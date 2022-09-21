import os
import cv2
import numpy as np
from utilss import *
import math
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Init models
model_detect_corner = torch.hub.load('ultralytics/yolov5', 'custom', path='src/weights/detect_corner_v2.pt')
model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='src/weights/plate_yolo10k.pt')

def draw(img, bbox):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)

def get_center(bbox):
    cx, cy = bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2
    return [cx, cy]

def transform_plate(img, pt_A, pt_B, pt_C, pt_D):
    # A-B-C-D : counter-clockwise
    w, h = img.shape[:2]

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])

    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    img = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    return img

def draw_corner(img, t):
    # bbox = np.int32(np.array(t)[:, :4][np.where(np.array(t)[:, 4] > 0.6)])
    bbox = np.int32(np.array(t)[:, :4])
    for bb in bbox:
        draw(img, bb)

def interpolate_end_point(A, B, C, D, coefficient_expand = 1):
    # Tính chất hình bình hành: vector cặp cạnh đối diện luôn bằng nhau
    if len(A) == 0:
        x = B[0][0] + D[0][0] - C[0][0]
        y = B[0][1] + D[0][1] - C[0][1]
        A = [[x, y]]

    if len(B) == 0:
        x = A[0][0] + C[0][0] - D[0][0]
        y = A[0][1] + C[0][1] - D[0][1]
        B = [[x, y]]

    if len(C) == 0:
        x = B[0][0] + D[0][0] - A[0][0]
        y = B[0][1] + D[0][1] - A[0][1]
        C = [[x, y]]

    if len(D) == 0:
        x = A[0][0] + C[0][0] - B[0][0]
        y = A[0][1] + C[0][1] - B[0][1]
        D = [[x, y]]

    # expanded coordinate
    A = [A[0][0] - coefficient_expand, A[0][1] - coefficient_expand]
    B = [B[0][0] - coefficient_expand, B[0][1] + coefficient_expand]
    C = [C[0][0] + coefficient_expand, C[0][1] + coefficient_expand]
    D = [D[0][0] + coefficient_expand, D[0][1] - coefficient_expand]

    return A, B, C, D

def get_true_coord(A, B, C, D, bbox_plate):
    w, h = bbox_plate[2] - bbox_plate[0], bbox_plate[3] - bbox_plate[1]

    _dict = {'A': (A, (0, 0)),
             'B': (B, (0, h)),
             'C': (C, (w, h)),
             'D': (D, (w, 0))}

    for key in _dict.keys():
        if len(_dict[key][0]) > 1:
            temp = []
            for center in _dict[key][0]:
                dist = math.dist(center, _dict[key][1])
                temp.append((dist, center))
            temp = np.array(temp)
            _dict[key] = (temp[:, 1][np.where(np.argmax(temp[:, 0]))], '000')

    return interpolate_end_point(A, B, C, D)

def ABCD(bbox, img, bbox_plate):
    h, w = img.shape[:2]
    A = []
    B = []
    C = []
    D = []
    for bb in bbox:
        center = get_center(bb)
        if center[0] < w / 2:
            A.append(center) if center[1] < h / 2 else B.append(center)
        else:
            D.append(center) if center[1] < h / 2 else C.append(center)

    return get_true_coord(A, B, C, D, bbox_plate)

def detect_corner_and_transform(img, bbox_plate, is_draw=True):
    results = model_detect_corner(img)
    t = results.pandas().xyxy[0]
    bbox = np.int32(np.array(t)[:, :4])

    if is_draw:
        draw_corner(img, t)

    try:
        pt_A, pt_B, pt_C, pt_D = ABCD(bbox, img, bbox_plate)
        try:
            exp = 4
            pt_A = [pt_A[0] - exp, pt_A[1] - exp]
            pt_B = [pt_B[0] - exp, pt_B[1] + exp]
            pt_C = [pt_C[0] + exp, pt_C[1] + exp]
            pt_D = [pt_D[0] + exp, pt_D[1] - exp]
        except:
            pt_A, pt_B, pt_C, pt_D = pt_A, pt_B, pt_C, pt_D
        img = transform_plate(img, pt_A, pt_B, pt_C, pt_D)
    except:
        #TODO: return original plate
        return img
    return img

def expand_bbox(bbox, img, scale=0.1):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    h_expand = int((y2 - y1) * scale)

    x1_exp = 0 if (x1 - h_expand) < 0 else (x1 - h_expand)
    y1_exp = 0 if (y1 - h_expand) < 0 else (y1 - h_expand)
    x2_exp = W if (x2 + h_expand) > W else (x2 + h_expand)
    y2_exp = H if (y2 - h_expand) > H else (y2 + h_expand)

    return (x1_exp, y1_exp, x2_exp, y2_exp)

def detect_plate(img):
    results = model_detect(img)
    t = results.pandas().xyxy[0]
    if len(t) > 0:
        # TODO: check area and confident
        bbox = np.int32(np.array(t)[:, :4][np.argmax(np.array(t)[:, 4])]) # Max confident
        bbox_exp = expand_bbox(bbox, img)
        plate = crop(img, bbox_exp)
        return plate, bbox
    else:
        return None, None

def process_image(path='data/private_test/GOOD/76C06593.jpg'):
    img = cv2.imread(path)
    img_ori = img.copy()
    plate, bbox = detect_plate(img)
    plate_transform = detect_corner_and_transform(plate, bbox)
    # cv2.imwrite('test1.png', plate)
    # img = cv2.resize(img, tuple(np.array(list(img.shape[:2])) * 10)[::-1])
    cv2.imshow('original', plate)
    cv2.imshow('result', plate_transform)
    cv2.waitKey(0)

def process_folder(root='data/private_test/GOOD/'):
    lst_img = [root + name for name in os.listdir(root) if name.endswith(('jpg'))]
    for path in lst_img:
        process_image(path)

process_image('data/private_test/GOOD/76C06323(2).jpg')
# process_image('88H0009.jpg')
# process_folder()
