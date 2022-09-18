import cv2
import numpy as np
from utilss import *
import math
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Init models
model_detect_corner = torch.hub.load('ultralytics/yolov5', 'custom', path='src/weights/detect_corner.pt')
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

def get_true_coord(A, B, C, D, bbox_plate):
    w, h = bbox_plate[2] - bbox_plate[0], bbox_plate[3] - bbox_plate[1]

    if len(A) > 1:
        temp = []
        for center in A:
            dist = math.dist(center, (0, 0))
            temp.append((dist, center))
        temp = np.array(temp)
        A = temp[:, 1][np.where(np.argmax(temp[:, 0]))]

    if len(B) > 1:
        temp = []
        for center in B:
            dist = math.dist(center, (0, h))
            temp.append((dist, center))
        temp = np.array(temp)
        B = temp[:, 1][np.where(np.argmax(temp[:, 0]))]

    if len(C) > 1:
        temp = []
        for center in C:
            dist = math.dist(center, (w, h))
            temp.append((dist, center))
        temp = np.array(temp)
        C = temp[:, 1][np.where(np.argmax(temp[:, 0]))]

    if len(D) > 1:
        temp = []
        for center in D:
            dist = math.dist(center, (w, 0))
            temp.append((dist, center))
        temp = np.array(temp)
        D = temp[:, 1][np.where(np.argmax(temp[:, 0]))]

    return A[0], B[0], C[0], D[0]


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

    A, B, C, D = get_true_coord(A, B, C, D, bbox_plate)
    return A, B, C, D

def detect_corner(img, bbox_plate):
    results = model_detect_corner(img)
    t = results.pandas().xyxy[0]
    bbox = np.int32(np.array(t)[:, :4])

    draw_corner(img, t)

    try:
        pt_A, pt_B, pt_C, pt_D = ABCD(bbox, img, bbox_plate)
        img = transform_plate(img, pt_A, pt_B, pt_C, pt_D)
    except:
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

img = cv2.imread('data/private_test/GOOD/76C06181(1).jpg')
img_ori = img.copy()
plate, bbox = detect_plate(img)
plate_transform = detect_corner(plate, bbox)
cv2.imwrite('test1.png', plate)
# img = cv2.resize(img, tuple(np.array(list(img.shape[:2])) * 10)[::-1])
cv2.imshow('original', plate)
cv2.imshow('result', plate_transform)
cv2.waitKey(0)