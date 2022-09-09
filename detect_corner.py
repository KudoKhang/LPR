import cv2

from utilss import *
# Init models
model_detect_corner = torch.hub.load('ultralytics/yolov5', 'custom', path='src/weights/detect_corner.pt', force_reload=True)

def draw(img, bbox):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)

def get_center(bbox):
    cx, cy = bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2
    return [cx, cy]

def transform_plate(img, pt_A, pt_B, pt_C, pt_D):
    w, h = img.shape[:2]

    # pt_A = [8, 4]
    # pt_B = [4, 71]
    # pt_C = [88, 69]
    # pt_D = [92, 5]

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

def detect(img):
    results = model_detect_corner(img)
    t = results.pandas().xyxy[0]
    A = np.int32(np.array(t)[:, :4][np.where(np.array(t)[:, 6] == 'A')])[0]
    B = np.int32(np.array(t)[:, :4][np.where(np.array(t)[:, 6] == 'B')])[0]
    C = np.int32(np.array(t)[:, :4][np.where(np.array(t)[:, 6] == 'C')])[0]
    D = np.int32(np.array(t)[:, :4][np.where(np.array(t)[:, 6] == 'D')])[0]
    draw(img, A)
    draw(img, B)
    draw(img, C)
    draw(img, D)
    pt_A = get_center(A)
    pt_B = get_center(B)
    pt_C = get_center(C)
    pt_D = get_center(D)
    img = transform_plate(img, pt_A, pt_B, pt_C, pt_D)
    return img

img = cv2.imread('data/92.jpg')
img_ori = img.copy()
img = detect(img)
cv2.imwrite('test1.png', img)
# img = cv2.resize(img, tuple(np.array(list(img.shape[:2])) * 10)[::-1])
cv2.imshow('original', img_ori)
cv2.imshow('result', img)
cv2.waitKey(0)