import cv2
import numpy as np
from skimage.filters import threshold_local

def canny_edge_detection(image_path, blur_ksize=5, threshold1=100, threshold2=200):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    img_canny = cv2.Canny(img_gaussian, threshold1, threshold2)
    return img_canny

image_path = 'data/plate.png'
img = cv2.imread(image_path)
img_canny = canny_edge_detection(image_path, 5, 10, 50)
contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def get_max_area(bbox):
    if len(bbox) == 1:
        return bbox[0]
    if len(bbox) == 0:
        return "No plate detection!"
    else:
        area = [(bbox[:,2] - bbox[:,0]) * (bbox[:,3] - bbox[:,1])]
        index_max = int(np.where(max(area))[0])
        return bbox[index_max]

lst_coord_and_area = []
for c in contours:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    lst_coord_and_area.append((x, y, w, h))
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)

lst_coord_and_area = np.array(lst_coord_and_area)
max_bbox = get_max_area(lst_coord_and_area)

d = lst_coord_and_area[:, 4](np.argmax(np.array(lst_coord_and_area)[:, -1]))

print('p')