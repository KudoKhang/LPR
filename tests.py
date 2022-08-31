import cv2
import numpy as np
from skimage.filters import threshold_local

def canny_edge_detection(image_path, blur_ksize=5, threshold1=100, threshold2=200):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    img_canny = cv2.Canny(img_gaussian, threshold1, threshold2)
    return img_canny

def denoise(char):
    # Remove noise in character binary via findContour --> calculate Area --> Compare with threshold area
    char = np.uint8(char)
    contours, hierarchy = cv2.findContours(char, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lst_coord_and_area = []
    for c in contours:
        area =cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(char, (x, y), (x + w, y + h), (255, 255, 255), 1)
        lst_coord_and_area.append((x, y , w, h, area))

    if len(lst_coord_and_area) > 0:
        lst_coord_and_area.pop(np.argmax(np.array(lst_coord_and_area)[:, -1]))

    return char

image_path = 'data/plate.png'
img = cv2.imread(image_path)
img_canny = canny_edge_detection(image_path, 5, 10, 50)
img_canny = denoise(img_canny)
# contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

d = lst_coord_and_area[:, 4](np.argmax(np.array(lst_coord_and_area)[:, -1]))

print('p')