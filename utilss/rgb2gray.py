import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_local
import imutils
from tqdm import tqdm
import sys

# sys.path.extend(['/Users/NghiaKhang/Coding/Projects/License Plate Detection/LPR'])
# print(os.getcwd())

root = '../character_4k/'
gray_folder = '../character_4k_binary/'
"""
    Convert RGB image to Binary image
"""

def thresold(img):
    V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
    # adaptive threshold
    T = threshold_local(V, 15, offset=10, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)
    thresh = imutils.resize(thresh, height=100)
    thresh = cv2.medianBlur(thresh, 5)
    return thresh

def padding(img):
    h, w = img.shape[:2]
    if h > w:
        bg = np.zeros((h, h))
        x = int((h - w) / 2)
        bg[0:h, x: x + w] = img
    else:
        bg = np.zeros((w, w))
        x = int((w - h) / 2)
        bg[x: x + h, 0: w] = img
    # bg = cv2.resize(bg, (28, 28))
    return bg


def denoise(char):
    char = np.uint8(char)
    _, mask = cv2.threshold(char, 15, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    l = []
    for c in contours:
        area =cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        l.append((x, y , w, h, area))

    if len(l) > 0:
        l.pop(np.argmax(np.array(l)[:, -1]))

    for bb in l:
        x, y, w, h = bb[:4]
        points = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        cv2.fillPoly(char, pts=[points], color=(0, 0, 0))
    return char


# HAVE SUB FOLDER
def have_sub_folder():
    for folder in tqdm(os.listdir(root)):
        if folder == '.DS_Store':
            continue
        path_image = [name for name in os.listdir(os.path.join(root, folder)) if name.endswith('jpg')]
        os.makedirs(os.path.join(gray_folder, folder), exist_ok=True)

        for path in path_image:
            img = cv2.imread(os.path.join(root, folder, path))
            img = thresold(img)
            img = padding(img)
            img = denoise(img)
            cv2.imwrite(os.path.join(gray_folder, folder, path), img)

# have_sub_folder()

# NON-SUB FOLDER
def non_sub_folder():
    os.makedirs(gray_folder, exist_ok=True)
    for path_image in tqdm(os.listdir(root)):
        if path_image == '.DS_Store':
            continue
        img = cv2.imread(os.path.join(root, path_image))
        img = thresold(img)
        img = padding(img)
        img = denoise(img)
        cv2.imwrite(os.path.join(gray_folder, path_image), img)

non_sub_folder()

print('Done!')