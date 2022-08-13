import os
from tqdm import tqdm
import cv2
from pathlib import Path
import argparse
import time
import torch
import numpy as np
from skimage.filters import threshold_local
import imutils
from skimage import measure
import warnings
warnings.filterwarnings('ignore')

def get_max_area(bbox):
    if len(bbox) == 1:
        return bbox[0]
    if len(bbox) == 0:
        return "No plate detection!"
    else:
        area = [(bbox[:,2] - bbox[:,0]) * (bbox[:,3] - bbox[:,1])]
        index_max = int(np.where(max(area))[0])
        return bbox[index_max]

def crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def convert2Square(image):
    img_h = image.shape[0]
    img_w = image.shape[1]

    # if height > width
    if img_h > img_w:
        diff = img_h - img_w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = np.zeros(shape=(img_h, (diff//2) + 1))

        squared_image = np.concatenate((x1, image, x2), axis=1)
    elif img_w > img_h:
        diff = img_w - img_h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1

        squared_image = np.concatenate((x1, image, x2), axis=0)
    else:
        squared_image = image

    return squared_image

def format(candidates, h_avg):
    #TODO: Tại vị trí thứ 3 luôn luôn là chữ,
    # còn các vị trí khác luôn luôn là số
    first_line = []
    second_line = []

    """ IDEA
        - Find y_max, for y .. compare y_max
    """

    lst = []
    l = np.array(candidates)[:, 1]
    for l_sub in l:
        lst.append(list(l_sub))
    y_max = max(np.array(lst)[:, 0])

    for candidate, coordinate in candidates:
        if coordinate[0] + 0.75 * h_avg > y_max:
            first_line.append((candidate, coordinate[1]))
        else:
            second_line.append((candidate, coordinate[1]))

    def take_second(s):
        return s[1]

    first_line = sorted(first_line, key=take_second)
    second_line = sorted(second_line, key=take_second)

    if len(second_line) == 0:  # if license plate has 1 line
        license_plate = "".join([str(ele[0]) for ele in first_line])
        license_plate = license_plate[:3] + '-' + license_plate[3:]
    else:  # if license plate has 2 lines
        license_plate = "".join([str(ele[0]) for ele in second_line]) + "-" +  "".join([str(ele[0]) for ele in first_line])
        # license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" +  "".join([str(ele[0]) for ele in second_line])

    return license_plate


def segmentation(LpRegion):
    candidates = []
    # apply thresh to extracted licences plate
    V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]

    # adaptive threshold
    T = threshold_local(V, 15, offset=10, method="gaussian")
    thresh = (V > T).astype("uint8") * 255

    # convert black pixel of digits to white pixel
    thresh = cv2.bitwise_not(thresh)
    thresh = imutils.resize(thresh, width=400)
    thresh = cv2.medianBlur(thresh, 5)

    # connected components analysis
    labels = measure.label(thresh, connectivity=2, background=0)

    # loop over the unique components
    for label in np.unique(labels):
        # if this is background label, ignore it
        if label == 0:
            continue

        # init mask to store the location of the character candidates
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255

        # find contours from mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(contour)

            # rule to determine characters
            aspectRatio = w / float(h)
            solidity = cv2.contourArea(contour) / float(w * h)
            heightRatio = h / float(LpRegion.shape[0])

            if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                # extract characters
                candidate = np.array(mask[y:y + h, x:x + w])
                square_candidate = convert2Square(candidate)
                square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                square_candidate = square_candidate.reshape((28, 28, 1))
                candidates.append((square_candidate, (y, x)))
    return candidates

def draw_labels_and_boxes(image, labels, boxes):
    x_min = round(boxes[0])
    y_min = round(boxes[1])
    x_max = round(boxes[2])
    y_max = round(boxes[3])

    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
    image = cv2.putText(image, labels, (x_min - 40, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

    return image

def detect_char_rgb(lpRegion):
    condidates = []
    results = model_detect_character(lpRegion)
    t = results.pandas().xyxy[0]
    bbox = list(np.int32(np.array(t)[:, :4]))
    bbox.sort(key=lambda x: x[0])

    if len(bbox) > 0:
        for bb in bbox:
            x1, y1, x2, y2 = bb
            cv2.rectangle(lpRegion, (x1,y1), (x2,y2), (0,255,0), 1)
            char = lpRegion.copy()[y1:y2, x1:x2]

            bg = np.zeros((char.shape[0], char.shape[0]))
            x = int((char.shape[0] - char.shape[1]) / 2)

            char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
            bg[0:char.shape[0], x: x + char.shape[1]] = char
            bg = cv2.resize(bg, (28, 28))

            condidates.append((bg, (y1, x1)))
    return condidates
