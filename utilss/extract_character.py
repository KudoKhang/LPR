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

model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='../src/weights/plate_yolo10k.pt', force_reload=True)  # local model

image_path = [name for name in os.listdir('../../Data10k') if name.endswith('jpg')]


output_character = 'test/'

os.makedirs(output_character, exist_ok=True)

for path in image_path[:3]:
    img = cv2.imread('../Data10k/' + path)
    results = model_detect(img)
    t = results.pandas().xyxy[0]
    bbox = np.int32(np.array(t)[:,:4][np.where(np.array(t)[:,6] == 'plate')]) # x1,y1,x2,y2
    bbox = get_max_area(bbox)
    plate = crop(img, bbox)

    candidates = segmentation(plate)

    id = path.split('.')[0]
    for idx, i in tqdm(enumerate(candidates)):
        # name = str(time.time()).split('.')[-1]
        name = id + '_' + str(idx)
        cv2.imwrite(f'{output_character}/{name}.png', i[0])
