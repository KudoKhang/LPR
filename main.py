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
from src.char_classification.model import CNN_Model
from utilss.functions import *


ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='src/weights/plate_yolo10k.pt', force_reload=True)  # detect licence plate
model_detect_character = torch.hub.load('ultralytics/yolov5', 'custom', path='src/weights/character_yolo.pt') # detect character
CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight_character_gray_clean.h5' # Classify character
recogChar = CNN_Model(trainable=False).model
recogChar.load_weights(CHAR_CLASSIFICATION_WEIGHTS)

def detect_char(lpRegion):
    # TODO: Take 8 characters bbox with the most confidence
    condidates = []
    results = model_detect_character(lpRegion)
    t = results.pandas().xyxy[0]
    bbox = list(np.int32(np.array(t)[:, :4]))

    # bbox.sort(key=lambda x: x[0])

    height_char = []

    if len(bbox) > 0:
        for bb in bbox:
            x1, y1, x2, y2 = bb
            height_char.append(y2 - y1)
            cv2.rectangle(lpRegion, (x1,y1), (x2,y2), (0,255,0), 1)
            char = lpRegion.copy()[y1:y2, x1:x2]
            V = cv2.split(cv2.cvtColor(char, cv2.COLOR_BGR2HSV))[2]
            T = threshold_local(V, 31, offset=10, method="gaussian")
            thresh = (V > T).astype("uint8") * 255
            thresh = cv2.bitwise_not(thresh)

            h = 200
            thresh = imutils.resize(thresh, height=h)
            thresh = cv2.medianBlur(thresh, 5)

            bg = np.zeros((h, h))
            x = int((h - thresh.shape[1]) / 2)
            bg[0:h, x: x + thresh.shape[1]] = thresh
            bg = cv2.resize(bg, (28, 28))
            bg = denoise(bg)
            condidates.append((bg, (y1, x1)))

    return condidates, sum(height_char) / len(height_char)


def recognizeChar(candidates):
    characters = []
    coordinates = []

    for char, coordinate in candidates:
        characters.append(char)
        coordinates.append(coordinate)

    characters = np.array(characters)
    result = recogChar.predict_on_batch(characters)
    result_idx = np.argmax(result, axis=1)

    candidates = []
    for i in range(len(result_idx)):
        if result_idx[i] == 31:  # if is background or noise, ignore it
            continue
        candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    return candidates


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

def detect_plate(img):
    results = model_detect(img)
    t = results.pandas().xyxy[0]
    bbox = np.int32(np.array(t)[:, :4][np.argmax(np.array(t)[:, 4])])
    plate = crop(img, bbox)
    return plate, bbox

def E2E(image):
    plate, bbox = detect_plate(image)
    candidates, h_avg = detect_char(plate)
    candidates = recognizeChar(candidates)
    license_plate = format(candidates, h_avg)
    print(license_plate)
    img = draw_labels_and_boxes(image, license_plate, bbox)
    return img

def process_folder(path_folder='data/pravite_test_500', output_folder='output/results_clean/'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    image_path = [name for name in os.listdir(path_folder) if name.endswith(('jpg', 'png'))]
    os.remove('tests/error.txt')
    for path in tqdm(image_path):
        try:
            img = cv2.imread(os.path.join(path_folder, path))
            img = E2E(img)
            cv2.imwrite(os.path.join(output_folder, path), img)
        except:
            with open('tests/error.txt', 'a') as f:
                f.write(f"{path} \n")
                f.close()
            print(path)

def process_image(image_path):
    img = cv2.imread(image_path)
    img = E2E(img)
    cv2.imshow('result', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    process_folder('data/pravite_test_500/', 'output/results_500')
    # process_image('data/pravite_test_500/49.jpg')