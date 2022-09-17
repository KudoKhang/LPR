from LPRPredict import *
import argparse
import os
import cv2
import numpy as np
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/private_test/GOOD/36C17806.jpg", help="Path to input image")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    start = time.time()
    LPRPredictor = LicensePlateRecognition()
    img = cv2.imread(args.input)
    result = LPRPredictor.predict(img)
    end = round((time.time() - start) * 1e3, 2)
    print(f'Time to inference: {end}')
    print(result)