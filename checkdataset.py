import cv2
import numpy as np

img = cv2.imread('../character_dataset37k/0/0.jpg')
img = cv2.resize(img, (280, 280))
cv2.imshow('img', img)
cv2.waitKey(0)