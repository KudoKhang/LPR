import cv2
import numpy as np
from skimage.filters import threshold_local

char_ori = cv2.imread('../debug/2.png')
char = cv2.resize(char_ori, tuple([a * 10 for a in char_ori.shape[:2][::-1]]))
V = cv2.split(cv2.cvtColor(char, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 31, offset=10, method="gaussian")
thresh = (V > T).astype("uint8") * 255
thresh = cv2.bitwise_not(thresh)

char_gray = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
th2 = cv2.adaptiveThreshold(char_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(char_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
blur = cv2.GaussianBlur(char_gray,(1,1),1)
ret3,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
th4_inv = cv2.bitwise_not(th4)
th4_ori = cv2.resize(th4_inv, char_ori.shape[:2][::-1])
print('p')
