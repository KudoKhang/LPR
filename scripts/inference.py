import sys

sys.path.insert(0, ".")
from LPR.LPRPredict import *

lpr_predictor = LicensePlateRecognition()

image = cv2.imread("data/30G42717.jpg")

output = lpr_predictor.predict(image)
print(output)
