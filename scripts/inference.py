import cv2

from lpr.main import LicensePlateRecognition
from lpr.utils.config import cfg

lpr_predictor = LicensePlateRecognition()

image = cv2.imread(cfg.image_test)

output = lpr_predictor.predict(image)
print(output)
