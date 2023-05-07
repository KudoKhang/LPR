import sys

sys.path.insert(0, ".")
from LPR.LicensePlateRecognition import *
from LPR.utils.config import cfg

cfg = cfg(config_name="config")

lpr_predictor = LicensePlateRecognition()

image = cv2.imread(cfg.image_test)

output = lpr_predictor.predict(image)
print(output)
