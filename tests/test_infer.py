import sys

sys.path.insert(0, ".")
from LPR.LicensePlateRecognition import *
from LPR.utils.config import cfg

cfg = cfg(config_name="config")


def test_infer():
    lpr_predictor = LicensePlateRecognition()
    image = cv2.imread(cfg.image_test)
    output = lpr_predictor.predict(image)
    assert output == ("51C45736", "[266 144 351 212]")
