import cv2

from lpr.main import LicensePlateRecognition
from lpr.utils.config import cfg


def test_infer():
    lpr_predictor = LicensePlateRecognition()
    image = cv2.imread(cfg.image_test)
    output = lpr_predictor.predict(image)
    assert output == ("51C45736", "[266 144 351 212]")
