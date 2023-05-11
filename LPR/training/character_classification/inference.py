import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from LPR.functions import *
from LPR.models.classifier import ALPHA_DICT, CNN_Model
from LPR.utils.config import cfg

cfg = cfg(config_name="config")

input_data = cv2.imread("data/char_3.png", 0)
input_data = np.expand_dims(input_data, 0)
print(input_data.dtype)

model_recognize_character = CNN_Model(trainable=False).model
model_recognize_character.load_weights(cfg.classify.path_local)

result = model_recognize_character.predict_on_batch(input_data)
idx = np.argmax(result)

print(result)
print(idx)
print(ALPHA_DICT[idx])
