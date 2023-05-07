import argparse
import os

# other lib
import sys
import time
import warnings
from pathlib import Path

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from skimage import measure
from skimage.filters import threshold_local

# pytorch
from torchvision import transforms
from tqdm import tqdm

# from LPR.src.character_classification.model import CNN_Model
from LPR.functions.functions import *

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
