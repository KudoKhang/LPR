import os
from tqdm import tqdm
import cv2
from pathlib import Path
import argparse
import time
import torch
import numpy as np
from skimage.filters import threshold_local
import imutils
from skimage import measure
from src.char_classification.model import CNN_Model
from utilss.functions import *
#pytorch
from torchvision import transforms
import torchvision

#other lib
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import math