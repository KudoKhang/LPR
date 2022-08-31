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
import torch
from torchvision import transforms
import torchvision

#other lib
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image