import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Split data to Ditgits and Alphas folder

path_root = "../character_dataset37k/"

os.makedirs("digits", exist_ok=True)
os.makedirs("alphas", exist_ok=True)

digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Background"]

for path in tqdm(os.listdir(path_root)):
    if path == ".DS_Store":
        continue
    if path in digits:
        shutil.copytree(f"{path_root}/{path}", f"digits/{path}")
    else:
        shutil.copytree(f"{path_root}/{path}", f"alphas/{path}")

print(f"Split {path_root} data to alphas and digits folder! ")
