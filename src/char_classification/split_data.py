import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm

# Split data to Ditgits and Alphas folder

path_root = '../character_GRAY_clean/'

os.makedirs('src/char_classification/digits', exist_ok=True)
os.makedirs('src/char_classification/alphas', exist_ok=True)

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Background']

for path in tqdm(os.listdir(path_root)):
    if path == '.DS_Store':
        continue
    if path in digits:
        shutil.copytree(f'{path_root}/{path}', f'src/char_classification/digits/{path}')
    else:
        shutil.copytree(f'{path_root}/{path}', f'src/char_classification/alphas/{path}')

print(f'Split {path_root} data to alphas and digits folder! ')