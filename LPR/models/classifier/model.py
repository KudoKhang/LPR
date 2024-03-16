ALPHA_DICT = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "R",
    14: "S",
    15: "T",
    16: "U",
    17: "V",
    18: "X",
    19: "Y",
    20: "Z",
    21: "0",
    22: "1",
    23: "2",
    24: "3",
    25: "4",
    26: "5",
    27: "6",
    28: "7",
    29: "8",
    30: "9",
    31: "Background",
}

import sys

sys.path.insert(0, ".")


import cv2
import numpy as np
import torch
import torch.nn as nn


class CNN_Model_Pytorch(nn.Module):
    def __init__(self):
        super(CNN_Model_Pytorch, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 32)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.conv6(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(-1, 64)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)

        return x
