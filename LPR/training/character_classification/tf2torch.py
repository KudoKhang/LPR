import sys

sys.path.insert(0, ".")


import cv2
import numpy as np
import torch
import torch.nn as nn
from keras.models import load_model

from LPR.models.classifier import ALPHA_DICT

# Load the Keras model
keras_model = load_model("checkpoints/classify_character.h5")

# Get the Keras model's weights as a NumPy array
keras_weights = keras_model.get_weights()


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


torch_model = CNN_Model_Pytorch()


def convert_checkpoint():
    for i, (name, param) in enumerate(torch_model.named_parameters()):
        if "weight" in name:
            w = torch.from_numpy(keras_weights[i])
            if w.dim() == 3:
                w = w.t()
            elif w.dim() == 1:
                pass
            else:
                assert w.dim() == 4
                w = w.permute(3, 2, 0, 1)

            param.data = w

    # Save the PyTorch model's weights as a state dictionary
    torch.save(torch_model.state_dict(), "checkpoints/classify_character.pt")


def infer():
    # Load the PyTorch model's state dictionary from the checkpoint file
    checkpoint = torch.load("checkpoints/classify_character.pt")
    torch_model.load_state_dict(checkpoint)

    # Put the PyTorch model in evaluation mode
    torch_model.eval()

    # Load your input data and preprocess it as necessary
    # input_data = np.random.rand(1, 28, 28, 1)  # replace with your actual input data
    input_data = cv2.imread("data/char_3.png", 0)
    input_data = np.expand_dims(input_data, 0)
    input_data = np.expand_dims(input_data, -1)
    input_data_2 = input_data.copy()

    input_tensor = torch.from_numpy(input_data.transpose(0, 3, 1, 2)).float()
    print(input_tensor.shape)

    # Pass the preprocessed input data through the PyTorch model
    output_tensor = torch_model.forward(input_tensor)

    # Use the output of the PyTorch model to make predictions or perform other downstream tasks
    predictions = output_tensor.detach().numpy()
    print(predictions)
    print(np.argmax(predictions))
    print(ALPHA_DICT[np.argmax(predictions)])


if __name__ == "__main__":
    # convert_checkpoint()
    infer()
