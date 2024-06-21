import torch
import torch.nn as nn

## Class for creating a simple FF network for the "actor"

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x3 = self.relu(self.fc3(x2))
        x4 = self.relu(self.fc4(x3))
        return x4
