import torch
import torch.nn as nn

## Class for creating a simple FF network for the "critic"

class Critic(nn.Module):
    def __init__(self, input_size1,input_size2,output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size1, 32)#action vec to dim (32)
        self.fc2 = nn.Linear(input_size2, 32)#state vec to dim (32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x3 = self.relu(self.fc1(x1))
        x4 = self.relu(self.fc2(x2))
        x5 = x3+x4
        x6 = self.relu(self.fc3(x5))
        x7 = self.relu(self.fc4(x6))
        x8 = self.fc5(x7)#took out relu b/c negative action-state scores seem reasonable?
        return x8
