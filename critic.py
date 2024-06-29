import torch
import torch.nn as nn

## Class for creating a simple FF network for the "critic"

class Critic(nn.Module):
    def __init__(self, input_size,output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size2, 32)#state vec to dim (32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.relu(self.fc3(x))
        
        x= self.fc4(x)#took out relu b/c negative scores seem reasonable?
        return x
