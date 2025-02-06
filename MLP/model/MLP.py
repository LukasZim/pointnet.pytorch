import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)



class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # self.layers = nn.Sequential(
        #     nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2),
        #     nn.Conv1d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2),
        #     nn.Linear(128, 1),
        #     nn.ReLU()
        # )
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=11, padding=5)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(256, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        # x = torch.relu(self.linear(x))
        return x
        # return self.layers(x)