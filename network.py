import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()

        self.twin = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),      

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.body = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.line = nn.Linear(16384, 1)
    
    def forward(self, x, y):
        x = self.twin(x)
        y = self.twin(y)
        z = torch.add(x, y)
        z = self.body(z)
        z = nn.Flatten(0, -1)(z)
        z = self.line(z)
        return z