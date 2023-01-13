import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()

        self.twin = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(),

            nn.Conv2d(8, 16, 3, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout()
        )

        self.body = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.line = nn.Linear(128, 1)
    
    def forward(self, x, y):
        x = self.twin(x)
        y = self.twin(y)
        z = torch.add(x, y)
        z = self.body(z)
        z = nn.Flatten()(z),

        return z