import torch
import torch.nn as nn
from network import Siamese
from torchvision import transforms

def train():
    return

def test():
    return

def plot():
    return



net = Siamese()

img_x = torch.rand(3, 128, 128) #torch.tensor((128, 128, 3)) 
img_y = torch.rand(3, 128, 128) #torch.tensor((128, 128, 3))


out = net(img_x, img_y) 
print(out)