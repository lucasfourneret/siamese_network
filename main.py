import torch
import torch.nn as nn
from network import Siamese
from torchvision import transforms
import numpy as np

def train(model, device, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = []
    for img_x, img_y, label in dataloader:
        out = model(img_x, img_y)
        loss = loss_fn(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def test(model, device, dataloader, loss_fn):
    model.eval()
    with torch.no_grad():
        for img_x, img_y, label in dataloader:
            out = model(img_x, img_y)
            loss = loss_fn(out, label)

    return

def plot(losses):
    return

net = Siamese(28, 28)

tensor_size = [3,128,128]

img_x = torch.rand(tensor_size)
img_y = torch.rand(tensor_size)
out = net(img_x, img_y)
print(out)
print(out.size())