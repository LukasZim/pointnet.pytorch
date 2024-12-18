import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from MLP.data_loader.data_loader import FractureData, FractureDataLoader
from MLP.model import model
from MLP.model.loss import l1_loss
from MLP.model.model import MLP
from MLP.test import visualize
from MLP.visualize import load_mesh_from_file


def save_checkpoint(epoch, model, optimizer, path, X, y, X_test, y_test, X_train, y_train, mesh_path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'X': X,
        'y': y,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train,
        'mesh_path': mesh_path
    }
    torch.save(state, f'{path}/{epoch}.pth')

path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/79.csv"
mesh_path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj"

dataloader, X, y, X_train, y_train, X_test, y_test = FractureDataLoader(path)
mlp = MLP(3)
loss_function = l1_loss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)

for epoch in (range(0, 100)):
    print(f'Starting Epoch {epoch + 1}')

    current_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        optimizer.zero_grad()

        outputs = mlp(inputs)

        loss = loss_function(outputs, targets)

        loss.backward()

        optimizer.step()

        current_loss += loss.item()

        if i == 0:
            print(f"loss after mini-batch %5d: %.3f" % (i + 1, current_loss / 50))
            current_loss = 0.0

    print(f'Finished Epoch {epoch + 1}')
    # if epoch % 5 == 0:
    save_checkpoint(epoch, mlp, optimizer, "checkpoints", X, y, X_test, y_test, X_train, y_train, mesh_path)

print("Training Finished")

visualize()
