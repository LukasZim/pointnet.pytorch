import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from MLP.data_loader.data_loader import FractureDataset, FractureDataLoader
from MLP.model import model
from MLP.model.loss import *
from MLP.model.model import MLP
from MLP.test import visualize
from MLP.visualize import load_mesh_from_file
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(epoch, model, optimizer, path, dataset, mesh_path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'dataset': dataset,
        'mesh_path': mesh_path
    }
    torch.save(state, f'{path}/{epoch}.pth')

tensorboard_writer = SummaryWriter(log_dir="runs/MLP")

path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/"
mesh_path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj"

dataloader, dataset = FractureDataLoader(path)
mlp = MLP(9)
loss_function = custom_loss
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)



for epoch in (range(0, 1000)):
    print(f'Starting Epoch {epoch + 1}')

    current_loss = 0.0

    for i, data in enumerate(dataloader):
        inputs, targets, impulse = data
        # for coordinates, udf_value in zip(inputs, targets):
            # coordinates = inputs[chunk_index]
            # udf_value = targets[chunk_index]
        coordinates = inputs
        udf_value = targets
        coordinates, udf_value = coordinates.float(), udf_value.float()
        udf_value = udf_value.reshape((udf_value.shape[0], 1))

        optimizer.zero_grad()
        tens = torch.cat((coordinates, impulse), dim=1)
        outputs = mlp(tens)

        loss = loss_function(outputs, udf_value)

        loss.backward()

        optimizer.step()

        current_loss += loss.item()

        if i == 0:
            print(f"loss after mini-batch %5d: %.3f" % (i + 1, current_loss / 50))
            current_loss = 0.0
    tensorboard_writer.add_scalar('loss/train', current_loss, epoch)
    print(f'Finished Epoch {epoch + 1}')
    # if epoch % 5 == 0:
    save_checkpoint(epoch, mlp, optimizer, "checkpoints", dataset, mesh_path)

print("Training Finished")

visualize()
