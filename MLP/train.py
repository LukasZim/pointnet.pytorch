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

tensorboard_writer = SummaryWriter(log_dir=f"runs/MLP_{time.strftime('%Y%m%d-%H%M%S')}")

path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/"
mesh_path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj"

train_dataloader, train_dataset = FractureDataLoader(path, type="train")
test_dataloader, test_dataset = FractureDataLoader(path, type="test")
mlp = MLP(9)
loss_function = custom_loss
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)



for epoch in (range(0, 400)):
    print(f'Starting Epoch {epoch + 1}')

    training_loss = 0.0
    mlp.train()
    for i, data in enumerate(train_dataloader):
        [inputs, targets, impulses, label_gt, label_edge] = data
        for index, (coordinates, udf_value, impulse) in enumerate(zip(inputs, targets, impulses)):
            coordinates, udf_value = coordinates.float(), udf_value.float()
            udf_value = udf_value.reshape((udf_value.shape[0], 1))

            optimizer.zero_grad()
            tens = torch.cat((coordinates, impulse.unsqueeze(0).repeat(coordinates.size(0), 1)), dim=1)

            outputs = mlp(tens)
            loss = loss_function(outputs, udf_value)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if i == 0 and index == 0:
                print(f"loss after mini-batch %5d: %.3f" % (i + 1, training_loss / 50))


    mlp.eval()
    testing_loss = 0.0
    for i, data in enumerate(test_dataloader):
        [inputs, targets, impulses] = data
        for index, (coordinates, udf_value, impulse) in enumerate(zip(inputs, targets, impulses)):


            coordinates, udf_value = coordinates.float(), udf_value.float()
            udf_value = udf_value.reshape((udf_value.shape[0], 1))

            optimizer.zero_grad()
            tens = torch.cat((coordinates, impulse.unsqueeze(0).repeat(coordinates.size(0), 1)), dim=1)

            outputs = mlp(tens)
            loss = loss_function(outputs, udf_value)
            # loss.backward()
            # optimizer.step()

            testing_loss += loss.item()
    #
    # tensorboard_writer.add_scalar('Loss/Train', training_loss / len(train_dataset), epoch)
    # tensorboard_writer.add_scalar('Loss/Test', testing_loss / len(test_dataset), epoch)
    tensorboard_writer.add_scalars("Loss", {"Train": training_loss / len(train_dataset), "Test": testing_loss / len(test_dataset)}, epoch)
    print(f'Finished Epoch {epoch + 1}')
    # if epoch % 5 == 0:
    save_checkpoint(epoch, mlp, optimizer, "checkpoints", train_dataset, mesh_path)

print("Training Finished")

visualize()
