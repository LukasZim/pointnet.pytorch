import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.data import Data
from tqdm import tqdm

from MLP.data_loader.data_loader import FractureDataset, FractureDataLoader
from MLP.helper_functions import append_impulse_to_data, get_screenshot_polyscope
from MLP.metrics import minimum_chamfer_distance, calculate_n_minimum_chamfer_values, get_model_output_from_index
from MLP.model import MLP
from MLP.model.loss import *
from MLP.model.MLP import MLP, CNN
from MLP.region_growing import RegionGrowing
from MLP.test import visualize
from MLP.visualize import load_mesh_from_file
from torch.utils.tensorboard import SummaryWriter
from MLP.model.deltanet_regression import DeltaNetRegression




def save_checkpoint(epoch, model, optimizer, path, dataset, mesh_path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'dataset': dataset,
        'mesh_path': mesh_path
    }
    torch.save(state, f'{path}/{epoch}.pth')

def run_model(points, udf_values, impulses, used_model, train=False, requires_grad=False):
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)
    if not isinstance(impulses, torch.Tensor):
        impulses = torch.tensor(impulses)
    if not isinstance(udf_values, torch.Tensor):
        udf_values = torch.tensor(udf_values)

    points, udf_values = points.float(), udf_values.float()

    if train:
        optimizer.zero_grad()

    points_with_impulses = torch.cat((points, impulses.unsqueeze(0).permute(1, 0, 2).repeat(1, points.size(1), 1)),
                                     dim=2)
    points_with_impulses = points_with_impulses.permute(0, 2, 1).float()
    if requires_grad:
        points_with_impulses.requires_grad = True

    model_output = used_model(points_with_impulses)

    return model_output, points_with_impulses


def run_epoch(model, dataloader, train=True):
    loss = 0.0

    if train:
        model.train()
    else:
        model.eval()
    for i, data in enumerate(dataloader):
        [points, udf_values, impulses, gt_labels, label_edge] = data

        model_output, _ = run_model(points, udf_values, impulses, model, train=train)
        loss = loss_function(model_output, udf_values)
        if train:
            loss.backward()
            optimizer.step()

        loss += loss.item()

        if i == 0:
            print(f"loss after mini-batch %5d: %.3f" % (i + 1, loss / 50))

    return loss

if __name__ == '__main__':
    tensorboard_writer = SummaryWriter(log_dir=f"runs/MLP_{time.strftime('%Y%m%d-%H%M%S')}")

    path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/"
    mesh_path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj"

    train_dataloader, train_dataset = FractureDataLoader(path, type="train")
    test_dataloader, test_dataset = FractureDataLoader(path, type="test")
    model = CNN(9)
    # model = DeltaNetRegression(9)
    loss_function = custom_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in (range(0, 400)):
        print(f'Starting Epoch {epoch + 1}')
        training_loss = run_epoch(model, train_dataloader, train=True)
        testing_loss = run_epoch(model, test_dataloader, train=False)

        mesh = load_mesh_from_file(mesh_path)
        chamfer_value = calculate_n_minimum_chamfer_values(test_dataset, model, mesh)

        tensorboard_writer.add_scalars("Loss", {"Train": training_loss /len(mesh.vertices) , "Test": testing_loss / len(mesh.vertices)}, epoch)
        time_start = time.time()
        if epoch % 10 == 0:
            predicted_labels, predicted_udf, gt_labels, gt_udf = get_model_output_from_index(0, test_dataset, mesh, model)

            ss_predicted_labels = get_screenshot_polyscope(mesh, predicted_labels)
            ss_gt_labels = get_screenshot_polyscope(mesh, gt_labels)
            ss_gt_udf = get_screenshot_polyscope(mesh, gt_udf)
            ss_predicted_udf = get_screenshot_polyscope(mesh, predicted_udf)

            tensorboard_writer.add_image(f"predicted_labels/{epoch}", ss_predicted_labels, dataformats="HWC")
            tensorboard_writer.add_image(f"gt_labels/{epoch}", ss_gt_labels, dataformats="HWC")
            tensorboard_writer.add_image(f"gt_udf/{epoch}", ss_gt_udf, dataformats="HWC")
            tensorboard_writer.add_image(f"predicted_udf/{epoch}", ss_predicted_udf, dataformats="HWC")
            print("screenshots take this long", time.time() - time_start)
        tensorboard_writer.add_scalar('chamfer', chamfer_value, epoch)

        print(f'Finished Epoch {epoch + 1}')
        save_checkpoint(epoch, model, optimizer, "checkpoints", train_dataset, mesh_path)

    print("Training Finished")

    visualize()

