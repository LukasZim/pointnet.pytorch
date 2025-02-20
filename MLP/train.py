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
from MLP.metrics import minimum_chamfer_distance, calculate_n_minimum_chamfer_values, get_model_output_from_index, \
    contour_chamfer_distance
from MLP.model import MLP
from MLP.model.loss import *
from MLP.model.MLP import MLP, CNN, MLP_constant
from MLP.region_growing import RegionGrowing
from MLP.test import visualize
from MLP.visualize import load_mesh_from_file
from torch.utils.tensorboard import SummaryWriter
from MLP.model.deltanet_regression import DeltaNetRegression


# def do_visualize_quick(mesh, dataset, model):
#     import polyscope as ps
#     ps.set_window_size(1920, 1080)
#     ps.init()
#
#     pcd, udf, impulse, label_gt, label_edge = dataset.get_GT(0)
#     # model.eval()
#     model_output , _ = run_model(pcd, udf, impulse, model, train=False)
#
#     model.train()
#
#
#     vertices = np.asarray(mesh.vertices)
#     faces = np.asarray(mesh.triangles)
#
#     gt = udf
#     predicted = model_output.detach().numpy().squeeze()
#
#
#
#     ps.register_surface_mesh("UDF mesh", vertices, faces, smooth_shade=True)
#
#     ps.get_surface_mesh("UDF mesh").add_scalar_quantity("GT distance scalar", gt, defined_on="vertices",
#                                                         enabled=True)
#     ps.get_surface_mesh("UDF mesh").add_scalar_quantity("pred distance scalar", predicted, defined_on="vertices",
#                                                         enabled=True)
#
#     ps.look_at((0., 0., 2.5), (0, 0, 0))
#     ps.show()

def save_checkpoint(epoch, model, optimizer, path, dataset, mesh_path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'dataset': dataset,
        'mesh_path': mesh_path
    }
    torch.save(state, f'{path}/{epoch}.pth')

def run_model(points, udf_values, impulses, used_model, optimizer=None, train=False):
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points).float()
    if not isinstance(impulses, torch.Tensor):
        impulses = torch.tensor(impulses).float()
    if not isinstance(udf_values, torch.Tensor):
        udf_values = torch.tensor(udf_values).float()

    points, udf_values = points.float(), udf_values.float()

    if train:
        optimizer.zero_grad()

    points_with_impulses = torch.cat((points, impulses.unsqueeze(0).repeat(points.size(0), 1)), dim=1).float()
    if not train:
        points_with_impulses.requires_grad = True

    model_output = used_model(points_with_impulses)

    return model_output.squeeze(), points_with_impulses


def run_epoch(model, dataloader, optimizer, loss_function = custom_loss, train=True):
    training_loss = 0.0

    if train:
        model.train()
    else:
        model.eval()
    size = 0
    for i, data in enumerate(dataloader):
        [points, udf_values, impulses, gt_labels, label_edges] = data
        for index, (coordinates, udf_value, impulse, gt_label, label_edge) in enumerate(zip(points, udf_values, impulses, gt_labels, label_edges)):
            size += 1

            model_output, _ = run_model(coordinates, udf_value, impulse, model, optimizer=optimizer, train=train)
            loss = loss_function(model_output, udf_value)
            if train:
                loss.backward()
                optimizer.step()

            training_loss += loss.item()

            if i == 0:
                print(f"loss after mini-batch %5d: %.3f" % (i + 1, loss / 50))
                # if index ==0 and epoch % 50 == 0:
                #     if train:
                #         do_visualize_quick(mesh, dataset, model)

            # print(i)

    return training_loss / size

if __name__ == '__main__':
    tensorboard_writer = SummaryWriter(log_dir=f"runs/MLP_{time.strftime('%Y%m%d-%H%M%S')}")

    path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/"
    mesh_path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj"

    train_dataloader, train_dataset = FractureDataLoader(path, type="train")
    test_dataloader, test_dataset = FractureDataLoader(path, type="test")
    model = MLP_constant(9, 4, 256)

    loss_function = custom_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in tqdm(range(0, 200)):
        # print(f'Starting Epoch {epoch + 1}')

        mesh = load_mesh_from_file(mesh_path)

        training_loss = run_epoch(model, train_dataloader, optimizer, loss_function, train=True)
        testing_loss = run_epoch(model, test_dataloader, optimizer, loss_function, train=False)

        # chamfer_value = calculate_n_minimum_chamfer_values(test_dataset, model, mesh)
        # edge_chamfer_value = calculate_n_minimum_chamfer_values(test_dataset, model, mesh, edge=True)

        tensorboard_writer.add_scalars("Loss", {"Train": training_loss  , "Test": testing_loss}, epoch)
        time_start = time.time()
        if epoch % 10 == 100:
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
        # tensorboard_writer.add_scalar('chamfer', chamfer_value, epoch)
        # tensorboard_writer.add_scalar('edge_chamfer', edge_chamfer_value, epoch)


        # print(f'Finished Epoch {epoch + 1}')
        save_checkpoint(epoch, model, optimizer, "checkpoints", train_dataset, mesh_path)

    print("Training Finished")

    # visualize()

def train_model(num_epochs, complexity, model, tensorboard_writer, mesh, train_dataloader, train_dataset, test_dataloader, test_dataset, loss_function=custom_loss):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in tqdm(range(0, num_epochs)):
        training_loss = run_epoch(model, train_dataloader, optimizer, loss_function, train=True)
        testing_loss = run_epoch(model, test_dataloader, optimizer, loss_function, train=False)

        # chamfer_value = calculate_n_minimum_chamfer_values(test_dataset, model, mesh)
        edge_chamfer_value, num_non_fractures = calculate_n_minimum_chamfer_values(test_dataset, model, mesh, num_chamfer_values=10, edge=True)

        tensorboard_writer.add_scalars("Loss", {f"Train_MLP_{complexity}": training_loss  , f"Test_MLP_{complexity}": testing_loss }, epoch)

        if epoch == (num_epochs - 1):
            predicted_labels, predicted_udf, gt_labels, gt_udf = get_model_output_from_index(0, test_dataset, mesh, model)

            ss_predicted_labels = get_screenshot_polyscope(mesh, predicted_labels)
            ss_gt_labels = get_screenshot_polyscope(mesh, gt_labels)
            ss_gt_udf = get_screenshot_polyscope(mesh, gt_udf)
            ss_predicted_udf = get_screenshot_polyscope(mesh, predicted_udf)

            tensorboard_writer.add_image(f"predicted_labels_MLP_{complexity}", ss_predicted_labels, dataformats="HWC")
            tensorboard_writer.add_image(f"gt_labels_MLP_{complexity}", ss_gt_labels, dataformats="HWC")
            tensorboard_writer.add_image(f"gt_udf_MLP_{complexity}", ss_gt_udf, dataformats="HWC")
            tensorboard_writer.add_image(f"predicted_udf_MLP_{complexity}", ss_predicted_udf, dataformats="HWC")

        # tensorboard_writer.add_scalar('chamfer', chamfer_value, epoch)
        tensorboard_writer.add_scalar(f'edge_chamfer_MLP_{complexity}', edge_chamfer_value, epoch)
        tensorboard_writer.add_scalar(f'num_non_fractures_MLP_{complexity}', num_non_fractures, epoch)


        # # print(f'Finished Epoch {epoch + 1}')
        # save_checkpoint(epoch, model, optimizer, "checkpoints", train_dataset, mesh_path)

    print("Training Finished")
    return model

