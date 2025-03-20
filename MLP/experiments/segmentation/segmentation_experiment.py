import time

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, GenerateMeshNormals

from MLP.metrics import calculate_n_minimum_chamfer_values, n_chamfer_values_deltaconv, contour_chamfer_distance
from MLP.model.MLP import MLP_constant, MLP
from MLP.model.deltanet_regression import DeltaNetRegression
from MLP.model.loss import custom_loss
from MLP.segmentation_approaches.divergence import DivergenceSegmentation
from MLP.segmentation_approaches.felzenszwalb.felzenszwalb import FelzensZwalbSegmentation
from MLP.segmentation_approaches.minimum_cut import minimum_cut_segmentation
from MLP.segmentation_approaches.region_growing import RegionGrowing
from MLP.train import train_model, run_epoch
from MLP.train_deltaconv import train_deltaconv, evaluate
from MLP.visualize import load_mesh_from_file
import deltaconv.transforms as T

if __name__ == "__main__":
    # define the datasets for MLP
    path = "../datasets/bunny/"
    mesh_path = "../datasets/bunny/bunny.obj"

    # load MLP validation dataset
    from MLP.data_loader.data_loader import FractureDataLoader, FractureGeomDataset
    train_dataloader_mlp, train_dataset_mlp = FractureDataLoader(path, type="train")
    test_dataloader_mlp, test_dataset_mlp = FractureDataLoader(path, type="test")
    validate_dataloader_mlp, validate_dataset_mlp = FractureDataLoader(path, type="validate")
    mesh = load_mesh_from_file(mesh_path)

    #define everything for DeltaConv
    path = "/home/lukasz/Documents/pointnet.pytorch/MLP/datasets/"
    dataset_name = "bunny"
    batch_size = 6
    num_workers = 4

    # load DC validation dataset
    pre_transform = Compose((
        T.NormalizeScale(),
        GenerateMeshNormals(),
        # T.SamplePoints(args.num_points * args.sampling_margin, include_normals=True, include_labels=True),
        # T.GeodesicFPS(args.num_points)
    ))


    validation_dataset = FractureGeomDataset(path, 'validation', dataset_name=dataset_name, pre_transform=pre_transform)

    validation_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)

    #TODO: load alternative datasets (other mesh, other simulator etc.)

    # load trained MLP model

    MLP_model = MLP(9)
    state = torch.load("checkpoints/980.pth")
    MLP_model.load_state_dict(state['state_dict'])

    # load trained DC model
    DC_path = "/home/lukasz/Documents/pointnet.pytorch/MLP/runs/shapeseg/11Feb25_01_58/checkpoints/best.pt"
    state_dict = torch.load(DC_path)
    DC_model = DeltaNetRegression(9)
    DC_model.load_state_dict(state_dict)

    MLP_region_growing = []
    MLP_region_growing_non_fractures = 0
    MLP_region_growing_time = []
    MLP_fzs = []
    MLP_fzs_non_fractures = 0
    MLP_fzs_time = []
    MLP_fzs_div = []
    MLP_fzs_div_non_fractures = 0
    MLP_fzs_div_time = []

    # loop over MLP dataset
    for i in range(validate_dataset_mlp.get_GT_size()):

        # get model UDF output
        X, y, impulse, label_gt, label_edge = validate_dataset_mlp.get_GT(i)

        from MLP.train import run_model

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        impulse = torch.from_numpy(impulse).float()

        MLP_model.eval()
        outputs, test_data = run_model(X, y, impulse, MLP_model, train=False)
        test_targets = y.float()

        outputs.sum().backward()
        gradients = test_data.grad

        print("gradient:", test_data.grad)
        predicted_udf = outputs.squeeze().tolist()

        predicted_udf = np.array(predicted_udf)
        test_targets = np.array(test_targets)

        # region growing
        region_growing_time = time.time()

        region_growing = RegionGrowing(mesh, predicted_udf, test_targets)
        labels = region_growing.calculate_region_growing()
        region_growing_duration = time.time() - region_growing_time

        print("region growing duration: ", region_growing_duration)
        region_growing_chamfer, _ = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                                             labels, label_gt)

        # FelzensZwalb

        t = time.time()
        fzs = FelzensZwalbSegmentation(mesh, labels, test_targets)
        labels = fzs.segment(2, 20)

        fzs_duration = time.time() - t
        print("region growing duration: ", fzs_duration)
        fzs_chamfer, _ = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                                             labels, label_gt)



        # Felzenszwalb divergence
        t = time.time()
        div = DivergenceSegmentation(mesh, predicted_udf, test_targets, gradients[:, :3])
        labels = div.calculate_divergence()
        fzs = FelzensZwalbSegmentation(mesh, labels, test_targets)
        labels = fzs.segment(0, 10)

        fzs_div_duration = time.time() - t

        chamfer_time = time.time()
        fzs_div_chamfer, _ = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles),labels, label_gt)

        # minimum cut
        # t = time.time()
        # a = minimum_cut_segmentation()


        # store results
        # region growing
        if region_growing_chamfer == float('inf'):
            MLP_region_growing_non_fractures += 1
        else:
            MLP_region_growing.append(region_growing_chamfer)
        MLP_region_growing_time.append(region_growing_duration)

        # felzenszwalb
        if fzs_chamfer == float('inf'):
            MLP_fzs_non_fractures += 1
        else:
            MLP_fzs.append(fzs_chamfer)
        MLP_fzs_time.append(fzs_duration)

        # felzenszwalb divergence
        if fzs_div_chamfer == float('inf'):
            MLP_fzs_div_non_fractures += 1
        else:
            MLP_fzs_div.append(fzs_div_chamfer)
        MLP_fzs_div_time.append(fzs_div_duration)

    # do same loop for DC

    DC_region_growing = []
    DC_region_growing_non_fractures = 0
    DC_region_growing_time = []
    DC_fzs = []
    DC_fzs_non_fractures = 0
    DC_fzs_time = []
    DC_fzs_div = []
    DC_fzs_div_non_fractures = 0
    DC_fzs_div_time = []
    for data in validation_dataset:
        outputs = DC_model(data).squeeze()

        predicted_udf = np.array(outputs.squeeze().tolist())

        gt_udf = data.y.cpu().numpy()

        region_growing = RegionGrowing(mesh, predicted_udf, gt_udf)
        labels = region_growing.calculate_region_growing()

        gt_labels = data.gt_label.cpu().numpy()
        chamfer = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles), labels, gt_labels)

        if chamfer == float('inf'):
            num_non_fractures += 1
        else:
            chamfer_values.append(chamfer)



    # save results to disk


def store(region_growing_list, region_growing_non_fractures, region_growing_time_list, fzs_list, fzs_non_fractures, fzs_time_list, fzs_div_list, fzs_div_non_fractures, fzs_div_time_list, predicted_udf, test_targets, label_gt, gradients):
    # region growing
    region_growing_time = time.time()

    region_growing = RegionGrowing(mesh, predicted_udf, test_targets)
    labels = region_growing.calculate_region_growing()
    region_growing_duration = time.time() - region_growing_time

    print("region growing duration: ", region_growing_duration)
    region_growing_chamfer, _ = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                                         labels, label_gt)

    # FelzensZwalb

    t = time.time()
    fzs = FelzensZwalbSegmentation(mesh, labels, test_targets)
    labels = fzs.segment(2, 20)

    fzs_duration = time.time() - t
    print("region growing duration: ", fzs_duration)
    fzs_chamfer, _ = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                              labels, label_gt)

    # Felzenszwalb divergence
    t = time.time()
    div = DivergenceSegmentation(mesh, predicted_udf, test_targets, gradients[:, :3])
    labels = div.calculate_divergence()
    fzs = FelzensZwalbSegmentation(mesh, labels, test_targets)
    labels = fzs.segment(0, 10)

    fzs_div_duration = time.time() - t

    chamfer_time = time.time()
    fzs_div_chamfer, _ = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles), labels,
                                                  label_gt)

    # minimum cut
    # t = time.time()
    # a = minimum_cut_segmentation()

    # store results
    # region growing
    if region_growing_chamfer == float('inf'):
        region_growing_non_fractures += 1
    else:
        region_growing_list.append(region_growing_chamfer)
    region_growing_time_list.append(region_growing_duration)

    # felzenszwalb
    if fzs_chamfer == float('inf'):
        fzs_non_fractures += 1
    else:
        fzs_list.append(fzs_chamfer)
    fzs_time_list.append(fzs_duration)

    # felzenszwalb divergence
    if fzs_div_chamfer == float('inf'):
        fzs_div_non_fractures += 1
    else:
        fzs_div_list.append(fzs_div_chamfer)
    fzs_div_time_list.append(fzs_div_duration)

    return region_growing_list, region_growing_non_fractures, region_growing_time_list, fzs_list, fzs_non_fractures, fzs_time_list, fzs_div_list, fzs_div_non_fractures, fzs_div_time_list