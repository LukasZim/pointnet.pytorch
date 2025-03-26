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

from tqdm import tqdm

def visualize(mesh, outputs, gt_label, labels_region_growing, labels_fzs, fzs_div_labels, gt_udf, gradients, index, using_deltaconv):
    import polyscope as ps

    ps.set_window_size(1920, 1080)
    ps.init()

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    ps.register_surface_mesh("UDF mesh", vertices, faces, smooth_shade=True)
    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("predicted", outputs,
                                                        defined_on="vertices",
                                                        enabled=True)
    # if hasattr(data, "norm"):
    #     ps.get_surface_mesh("UDF mesh").add_vector_quantity('normals', data.norm.cpu().numpy(),
    #                                                         defined_on="vertices", enabled=False)
    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("GT", gt_label, defined_on="vertices",
                                                        enabled=False)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("region", labels_region_growing, defined_on="vertices",
                                                        enabled=False)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("felzenzshalb", labels_fzs, defined_on="vertices",
                                                        enabled=False)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("felzenzshalb div", fzs_div_labels, defined_on="vertices",
                                                        enabled=False)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("gt udf", gt_udf, defined_on="vertices",
                                                        enabled=False)

    ps.get_surface_mesh("UDF mesh").add_vector_quantity("gradients", gradients,
                                                                 enabled=False)


    ps.look_at((0., 0., 2.5), (0, 0, 0))
    ps.show()

    # ps.screenshot(f"/home/lukasz/Documents/pointnet.pytorch/MLP/experiments/segmentation/output_images/{'MLP' if not using_deltaconv else 'DC'}_{index}.png")

import polyscope as ps
import os

ps.set_window_size(1920, 1080)
ps.init()

def visualize_toggle(mesh, outputs, gt_label, labels_region_growing, labels_fzs, fzs_div_labels, gt_udf, gradients, index, using_deltaconv):


    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    ps.register_surface_mesh("UDF mesh", vertices, faces, smooth_shade=True)

    # Dictionary of scalar quantities to visualize
    scalar_quantities = {
        "predicted": outputs,
        "GT": gt_label,
        "region": labels_region_growing,
        "felzenzshalb": labels_fzs,
        "felzenzshalb div": fzs_div_labels,
        "gt udf": gt_udf
    }

    # Register scalar quantities but enable only one at a time for screenshots
    for name, data in scalar_quantities.items():
        ps.get_surface_mesh("UDF mesh").add_scalar_quantity(name, data, defined_on="vertices", enabled=True)

        # Set the view
        ps.look_at((0., 0., 2.5), (0, 0, 0))

        # Define the output path
        output_path = f"/home/lukasz/Documents/pointnet.pytorch/MLP/experiments/segmentation/output_images/{'MLP' if not using_deltaconv else 'DC'}_{index}_{name}.png"

        # Take a screenshot
        ps.screenshot(output_path)

        # Disable the scalar quantity after screenshot to avoid overlap
        # ps.get_surface_mesh("UDF mesh").remove_scalar_quantity(name)

    # Add vector quantity (gradients) but don't enable by default
    # ps.get_surface_mesh("UDF mesh").add_vector_quantity("gradients", gradients, enabled=False)
    # ps.shutdown()
    ps.remove_all_structures()


def store(region_growing_list, region_growing_non_fractures, region_growing_time_list, fzs_list, fzs_non_fractures,
          fzs_time_list, fzs_div_list, fzs_div_non_fractures, fzs_div_time_list, predicted_udf, test_targets, label_gt,
          gradients, using_deltaconv, index):
    # region growing
    region_growing_time = time.time()

    region_growing = RegionGrowing(mesh, predicted_udf, test_targets)
    labels_region_growing = region_growing.calculate_region_growing()
    region_growing_duration = time.time() - region_growing_time

    # print("region growing duration: ", region_growing_duration)
    region_growing_chamfer = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                                         labels_region_growing, label_gt)

    # FelzensZwalb

    t = time.time()
    fzs = FelzensZwalbSegmentation(mesh, predicted_udf, test_targets)
    labels_fzs = fzs.segment(2, 20)

    fzs_duration = time.time() - t
    # print("felzenzswalb duration: ", fzs_duration)
    fzs_chamfer = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                              labels_fzs, label_gt)

    # Felzenszwalb divergence
    t = time.time()
    div = DivergenceSegmentation(mesh, predicted_udf, test_targets, gradients[:, :3])
    div_output = div.calculate_divergence()
    fzs = FelzensZwalbSegmentation(mesh, div_output, test_targets)
    fzs_div_labels = fzs.segment(0, 10)

    fzs_div_duration = time.time() - t

    # print("felzenzswalb divergence duration: ", fzs_div_duration)
    fzs_div_chamfer = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles), fzs_div_labels,
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
    print(region_growing_chamfer, fzs_chamfer, fzs_div_chamfer)
    if using_deltaconv:
        visualize_toggle(mesh, outputs.detach().cpu().numpy(), data.gt_label.cpu().numpy(), labels_region_growing,
                         labels_fzs, fzs_div_labels, data.y.cpu().numpy(), gradients[:, :3].detach().cpu().numpy(),
                         index, using_deltaconv)
        # visualize(mesh, outputs.detach().cpu().numpy(), data.gt_label.cpu().numpy(), labels_region_growing,
        #                  labels_fzs, fzs_div_labels, data.y.cpu().numpy(), gradients[:, :3].detach().cpu().numpy(),
        #                  index, using_deltaconv)
    else:
        visualize_toggle(mesh, outputs.detach().cpu().numpy(), label_gt, labels_region_growing, labels_fzs,
                         fzs_div_labels, y.cpu().numpy(), gradients[:, :3].detach().cpu().numpy(), index,
                         using_deltaconv)
        # visualize(mesh, outputs.detach().cpu().numpy(), label_gt, labels_region_growing, labels_fzs,
        #                  fzs_div_labels, y.cpu().numpy(), gradients[:, :3].detach().cpu().numpy(), index,
        #                  using_deltaconv)

    return region_growing_list, region_growing_non_fractures, region_growing_time_list, fzs_list, fzs_non_fractures, fzs_time_list, fzs_div_list, fzs_div_non_fractures, fzs_div_time_list


if __name__ == "__main__":
    import pickle
    import datetime

    # Generate filename with current date and time
    filename = datetime.datetime.now().strftime("Segmentation_experiment_%Y-%m-%d_%H-%M-%S") + ".pkl"

    # define the datasets for MLP
    path = "/home/lukasz/Documents/pointnet.pytorch/MLP/datasets/bunny"
    mesh_path = "/home/lukasz/Documents/pointnet.pytorch/MLP/datasets/bunny/bunny.obj"

    # load MLP validation dataset
    from MLP.data_loader.data_loader import FractureDataLoader, FractureGeomDataset

    train_dataloader_mlp, train_dataset_mlp = FractureDataLoader(path, type="train")
    test_dataloader_mlp, test_dataset_mlp = FractureDataLoader(path, type="test")
    validate_dataloader_mlp, validate_dataset_mlp = FractureDataLoader(path, type="validate")
    mesh = load_mesh_from_file(mesh_path)

    # define everything for DeltaConv
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

    # TODO: load alternative datasets (other mesh, other simulator etc.)

    # load trained MLP model

    MLP_model = MLP(9)
    state = torch.load("/home/lukasz/Documents/pointnet.pytorch/MLP/checkpoints/980.pth")
    MLP_model.load_state_dict(state['state_dict'])

    # load trained DC model
    # DC_path = "/home/lukasz/Documents/pointnet.pytorch/MLP/runs/shapeseg/21Mar25_02_35/checkpoints/best.pt"
    DC_path = "/home/lukasz/Documents/pointnet.pytorch/MLP/runs/shapeseg/26Mar25_00_41/checkpoints/best.pt"
    state_dict = torch.load(DC_path)
    # DC_model = DeltaNetRegression(in_channels=9, conv_channels=[128] * 5, mlp_depth=3, embedding_size=512, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1).to('cuda')
    DC_model = DeltaNetRegression(in_channels=9, conv_channels=[256] * 4, mlp_depth=3, embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=2).to('cuda')

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

    # # loop over MLP dataset
    for i in tqdm(range(validate_dataset_mlp.get_GT_size())):
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

        # print("gradient:", test_data.grad)
        predicted_udf = outputs.squeeze().tolist()

        predicted_udf = np.array(predicted_udf)
        test_targets = np.array(test_targets)

        MLP_region_growing, MLP_region_growing_non_fractures, MLP_region_growing_time, MLP_fzs, MLP_fzs_non_fractures, MLP_fzs_time, MLP_fzs_div, MLP_fzs_div_non_fractures, MLP_fzs_div_time = store(
            MLP_region_growing, MLP_region_growing_non_fractures, MLP_region_growing_time, MLP_fzs,
            MLP_fzs_non_fractures, MLP_fzs_time, MLP_fzs_div, MLP_fzs_div_non_fractures, MLP_fzs_div_time,
            predicted_udf, test_targets, label_gt,
            gradients, False, i
        )

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
    for index, data in enumerate(tqdm(validation_loader)):
        data = data.to('cuda')
        data.pos.requires_grad = True
        outputs = DC_model(data).squeeze()
        outputs.sum().backward()
        gradients = data.pos.grad.cpu()

        predicted_udf = np.array(outputs.squeeze().tolist())

        gt_udf = data.y.cpu().numpy()

        region_growing = RegionGrowing(mesh, predicted_udf, gt_udf)
        labels = region_growing.calculate_region_growing()

        gt_labels = data.gt_label.cpu().numpy()
        chamfer = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles), labels, gt_labels)

        DC_region_growing, DC_region_growing_non_fractures, DC_region_growing_time, DC_fzs, DC_fzs_non_fractures, DC_fzs_time, DC_fzs_div, DC_fzs_div_non_fractures, DC_fzs_div_time = store(
            DC_region_growing, DC_region_growing_non_fractures, DC_region_growing_time, DC_fzs, DC_fzs_non_fractures,
            DC_fzs_time, DC_fzs_div, DC_fzs_div_non_fractures, DC_fzs_div_time, predicted_udf, gt_udf, data.gt_label, gradients, True,
            index)



    # save results to disk

    # Save data to file
    with open(filename, 'wb') as file:
        pickle.dump({
            "DC_region_growing": DC_region_growing,
            "DC_region_growing_non_fractures": DC_region_growing_non_fractures,
            "DC_region_growing_time": DC_region_growing_time,
            "DC_fzs": DC_fzs,
            "DC_fzs_non_fractures": DC_fzs_non_fractures,
            "DC_fzs_time": DC_fzs_time,
            "DC_fzs_div": DC_fzs_div,
            "DC_fzs_div_non_fractures": DC_fzs_div_non_fractures,
            "DC_fzs_div_time": DC_fzs_div_time,
            "MLP_region_growing": MLP_region_growing,
            "MLP_region_growing_non_fractures": MLP_region_growing_non_fractures,
            "MLP_region_growing_time": MLP_region_growing_time,
            "MLP_fzs": MLP_fzs,
            "MLP_fzs_non_fractures": MLP_fzs_non_fractures,
            "MLP_fzs_time": MLP_fzs_time,
            "MLP_fzs_div": MLP_fzs_div,
            "MLP_fzs_div_non_fractures": MLP_fzs_div_non_fractures,
            "MLP_fzs_div_time": MLP_fzs_div_time,
        }, file)
