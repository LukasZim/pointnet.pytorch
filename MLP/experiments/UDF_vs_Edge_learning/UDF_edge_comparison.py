import time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, GenerateMeshNormals

from MLP.metrics import calculate_n_minimum_chamfer_values, n_chamfer_values_deltaconv
from MLP.model.MLP import MLP_constant
from MLP.model.deltanet_regression import DeltaNetRegression
from MLP.model.loss import adjusted_l1_loss
from MLP.train import train_model, run_epoch
from MLP.train_deltaconv import train_deltaconv, evaluate
from MLP.visualize import load_mesh_from_file
import deltaconv.transforms as T


# create a function that takes complexity as input
# and outputs a DeltaConv model of that complexity
def create_DeltaConv_model(complexity):
    return DeltaNetRegression(9, complexity * [128], 2, embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1)

if __name__ == '__main__':
    # set some global variables
    max_epochs = 50

    # define the datasets for MLP
    path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/"
    mesh_path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj"

    from MLP.data_loader.data_loader import FractureDataLoader, FractureGeomDataset
    train_dataloader_mlp, train_dataset_mlp = FractureDataLoader(path, type="train")
    test_dataloader_mlp, test_dataset_mlp = FractureDataLoader(path, type="test")
    validate_dataloader_mlp, validate_dataset_mlp = FractureDataLoader(path, type="validate")
    mesh = load_mesh_from_file(mesh_path)

    #define everything for DeltaConv
    path = "/home/lukasz/Documents/pointnet.pytorch/MLP/datasets"
    dataset_name = "bunny"
    batch_size = 6
    num_workers = 4


    # Apply pre-transformations: normalize, get mesh normals, and sample points on the mesh.
    pre_transform = Compose([
        T.NormalizeScale(),
        GenerateMeshNormals(),
        # T.SamplePoints(args.num_points * args.sampling_margin, include_normals=True, include_labels=True),
        # T.GeodesicFPS(args.num_points)
    ])

    # pre_transform = T.GeodesicFPS(args.num_points)
    # Transformations during training: random scale, rotation, and translation.
    transform = Compose([
        T.RandomScale((0.8, 1.2)),
        T.RandomRotate(360, axis=2),
        T.RandomTranslateGlobal(0.1)
    ])

    train_dataset = FractureGeomDataset(path, 'train', dataset_name=dataset_name, pre_transform=pre_transform, )
    test_dataset = FractureGeomDataset(path, 'test', dataset_name=dataset_name, pre_transform=pre_transform)
    validation_dataset = FractureGeomDataset(path, 'validation', dataset_name=dataset_name, pre_transform=pre_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


    chamfer_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False
    )


    tensorboard_writer = SummaryWriter(log_dir=f"runs/experiment_complexity_{time.strftime('%Y%m%d-%H%M%S')}")
    csv_filename = f'complexity_comparison_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    # create a loop that loops over complexity values in a range

    print("Start Deltaconv training")

    delta_conv_model = create_DeltaConv_model(2)
    trained_deltaconv_model = train_deltaconv(writer=tensorboard_writer, epochs= max_epochs, model=delta_conv_model, train_loader=train_loader, test_loader=test_loader, validation_loader=validation_loader, complexity=complexity,
                    chamfer_loader=chamfer_loader, validation_dataset=validation_dataset)



    validation_loss_deltaconv, _ = evaluate(trained_deltaconv_model, 'cuda', validation_loader, adjusted_l1_loss)
    validation_loss_deltaconv = np.mean(validation_loss_deltaconv)
    edge_chamfer_value_deltaconv, num_non_fractures_deltaconv, _ = n_chamfer_values_deltaconv(chamfer_loader,
                                                                               trained_deltaconv_model,
                                                                               num_chamfer_values=100, edge=True)




    print("start evaluation")
    # loop over the validation set
    # and calculate the loss for both models




    # also store in tensorboard (optional)
    tensorboard_writer.add_scalars(f'Validation_loss', {"MLP": validation_loss_mlp, "DeltaConv": validation_loss_deltaconv,}, complexity)

    tensorboard_writer.add_scalars(f'Validation_chamfer', {"MLP": edge_chamfer_value, 'DeltaConv': edge_chamfer_value_deltaconv}, complexity)

    tensorboard_writer.add_scalars(f'Validation_chamfer_non_fractures', {"MLP": num_non_fractures, 'DeltaConv': num_non_fractures_deltaconv}, complexity)


    data = {
        'complexity': complexity,
        'mlp_loss': validation_loss_mlp,
        'mlp_edge_chamfer_value': edge_chamfer_value,
        'mlp_num_non_fractures': num_non_fractures,

        'deltaconv_loss': validation_loss_deltaconv,
        'deltaconv_edge_chamfer_value': edge_chamfer_value_deltaconv,
        'deltaconv_num_non_fractures': num_non_fractures_deltaconv,
    }
    df = pd.DataFrame([data])
    df.to_csv(csv_filename, mode='a', header = (complexity ==1), index=False)
    print("done with complexity:", complexity)
    # write the PD object to file, so I can later use it to plot
    # for this use an appropriate name





