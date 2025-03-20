import time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, GenerateMeshNormals

from MLP.metrics import calculate_n_minimum_chamfer_values, n_chamfer_values_deltaconv
from MLP.model.MLP import MLP_constant
from MLP.model.deltanet_regression import DeltaNetRegression
from MLP.model.loss import custom_loss
from MLP.train import train_model, run_epoch
from MLP.train_deltaconv import train_deltaconv, evaluate
from MLP.visualize import load_mesh_from_file
import deltaconv.transforms as T


# create a function that takes complexity as input
# and outputs a MLP of that complexity
def create_MLP_model(complexity):
    # for now use a constant layer_size and varying number of layers
    return MLP_constant(9, complexity, 128)


# create a function that takes complexity as input
# and outputs a DeltaConv model of that complexity
def create_DeltaConv_model(complexity):
    return DeltaNetRegression(9, complexity * [128], 2, embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1)

if __name__ == '__main__':
    # set some global variables
    max_epochs = 50

    # define the datasets for MLP
    path = "../datasets/bunny/"
    mesh_path = "../datasets/bunny/bunny.obj"

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


    # Apply pre-transformations: normalize, get mesh normals, and sample points on the mesh.
    pre_transform = Compose((
        T.NormalizeScale(),
        GenerateMeshNormals(),
        # T.SamplePoints(args.num_points * args.sampling_margin, include_normals=True, include_labels=True),
        # T.GeodesicFPS(args.num_points)
    ))

    # pre_transform = T.GeodesicFPS(args.num_points)
    # Transformations during training: random scale, rotation, and translation.
    transform = Compose((
        T.RandomScale((0.8, 1.2)),
        T.RandomRotate(360, axis=2),
        T.RandomTranslateGlobal(0.1)
    ))

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
    for complexity in range(1, 8):
        print("CURRENT COMPLEXITY: ", complexity)
        print("Start Deltaconv training")

        delta_conv_model = create_DeltaConv_model(complexity)
        trained_deltaconv_model = train_deltaconv(writer=tensorboard_writer, epochs= max_epochs, model=delta_conv_model, train_loader=train_loader, test_loader=test_loader, validation_loader=validation_loader, complexity=complexity,
                        chamfer_loader=chamfer_loader, validation_dataset=validation_dataset)



        deltaconv_losses = evaluate(trained_deltaconv_model, 'cuda', validation_loader, custom_loss)
        edge_chamfer_value_deltaconv, num_non_fractures_deltaconv, edge_chamfer_values_list_deltaconv = n_chamfer_values_deltaconv(chamfer_loader,
                                                                                   trained_deltaconv_model,
                                                                                   num_chamfer_values=100, edge=True)

        validation_loss_deltaconv = np.mean(deltaconv_losses)
        validation_variance_deltaconv = np.var(deltaconv_losses)
        validation_median_deltaconv = np.median(deltaconv_losses)
        validation_min_deltaconv = np.min(deltaconv_losses) if len(deltaconv_losses) > 0 else float('inf')
        validation_max_deltaconv = np.max(deltaconv_losses) if len(deltaconv_losses) > 0 else float('inf')

        chamfer_mean_deltaconv = np.mean(edge_chamfer_values_list_deltaconv)
        chamfer_variance_deltaconv = np.var(edge_chamfer_values_list_deltaconv)
        chamfer_median_deltaconv = np.median(edge_chamfer_values_list_deltaconv)
        chamfer_min_deltaconv = np.min(edge_chamfer_values_list_deltaconv) if len(edge_chamfer_values_list_deltaconv) > 0 else float('inf')
        chamfer_max_deltaconv = np.max(edge_chamfer_values_list_deltaconv) if len(edge_chamfer_values_list_deltaconv) > 0 else float('inf')





        # call the training of the MLP
        # and output the best performing model on the testing set
        print("start MLP training")
        mlp_model = create_MLP_model(complexity)

        trained_mlp_model = train_model(max_epochs, complexity, mlp_model, tensorboard_writer=tensorboard_writer, mesh=mesh,
                    train_dataloader=train_dataloader_mlp, train_dataset=train_dataset_mlp, test_dataloader=test_dataloader_mlp,
                    test_dataset=test_dataset_mlp, mesh_path=mesh_path,)



        print("start evaluation")
        # loop over the validation set
        # and calculate the loss for both models
        _, mlp_losses = run_epoch(trained_mlp_model, validate_dataloader_mlp, optimizer=None, train=False)
        edge_chamfer_value, num_non_fractures, edge_chamfer_values_list = calculate_n_minimum_chamfer_values(validate_dataset_mlp, trained_mlp_model, mesh, num_chamfer_values=100, edge=True)


        validation_loss_mlp = np.mean(mlp_losses)
        validation_variance_mlp = np.var(mlp_losses)
        validation_median_mlp = np.median(mlp_losses)
        validation_min_mlp = np.min(mlp_losses) if len(mlp_losses) > 0 else float('inf')
        validation_max_mlp = np.max(mlp_losses) if len(mlp_losses) > 0 else float('inf')

        chamfer_mean_mlp = np.mean(edge_chamfer_values_list)
        chamfer_variance_mlp = np.var(edge_chamfer_values_list)
        chamfer_median_mlp = np.median(edge_chamfer_values_list)
        chamfer_min_mlp = np.min(edge_chamfer_values_list) if len(edge_chamfer_values_list) > 0 else float('inf')
        chamfer_max_mlp = np.max(edge_chamfer_values_list) if len(edge_chamfer_values_list) > 0 else float('inf')


        # also store in tensorboard (optional)
        tensorboard_writer.add_scalars(f'Validation_loss', {"MLP": validation_loss_mlp, "DeltaConv": validation_loss_deltaconv,}, complexity)

        tensorboard_writer.add_scalars(f'Validation_chamfer', {"MLP": edge_chamfer_value, 'DeltaConv': edge_chamfer_value_deltaconv}, complexity)

        tensorboard_writer.add_scalars(f'Validation_chamfer_non_fractures', {"MLP": num_non_fractures, 'DeltaConv': num_non_fractures_deltaconv}, complexity)


        data = {
            'complexity': complexity,

            'mlp_loss': validation_loss_mlp,
            'mlp_validation_variance': validation_variance_mlp,
            'mlp_validation_median': validation_median_mlp,
            'mlp_validation_min': validation_min_mlp,
            'mlp_validation_max': validation_max_mlp,

            'mlp_edge_chamfer_value': edge_chamfer_value,
            'mlp_num_non_fractures': num_non_fractures,
            'mlp_chamfer_variance': chamfer_variance_mlp,
            'mlp_chamfer_median': chamfer_median_mlp,
            'mlp_chamfer_min': chamfer_min_mlp,
            'mlp_chamfer_max': chamfer_max_mlp,


            'deltaconv_loss': validation_loss_deltaconv,
            'deltaconv_validation_variance': validation_variance_deltaconv,
            'deltaconv_validation_median': validation_median_deltaconv,
            'deltaconv_validation_min': validation_min_deltaconv,
            'deltaconv_validation_max': validation_max_deltaconv,

            'deltaconv_edge_chamfer_value': edge_chamfer_value_deltaconv,
            'deltaconv_num_non_fractures': num_non_fractures_deltaconv,
            'deltaconv_chamfer_variance': chamfer_variance_deltaconv,
            'deltaconv_chamfer_median': chamfer_median_deltaconv,
            'deltaconv_chamfer_min': chamfer_min_deltaconv,
            'deltaconv_chamfer_max': chamfer_max_deltaconv,
        }
        df = pd.DataFrame([data])
        df.to_csv(csv_filename, mode='a', header = (complexity ==1), index=False)
        print("done with complexity:", complexity)
    # write the PD object to file, so I can later use it to plot
    # for this use an appropriate name





