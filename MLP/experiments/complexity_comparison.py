import time

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from MLP.data_loader.data_loader import FractureDataLoader
from MLP.metrics import calculate_n_minimum_chamfer_values
from MLP.model.MLP import MLP_constant
from MLP.model.deltanet_regression import DeltaNetRegression
from MLP.train import train_model, run_epoch
from MLP.visualize import load_mesh_from_file


# create a function that takes complexity as input
# and outputs a MLP of that complexity
def create_MLP_model(complexity):
    # for now use a constant layer_size and varying number of layers
    return MLP_constant(9, complexity, 128)


# create a function that takes complexity as input
# and outputs a DeltaConv model of that complexity
def create_DeltaConv_model(complexity):
    return DeltaNetRegression(9, complexity * [128], complexity, embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1)

if __name__ == '__main__':
    # set some global variables
    max_epochs = 2

    # define the datasets for MLP and DeltaConv
    path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/"
    mesh_path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj"

    train_dataloader_mlp, train_dataset_mlp = FractureDataLoader(path, type="train")
    test_dataloader_mlp, test_dataset_mlp = FractureDataLoader(path, type="test")
    validate_dataloader_mlp, validate_dataset_mlp = FractureDataLoader(path, type="validate")
    mesh = load_mesh_from_file(mesh_path)

    tensorboard_writer = SummaryWriter(log_dir=f"runs/experiment_complexity_{time.strftime('%Y%m%d-%H%M%S')}")
    csv_filename = f'complexity_comparison_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    # create a loop that loops over complexity values in a range
    for complexity in range(1, 11):

        # call the training of the MLP
        # and output the best performing model on the testing set
        mlp_model = create_MLP_model(complexity)

        trained_mlp_model = train_model(max_epochs, complexity, mlp_model, tensorboard_writer=tensorboard_writer, mesh=mesh,
                    train_dataloader=train_dataloader_mlp, train_dataset=train_dataset_mlp, test_dataloader=test_dataloader_mlp,
                    test_dataset=test_dataset_mlp)


        # TODO: call the training of the DeltaConv
        # and output the best performing model on the testing set

        # loop over the validation set
        # and calculate the loss for both models
        validation_loss_mlp = run_epoch(trained_mlp_model, validate_dataloader_mlp, optimizer=None, train=False)
        edge_chamfer_value, num_non_fractures = calculate_n_minimum_chamfer_values(validate_dataset_mlp, trained_mlp_model, mesh, num_chamfer_values=100, edge=True)


        #TODO: store these values in a PD object

        # also store in tensorboard (optional)
        tensorboard_writer.add_scalars(f'Validation_loss', {"MLP_loss": validation_loss_mlp, }, complexity)

        tensorboard_writer.add_scalars(f'Validation_chamfer', {"MLP_loss": edge_chamfer_value, }, complexity)

        tensorboard_writer.add_scalars(f'Validation_chamfer_non_fractures', {"MLP_loss": num_non_fractures, }, complexity)


        data = {'complexity': complexity, 'mlp_loss': validation_loss_mlp, 'mlp_edge_chamfer_value': edge_chamfer_value, 'mlp_num_non_fractures': num_non_fractures}
        df = pd.DataFrame([data])
        df.to_csv(csv_filename, mode='a', header = (complexity ==1), index=False)
    # write the PD object to file, so I can later use it to plot
    # for this use an appropriate name





