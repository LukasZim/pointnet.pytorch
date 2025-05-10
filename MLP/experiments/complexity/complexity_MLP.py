import time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, GenerateMeshNormals

from MLP.metrics import calculate_n_minimum_chamfer_values, n_chamfer_values_deltaconv
from MLP.model.MLP import MLP_constant
from MLP.model.deltanet_regression import DeltaNetRegression
from MLP.model.loss import adjusted_l1_loss, adjusted_l2_loss
from MLP.train import train_model, run_epoch
from MLP.train_deltaconv import train_deltaconv, evaluate, evaluate_with_time
from MLP.visualize import load_mesh_from_file
import deltaconv.transforms as T


# create a function that takes complexity as input
# and outputs a MLP of that complexity
def create_MLP_model(complexity):
    # for now use a constant layer_size and varying number of layers
    return MLP_constant(9, complexity, int(128 * (1 + complexity / 10)))



if __name__ == '__main__':
    # set some global variables
    max_epochs = 50
    dataset_name = "bunny"


    # define the datasets for MLP
    path = f"/home/lukasz/Documents/thesis_pointcloud/datasets/{dataset_name}/"
    mesh_path = f"/home/lukasz/Documents/thesis_pointcloud/datasets/{dataset_name}/{dataset_name}.obj"

    from MLP.data_loader.data_loader import FractureDataLoader, FractureGeomDataset
    train_dataloader_mlp, train_dataset_mlp = FractureDataLoader(path, type="train", batch_size=25)
    test_dataloader_mlp, test_dataset_mlp = FractureDataLoader(path, type="test")
    validate_dataloader_mlp, validate_dataset_mlp = FractureDataLoader(path, type="validate")
    mesh = load_mesh_from_file(mesh_path)




    tensorboard_writer = SummaryWriter(log_dir=f"runs/mlp_experiment_complexity_{time.strftime('%Y%m%d-%H%M%S')}")
    csv_filename = f'mlp_complexity_comparison_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    # create a loop that loops over complexity values in a range
    for complexity in range(0, 50):
        print("CURRENT COMPLEXITY: ", complexity)

        mlp_model = create_MLP_model(complexity).to('cuda')

        trained_mlp_model = train_model(max_epochs, complexity, mlp_model, tensorboard_writer=tensorboard_writer, mesh=mesh,
                    train_dataloader=train_dataloader_mlp, train_dataset=train_dataset_mlp, test_dataloader=test_dataloader_mlp,
                    test_dataset=test_dataset_mlp, mesh_path=mesh_path,)



        print("start evaluation")
        # loop over the validation set
        # and calculate the loss for both models
        _, mlp_losses, durations = run_epoch(trained_mlp_model, validate_dataloader_mlp, optimizer=None, train=False)
        mlp_losses_list = []
        mlp_durations_list = []
        for i in range(5):
            _, mlp_losses, durations = run_epoch(trained_mlp_model, validate_dataloader_mlp, optimizer=None,
                                                 train=False)
            mlp_losses_list.append(mlp_losses)
            mlp_durations_list.append(durations)
        mlp_losses = np.mean(mlp_losses_list, axis=0)
        durations  = np.mean(mlp_durations_list, axis=0)

        MLP_evaluation_time = np.mean(durations)
        time_start = time.time()
        edge_chamfer_value, num_non_fractures, edge_chamfer_values_list = calculate_n_minimum_chamfer_values(validate_dataset_mlp, trained_mlp_model, mesh, num_chamfer_values=100000, edge=True)
        MLP_chamfer_time = time.time() - time_start


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
        tensorboard_writer.add_scalars(f'Validation_loss', {"MLP": validation_loss_mlp, }, complexity)

        tensorboard_writer.add_scalars(f'Validation_chamfer', {"MLP": edge_chamfer_value, }, complexity)

        tensorboard_writer.add_scalars(f'Validation_chamfer_non_fractures', {"MLP": num_non_fractures, }, complexity)


        data = {
            'complexity': complexity,

            'mlp_loss': validation_loss_mlp,
            'mlp_validation_variance': validation_variance_mlp,
            'mlp_validation_median': validation_median_mlp,
            'mlp_validation_min': validation_min_mlp,
            'mlp_validation_max': validation_max_mlp,
            'MLP_evaluation_time': MLP_evaluation_time,
            'MLP_chamfer_time': MLP_chamfer_time,
            'MLP_evaluation_size': validate_dataset_mlp.get_GT_size(),
            'MLP_losses': mlp_losses,

            'mlp_edge_chamfer_value': edge_chamfer_value,
            'mlp_num_non_fractures': num_non_fractures,
            'mlp_chamfer_variance': chamfer_variance_mlp,
            'mlp_chamfer_median': chamfer_median_mlp,
            'mlp_chamfer_min': chamfer_min_mlp,
            'mlp_chamfer_max': chamfer_max_mlp,
            'mlp_chamfer_values_list': edge_chamfer_values_list,

        }
        df = pd.DataFrame([data])
        df.to_csv(csv_filename, mode='a', header = (complexity ==0), index=False)
        print("done with complexity:", complexity)
    # write the PD object to file, so I can later use it to plot
    # for this use an appropriate name





