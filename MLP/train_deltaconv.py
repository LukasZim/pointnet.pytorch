import os, time, argparse
import os.path as osp

import numpy as np
from progressbar import progressbar

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.transforms import Compose, GenerateMeshNormals
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from MLP.metrics import n_chamfer_values_deltaconv
from MLP.model.deltanet_regression import DeltaNetRegression
from MLP.model.loss import l2_loss, l1_loss, custom_loss
import deltaconv.transforms as T
from deltaconv.models import DeltaNetSegmentation

from deltaconv.experiments.utils import calc_loss

from MLP.visualize import create_mesh_from_faces_and_vertices, load_mesh_from_file


def train(args, writer, train_loader, test_loader, validation_loader, validation_dataset, chamfer_loader, model=None, complexity=0):
    # Data preparation
    # ----------------

    # Path to the dataset folder
    # The dataset will be downloaded if it is not yet available in the given folder.



    # Load datasets.

    # And setup DataLoaders for each dataset.

    # Model and optimization
    # ----------------------

    # Create the model.
    if model is None:
        model = DeltaNetRegression(
            in_channels=9,   # There are eight segmentation classes
            conv_channels=[256] * 4,  # We use 8 convolution layers, each with 128 channels
            # conv_channels=[32]*4,                   # This also works with fewer layers and channels, e.g., 6 layers and 32 channels
            mlp_depth=3,  # Each convolution uses MLPs with only one layer (i.e., perceptrons)
            embedding_size=1024,  # Embed the features in 512 dimensions after convolutions
            num_neighbors=args.k,  # The number of neighbors is given as an argument
            grad_regularizer=args.grad_regularizer,  # The regularizer value is given as an argument
            grad_kernel_width=args.grad_kernel,  # The kernel width is given as an argument
        ).to(args.device)
    else:
        model = model.to(args.device)

    loss_function = custom_loss

    if not args.evaluating:
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
        # Train the model
        # ---------------

        best_validation = float('inf')
        best_validation_test_score = 0
        for epoch in tqdm(range(0, args.epochs )):
            training_loss = train_epoch(epoch, model, args.device, optimizer, train_loader, writer, loss_function)
            torch.cuda.empty_cache()
            validation_accuracy = np.mean(evaluate(model, args.device, validation_loader, loss_function, ))
            torch.cuda.empty_cache()
            writer.add_scalar('validation accuracy', validation_accuracy, epoch)
            test_accuracy = np.mean(evaluate(model, args.device, test_loader, loss_function, ))
            writer.add_scalar('test accuracy', test_accuracy, epoch)
            writer.add_scalars("Loss", {f"train_DC_{complexity}": training_loss,
                                                    f"Test_DC_{complexity}": validation_accuracy}, epoch)

            chamfer_value, num_non_fractures, _ = n_chamfer_values_deltaconv(chamfer_loader,
                                       model,
                                       num_chamfer_values=10, edge=True)

            writer.add_scalar(f'DC_chamfer_{complexity}', chamfer_value, epoch)
            writer.add_scalar(f'DC_non_fracture_{complexity}', num_non_fractures, epoch)

            torch.cuda.empty_cache()
            print("deltaconv validation and test accuracy: ", validation_accuracy, test_accuracy)
            if validation_accuracy < best_validation:
                best_validation = validation_accuracy
                best_validation_test_score = test_accuracy
                torch.save(model.state_dict(), osp.join(args.checkpoint_dir, 'best.pt'))
            scheduler.step()
            # if epoch % 10 == 0:
            #     visualize_model_output(model, test_loader, args.device, test_dataset)
    else:
        model.load_state_dict(torch.load(args.checkpoint))
        best_validation_test_score = np.mean(evaluate(model, args.device, test_loader, loss_function, ))

    print("Test accuracy: {}".format(best_validation_test_score))
    # visualize_model_output(model, test_loader, args.device, test_dataset)
    return model


def visualize_model_output(model, loader, device, dataset):
    model.eval()
    limit = 1
    i = 0

    mesh_path = "/home/lukasz/Documents/pointnet.pytorch/MLP/datasets/bunny/bunny.obj"
    mesh = load_mesh_from_file(mesh_path)
    for batch in loader:
        if i >= limit:
            break
        i += 1
        data = batch
        # vertices =  data.pos.cpu().numpy()
        # faces = dataset.mesh_triangles
        # vertices = dataset.mesh_vertices

        faces = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)

        out = model(data.to(device))
        for i in range(batch.num_graphs):
            out_i = out[batch.batch == i]
            outputs = out_i.detach().cpu().numpy().reshape(-1)

            data = batch[i]

            import polyscope as ps

            ps.set_window_size(1920, 1080)
            ps.init()



            ps.register_surface_mesh("UDF mesh", vertices, faces, smooth_shade=True)
            ps.get_surface_mesh("UDF mesh").add_scalar_quantity("predicted", outputs, defined_on="vertices",
                                                                enabled=True)
            if hasattr(data, "norm"):
                ps.get_surface_mesh("UDF mesh").add_vector_quantity('normals', data.norm.cpu().numpy(), defined_on="vertices", enabled=False)
            ps.get_surface_mesh("UDF mesh").add_scalar_quantity("GT", data.gt_label.cpu().numpy(), defined_on="vertices", enabled=False)
            ps.get_surface_mesh("UDF mesh").add_scalar_quantity("gt udf", data.y.cpu().numpy(), defined_on="vertices", enabled=False)

            ps.register_point_cloud("XD", vertices, enabled=False)
            ps.get_point_cloud("XD").add_scalar_quantity("GT distance scalar", outputs, )

            ps.register_point_cloud("transformed_points", data.pos.cpu().numpy(), enabled=False)
            if hasattr(data, "norm"):
                ps.get_point_cloud("transformed_points").add_vector_quantity("normal2s", data.norm.cpu().numpy(),  enabled=False)
            ps.get_point_cloud("transformed_points").add_scalar_quantity("UDF", outputs, enabled=True)

            ps.look_at((0., 0., 2.5), (0, 0, 0))
            ps.show()


def train_epoch(epoch, model, device, optimizer, loader, writer, loss_function):
    """Train the model for one iteration on each item in the loader."""
    model.train()
    losses = []

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        out = model(data.to(device))
        out = out.squeeze()
        loss = loss_function(out, data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        torch.cuda.empty_cache()
        del data , out, loss
    writer.add_scalar('training loss',
                      np.mean(losses),
                      epoch)

    model.train()
    return np.mean(losses)


def evaluate(model, device, loader, loss_function):
    """Evaluate the model for on each item in the loader."""
    model.eval()
    losses = []
    # print("evaluata")
    visualized = False
    for data in loader:
        data = data.to(device)

        # data.x = data.x.clone().detach().requires_grad_(True)
        out = model(data)
        out = out.squeeze()

        if not visualized:
            visualized = True

            # visualize(mesh, out.detach().cpu().numpy(), data.gt_label.cpu().numpy(), data.gt_label.cpu().numpy(), data.gt_label.cpu().numpy(), data.gt_label.cpu().numpy(), data.y.cpu().numpy(), None )
            # visualize(mesh, outputs.detach().cpu().numpy(), data.gt_label.cpu().numpy(), labels_region_growing,
            #           labels_fzs, fzs_div_labels, data.y.cpu().numpy(), gradients[:, :3].detach().cpu().numpy())

        loss = loss_function(out, data.y)
        # loss.backward()
        # print(data.x.grad)
        # np.mean(np.abs(data.x.grad.cpu().numpy()), axis=0)
        losses.append(loss.item())


        torch.cuda.empty_cache()
        del data , out, loss
        torch.cuda.empty_cache()
    return losses


def evaluate_with_time(model, device, loader, loss_function):
    """Evaluate the model for on each item in the loader."""
    model.eval()
    losses = []
    times = []
    # print("evaluata")
    for data in loader:
        data = data.to(device)

        # data.x = data.x.clone().detach().requires_grad_(True)
        start_model = time.time()
        out = model(data)
        out = out.squeeze()
        duration = time.time() - start_model

        loss = loss_function(out, data.y)
        # loss.backward()
        # print(data.x.grad)
        # np.mean(np.abs(data.x.grad.cpu().numpy()), axis=0)
        losses.append(loss.item())
        times.append(duration)


        torch.cuda.empty_cache()
        del data , out, loss
        torch.cuda.empty_cache()
    return losses, times





if __name__ == "__main__":

    from data_loader.data_loader import FractureGeomDataset
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeltaNet Segmentation')
    # Optimization hyperparameters.
    parser.add_argument('--batch_size', type=int, default=6, metavar='batch_size',
                        help='Size of batch (default: 8)')
    parser.add_argument('--epochs', type=int, default=300, metavar='num_epochs',
                        help='Number of episode to train (default: 50)')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Number of points to use (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                        help='Learning rate (default: 0.005)')

    # DeltaConv hyperparameters.
    parser.add_argument('--k', type=int, default=20, metavar='K',
                        help='Number of nearest neighbors to use (default: 20)')
    parser.add_argument('--grad_kernel', type=float, default=2, metavar='h',
                        help='Kernel size for WLS, as a factor of the average edge length (default: 1)')
    parser.add_argument('--grad_regularizer', type=float, default=0.001, metavar='lambda',
                        help='Regularizer lambda to use for WLS (default: 0.001)')

    # Dataset generation arguments.
    parser.add_argument('--sampling_margin', type=int, default=8, metavar='sampling_margin',
                        help='The number of points to sample before using FPS to downsample (default: 8)')

    # Logging and debugging.
    parser.add_argument('--logdir', type=str, default='', metavar='logdir',
                        help='Root directory of log files. Log is stored in LOGDIR/runs/EXPERIMENT_NAME/TIME. (default: FILE_PATH)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Evaluation.
    # parser.add_argument('--checkpoint', type=str, default='/home/lukasz/Documents/pointnet.pytorch/MLP/runs/shapeseg/06Feb25_17_33/checkpoints/best.pt',
    #                     help='Path to the checkpoint to evaluate. The script will only evaluate if a path is given.')

    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to the checkpoint to evaluate. The script will only evaluate if a path is given.')

    args = parser.parse_args()

    # If a checkpoint is given, evaluate the model rather than training.
    args.evaluating = args.checkpoint != ''

    # Determine the device to run the experiment
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Name the experiment, used to store logs and checkpoints.
    args.experiment_name = 'shapeseg'
    run_time = time.strftime("%d%b%y_%H_%M", time.localtime(time.time()))

    writer = None
    if not args.evaluating:
        # Set log directory and create TensorBoard writer in log directory.
        if args.logdir == '':
            args.logdir = osp.dirname(osp.realpath(__file__))
        args.logdir = osp.join(args.logdir, 'runs', args.experiment_name, run_time)
        writer = SummaryWriter(args.logdir)

        # Create directory to store checkpoints.
        args.checkpoint_dir = osp.join(args.logdir, 'checkpoints')
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        # Write experimental details to log directory.
        experiment_details = args.experiment_name + '\n--\nSettings:\n--\n'
        for arg in vars(args):
            experiment_details += '{}: {}\n'.format(arg, getattr(args, arg))
        with open(os.path.join(args.logdir, 'settings.txt'), 'w') as f:
            f.write(experiment_details)

        # And show experiment details in console.
        print(experiment_details)
        print('---')
        print('Training...')
    else:
        print('Evaluating {}...'.format(args.experiment_name))

    # Start training process
    torch.manual_seed(args.seed)

    path = "/home/lukasz/Documents/pointnet.pytorch/MLP/datasets"
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
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)




    train(args, writer, train_loader, test_loader, validation_loader, validation_dataset, chamfer_loader)

def train_deltaconv(writer, epochs, model, train_loader, test_loader, validation_loader, validation_dataset, chamfer_loader, complexity):
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeltaNet Segmentation')
    # Optimization hyperparameters.
    parser.add_argument('--batch_size', type=int, default=6, metavar='batch_size',
                        help='Size of batch (default: 8)')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='num_epochs',
                        help='Number of episode to train (default: 50)')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Number of points to use (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='Learning rate (default: 0.005)')

    # DeltaConv hyperparameters.
    parser.add_argument('--k', type=int, default=20, metavar='K',
                        help='Number of nearest neighbors to use (default: 20)')
    parser.add_argument('--grad_kernel', type=float, default=1, metavar='h',
                        help='Kernel size for WLS, as a factor of the average edge length (default: 1)')
    parser.add_argument('--grad_regularizer', type=float, default=0.001, metavar='lambda',
                        help='Regularizer lambda to use for WLS (default: 0.001)')

    # Dataset generation arguments.
    parser.add_argument('--sampling_margin', type=int, default=8, metavar='sampling_margin',
                        help='The number of points to sample before using FPS to downsample (default: 8)')

    # Logging and debugging.
    parser.add_argument('--logdir', type=str, default='', metavar='logdir',
                        help='Root directory of log files. Log is stored in LOGDIR/runs/EXPERIMENT_NAME/TIME. (default: FILE_PATH)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Evaluation.
    # parser.add_argument('--checkpoint', type=str, default='/home/lukasz/Documents/pointnet.pytorch/MLP/runs/shapeseg/06Feb25_17_33/checkpoints/best.pt',
    #                     help='Path to the checkpoint to evaluate. The script will only evaluate if a path is given.')

    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to the checkpoint to evaluate. The script will only evaluate if a path is given.')

    args = parser.parse_args()

    # If a checkpoint is given, evaluate the model rather than training.
    args.evaluating = args.checkpoint != ''

    # Determine the device to run the experiment
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Name the experiment, used to store logs and checkpoints.
    args.experiment_name = 'shapeseg'
    run_time = time.strftime("%d%b%y_%H_%M", time.localtime(time.time()))

    if not args.evaluating:
        # Set log directory and create TensorBoard writer in log directory.
        if args.logdir == '':
            args.logdir = osp.dirname(osp.realpath(__file__))
        args.logdir = osp.join(args.logdir, 'runs', args.experiment_name, run_time)

        # Create directory to store checkpoints.
        args.checkpoint_dir = osp.join(args.logdir, 'checkpoints')
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        # Write experimental details to log directory.
        experiment_details = args.experiment_name + '\n--\nSettings:\n--\n'
        for arg in vars(args):
            experiment_details += '{}: {}\n'.format(arg, getattr(args, arg))
        with open(os.path.join(args.logdir, 'settings.txt'), 'w') as f:
            f.write(experiment_details)

        # And show experiment details in console.
        print(experiment_details)
        print('---')
        print('Training...')
    else:
        print('Evaluating {}...'.format(args.experiment_name))

    # Start training process
    torch.manual_seed(args.seed)
    return train(args, writer, train_loader, test_loader, validation_loader, validation_dataset, chamfer_loader, model=model, complexity=complexity)
