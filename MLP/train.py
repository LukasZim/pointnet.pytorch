import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from MLP.data_loader.data_loader import FractureData, FractureDataLoader
from MLP.model import model
from MLP.model.loss import l1_loss
from MLP.model.model import MLP
from MLP.visualize import load_mesh_from_file


def save_checkpoint(epoch, model, optimizer, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, f'{path}/{epoch}.pth')

path = "/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/0.csv"

dataloader, X, y, X_train, y_train, X_test, y_test = FractureDataLoader(path)
mlp = MLP(3)
loss_function = l1_loss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)

for epoch in range(0, 100):
    print(f'Starting Epoch {epoch + 1}')

    current_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        optimizer.zero_grad()

        outputs = mlp(inputs)

        loss = loss_function(outputs, targets)

        loss.backward()

        optimizer.step()

        current_loss += loss.item()

        if i == 0:
            print(f"loss after mini-batch %5d: %.3f" % (i + 1, current_loss / 50))
            current_loss = 0.0

    print(f'Finished Epoch {epoch + 1}')
    if epoch % 5 == 0:
        save_checkpoint(epoch, mlp, optimizer, "checkpoints")

print("Training Finished")

test_data = torch.from_numpy(X).float()
test_data.requires_grad = True
test_targets = torch.from_numpy(y).float()

mlp.eval()

# with torch.no_grad():
outputs = mlp(test_data)

# loss = loss_function(outputs, test_targets.reshape((test_targets.shape[0], 1)))

outputs.sum().backward()
gradients = test_data.grad
print("gradient:", test_data.grad)
predicted_labels = outputs.squeeze().tolist()

predicted_labels = np.array(predicted_labels)
test_targets = np.array(test_targets)

mse = mean_squared_error(test_targets, predicted_labels)
r2 = r2_score(test_targets, predicted_labels)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


def visualize_UDF(vertices, distances, distances_GT, use_sqrt_ratios=False):
    """
    Takes a mesh and a set of distances per vertex and Visualizes these on the mesh.
    :param mesh: mesh to visualize
    :param distances: list of distances per vertex
    :param use_sqrt_ratios: Whether to use square roots to exaggerate differences in distance close to contact points
    :return: Nothing, opens a window with the visualization
    """

    import open3d as o3d
    if use_sqrt_ratios:
        ratios = np.sqrt(distances / np.max(distances))
    else:
        ratios = distances / np.max(distances)
    colors = 1 - (ratios[:, None] * np.array([1.0, 1.0, 1.0]))
    # print(ratios)
    # print(colors)

    mesh = load_mesh_from_file("/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj")

    if use_sqrt_ratios:
        ratios_GT = np.sqrt(distances_GT / np.max(distances))
    else:
        ratios_GT = distances_GT / np.max(distances)
    colors_GT = 1 - (ratios_GT[:, None] * np.array([1.0, 1.0, 1.0]))
    # print(ratios_GT)
    # print(colors_GT)

    mesh_GT = load_mesh_from_file("/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj")

    distances_diff = np.abs(distances_GT - distances)
    if use_sqrt_ratios:
        ratios_diff = np.sqrt(distances_diff / np.max(distances))
    else:
        ratios_diff = distances_diff / np.max(distances)
    ratios_diff = ratios_diff * 10
    colors_diff = (ratios_diff[:, None] * np.array([1.0, 1.0, 1.0]))
    mesh_diff = load_mesh_from_file("/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj")

    # # pcd = o3d.geometry.PointCloud()
    # # v = vertices.detach().cpu().numpy()
    # print("converted")
    # # pcd.points = o3d.utility.Vector3dVector(v)
    # print("set points")
    # # c=colors
    # # pcd.colors = o3d.utility.Vector3dVector(c)

    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh_GT.compute_vertex_normals()
    mesh_GT.vertex_colors = o3d.utility.Vector3dVector(colors_GT)
    translation_vector = np.array([1.5, 0, 0])

    # Create a translation matrix
    translation_matrix = np.eye(4)  # 4x4 identity matrix
    translation_matrix[:3, 3] = translation_vector
    mesh_GT.transform(translation_matrix)

    mesh_diff.compute_vertex_normals()
    mesh_diff.vertex_colors = o3d.utility.Vector3dVector(colors_diff)
    translation_vector = np.array([-1.5, 0, 0])

    # Create a translation matrix
    translation_matrix = np.eye(4)  # 4x4 identity matrix
    translation_matrix[:3, 3] = translation_vector
    mesh_diff.transform(translation_matrix)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set camera parameters if needed
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.5)

    # Add mesh to visualizer
    vis.add_geometry(mesh)
    vis.add_geometry(mesh_GT)
    vis.add_geometry(mesh_diff)

    # Configure to disable all light effects
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([1.0, 1.0, 1.0])  # White background
    opt.show_coordinate_frame = False
    opt.light_on = False
    # opt.mesh_show_wireframe = True

    # Start the visualizer
    vis.run()

    # Destroy the visualizer window
    vis.destroy_window()


def visualize_UDF_polyscope(gradients, distances, use_sqrt_ratios=False):
    import polyscope as ps

    ps.set_window_size(1920, 1080)
    ps.init()

    mesh = load_mesh_from_file("/home/lukasz/Documents/thesis_pointcloud/datasets/bunny/bunny.obj")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    ps.register_surface_mesh("UDF mesh", vertices, faces, smooth_shade=True)

    gradients = gradients.detach().cpu().numpy()

    lens = np.linalg.norm(gradients, axis=1)
    print(lens)
    print(np.max(lens))
    print(np.min(lens))
    print(np.mean(lens))
    print(np.median(lens))

    ps.get_surface_mesh("UDF mesh").add_vector_quantity("gradients vector", gradients, defined_on="vertices",
                                                        enabled=True)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("distance scalar", distances, defined_on="vertices",
                                                        enabled=True)

    ps.look_at((0., 0., 2.5), (0, 0, 0))

    ps.screenshot("screenshots/screenshot_" + time.strftime("%Y%m%d-%H%M%S") + ".png", transparent_bg=True)
    ps.show()


visualize_UDF_polyscope(gradients, predicted_labels)

# visualize_UDF(vertices=test_data, distances=predicted_labels, distances_GT=y)
