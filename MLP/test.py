import time

import torch
from sklearn.metrics import mean_squared_error, r2_score

from MLP.model.model import MLP
import numpy as np

from MLP.visualize import load_mesh_from_file


def calculate_gradient(model, point):
    t = torch.from_numpy(point).float()
    t.requires_grad = True

    output = model(t)
    output.sum().backward()
    grad = t.grad

    return grad.detach().cpu().numpy()

def calculate_UDF(model, point):
    t = torch.from_numpy(point).float()
    output = model(t)
    return output.detach().cpu().numpy()

def dense_PC(model, mesh, GT_distances, delta = 0.05, num_steps=4, n = 1000):
    # grab m points from the mesh
    np.random.seed(0)
    vertices = np.asarray(mesh.vertices)
    P_init = vertices
    # throw away the points far away (f(p) > lambda)
    mask = GT_distances < delta
    P_init = P_init[mask]

    # for i ... num_steps:
    for i in range(num_steps):
        for index, point in enumerate(P_init):
            # p = p - f(p) * gradient(f(p)) / || gradient(f(p)) ||
            gradient = calculate_gradient(model, point)
            new_point = point - calculate_UDF(model, point) * gradient / np.linalg.norm(gradient)
            P_init[index] = new_point
            # P_init[index] = vertices[np.argmin(np.linalg.norm(new_point - vertices, axis=1))]


            # Pdense: n points drawn from p with replacement
    # P_dense = np.random.choice(P_init, size=n, replace=True)
    P_dense = P_init[np.random.randint(P_init.shape[0], size=n), :]

    # Pdense = { p + d | p in Pdense and d stddev(0, delta/3)}
    normal_dist = np.random.normal(0, delta / 30, size=P_dense.shape)
    P_dense = P_dense + normal_dist

    # for i ... num_steps:
    for i in range(num_steps):
        for index, point in enumerate(P_dense):
    #         # p = p - f(p) * gradient(f(p)) / || gradient(f(p)) ||
            gradient = calculate_gradient(model, point)
            P_dense[index] = point - calculate_UDF(model, point) * gradient / np.linalg.norm(gradient) * 0.5
        # p = p - f(p) * gradient(f(p)) / || gradient(f(p)) ||

    mask = (calculate_UDF(model, P_dense) < delta).ravel()
    P_dense = P_dense[mask]

    # P_dense[index] = vertices[np.argmin(np.linalg.norm(new_point - vertices, axis=1))]
    for index, point in enumerate(P_dense):
        P_dense[index] = vertices[np.argmin(np.linalg.norm(point - vertices, axis=1))]
    return P_dense



def print_array_properties(arr):
    arr = np.linalg.norm(arr, axis=1)

    print("max", np.max(arr))
    print("min", np.min(arr))
    print("mean", np.mean(arr))
    print("std", np.std(arr))
    print("median", np.median(arr))

def visualize_UDF_polyscope(gradients, distances, GT_distances, mesh_path, model, use_sqrt_ratios=False):
    import polyscope as ps

    ps.set_window_size(1920, 1080)
    ps.init()

    mesh = load_mesh_from_file(mesh_path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    ps.register_surface_mesh("UDF mesh", vertices, faces, smooth_shade=True)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("GT distance scalar", GT_distances, defined_on="vertices",
                                                        enabled=True)


    gradients = gradients.detach().cpu().numpy()

    print_array_properties(gradients)

    ps.get_surface_mesh("UDF mesh").add_vector_quantity("gradients vector", gradients, defined_on="vertices",
                                                        enabled=True)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("distance scalar", distances, defined_on="vertices",
                                                        enabled=True)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("difference", (distances - GT_distances), defined_on="vertices")

    points = dense_PC(model, mesh, GT_distances)

    ps.register_point_cloud("Dense fracture PC", points, enabled=True)

    ps.look_at((0., 0., 2.5), (0, 0, 0))

    ps.screenshot("screenshots/screenshot_" + time.strftime("%Y%m%d-%H%M%S") + ".png", transparent_bg=True)
    ps.show()

def visualize():
    mlp = MLP(3)
    state = torch.load("checkpoints/95.pth")
    mlp.load_state_dict(state['state_dict'])

    X = state['X']
    y = state['y']
    mesh_path = state['mesh_path']


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





    visualize_UDF_polyscope(gradients, predicted_labels, test_targets, mesh_path, mlp)

if __name__ == '__main__':
    visualize()