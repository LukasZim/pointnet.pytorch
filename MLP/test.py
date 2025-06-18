import os
import random
import time
from glob import glob

import torch
from sklearn.metrics import mean_squared_error, r2_score, normalized_mutual_info_score, adjusted_mutual_info_score, \
    v_measure_score
from torch.utils.tensorboard import SummaryWriter

from MLP.data_loader.data_loader import FractureDataLoader
from MLP.metrics import minimum_chamfer_distance, contour_chamfer_distance
from MLP.model.MLP import MLP
import numpy as np

from MLP.path import Path
from MLP.segmentation_approaches.divergence import DivergenceSegmentation
from MLP.segmentation_approaches.felzenszwalb.felzenszwalb import FelzensZwalbSegmentation
from MLP.segmentation_approaches.hierarchical_clustering import HierarchicalClustering
from MLP.segmentation_approaches.region_growing import RegionGrowing, LocalExtremes
from MLP.segmentation_approaches.watershed import Watershed
from MLP.train import run_epoch
from MLP.visualize import load_mesh_from_file
from sklearn.metrics import adjusted_rand_score

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


def visualize_UDF_polyscope(gradients, distances, GT_distances, mesh, model, part_labels, impulse, label_gt, use_sqrt_ratios=False):
    import polyscope as ps

    ps.set_window_size(1920, 1080)
    ps.init()


    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    ps.register_surface_mesh("UDF mesh", vertices, faces, smooth_shade=True)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("GT distance scalar", GT_distances, defined_on="vertices",
                                                        enabled=True)


    gradients = gradients.detach().cpu().numpy()

    print_array_properties(gradients)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("label scalar", part_labels, defined_on="vertices", enabled=True)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("label GT", label_gt, defined_on="vertices", enabled=False)

    ps.get_surface_mesh("UDF mesh").add_vector_quantity("gradients vector", gradients.reshape(-1, 3), defined_on="vertices",
                                                        enabled=False)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("distance scalar", distances, defined_on="vertices",
                                                        enabled=False)

    ps.get_surface_mesh("UDF mesh").add_scalar_quantity("difference", (distances - GT_distances), defined_on="vertices")

    # points = dense_PC(model, mesh, GT_distances)
    points = impulse[:3].reshape(-1, 3)
    ps.register_point_cloud("Impact point", points, enabled=True)

    # ps.register_point_cloud("Dense fracture PC", points, enabled=True)

    ps.look_at((0., 0., 2.5), (0, 0, 0))

    ps.screenshot("screenshots/screenshot_" + time.strftime("%Y%m%d-%H%M%S") + ".png", transparent_bg=True)
    ss = ps.screenshot_to_buffer(transparent_bg=False)
    ss3 = ss[:,:,:3]
    # tensorboard_writer.add_image("labelling", ss3, dataformats="HWC")
    # tensorboard_writer.close()
    ps.show()




def visualize():
    log_root = "runs"
    latest_run = max(glob(f"{log_root}/*"), key=os.path.getmtime)
    print(latest_run)
    tensorboard_writer = SummaryWriter(log_dir="runs/MLP")

    model = MLP(9).to('cuda')
    state = torch.load("checkpoints/980.pth")
    model.load_state_dict(state['state_dict'])
    # index_to_use = 63
    index_to_use = 69

    validate_dataloader, validate_dataset = FractureDataLoader(Path().path, type="validate")
    X, y, impulse, label_gt, label_edge = validate_dataset.get_GT(index_to_use)

    run_epoch(model, validate_dataloader, None, train=False)

    mesh_path = state['mesh_path']
    mesh = load_mesh_from_file(mesh_path)

    # random_vertex = np.asarray(mesh.vertices)[random.randint(0, len(mesh.vertices) - 1)]
    # impulse[0] = random_vertex[0]
    # impulse[1] = random_vertex[1]
    # impulse[2] = random_vertex[2]



    time_start = time.time()

    from MLP.train import run_model
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    impulse = torch.from_numpy(impulse).float()

    model.eval()
    outputs, test_data = run_model(X.to('cuda'), y.to('cuda'), impulse.to('cuda'), model.to('cuda'), train=False)
    test_targets = y.float()

    print("duration_eval: ", time.time() - time_start)
    # loss = loss_function(outputs, test_targets.reshape((test_targets.shape[0], 1)))

    outputs.sum().backward()
    gradients = test_data.grad
    # gradients = gradients.squeeze().permute(1,0)
    print("gradient:", test_data.grad)
    predicted_udf = outputs.squeeze().tolist()

    predicted_udf = np.array(predicted_udf)
    test_targets = np.array(test_targets)

    mse = mean_squared_error(test_targets, predicted_udf)
    r2 = r2_score(test_targets, predicted_udf)
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)


    tensorboard_writer.add_scalar("MSE", mse)
    tensorboard_writer.add_scalar("R2", r2)


    # labels = segment_UDF(mesh, predicted_udf, test_targets)
    region_growing_time = time.time()
    # div = DivergenceSegmentation(mesh, predicted_udf, test_targets, gradients[:,:3])
    # labels_div = div.calculate_divergence()
    # fzs = FelzensZwalbSegmentation(mesh, labels_div, test_targets, False)
    # # labels = fzs.segment(5, 20)
    #
    # labels = fzs.segment(500, 20)
    watershed = Watershed(mesh, predicted_udf)
    labels = watershed.calculate_watershed()

    # h_cluster = HierarchicalClustering(mesh, predicted_udf)
    # labels = h_cluster.calculate_hierarchical_clustering()

    # LE = LocalExtremes(mesh, test_targets, test_targets)
    # labels = LE.local_extremes()
    # region_growing = RegionGrowing(mesh, predicted_udf, test_targets)
    # labels = region_growing.calculate_region_growing()
    print("region growing duration: ", time.time() - region_growing_time)

    chamfer_time = time.time()
    chamfer, key_map = minimum_chamfer_distance(np.asarray(mesh.vertices), labels, label_gt)
    print("chamfer duration: ", time.time() - chamfer_time)
    print('chamfer distance =', chamfer)
    print("champfer key mapping", key_map)
    # print(np.max(labels_div))
    # duration = time.time() - time_start
    # print("duration:",duration)
    # labels = np.array([remap_label(label, key_map) for label in labels])


    rand_score = adjusted_rand_score(labels, label_gt)
    print("rand score", rand_score)

    norm_info_score = adjusted_mutual_info_score(labels, label_gt)
    print('adj info score =', norm_info_score)

    v_measure = v_measure_score(labels, label_gt)
    print("v_measure =", v_measure)

    print('chamfer distance =', chamfer)

    contour_chamfer_value = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles), labels, label_gt )

    print("contour chamfer distance =", contour_chamfer_value)

    visualize_UDF_polyscope(gradients[:, :3], predicted_udf, test_targets, mesh, model, labels, impulse, label_gt, tensorboard_writer)

def remap_label(label, key_map):
    for (new_label, old_label) in key_map:
        if old_label == label:
            return new_label
    return -1

if __name__ == '__main__':
    visualize()