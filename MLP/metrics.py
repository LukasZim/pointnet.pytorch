import time

import numpy as np
import torch
from scipy.spatial import KDTree

from MLP.helper_functions import append_impulse_to_data
from MLP.region_growing import RegionGrowing


def calculate_n_minimum_chamfer_values(dataset, model, mesh, num_values=10):
    start_time = time.time()
    chamfer_values = []

    for _ in range(num_values):
        pcd, gt_udf, impulse, gt_labels, edge_labels = dataset.get_random_GT()
        labels, predicted_udf = get_model_output(mesh, pcd, impulse, model, gt_udf)

        chamfer, _ = minimum_chamfer_distance(np.asarray(mesh.vertices), labels, gt_labels)

        chamfer_values.append(chamfer)
    print("Chamfer values calculation Duration:", time.time() - start_time)
    return np.mean(chamfer_values)


def get_model_output(mesh, pcd, impulse, model, gt_udf):
    from MLP.train import run_model
    pcd = torch.from_numpy(pcd).float().unsqueeze(0)
    impulse = torch.from_numpy(impulse).float().unsqueeze(0)
    gt_udf = torch.from_numpy(gt_udf).float()
    outputs, _ = run_model(pcd, gt_udf, impulse, model, train=False)

    predicted_udf = np.array(outputs.squeeze().tolist())

    region_growing = RegionGrowing(mesh, predicted_udf, gt_udf)
    labels = region_growing.calculate_region_growing()
    return labels, predicted_udf

def get_model_output_from_index(index, dataset, mesh, model):
    pcd, gt_udf, impulse, gt_labels, edge_labels = dataset.get_GT(index)
    labels, predicted_udf = get_model_output(mesh, pcd, impulse, model, gt_udf)

    return  labels, predicted_udf, gt_labels, gt_udf




def minimum_chamfer_distance(vertices, predicted_labels, GT_labels):
    # create a list of all groups of vertices for predicted labels
    predicted_combined = list(zip(predicted_labels, vertices))
    predicted_vertex_groups = {}
    for label, vertex in predicted_combined:
        if not label in  predicted_vertex_groups.keys():
            predicted_vertex_groups[label] = []
        predicted_vertex_groups[label].append(vertex)

    # do the same for GT label groups
    gt_combined = list(zip(GT_labels, vertices))
    gt_vertex_groups = {}
    for label, vertex in gt_combined:
        if not label in gt_vertex_groups.keys():
            gt_vertex_groups[label] = []
        gt_vertex_groups[label].append(vertex)

    # loop over all predicted segments
    distances = []
    key_map = []
    for predicted_label, predicted_group in predicted_vertex_groups.items():
        # set min distance to infinity
        min_distance = float('inf')
        cur_gt_label = -1

        # loop over all GT segments
        for gt_label, gt_group in gt_vertex_groups.items():
            # if closer, overwrite minimum
            distance = chamfer_distance(predicted_group, gt_group)
            if distance < min_distance:
                min_distance = distance
                cur_gt_label = gt_label

        # store minimum
        distances.append(min_distance)
        key_map.append((cur_gt_label, predicted_label))

    # return mean minimum distance
    return np.mean(distances), key_map


def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points.
    :param A: a list of points.
    :param B: a list of points.
    :return: chamfer distance between A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]

    tree = KDTree(A)
    dist_B = tree.query(B)[0]

    return np.mean(dist_A) + np.mean(dist_B)

