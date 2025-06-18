import numpy as np
import torch
from scipy.spatial import KDTree
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, mean_squared_error, r2_score
from tqdm import tqdm

from MLP.segmentation_approaches.felzenszwalb.felzenszwalb import FelzensZwalbSegmentation
from MLP.segmentation_approaches.minimum_cut import minimum_cut_segmentation
# from MLP.segmentation_approaches.region_growing import RegionGrowing
from MLP.segmentation_approaches.watershed import Watershed
from MLP.visualize import create_mesh_from_faces_and_vertices


def calculate_n_minimum_chamfer_values(dataset, model, mesh, num_chamfer_values=10, edge=False):
    chamfer_values = []
    num_non_fractures = 0

    for i in range(min(num_chamfer_values, dataset.get_GT_size())):
        pcd, gt_udf, impulse, gt_labels, edge_labels = dataset.get_GT(i)
        labels, predicted_udf = get_model_output(mesh, pcd, impulse, model, gt_udf)
        if edge:
            chamfer = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles), labels, gt_labels)
        else:
            chamfer, _ = minimum_chamfer_distance(np.asarray(mesh.vertices), labels, gt_labels)
        if chamfer == float('inf'):
            num_non_fractures += 1
        else:
            chamfer_values.append(chamfer)

    if len(chamfer_values) == 0:
        return -1, num_non_fractures, []


    return np.mean(chamfer_values), num_non_fractures, chamfer_values


def n_chamfer_values_deltaconv(loader, model, num_chamfer_values=10, edge=False, visualize=False):
    chamfer_values = []
    num_non_fractures = 0
    index = 0
    for data in (loader):
        if index >= num_chamfer_values:
            break
        index += 1

        mesh = create_mesh_from_faces_and_vertices(data.face.T, data.pos)
        labels, predicted_udf = get_model_output_DC(data, model, mesh, visualize)


        gt_labels = data.gt_label.cpu().numpy()
        if edge:
            chamfer = contour_chamfer_distance(np.asarray(mesh.vertices), np.asarray(mesh.triangles), labels, gt_labels)
        else:
            chamfer, _ = minimum_chamfer_distance(np.asarray(mesh.vertices), labels, gt_labels)
        if chamfer == float('inf'):
            num_non_fractures += 1
        else:
            chamfer_values.append(chamfer)

    if len(chamfer_values) == 0:
        return float('inf'), num_non_fractures, []


    return np.mean(chamfer_values), num_non_fractures, chamfer_values


def get_model_output(mesh, pcd, impulse, model, gt_udf):
    from MLP.train import run_model
    pcd = torch.from_numpy(pcd).float()
    impulse = torch.from_numpy(impulse).float()
    gt_udf = torch.from_numpy(gt_udf).float()
    outputs, _ = run_model(pcd, gt_udf, impulse, model,  train=False)

    predicted_udf = np.array(outputs.squeeze().tolist())

    # region_growing = RegionGrowing(mesh, predicted_udf, gt_udf)
    # labels = region_growing.calculate_region_growing()

    watershed = Watershed(mesh, predicted_udf)
    labels = watershed.calculate_watershed()







    # minimum_cut = minimum_cut_segmentation(mesh, predicted_udf, gt_udf)
    # labels = minimum_cut.calculate_minimum_cut()
    return labels, predicted_udf


def get_model_output_DC(data, model, mesh, show=False):
    data = data.to('cuda')

    pcd = data.pos.cpu().numpy()
    gt_udf = data.y.cpu().numpy()
    impulse = data.impulse.cpu().numpy()


    outputs = model(data).squeeze()

    predicted_udf = np.array(outputs.squeeze().tolist())

    # region_growing = RegionGrowing(mesh, predicted_udf, gt_udf)
    # labels = region_growing.calculate_region_growing()
    watershed = Watershed(mesh, predicted_udf)
    labels = watershed.calculate_watershed()

    from MLP.experiments.segmentation.segmentation_experiment import visualize
    if show:
        visualize(mesh, outputs.detach().cpu().numpy(), data.gt_label.cpu().numpy(), labels,
                  data.gt_label.cpu().numpy(), data.gt_label.cpu().numpy(), data.y.cpu().numpy(), None, impulse=impulse)
    # v
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
    A = np.array(A)
    B = np.array(B)

    tree = KDTree(B)
    dist_A = tree.query(A)[0]

    tree = KDTree(A)
    dist_B = tree.query(B)[0]

    return np.mean(dist_A) + np.mean(dist_B)

def contour_chamfer_distance(vertices, triangles, predicted_labels, gt_labels):
    # intialize a set of indices of edge points for predicted and gt
    predicted_set = set()
    gt_set = set()
    # loop over all edges
    for [t1,t2,t3] in triangles:
        edges = [(t1, t2), (t2, t3), (t3, t1)]
        for (v1, v2) in edges:
            # check if vertices on edge have the same label
            if not predicted_labels[v1] == predicted_labels[v2]:
                # if not add to either gt_set of predicted_set
                predicted_set.update((v1, v2))
            if not gt_labels[v1] == gt_labels[v2]:
                gt_set.update((v1, v2))
    predicted_edge_points = [vertices[x] for x in predicted_set]
    gt_edge_points = [vertices[x] for x in gt_set]
    # return chamfer distance between point_sets gt_set and predicted_set
    if len(predicted_edge_points) == 0 and not len(gt_edge_points) == 0:
        return float('inf')
    if len(gt_edge_points) == 0 and not len(predicted_edge_points) == 0:
        return float('inf')
    if len (predicted_edge_points) == 0 and len(gt_edge_points) == 0:
        return 0
    return chamfer_distance(predicted_edge_points, gt_edge_points)


# def calc_rand_values(loader, model, mesh, use_deltaconv=False):
#     if use_deltaconv:
#         return dc_calc_rand_values(loader, model)
#     return mlp_calc_rand_values(loader, model, mesh)


def dc_calc_rand_values(loader, model):
    rand_values = []
    for data in loader:

        mesh = create_mesh_from_faces_and_vertices(data.face.T, data.pos)
        labels, predicted_udf = get_model_output_DC(data, model, mesh)


        gt_labels = data.gt_label.cpu().numpy()
        rand_values.append(adjusted_rand_score(labels, gt_labels))
    return np.mean(rand_values)

def mlp_calc_rand_values(dataset, model, mesh):
    rand_values = []

    for i in range(dataset.get_GT_size()):
        pcd, gt_udf, impulse, gt_labels, edge_labels = dataset.get_GT(i)
        labels, predicted_udf = get_model_output(mesh, pcd, impulse, model, gt_udf)

        rand_values.append(adjusted_rand_score(labels, gt_labels))
    return np.mean(rand_values)


def dc_calc_norm_info_score(loader, model):
    adjusted_mutual_info_score_values = []
    for data in loader:
        mesh = create_mesh_from_faces_and_vertices(data.face.T, data.pos)
        labels, predicted_udf = get_model_output_DC(data, model, mesh)

        gt_labels = data.gt_label.cpu().numpy()
        adjusted_mutual_info_score_values.append(adjusted_mutual_info_score(labels, gt_labels))
    return np.mean(adjusted_mutual_info_score_values)


def mlp_calc_norm_info_score(dataset, model, mesh):
    adjusted_mutual_info_score_values = []

    for i in range(dataset.get_GT_size()):
        pcd, gt_udf, impulse, gt_labels, edge_labels = dataset.get_GT(i)
        labels, predicted_udf = get_model_output(mesh, pcd, impulse, model, gt_udf)

        adjusted_mutual_info_score_values.append(adjusted_mutual_info_score(labels, gt_labels))
    return np.mean(adjusted_mutual_info_score_values)


def dc_mse_loss(loader, model):
    mse_values = []
    for data in loader:
        mesh = create_mesh_from_faces_and_vertices(data.face.T, data.pos)
        _, predicted_udf = get_model_output_DC(data, model, mesh)

        gt_udf = data.y.cpu().numpy()
        mse_values.append(mean_squared_error(predicted_udf, gt_udf))
    return np.mean(mse_values)


def mlp_mse_loss(dataset, model, mesh):
    mse_values = []

    for i in range(dataset.get_GT_size()):
        pcd, gt_udf, impulse, gt_labels, edge_labels = dataset.get_GT(i)
        labels, predicted_udf = get_model_output(mesh, pcd, impulse, model, gt_udf)

        mse_values.append(mean_squared_error(predicted_udf, gt_udf))
    return np.mean(mse_values)

def dc_r2_values(loader, model):
    r2_values = []
    for data in loader:
        mesh = create_mesh_from_faces_and_vertices(data.face.T, data.pos)
        _, predicted_udf = get_model_output_DC(data, model, mesh)

        gt_udf = data.y.cpu().numpy()
        r2_values.append(r2_score(predicted_udf, gt_udf))
    return np.mean(r2_values)


def mlp_r2_values(dataset, model, mesh):
    r2_values = []

    for i in range(dataset.get_GT_size()):
        pcd, gt_udf, impulse, gt_labels, edge_labels = dataset.get_GT(i)
        labels, predicted_udf = get_model_output(mesh, pcd, impulse, model, gt_udf)

        r2_values.append(r2_score(predicted_udf, gt_udf))
    return np.mean(r2_values)