import numpy as np
from scipy.spatial import KDTree


def minimum_chamfer_distance(vertices, predicted_labels, GT_labels):
    pass
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

