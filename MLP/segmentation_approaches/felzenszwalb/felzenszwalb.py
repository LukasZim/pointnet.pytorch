import math

import networkx as nx
import numpy as np
from .disjoint_set import DisjointSet
from .utils import smoothen, difference, get_random_rgb_image



class FelzensZwalbSegmentation:
    def __init__(self, mesh, predicted_udf, gt_udf, not_divergent):
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        self.predicted_udf = predicted_udf
        self.gt_udf = gt_udf
        self.normal = not_divergent

    def segment_graph(self, num_vertices, num_edges, edges, c):
        edges[0: num_edges, :] = edges[edges[0: num_edges, 2].argsort()]
        u = DisjointSet(num_vertices)
        threshold = np.zeros(shape=num_vertices, dtype=float)
        for i in range(num_vertices):
            threshold[i] = c
        for i in range(num_edges):
            pedge = edges[i, :]
            a = u.find(pedge[0])
            b = u.find(pedge[1])
            if a != b:
                if (pedge[2] <= threshold[a]) and (pedge[2] <= threshold[b]):
                    u.join(a, b)
                    a = u.find(a)
                    threshold[a] = pedge[2] + (c / u.size(a))
        return u

    def segment(self, k, min_size):


        nxg = self.create_graph()

        # build graph
        edges_size = nxg.number_of_edges()
        edges = np.zeros(shape=(edges_size, 3), dtype=object)
        num = 0

        for u, v, data in nxg.edges(data=True):
            edges[num, 0] = u
            edges[num, 1] = v
            if self.normal:
                edges[num, 2] = math.pow(10 * (np.max(self.predicted_udf) - data['udf']), 10)
            else:
                edges[num, 2] = data['udf']
            # edges[num, 2] = np.max(self.predicted_udf) - data['udf']
            num += 1


        u = self.segment_graph(len(self.vertices), num, edges, k)
        for i in range(num):
            a = u.find(edges[i, 0])
            b = u.find(edges[i, 1])
            if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
                u.join(a, b)
        num_cc = u.num_sets()
        vals = []
        for i in range(len(self.vertices)):
            val = u.find(i)
            vals.append(val)

        # print(np.unique(vals, return_counts=True))
        return np.array(vals)



    def create_graph(self):
        G = nx.Graph()
        # G.add_nodes_from(self.vertices)

        edges = [(x, y) for [a, b, c] in self.faces for x, y in [(a, b), (a, c), (b, c)]]
        # G.add_edges_from(all_edges)
        G.add_edges_from([(edges[i][0], edges[i][1], {"udf": (self.predicted_udf[edges[i][0]] + self.predicted_udf[edges[i][0]]) / 2}) for i in range(len(edges))])

        # node_value_dict = dict(zip(G.nodes, self.predicted_udf))
        # nx.set_node_attributes(G, node_value_dict, 'udf')
        return G