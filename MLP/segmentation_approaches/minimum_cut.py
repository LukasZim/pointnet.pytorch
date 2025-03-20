import networkx as nx
import numpy as np
from networkx import minimum_cut


class minimum_cut_segmentation:
    def __init__(self, mesh, predicted_udf, gt_udf):
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        self.predicted_udf = predicted_udf
        self.gt_udf = gt_udf


    def calculate_minimum_cut(self):
        # pass
        # turn
        graph = self.create_graph()
        _, partition = minimum_cut(graph, np.random.choice(list(range(len(self.vertices)))), np.random.choice(list(range(len(self.vertices)))), capacity='udf')
        #TODO: use the partition somehow. Need to check what it looks like
        raise Exception("not implemented yet lol!!!!")


    def create_graph(self):
        G = nx.Graph()
        # G.add_nodes_from(self.vertices)

        edges = [(x, y) for [a, b, c] in self.faces for x, y in [(a, b), (a, c), (b, c)]]
        # G.add_edges_from(all_edges)
        G.add_edges_from([(edges[i][0], edges[i][1], {"udf": (self.predicted_udf[edges[i][0]] + self.predicted_udf[edges[i][0]]) / 2}) for i in range(len(edges))])

        # node_value_dict = dict(zip(G.nodes, self.predicted_udf))
        # nx.set_node_attributes(G, node_value_dict, 'udf')
        return G