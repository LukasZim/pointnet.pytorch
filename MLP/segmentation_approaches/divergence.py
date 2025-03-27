import networkx as nx
import numpy as np


class DivergenceSegmentation:
    def __init__(self, mesh, predicted_udf, gt_udf, gradients):
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        self.predicted_udf = predicted_udf
        self.gt_udf = gt_udf
        self.gradients = gradients


    def calculate_divergence(self):
        divergence = []
        G = self.create_graph()

        for i in G.nodes:
            div = 0
            pos_i = np.array(self.vertices[i])
            F_i = np.array(self.gradients[i])


            num_neighbours = 0
            for j in G.neighbors(i):
                num_neighbours += 1
                pos_j = np.array(self.vertices[j])
                F_j = np.array(self.gradients[j])

                direction = pos_j - pos_i
                distance = np.linalg.norm(direction)
                if distance > 0:
                    unit_vector = direction / distance
                    flux = (F_j - F_i) @ unit_vector  # Dot product for flow change
                    div += flux / distance  # Normalize by distance

            div = div / num_neighbours
            divergence.append(div if div > 0 else 0)
            # divergence.append(div)

        return np.array(divergence)

    def create_graph(self):
        G = nx.Graph()
        # G.add_nodes_from(self.vertices)

        edges = [(x, y) for [a, b, c] in self.faces for x, y in [(a, b), (a, c), (b, c)]]
        # G.add_edges_from(all_edges)
        G.add_edges_from([(edges[i][0], edges[i][1], {"udf": (self.predicted_udf[edges[i][0]] + self.predicted_udf[edges[i][0]]) / 2}) for i in range(len(edges))])

        # node_value_dict = dict(zip(G.nodes, self.predicted_udf))
        # nx.set_node_attributes(G, node_value_dict, 'udf')
        return G



def compute_graph_divergence(G, gradients):

    divergence = {}

    for i in G.nodes:
        div = 0
        pos_i = np.array(G.nodes[i]['pos'])
        F_i = np.array(gradients[i])

        for j in G.neighbors(i):
            pos_j = np.array(G.nodes[j]['pos'])
            F_j = np.array(gradients[j])

            direction = pos_j - pos_i
            distance = np.linalg.norm(direction)
            if distance > 0:
                unit_vector = direction / distance
                flux = (F_j - F_i) @ unit_vector  # Dot product for flow change
                div += flux / distance  # Normalize by distance

        divergence[i] = div

    return divergence


# # Example Graph with Nodes and Vectors
# G = nx.grid_2d_graph(3, 3)  # 3x3 grid graph
# positions = {node: np.array(node) for node in G.nodes}
# vectors = {node: np.array([np.sin(node[0]), np.cos(node[1])]) for node in G.nodes}
#
# # Assign positions to graph
# nx.set_node_attributes(G, positions, 'pos')
#
# # Compute divergence
# divergence = compute_graph_divergence(G, vectors)
#
# print("Divergence at each node:", divergence)