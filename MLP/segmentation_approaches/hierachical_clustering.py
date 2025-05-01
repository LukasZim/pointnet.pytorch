import networkx as nx
import numpy as np

class HierarchicalClustering:

    def __init__(self, mesh, UDF, threshold = 0.2):
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        self.UDF = UDF
        self.threshold = threshold
        self.G = self.create_graph()

    def create_graph(self):
        # create a graph with all edges of the mesh
        G = nx.Graph()
        all_edges = [(x,y) for [a,b,c] in self.faces for x,y in [(a,b), (a,c), (b,c)]]
        G.add_edges_from(all_edges)

        # take udf property from vertices and add them to graph nodes
        node_value_dict = dict(zip(G.nodes, self.UDF))
        nx.set_node_attributes(G, node_value_dict, 'udf')
        return G