import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn
from sklearn.preprocessing import normalize

class K_Means:

    def __init__(self, mesh, UDF, num_clusters):
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        self.UDF = UDF
        self.G = self.create_graph()
        self.num_clusters = num_clusters

    def create_graph(self):
        # create a graph with all edges of the mesh
        G = nx.Graph()
        all_edges = [(x, y) for [a, b, c] in self.faces for x, y in [(a, b), (a, c), (b, c)]]
        G.add_edges_from(all_edges)

        udf = [x if x < 0.1 else 0.1 for x in self.UDF ]

        # take udf property from vertices and add them to graph nodes
        node_value_dict = dict(zip(G.nodes, udf))
        nx.set_node_attributes(G, node_value_dict, 'udf')
        return G


    def segment(self):
        G = self.G
        # Use node features – here: adjacency matrix rows (simple baseline)
        A = normalize(nx.adjacency_matrix(G).todense())
        # X = normalize(np.array(A))  # Normalize for better clustering

        # X = np.hstack((self.vertices, self.UDF[:,np.newaxis]))

        udf = np.array([x if x < 0.5 else 0.5 for x in self.UDF ])
        udf = np.array(self.UDF)

        X = udf[:,np.newaxis]

        # X = np.hstack((self.vertices, udf[:,np.newaxis]))
        # X = (A * self.UDF)

        # Apply KMeans clustering
        kmeans = sklearn.cluster.KMeans(n_clusters=self.num_clusters, random_state=0).fit(X)
        labels = kmeans.labels_

        return labels