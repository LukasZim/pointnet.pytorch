import networkx as nx
import numpy as np

class HierarchicalClustering:

    def __init__(self, mesh, UDF, threshold = 0.2):
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        self.UDF = UDF
        self.threshold = threshold
        self.G = self.create_graph()


    def calculate_hierarchical_clustering(self):
        G = self.G
        # retrieve all edges and order them by combined vertex UDF
        edges = list(G.edges())
        edges = sorted(edges, key=lambda e:  -(self.UDF[e[0]] + self.UDF[e[1]]))

        labels = list(range(len(self.vertices)))
        for (x, y) in edges:
            x_label = labels[x]
            y_label = labels[y]

            if self.UDF[x] + self.UDF[y] < self.threshold:
                break

            if not x_label == y_label:
                new_label = min(x_label, y_label)
                for index, value in enumerate(labels):
                    if value == x_label or value == y_label:
                        labels[index] = new_label
        # merge them into same group

        # keep merging until sum is too small
        return np.array(labels)




    def create_graph(self):
        # create a graph with all edges of the mesh
        G = nx.Graph()
        all_edges = [(x,y) for [a,b,c] in self.faces for x,y in [(a,b), (a,c), (b,c)]]
        G.add_edges_from(all_edges)

        # take udf property from vertices and add them to graph nodes
        node_value_dict = dict(zip(G.nodes, self.UDF))
        nx.set_node_attributes(G, node_value_dict, 'udf')
        return G