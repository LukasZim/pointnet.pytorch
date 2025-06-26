import networkx as nx
import numpy as np


class Watershed:

    def __init__(self, mesh, UDF, threshold = 0.2):
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        self.UDF = UDF
        self.G = self.create_graph()
        self.threshold = threshold

    def calculate_watershed(self):
        # create graph
        G = self.G
        # find local maxima + give them labels
        local_maxima, local_maxima_labels = self.find_local_max()

        # find flat areas and set them as minimum or plateau

        # loop through pleateaus

        # loop through other vertices
        visited = [False] * len(self.vertices)
        labels = [-1] * len(self.vertices)
        queue = []
        queue.extend(local_maxima)
        for i, value in enumerate(local_maxima):
            labels[value] = local_maxima_labels[i]

        while len(queue) > 0:
            node = queue.pop(0)
            visited[node] = True
            udf = self.UDF[node]
            neighbours = list(G.neighbors(node))
            for neighbour in neighbours:
                if not visited[neighbour] and self.UDF[neighbour] <= udf and labels[neighbour] == -1:
                    queue.append(neighbour)
                    labels[neighbour] = labels[node]
                if not labels[neighbour] == -1 and (self.UDF[neighbour] >= self.threshold or self.UDF[node] >= self.threshold) and not labels[node] == labels[neighbour]:
                    local_maxima_labels = self.merge_labels(local_maxima_labels, labels[node], labels[neighbour])

        # for node in self.vertices:
        #     label = labels[node]
        #     neighbours = list(G.neighbors(node))
        #     for neighbour in neighbours:
        result = [local_maxima_labels[i] for i in labels]
        return np.array(result)
        # merge regions whose depth is below threshold


    def merge_labels(self, labels, l1, l2):
        new_label = min(l1, l2)
        return [new_label if item == l1 or item == l2 else item for item in labels]

    def create_graph(self):
        # create a graph with all edges of the mesh
        G = nx.Graph()
        all_edges = [(x,y) for [a,b,c] in self.faces for x,y in [(a,b), (a,c), (b,c)]]
        G.add_edges_from(all_edges)

        # take udf property from vertices and add them to graph nodes
        node_value_dict = dict(zip(G.nodes, self.UDF))
        nx.set_node_attributes(G, node_value_dict, 'udf')
        return G

    def find_local_max(self):
        G = self.G
        local_max = []
        for node, attrs in G.nodes(data=True):
            neighbours = list(G.neighbors(node))
            is_max = True
            for neighbour in neighbours:
                if self.UDF[neighbour] >= self.UDF[node]:
                    is_max = False
                    break
            if is_max:
                local_max.append(node)

        return local_max, list(range(len(local_max)))

