import numpy as np

import networkx as nx

class RegionGrowing:
    def __init__(self, mesh, predicted_UDF, gt_UDF,):
        self.threshold_vertices = None
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        self.udf = predicted_UDF
        self.gt_udf = gt_UDF
        self.threshold_value = np.max(predicted_UDF)
        self.labeling = np.full(len(self.vertices), -1)


    def threshold(self):
        # threshold the udf
        mask = self.udf >= self.threshold_value
        threshold_vertices = self.vertices[mask]
        self.threshold_vertices = threshold_vertices

    def most_common(self, lst):
        return max(set(lst), key=lst.count)

    def calculate_region_growing(self):
        # threshold the udf
        self.threshold()

        # calculate the groups
        G = nx.Graph()
        all_edges = [(x,y) for [a,b,c] in self.faces for x,y in [(a,b), (a,c), (b,c)]]
        G.add_edges_from(all_edges)
        print(G)
        print(G[1])
        node_value_dict = dict(zip(G.nodes, self.udf))
        nx.set_node_attributes(G, node_value_dict, 'udf')
        print(G)
        print(G[1])

        visited = set()
        num_groups = 0
        while len(visited) < len(self.vertices):
            if self.threshold_value < 0:
                self.threshold_value = np.min(self.udf)

            target_nodes = set([(n, attrs.get('udf')) for n, attrs in G.nodes(data=True) if attrs.get("udf") >= self.threshold_value]) - visited
            target_nodes = sorted(target_nodes, key=lambda tup: tup[1])

            for node, udf_value in target_nodes:
                if self.labeling[node] == -1:
                    neighbour_labels = [self.labeling[i] for i in list(G.neighbors(node))]
                    neighbour_labels = [x for x in neighbour_labels if x != -1]
                    if len(neighbour_labels) == 0:
                        self.labeling[node] = num_groups
                        num_groups += 1
                    else:
                        self.labeling[node] = self.most_common(neighbour_labels)
                    visited.add(node)

            self.threshold_value -= .004
        print(num_groups)

        for node, udf in set([(n, attrs.get('udf')) for n, attrs in G.nodes(data=True)]):
            neighbour_labels = [(i,self.labeling[i]) for i in list(G.neighbors(node))]
            for neighbour, neighbour_label in neighbour_labels:
                own_label = self.labeling[node]
                if neighbour_label != own_label:
                    if self.udf[node] > 0.1 and self.udf[neighbour] > 0.1:
                        # we merge the groups

                        new_label = min(neighbour_label, own_label)
                        self.labeling = [new_label if label == neighbour_label or label == own_label else label for label in self.labeling ]
                        # break

                        print(own_label, neighbour_label, new_label)
                        print(np.unique(self.labeling, return_counts=True))
                        print("========================")
                        break



        print(num_groups)
        print(np.max(self.labeling))
        print(np.unique(self.labeling, return_counts=True))
        return np.array(self.labeling)


        # slowly grow back labels
        # region = np.argmax(self.udf)
        # visited = {region}
        # region = [region]
        # return labels



