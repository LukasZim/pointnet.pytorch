import networkx as nx
import numpy as np
import trimesh
import numpy as np
from scipy.integrate import solve_ivp

def mesh_to_graph(faces, udf):
    # create a graph with all edges of the mesh
    G = nx.Graph()
    all_edges = [(x, y) for [a, b, c] in faces for x, y in [(a, b), (a, c), (b, c)]]
    G.add_edges_from(all_edges)

    # take udf property from vertices and add them to graph nodes
    node_value_dict = dict(zip(G.nodes, udf))
    nx.set_node_attributes(G, node_value_dict, 'udf')


def gradient_flow(x, gradients, vertices):
    idx = np.argmin(np.linalg.norm(vertices - x, axis=1))
    return gradients[idx]


def trace_streamline(vertices, gradients, start, max_steps=100, step_size=0.05):
    streamline = [start]
    current = start
    for _ in range(max_steps):
        idx = np.argmin(np.linalg.norm(vertices - current, axis=1))
        grad = gradients[idx]
        current = current + step_size * grad
        streamline.append(current)

        # Stop if flow is oscillating or out of bounds
        if len(streamline) > 3 and np.linalg.norm(streamline[-1] - streamline[-3]) < 1e-4:
            break

    return np.array(streamline)

def calculate(vertices, faces, udf, gradients):
    from scipy.spatial import KDTree
    import networkx as nx

    # Build k-d tree for fast lookup
    tree = KDTree(vertices)
    streamlines = [trace_streamline(vertices, gradients, v) for v in vertices]

    # Use last streamline point as convergence label
    end_points = np.array([s[-1] for s in streamlines])

    # Group vertices by closest endpoints
    graph = nx.Graph()
    threshold = 0.1  # Merging threshold

    for i, p1 in enumerate(end_points):
        for j, p2 in enumerate(end_points[i + 1:], start=i + 1):
            if np.linalg.norm(p1 - p2) < threshold:
                graph.add_edge(i, j)

    components = list(nx.connected_components(graph))

    print(f"Found {len(components)} regions")


