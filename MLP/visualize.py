import open3d as o3d
import potpourri3d as pp3d

def create_mesh_from_faces_and_vertices(faces, vertices):
    """
    Creates a mesh from a set of faces and vertices.
    :param faces: numpy array of faces
    :param vertices: numpy array of vertices
    :return: open3d.geometry.TriangleMesh
    """
    # create empty variable for "fine_mesh"
    fine_mesh = o3d.geometry.TriangleMesh()
    # retrieve and set fine vertices
    fine_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # retrieve and set fine faces
    fine_mesh.triangles = o3d.utility.Vector3iVector(faces)
    fine_mesh.compute_vertex_normals()
    return fine_mesh


def load_mesh_from_file(filepath):
    """
    Load as mesh from a file and attempts to fix the points, such that overlapping points are merged automatically
    :param filepath: path to the mesh to load
    :return: mesh loaded in that was fixed by potpourri3d
    """

    vertices, faces = pp3d.read_mesh(filepath)
    # mesh2 = o3d.io.read_triangle_mesh("data/bunny_oded.obj")
    mesh = create_mesh_from_faces_and_vertices(faces, vertices)
    return mesh