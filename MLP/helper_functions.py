import torch

def append_impulse_to_data(data, impulse):
    test_data = torch.from_numpy(data).float()
    impulse_input = torch.from_numpy(impulse).unsqueeze(0).expand(test_data.size(0), -1).float()
    test_data = torch.cat((test_data, impulse_input), 1)
    return test_data

def get_screenshot_polyscope(mesh, labels):
    import polyscope as ps
    import numpy as np

    ps.set_window_size(1920, 1080)
    ps.init()

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    ps.look_at((0., 0., 2.5), (0, 0, 0))
    ps.register_surface_mesh("mesh", vertices, faces, smooth_shade=True)

    ps.get_surface_mesh("mesh").add_scalar_quantity("values", labels, defined_on="vertices", enabled=True)
    ss = ps.screenshot_to_buffer(transparent_bg=False)
    ss3 = ss[:, :, :3]
    ps.show(0)
    return ss3