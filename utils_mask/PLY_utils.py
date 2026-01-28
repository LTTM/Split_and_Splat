from plyfile import PlyData, PlyElement
import numpy as np
import os


def filter_points(ply, black_th = -1.75, alpha_th = 4.5):
    """Filter points in a PLY file based on color and opacity."""
    vertices = ply['vertex']
   
    f_dc = np.stack([
        vertices['f_dc_0'],
        vertices['f_dc_1'],
        vertices['f_dc_2']
    ], axis=1)
    
    is_black = np.all(f_dc < black_th, axis=1)

    # Mask: keep only non-black Gaussians
    keep_mask = ~is_black
    filtered_array = vertices[keep_mask]
    

    # Mask: near transparent Gaussians
    opacity = filtered_array['opacity']
    is_transparent = opacity > alpha_th

    keep_mask = ~is_transparent
    filtered_array = filtered_array[keep_mask]

    return filtered_array


def combine_ply(path_ply_a, path_ply_b, save_path, filter = False, is_blender = False):
    """Combine two PLY files into one, optionally filtering points based on color and opacity."""
    if(filter):
        ply_a = filter_points(PlyData.read(path_ply_a))
        ply_b = filter_points(PlyData.read(path_ply_b))


        vertex_a = ply_a.data
        vertex_b = ply_b.data
    else:
        ply_a = PlyData.read(path_ply_a)
        ply_b = PlyData.read(path_ply_b)


        vertex_a = ply_a["vertex"].data
        vertex_b = ply_b["vertex"].data



    ply = np.concatenate((vertex_a, vertex_b))
    if(is_blender):
        path_ply = os.path.join(save_path, "points3D.ply")
    else:
        os.makedirs(os.path.join(save_path,"sparse", "0"), exist_ok=True)
        path_ply = os.path.join(save_path,"sparse", "0", "points3D.ply")

    el = PlyElement.describe(ply, 'vertex')
    PlyData([el]).write(path_ply)

def combine_sparse_ply(path_ply_a, path_ply_b, save_path, filter = False):
    """Combine two PLY files into one, optionally filtering points based on color and opacity."""
    if(filter):
        ply_a = filter_points(PlyData.read(path_ply_a))
        ply_b = filter_points(PlyData.read(path_ply_b))


        vertex_a = ply_a.data
        vertex_b = ply_b.data
    else:
        ply_a = PlyData.read(path_ply_a)
        ply_b = PlyData.read(path_ply_b)


        vertex_a = ply_a["vertex"].data
        vertex_b = ply_b["vertex"].data


    # Make writable copies
    vertex_a = vertex_a.copy()
    vertex_b = vertex_b.copy()

    # Set f_dc parameters for vertex_b to -1.75
    for field in ['f_dc_0', 'f_dc_1', 'f_dc_2']:
        if field in vertex_b.dtype.names:
            vertex_b[field] = -1.75

    ply = np.concatenate((vertex_a, vertex_b))
    os.makedirs(os.path.join(save_path, "sparse", "0"), exist_ok=True)
    path_ply = os.path.join(save_path, "sparse", "0", "points3D.ply")

    el = PlyElement.describe(ply, 'vertex')
    PlyData([el]).write(path_ply)

def get_bounding_box(ply, S = 2):
    """Get the bounding box of a PLY file."""
    
    filtered_array = filter_points(ply)
    if len(filtered_array) == 0:
        # Return a default value or handle as needed
        return None, None

    positions = np.vstack([filtered_array['x'], filtered_array['y'], filtered_array['z']]).T

    min_x, max_x    =  [positions[:,0].min(), positions[:,0].max()]
    min_y, max_y    =  [positions[:,1].min(), positions[:,1].max()]
    min_z, max_z    =  [positions[:,2].min(), positions[:,2].max()]


    bb_min = [min_x, min_y, min_z]
    bb_max = [max_x, max_y, max_z]


    return [bb_min, bb_max]

def compute_collisons(ply_a, ply_b):
    """Compute the number of points in ply_b that are within the bounding box of ply_a."""
    min_bb_a, max_bb_a =get_bounding_box(ply_a)
    
    if min_bb_a is None or max_bb_a is None: return 0

    vertices_b = ply_b['vertex']
    positions_b = np.vstack([vertices_b['x'], vertices_b['y'], vertices_b['z']]).T

    mask = np.all((positions_b >= min_bb_a) & (positions_b <= max_bb_a), axis=1)

    return len(positions_b[mask])

def compute_centroid(ply):
    """Compute the centroid of a PLY file."""
    vertices = ply['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    centroid = np.mean(positions, axis=0)
    return centroid
