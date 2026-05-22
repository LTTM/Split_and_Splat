import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.cameras import CustomCam
from scene import GaussianModel
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
import math
import json
import torch
import torchvision
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


from gaussian_renderer import render_simple
from utils_mask.PLY_utils import compute_centroid, get_bounding_box

objects_path = "./data/figurines/figurines_masks"
dataset_path = "./data/figurines/"

objects = os.listdir(objects_path)
obj = []

for o in tqdm(objects, desc="Computing centroids"):
   
        flag = o.split(".")[0]
        try:
            test = int(flag)
        except ValueError:
            continue
        object_path = os.path.join(objects_path,'PLY', o +".ply")
        ply = PlyData.read(object_path)
        centroid = compute_centroid(ply)
        vertex = ply['vertex']
        color, *_ = np.vstack([vertex['id_0'], vertex['id_1'], vertex['id_2']]).T
        bb = get_bounding_box(ply)
        bb = np.array(bb, dtype=np.float32)  
        size = np.linalg.norm(bb[1] - bb[0])  
        obj.append({
             "id": o,
             "color": color,
             "centroid": centroid,
             "size": size
        })

#Pretty printer
def pretty_print(obj):
    """Print id, color, centroid and size for each object in the list."""
    for o in obj:
        print(f"ID: {o['id']}, Color: {o['color']}, Centroid: {o['centroid']}, Size: {o['size']}")
    return


def compute_spherical_cam_params(center, radius, n_elev, n_azim):
    """
    Returns a numpy array of shape (N, 4, 4) with dtype float32 for spherical cameras around center.
    Each c2w is a torch tensor.
    """
    cam_params_list = []
    for elev in np.linspace(10, 170, n_elev):
        elev_rad = math.radians(elev)
        for azim in np.linspace(0, 360, n_azim, endpoint=False):
            azim_rad = math.radians(azim)
            x = center[0] + radius * math.sin(elev_rad) * math.cos(azim_rad)
            y = center[1] + radius * math.sin(elev_rad) * math.sin(azim_rad)
            z = center[2] + radius * math.cos(elev_rad)
            cam_pos = np.array([x, y, z])
            forward = (np.array(center) - cam_pos)
            forward = forward / np.linalg.norm(forward)
            up = np.array([0, 0, 1])
            right = np.cross(up, forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)
            R = np.stack([right, up, forward], axis=1)  # 3x3
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R
            c2w[:3, 3] = cam_pos
            cam_params_list.append(torch.from_numpy(c2w))
    cam_params_tensor = torch.stack(cam_params_list, dim=0)  # shape (N, 4, 4)
    return cam_params_tensor


if __name__ == "__main__":
    gs_path = "./data/figurines/figurines_masks/PLY/"

    for o in tqdm(obj, desc="Computing object renders"):
        # if(o["id"] != "12_13"):
        #     continue
        gs = GaussianModel(sh_degree=0)
        gs.load_ply(os.path.join(gs_path, o["id"]+".ply"))
        gs = gs.filter_points()

        center = o["centroid"]
        color_id = o["color"]
        size = o["size"]
        i=0
        n_elev = 6
        n_azim = 12
        radius = 2.0 * size
        cam_params = compute_spherical_cam_params(center, radius, n_elev, n_azim)

    #     #TODO: save cameras
        camera_path = os.path.join("./data/figurines_mask", o["id"], "cameras")
        os.makedirs(camera_path, exist_ok=True)
    
        batch_size = cam_params.shape[0]

        for c in tqdm(cam_params, desc="Cameras rendering"):
            
            cam_param = c.to("cuda")  # Use the first camera parameters for testing
            fov = 45
            resolution = 1120
            background_color= torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
            fov_rad = fov / 360 * 2 * np.pi
            render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_param)
            render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))  
            output_path = os.path.join("./data/figurines_mask", o["id"], "rendered")
            os.makedirs(output_path, exist_ok=True)
            img = render["render"]

            torchvision.utils.save_image(img, os.path.join(output_path, '{0:05d}'.format(i) + ".png"))
            torch.save(cam_param, os.path.join(camera_path, '{0:05d}'.format(i) + ".pt"))
            i += 1
