import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy as np
from plyfile import PlyData

from PIL import Image
from tqdm import tqdm
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
 
from scene.colmap_loader import from3Dto2D, extract_by_name, read_extrinsics_binary, read_intrinsics_binary

from utils_mask.SAM_2_utils import save_mask
from utils.sh_utils import SH2RGB

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, ArgumentParser, get_combined_args
from utils_mask.mask_filters import mask_filter
from utils.general_utils import safe_state
from utils_mask.PLY_utils import compute_centroid

def get_masks(dataset_path, objects_path):
    """Composite per-object RGBA masks for every image into a single full-scene mask and save under dataset_path/mask/."""
    
    images_path = os.path.join(dataset_path, "images")
    images = os.listdir(images_path)
    objects = os.listdir(objects_path)

    
    for i in tqdm(images, desc="Computing partial masks"): 
        i_png= i.replace(".JPEG", ".png")
        image_path = os.path.join(images_path, i)
        image = Image.open(image_path).convert("RGBA")

        mask_final = Image.new("RGBA", image.size, (0, 0, 0, 0))

        for o in objects:
            
            mask_path = os.path.join(objects_path, o, "mask", i_png)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("RGBA")
            else:
                mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
        
            mask_final = Image.alpha_composite(mask_final, mask)
        
        
        os.makedirs(os.path.join(dataset_path, "mask"), exist_ok=True)
        mask_path = os.path.join(dataset_path, "mask", i_png)

        mask_final.save(mask_path)


objects_path = "./data/figurines/figurines_mask/"
dataset_path = "./data/figurines/"
# 1- Compute all the masks 
#get_masks(dataset_path, objects_path)

# 2-Compute 3D centroid of each object 

objects = os.listdir(objects_path)
obj = []
for o in tqdm(objects, desc="Computing centroids"):
        if o == "PLY" or o == "final": continue
        object_path = os.path.join(objects_path,'PLY', o+".ply")
        ply = PlyData.read(object_path)
        centroid = compute_centroid(ply)
        vertex = ply['vertex']
        color, *_ = np.vstack([vertex['id_0'], vertex['id_1'], vertex['id_2']]).T
        obj.append({
             "id": o,
             "color": color,
             "centroid": centroid
        })
print(len(obj))
# 3-function to compute 2D point from 3D centroid and viewpoint: from3Dto2D
# 4-Use the centroid as input for the segmentatation
# 4.5- use point as sam 2 input 

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda"))

predictor = SAM2ImagePredictor(sam2_model)

cameras_extrinsic_file = os.path.join(dataset_path, "sparse/0", "images.bin")
cameras_intrinsic_file = os.path.join(dataset_path, "sparse/0", "cameras.bin")
cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)



def filter_3D(dataset, iteration, pipe):
    """Load the Gaussian scene at the given iteration and run mask_filter for every object."""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cameras = scene.getTrainCameras()
        for o in obj:
            o_id = o["id"]
            color = o["color"]
            mask_filter(gaussians, cameras , pipe, dataset, o_id, color)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    filter_3D(model.extract(args), args.iteration, pipeline.extract(args))


