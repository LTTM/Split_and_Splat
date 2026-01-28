import sys
sys.path.append('/home/leo/Desktop/Leo/SEQUOIA/modular-GS')

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy as np
import os
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


# select the device for computation
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# elif device.type == "mps":
#     print(
#         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#         "give numerically different outputs and sometimes degraded performance on MPS. "
#         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#     )


def get_masks(dataset_path, objects_path):
    
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



# images = os.listdir(os.path.join(dataset_path, "images"))

# for i in tqdm(images, desc="Computing new mask points"):
#     i_png= i.replace(".JPEG", ".png")
#     image_path = os.path.join(dataset_path, "full_mask", i_png)
#     image = Image.open(image_path).convert("RGBA")

#     image_alpha = np.array(image.split()[-1])  # Convert to NumPy array

#     extrinsic_image = extract_by_name(cam_extrinsics, i_png)
#     try:
#         camera_id = extrinsic_image.camera_id
#     except:
#         continue

#     intrinsic_image = cam_intrinsics[camera_id]

#     for o in obj:
#         centroid = torch.tensor(o["centroid"])
#         x, y = from3Dto2D(centroid, image, intrinsic_image, extrinsic_image)
#         # Boundary check to ensure point_2d is within image_alpha dimensions
#         if 0 <= y < image_alpha.shape[0] and 0 <= x < image_alpha.shape[1]:
#             if image_alpha[int(y), int(x)] == 0:
#                 original_image_path = os.path.join(dataset_path, "images", i)
#                 original_image = Image.open(original_image_path).convert("RGB")

#                 predictor.set_image(original_image)
#                 input_point = np.array([[x.item(),y.item()]])                
           
#                 input_label = np.array([1])
#                 masks, scores, logits = predictor.predict(
#                     point_coords=input_point,
#                     point_labels=input_label,
#                     multimask_output=False,
#                 )
#                 save_mask(masks[0], original_image, SH2RGB(o["color"]), i, o["id"])
           
#             predictor.reset_predictor()

# 5- 3D conflict solver




      


def filter_3D(dataset,iteration, pipe):
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

# 6- Update masks
masks_path = os.path.join(objects_path, "final/mask")
save_path = os.path.join(objects_path, "final/mask_extra_png")

for o in tqdm(obj, desc="Updating masks"):
    o_id = o["id"]
    rec_masks_path = os.path.join(objects_path, "final/mask_restored", o_id, "mask")
  
    if not os.path.exists(rec_masks_path):
        continue


    for i in tqdm(os.listdir(rec_masks_path), desc=f"Updating masks for {o_id}"):
        i_png = i.replace(".JPEG", ".png")
      
        mask_path = os.path.join(masks_path, i_png)
        
        if(not os.path.exists(mask_path)):
            print(f"Mask not found for {i_png} for {o_id}, skipping...")
            continue
        mask = Image.open(mask_path).convert("RGBA")
        rec_mask = Image.open(os.path.join(rec_masks_path, i_png)).convert("RGBA")


        new_mask = Image.new("RGBA", mask.size, (0, 0, 0, 0))
        new_mask = Image.alpha_composite(mask, rec_mask)


        new_mask = Image.alpha_composite(mask, rec_mask)

        os.makedirs(save_path, exist_ok=True)
        new_mask.save(mask_path)

        os.makedirs(os.path.join(save_path, "mask_extra"), exist_ok=True)
        mask_extra_path = os.path.join(save_path, "mask_extra", i_png)
        new_mask.save(mask_extra_path)

        mask_alpha = new_mask.split()[-1]
      
        image_path = os.path.join(dataset_path, "images", i)
        try:
            image = Image.open(image_path).convert("RGBA")
        except:
            continue
        image.putalpha(mask_alpha)

        # image = Image.alpha_composite(image_a, image_b)

        
        os.makedirs(os.path.join(save_path, "images_extra"), exist_ok=True)
        images_path = os.path.join(save_path, "images_extra", i_png)
        image.save(images_path)

