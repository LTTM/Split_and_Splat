import torchvision.transforms.functional as F

import os
import torch
import open3d as o3d
import pycolmap
import numpy as np
import sam2
from plyfile import PlyData, PlyElement
from PIL import Image as IMG, ImageFilter
import cv2
from PIL import Image as IMG
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_erosion
from arguments import ModelParams, PipelineParams, ArgumentParser, get_combined_args
from utils.general_utils import safe_state
from scene import Scene, GaussianModel

from gaussian_renderer import render
import matplotlib
matplotlib.use('Agg')  # Use headless backend (no display needed)
import matplotlib.pyplot as plt



# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")



# Dataset-specific image sizes (H, W)
DATASET_IMAGE_SIZES = {
    "figurines": (728, 986),
    "ramen": (731, 988),
    "teatime": (730, 988),
    "waldo_kitchen": (725, 985),
}

##########################
ACCURACY_LABELS = 0.7
EPSILON = 0.01


def filter_points(ply, black_th = -1.75, alpha_th = 4.5):
    """Filter points in a PLY file based on color and opacity, returns only XYZ coordinates."""
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

    # Extract XYZ coordinates
    xyz = np.vstack([
        filtered_array['x'],
        filtered_array['y'],
        filtered_array['z']
    ]).T

    return xyz

def compute_IoU(mask_1, mask_2):
    """
    Compute Intersection over Union between two binary masks.
    
    Args:
        mask_1: First binary mask as numpy array
        mask_2: Second binary mask as numpy array
    
    Returns:
        float: IoU score between 0 and 1
    """
    # Convert masks to boolean arrays
    mask_1 = mask_1.astype(bool)
    mask_2 = mask_2.astype(bool)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask_1, mask_2).sum()
    union = np.logical_or(mask_1, mask_2).sum()
    
    # Return IoU score, handle division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def mask_to_array(masks):
    """Collapse a list or array of binary masks into a single uint8 mask via logical OR."""
    if isinstance(masks, list):
        masks = np.array(masks)

    # If multiple masks, collapse into one (any overlap counts as mask)
    mask = (np.sum(masks, axis=0) > 0).astype(np.uint8)

    # Return just the binary mask array instead of RGBA image
    return mask

def test_mask(img, mask, input_point):
    """Plot the image with SAM2 input points alongside the predicted mask overlay for visual debugging."""
    plt.figure(figsize=(12, 6))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.scatter(input_point[:, 0], input_point[:, 1], c='red', s=50, marker='x', label='Input Points')
    plt.title('Image with Input Points')
    plt.axis('off')
    plt.legend()

    # Plot image with mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    mask_overlay = np.zeros_like(img[:,:,0], dtype=float)
    mask_overlay[mask > 0] = 1
    plt.imshow(mask_overlay, alpha=0.5, cmap='cool')
    plt.scatter(input_point[:, 0], input_point[:, 1], c='red', s=50, marker='x', label='Input Points')
    plt.title('Mask Overlay with Input Points')
    plt.axis('off')
    plt.legend()

    plt.tight_layout()
    plt.show()

def get_render_mask(camera, dataset, pipe, obj_id):
    """Render both the object mask and the full scene with the object removed for a given camera view."""
    bg_color = [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        render_pkg = render(camera, gaussians, pipe, bg, use_trained_exp = dataset.train_test_exp, separate_sh=False,  id_filter=obj_id, mask_only=True)
        remove_pkg = render(camera, gaussians, pipe, bg, use_trained_exp = dataset.train_test_exp, separate_sh=False,  id_filter=obj_id, mask_only=False, remove=True)

    return render_pkg["mask"], remove_pkg["render"]

def get_render_view(camera, dataset, pipe, masks = False):
    """Render the full scene from a given camera. If masks=True also returns the mask channel."""
    bg_color = [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        render_pkg = render(camera, gaussians, pipe, bg, use_trained_exp = dataset.train_test_exp, separate_sh=False)
        
    if(masks):
        return render_pkg["render"], render_pkg["mask"]
    
    return render_pkg["render"]


def erode_mask(mask: np.ndarray, size):
    """Erode a binary mask with kernel size and iteration count scaled to the relative mask size."""
    mask = mask.astype(bool)
    if size < 0.10:
        return binary_erosion(mask, structure=np.ones((3,3)), iterations=3)
    elif size < 0.30:
        return binary_erosion(mask, structure=np.ones((5,5)), iterations=2)
    else:
        return binary_erosion(mask, structure=np.ones((5,5)), iterations=2)

def crop_to_mask(img: IMG.Image, mask_np: np.ndarray, size=224,offset=10):
    """
    Crop the smallest possible square that fully contains the mask,
    ensuring the crop stays inside the image bounds. Then resize to (size, size).
    """
    H, W = mask_np.shape

    # Get mask coordinates
    coords = np.column_stack(np.where(mask_np > 0.05))
    if coords.size == 0:
        # Fallback: center crop
        cx, cy = W // 2, H // 2
        half = min(cx, cy, size // 2)
        left = cx - half
        top = cy - half
        right = cx + half
        bottom = cy + half
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)


        # Apply offset BEFORE computing the square
        x_min -= offset
        y_min -= offset
        x_max += offset
        y_max += offset

        # Compute bounding box size
        width = x_max - x_min
        height = y_max - y_min
        side = max(width, height)  # smallest square covering the mask

        # Expand square slightly to avoid cutting off edges
        side = int(np.ceil(side * 1.05))  # +5% margin

        # Center of the mask
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        # Determine square coordinates
        half = side // 2
        left = cx - half
        top = cy - half
        right = cx + half
        bottom = cy + half

        # Clamp to image bounds
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > W:
            left -= (right - W)
            right = W
        if bottom > H:
            top -= (bottom - H)
            bottom = H

        # Re-clamp again (just in case)
        left = max(0, left)
        top = max(0, top)
        right = min(W, right)
        bottom = min(H, bottom)

    # Perform crop
    cropped = img.crop((left, top, right, bottom))

    # Resize to target square
    cropped = cropped.resize((size, size), IMG.Resampling.LANCZOS)
    
    return cropped

def resize_mask_to_image(mask_bool: np.ndarray, target_size):
    """
    Resize a boolean mask to match image size.
    
    mask_bool: np.ndarray (H0, W0), dtype=bool
    target_size: (W, H) from PIL image.size
    
    Returns:
        np.ndarray (H, W), dtype=bool
    """
    W, H = target_size
    return cv2.resize(
        mask_bool.astype(np.uint8),
        (W, H),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    
def mask_black_background(img: IMG.Image, mask_bool: np.ndarray):
    """Mask everything outside the mask to black, crop, resize to 224x224 and return an uncropped version"""
    img = img.convert("RGB")
    img_np = np.array(img)

    # resize mask exactly to image
    mask_bool_resized = resize_mask_to_image(mask_bool, img.size)

    masked_np = img_np * mask_bool_resized[..., None]  # boolean masking
    masked_img = IMG.fromarray(masked_np.astype(np.uint8))

    cropped = crop_to_mask(masked_img, mask_bool_resized)

    return masked_img, cropped

def mask_white_background(img: IMG.Image, mask: IMG.Image):
    """Mask everything outside the mask to white, crop, resize to 224x224."""
    img = img.convert("RGB")
    mask = mask.convert("L").resize(img.size, IMG.Resampling.LANCZOS)
    mask_np = np.array(mask) / 255.0

    img_np = np.array(img)

    white_bg = np.ones_like(img_np) *255

    masked_np = img_np * mask_np[..., None] + white_bg * (1 - mask_np[..., None])
    masked_img = IMG.fromarray(masked_np.astype(np.uint8))

    return crop_to_mask(masked_img, mask_np)

def mask_blur_background(img: IMG.Image, mask: IMG.Image, blur_radius=10):
    """Mask everything outside the mask to blurred version, crop, resize."""
    img = img.convert("RGB")
    mask = mask.convert("L").resize(img.size, IMG.Resampling.LANCZOS)
    mask_np = np.array(mask) / 255.0
    img_np = np.array(img)

    blurred_np = np.array(img.filter(ImageFilter.GaussianBlur(radius=blur_radius)))
    masked_np = img_np * mask_np[..., None] + blurred_np * (1 - mask_np[..., None])
    masked_img = IMG.fromarray(masked_np.astype(np.uint8))

    return crop_to_mask(masked_img, mask_np)


def is_mask_significant(mask_np, threshold=0.01):
    """
    Returns True if the mask has enough positive pixels to process.
    
    mask_np: binary numpy array (0/1 or 0/255)
    threshold: minimum fraction of positive pixels to consider significant
    """
    total_pixels = mask_np.size
    positive_pixels = np.count_nonzero(mask_np)
    fraction = positive_pixels / total_pixels

    return fraction >= threshold
###########################

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument( "-dataset", type=str, default=None, help="Dataset where compute the views")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    pipeline = pipeline.extract(args)
    model = model.extract(args)
    iteration = args.iteration

    # Initialize system state (RNG)
    safe_state(args.quiet)

    args.is_instance = False

    DATASET = args.dataset
    
    H,W = DATASET_IMAGE_SIZES[DATASET]
    
    gaussians = GaussianModel(model.sh_degree)

    
    scene = Scene(model, gaussians, load_iteration=iteration, shuffle=False)
   
    #scene = scene_old.filter_gaussian()

    cameras = scene.getTrainCameras()
 

#SAM model

    objects_path = "./outputs/" + DATASET + "/PLY_ref"

    objects = os.listdir(objects_path)
    obj = []


    for o in tqdm(objects, desc="Computing centroids"):
            o = o.replace(".ply","")
            try:
                int(o)
            except:
                continue
            if o == "PLY" or o == "final": continue
            try:
                ply_path = os.path.join("./output",DATASET,"PLY_ref", o+".ply")
    
                ply = PlyData.read(ply_path)
            except:
                continue
            vertex = ply['vertex']
            color, *_ = np.vstack([vertex['id_0'], vertex['id_1'], vertex['id_2']]).T
            obj.append({
                "id": o,
                "color": color,
            })
            del ply
            torch.cuda.empty_cache()

    for o in obj:
        iter = 0
        # 2-iterate over each view and check IoU with original mask
        for cam in tqdm(cameras,  desc=f"computing masks for object {o["id"]}", total=len(cameras)):

  
            iter +=1
            
            # Open image, depth and masks

            img_name = getattr(cam, "image_name", None)
            image_name = f"{img_name}"
            
            # render the view
            rendered_mask, removed_test = get_render_mask(cam, model, pipeline, o["color"])
            
            
            img_pil = F.to_pil_image(removed_test)
            img_pil = img_pil.resize((W, H), IMG.Resampling.LANCZOS)
            os.makedirs(os.path.join("./testing_view",DATASET,"removed_test", o["id"]), exist_ok=True)
            img_pil.save(os.path.join("./testing_view",DATASET, "removed_test",   o["id"],  image_name))
        


            if rendered_mask.ndim == 3 and rendered_mask.shape[0] in [3, 4]:  # (C,H,W)
                gray_mask = rendered_mask.mean(dim=0)
            else:
                gray_mask = rendered_mask

            # Create binary mask: 0 for black, 1 otherwise
            rendered_mask = (gray_mask > 0.05)
            
             
            rendered_mask = rendered_mask.detach().cpu().numpy()
            size = rendered_mask.sum()/(H*W)
            mask_bool = erode_mask(rendered_mask, size)

            # --- resize mask to match image resolution ---
            
            mask_bool = resize_mask_to_image(mask_bool, (992,736))

            # --- convert to image ONLY for saving / visualization ---
            mask_img_np = (mask_bool.astype(np.uint8) * 255)
            mask_img = IMG.fromarray(mask_img_np)
 
            os.makedirs(os.path.join("./testing_view",DATASET,"boolean_masks",image_name.replace(".jpg", "")), exist_ok=True)
            mask_img.save(os.path.join("./testing_view",DATASET, "boolean_masks", image_name.replace(".jpg", ""), o["id"]+".png"))

            

            # Mask original image
            try:
                original_image_path = os.path.join("./data", DATASET, "images", image_name)
                original_image = IMG.open(original_image_path)
            except:
                #print(f"Cannot find {image_name}")
                continue

            out_masked, out_black = mask_black_background(original_image, mask_bool)
            out_white = mask_white_background(original_image, mask_img)
            out_blur = mask_blur_background(original_image, mask_img)
            out_keep = crop_to_mask(original_image, mask_bool)
            
            os.makedirs(os.path.join("./testing_view", DATASET, "masked", o["id"]), exist_ok=True)
            out_masked.save(os.path.join("./testing_view", DATASET, "masked", o["id"],  image_name))
            
            if not is_mask_significant(mask_bool, threshold=0.01):

                #print(f"Skipping {image_name}, mask almost empty")
                continue  

            os.makedirs(os.path.join("./testing_view", DATASET,"black_bg", o["id"]), exist_ok=True)
            out_black.save(os.path.join("./testing_view", DATASET, "black_bg", o["id"], image_name))

            os.makedirs(os.path.join("./testing_view", DATASET,"white_bg", o["id"]), exist_ok=True)
            out_white.save(os.path.join("./testing_view", DATASET, "white_bg", o["id"], image_name))

            os.makedirs(os.path.join("./testing_view", DATASET, "blurred_bg", o["id"]), exist_ok=True)
            out_blur.save(os.path.join("./testing_view", DATASET, "blurred_bg", o["id"],  image_name))

            os.makedirs(os.path.join("./testing_view", DATASET, "crop", o["id"]), exist_ok=True)
            out_keep.save(os.path.join("./testing_view", DATASET, "crop", o["id"],  image_name))
            

    i = 1
    for cam in tqdm(cameras):
        cam_id = getattr(cam, "uid", None)+1
        
        img_name = getattr(cam, "image_name", None)
        image_name = f"{img_name}.JPEG"

        # Generate render of the given view
        rendered_img, rendered_mask = get_render_view(cam, model, pipeline, masks=True)

        # view
        img_pil = F.to_pil_image(rendered_img)
        img_pil = img_pil.resize((W, H), IMG.Resampling.LANCZOS)
        os.makedirs(os.path.join("./testing_view",DATASET,"view_test"), exist_ok=True)
        img_pil.save(os.path.join("./testing_view",DATASET, "view_test", f"gs_{i}.JPEG"))
        
        # mask
        mask_pil = F.to_pil_image(rendered_mask)
        mask_pil = mask_pil.resize((W, H), IMG.Resampling.LANCZOS)
        os.makedirs(os.path.join("./testing_view",DATASET,"mask_test"), exist_ok=True)
        mask_pil.save(os.path.join("./testing_view",DATASET, "mask_test", f"gs_{i}.JPEG"))
        i=i+1
