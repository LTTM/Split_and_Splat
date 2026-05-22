import os
# if using Apple MPS, fall back to CPU for unsupported ops

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as IMG
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from tqdm import tqdm
import gc
import json
import random as rnd
from pathlib import Path

import uuid
import argparse



device = "cuda"

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_anns(anns, borders=False):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def show_mask(mask, ax, frame, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    img_to_save = (mask_image[:, :, :3] * 255).astype(np.uint8)
    os.makedirs(f"./output/{DATASET}_autoseg_mask/{frame}", exist_ok=True)
    IMG.fromarray(img_to_save).save(f"./output/{DATASET}_autoseg_mask/{frame}/{obj_id}.png")
 
def save_mask(masks, prefix):
    """
    Save only the 'segmentation' arrays from a list of mask dicts to a .npz file.
    """
    segms = {}
    for idx, m in enumerate(masks):
        segms[f"mask_{idx}"] = m["segmentation"].astype(np.uint8)
    np.savez_compressed(f"./data/{prefix}_segmentations_only.npz", **segms)

    return list(segms.values())

def load_mask(prefix, as_numpy=True):
    """
    Load only the segmentation arrays saved in a .npz file.
    Returns a tensor of shape (N, H, W).
    """
    segms = np.load(f"./data/{prefix}_segmentations_only.npz")
    if as_numpy:
        masks = [segms[key] for key in sorted(segms.files, key=lambda x: int(x.split('_')[1]))]
    else:
        masks = [torch.from_numpy(segms[key]) for key in sorted(segms.files, key=lambda x: int(x.split('_')[1]))]
    return masks

def getMask(image, output_path, configs, preload = False):
    """
    
    """
    
    
    image = np.array(image.convert("RGB"))
    os.makedirs(output_path, exist_ok=True)

    generated_mask = []
    if(not preload):
        for cfg in configs:
            #print(f"\nRunning configuration: {cfg['name']}")
            mask_generator = SAM2AutomaticMaskGenerator(**{k: v for k, v in cfg.items() if k != "name"})

            try:
                
                masks1 = mask_generator.generate(image)


                masks = save_mask(masks1, f"masks_{cfg['name']}")
                
    
                generated_mask.append(masks)
            except Exception as e:
                print("NO valid mask!")
                print(e)
                continue

    # 1-IoU between the generated mask
    else:
        for cfg in configs:
           # print(f"\nRunning configuration: {cfg['name']}")
        
            load = load_mask(f"masks_{cfg['name']}")
            generated_mask.append(load)

    #print("merging...")   
    final_masks = []
    if(len(generated_mask) == 0 ): return final_masks
    for i in range(len(generated_mask)):
        if(i == 0): 
            sorted_masks_0 = sorted(list(generated_mask[i]), key=lambda x: np.sum(x), reverse=True)
            cleaned_mask_0 = useless_mask(sorted_masks_0)
            final_masks = cleaned_mask_0
        else: final_masks = MaskMerging(final_masks, generated_mask[i])
    plt.close("all")
    final_masks = useless_mask(final_masks)
    return final_masks
 
def MaskMerging(merged, mask):

    mask = sorted(mask, key=lambda x: np.sum(x), reverse=True)

    for m1 in mask:  # iterate over new masks
        merged_flag = False
        for i, m2 in enumerate(merged): # iterate over old masks
            # Check overlap
            intersection = np.logical_and(m1, m2).sum()

            if intersection == 0:
                continue

            # areas of each mask
            area1 = m1.sum()
            area2 = m2.sum()

            # relative overlap (intersection over smaller mask)
            smaller_area = min(area1, area2)
            iou_score = intersection / smaller_area

            if(iou_score>0.3): #Overlap: mask cannot be added
           
                merged[i] = np.logical_or(m1, m2) #merge the masks
                merged_flag = True

        if not merged_flag:
            merged.append(m1) # no overlap with any existing mask

    return merged

def overlap_bins(bin_rep, bins, masks):
    """
    - bin_rep: dictionary bin-> larger mask
    - bins: bins containing overlapping masks, assume b[0] is the largest one
    - masks: mask list obtained from SAM 2

    Update the bins, if no overlap create new bin if overlap add to the bin
    """
    masks = useless_mask(masks)
    for m1 in masks:  # iterate over new masks
        placed = False

        for bin_idx, b in enumerate(bins):  # check existing bins
       
            bin_smaller = bin_rep[bin_idx]
            intersection = np.logical_and(m1, bin_smaller).sum()
            
            if intersection == 0:
                continue  # no overlap, check next bin
            
            # areas
            area1 = m1.sum()
            area2 = bin_smaller.sum()
            smaller_area = min(area1, area2)

            iou_score = intersection / smaller_area if smaller_area > 0 else 0
            
            if iou_score > 0.5: #more then 50% of the mask overlap with the representative
                
                # Add to this bin
                b.append(m1)
                placed = True

                if(area1<area2):
                    bin_rep[bin_idx] = m1
            
        
        if not placed:
            # Create new bin with this mask
            bins.append([m1])
            bin_rep[len(bin_rep.keys())] = m1
          
    
    return bins, bin_rep

def majority_voting(bins):
    """
    - bins: list of bins, each bin is a list of overlapping masks (numpy arrays)

    Performs majority voting inside each bin:
      * For each bin, compute a majority-vote mask.
      * Masks that disagree with the majority are split into new bins.
    """
    new_bins = []

    for b in bins:
        if len(b) == 1:
            # Single mask, keep as is
            new_bins.append(b)
            continue

        # Stack masks for voting
        stack = np.stack(b)  # shape: (num_masks, H, W)
        votes = stack.sum(axis=0)

        # Majority threshold (more than half)
        threshold = len(b) / 2
        majority_mask = votes > threshold

        # Compare each mask to majority
        agree_bin = []
        disagree_bin = []

        for mask in b:
            intersection = np.logical_and(mask, majority_mask).sum()
            union = np.logical_or(mask, majority_mask).sum()
            iou = intersection / union if union > 0 else 0

            if iou > 0.5:  # sufficiently agrees with majority
                agree_bin.append(mask)
            else:
                disagree_bin.append(mask)

        # Add the agreeing masks as one bin
        if agree_bin:
            new_bins.append(agree_bin)
        # Add the disagreeing masks as their own bins
        for m in disagree_bin:
            new_bins.append([m])

    return new_bins

def useless_mask(masks, min_ratio=0.0025, iou_threshold=0.95):
    filtered_masks = []
    
    for m in masks:
        total_pixels = m.size
        mask_pixels = np.count_nonzero(m)
        
        # Skip if too small
        if mask_pixels / total_pixels < min_ratio:
            continue
        
        keep = True

        for fm in filtered_masks:
            # Compute IoU with existing mask
            intersection = np.logical_and(m, fm).sum()
            area1 = m.sum()
            area2 = fm.sum()
            smaller_area = min(area1, area2)
            iou_score = intersection / smaller_area
            
            # If it's basically the same (IoU > threshold), drop it
            if iou_score > iou_threshold:
                keep = False
                break
        
        if keep:
            filtered_masks.append(m)
    
    return filtered_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation evaluation")

    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="scene to compute the masks on"
    )

    parser.add_argument(
        "--scene_path",
        type=str,
        default="./data",
        help="path to the scene"
    )
    
    
    args = parser.parse_args()

    DATASET = args.scene
    VIDEO_PATH = args.scene_path
    BASE_DIR = Path(__file__).resolve().parent

    sam2_checkpoint_init = (
        BASE_DIR / ".." / "checkpoints" / "sam2.1_hiera_large.pt"
    ).resolve() # link to download checkpoints -> SAM2 github page
    
    model_cfg_init = "./configs/sam2.1/sam2.1_hiera_l.yaml"
    

    sam2 = build_sam2(model_cfg_init, sam2_checkpoint_init, device=device, apply_postprocessing=False)

    sam2_post_proc = build_sam2(model_cfg_init, sam2_checkpoint_init, device=device, apply_postprocessing=True)


    configs = [
            {
                "name": "uber_huge",
                "model": sam2,
                "points_per_side": 1,
                "points_per_batch": 1,
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.85,
                "stability_score_offset": 0.85,
                "mask_threshold": 0.5,
                "crop_n_layers": 1,
                "box_nms_thresh": 0.3,
                "use_m2m": False
            },

            {
                "name": "very_coarse",
                "model": sam2,
                "points_per_side": 4,
                "points_per_batch": 4,
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.9,
                "stability_score_offset": 0.85,
                "mask_threshold": 0.5,
                "crop_n_layers": 1,
                "box_nms_thresh": 0.3,
                "use_m2m": False
            },

            {
                "name": "coarse",
                "model": sam2,
                "points_per_side": 8,
                "points_per_batch": 8,
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.85,
                "stability_score_offset": 0.85,
                "mask_threshold": 0.3,
                "crop_n_layers": 1,
                "box_nms_thresh": 0.5,
                "use_m2m": False
            },

        
            {
                "name": "fine",
                "model": sam2,
                "points_per_side": 16,
                "points_per_batch": 64,
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.9,
                "stability_score_offset": 1,
                "mask_threshold": 0.3,
                "crop_n_layers": 1,
                "box_nms_thresh": 0.3,
                "use_m2m": False
            }
            
    ]


    video_dir = os.path.join(VIDEO_PATH,DATASET, "images")
    files = sorted(os.listdir(video_dir))

    # Rename files in sequence
    for index, filename in enumerate(files):
        if filename.lower().endswith(".png") or filename.endswith(".jpg"):  # Filter for JPEG files
    
            new_name = filename.replace("jpg", "JPEG")
            new_name = new_name.replace("png", "JPEG")
            new_name = new_name.replace("frame_", "")
         
            old_path = os.path.join(video_dir, filename)
            new_path = os.path.join(video_dir, new_name)
            os.rename(old_path, new_path)


    print("Renaming complete!")

    best_views = os.listdir (os.path.join(VIDEO_PATH,DATASET, "images"))

   

    best_views = sorted(best_views)
    #### SUBSAMPLE SCANET ####
     
    #best_views =  best_views[::20]

    for i, v in enumerate(tqdm((best_views))):
        
        #print(f"Generating masks view {v}")
        frame_name = (v).replace(".jpg", ".JPEG")
        frame = (frame_name).replace(".JPEG", "")

        f = IMG.open(os.path.join(VIDEO_PATH,DATASET, "images", frame_name))
        
        f_masks =  getMask(f,f"./output/{DATASET}/{frame}" ,configs= configs, preload=False)
        
        # #3-plot final mask
        plt.figure(figsize=(20, 20))
        plt.imshow(f)
        for out_obj_id, out_mask in enumerate(f_masks):
            show_mask(out_mask, plt.gca(),frame,  obj_id = out_obj_id, random_color=True)
        plt.axis('off')
        # os.makedirs(f"./output/full_mask_{DATASET}", exist_ok=True)
        # plt.savefig(f"./output/full_mask_{DATASET}/{frame}.png")
        plt.close("all")

       
        #print(f"Number of intances at frame 0: {len(f_masks)}")

        del f, f_masks
        gc.collect()
        if torch.cuda.is_available():
                torch.cuda.empty_cache()

