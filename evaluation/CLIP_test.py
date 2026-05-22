################################################################################
# Split&Splat - Copyright (c) 2026, MEDIALab, University of Padova.
#
# Author(s):
#  Leonardo Monchieri (leonardo.monchieri@unipd.it)
#  Elena Camuffo (elenacamuffo97@gmail.com)
#  Francesco Barbato (francesco.barbato@dei.unipd.it)
#  Pietro Zanuttigh (zanuttigh@dei.unipd.it)
#  Simone Milani (simone.milani@dei.unipd.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
################################################################################

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pycocotools import mask as mask_utils
import json
import os
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse

def plot_pred_mask(pred_mask, title="Predicted Mask"):
    """
    Plots a predicted mask with different colors for each instance.
    """
    pred_mask_vis = pred_mask.copy()
    # Set background to 0 (optional)
    pred_mask_vis[pred_mask_vis == -1] = 0

    # Use a discrete colormap
    num_instances = len(np.unique(pred_mask_vis))
    cmap = plt.cm.get_cmap('tab20', num_instances)

    plt.figure(figsize=(8, 8))
    plt.imshow(pred_mask_vis, cmap=cmap, interpolation='nearest')
    plt.colorbar(ticks=np.arange(num_instances), label="Instance ID")
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load descriptors
DATASETS = ["figurines", "ramen", "teatime", "waldo_kitchen"]

# Dataset-specific image sizes (H, W)
DATASET_IMAGE_SIZES = {
    "figurines": (728, 986),
    "ramen": (731, 988),
    "teatime": (730, 988),
    "waldo_kitchen": (725, 985),
}


######################
# COMPUTE SIMILARITY #
######################
def get_correlation(dataset):
    """
    Given a dateset import its textual and intstance descriptros.
    Compute the cosine similairty between each query and each object.
    Assign to each label the best match w.r.t. a correlation thrashold DELTA.
    
    return: topk_mathces{[]}: a dictionaey where each label contains the associated instance/s 
    """
    
    image_descriptors = torch.load(f"./data/{dataset}/clip_descriptors.pt", weights_only=False)
    label_descriptors = torch.load(f"./data/{dataset}/all_text_descriptors.pt", weights_only=False)

    image_names = list(image_descriptors.keys())
    label_names = list(label_descriptors.keys())

    image_embs = torch.stack([image_descriptors[n] for n in image_names])
    label_embs = torch.stack([label_descriptors[n] for n in label_names]).squeeze(1)

    image_embs = F.normalize(image_embs, dim=1)
    label_embs = F.normalize(label_embs, dim=1)

    # Similarity matrix: labels x images
    similarity_matrix = label_embs @ image_embs.T
    topk_matches = {}
    
    DELTA = 0.02  # Correlation thrashold 

    topk_matches = {}

    # Assign to each label the best match instance/s
    
    for i, lbl in enumerate(label_names):
        sims = similarity_matrix[i]  

        max_sim = sims.max()
        sorted_indices = torch.argsort(sims, descending=True)

        topk_matches[lbl] = [
            image_names[idx]
            for idx in sorted_indices
            if sims[idx] >= max_sim - DELTA
        ]
        
    return topk_matches

###############
# COMPUTE IoU #
###############

def gt_json_to_label_array(json_path):
    """
    Given a path ,open the JSON containing the GT segmenetation.
    Convert the JSON GT segmentation to a boolean mask.
    
    return: the boolean masks of object segmented in the JSON file
    """
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    height = data['info']['height']
    width = data['info']['width']

    gt_objects = []
    for obj in data['objects']:
        cat = obj['category']
        poly = []
        mask = np.zeros((height, width), dtype=bool)
        for xy_pair in obj['segmentation']:
            poly.extend([float(xy_pair[0]), float(xy_pair[1])])
        rle = mask_utils.frPyObjects([poly], height, width)
        mask = np.logical_or(mask, mask_utils.decode(rle)[:, :, 0].astype(bool))
        gt_objects.append((cat, mask))
    return gt_objects


def assign_instances_greedy(gt_objects, pred_masks, topk_matches):
    """
    Assign predicted instances to GT objects.
    - Each predicted instance can be assigned ONLY ONCE
    - For each GT object, pick the candidate with the associated wuery with maximum IoU
    """
    
    final_label_to_instance = {}
    used_instances = set()   

    for gt_idx, (gt_label, gt_mask) in enumerate(gt_objects):

        candidates = topk_matches[gt_label]

        best_iou = -1.0
        best_candidate = None

        for pred_name in candidates:
            
            pred_name_norm = str(int(pred_name))

          
            if pred_name_norm in used_instances:        # skip if already used 
                continue
            
            if pred_name_norm not in pred_masks:        # skip if not correlated
                continue

            pred_mask = pred_masks[pred_name_norm]
            union = (gt_mask | pred_mask).sum()
            if union == 0:
                continue

            iou_val = (gt_mask & pred_mask).sum() / union

            if iou_val > best_iou:
                best_iou = iou_val
                best_candidate = pred_name_norm

        # assign (or None if nothing matched)
        final_label_to_instance[(gt_idx, gt_label)] = best_candidate
        if best_candidate is not None:
            used_instances.add(best_candidate)          # mark the instance as used

    return final_label_to_instance

def compute_metrics(gt_objects, pred_masks, final_label_to_instance):
    """
    Compute the mIoU between gt_objects masks and pred_masks
    
    return:  
        - mIoU, mAcc(25) and mAcc(50) float
        - ious: containing  all the computed IoU dictionary
    """
    ious = {}
    for (gt_idx, lbl), pred_name in final_label_to_instance.items():
        gt_mask = gt_objects[gt_idx][1]
        if pred_name is not None:
            pred_mask = pred_masks[pred_name]
            ious[f"{lbl}_{gt_idx}"] = float((gt_mask & pred_mask).sum() / (gt_mask | pred_mask).sum())
        else:
            ious[f"{lbl}_{gt_idx}"] = 0.0

    mIoU = np.mean(list(ious.values()))
    mAcc25 = np.mean([v >= 0.25 for v in ious.values()])
    mAcc50 = np.mean([v >= 0.5 for v in ious.values()])

    return mIoU, mAcc25, mAcc50, ious



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation evaluation")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output and visualizations"
    )
    
    parser.add_argument(
        "--plot_pred",
        action="store_true",
        help="Enable plot predicted mask"
    )

    args = parser.parse_args()
    
    
    for ds in DATASETS:
        print(f"\n\n=========== {str(ds).upper()} ===========")
        if ds not in DATASET_IMAGE_SIZES:
            raise ValueError(f"Unknown dataset: {ds}")
        
        image_height, image_width = DATASET_IMAGE_SIZES[ds]

        test_dir = f"./testing_view/{ds}/test/"

        # Count files matching the pattern gt_*.json
        NUM_TESTS = len(glob.glob(os.path.join(test_dir, "gt_*.json")))
        
        #Get correlation label-instances
        topk_matches = get_correlation(ds)


        tot_mIoU = 0
        tot_mAcc25 = 0
        tot_mAcc50 = 0
        

        for ID in range(1, NUM_TESTS+1):
            
            gt_json_path = f"./testing_view/{ds}/test/gt_{ID}.json"
            pred_mask_path = f"./testing_view/{ds}/test/pred_{ID}.npy"

            gt_objects = gt_json_to_label_array(gt_json_path)
            pred_mask = np.load(pred_mask_path).reshape(image_height, image_width)
            if(args.plot_pred):
                plot_pred_mask(pred_mask, title=f"Predicted Mask - Scene {ID}")

            # per-instance predicted masks
            pred_instance_names = np.unique(pred_mask)
            pred_masks = {str(inst_id): (pred_mask == inst_id) for inst_id in pred_instance_names if inst_id != -1}
            
            # assign instances
            final_label_to_instance = assign_instances_greedy(gt_objects, pred_masks, topk_matches)

            # compute metrics
            mIoU, mAcc25, mAcc50, ious = compute_metrics(gt_objects, pred_masks, final_label_to_instance)

            if(args.verbose):
                print(f"🔹 SCENE {ID}")
                print("Per-object IoU:")
                for item, score in sorted(ious.items(), key=lambda x: x[1], reverse=True):
                    print(f"{item:25} : {score:.3f}")
                print(f"Scene mIoU: {mIoU:.4f}, mAcc25: {mAcc25:.4f}, mAcc50: {mAcc50:.4f}\n")

            tot_mIoU += mIoU
            tot_mAcc25 += mAcc25
            tot_mAcc50 += mAcc50

        print("----- FINAL RESULTS -----")
        print(f"Final mIoU: {tot_mIoU / NUM_TESTS:.4f}")
        print(f"Final mAcc25: {tot_mAcc25 / NUM_TESTS:.4f}")
        print(f"Final mAcc50: {tot_mAcc50 / NUM_TESTS:.4f}")
    print("\n\n")