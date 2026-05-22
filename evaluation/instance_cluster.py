################################################################################
# Part of the code adapted from: https://github.com/changandao/VALA
# Copyright (c) 2025. 
#
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

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
import torch
import hashlib
import json
import os
import argparse

def calculate_per_instance_iou(pred, gt):
    """
    Compute IoU for each GT instance individually.

    Returns:
        per_instance_iou: dict {gt_instance_id: IoU value}
    """
    gt_instances = np.unique(gt)
    pred_instances = np.unique(pred)

    per_instance_iou = {}
    for gt_id in gt_instances:
        gt_mask = gt == gt_id

        best_iou = 0
        for pred_id in pred_instances:
            pred_mask = pred == pred_id
            intersection = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou

        per_instance_iou[gt_id] = best_iou

    return per_instance_iou

def triple_hash(tup):
    """
    Generate a univoque integere id starting from the id triplette (tup) 
    """
    s = f"{tup[0]}_{tup[1]}_{tup[2]}"
    # take first 8 hex digits → 32 bit
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

def read_ply_points_and_attrs(ply_path, ids_attr):
    """
    Read XYZ and extract ids parameter from a PLY file.
    """
    ply = PlyData.read(ply_path)
    vertex = ply['vertex'].data
    
    pts = np.vstack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ]).T
    
    extras = {}
    for a in ids_attr:
        extras[a] = np.array(vertex[a])
    return pts, extras

def assign_to_nearest_gaussian(gt_pts, gauss_pts, gauss_labels):
    """
    Assign each GT point to nearest Gaussian center and relative labels.
    """
    tree = KDTree(gauss_pts)
    dist, idx = tree.query(gt_pts, k=2)

    return gauss_labels[idx[:, 0]]

def write_ply_with_labels(filename, pts, labels):
    """
    Save points with predicted label and random color per label to a PLY file.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    # Assign a random color to each label
    rng = np.random.default_rng(42)  # fixed seed for reproducibility (optional)
    label_to_color = {
        lbl: rng.integers(0, 256, size=3, dtype=np.uint8)
        for lbl in unique_labels
    }

    # Create color array per point
    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
    for i, lbl in enumerate(labels):
        colors[i] = label_to_color[lbl]

    # Define PLY vertex structure
    vertex = np.zeros(pts.shape[0], dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('pred', 'i4')
    ])

    vertex['x'] = pts[:, 0]
    vertex['y'] = pts[:, 1]
    vertex['z'] = pts[:, 2]
    vertex['red'] = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue'] = colors[:, 2]
    vertex['pred'] = labels

    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(filename)

def write_gt_ply(filename, pts, gt_labels):
    """
    Save GT point cloud with GT instance labels (standalone PLY).
    """
    gt_labels = np.asarray(gt_labels)

    unique_labels = np.unique(gt_labels)
    rng = np.random.default_rng(42)

    label_to_color = {
        lbl: rng.integers(0, 256, size=3, dtype=np.uint8)
        for lbl in unique_labels if lbl >= 0
    }

    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
    for i, lbl in enumerate(gt_labels):
        if lbl >= 0:
            colors[i] = label_to_color[lbl]

    vertex = np.zeros(pts.shape[0], dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('gt', 'i4')
    ])

    vertex['x'] = pts[:, 0]
    vertex['y'] = pts[:, 1]
    vertex['z'] = pts[:, 2]
    vertex['red'] = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue'] = colors[:, 2]
    vertex['gt'] = gt_labels

    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(filename)

#*********************************
#************ TESTING ************
#*********************************
#(copyed from Instance GS) 

def calculate_iou(pred, gt, pred_cnt, gt_cnt):
    iou = np.zeros((gt_cnt, pred_cnt))
    for cls in range(gt_cnt):
        for pred_cls in range(pred_cnt):
            intersection = np.sum((pred == pred_cls) & (gt == cls))
            union = np.sum((pred == pred_cls) | (gt == cls))
            iou[cls, pred_cls] = intersection / union if union > 0 else 0
    return iou

def calculate_miou_and_macc(pred, gt):
    pred_cnt = np.unique(pred).shape[0]
    gt_cnt = np.unique(gt).shape[0]
    iou = calculate_iou(pred, gt, pred_cnt, gt_cnt)
    max_ious = np.max(iou, axis=1)
    miou = np.mean(max_ious)
    return miou, max_ious

def load_scannet_gt(scene_path, scene_name):
    # read GT PLY (only for vertex count)
    ply = PlyData.read(
        os.path.join(scene_path, f"{scene_name}_vh_clean_2.labels.ply")
    )
    n_points = ply['vertex'].count

    # read segmentation indices
    with open(os.path.join(scene_path, f"{scene_name}_vh_clean_2.0.010000.segs.json")) as f:
        seg_indices = json.load(f)['segIndices']

    # read aggregation
    with open(os.path.join(scene_path, f"{scene_name}_vh_clean.aggregation.json")) as f:
        segGroups = json.load(f)['segGroups']

    points2label = {}
    for inst_id, obj in enumerate(segGroups):
        for seg_id in obj['segments']:
            points2label[seg_id] = inst_id

    GT_instance_label = np.full(n_points, -1, dtype=np.int32)
    for i, seg_id in enumerate(seg_indices):
        if seg_id in points2label:
            GT_instance_label[i] = points2label[seg_id]

    return GT_instance_label

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a ScanNet scene")
    parser.add_argument(
        "--scene",
        type=str,
        required=False,
        help="Scene name, e.g. scene0200_00"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()

    scene_name = args.scene
    
    verbose = args.verbose
    
    
    if scene_name== None:
        scenes = [ 'scene0000_00', 'scene0062_00', 'scene0070_00', 'scene0097_00', 'scene0140_00',
                    'scene0200_00', 'scene0347_00', 'scene0400_00', 'scene0590_00', 'scene0645_00']
    else:
        scenes= [scene_name]

    for scene_name in scenes:
        # PATHS
        scene_path = f"./data/{scene_name}"    # Scene test path
        pred_pt = "pred_indices.pt"                                                     # Produced by Gaussian reconstruction

        gt_ply = os.path.join(scene_path, f"{scene_name}_vh_clean_2.labels.ply")        # GT PLY
        gauss_ply = os.path.join(scene_path, "pred.ply")                                # Gaussian reconstruction PLY                       

        out_pred_ply = f"{scene_name}_predicted_labels.ply"                                           # GT with predicted label assigned

        # === READ POINT CLOUDS ===
        # GT points only need xyz
        gt_pts, _ = read_ply_points_and_attrs(gt_ply, ids_attr=[])

        # Gaussian PLY: read xyz + triplette ids
        gauss_pts, extras = read_ply_points_and_attrs(
            gauss_ply, ids_attr=['id_0', 'id_1', 'id_2']
        )

        # === ENCODE INSTANCE LABELS ===
        id0 = extras['id_0']
        id1 = extras['id_1']
        id2 = extras['id_2']

        triples = zip(id0.tolist(), id1.tolist(), id2.tolist())
        gauss_labels = np.array([triple_hash(t) for t in triples], dtype=np.int64)

        # === NEAREST NEIGHBOR ASSIGNMENT ===
        pred_labels = assign_to_nearest_gaussian(gt_pts, gauss_pts, gauss_labels)

        # === SAVE PYTORCH TENSOR FOR SCANNET EVAL === (optional)
        pred_tensor = torch.tensor(pred_labels, dtype=torch.long)
        torch.save(pred_tensor, pred_pt)
       
        # === OPTIONAL: SAVE PLY OF GT POINTS WITH PRED LABELS === (optional)
        write_ply_with_labels(out_pred_ply, gt_pts, pred_labels)

        #*********************************
        #************ TESTING ************
        #*********************************
        
        # ---------- LOAD DATA ----------
        GT = load_scannet_gt(scene_path, scene_name)
        pred = torch.load(pred_pt).numpy()
        
        out_gt_ply = f"{scene_name}_gt.ply"
        write_gt_ply(out_gt_ply, gt_pts, GT)

        assert len(GT) == len(pred), "GT and prediction size mismatch!"

        # ---------- APPLY MASK ----------
        mask = GT >= 0
        GT = GT[mask]
        pred = pred[mask]

        # ---------- MAKE PRED LABELS CONTIGUOUS ----------
        _, pred = torch.unique(torch.tensor(pred), return_inverse=True)
        pred = pred.numpy()

        # ---------- EVALUATE ----------
        miou, ious = calculate_miou_and_macc(pred, GT)
        
        m50 = np.mean(ious > 0.5)
        m25 = np.mean(ious > 0.25)

        print(f"{scene_name}")
        print(f"mIoU: {miou * 100:.2f}")
        print(f"m50 : {m50 * 100:.2f}")
        print(f"m25 : {m25 * 100:.2f}")
        
        if(verbose):
            # ---------- COMPUTE PER-INSTANCE IOU ----------
            per_instance_iou = calculate_per_instance_iou(pred, GT)

            # ---------- PRINT TABLE ----------
            print("\nPer-instance IoU:")
            print(f"{'Instance':>10} | {'IoU':>6}")
            print("-" * 20)
            for inst_id, iou_val in per_instance_iou.items():
                print(f"{inst_id:10d} | {iou_val:6.3f}")
            
            print("-" * 20)
            print(f"mean instance mIoU: { np.mean(list(per_instance_iou.values()))}")
        print("\n")    
        print("#"*100)
       