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

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
import torch
import torch.nn.functional as F
import json
import os
import argparse
import hashlib

# ============================================================
# -------------------- Utility functions ---------------------
# ============================================================

def calculate_per_instance_iou(pred, gt):
    gt_instances = np.unique(gt)
    pred_instances = np.unique(pred)

    per_instance_iou = {}
    for gt_id in gt_instances:
        gt_mask = gt == gt_id
        best_iou = 0.0
        for pred_id in pred_instances:
            pred_mask = pred == pred_id
            inter = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)
            iou = inter / union if union > 0 else 0.0
            best_iou = max(best_iou, iou)
        per_instance_iou[gt_id] = best_iou

    return per_instance_iou


def read_ply_points_and_attrs(ply_path, ids_attr):
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"].data

    pts = np.vstack([
        vertex["x"],
        vertex["y"],
        vertex["z"]
    ]).T

    extras = {}
    for a in ids_attr:
        extras[a] = np.array(vertex[a])

    return pts, extras


def assign_to_nearest_gaussian(gt_pts, gauss_pts, gauss_labels):
    tree = KDTree(gauss_pts)
    _, idx = tree.query(gt_pts, k=1)
    return gauss_labels[idx[:, 0]]


def write_ply_with_labels(filename, pts, labels):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    rng = np.random.default_rng(42)
    label_to_color = {
        lbl: rng.integers(0, 256, size=3, dtype=np.uint8)
        for lbl in unique_labels
    }

    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
    for i, lbl in enumerate(labels):
        colors[i] = label_to_color[lbl]

    vertex = np.zeros(pts.shape[0], dtype=[
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("pred", "i4"),
    ])

    vertex["x"] = pts[:, 0]
    vertex["y"] = pts[:, 1]
    vertex["z"] = pts[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    vertex["pred"] = labels

    PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(filename)


def write_gt_ply(filename, pts, gt_labels):
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
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("gt", "i4"),
    ])

    vertex["x"] = pts[:, 0]
    vertex["y"] = pts[:, 1]
    vertex["z"] = pts[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    vertex["gt"] = gt_labels

    PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(filename)


# ============================================================
# -------------------- Evaluation metrics --------------------
# ============================================================

def calculate_iou(pred, gt, pred_cnt, gt_cnt):
    iou = np.zeros((gt_cnt, pred_cnt))
    for g in range(gt_cnt):
        for p in range(pred_cnt):
            inter = np.sum((gt == g) & (pred == p))
            union = np.sum((gt == g) | (pred == p))
            iou[g, p] = inter / union if union > 0 else 0
    return iou


def calculate_miou_and_macc(pred, gt):
    pred_cnt = np.unique(pred).shape[0]
    gt_cnt = np.unique(gt).shape[0]
    iou = calculate_iou(pred, gt, pred_cnt, gt_cnt)
    max_ious = np.max(iou, axis=1)
    miou = np.mean(max_ious)
    return miou, max_ious


def load_scannet_gt(scene_path, scene_name):
    ply = PlyData.read(
        os.path.join(scene_path, f"{scene_name}_vh_clean_2.labels.ply")
    )
    n_points = ply["vertex"].count

    with open(os.path.join(scene_path, f"{scene_name}_vh_clean_2.0.010000.segs.json")) as f:
        seg_indices = json.load(f)["segIndices"]

    with open(os.path.join(scene_path, f"{scene_name}_vh_clean.aggregation.json")) as f:
        segGroups = json.load(f)["segGroups"]

    points2label = {}
    for inst_id, obj in enumerate(segGroups):
        for seg_id in obj["segments"]:
            points2label[seg_id] = inst_id

    GT = np.full(n_points, -1, dtype=np.int32)
    for i, seg_id in enumerate(seg_indices):
        if seg_id in points2label:
            GT[i] = points2label[seg_id]

    return GT


# ============================================================
# ---------------------------- MAIN --------------------------
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=False)
    parser.add_argument("-verbose", action="store_true")
    args = parser.parse_args()

    scenes = [args.scene] if args.scene else [
        "scene0062_00"
        #"scene0000_00", "scene0062_00", "scene0070_00",
        #"scene0097_00", "scene0140_00", "scene0200_00",
        #"scene0347_00", "scene0400_00", "scene0590_00",
        #"scene0645_00"
    ]

    # ------------------------------------------------------------
    # Load CLIP descriptors
    # ------------------------------------------------------------
    print("Loading CLIP descriptors...")
    instance_desc = torch.load("clip_descriptors.pt")       # key: "id0_id1_id2"
    text_desc = torch.load("all_text_descriptors.pt")       # key: class name

    for k in instance_desc:
        instance_desc[k] = F.normalize(instance_desc[k].float(), dim=-1)

    for k in text_desc:
        text_desc[k] = F.normalize(text_desc[k].float(), dim=-1)

    class_names = list(text_desc.keys())
    text_feats = torch.cat([text_desc[c] for c in class_names], dim=0)

    # ------------------------------------------------------------
    # Instance → semantic class mapping
    # ------------------------------------------------------------
    instance_to_class = {}
    for inst_key, inst_feat in instance_desc.items():
        sim = inst_feat @ text_feats.T
        cls_id = torch.argmax(sim, dim=1).item()
        instance_to_class[inst_key] = cls_id + 1  # background = 0

    # ------------------------------------------------------------
    # Process scenes
    # ------------------------------------------------------------
    for scene_name in scenes:

        print(f"\nProcessing {scene_name}")
        scene_path = f"./data/{scene_name}"

        gt_ply = os.path.join(scene_path, f"{scene_name}_vh_clean_2.labels.ply")
        gauss_ply = os.path.join(scene_path, "pred.ply")

        gt_pts, _ = read_ply_points_and_attrs(gt_ply, ids_attr=[])
        gauss_pts, extras = read_ply_points_and_attrs(
            gauss_ply, ids_attr=["id_0", "id_1", "id_2"]
        )

        id0, id1, id2 = extras["id_0"], extras["id_1"], extras["id_2"]

        gauss_labels = np.zeros(len(id0), dtype=np.int64)
        missing = 0
        for i, (a, b, c) in enumerate(zip(id0, id1, id2)):
            key = f"{a}_{b}_{c}"
            if key in instance_to_class:
                gauss_labels[i] = instance_to_class[key]
            else:
                gauss_labels[i] = 0
                missing += 1

        if missing > 0:
            print(f"⚠️ Missing CLIP descriptors for {missing} Gaussians")

        pred_labels = assign_to_nearest_gaussian(gt_pts, gauss_pts, gauss_labels)

        torch.save(torch.tensor(pred_labels), "pred_indices.pt")
        write_ply_with_labels(f"{scene_name}_predicted_labels.ply", gt_pts, pred_labels)

        GT = load_scannet_gt(scene_path, scene_name)
        mask = GT >= 0
        GT = GT[mask]
        pred = pred_labels[mask]

        _, pred = torch.unique(torch.tensor(pred), return_inverse=True)
        pred = pred.numpy()

        miou, ious = calculate_miou_and_macc(pred, GT)
        m50 = np.mean(ious > 0.5)
        m25 = np.mean(ious > 0.25)

        print(f"mIoU: {miou * 100:.2f}")
        print(f"m50 : {m50 * 100:.2f}")
        print(f"m25 : {m25 * 100:.2f}")

        if args.verbose:
            per_inst = calculate_per_instance_iou(pred, GT)
            print(f"mean instance mIoU: {np.mean(list(per_inst.values())):.4f}")
