import os
import numpy as np
from PIL import Image
from itertools import combinations
import argparse

# -------------------------
# Loading masks
# -------------------------
def load_instance_masks(instance_dir):
    """Load all PNG masks in instance_dir/masks/ as boolean numpy arrays keyed by filename."""
    masks_dir = os.path.join(instance_dir, "masks")
    masks = {}

    for fname in os.listdir(masks_dir):
        if fname.endswith(".png"):
            path = os.path.join(masks_dir, fname)
            mask = np.array(Image.open(path).convert("L")) > 0
            masks[fname] = mask

    return masks


# -------------------------
# Mask metrics
# -------------------------
def mask_iou(mask1, mask2):
    """Compute Intersection over Union between two boolean numpy masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def mask_containment(mask_a, mask_b):
    """
    Percentage of mask_a that lies inside mask_b
    """
    area_a = mask_a.sum()
    if area_a == 0:
        return 0.0

    intersection = np.logical_and(mask_a, mask_b).sum()
    return intersection / area_a


# -------------------------
# Multi-view aggregation
# -------------------------
def instance_containment(
    masks_a,
    masks_b,
    reduction="mean",
    min_shared_views=1,
):
    """
    Compute how much of instance A is contained inside instance B across shared views.
    Returns (score, n_shared_views); score is 0.0 if fewer than min_shared_views views are shared.
    """
    shared_views = set(masks_a.keys()) & set(masks_b.keys())

    if len(shared_views) < min_shared_views:
        return 0.0, 0

    scores = [
        mask_containment(masks_a[v], masks_b[v])
        for v in shared_views
    ]

    if reduction == "mean":
        score = float(np.mean(scores))
    elif reduction == "max":
        score = float(np.max(scores))
    else:
        raise ValueError("reduction must be 'mean' or 'max'")

    return score, len(shared_views)


# -------------------------
# Instance relationship search
# -------------------------
def find_instance_containment(
    root_dir,
    containment_threshold=0.8,
    min_shared_views=2,
):
    """
    Find all (child, parent) instance pairs in root_dir where one instance is largely contained in the other.
    Returns a list of dicts with keys: child, parent, containment, shared_views.
    """
    instances = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    instance_masks = {
        inst: load_instance_masks(inst)
        for inst in instances
    }

    results = []

    for a, b in combinations(instances, 2):
        a_in_b, shared = instance_containment(
            instance_masks[a],
            instance_masks[b],
            min_shared_views=min_shared_views,
        )

        b_in_a, _ = instance_containment(
            instance_masks[b],
            instance_masks[a],
            min_shared_views=min_shared_views,
        )

        if a_in_b >= containment_threshold:
            results.append({
                "child": os.path.basename(a),
                "parent": os.path.basename(b),
                "containment": a_in_b,
                "shared_views": shared,
            })

        if b_in_a >= containment_threshold:
            results.append({
                "child": os.path.basename(b),
                "parent": os.path.basename(a),
                "containment": b_in_a,
                "shared_views": shared,
            })

    return results
# RUN AFTER REFINEMENT

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation evaluation")

    parser.add_argument(
        "--scene",
        type=str, 
        default=None,
        help="scene to clean masks"
    )
    
    args = parser.parse_args()

    SCENE = args.scene

    root = os.path.join("./data", SCENE,"masks")
    res = find_instance_containment(root)

    res_sorted = sorted(res, key=lambda x: x["containment"], reverse=True)

    for r in res_sorted:
        print(
            f"child {r['child']} <-> parent: {r['parent']} | "
            f"containment={r['containment']:.3f}, "
            f"shared_views={r['shared_views']}, "
        )