import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image as IMG
from tqdm import tqdm
import torch
import traceback
from scipy.spatial import ConvexHull
import shutil

from sklearn.cluster import DBSCAN

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib
matplotlib.use("TkAgg")  
import open3d as o3d

from collections import Counter
from sklearn.neighbors import LocalOutlierFactor
import cv2

from scipy.optimize import minimize

import pycolmap

from matplotlib import colors as clrs

import argparse

from scipy.ndimage import binary_erosion, binary_dilation
import sys
sys.path.append("./point_projection")

import point_projection_cuda as ppc

from pathlib import Path


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


#intialize the model
BASE_DIR = Path(__file__).resolve().parent

sam2_checkpoint_init = (
        BASE_DIR / ".." / "checkpoints" / "sam2.1_hiera_large.pt"
    ).resolve()
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint_init, device=device)

predictor = SAM2ImagePredictor(sam2_model)


def update_label_points(labels, pointlabels, GT, pcd = None):
    """
    - labels: dictionary labels-> (point_id)
    - pointlabels: dictionart point_id-> [labels]
    For each label update their set of points accordingly with the majority voting.
    Remove labels which contains less then 10 points and update accortingly the pointlabels datastructure
    EXTRA: cluster the final pcd
    """
   
    for p in pointlabels:
        label_prob = get_label(pointlabels[p])

        major_label, prob = label_prob[0] if label_prob else (-1, -1)
        

        for l in labels: 
            if((l!=major_label or prob<ACCURACY_LABELS) and p in labels[l] ):
                labels[l].remove(p)
            elif(l==major_label and p not in labels[l]):
                if(prob >= ACCURACY_LABELS):
                    labels[l].add(p)
           
               

    to_remove = [l for l, pts in labels.items() if len(pts) < 5]
    for l in to_remove:
        for p in labels[l]:
            if l in pointlabels[p]:
                del pointlabels[p][l]
        del labels[l]   

    
    #Refactor labels
    
    # Step 3: Reindex labels (1 to N)
    old_to_new = {old: new+1 for new, old in enumerate(sorted(labels.keys()))}
    new_labels = {old_to_new[old]: pts for old, pts in labels.items()}

    # Step 4: Update pointlabels with new indices
    new_pointlabels = {}

    for p, labs in pointlabels.items():
        new_pointlabels[p] = {
            old_to_new[l]: v
            for l, v in labs.items()
            if l in old_to_new
        }

    return new_labels, new_pointlabels


##################################################################################
EPSILON = 0.02

ACCURACY_LABELS = 0.7

def erode_mask(mask, size, a=0.10, b=0.30):
    """
    Erodes a binary mask with a kernel size and iteration count chosen
    based on the relative object size to remove boundary noise.
    """
    if(size<a):
        mask_erosion = binary_erosion(mask, structure=np.ones((3, 3)), iterations=3).astype(mask.dtype)
    elif(size<b):
        mask_erosion = binary_erosion(mask, structure=np.ones((5, 5)), iterations=2).astype(mask.dtype)
    else:
        mask_erosion = binary_erosion(mask, structure=np.ones((5, 5)), iterations=2).astype(mask.dtype)
    return mask_erosion

def clean_labels(image, img, camera, depth, pcd, labels, pointlabels):
    """ 
    Use visible labels points to generate mask of view i.
    If 2 label generate the same mask (high IoU rate): merge the labels
    """
  
    masks = []  
    n = 10
    H, W = camera.height, camera.width

    l3 = max(labels)

    for l, point_ids in labels.items():
        label_points = [pcd[i] for i in point_ids]
        #_, visible_label_2D, _ = get2D_masked_pcd(image, camera, depth, label_points)

        #to tensor
        t_pcd_points = torch.from_numpy(label_points).float().cuda()                         # Point cloud CUDA
        t_depth = torch.from_numpy(depth).float().cuda()                                        # Depth map CUDA
        t_points_2D = torch.full((H,W), -1, device="cuda", dtype=torch.float32)                 # 2D-> 3D ids CUDA (init to -1)
        t_computed_depth = torch.full((H,W), float('inf'), device="cuda", dtype=torch.float32)  # z_buffer CUDA (init to inf)
    
        ppc.pcd2D(t_pcd_points, t_depth, extrinsics, intrinsics, EPSILON, t_points_2D, t_computed_depth)

        ys, xs = torch.where(t_points_2D != -1)
        visible_label_2D = torch.stack([xs, ys], dim=1).detach().cpu().numpy()
        #visible_ids = t_points_2D[ys, xs].long().detach().cpu().numpy()
        #visible_point3D = pcd_points[visible_ids]

        if(visible_label_2D is None): continue
        p = greedy_coreset_2D(visible_label_2D, n)
        input_point = np.array(p)
        input_label = np.ones(len(p))    

        predictor.set_image(img)
        mask_orignal, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = mask_to_array(mask_orignal)

        save_mask(mask, image.name, str(l))

        masks.append((l, mask))
    
    
    label_merge = []
    merged_labels = set()

    for i, m1 in enumerate(masks):
        m1_label = m1[0]
        m1_mask = m1[1]
        
        size = m1_mask.sum()/(H*W)
        mask_erosion_1 = erode_mask(m1_mask, size)

        for j in range(i + 1, len(masks)):
            
            m2 = masks[j]
            m2_label = m2[0]
            m2_mask = m2[1]

            size = m2_mask.sum()/(H*W)
            mask_erosion_2 = erode_mask(m2_mask, size)

            # compute intersection and union
            intersection = (mask_erosion_1 & mask_erosion_2).sum()
            union = (mask_erosion_1 | mask_erosion_2).sum()

            if intersection == 0:
                continue

            # compute the overlap score
            area1 = m1_mask.sum()
            area2 = m2_mask.sum()

            smaller_area = min(area1, area2)
            overlap_score = intersection / smaller_area

            # standard IoU (intersection over union)
            iou_score = intersection / union if union > 0 else 0

            w_iou = 1
            w_overlap = 0.75
            if((w_iou * iou_score) + (w_overlap * overlap_score) > 1.50):
                label_merge.append((m1_label, m2_label))
                merged_labels.add(m1_label)
                merged_labels.add(m2_label)
            
    new_labels = {}
    for l1, l2 in label_merge:
        l3 = l3 + 1
  

        new_labels[l3] = labels[l1] | labels[l2]
        for p in  new_labels[l3]:
            pointlabels[p] = [l3 if (lbl == l1 or lbl == l2) else lbl for lbl in pointlabels[p]]



    old_labels = {k: v for k, v in labels.items() if k not in merged_labels and len(v) > 10}
    new_labels.update(old_labels)

    #Refactor labels
    # Step 2: Remove empty label entries
    new_labels = {l: pts for l, pts in new_labels.items() if len(pts) > 0}
    
    # Step 3: Reindex labels (1 to N)
    old_to_new = {old: new+1 for new, old in enumerate(sorted(new_labels.keys()))}
    new_labels = {old_to_new[old]: pts for old, pts in new_labels.items()}

    # Step 4: Update pointlabels with new indices
    new_pointlabels = {}
    for p, labs in pointlabels.items():
        new_pointlabels[p] = [old_to_new[l] for l in labs if l in old_to_new]

    return new_labels, new_pointlabels

def get_label(labels):
    """
    - labels: list of labels [labels]
    Compute the majority voting on the list of labels, weight are defined as follow:
    - 1
    To avoid conflict we give a +0.25 to the first label assigned

    return the majority label and the counter
    
    """

    if not labels:
        return []

    counter = Counter(labels)

    total_weight = sum(counter.values())

    # Compute normalized probabilities
    probabilities = [(label, weight / total_weight) for label, weight in counter.items()]
    probabilities.sort(key=lambda x: x[1], reverse=True)

    return probabilities
    
def majority_per_key(data):
    """
    - data: dictionary point_id->[labels]
    - GT: list of zero truth label [gt labels]
    for each point compute the majority voting and return a dictionary point_id -> majority_label
    """
    result = {}
    for k, labels in data.items():
        label_prob = get_label(labels)
        majority, prob = label_prob[0] if label_prob else (-1, -1)
        if(majority==-1 or prob < ACCURACY_LABELS):
            continue
        result[k] = majority
    return result

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_masks(image, masks, scores, point_coords=None, input_labels=None, borders=True, img_id=None, title = "0"):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())

        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if not os.path.isdir("./mask_test/"+ img_id ):
            os.makedirs("./mask_test/"+ img_id )
        plt.savefig(f"./mask_test/{img_id}/{title}")

def show_masks(image, masks, scores, point_coords=None, input_labels=None, borders=True, img_id= None, title = "0"):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())

        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if not os.path.isdir("./mask_test/"+ img_id ):
            os.makedirs("./mask_test/"+ img_id )
        plt.savefig(f"./mask_test/{img_id}/{title}")

def mask_to_array(masks):
    if isinstance(masks, list):
        masks = np.array(masks)
    
    # If multiple masks, collapse into one (any overlap counts as mask)
    mask = (np.sum(masks, axis=0) > 0).astype(np.uint8)
    
    # Return just the binary mask array instead of RGBA image
    return mask

def greedy_coreset_2D(S, n, alpha=0.3, exclude_hull=True):
    """
    Greedy coreset selection for 2D points, avoiding border points.

    Parameters:
    -----------
    S : np.ndarray or list
        Input 2D points of shape (num_points, 2)
    n : int
        Desired coreset size
    alpha : float
        Penalty weight for distance from mean (0 = ignore, 1 = strong penalty)
    exclude_hull : bool
        Whether to exclude convex hull (outer boundary) points before selection

    Returns:
    --------
    C : list
        List of selected coreset points
    """
    # Convert input to list of lists
    if isinstance(S, np.ndarray):
        S = S.tolist()
    C = []

    original_S = S.copy()
    
    # --- Step 0: Optionally exclude convex hull points ---
    if exclude_hull and len(S) > 3:
        hull = ConvexHull(S)
        hull_points = set(tuple(S[i]) for i in hull.vertices)
        S = [s for s in S if tuple(s) not in hull_points]

        if len(S) == 0:
            S = original_S.copy()

    # --- Step 1: Pick the point closest to the mean ---
    mean_S = np.mean(S, axis=0)
    x0 = min(S, key=lambda s: np.linalg.norm(np.array(s) - mean_S))
    C.append(x0)
    S.remove(x0)

    # --- Step 2: Iteratively add points farthest from current coreset,
    #              with a penalty for being far from the mean ---
    while len(C) < n and S:
        def score(s):
            s = np.array(s)
            dist_to_C = min(np.linalg.norm(s - np.array(c)) for c in C)
            dist_from_mean = np.linalg.norm(s - mean_S)
            # Penalize being far from the mean
            return dist_to_C - alpha * dist_from_mean

        y = max(S, key=score)
        C.append(y)
        S.remove(y)

    return C

def save_mask(mask, title, img_id):
    alpha = (mask.astype(np.uint8)) * 255
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., 0:3] = 255  
    rgba[..., 3] = alpha 

    
    img = IMG.fromarray(rgba, mode="RGBA")
    if not os.path.isdir("./mask_test_png/"+ title.replace(".jpg", ".png") ):
        os.makedirs("./mask_test_png/"+ title.replace(".jpg", ".png") )
    img.save(os.path.join("./mask_test_png/", title.replace(".jpg", ".png"), img_id+".png"))

def plt_clusters(pointsLabels, labels, gt_labels):

    """
    - pointlabels: dictionart point_id-> [labels]
    - labels: dictionary labels-> (point_id)
    - GT: list of zero truth points/ground truth [points]

    Given a set of labeled points generate a colored open3D point cloud for each and plot them together
    """
    unique_labels = labels.keys()

    # Modern API for colormap
    from matplotlib.colors import ListedColormap
    points = np.asarray(pcd.points)
    
    # Generate 100 random RGB colors
    np.random.seed(42)  # for reproducibility (remove if you want different results each time)
    colors = np.random.rand(300, 3)  # shape (100, 3) for RGB

    # Create a colormap
    colormap = ListedColormap(colors, name="random_colormap")

    #print(len(unique_labels))
    label_pcds = []

    majority_labels = majority_per_key(pointsLabels)
    for label in unique_labels:
    
        ids_for_label = [pid for pid, l in majority_labels.items() if l == label]
  
        

        
        if len(ids_for_label) == 0:
            continue
        
        # collect their 3D coordinates and colors
        pts = np.array([points[pid] for pid in ids_for_label])
        # cols = np.tile(np.array(colormap(label)[:3]), (len(pts), 1))  # uniform label color
        # pcd_label = o3d.geometry.PointCloud()
        # pcd_label.points = o3d.utility.Vector3dVector(pts)
        # pcd_label.colors = o3d.utility.Vector3dVector(cols)
        # label_pcds.append(pcd_label)
        


        
        
        # === Cluster within this label ===
        db = DBSCAN(eps=0.05, min_samples=10).fit(pts)
        cluster_labels = db.labels_
        
        # Ignore noise (-1)
        valid_clusters = cluster_labels[cluster_labels != -1]
        if len(valid_clusters) == 0:
            continue
        
        
        valid_clusters = [c for c in np.unique(cluster_labels) if c != -1]
        n_clusters = len(valid_clusters)
        if(n_clusters>4):
            cols = np.tile(np.array(colormap(label)[:3]), (len(pts), 1))  # uniform label color
            pcd_label = o3d.geometry.PointCloud()
            pcd_label.points = o3d.utility.Vector3dVector(pts)
            pcd_label.colors = o3d.utility.Vector3dVector(cols)
            label_pcds.append(pcd_label)

        # Find the largest cluster
        cluster_counts = Counter(valid_clusters)
        biggest_cluster_id = cluster_counts.most_common(1)[0][0]
        
        # Filter points for the biggest cluster
        cluster_mask = (cluster_labels == biggest_cluster_id)
        pts_biggest = pts[cluster_mask]
        
        if len(pts_biggest) <= 10:
            continue
        cols = np.tile(np.array(colormap(label)[:3]), (len(pts_biggest), 1))
        
    
        # if len(pts) <= 10:
        #     continue
        
        # build point cloud
        pcd_label = o3d.geometry.PointCloud()
        pcd_label.points = o3d.utility.Vector3dVector(pts_biggest)
        pcd_label.colors = o3d.utility.Vector3dVector(cols)
     
        #o3d.visualization.draw_geometries([pcd_label])   
        #o3d.io.write_point_cloud(f"output_{label}.ply", pcd_label)
        #print(f"Saved point cloud as output_{label}.ply")
        label_pcds.append(pcd_label)

    # === Visualize all objects together, colored by label ===
    o3d.visualization.draw_geometries(label_pcds)   

def DBSCAN_points(
        t_pcd_points, 
        t_points_2D,     
        eps=0.10,           
        min_samples=10
    ):

    ids = (t_points_2D[t_points_2D!=-1]).detach().cpu().numpy()
    points3D = t_pcd_points[ids].detach().cpu().numpy()
    points2D_map = t_points_2D.detach().cpu().numpy()


    H, W = points2D_map.shape



    # ----------------------------------------
    # Run DBSCAN on the 3D points
    # ----------------------------------------
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points3D)

    if labels.size == 0 or np.all(labels == -1):
        # no cluster detected → return original
        xy = np.column_stack(np.where(points2D_map >= 0))
        return points3D, xy, ids

    # ----------------------------------------
    # Extract largest valid cluster
    # ----------------------------------------
    positive_labels = labels[labels >= 0]
    if positive_labels.size == 0:
        xy = np.column_stack(np.where(points2D_map >= 0))
        return points3D, xy, ids

    largest_cluster = np.bincount(positive_labels).argmax()
    indices = np.where(labels == largest_cluster)[0]

    if len(indices) <= 10:
        # cluster too small → return original
        xy = np.column_stack(np.where(points2D_map >= 0))
        return points3D, xy, ids

    # ----------------------------------------
    # Filter 3D, 2D, colors, ids
    # ----------------------------------------
    points3D_f = points3D[indices]
    ids_f = ids[indices]

    # ----------------------------------------
    # Extract (x,y) pixels corresponding to KEPT indices
    # ----------------------------------------
    points2D_kept_map = np.full_like(points2D_map, -1)

    # fill new map with kept indices
    for new_i, old_i in enumerate(indices):
        points2D_kept_map[points2D_map == old_i] = new_i

    # get (x,y) pixel coordinates
    xy = np.column_stack(np.where(points2D_kept_map >= 0))

    return points3D_f, xy, ids_f

######################################################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation evaluation")

    parser.add_argument(
        "-dataset",
        type=str, 
        default=None,
        help="Dataset to perform propagation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable plot intermidiate results"
    )
    
    args = parser.parse_args()
     
    DATASET = args.dataset
    
    VERBOSE = args.verbose


    #### PATH ####


    image_path = os.path.join("./data", DATASET, "images")
    colmap_model_path = os.path.join("./data", DATASET, "sparse/0")



    depth_path = os.path.join("./data", DATASET, "depth")
    mask_folder =  os.path.join("./output", f"{DATASET}_autoseg_mask")

    # Load GS reconstruction
    ply_path = os.path.join("./data", DATASET,"sparse/0/points3D.ply")


    recon = pycolmap.Reconstruction(colmap_model_path)
    images = recon.images

    sparse_points = recon.points3D

    # Generate pcd from sparse reconstruction
    xyz = np.array([p.xyz for p in sparse_points.values()])
    colors = np.array([p.color for p in sparse_points.values()]) / 255.0  # normalize to [0,1]
    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors.size > 0:
        sparse_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Generate pcd from GS dense reconstruction
    pcd = o3d.io.read_point_cloud(ply_path)


    if hasattr(pcd, 'points'):
        pcd_points = np.asarray(pcd.points)
    else: pcd_points = pcd

    segmented_pcd = o3d.geometry.PointCloud()
    labels_counter = 0

    pointsLabels = {}   # dictionary: point_id → [label]
    labelPoints = {}    # dictionary: label → {point_id}
    gt_labels_idx = 1

    projected_view = {}


    iter = 0 

    for image_id in tqdm(list(images)[::-1]):
        # Open image, depth and masks

        img_meta = images[image_id]
        cam_meta = img_meta.camera
        image_id = img_meta.image_id

        ext = img_meta.cam_from_world.matrix()

        extrinsics = torch.tensor(img_meta.cam_from_world.matrix(), device="cuda", dtype=torch.float32).contiguous()
        intrinsics = torch.tensor(cam_meta.params, device="cuda", dtype=torch.float32).reshape(2, 2)
        
        # COLMAP data   
        image_name = img_meta.name


        H, W = cam_meta.height, cam_meta.width

        
        # Load image
        img_name_2 = image_name.replace(".jpg", ".JPEG")
        img = IMG.open(os.path.join(image_path, img_name_2)).convert("RGB")
        img = np.array(img)

            
        # Load depth
        dm_name = image_name.replace(".jpg", "_pred.npy")
        dm_name = dm_name.replace(".JPEG", "_pred.npy")
        depth = np.load(os.path.join(depth_path, dm_name)).astype(np.float32)

        # Load mask
        mask_name = image_name.replace(".jpg", ".JPEG")
        mask_name = mask_name.replace(".JPEG", "")
        masks = os.listdir(os.path.join(mask_folder, mask_name))

        
        #Compute visible points from the view

        #to tensor
        t_pcd_points = torch.from_numpy(pcd_points).float().cuda()                              # Point cloud CUDA
        t_depth = torch.from_numpy(depth).float().cuda()                                        # Depth map CUDA
        t_points_2D = torch.full((H,W), -1, device="cuda", dtype=torch.float32)                 # 2D-> 3D ids CUDA (init to -1)
        t_computed_depth = torch.full((H,W), float('inf'), device="cuda", dtype=torch.float32)  # z_buffer CUDA (init to inf)
        
        
        ppc.pcd2D(t_pcd_points, t_depth, extrinsics, intrinsics, EPSILON, t_points_2D, t_computed_depth)

        ys, xs = torch.where(t_points_2D != -1)
        visible_point2D = torch.stack([xs, ys], dim=1).detach().cpu().numpy()
        visible_ids = t_points_2D[ys, xs].long().detach().cpu().numpy()
        visible_point3D = pcd_points[visible_ids]

        projected_point = {}
        for idx, p2D in enumerate(visible_point2D):
            key = (p2D[0], p2D[1])
            projected_point[key] = visible_ids[idx] 
        projected_view[image_id] = projected_point

        if(iter==0): 

            # GT Mask: project mask and generate initial 3D point instances
            for i, m in enumerate(masks):
                
                #print(f"Image: {image_name} Mask: {m}")
                mask_img = IMG.open(os.path.join(mask_folder, mask_name, m)).convert("L")
                gray = np.array(mask_img)
                mask_0 = gray > 0 

                #Adaptive erosion: based on the size of the mask define the erosione level
                size = mask_0.sum()/(H*W)
                mask_erosion = erode_mask(mask_0, size)
                
                #run mask projection

                #to tensor
                t_pcd_points = torch.from_numpy(visible_point3D).float().cuda()                         # Point cloud CUDA
                t_depth = torch.from_numpy(depth).float().cuda()                                        # Depth map CUDA
                t_points_2D = torch.full((H,W), -1, device="cuda", dtype=torch.float32)                 # 2D-> 3D ids CUDA (init to -1)
                t_computed_depth = torch.full((H,W), float('inf'), device="cuda", dtype=torch.float32)  # z_buffer CUDA (init to inf)
                t_mask = torch.from_numpy(mask_erosion).to("cuda").bool()                               # boolean mask CUDA

                ppc.pcd2D_mask(t_pcd_points, t_depth, extrinsics, intrinsics, EPSILON, t_points_2D, t_computed_depth, t_mask)

                ids = (t_points_2D[t_points_2D!=-1]).detach().cpu().numpy()
                if len(ids)==0: continue
                _, p2D, ids = DBSCAN_points(t_pcd_points, t_points_2D)

                if (p2D is None): continue

                # Create the new label set
                labelPoints[gt_labels_idx] = set()

                # Update data structures
                for id in ids: 
                    id = int(id)
                    id = visible_ids[id] # Get "global id"
                    
                    
                    if id in pointsLabels: #To avoid duplicates update the datastructures
                        for old_label in pointsLabels[id]:
                            if id in labelPoints[old_label]:
                                labelPoints[old_label].remove(id)

                    # Create the point label list
                    if id not in pointsLabels:
                        pointsLabels[id] = {}
                        #pointsLabels.append(i)
                        pointsLabels[id][gt_labels_idx] = 1.25
                
                    # Add the label to the list
                    elif gt_labels_idx not in  pointsLabels[id]:
                        pointsLabels[id][gt_labels_idx] = 1.00
                    else:       
                        pointsLabels[id][gt_labels_idx]+= 1.00        
            
                    # Add the point_id to the label list
                    labelPoints[gt_labels_idx].add(id)
                
                gt_labels_idx +=1 #index update

            #create the list of zero-truth labels
            gt_labels = set(labelPoints.keys())

            
            if(VERBOSE): plt_clusters(pointsLabels, labelPoints, gt_labels) # initial test
        
        else:
            """
            Iterate over the instances
            1- Get visible 3D points DONE
            2- Project each instance visible point onto the mask of view_i
            3- Get mask where points falls in and assign to label l
                3a- If points of instance l fall on two different masks, merge
                3b- If an already assigned label l want to be assigned to the same mask, split
                3c- No overlap: new object
            4- Peoject the new mask point to the visible point to update the instance rec
            5- if some mask has not been touched, project them as new instance    
            """

            assigned_labels = {}

            mask_labels = np.full((H, W), -1)
            mask_new_labels = np.full((H, W), -1)

            # Update the datasctructures accordingly with the last iteration

            labelPoints, pointsLabels  = update_label_points(labelPoints, pointsLabels, gt_labels,  np.asarray(pcd.points))

            # Step 2: backproject the 3D to obtain a 2D "virtual mask"

            for l in labelPoints:
                points3D_label = labelPoints[l]
                for p_id in points3D_label:    
                    # Get 2D point index
                    if p_id not in visible_ids: continue
                    p = pcd.points[p_id]
                    uv = cam_meta.img_from_cam(img_meta.cam_from_world * p)
                    
                    if uv is None: continue
                    u, v = uv[0]
                            
                    x, y = int(round(u)), int(round(v))
                    if x < 0 or x >= W-2 or y < 0 or y >= H-2: continue

                    if(mask_labels[y,x]==-1): #Avoid overlaps
                        mask_labels[y,x] = l    
                
        
            # Step 3: iterate over the masks of the current view to update the instances
            for i, m in enumerate(masks): 
                
                #load the curernt mask
                mask_img = IMG.open(os.path.join(mask_folder, mask_name, m)).convert("L")
                gray = np.array(mask_img)
                mask_0 = gray > 0   

                size = mask_0.sum()/(H*W)

                # Aggressive erosion to avoid distructive overlaps
                mask_erosion = erode_mask(mask_0, size, 0.05, 0.5)

                # Get overlaps between the "virtual mask" and the current mask
                label_in_mask = np.unique(mask_labels[mask_erosion])
                label_in_mask = label_in_mask[label_in_mask >= 0] # remove -1 labels(void)

                # 3c: No overlap: no intersection with an existing mask:
                # - new object
                # - point of an existing mask not intersected with the pre-dicovered ones

                if len(label_in_mask) == 0: 
            
    
                    
                    # Step 5
                    size = mask_0.sum()/(H*W)
                    if(size<0.05):
                        mask_erosion = binary_erosion(mask_0, structure=np.ones((3, 3)), iterations=2).astype(mask_0.dtype)
                    elif(size<0.10):
                        mask_erosion = binary_erosion(mask_0, structure=np.ones((5, 5)), iterations=2).astype(mask_0.dtype)
                    else:
                        mask_erosion = binary_erosion(mask_0, structure=np.ones((5, 5)), iterations=3).astype(mask_0.dtype)

                    # Compute point_ids involved in the new masked object
                
                    #to tensor

                    t_pcd_points = torch.from_numpy(visible_point3D).float().cuda()                         # Point cloud CUDA
                    t_depth = torch.from_numpy(depth).float().cuda()                                        # Depth map CUDA
                    t_points_2D = torch.full((H,W), -1, device="cuda", dtype=torch.float32)                 # 2D-> 3D ids CUDA (init to -1)
                    t_computed_depth = torch.full((H,W), float('inf'), device="cuda", dtype=torch.float32)  # z_buffer CUDA (init to inf)
                    t_mask = torch.from_numpy(mask_erosion).to("cuda").bool()                               # boolean mask CUDA

                    ppc.pcd2D_mask(t_pcd_points, t_depth, extrinsics, intrinsics, EPSILON, t_points_2D, t_computed_depth,t_mask)

                    ids = (t_points_2D[t_points_2D!=-1]).detach().cpu().numpy()
                    if len(ids)==0: continue
                    _, p2D, ids = DBSCAN_points(t_pcd_points, t_points_2D)
                
                # if (ids is None): continue
                    
                    # new id(incremental ids)
                    label = max(labelPoints)+1
        
                    overlapping_label = []
                    # Update data structures
                    for id in ids: 
                        id = int(id)             
                        id = visible_ids[id] # Get "global id"
                    
                        if id in pointsLabels:
                            label_prob = get_label(pointsLabels[id])
                            overlap, _ =  label_prob[0] if label_prob else (-1, -1)
                            overlapping_label.append(overlap)
                    
        
                    labelPoints[label] = set()
                    for id in ids: 
                        id = int(id)
                        id = visible_ids[id] # Get "global id"

                    
                        if id not in pointsLabels:
                            pointsLabels[id] = {}
                            labelPoints[label].add(id)
                            pointsLabels[id][label]= 1.25
                        
                        
                        elif label not in  pointsLabels[id]:
                            pointsLabels[id][label]= 1.00
                        else:       
                            pointsLabels[id][label]+= 1.00 
                        
                    
                    # else:
                    #     continue
            
                # 3a: One label fits this mask and has not been added yet, update
                elif len(label_in_mask) == 1: 
        
                    #print(f"OVERLAP! for {m}")
                    
                    l = label_in_mask[0]
                    assigned_labels[l] = i

                # 3b: Minor overlap between 2 masks, check the overlap:
                # - if a new object overlap with a zero truth one assign just the "major label"(gt label) to the object
                # - if a new object overlap with another new object assigne the "oldest" label to the object (the lesser one)
                else:
                    major_label = (
                        lambda valid: None if valid.size == 0 else np.bincount(valid).argmax()
                    )(mask_labels[mask_erosion].flatten()[mask_labels[mask_erosion].flatten() != -1])

                    size = mask_0.sum()/(H*W)
                    mask_erosion = erode_mask(mask_0, size, 0.05, 0.3)
                    
                    # Get points involved and assign to them the major_label  
                            
                    #to tensor
                    t_pcd_points = torch.from_numpy(visible_point3D).float().cuda()                         # Point cloud CUDA
                    t_depth = torch.from_numpy(depth).float().cuda()                                        # Depth map CUDA
                    t_points_2D = torch.full((H,W), -1, device="cuda", dtype=torch.float32)                 # 2D-> 3D ids CUDA (init to -1)
                    t_computed_depth = torch.full((H,W), float('inf'), device="cuda", dtype=torch.float32)  # z_buffer CUDA (init to inf)
                    t_mask = torch.from_numpy(mask_erosion).to("cuda").bool()                               # boolean mask CUDA

                    ppc.pcd2D_mask(t_pcd_points, t_depth, extrinsics, intrinsics, EPSILON, t_points_2D, t_computed_depth,t_mask)

                    ids = (t_points_2D[t_points_2D!=-1]).detach().cpu().numpy()
                    if len(ids)==0: continue
                    _, p2D, ids = DBSCAN_points(t_pcd_points, t_points_2D)
                        
                    for id in ids:
                        id = int(id)
                        id = visible_ids[id] 
                    
                        if id not in pointsLabels:
                            pointsLabels[id] = {}
                            pointsLabels[id][major_label] = 1.00
                        
                        elif major_label not in  pointsLabels[id]:
                            pointsLabels[id][major_label] = 0.75
                        else:       
                            pointsLabels[id][major_label]+= 0.75
                            
                        labelPoints[major_label].add(id)

                    
            # Step 4: update the existing object wiht the new discovered masks
            for l, mask_idx in assigned_labels.items():
                mask_path = os.path.join(mask_folder, mask_name, masks[mask_idx])
                mask_img = IMG.open(mask_path).convert("L")
                gray = np.array(mask_img)
                mask_0 = gray > 0
            
                size = mask_0.sum()/(H*W) 
                mask_erosion = erode_mask(mask_0, size, 0.05, 0.3)


                #to tensor
                t_pcd_points = torch.from_numpy(visible_point3D).float().cuda()                         # Point cloud CUDA
                t_depth = torch.from_numpy(depth).float().cuda()                                        # Depth map CUDA
                t_points_2D = torch.full((H,W), -1, device="cuda", dtype=torch.float32)                 # 2D-> 3D ids CUDA (init to -1)
                t_computed_depth = torch.full((H,W), float('inf'), device="cuda", dtype=torch.float32)  # z_buffer CUDA (init to inf)
                t_mask = torch.from_numpy(mask_erosion).to("cuda").bool()                               # boolean mask CUDA

                ppc.pcd2D_mask(t_pcd_points, t_depth, extrinsics, intrinsics, EPSILON, t_points_2D, t_computed_depth,t_mask)

                ids = (t_points_2D[t_points_2D!=-1]).detach().cpu().numpy()
                if len(ids)==0: continue
                _, p2D, ids = DBSCAN_points(t_pcd_points, t_points_2D)


                # Update data structures
                for id in ids:
                    id = int(id) 
                    id = visible_ids[id] # Get "global id"

            
                    if id not in pointsLabels:
                        pointsLabels[id] = {}
                        pointsLabels[id][l] = 1.25
                    
                    elif l not in pointsLabels[id]:
                        pointsLabels[id][l] = 1.00
                    else:       
                        pointsLabels[id][l]+= 1.00 
            
        iter+=1

        
    labelPoints, pointsLabels  = update_label_points(labelPoints, pointsLabels, gt_labels,  np.asarray(pcd.points))

    if(VERBOSE): plt_clusters(pointsLabels, labelPoints, gt_labels) # PLOT to test results

    pointsLabels_def = majority_per_key(pointsLabels)


    #Point clustering
    labeled_clusters = {}
    for label in tqdm(labelPoints):
        
        p3D_ids = list(labelPoints[label])
        points3D = np.array([pcd.points[p3D_id] for p3D_id in p3D_ids])
    
        if points3D is None or len(points3D) == 0:
            continue

        # --- DBSCAN clustering on 3D points ---
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(points3D)
        labels_db = clustering.labels_

        valid_mask = labels_db != -1
        valid_points = np.array(p3D_ids)[valid_mask]
        valid_labels = labels_db[valid_mask]

        if len(valid_labels) == 0:
            continue
        
        
        # Find the largest cluster
        unique, counts = np.unique(valid_labels, return_counts=True)
        largest_cluster_id = unique[np.argmax(counts)]

        if(len(unique)>3): #More then 3 cluster, is not just a mislabeling
            labeled_clusters[label] = p3D_ids
            
            # --- Save cluster with color as PLY ---
            cluster_points_to_save = np.array([pcd.points[idx] for idx in p3D_ids])

            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points_to_save)
            output_dir = f"./output/{DATASET}_masks/{label}"
            os.makedirs(output_dir, exist_ok=True)
            ply_filename = os.path.join(output_dir, f"label_{label}.ply")
            o3d.io.write_point_cloud(ply_filename, cluster_pcd)

            del cluster_pcd
            continue
        # Keep only points from the largest cluster
        cluster_points = valid_points[valid_labels == largest_cluster_id]
        cluster_points_set = set(cluster_points.tolist())  # For fast lookup
        if(len(cluster_points))<10: continue
        
        labeled_clusters[label] = cluster_points_set

    
        # --- Save cluster with color as PLY ---
        cluster_points_to_save = np.array([pcd.points[idx] for idx in cluster_points.tolist()])

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points_to_save)

        output_dir = f"./output/{DATASET}_masks/{label}"
        os.makedirs(output_dir, exist_ok=True)
        ply_filename = os.path.join(output_dir, f"label_{label}.ply")
        o3d.io.write_point_cloud(ply_filename, cluster_pcd)
        del cluster_pcd

    clustered_points = set().union(*labeled_clusters.values())
    possible_merges = {}

    for image_id in tqdm(projected_view):

        
        img_meta = images[image_id]
        cam_meta = img_meta.camera
        
        
        # COLMAP data   
        image_name = img_meta.name
        save_name = image_name.replace(".jpg", ".png")
        save_name = save_name.replace(".JPEG", ".png")

        img_name_2 = image_name.replace(".jpg", ".JPEG")
        img = IMG.open(os.path.join(image_path, img_name_2)).convert("RGB")
        img = np.array(img)

        # Load mask
        mask_name = image_name.replace(".jpg", ".JPEG")
        mask_name = mask_name.replace(".JPEG", "")
        masks = os.listdir(os.path.join(mask_folder, mask_name))


        projected_points = projected_view[image_id]

        mask_labels = np.full((H, W), -1)

        for p2D, p3D_id in projected_points.items():
            if(p3D_id not in pointsLabels_def or p3D_id not in clustered_points): continue
            x, y = p2D
            label_prob = pointsLabels_def[p3D_id]

            major_label = label_prob if label_prob else -1
            if(major_label==-1): continue
            
            mask_labels[y, x] = major_label
        
        visiblelabels = np.unique(mask_labels)
        visiblelabels = set(visiblelabels[visiblelabels >= 0])

        created_mask = set()
        for m in masks:
        #load the curernt mask
            mask_img = IMG.open(os.path.join(mask_folder, mask_name, m)).convert("L")
            gray = np.array(mask_img)
            mask_0 = gray > 0   

            size = mask_0.sum()/(H*W)

            #Aggressive erosion to avoid distructive overlaps
            mask_erosion = erode_mask((mask_0), size)

            # Get overlaps between the "virtual mask" and the current mask
            label_in_mask = np.unique(mask_labels[mask_erosion])
            label_in_mask = label_in_mask[label_in_mask >= 0]   

            if(len(label_in_mask)>0 and label_in_mask[0] == 0): "CHECK"
            
            if (len(label_in_mask)==1 and label_in_mask[0] not in created_mask): # not created mask
                print("not created mask")
                label = label_in_mask[0]
                if(label not in labeled_clusters): 
                    print(label)
                    continue

                # Merge using bitwise OR (logical union)
                mask_bw = (mask_0.astype(np.uint8)) * 255

                # Convert back to PIL
                mask_bw_img = IMG.fromarray(mask_bw)
                
                os.makedirs(f"./output/{DATASET}_masks/{label}", exist_ok=True)
                mask_bw_img.save(os.path.join(f"./output/{DATASET}_masks/{label}", f"{save_name}"))

                #save the image
                if label in visiblelabels:
                    visiblelabels.remove(label)
                created_mask.add(label)

            elif (len(label_in_mask)==1 and label_in_mask[0] in created_mask): 
                """
                OVERSEGMENTATION: more masks of the view i relay on the same obejct, we need to merge the masks
                if a mask of the same object is already present, load and merge with  the new one(bitwise or) if not save the new one as ground mask 
                """
                print("old mask already visited")
                label = label_in_mask[0]
                if(label not in labeled_clusters): continue
                try:
                    mask_img_1 = IMG.open(os.path.join(f"./output/{DATASET}_masks/{label}/{save_name}")).convert("L") 
                    mask_np_1 = np.array(mask_img_1)
                    mask_1 = (mask_np_1 > 0).astype(np.uint8)*255   
                
                except:
                    # save the mask
                    mask_bw = (mask_0.astype(np.uint8)) * 255

                    # Convert back to PIL
                    mask_bw_img = IMG.fromarray(mask_bw)
                    
                    os.makedirs(f"./output/{DATASET}_masks/{label}", exist_ok=True)
                    mask_bw_img.save(os.path.join(f"./output/{DATASET}_masks/{label}", f"{save_name}"))

                    #save the image
                    if label in visiblelabels:
                        visiblelabels.remove(label)
                    created_mask.add(label)
                    continue

                mask_0 = (mask_0.astype(np.uint8)) * 255
                
                # Merge using bitwise OR (logical union)
                merged_mask_np = (mask_0 | mask_1)
                merged_mask_np = merged_mask_np


                #Check if the new mask overlap with different objects

                size = merged_mask_np.sum()/(H*W)

                mask_erosion = erode_mask((merged_mask_np), size)
                
                label_in_mask = np.unique(mask_labels[mask_erosion])
                label_in_mask = label_in_mask[label_in_mask >= 0]  

                # Convert back to PIL
                merged_mask = IMG.fromarray(merged_mask_np)


                if label in visiblelabels:
                        visiblelabels.remove(label)
                created_mask.add(label)

                merged_mask.save(os.path.join(f"./output/{DATASET}_masks/{label}", f"{save_name}"))

            
            elif(len(label_in_mask)>=2): # get the most represented label
                print("Overlapping mask")
                major_label = (
                    lambda valid: None if valid.size == 0 else np.bincount(valid).argmax()
                )(mask_labels[mask_erosion].flatten()[mask_labels[mask_erosion].flatten() != -1])
                
                if(major_label not in labeled_clusters): continue
                try:
                    mask_img_1 = IMG.open(os.path.join(f"./output/{DATASET}_masks/{major_label}/{save_name}")).convert("L") 
                    mask_np_1 = np.array(mask_img_1)
                    mask_1 = (mask_np_1 > 0).astype(np.uint8)*255    
                except:
                    # save the mask
                    mask_bw = (mask_0.astype(np.uint8)) * 255

                    # Convert back to PIL
                    mask_bw_img = IMG.fromarray(mask_bw)
                    
                    os.makedirs(f"./output/{DATASET}_masks/{major_label}", exist_ok=True)
                    mask_bw_img.save(os.path.join(f"./output/{DATASET}_masks/{major_label}", f"{save_name}"))

                    #save the image
                    if major_label in visiblelabels:
                        visiblelabels.remove(major_label)
                    created_mask.add(major_label)
                    continue

                mask_0 = (mask_0.astype(np.uint8)) * 255
                # Merge using bitwise OR (logical union)
                merged_mask_np = (mask_0 | mask_1)
                merged_mask_np = merged_mask_np

                #Check if the new mask overlap with different objects

                size = merged_mask_np.sum()/(H*W)

                mask_erosion = erode_mask((merged_mask_np), size)
                
                label_in_mask = np.unique(mask_labels[mask_erosion])
                label_in_mask = label_in_mask[label_in_mask >= 0]  

                # Convert back to PIL
                merged_mask = IMG.fromarray(merged_mask_np)

                created_mask.add(major_label)

                merged_mask.save(os.path.join(f"./output/{DATASET}_masks/{major_label}", f"{save_name}"))


        for label in (visiblelabels - created_mask):
            if label not in labeled_clusters: continue
            p2D_label = []
            for p2D, p3D_id in projected_points.items():
                if(p3D_id not in pointsLabels_def or p3D_id not in labeled_clusters[label]): continue

                label_prob = pointsLabels_def[p3D_id]

                major_label = label_prob if label_prob else  -1
                if(major_label==-1): continue
                elif (major_label == label):
                    p2D_label.append(p2D)      
                    ### GET 2D POINT OF LABEL IN THE VIEW USING projected_points
                    #p2D = [p for p, p3D_id in projected_points.items() if p3D_id in pointsLabels and get_label(pointsLabels[p3D_id], gt_labels)[0][0] == label]
            print(f"label {label} found {len(p2D_label)} points")
            if(len(p2D_label)<10): continue
            p = greedy_coreset_2D(p2D_label, 10)
            input_point = np.array(p)
            input_label = np.ones(len(p))    

            predictor.set_image(img)
            mask_orignal, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            mask = mask_to_array(mask_orignal)*255 
    
            # Plot 2D label points (in red)
            if len(input_point) > 0:
                plt.scatter(input_point[:, 0], input_point[:, 1], c='red', s=30, marker='o', label='Label Points')

            print("NEW MASK!")
            
            os.makedirs(f"./output/{DATASET}_masks/{label}", exist_ok=True)

            merged_mask = IMG.fromarray(mask)

            merged_mask.save(os.path.join(f"./output/{DATASET}_masks/{label}", f"{save_name}"))

        plt.close("all")
            


    new_labels = {}

    def find_root(label):
        """Recursively find the final merged label."""
        while label in new_labels:
            label = new_labels[label]
        return label

    for p in possible_merges:
        
        l_1, l_2 = p
        
        l_1 = find_root(l_1)
        l_2 = find_root(l_2)

        v_1 = len([f for f in os.listdir(f"./output/{DATASET}_masks/{l_1}")])
        v_2 = len([f for f in os.listdir(f"./output/{DATASET}_masks/{l_2}")])
        min_views = min (v_1, v_2)
        prob = possible_merges[p]/min_views

        if prob>=0.9: # merge labels:
            src = f"./output/{DATASET}_masks/{l_2}"
            dest = f"./output/{DATASET}_masks/{l_1}"

            # Merge all files (rename if duplicates exist)
            for file_name in os.listdir(src):
                src_path = os.path.join(src, file_name)
                dest_path = os.path.join(dest, file_name)
                # Load src mask
                src_mask = np.array(IMG.open(src_path).convert("L")) > 0

                if os.path.exists(dest_path):
                    # If exists, load dest mask and OR them
                    dest_mask = np.array(IMG.open(dest_path).convert("L")) > 0
                    merged_mask = np.logical_or(dest_mask, src_mask).astype(np.uint8) * 255
                else:
                    merged_mask = src_mask.astype(np.uint8) * 255

                # Save merged mask back to dest
                IMG.fromarray(merged_mask).save(dest_path)
            
            shutil.rmtree(src)


            # Update mappings (including transitive)
            new_labels[l_2] = l_1
            # propagate transitive merges
            for k, v in list(new_labels.items()):
                if v == l_2:
                    new_labels[k] = l_1
        
