import sys
sys.path.append('.')

import os
import torch
import open3d as o3d
import pycolmap
import numpy as np
import sam2

import cv2
from PIL import Image as IMG
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from scipy.spatial import ConvexHull

from arguments import ModelParams, PipelineParams, ArgumentParser, get_combined_args
from utils.general_utils import safe_state
from scene import Scene, GaussianModel

from gaussian_renderer import render
import matplotlib
matplotlib.use('Agg')  # Use headless backend (no display needed)
import matplotlib.pyplot as plt


sys.path.append("./point_projection")
import point_projection_cuda as ppc


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


SCENE = "scene0070_00"



##########################
ACCURACY_LABELS = 0.7

DOWNSAMPLE = 100000
EPSILON = 0.01

def random_downsample(points, target_num):
    """Randomly subsample target_num rows from a numpy point array without replacement."""
    idx = np.random.choice(points.shape[0], size=target_num, replace=False)
    return points[idx]


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
        try: 
            hull = ConvexHull(S)
        except:
            return None
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

def get2D_masked_pcd(image, camera, depth, pcd, depth_trashold = EPSILON):
    """
    - image: Extriniscs, PyColmapImage
    - camera: Intrinsics, PyColmapCamera
    - pcd: open3D pointcloud or list of 3D points 
    - depth: depth map, float numpy array np.array(H,W,1)
    - depth_trashold: float which define what is the acceptable error in the delta
    
    3D->2D
    Backproject a 3D pcd onto a 2D image and return just the surface points as open3D point cloud, their 2D coordinates and integer ids
    """
    points3D_filtered = []
    colors = []


    points = pcd
    
    # if hasattr(pcd, 'points'):
    #     points = np.asarray(pcd.points)
    # else: points = pcd

    H, W = camera.height, camera.width

    extrinsics = image.cam_from_world

    pcd_size = len(points)

    points2D = []
    z_buffer ={}
    epsilon = 0.5
    
    for id, p in enumerate(points):
        uv = camera.img_from_cam(image.cam_from_world * p)
        
        if uv is None: continue
        u, v = uv[0]
            
        i, j = int(round(u)), int(round(v))
        if i < 0 or i >= W or j < 0 or j >= H:
            continue

        p_e = np.append(p, 1.0)

        p_cam = extrinsics.matrix() @ p_e

        z = p_cam[2]

        # Compare to depth map
        if abs(depth[j,i] - z) <= depth_trashold and z >= 0:  # only keep if closer
           if((i,j) not in z_buffer or z < z_buffer[(i,j)]["z"]):
     
                z_buffer[(i,j)] = {
                    "z": z,
                    "point3D" : p[:3] ,
                    "id" : id
                }

    points2D = np.array(list(z_buffer.keys()))
    points3D_filtered = np.array([v["point3D"] for v in z_buffer.values()])
    ids =  np.array([v["id"] for v in z_buffer.values()])
    
    filtered_pcd_size = len(points3D_filtered)
    if(filtered_pcd_size==0 or pcd_size/filtered_pcd_size <= ACCURACY_LABELS):
        print("No point founded!")
        return None, None, None
    points3D_filtered = np.array(points3D_filtered)
    
    pcd_filter = o3d.geometry.PointCloud()
    pcd_filter.points = o3d.utility.Vector3dVector(points3D_filtered)
    #pcd_filter.colors = o3d.utility.Vector3dVector(np.array(colors))

    points2D = np.array(points2D)

    return points3D_filtered, points2D, ids

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

def get_render_mask(camera, dataset, pipe):
    """Render the scene from a given camera and return the RGB render tensor."""
    bg_color = [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_pkg = render(camera, gaussians, pipe, bg, use_trained_exp = dataset.train_test_exp, separate_sh=False)

    return render_pkg["render"]

def load_image(path):
    """Load a depth image from path and normalize by dividing by 1000 (mm → m)."""
    return np.array(IMG.open(path))/1000.0

def load_matrix_from_txt(path):
    """Load a 4×4 matrix from a whitespace-separated text file."""
    with open(path) as f:
        vals = [float(v) for v in f.read().split()]
    return np.array(vals).reshape(4, 4)

###########################
if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--instance_test", default="0", type=str, help="Instance ID to process (e.g., '81').")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    pipeline = pipeline.extract(args)
    model = model.extract(args)
    iteration = args.iteration

    # Initialize system state (RNG)
    safe_state(args.quiet)

    args.is_instance = False


    gaussians = GaussianModel(model.sh_degree)

    scene = Scene(model, gaussians, load_iteration=iteration, shuffle=False)

    id_color = scene.get_id()

    scene = scene.filter_gaussian()


    cameras = scene.getTrainCameras()

    INSTANCE_TEST = args.instance_test
    print(f"Processing Instance ID: {INSTANCE_TEST}")

    PLY_PATH = os.path.join("./output", SCENE, "PLY")

    COLMAP_PATH = os.path.join("./data", SCENE, "sparse/0")
    DEPTH_PATH = os.path.join("./data", SCENE, "depth")
    IMAGE_PATH = os.path.join("./data", SCENE, "images")
    INSTANCE_PATH = os.path.join("./data", SCENE, SCENE+"_masks", INSTANCE_TEST)


    #SAM model

    sam_dir = str(os.path.dirname(sam2.__file__))

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)


    # 1-Open the instance reconstruction and filter occlusion

    image_path = os.path.join("./data/scanNet", SCENE, "images")
    intrinsics_depth = os.path.join("./data/scanNet", SCENE, "intrinsic/intrinsic_depth.txt")
    #intrinsics_img= os.path.join("./data/scanNet", SCENE, "calibration/instrinsic/intrinsic_color.txt" )

    pose_path = os.path.join("./data/scanNet", SCENE, "pose")

    intr = load_matrix_from_txt(intrinsics_depth)
    intr_np = np.array([
            [intr[0,0], intr[1,1], intr[0,2], intr[1,2]]    # cx, cy
        ])


    depth_path = os.path.join("./data/scanNet", SCENE, "depth")
    mask_folder =  os.path.join("./data/scanNet", SCENE, "masks")

    # Load GS reconstruction
    ply_path = os.path.join("./output", SCENE,"raw", INSTANCE_TEST, "point_cloud/iteration_10000/point_cloud.ply")


    pcd = o3d.io.read_point_cloud(ply_path)

    # Nx3 points
    pcd_points = np.asarray(pcd.points)

    pcd_points = random_downsample(pcd_points, DOWNSAMPLE)



    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(pcd_points)


    o3d.visualization.draw_geometries([pcd])   

    labels_counter = 0

    pointsLabels = {}   # dictionary: point_id → [label]
    labelPoints = {}    # dictionary: label → {point_id}
    gt_labels_idx = 1

    projected_view = {}

    images = sorted(os.listdir(image_path))

    # 2-iterate over each view and check IoU with original mask
    
    iter = 0

    for cam in tqdm(cameras):
        # Open image, depth and masks
    
        if(iter>len(images)): break
        iter +=1

        img_name = getattr(cam, "image_name", None)
       
        # try:
        #     img_meta = images[cam_id]
        #     cam_meta = img_meta.camera
        #     image_id = img_meta.image_id
        # except:
        #     continue


        image_name = f"{img_name}.JPEG"
        extr_path  = image_name.replace(".jpg", ".txt")
        extr_path  = extr_path.replace(".JPEG", ".txt")
        extr = load_matrix_from_txt(os.path.join(pose_path, extr_path))

        # extr_inv = np.linalg.inv(extr)
        # extr_inv = extr_inv[:3, :]

        extr_inv = np.linalg.inv(extr)        # world → camera
        extr_inv = extr_inv[:3, :].astype(np.float32).reshape(12)  # 3x4 row-major

        extrinsics = torch.tensor(extr_inv, device="cuda", dtype=torch.float32).contiguous()
        intrinsics = torch.tensor(intr_np, device="cuda", dtype=torch.float32)

        
        # Load image
        #try:
            #img_name_2 = image_name.replace(".jpg", ".JPEG")
        img = IMG.open(os.path.join(image_path, image_name)).convert("RGB")
        W, H = img.size
        img = np.array(img)
        # except:
        #     continue

        # Load depth

        # Load depth
        dm_name = image_name.replace(".jpg", ".png")
        dm_name = dm_name.replace(".JPEG", ".png")
        #depth = np.load(os.path.join(depth_path, dm_name)).astype(np.float32)
        depth = load_image(os.path.join(depth_path, dm_name))

        # Load mask
        try:
            mask_name = image_name.replace(".jpg", ".png")
            mask_name = mask_name.replace(".JPEG", ".png")
            mask_img = IMG.open(os.path.join(INSTANCE_PATH,"masks", mask_name)).convert("RGBA")#.convert("L")
            # gray = np.array(mask_img)
            # mask = gray > 0
            # Extract the alpha channel (4th channel)
            alpha = np.array(mask_img)[:, :, 3]

            # Mask is True where alpha > 0 (visible areas)
            mask = alpha > 0
        except:
            mask = None
        
        

        #compute the mask s

        #Compute visible points from the view
        #visible_point3D, visible_point2D, visible_ids = get2D_masked_pcd(img_meta, cam_meta, depth, instance_pcd, depth_trashold=0.01)


        t_pcd_points = torch.from_numpy(pcd_points).float().cuda()                         # Point cloud CUDA
        t_depth = torch.from_numpy(depth).float().cuda()                                        # Depth map CUDA
        t_points_2D = torch.full((H,W), -1, device="cuda", dtype=torch.float32)                 # 2D-> 3D ids CUDA (init to -1)
        t_computed_depth = torch.full((H,W), float('inf'), device="cuda", dtype=torch.float32)  # z_buffer CUDA (init to inf)
    
        ppc.pcd2D(t_pcd_points, t_depth, extrinsics, intrinsics, EPSILON, t_points_2D, t_computed_depth)

    
        ys, xs = torch.where(t_points_2D != -1)
        visible_point2D = torch.stack([xs, ys], dim=1).detach().cpu().numpy()
        visible_ids = t_points_2D[ys, xs].long().detach().cpu().numpy()
        visible_point3D = pcd_points[visible_ids]



        if(visible_point2D is None or len(visible_point2D)==0): continue


        p = greedy_coreset_2D(visible_point2D, 5)
        if p is None: continue

        input_point = np.array(p)
        input_label = np.ones(len(p))    

        predictor.set_image(img)
        mask_orignal, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )




        plt.figure(figsize=(10, 8))

        # Draw the image first (background)
        plt.imshow(img)               # img is H×W×3 RGB

        # Then draw the points (foreground)
        plt.scatter(
            visible_point2D[:, 0],    # x
            visible_point2D[:, 1],    # y
            s=8,
            c='red',
            marker='o'
        )

        # plt.axis('off')               # optional
        # plt.tight_layout()
        # plt.savefig(image_name)
        # plt.close()

        new_mask = mask_to_array(mask_orignal)
        new_mask = (new_mask > 0).astype(np.uint8)


        # Visualization
        # Create figure and axis
        #test_mask(img, new_mask, input_point)
    

        # Generate render of the given view

        rendered_mask = get_render_mask(cam, model, pipeline)

        if rendered_mask.ndim == 3 and rendered_mask.shape[0] in [3, 4]:  # (C,H,W)
            gray_mask = rendered_mask.mean(dim=0)
        else:
            gray_mask = rendered_mask

        # Create binary mask: 0 for black, 1 otherwise
        rendered_mask = (gray_mask > 0.2).to(torch.uint8)
        rendered_mask = rendered_mask.detach().cpu().numpy()

        iou_score_new = compute_IoU(rendered_mask, new_mask)

        #print(f"IoU with new:{iou_score_new}")


        # Ensure all masks are in numpy uint8 format
        rendered_mask_np = rendered_mask.astype(np.uint8)
        new_mask_np = new_mask.astype(np.uint8)
        mask_np = mask.astype(np.uint8) if mask is not None else None

      
        if(mask is not None): 
            """ 
                If the mask is not None, compare IoU of the original mask with the new one and keep the best
            """
            iou_score_old = compute_IoU(rendered_mask, mask)
            #print(f"IoU with old:{iou_score_old}")

            if (iou_score_new < iou_score_old):
                continue

        if(mask is not None): 

            if (iou_score_new < iou_score_old):
                continue
        if(iou_score_new<0.05): continue

        alpha = (new_mask.astype(np.uint8)) * 255

        #small dilation
        try:
            kernel = np.ones((3, 3), dtype=np.uint8)
            alpha_dilated = cv2.dilate(alpha, kernel, iterations=1)
            alpha = alpha_dilated
        except Exception as e:
            # If dilation fails for any reason, fall back to the original alpha
            print(f"Warning: dilation failed ({e}), using original alpha.")
            pass
        
        color = id_color.detach().cpu().numpy().astype(np.uint8)

        rgba = np.zeros((new_mask.shape[0], new_mask.shape[1], 4), dtype=np.uint8)
        rgba[..., 0:3] = 255  
        rgba[..., 3] = alpha 

        save_name = image_name.replace(".jpg", ".png")
        save_name = save_name.replace(".JPEG", ".png")
        img = IMG.fromarray(rgba, mode="RGBA")
        os.makedirs(os.path.join(INSTANCE_PATH, "mask_extra"), exist_ok=True)
        mask_extra_path = os.path.join(INSTANCE_PATH, "mask_extra", save_name)
        img.save(mask_extra_path)

        
