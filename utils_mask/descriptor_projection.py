import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.cameras import CustomCam
from scene import GaussianModel

import torch

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph

from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

from plyfile import PlyData

import matplotlib.pyplot as plt

from utils_mask.PLY_utils import compute_centroid, get_bounding_box
from utils_mask.Get_cameras import compute_spherical_cam_params


import torchvision
from tqdm import tqdm
from gaussian_renderer import render_simple

from PIL import Image

import numpy as np
import torch.nn.functional as F

from sklearn.cluster import KMeans

from utils.sh_utils import RGB2SH
import cv2 as cv



def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine distance between two 1D tensors of shape (384,)"""
    return 1 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def jaccard_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Jaccard distance between two 1D tensors of shape (384,). Assumes binary or thresholded vectors."""
    a_bin = (a > 0).float()
    b_bin = (b > 0).float()
    intersection = (a_bin * b_bin).sum().item()
    union = ((a_bin + b_bin) > 0).float().sum().item()
    if union == 0:
        return 0.0
    return 1 - intersection / union

id = "04"

# Set you path
features_source = "../Modular-GS/data/figurines_mask/"+id+"/features"
camera_sources = "../Modular-GS/data/figurines_mask/"+id+"/cameras"

cameras = os.listdir(camera_sources)

size = 560

gs_path = "./data/figurines/figurines_masks/PLY/"+id+".ply"
gs = GaussianModel(sh_degree=0)
gs.load_ply(gs_path)
gs = gs.filter_points()


i=0
n_elev = 6
n_azim = 12
radius = 2.0 

# Create a dict mapping tuple (x, y, z) to its index in gs.get_xyz
xyz_to_index = {tuple(gs.get_xyz[i].tolist()): i for i in range(gs.get_xyz.shape[0])}


for camera in tqdm(cameras, desc="Descriptors projection...", total=len(cameras)):
   
    c = torch.load(os.path.join(camera_sources, camera), weights_only= True)
    descriptors = torch.load(os.path.join(features_source, camera), weights_only= True)
    cam_param = c.to("cuda")  # Use the first camera parameters for testing
    fov = 45
    resolution = 1120
    background_color= torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
    fov_rad = fov / 360 * 2 * np.pi
    render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_param)
    render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"), dense_rep=True)  


    meaning_pixels = render["pixels"]

    desc_dict = {}

    # Generate a random tensor of shape (W*H, 3)
    W = H = resolution
    
    descriptors = descriptors.permute(2, 0, 1).unsqueeze(0)  # (1, 384, 40, 40)
    descriptors = F.interpolate(descriptors, size=(1120, 1120), mode='bilinear', align_corners=False)
    descriptors = descriptors.squeeze(0).permute(1, 2, 0)  # (1120, 1120, 384)
    

    for g in meaning_pixels:
        pixels = meaning_pixels[g]
        desc = torch.zeros(384, device="cuda")  # Initialize desc as a tensor of zeros with the same shape as random_pixels[p]
        k = 0
        for p in pixels:
        
            tmp_desc = descriptors[p[1], p[0]] 
            
            desc += tmp_desc
            k += 1
        if k > 0:
            desc = desc / k

        gs.set_desc(g, desc)
        
     
desc_tensor = gs.get_desc  # Assuming this returns a tensor of shape (N, 384)


if isinstance(desc_tensor, torch.Tensor):
    nan_mask = torch.isnan(desc_tensor)
    nan_rows = nan_mask.any(dim=1)
    num_nan_desc = nan_rows.sum().item()

    coords = gs.get_xyz.detach().cpu().numpy()  
    if num_nan_desc > 0:
        from sklearn.neighbors import NearestNeighbors
        valid_mask = ~nan_rows.cpu().numpy()
        valid_coords = coords[valid_mask]
        valid_desc = desc_tensor[valid_mask]
        nan_indices = nan_rows.nonzero(as_tuple=True)[0]
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(valid_coords)
        for idx in nan_indices:
            query_coord = coords[idx]
            distances, indices = nbrs.kneighbors([query_coord])
            neighbor_descs = valid_desc[indices[0]]
            mean_desc = neighbor_descs.mean(dim=0)
            gs.set_desc(int(idx.item()), mean_desc)
            # gs.set_desc(int(idx.item()), torch.full_like(valid_desc[0], 0))
            # gs._opacity[int(idx.item())] = -1

    # PCA and plot descriptors colored by first PC
    desc_np = desc_tensor.detach().cpu().numpy()  # (N, 384)

    
    ################## SPECTRAL CLUSTERING ##################

    A = kneighbors_graph(desc_np, n_neighbors=10, include_self=True)
    A = 0.5 * (A + A.T)
    L = laplacian(A, normed=True)


    eigvals, eigvecs = eigsh(L, k=20, which='SM')

    eigvals_sorted = np.sort(eigvals)

    gaps = np.diff(eigvals_sorted)
    estimated_k = np.argmax(gaps) + 1
    
    clusters = KMeans(n_clusters = estimated_k )
    cluster_labels = clusters.fit_predict(desc_np)

    ########################################################

    # 3D plot of clusters using the original coordinates
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=cluster_labels, cmap='tab10', s=8, alpha=0.7)
    ax.set_title('KMeans Clusters in 3D Space')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    plt.show()

    pca = PCA(n_components=3)
    desc_pca = pca.fit_transform(desc_np)
    # Normalize for coloring
    color_vals = (desc_pca[:, 0] - desc_pca[:, 0].min()) / (np.ptp(desc_pca[:, 0]) + 1e-8)
    norm_pca = desc_pca.copy() - desc_pca.min()
    norm_pca /= norm_pca.max()
    # Set color of each gaussian as norm_pca (N, 3)
    # norm_pca shape: (N, 3), _features_dc shape: (N, 3, 1)
    norm_pca_tensor = torch.tensor(norm_pca, dtype=gs._features_dc.dtype, device=gs._features_dc.device)  # (N, 3)
    norm_pca_tensor = norm_pca_tensor.unsqueeze(1)  # (N, 1, 3)
    with torch.no_grad():
        gs._features_dc.copy_(RGB2SH(norm_pca_tensor))
        gs._features_rest = torch.zeros_like(gs._features_rest)
    
    output_path = os.path.join("./data/figurines_mask",str(id), "test_descriptors")
    os.makedirs(output_path, exist_ok=True)
    desc_output_path = os.path.join("./data/figurines_mask",str(id), "dino_desc")
    os.makedirs(desc_output_path, exist_ok=True)
    # Render an image of gs and plot it
    
    # pca_desc = PCA(n_components=3).fit_transform(all_descrptors)
    # pca_img = pca_desc - pca_desc.min(axis=0)
    # pca_img /= (pca_img.max(axis=0) + 1e-8)

pca = IncrementalPCA(n_components=3)

cosine_similarity_oveall = []
jaccard_distance_overall = []
for camera in tqdm(cameras, desc="Cameras test rendering and descriptors PCA", total=len(cameras)):

    c = torch.load(os.path.join(camera_sources, camera), weights_only= True)
    descriptors = torch.load(os.path.join(features_source, camera), weights_only= True)
    
    cam_param = c.to("cuda")  # Use the first camera parameters for testing
    fov = 45
    resolution = 1120
    background_color= torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
    fov_rad = fov / 360 * 2 * np.pi
    render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_param)
    render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"), dense_rep=False)  
    img = render["render"]
    cam_id = camera.replace("pt","")
    torchvision.utils.save_image(img, os.path.join(output_path, cam_id + "png"))
      

    pixels = render["pixels"]
       
    descriptors = descriptors.permute(2, 0, 1).unsqueeze(0)  # (1, 384, 40, 40)
    descriptors = F.interpolate(descriptors, size=(1120, 1120), mode='bilinear', align_corners=False)
    descriptors = descriptors.squeeze(0).permute(1, 2, 0)  # (1120, 1120, 384)

    # Compute and plot PCA only over those values in pixel (a map id:[pixel.x, pixel.y]) and cosine similarity
    pixel_indices = []
    all_descs = torch.zeros((resolution, resolution, 384), device=descriptors.device)
    cosine_sim = []
    jaccard_dists = []

    for g in pixels:
      
        px = pixels[g]
        pixel_indices =  pixel_indices + px  
        for p in px:
            d1 = descriptors[p[1], p[0]]
            d2 = desc_tensor[g]
            cos_sim = cosine_distance(d1,d2)
            jaccard_dist = jaccard_distance(d1,d2)
            cosine_sim.append(cos_sim)
            jaccard_dists.append(jaccard_dist)

        
    cosine_sim = np.array(cosine_sim)
    jaccard_dists = np.array(jaccard_dists)
    if len(pixel_indices) == 0:
        continue
    cosine_similarity_oveall.append(cosine_sim.mean())
    jaccard_distance_overall.append(jaccard_dists.mean())
    descs = torch.stack([descriptors[p[1], p[0]] for p in pixel_indices], dim=0).cpu().numpy()  # (num_pixels, 384)
    pca.partial_fit(descs.reshape(-1, descs.shape[-1]))
       

mean = pca.mean_
components = pca.components_

cosine_similarity_oveall = np.array(cosine_similarity_oveall)
cosine_sim_mean = cosine_similarity_oveall.mean()
print(f"Overall mean cosine distance: {cosine_sim_mean}")

jaccard_distance_overall = np.array(jaccard_distance_overall)
jaccard_dist_mean = jaccard_distance_overall.mean()
print(f"Overall mean jaccard distance: {jaccard_dist_mean}")

images = []

for camera in tqdm(cameras, desc="Generate descriptors render and images...", total=len(cameras)):

    c = torch.load(os.path.join(camera_sources, camera), weights_only= True)
    descriptors = torch.load(os.path.join(features_source, camera), weights_only= True)
    
    cam_param = c.to("cuda")  # Use the first camera parameters for testing
    fov = 45
    resolution = 1120
    background_color= torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
    fov_rad = fov / 360 * 2 * np.pi
    render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_param)
    render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"), dense_rep=False)  
    img = render["render"]
    cam_id = camera.replace("pt","")
    torchvision.utils.save_image(img, os.path.join(output_path, cam_id + "png"))
      

    pixels = render["pixels"]
       
    #descriptors = descriptors.view(40, 40, 384)
    descriptors = descriptors.permute(2, 0, 1).unsqueeze(0)  # (1, 384, 40, 40)
    descriptors = F.interpolate(descriptors, size=(1120, 1120), mode='bilinear', align_corners=False)
    descriptors = descriptors.squeeze(0).permute(1, 2, 0)  # (1120, 1120, 384)

   
    # Compute and plot PCA only over those values in pixel (a map id:[pixel.x, pixel.y])
    pixel_indices = []
    for g in pixels:
        pixel_indices =  pixel_indices + pixels[g]  # list of (x, y) tuples
    if len(pixel_indices) == 0:
        continue
    
   
    descs = torch.stack([descriptors[p[1], p[0]] for p in pixel_indices], dim=0)  # (num_pixels, 384)
    descs = descs.cpu()  # Ensure tensor is on CPU before numpy conversion
    descs -= torch.from_numpy(mean).to(descs)
    im = descs @ torch.from_numpy(components.T).to(descs)
    blank_img = torch.zeros((resolution, resolution, 3), dtype=im.dtype)
    for idx, (x, y) in enumerate(pixel_indices):
        blank_img[y, x] = im[idx]
    images.append(blank_img)

images = torch.stack(images)
images -= torch.min(images)
images /= torch.max(images)

for i, c in enumerate(tqdm(cameras, 'Saving RGB embeddings...')):
    im = images[i].cpu().numpy()
    im = np.clip(np.round(255*im), 0, 255).astype(np.uint8)

    cv.imwrite(os.path.join(desc_output_path, c.replace(".pt",".png")), im)

if __name__ == "__main__":
    pca = IncrementalPCA(n_components=3)
    embeddings = sorted([f for f in os.listdir("raw_embeddings") if f.endswith(".npy") and not f.startswith(".")])

    for emb_path in tqdm(embeddings):
        emb = np.load("raw_embeddings/"+emb_path)
        pca.partial_fit(emb.reshape(-1, emb.shape[-1])) # flatten

    print(100*pca.explained_variance_ratio_)
    np.savez("pca.npz",
             mean=pca.mean_,
             components=pca.components_)
