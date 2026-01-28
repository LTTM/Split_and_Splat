#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import numpy as np
import random

from utils.sh_utils import SH2RGB
#########################################################################################

def pixels_to_xy_dict(pixels, means3D):
    if hasattr(pixels, 'detach'):
        pixels_np = pixels.detach().cpu().numpy()
    else:
        pixels_np = pixels
    if hasattr(means3D, 'detach'):
        means3D_np = means3D.detach().cpu().numpy()
    else:
        means3D_np = means3D
    xy_dict = {}
    for row in pixels_np:
        x, y, idx = row
        idx = int(idx)
        if idx == -1:
            continue
        xy_dict[(int(x), int(y))] = means3D_np[idx]
    return xy_dict

def pixels_to_gs_dict(pixels):
    if hasattr(pixels, 'detach'):
        pixels_np = pixels.detach().cpu().numpy()
    else:
        pixels_np = pixels
    gs_dict = {}
    for row in pixels_np:
        x, y, idx = row
        idx = int(idx)
        if idx == -1:
            continue
        if idx not in gs_dict:
            gs_dict[idx] = [(int(x), int(y))]
        else:
            gs_dict[idx].append((int(x), int(y)))
    return gs_dict

def compute_depth_map_from_pixels(pc, viewpoint_camera, pixels_dict):
    """
    pc: GaussianModel (has get_xyz())
    viewpoint_camera: camera object with world_view_transform and image size
    pixels_dict: (N, 3) tensor or array with [x, y, gaussian_idx]
    """


    # 1. Get 3D positions of Gaussians
    means3D = pc.get_xyz  # (N, 3)
    
    # 2. Convert to homogeneous coordinates
    ones = torch.ones_like(means3D[:, :1])
    means3D_h = torch.cat([means3D, ones], dim=1)  # (N, 4)
    
    # 3. Transform to camera space
    view_matrix = viewpoint_camera.world_view_transform  # (4, 4)
    camera_coords = (view_matrix @ means3D_h.T).T  # (N, 4)
    gaussian_depths = camera_coords[:, 2]  # Z component => depth
    
    
    # 5. Initialize depth map
    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width
    depth_map = torch.ones((H, W), dtype=torch.float32, device=gaussian_depths.device) * float('inf')
    
    # 6. Fill in depth map
    for gs_idx, pixel_coords in pixels_dict.items():
        depth = gaussian_depths[gs_idx].item()
        for x, y in pixel_coords:
            # Ensure pixel coords are within bounds
            if 0 <= y < H and 0 <= x < W:
                if depth < depth_map[y, x]:
                    depth_map[y, x] = depth
    
    # 7. Optional: replace inf with 0 (or any value)
    depth_map[depth_map == float('inf')] = 0.0
    
    return depth_map



def compute_image_from_pixels(pc, viewpoint_camera, pixels_dict):
    """
    pc: GaussianModel (has get_xyz())
    viewpoint_camera: camera object with world_view_transform and image size
    pixels_dict: (N, 3) tensor or array with [x, y, gaussian_idx]
    """


    # 1. Get 3D positions of Gaussians
    means3D = pc.get_xyz  # (N, 3)
    print(means3D.shape)
    
    device = means3D.device

    
    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width
    image = torch.ones((H, W, 3), dtype=torch.float32, device=device) * float('inf')
    

    for gs_idx, pixel_coords in pixels_dict.items():
            # Generate a random color (RGB)
        color_rgb = torch.tensor([
            random.random(),  # Red channel [0, 1]
            random.random(),  # Green channel [0, 1]
            random.random()   # Blue channel [0, 1]
        ], dtype=torch.float32, device=device)
        
        for x, y in pixel_coords:
            # Ensure pixel coords are within bounds
            if 0 <= y < H and 0 <= x < W:
                
                image[y, x] = color_rgb

               
    plt.figure(figsize=(8, 6))
    plt.imshow(image.detach().cpu().numpy())
    plt.axis("off")
    plt.savefig("color_image_"+viewpoint_camera.image_name)
    plt.close() 

#########################################################################################

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, id_filter = None, mask_only = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if(mask_only):
        pc = pc.filter_points()
    if(id_filter is not None):
        pc = pc.filter_by_id(id_filter)
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

 
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        #return_accumulation = True,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    mask_opacity = torch.zeros_like(pc.get_opacity)+1


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color
    

    mask_colors_precomp = None
    try:
        if pipe.convert_SHs_python:
            shs_view = pc.get_id_color.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_id_color.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            mask_colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                mask_dc, mask_shs = pc.get_id, pc.get_features_rest
            else:
                mask_shs = pc.get_id_color
    except:
        mask_colors_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
    
        mask_image, _, _, pixels_dict = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = mask_dc,
            shs = mask_shs,
            colors_precomp = mask_colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        rendered_image, radii, depth_image, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    
    else:
        mask_image, _, _, pixels_dict = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = mask_shs,
            colors_precomp = mask_colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        rendered_image, radii, depth_image, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    
    
    
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
        mask_image = torch.matmul(mask_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    mask_image = mask_image.clamp(0, 1)
    
    out = {
        "render": rendered_image,
        "mask": mask_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,
        "pixels" : pixels_dict   #HxW, 3     
        }
    
    return out



def render_simple(viewpoint_camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, debug=False, id_filter=None, dense_rep=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    if(id_filter is not None):
        pc = pc.filter_by_id(id_filter, keep_occlusions= False)
      
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H= int(viewpoint_camera.image_height)
    W= int(viewpoint_camera.image_width)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=debug,
        antialiasing= None
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    convert_shs_python = False
    colors_precomp = None
    if override_color is None:
        if convert_shs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    if(dense_rep==True): opacity = torch.zeros_like(pc.get_opacity)+1
    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    #rendered_image, radii
    rendered_image, radii, rendered_depth, pixels = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    #pixels = pixels.view(H,W)
    
    pixels_dict = pixels_to_gs_dict(pixels)
    
    #depth_image =  compute_depth_map_from_pixels(pc, viewpoint_camera, pixels_dict)


    #plot_xy_dict_2d3d(pixels_dict, rendered_image)
    #plot_gs_dict_2d3d(pixels_dict, means3D, rendered_image)
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": rendered_depth,
        "pixels": pixels_dict
    }

   

def plot_xy_dict_2d3d(xy_dict, render=None):
    import matplotlib.pyplot as plt
    xy = np.array(list(xy_dict.keys()))
    vals = np.array(list(xy_dict.values()))
    fig = plt.figure(figsize=(18, 5) if render is not None else (12, 5))
    ax1 = fig.add_subplot(1, 3, 1) if render is not None else fig.add_subplot(1, 2, 1)
    ax1.scatter(xy[:, 0], xy[:, 1], s=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Pixel (x, y) keys')
    ax1.axis('equal')
    # Make ax1 the same size as ax3 (image aspect ratio)
    if render is not None:
        img = render.detach().cpu().numpy() if hasattr(render, 'detach') else render
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        h, w = img.shape[:2]
        aspect = w / h
        ax1.set_aspect(aspect)
    ax1.invert_yaxis()  # Flip y-axis for image-style display
    ax2 = fig.add_subplot(1, 3, 2, projection='3d') if render is not None else fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(vals[:, 0], vals[:, 1], vals[:, 2], s=2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('means3D values')
    if render is not None:
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(img)
        ax3.set_title('Rendered Image')
        ax3.axis('off')
    plt.tight_layout()
    plt.show()

def plot_gs_dict_2d3d(gs_dict, means3D, render=None):
    import matplotlib.pyplot as plt
    import numpy as np
    # gs_dict: {gs_idx: [(x, y), ...]}
    # For visualization, flatten all (x, y) for all gs_idx
    xy = np.array([xy for pixels in gs_dict.values() for xy in pixels])
    gs_indices = np.array([idx for idx, pixels in gs_dict.items() for _ in pixels])
    # Get 3D points for each gs_idx
    means3D_np = means3D.detach().cpu().numpy() if hasattr(means3D, 'detach') else means3D
    points3D = np.array([means3D_np[idx] for idx in gs_indices])
    fig = plt.figure(figsize=(18, 5) if render is not None else (12, 5))
    ax1 = fig.add_subplot(1, 3, 1) if render is not None else fig.add_subplot(1, 2, 1)
    ax1.scatter(xy[:, 0], xy[:, 1], s=2, c=gs_indices, cmap='tab20', alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Pixel (x, y) for each GS index')
    ax1.axis('equal')
    if render is not None:
        img = render.detach().cpu().numpy() if hasattr(render, 'detach') else render
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        h, w = img.shape[:2]
        aspect = w / h
        ax1.set_aspect(aspect)
    ax1.invert_yaxis()
    # 3D plot: show the actual 3D points
    ax2 = fig.add_subplot(1, 3, 2, projection='3d') if render is not None else fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], s=2, c=gs_indices, cmap='tab20', alpha=0.7)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D points for each GS index')
    if render is not None:
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(img)
        ax3.set_title('Rendered Image')
        ax3.axis('off')
    plt.tight_layout()
    plt.show()