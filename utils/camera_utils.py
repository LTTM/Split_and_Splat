################################################################################
# Part of the code adapted from: https://github.com/graphdeco-inria/gaussian-splatting
# Copyright (c) 2023.
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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image 
import cv2
import os
import matplotlib.pyplot as plt

from scipy.ndimage import binary_erosion, binary_dilation


WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset, mask_color = None, black_filter =  True):
    
    if (args.is_instance):
        
        image = Image.open(cam_info.image_path).convert("RGBA")

        mask_path =  cam_info.image_path.replace("images", "masks")
        mask_path  = mask_path.replace(".jpg", ".png")
        mask_path  = mask_path.replace(".JPEG", ".png")
     
        mask_image = Image.open(mask_path)

    
           
        # Extract the alpha channel from the mask
        mask_alpha = mask_image.split()[-1].convert('L')

        if mask_color is not None:
            if(args.init_rec == True):
                try:
                    alpha_np = np.array(mask_alpha)
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    alpha_dilated = cv2.dilate(alpha_np, kernel, iterations=1)
                    mask_alpha = Image.fromarray(alpha_dilated)
                except Exception: # If dilation fails for any reason, fall back to the original alpha
                    pass

            image.putalpha(mask_alpha)
            # Create a new image with the same size as mask_image, filled with the mask_color
            color_layer = Image.new("RGBA", mask_image.size, tuple(mask_color))
            
            # Composite the color layer with the mask_image using the alpha channel as a mask
            color_layer.putalpha(mask_alpha)

            mask_image = color_layer

            mask_image.save(mask_path)

          #If an image is completely black skip it
        else:
            image.putalpha(mask_alpha)
        if(is_alpha_mostly_zero(mask_image) and black_filter):
            print(f"Caught black img {cam_info.image_path}")
            raise ValueError(f"Mask image at {mask_path} has an alpha channel that is all zeros. Please check the mask image.")

       
    else:
        image = Image.open(cam_info.image_path)     
        mask_image = Image.new("1", (image.width, image.height), 1)


    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
      
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        

    
    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, image_mask = mask_image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset, mask_color = None):
    camera_list = []
    black_camerea_list = []

    for id, c in enumerate(cam_infos):
        try:
            camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset, mask_color, black_filter = True))
        except FileNotFoundError:
            continue
        except ValueError:
            black_camerea_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset, mask_color, black_filter = False))
    
    return camera_list, black_camerea_list



def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

# Check if the alpha channel of an image is all zeros


def is_alpha_mostly_zero(image: Image.Image, threshold: float = 0.999) -> bool:
    """
    Checks if more than a given threshold of the alpha channel is zero.

    Args:
        image (PIL.Image.Image): The RGBA image to check.
        threshold (float): Fraction of alpha pixels that must be zero to return True (default 0.99).

    Returns:
        bool: True if more than threshold fraction of alpha values are zero, False otherwise.
    """
    if image.mode != "RGBA":
        raise ValueError("Image must be in RGBA mode to check alpha channel.")

    alpha = image.getchannel("A")
    alpha_array = np.array(alpha)

    zero_fraction = np.mean(alpha_array == 0)

    return zero_fraction >= threshold