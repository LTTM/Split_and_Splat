import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy.ndimage import binary_dilation

from tqdm import tqdm

import cv2
import shutil

from PLY_utils import combine_ply,combine_sparse_ply

from composition import compute_collision_matrices, print_highest_pairs_until_zero

import argparse


def compose_mask(path_a, path_b, images_path, scene, save_path):
    """
    Alpha-composite per-image masks from two instance directories and save the merged masks and images.
    The smaller mask is placed on top. Falls back to whichever mask is available if only one exists.
    """

    
    images_name = os.listdir(images_path)

    for img in tqdm(images_name, desc="Composing masks"):
       
        if not img.lower().endswith(".png"):
            img = os.path.splitext(img)[0] + ".png"

        image_a_path = os.path.join(path_a, img)

        mask_a_path = image_a_path.replace("images", "masks")
        
      
        try:   
            mask_a_image = Image.open(mask_a_path)
        except:
            mask_a_image = None
    

        # #Image B
        image_b_path = os.path.join(path_b, img)

        mask_b_path = image_b_path.replace("images", "masks")

        try:
            mask_b_image = Image.open(mask_b_path)
        except:
            #print(mask_b_path)
            mask_b_image = None

        if(mask_b_image is not None and mask_a_image is not None):

            # Convert to numpy arrays to compare non-transparent pixel counts (alpha > 0)
            mask_a_alpha = np.array(mask_a_image.split()[-1])
            mask_b_alpha = np.array(mask_b_image.split()[-1])

            # Count number of visible (non-zero alpha) pixels
            area_a = np.count_nonzero(mask_a_alpha)
            area_b = np.count_nonzero(mask_b_alpha)     

            mask = Image.new("RGBA", mask_b_image.size, (0, 0, 0, 0))

            if area_a < area_b:
                mask = Image.alpha_composite(mask, mask_b_image)  # larger first
                mask = Image.alpha_composite(mask, mask_a_image)  # smaller on top
            else:
                mask = Image.alpha_composite(mask, mask_a_image)
                mask = Image.alpha_composite(mask, mask_b_image)

            os.makedirs(os.path.join(save_path, "masks"), exist_ok=True)
            mask_path = os.path.join(save_path, "masks", img)
            mask.save(mask_path)
        
            #Image
            mask_alpha = mask.split()[-1]

            
            Image_path = os.path.join("./data/"+scene+"/images", img)
            try:
                image = Image.open(Image_path).convert("RGBA")
            except:
                continue
            image.putalpha(mask_alpha)

            # image = Image.alpha_composite(image_a, image_b)

            os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
            images_path = os.path.join(save_path, "images", img)
            image.save(images_path)

        elif(mask_b_image is not None):
            os.makedirs(os.path.join(save_path, "masks"), exist_ok=True)
            mask_path = os.path.join(save_path, "masks", img)
            mask_b_image.save(mask_path)

        elif(mask_a_image is not None):
            os.makedirs(os.path.join(save_path, "masks"), exist_ok=True)
            mask_path = os.path.join(save_path, "masks", img)
            mask_a_image.save(mask_path)
        else:
            continue





def combine(img_1, img_2, dataset_path, scene, is_blender):
    """
    Combine two images, their masks and their PLY from a dataset.
    
    Parameters:
    - img_1: Name of the first image directory.
    - img_2: Name of the second image directory.
    - dataset_path: Path to the dataset containing the images and masks.
    """

    mask_1 = os.path.join(dataset_path, img_1, "images")
    mask_2 = os.path.join(dataset_path, img_2, "images")

    images_path = os.path.join(dataset_path, "images")

    save_path = os.path.join(dataset_path, "combined",  img_1+"_"+img_2)
    os.makedirs(save_path, exist_ok=True)



    if(is_blender):
        src_transforms = os.path.join(dataset_path, "transforms_train.json")
        dst_transforms = os.path.join(save_path, "transforms_train.json")

        if os.path.exists(src_transforms):
            shutil.copy(src_transforms, dst_transforms)
        else:
            print(f"Warning: transforms_train.json not found at {src_transforms}")
    else: 
        src_transforms = os.path.join(dataset_path, "sparse")
        dst_transforms = os.path.join(save_path, "sparse")

        if os.path.exists(src_transforms):
            shutil.copytree(src_transforms, dst_transforms, dirs_exist_ok=True)
        else:
            print(f"Warning: transforms_train.json not found at {src_transforms}")     


    compose_mask(mask_1, mask_2, images_path, scene, save_path)

    ply_path = "./output/"+scene+"/tmp"
    ply_path_a = os.path.join(ply_path, img_1+".ply")
    ply_path_b = os.path.join(ply_path, img_2+".ply")

    combine_ply(ply_path_a, ply_path_b, save_path, False, is_blender)

      
    src_images = os.path.join(dataset_path, "images")
    dst_images = os.path.join(save_path, "images")

    if os.path.exists(src_images):
        shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
    else:
        print(f"Warning: images folder not found at {src_images}")

    print("Combination succeded!")
    # --- Move PLY files into old/ subfolder ---
    old_folder = os.path.join(ply_path, "old")
    os.makedirs(old_folder, exist_ok=True)

    for ply_file in [ply_path_a, ply_path_b]:
        if os.path.exists(ply_file):
            shutil.move(ply_file, old_folder)
        else:
            print(f"Warning: {ply_file} not found, cannot move.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a ScanNet scene")
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name, e.g. scene0200_00"
    )
    args = parser.parse_args()

    SCENE = args.scene

    PLY_PATH ="./output/"+SCENE+"/tmp"

    DATASET_PATH = "./data/"+SCENE+"/masks"
    cm = compute_collision_matrices(PLY_PATH)

    pairs = print_highest_pairs_until_zero(cm)

    for img_1, img_2, _ in pairs:
 
        print(f"combine: {img_1} - {img_2}")
        combine(img_1, img_2, DATASET_PATH, SCENE, True)


