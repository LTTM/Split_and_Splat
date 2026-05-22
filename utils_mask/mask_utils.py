
from PIL import Image
from scipy.ndimage import binary_dilation
import numpy as np
import os

import cv2



def get_background(masks_path,  dataset_path):
    """Givene a mask folder (i.e. ./data/figurine_mask) and a dataset path (i.e. ./data/figurine), extract the background and save it in the dataset path under bg_mask_png."""

    images_path = os.path.join(dataset_path, "images/")

    images = os.listdir(images_path)
    masks = os.listdir(masks_path)

    bg_mask_path = os.path.join(dataset_path,"bg_mask_png")
    os.makedirs(bg_mask_path, exist_ok=True)

    for i in images:
        image_path = os.path.join(images_path, i)
        image = Image.open(image_path).convert("RGBA")
        w, h = image.size
        mask = np.zeros((h, w), dtype=bool)
        for j in masks:
            mask_label = i.replace(".JPEG", ".npy")
            mask_path = os.path.join(masks_path, j, "mask", mask_label)
            boolean_mask = np.squeeze(np.load(mask_path))
            mask = mask | boolean_mask
        
        # Convert the boolean mask to uint8 (0 or 255)
        png_mask = mask.astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
        dilated_mask = cv2.dilate(png_mask, kernel, iterations=1)


        # Create an image from the numpy array
        mask_image = Image.fromarray(mask)
        
        image.putalpha(mask_image)

        # Ensure black background where alpha is 0
        background = Image.new("RGB", image.size, (0, 0, 0))
        image = Image.composite(image, background, mask_image)

        # Convert RGBA to RGB before saving
        image = image.convert("RGB")

        # Save the mask image
        image.save(os.path.join(bg_mask_path, i))

def dilate_mask(img_id, source_path, instance_path):
    """
    Dilate the mask images and save them in the specified directories.
    
    Parameters:
    img_id (str): Identifier for the image folder.
    source_path (str): Path to the dataset source images.
    mask_path (str): Path to the masks.

    i.e. 
        img_id = "final"
        source_path = "./data/figurines/images"
        mask_path = "./data/figurines_mask"
    """
    img_path = os.path.join(instance_path,img_id,"images")
    mask_path = os.path.join(instance_path, img_id, "mask_png")
    mask_png_path = os.path.join(instance_path, img_id,"mask_dilated_png")
    image_dilate_path = os.path.join(instance_path, img_id, "image_dilated")

    os.makedirs(os.path.join(mask_png_path), exist_ok=True)
    os.makedirs(os.path.join(image_dilate_path), exist_ok=True)
    images = os.listdir(mask_path)
    for i in images:
        if not i.endswith(".png"):
            continue  # Skip non-PNG files
        img_name = i.replace("png", "JPEG")
        original_image = os.path.join(source_path, img_name)
        
        image_mask = os.path.join(mask_path, i)
        mask = np.array(Image.open(image_mask))  # Convert image to array

        kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        dilated_mask_image = Image.fromarray(dilated_mask)  # Convert array back to image
        dilated_mask_image.save(os.path.join(mask_png_path, i))

        #Image
        mask_alpha = dilated_mask_image.split()[-1]
        
        
        try:
            image = Image.open(original_image).convert("RGBA")
        except:
            continue
        image.putalpha(mask_alpha)

        # image = Image.alpha_composite(image_a, image_b)

    
        images_path = os.path.join(image_dilate_path, i)
        image.save(images_path)


def maskTopng(img_path, mask_path, mask_png_path):
    """
    Convert boolean masks to PNG images and overlay them on the original images.
    
    Parameters:
    img_path (str): Path to the folder containing original images.
    mask_path (str): Path to the folder containing boolean masks in .npy format.
    mask_png_path (str): Path to save the converted PNG masks.
    """
    
    os.makedirs(mask_png_path, exist_ok=True)
    
     # List all files in the mask directory
     #
    images = os.listdir(mask_path)
    for i in images:
        mask_label = i.replace(".png", ".npy")
        image_label = i.replace(".npy", ".png")
        image_path = os.path.join(img_path, image_label)
        boolean_mask = np.squeeze(np.load(os.path.join(mask_path, mask_label)))
        image = Image.open(image_path).convert("RGBA")

        # Convert the boolean mask to uint8 (0 or 255)
        png_mask = boolean_mask.astype(np.uint8) * 255

        # Create an image from the numpy array
        mask_image = Image.fromarray(png_mask)
        
        image.putalpha(mask_image)


        mask_label_png = image.replace(".npy", ".png.png")
        save_path = os.path.join(mask_png_path, mask_label_png)

        # Save as PNG
        mask_image.save(save_path)

    


def image_from_mask(mask_path, image_path):
    """Apply each RGBA mask's alpha channel to the corresponding image, compositing onto a black background."""
    masks_list = os.listdir(mask_path)
    masked_image_path = mask_path.replace("/mask_png", "/images")

    for m in masks_list:
        try:
            mask_image = Image.open(os.path.join(mask_path, m)).convert("RGBA")
            mask_alpha = mask_image.getchannel("A")   # Extract alpha channel
            #mask_image = mask_image.transpose(Image.TRANSPOSE) 
            image = Image.open(os.path.join(image_path, m)).convert("RGBA")
        except:
            continue
        
        # alpha = np.array(mask_image)[:, :, 3]
        # mask = alpha > 0

        #mask = binary_dilation(mask, structure=np.ones((3, 3)), iterations=3).astype(mask.dtype)

        black_bg = Image.new("RGB", image.size, (0, 0, 0))
        result = Image.composite(image, black_bg, mask_alpha)

        # Save result
      

        # Save / show
        result.save(os.path.join(masked_image_path, m))
        #result.show()


img_id = "34"
mask_path = "./data/garden_4_mask/"+img_id+"/mask_png"
image_path = "./data/garden_4/images_png"
image_from_mask(mask_path, image_path)