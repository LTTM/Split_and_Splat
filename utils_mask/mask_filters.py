from gaussian_renderer import render
from utils.loss_utils import l1_loss
import torch
import os 
from PIL import Image
import numpy as np
from utils.general_utils import PILtoTorch
import shutil
import matplotlib.pyplot as plt


def image_filter(gaussians, cameras, pipe, dataset):
    """Move images and masks with L1 render error above threshold to discard folders."""
    counter = 0
    bg_color = [0, 0, 0] 
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    alpha = 0.001
    for v in cameras:
        render_pkg = render(v, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=False)

        image = render_pkg["render"]

        gt_image = v.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        os.makedirs(os.path.join(dataset.source_path, "images_discarted"), exist_ok=True)
        os.makedirs(os.path.join(dataset.source_path, "mask_discarted_png"), exist_ok=True)

        if(Ll1.item() > alpha):
            try:
                print(f"Image ID: {v.image_name}, L1 Loss: {Ll1.item()}")
                mask_path = os.path.join(dataset.source_path, "masks", f"{v.image_name}.png")
                image_path = os.path.join(dataset.source_path, "images", f"{v.image_name}.JPEG")
                new_mask_path = os.path.join(dataset.source_path, "mask_discarted_png", f"{v.image_name}.png")
                new_image_path = os.path.join(dataset.source_path, "images_discarted", f"{v.image_name}.JPEG")
                shutil.move(image_path, new_image_path) 
                shutil.move(mask_path, new_mask_path) 

            
                counter += 1
            except:
                continue
    
    print("Number of images removed: ", counter)

def mask_filter(gaussians, cameras, pipe, dataset, id, obj_id):
    """Move per-object masks (and their images) whose render IoU is below 0.25 to discard folders."""
    counter = 0
    bg_color = [0, 0, 0] 
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    os.makedirs(os.path.join(dataset.source_path, "images_discarted"), exist_ok=True)
    os.makedirs(os.path.join(dataset.source_path, "mask_discarted_png"), exist_ok=True)
    os.makedirs(os.path.join(dataset.source_path, "mask_restored", id, "mask_discrted_png"), exist_ok=True)

    for v in cameras:
        
        new_mask_path = os.path.join(dataset.source_path, "mask_restored", id, "mask", v.image_name.replace(".jpg", ".png"))
     
        if os.path.exists(new_mask_path):
            new_mask = Image.open(new_mask_path).convert("RGBA")
            
        else:
            continue
    
        render_pkg = render(v, gaussians, pipe, bg, use_trained_exp = dataset.train_test_exp, separate_sh=False, id_filter=obj_id)
        mask =  render_pkg["mask"]

        mask_np = mask.detach().cpu().numpy()
        
        mask_np = np.transpose(mask_np, (1, 2, 0))
        mask_black = np.all(mask_np <= 0.1, axis=-1)
        binary_mask = (~mask_black).astype(np.uint8)

        
        resolution = (v.image_width, v.image_height)
        resized_image_rgb = PILtoTorch(new_mask, resolution)
       
        new_image = resized_image_rgb[:3, ...]
        new_image = new_image.clamp(0.0, 1.0).cuda()

        new_image_np = new_image.permute(1, 2, 0).detach().cpu().numpy()
        new_image_black = np.all(new_image_np == 0, axis=-1)
        binary_new_image = (~new_image_black).astype(np.uint8)

        
        

        # Compute IoU between binary_gt_image and binary_mask
        intersection = np.logical_and(binary_mask, binary_new_image).sum()
        union = np.logical_or(binary_mask, binary_new_image).sum()
        iou = intersection / union if union > 0 else 0.0
       
        

        #mask = mask * torch.tensor(GT_alpha, dtype=torch.float32, device="cuda")/255  # Multiply mask by alpha channel

        Ll1 = l1_loss(mask, new_image)
        
        if(iou < 0.25):
            print(f"Image ID: {v.image_name}, L1 Loss: {Ll1.item()}")
            print(f"IoU (ID: {id}, Image: {v.image_name}): {iou:.4f}")

    
             # Plot the mask for visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title(f"Generated Mask - ID: {id}, Image: {v.image_name}")
            plt.imshow(binary_mask, cmap='gray')

            plt.subplot(1, 2, 2)
            plt.title(f"Ground Truth Mask - ID: {id}, Image: {v.image_name}")
            plt.imshow((binary_new_image), cmap='gray')

            plt.show()

            # Plot the mask for visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title(f"Generated Mask - ID: {id}, Image: {v.image_name}")
            plt.imshow(mask.permute(1, 2, 0).cpu().numpy())

            plt.subplot(1, 2, 2)
            plt.title(f"Ground Truth Mask - ID: {id}, Image: {v.image_name}")
            plt.imshow(new_image.permute(1, 2, 0).cpu().numpy())

            plt.show()


   
           
            mask_path = os.path.join(dataset.source_path, "mask", v.image_name)
            image_path = os.path.join(dataset.source_path, "images", v.image_name)
            new_masks_path = os.path.join(dataset.source_path, "mask_discarted_png", v.image_name)
            new_image_path = os.path.join(dataset.source_path, "images_discarted", v.image_name)
            if  not (os.path.exists(mask_path) and os.path.exists(image_path)):
                print(f"Image ID: {v.image_name} does not exist")
                continue
                
            shutil.move(image_path, new_image_path) 
            shutil.move(mask_path, new_masks_path) 

            new_obj_mask_path =  os.path.join(dataset.source_path, "mask_restored", id, "mask_discrted_png", v.image_name)
            shutil.move(new_mask_path, new_obj_mask_path) 

            counter += 1
    
    print("Number of masks removed: ", counter)