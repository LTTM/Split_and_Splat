import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from PIL import Image

def show_mask(mask, ax, id, color, borders = True):
    """Overlay a binary mask on a matplotlib axis with optional border contours."""
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.set_title(f"ID: {id}")
    ax.imshow(mask_image)

def save_mask(mask, image, color, label, id):
    """Save a SAM2 binary mask as PNG and .npy, and the masked original image, under output/SAM_2/figurines/<id>/."""
   
    png_mask = mask.astype(np.uint8) * 255
    alpha_mask = Image.fromarray(png_mask)

    h, w = mask.shape[-2:]
   
    mask_png = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_png = (mask_png * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
    mask_png = Image.fromarray(mask_png).convert("RGBA")
    mask_png.putalpha(alpha_mask)

   
    image.putalpha(alpha_mask)

    
    os.makedirs(os.path.join("./output/SAM_2/figurines/", str(id), "images"), exist_ok=True)
    os.makedirs(os.path.join("./output/SAM_2/figurines/", str(id), "mask"), exist_ok=True)
    os.makedirs(os.path.join("./output/SAM_2/figurines/", str(id), "mask_png"), exist_ok=True)

    mask_label = label.replace(".JPEG", ".png")
    path = os.path.join("./output/SAM_2/figurines/", str(id),"mask_png", mask_label)
    mask_png.save(path)
    
    mask_label_pkl = label.replace(".JPEG", "")
    path = os.path.join("./output/SAM_2/figurines/", str(id),"mask", mask_label_pkl)
    np.save(path, mask)


    path = os.path.join("./output/SAM_2/figurines/", str(id),"images", mask_label)
    image.save(path)