import cv2
import numpy as np
import os
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation evaluation")

    parser.add_argument(
        "-scene",
        type=str, 
        default=None,
        help="scene where to invert masks"
    )
    
    parser.add_argument(
        "-instance",
        type=str, 
        default=None,
        help="instanec where to convert masks"
    )
    
    args = parser.parse_args()
    
    SCENE = args.scene
    
    INSTANCE= args.instance
    
    MASK_PATH = os.path.join("./data", SCENE , INSTANCE, "masks")

    images = os.listdir(MASK_PATH)
    def invert_mask(input_path, output_path):
        # Load the image in grayscale
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {input_path}")
        
        # Invert the mask: black <-> white
        inverted = 255 - img
        # Save the inverted image
        cv2.imwrite(output_path, inverted)

    for i in images:
        image_path = os.path.join(MASK_PATH, i)
        invert_mask(image_path, image_path)

