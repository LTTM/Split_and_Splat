################################################################################
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

import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import argparse


# -------- CONFIGURATION --------


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------


def compute_clip_descriptors(main_dir):
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    descriptors = {}  


    subfolders = [
        os.path.join(main_dir, o)
        for o in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, o))
    ]

    for sub in tqdm(subfolders, desc="Processing objects"):
        # print(sub)
        rendered_dir = os.path.join(sub, "")
        if not os.path.isdir(rendered_dir):
            print(f"⚠️ Skipping {sub} (no 'rendered' folder)")
            continue

        img_files = [
            os.path.join(rendered_dir, f)
            for f in os.listdir(rendered_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg")) 
        ]

        if not img_files:
            print(f"⚠️ No images found in {rendered_dir}")
            continue

        embeddings = []

        for img_path in img_files:
           
            try:
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    embedding = model.encode_image(image)
                    embedding /= embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.cpu())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
   

        if embeddings:
            embeddings = torch.cat(embeddings, dim=0)
            mean_embedding = embeddings.mean(dim=0)
            descriptors[os.path.basename(sub)] = mean_embedding

    return descriptors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation evaluation")

    parser.add_argument(
        "-dataset",
        type=str, 
        default=None,
        help="Dataset where compute descriptors"
    )
    
    parser.add_argument(
        "-mode",
        type=str, 
        default="blurred_bg",
        help="crop setting to compute the descriptors (black_bg, white_bg, blurred_bg)"
    )
    args = parser.parse_args()
     
    DATASET = args.dataset
    MODE = args.mode
    
    # Edit path if needed
    MAIN_DIR = os.path.join("./testing_view",DATASET,MODE)  
    OUTPUT_FILE = os.path.join("./data", DATASET, "clip_descriptors.pt")  
    
    descriptors = compute_clip_descriptors(MAIN_DIR)
    torch.save(descriptors, OUTPUT_FILE)
    print(f"Saved descriptors for {len(descriptors)} instances from {DATASET} in {MODE} mode")