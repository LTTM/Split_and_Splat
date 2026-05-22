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
import cv2
import json
import numpy as np

# Root directory containing all views
DATASET = "ramen"
root_dir = "testing_view/"+ DATASET +"/boolean_masks"
test_view = ["00006.JPEG", "00024.JPEG", "00060.JPEG","00065.JPEG", "00081.JPEG", "00119.JPEG", "00128.JPEG"]

for view_name in os.listdir(root_dir):
    if view_name not in test_view : continue
    view_path = os.path.join(root_dir, view_name)
    if not os.path.isdir(view_path):
        continue

    # Extract group number from view folder name (e.g., "view_5" → 5)
    try:
        group = int(''.join([c for c in view_name if c.isdigit()])) if any(c.isdigit() for c in view_name) else view_name
    except ValueError:
        group = view_name

    annotations = []

    for filename in sorted(os.listdir(view_path)):
        if not filename.endswith(".png"):
            continue

        mask_path = os.path.join(view_path, filename)

        # category = filename without extension
        category = os.path.splitext(filename)[0]
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        except:
            continue
          # Convert mask to boolean
        mask_bool = mask > 127

        # Get coordinates of True pixels
        ys, xs = np.where(mask_bool)
        segmentation = [[int(x), int(y)] for x, y in zip(xs, ys)]

        if len(segmentation) == 0:
            continue

        # Compute area and bounding box
        area = float(len(segmentation))
        x_min, y_min = int(xs.min()), int(ys.min())
        x_max, y_max = int(xs.max()), int(ys.max())
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        obj_data = {
            "category": category,
            "group": category,
            "segmentation": segmentation,
            "area": area,
            "layer": 1.0,
            "bbox": bbox,
            "iscrowd": 0,
            "note": ""
        }

    

        annotations.append(obj_data)

    # Save JSON file for this view
    output_path = os.path.join(root_dir, f"{view_name}.json")
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"✅ Saved {output_path}")




# Example JSON entry
json_dir = "./testing_view/"+ DATASET  # Replace with your JSON path



# Determine mask size from bbox (or set manually)
# Here we assume all objects fit within 1000x1000
height, width = 731, 988

for json_file in os.listdir(json_dir):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(json_dir, json_file)
    with open(json_path, "r") as f:
        data = json.load(f)

    # Initialize array with -1
    group_array = np.full((height, width), -1, dtype=np.int32)


    # Fill in pixels with group_id
    for obj in data:
        group_id = obj["group"]
        for x, y in obj["segmentation"]:
            y = min(y, height-1)
            x = min(x, width-1)
            group_array[y, x] = group_id

    # Flatten to 1D
    group_array_1d = group_array.flatten()

    # Save as .npy file
    out_path = os.path.join( "./testing_view", DATASET, os.path.splitext(json_file)[0] + ".npy")
    
    np.save(out_path, group_array_1d)
    

print("Done! 1D arrays saved for all JSON files.")
mask = np.zeros((height, width), dtype=np.uint8)

for json_file in os.listdir(json_dir):
   if not json_file.endswith(".json"):
       continue

   json_path = os.path.join(json_dir, json_file)
   with open(json_path, "r") as f:
       data = json.load(f)

    # Create a blank mask per object
   for i, obj in enumerate(data):
       mask = np.zeros((height, width), dtype=np.uint8)

        # Fill in the True pixels from segmentation
       for point in obj["segmentation"]:
            x, y = point
            y = min(y, height-1)
            x = min(x, width-1)
            mask[y, x] = 255  # white pixel

       # Save mask
       mask_name = f"{os.path.splitext(json_file)[0]}_obj{i+1}.png"
       cv2.imwrite(os.path.join(json_dir, mask_name), mask)

print("Masks reconstructed from JSON!")