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

from scene.cameras import Camera

import os
import json
import numpy as np

import argparse


def load_transform_matrix(file_path):
    """
    Load the transform matrix from a text file.
    """
    with open(file_path, 'r') as file:
        matrix = [list(map(float, line.strip().split())) for line in file]
    return matrix

def process_directory(directory_path):
    """
    Process each directory and create a JSON file with the transform matrices.
    """
    color_dir = os.path.join(directory_path, "images")           # TODO
    pose_dir = os.path.join(directory_path, "pose")             # TODO
    intrinsic_dir = os.path.join(directory_path, "intrinsic")   # TODO

    # Check if both directories exist
    if not os.path.isdir(color_dir) or not os.path.isdir(pose_dir):
        return

    transform_data = {
        'w': 640,
        'h': 480,
        'fl_x': 577.590698,
        'fl_y': 578.729797,
        'cx': 318.905426,
        'cy': 242.683609,
        'frames': [],
    }

    print(color_dir)
    img_names = [
        os.path.splitext(f)[0].zfill(4) + ".JPEG"
        for f in os.listdir(color_dir)
        if f.endswith(".JPEG")
    ]

    print(img_names)

    img_names.sort(key=lambda x: os.path.splitext(x)[0])  # Sort by image number
    # Iterate over the color images
    for img_name in img_names:
        
        if img_name.endswith(".JPEG"):
            # Construct the corresponding pose file path
            pose_file = os.path.splitext(img_name)[0] + ".txt"
            pose_file_path = os.path.join(pose_dir, pose_file)

            intrinsic_file = os.path.splitext(img_name)[0] + ".txt"
            intrinsic_file_path = os.path.join(intrinsic_dir, intrinsic_file)

            print(intrinsic_file_path)

            # Check if the pose file exists
            if os.path.isfile(pose_file_path):
                transform_matrix = load_transform_matrix(pose_file_path)
                
                # note: colmap --> blender
                transform_matrix = np.array(transform_matrix)
                transform_matrix[:3, 1:3] *= -1     
                transform_matrix = transform_matrix.tolist()

            
                frame_data = {
                    "file_path": os.path.join("images", os.path.splitext(img_name)[0]),
                    "transform_matrix": transform_matrix
                }

                if os.path.isfile(intrinsic_file_path):
                    intrinsic_info = load_transform_matrix(intrinsic_file_path)
                    frame_data.update({
                        'fl_x': intrinsic_info[0][0],
                        'fl_y': intrinsic_info[1][1],
                        'cx':  intrinsic_info[0][2],
                        'cy': intrinsic_info[1][2]
                    })
                else:
                    frame_data.update({
                        'fl_x': 577.590698,
                        'fl_y': 578.729797,
                        'cx': 318.905426,
                        'cy': 242.683609,
                    })

                transform_data["frames"].append(frame_data)

    return transform_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation evaluation")

    parser.add_argument(
        "-dataset_path",
        type=str, 
        default="./data/Scanet",
        help="Path of Scanet scenes"
    )
    
    args = parser.parse_args()

    # Directory containing the scenes
    base_directory = args.dataset_path    

    # Process each scene directory and create JSON files
    for scene_dir in os.listdir(base_directory):        
        scene_path = os.path.join(base_directory, scene_dir)
        if os.path.isdir(scene_path):
            # Process the directory and get the transform data
            transform_data = process_directory(scene_path)

            print(scene_path)
            print(len(transform_data["frames"]))
            # Create the JSON file
            if transform_data:
                json_file_path = os.path.join(scene_path, "transforms_train.json")
                with open(json_file_path, 'w') as json_file:
                    json.dump(transform_data, json_file, indent=4)
