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

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import argparse
import os

def load_camera_pose(pose_file):
    """Extract transformation matrix from a camera_pose.ply file."""
    pcd = o3d.io.read_point_cloud(pose_file)

    # Extract vertex data (assuming pose stored as [x, y, z, qx, qy, qz, qw])
    points = np.asarray(pcd.points)

    if points.shape[1] < 7:
        raise ValueError("The PLY file does not contain quaternion data!")

    # Extract translation (first 3 columns)
    translation = points[:, :3]

    # Extract quaternion (next 4 columns)
    quaternions = points[:, 3:7]

    # Convert quaternion to rotation matrices
    rotation_matrices = np.array([Rotation.from_quat(q).as_matrix() for q in quaternions])

    # Construct 4x4 transformation matrices
    transformation_matrices = []
    for i in range(len(points)):
        T = np.eye(4)
        T[:3, :3] = rotation_matrices[i]  # Set rotation
        T[:3, 3] = translation[i]         # Set translation
        transformation_matrices.append(T)

    return transformation_matrices


def load_point_cloud(pointcloud_file):
    """Load point cloud and extract points + IDs (if available)."""
    pcd = o3d.io.read_point_cloud(pointcloud_file)
    points = np.asarray(pcd.points)
    # Check if IDs exist
    point_ids = np.arange(len(points))  # If no IDs are available
    return points, point_ids

def project_points(points, T, K):
    """Project 3D points using camera pose and intrinsics."""
    xys = []
    valid_ids = []

    for i, P_w in enumerate(points):
        # Convert to homogeneous coordinates
        P_w_h = np.append(P_w, 1)  
        P_c = T @ P_w_h  # Transform to camera space
        
        if P_c[2] <= 0:  # Ignore points behind the camera
            continue

        # Project using intrinsics
        p = K @ P_c[:3]
        x, y = p[0] / p[2], p[1] / p[2]

        xys.append((x, y))
        valid_ids.append(i)

    return np.array(xys), np.array(valid_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation evaluation")

    parser.add_argument(
        "-scene",
        type=str, 
        default=None,
        help="scene where compute cameras"
    )
    
    args = parser.parse_args()
    
    SCENE = args.scene
    
    # Load camera pose
    T = load_camera_pose(os.path.join("./data", SCENE, "sparse", ".ply" ))

    # Load point cloud
    points, point_ids = load_point_cloud(os.path.join("./data", SCENE, "sparse", ".pointcloud.ply"))

    # Define camera intrinsic matrix K (modify with actual values)
    K = np.array([[1000, 0, 640],  # fx, 0, cx
                [0, 1000, 360],  # 0, fy, cy
                [0, 0, 1]])       # 0,  0,  1

    # Project points
    xys, valid_ids = project_points(points, T, K)

    # Output results
    print("2D Image Coordinates (x, y):", xys)
    print("Point IDs:", valid_ids)