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
import collections
import struct
import json
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_erosion
import os
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

from PIL import Image as IMG
from numpy import random as rnd 
import torch

DOWNSAMPLE = 100000

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def erode_mask(mask, size):
    if(size<0.10):
        mask_erosion = binary_erosion(mask, structure=np.ones((3, 3)), iterations=3).astype(mask.dtype)
    elif(size<0.30):
        mask_erosion = binary_erosion(mask, structure=np.ones((5, 5)), iterations=2).astype(mask.dtype)
    else:
        mask_erosion = binary_erosion(mask, structure=np.ones((5, 5)), iterations=2).astype(mask.dtype)
    
    return mask_erosion


def random_downsample(points, target_num):
    idx = np.random.choice(points.shape[0], size=target_num, replace=False)
    return points[idx]

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_intrisics_JSON(path):
    cameras = {}
    with open(path, "r") as fid:
        data = json.load(fid)
        camera_id = 0
        model = data["modalities"]["rgb"]["camera_model"]
        width  = data["modalities"]["rgb"]["width"]
        height = data["modalities"]["rgb"]["height"]
        params = np.array(tuple(map(float, [data["modalities"]["rgb"]["fx"], 
                                            data["modalities"]["rgb"]["fy"], 
                                            data["modalities"]["rgb"]["cx"],
                                            data["modalities"]["rgb"]["cy"]])))
        cameras[camera_id] = Camera(id=camera_id, model=model,
                                    width=width, height=height,
                                    params=params)
    fid.close()
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_extrinsic_JSON(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        data = json.load(fid)
        
        for frame in data["modalities"]["rgb"]["frames"]:
            image_id = frame["frame_id"]
            camtoworld = np.array(frame["camtoworld"])  # 3x4 matrix

            # Extract rotation (3x3) and translation (3x1)
            
            tvec = camtoworld[:, 3]
            R_mat = camtoworld[:, :3]

            qvec = R.from_matrix(R_mat).as_quat()
            qvec = np.roll(qvec, shift=1)  # Reorder to w, x, y, z

            camera_id = 0
            image_name = frame["file_name"]
    
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=None, point3D_ids=None)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


"""
Extract from views 3D point related to a mask region
"""
def instance_filter(extrinsic, mask_path, instance_id):
    points3D_filter = []

    for i in extrinsic:
        image = extrinsic[i]
        image_name = image.name
        mask_label = image_name.replace(".png", ".npy")
        mask = np.squeeze(np.load(os.path.join(mask_path, mask_label)))

  
        for j, (x, y) in enumerate(image.xys):
            clipped_x = np.clip(x, 0, mask.shape[1]-1)
            clipped_y = np.clip(y, 0, mask.shape[0]-1)
           
            if (mask[int(clipped_y), int(clipped_x)] and 
                image.point3D_ids[j] != -1):
                points3D_filter.append(image.point3D_ids[j].item())
                

    filtered_3D = np.array(points3D_filter)
    filtered_3D = np.unique(filtered_3D)
       
    return filtered_3D

"""
Extract 3D point present with a correspondance in filter
"""
def filter_point3D_instance(xyzs_global, rgb_global, filter):
   
    num_points = xyzs_global.shape[0]
    filter = filter[filter <= num_points]

    
    # #Bounding box version
    xyzs_inst =  xyzs_global[filter]
    xyzs_glbal = np.delete(xyzs_global, filter, axis=0)
    

    min_corner_bb = np.min(xyzs_inst, axis = 0 ) 
    max_corner_bb = np.max(xyzs_inst, axis = 0 ) 

    print(min_corner_bb)
    print(max_corner_bb)

    bb_points = np.all((xyzs_glbal >= min_corner_bb) & (xyzs_glbal <= max_corner_bb), axis=1)
    outside_points = np.any((xyzs_glbal < min_corner_bb) | (xyzs_glbal > max_corner_bb), axis=1)
    


    n = xyzs_glbal[outside_points].shape[0]
    rgbs_glbl = np.zeros((n, 3), dtype=int)
    rgbs_inst = xyzs_glbal[bb_points]

   
    #return  np.vstack((xyzs_global[bb_points], xyzs_global[outside_points])), np.vstack((rgbs_inst, rgbs_glbl))
    #return  xyzs_global[bb_points], rgbs_inst
    return xyzs_inst, xyzs_glbal, min_corner_bb, max_corner_bb
    # # Msk point version
    # xyzs_inst =  xyzs_global[filter]
    # xyzs_glbal = np.delete(xyzs_inst, filter, axis=0)
    


    # n = rgb_global.shape[0] - rgb_global[filter].shape[0]
    # rgbs_glbl = np.zeros((n, 3), dtype=int)
    # rgbs_inst = rgb_global[filter]

    # #return xyzs_global[bb_points], xyzs_global[outside_points], rgbs_inst, rgbs_glbl
    # #return  np.vstack((xyzs_inst, xyzs_glbal)), np.vstack((rgbs_inst, rgbs_glbl))

def extract_by_name(elements, name):
    for i in elements:
  
        element= elements[i]
        if element.name == name:
            return element
    return None  # Return None if no match is found



def filterPLY(path_ply, folder_path, intrinsics, extrinsics):

    point_cloud = o3d.io.read_point_cloud(path_ply)
    


    points = np.asarray(point_cloud.points)
    filtered_3D = []
    images_path = os.path.join(folder_path, "images")
    images = os.listdir(images_path)
    
    for image_label in images:
        
        #image_path = os.path.join(images_path, image_label)
        #image = cv2.imread(image_path)
        img_lbl = image_label#.replace(".png", ".JPEG")

        if image_label.split(".")[-1].lower() != "png":
            image_label = os.path.splitext(image_label)[0] + ".png"

        extrinsic_image =  extract_by_name(extrinsics, img_lbl)

        try:
            camera_id = extrinsic_image.camera_id
        except:
            continue
     
        intrinsic_image =  intrinsics[camera_id]

        f_x, f_y = intrinsic_image.params[:2] # Focal lengths in x and y (pixels)
        c_x, c_y = intrinsic_image.params[2:] # Optical center (image center in pixels)

        R = qvec2rotmat(extrinsic_image.qvec)  
        t = extrinsic_image.tvec  

        K = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]])
    

        #mask_label = image_label.replace(".png", ".npy")

        mask_path = os.path.join(folder_path, "masks", image_label)

        # mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        try:
            mask_img = IMG.open(mask_path).convert("RGBA")
        except: 
            continue
       
        alpha = np.array(mask_img)[:, :, 3]
        mask = alpha > 0

        # mask = mask_img>127

        # mask = np.squeeze(mask)

        # plt.imshow(mask)
        # plt.savefig("test.png")

        #mask = np.squeeze(np.load(mask_path))
   
    
        for point in points:
            # Homogeneous coordinates of the 3D point
            point_homogeneous = np.append(point, 1)  # [X, Y, Z, 1]
            
            # Camera projection (R * point + t)
            projected_point = K @ (R @ point_homogeneous[:3] + t).reshape((3, 1))

            # Convert from homogeneous to 2D
            x_2d, y_2d = projected_point[0] / projected_point[2], projected_point[1] / projected_point[2]

            clipped_x = np.clip(x_2d, 0, mask.shape[1]-1)
            clipped_y = np.clip(y_2d, 0, mask.shape[0]-1)

            if mask[int(clipped_y), int(clipped_x)] :
                filtered_3D.append(point)
            

    filtered_3D = np.array(filtered_3D)

    #path 
    path_filtered_ply = os.path.join(folder_path, "points3d.ply")

   
    if(len(filtered_3D)== 0): 
        o3d.io.write_point_cloud(path_filtered_ply, point_cloud)
        print("NO POINTS!")
        return
    
    u, c = np.unique(filtered_3D, axis = 0, return_counts=True)
    dup = u[c > len(images)/2]

    u_1, c_1 = np.unique(filtered_3D, axis = 0, return_counts=True)
    dup_1 = u_1[c_1 > 5]

    # Find indices of matching points in the point cloud
    indices = np.where((points[:, None] == dup).all(axis=2).any(axis=1))[0]

    # Extract matched points
    filtered_pcd = point_cloud.select_by_index(indices)

    #Save the filtered point cloud to a PLY file
    #o3d.io.write_point_cloud(path_filtered_ply, filtered_pcd)

    min_corner_bb = np.min(dup_1, axis = 0 ) 
    max_corner_bb = np.max(dup_1, axis = 0 ) 

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_corner_bb, max_corner_bb)
    bbox.color = (1, 0, 0)  # Set bounding box color to red
    cropped_pcd = point_cloud.crop(bbox)

    print("Instance point: %d"%len(filtered_pcd.points))

    print("Bounding box point: %d"%len(cropped_pcd.points))

    #o3d.visualization.draw_geometries([point_cloud, bbox])

    path_filtered_ply = os.path.join(folder_path, "points3d.ply")
    # Save the filtered point cloud to a PLY file
    if(len(filtered_pcd.points)>10):
        o3d.io.write_point_cloud(path_filtered_ply, filtered_pcd)
    elif(len(cropped_pcd.points)>10):
        o3d.io.write_point_cloud(path_filtered_ply, cropped_pcd)
    else:
        o3d.io.write_point_cloud(path_filtered_ply, point_cloud)


    #return cropped_ply
def filterPLY_blender(path_ply, folder_path, intrinsics, extrinsics):

    point_cloud = o3d.io.read_point_cloud(path_ply)

    points = np.asarray(point_cloud.points)
    #points = random_downsample(points, DOWNSAMPLE)
    all_filtered_3D = []  # Store points from all images
    parent_path = os.path.dirname(os.path.abspath(folder_path))
    
    images_path = os.path.join(parent_path, "images")
    images = os.listdir(images_path)
    debug_path = os.path.join(folder_path, "projection_overlays")
    for image_label in images:
        
        #image_path = os.path.join(images_path, image_label)
        #image = cv2.imread(image_path)
        img_lbl = os.path.splitext(image_label)[0]

        if image_label.split(".")[-1].lower() != "png":
            image_label = os.path.splitext(image_label)[0] + ".png"

        try:
            extrinsic_image =  extrinsics[img_lbl]
        
            intrinsic_image =  intrinsics[img_lbl]
        except:
            continue

       # --- Intrinsics ---
        f_x = intrinsic_image["fx"]
        f_y = intrinsic_image["fy"]
        c_x = intrinsic_image["cx"]
        c_y = intrinsic_image["cy"]

        K = np.array([
            [f_x, 0.0, c_x],
            [0.0, f_y, c_y],
            [0.0, 0.0, 1.0]
        ])

        # --- Extrinsics ---
        R = extrinsic_image["R"]   # (3, 3) world → camera, already transposed as expected
        t = extrinsic_image["T"]   # (3,)
    

        #mask_label = image_label.replace(".png", ".npy")

        mask_path = os.path.join(folder_path, "masks", image_label)

        # mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        try:
            mask_img = IMG.open(mask_path).convert("RGBA")
            overlay_canvas = cv2.cvtColor(np.array(mask_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        except: 
            continue
       
        alpha = np.array(mask_img)[:, :, 3]
        mask = alpha > 0
        
        H, W = mask.shape[:2]
        size = mask.sum()/(H*W)
        mask = erode_mask((mask), size)
        
        
        z_buffer = {}
        for point in points:
            # Homogeneous coordinates of the 3D point
            point_homogeneous = np.append(point, 1)  # [X, Y, Z, 1]
            
            # Camera projection (R * point + t)
            projected_point = K @ (R @ point_homogeneous[:3] + t).reshape((3, 1))

            # Convert from homogeneous to 2D
            if projected_point[2]<0: continue
            
            x_2d, y_2d = projected_point[0] / projected_point[2], projected_point[1] / projected_point[2]
            
            depth = projected_point[2]
            
            ix, iy = int(x_2d), int(y_2d)

            # Check if point is within image boundaries
            # 3. Check image boundaries and Mask
            if 0 <= ix < mask.shape[1] and 0 <= iy < mask.shape[0]:
                if mask[iy, ix]:
                    # 4. Z-Buffer Check: Is this point closer than what we've seen at this pixel?
                    pixel_coord = (ix, iy)
                    if pixel_coord not in z_buffer or depth < z_buffer[pixel_coord][0]:
                        z_buffer[pixel_coord] = (depth, point)
                        
        filtered_3D = [data[1] for data in z_buffer.values()]
        all_filtered_3D.extend(filtered_3D)


        for (ix, iy), (depth, point) in z_buffer.items():
            cv2.circle(overlay_canvas, (ix, iy), 2, (0, 0, 255), -1)
            
        debug_path = os.path.join(folder_path, "mask_projections")
        os.makedirs(debug_path, exist_ok=True)
        
        save_path = os.path.join(debug_path, f"mask_overlay_{image_label}")
        cv2.imwrite(save_path, overlay_canvas)
    
    all_filtered_3D = np.array(all_filtered_3D)

    #path 
    path_filtered_ply = os.path.join(folder_path, "points3d.ply")

    if(len(all_filtered_3D)== 0): 
        o3d.io.write_point_cloud(path_filtered_ply, point_cloud)
        print("NO POINTS!")
        return
    
    # u, c = np.unique(filtered_3D, axis = 0, return_counts=True)
    # dup = u[c > len(images)/2]

    # u_1, c_1 = np.unique(filtered_3D, axis = 0, return_counts=True)
    # dup_1 = u_1[c_1>5]

    # Find indices of matching points in the point cloud
    #indices = np.where((points[:, None] == dup).all(axis=2).any(axis=1))[0]

    # Extract matched points
    #filtered_pcd = point_cloud.select_by_index(indices)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(all_filtered_3D)

    #Save the filtered point cloud to a PLY file
    #o3d.io.write_point_cloud(path_filtered_ply, filtered_pcd)

    min_corner_bb = np.min(all_filtered_3D, axis = 0 ) 
    max_corner_bb = np.max(all_filtered_3D, axis = 0 ) 

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_corner_bb, max_corner_bb)
    bbox.color = (1, 0, 0)  # Set bounding box color to red
    cropped_pcd = point_cloud.crop(bbox)

    print("Instance point: %d"%len(filtered_pcd.points))

    print("Bounding box point: %d"%len(cropped_pcd.points))

    #o3d.visualization.draw_geometries([point_cloud, bbox])

    path_filtered_ply = os.path.join(folder_path, "points3d.ply")
    # Save the filtered point cloud to a PLY file
    if(len(filtered_pcd.points)>=10):
        o3d.io.write_point_cloud(path_filtered_ply, filtered_pcd)
    elif(len(cropped_pcd.points)>=10):
        o3d.io.write_point_cloud(path_filtered_ply, cropped_pcd)
    else:
        o3d.io.write_point_cloud(path_filtered_ply, point_cloud)

def camera_matrix(intrinsic, extrinsic):
    f_x, f_y = intrinsic.params[:2]  # Focal lengths in x and y (pixels)
    c_x, c_y = intrinsic.params[2:]  # Optical center (image center in pixels)

    R = torch.tensor(qvec2rotmat(extrinsic.qvec), dtype=torch.float32)
    t = torch.tensor(extrinsic.tvec, dtype=torch.float32)

    K = torch.tensor([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ], dtype=torch.float32)

    return R, t, K

def from3Dto2D(point, image, intrinsic, extrinsic):
    point_homogeneous = torch.cat((point, torch.tensor([1.0])))  # [X, Y, Z, 1]
    R, t, K= camera_matrix(intrinsic, extrinsic) 
    projected_point = K @ (R @ point_homogeneous[:3] + t).view(3, 1)


    x_2d, y_2d = projected_point[0] / projected_point[2], projected_point[1] / projected_point[2]

    h,w = image.size
    clipped_x = torch.clamp(x_2d, 0, w - 1).long()
    clipped_y = torch.clamp(y_2d, 0, h - 1).long()

    return x_2d, y_2d

def filterPLY_torch(path_ply, folder_path, intrinsics, extrinsics):
    point_cloud = o3d.io.read_point_cloud(path_ply)
    points = torch.tensor(np.asarray(point_cloud.points), dtype=torch.float32)

    filtered_3D = []
    images_path = os.path.join(folder_path, "images")
    images = os.listdir(images_path)

    for image_label in images:
        extrinsic_image = extract_by_name(extrinsics, image_label)
        try:
            camera_id = extrinsic_image.camera_id
        except:
            continue

        intrinsic_image = intrinsics[camera_id]

        R, t, K= camera_matrix(intrinsic_image, extrinsic_image) 

        mask_label = image_label.replace(".png", ".npy")
        mask_path = os.path.join(folder_path, "mask", mask_label)
        mask = torch.tensor(np.squeeze(np.load(mask_path)), dtype=torch.bool)

        for point in points:
            point_homogeneous = torch.cat((point, torch.tensor([1.0])))  # [X, Y, Z, 1]
            projected_point = K @ (R @ point_homogeneous[:3] + t).view(3, 1)

            x_2d, y_2d = projected_point[0] / projected_point[2], projected_point[1] / projected_point[2]

            clipped_x = torch.clamp(x_2d, 0, mask.shape[1] - 1).long()
            clipped_y = torch.clamp(y_2d, 0, mask.shape[0] - 1).long()

            if mask[clipped_y, clipped_x]:
                filtered_3D.append(point.numpy())

    filtered_3D = torch.tensor(filtered_3D, dtype=torch.float32)
    unique_points, counts = torch.unique(filtered_3D, dim=0, return_counts=True)
    dup = unique_points[counts > len(images) / 2]

    min_corner_bb = torch.min(dup, dim=0).values
    max_corner_bb = torch.max(dup, dim=0).values

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_corner_bb.numpy(), max_corner_bb.numpy())
    bbox.color = (1, 0, 0)  # Set bounding box color to red
    cropped_pcd = point_cloud.crop(bbox)

    print("Bounding box point: %d" % len(cropped_pcd.points))

    path_filtered_ply = os.path.join(folder_path, "sparse/0/points3D.ply")
    o3d.io.write_point_cloud(path_filtered_ply, cropped_pcd)


def plot_ply(ply, image_path, intrinsics, extrinsics):

    points = np.asarray(ply.points)
    
    image = cv2.imread(image_path) 
  
    f_x, f_y = intrinsics.params[:2] # Focal lengths in x and y (pixels)
    c_x, c_y = intrinsics.params[2:] # Optical center (image center in pixels)

    R = qvec2rotmat(extrinsics.qvec)  
    t = extrinsics.tvec  

    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]])


    projected_points_2d = []
    for point in points:
        # Homogeneous coordinates of the 3D point
        point_homogeneous = np.append(point, 1)  # [X, Y, Z, 1]
            
        # Camera projection (R * point + t)
        projected_point = K @ (R @ point_homogeneous[:3] + t).reshape((3, 1))

        # Convert from homogeneous to 2D
        x_2d, y_2d = projected_point[0] / projected_point[2], projected_point[1] / projected_point[2]
        projected_points_2d.append([x_2d, y_2d])

    projected_points_2d = np.array(projected_points_2d).squeeze(axis=-1)
    
   

            
          
def filterPLY_img(path_ply, mask_path, intrinsics, extrinsics):

    point_cloud = o3d.io.read_point_cloud(path_ply)


    points = np.asarray(point_cloud.points)


    f_x,f_y = intrinsics.params[:2] # Focal lengths in x and y (pixels)
    c_x, c_y = intrinsics.params[2:] # Optical center (image center in pixels)

    R = qvec2rotmat(extrinsics.qvec)  
    t = extrinsics.tvec  

    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]])
    

    projected_points_2d = []

    mask = np.squeeze(np.load(mask_path))
    

    filtered_3D = []
    for point in points:
        # Homogeneous coordinates of the 3D point
        point_homogeneous = np.append(point, 1)  # [X, Y, Z, 1]
            
        # Camera projection (R * point + t)
        projected_point = K @ (R @ point_homogeneous[:3] + t).reshape((3, 1))

        # Convert from homogeneous to 2D
        x_2d, y_2d = projected_point[0] / projected_point[2], projected_point[1] / projected_point[2]

        clipped_x = np.clip(x_2d, 0, mask.shape[1]-1)
        clipped_y = np.clip(y_2d, 0, mask.shape[0]-1)


            
        if mask[int(clipped_y), int(clipped_x)] :
            projected_points_2d.append([x_2d, y_2d])
            filtered_3D.append(point)
            

    filtered_3D = np.array(filtered_3D)
    min_corner_bb = np.min(filtered_3D, axis = 0 ) 
    max_corner_bb = np.max(filtered_3D, axis = 0 ) 

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_corner_bb, max_corner_bb)
    bbox.color = (1, 0, 0)  # Set bounding box color to red
    point_cloud = point_cloud.crop(bbox)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_3D)
 

    return pcd

