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


import torch
import point_projection_cuda as cpp_ext

N = 1000      # number of points
H, W = 480, 640

# 3D points
points = torch.randn(N, 3, device="cuda", dtype=torch.float32).contiguous()

# Depth map
depth = torch.randn(H, W, device="cuda", dtype=torch.float32).contiguous()

# Extrinsics and intrinsics
extr = torch.eye(4, device="cuda", dtype=torch.float32).contiguous()
intr = torch.tensor([[500., 0., 320.],
                     [0., 500., 240.],
                     [0.,   0.,   1.]], device="cuda", dtype=torch.float32).contiguous()

depth_thresh = 0.1

# Allocate output tensors
points_2D = torch.full((H,W), 0, device="cuda", dtype=torch.float32)     # projected 2D points
computed_depth = torch.full((H,W), float('inf'), device="cuda", dtype=torch.float32)  # initialize z-buffer with +inf
mask = torch.full((H,W), False, device="cuda",  dtype=torch.bool)    # projected 2D points

# Call CUDA kernel
cpp_ext.pcd2D(points, depth, extr, intr, depth_thresh, points_2D, computed_depth)
cpp_ext.pcd2D_mask(points, depth, extr, intr, depth_thresh, points_2D, computed_depth, mask)

# Inspect first few results
print("First 10 projected points (2D):\n", points_2D[:10])
print("First 10 computed depths:\n", computed_depth[:10])
