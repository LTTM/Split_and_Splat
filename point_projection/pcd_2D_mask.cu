/*
* Split&Splat - Copyright (c) 2026, MEDIALab, University of Padova.
*
* Author(s):
*  Leonardo Monchieri (leonardo.monchieri@unipd.it)
*  Elena Camuffo (elenacamuffo97@gmail.com)
*  Francesco Barbato (francesco.barbato@dei.unipd.it)
*  Pietro Zanuttigh (zanuttigh@dei.unipd.it)
*  Simone Milani (simone.milani@dei.unipd.it)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#define BOX_SIZE 1024

#include <float.h>
#include <torch/extension.h>

#include "pcd_2D_mask.h"


static __device__ float fatomicMin(float *addr, float value)
{

        float old = *addr, assumed;

        if(old <= value) return old;
        do
        {
                assumed = old;

                old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

        }while(old!=assumed);

        return old;

}

__global__
void pcd_mask_to_zbuffer_kernel(
    float3* points,
    int N,
  	const float* __restrict__ depth,
    int H, int W,
    const float* __restrict__ extr,   // 4x4 matrix (row-major, length 16)
    const float* __restrict__ intr,   // [fx, fy, cx, cy]
    float depth_thresh,
    float* __restrict__ points_2D,
    float* __restrict__ computed_depth,
    const bool* __restrict__ mask
   // ZBufEntry* z_buffer   // H*W array
) {
   	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float fx = intr[0];
    float fy = intr[1];
    float cx = intr[2];
    float cy = intr[3];

	// 0. Get the point to compute
    float3 p = points[idx];

    // 1. Transform to camera frame
    float x = extr[0] * p.x + extr[1] * p.y + extr[2]  * p.z + extr[3];
    float y = extr[4] * p.x + extr[5] * p.y + extr[6]  * p.z + extr[7];
    float z = extr[8] * p.x + extr[9] * p.y + extr[10] * p.z + extr[11];



	// Point behind the pov
    if (z <= 0.0f) return;

    // 2. Project to 2D
    float u = fx * (x / z) + cx;
    float v = fy * (y / z) + cy;

 
    int i = __float2int_rn(u);
    int j = __float2int_rn(v);



    if (i < 0 || i >= W || j < 0 || j >= H)
        return;

    int pix = j * W + i;
    
    //Check if the point lie in the mask
    if(!mask[pix]) return;

    float d = depth[pix];
    if (fabsf(d - z) > depth_thresh)
        return;
    // 4. Atomic z-buffer update
    int idx2D = j * W + i;


    float old_z = fatomicMin(&computed_depth[idx2D], z);
    if (computed_depth[idx2D] == old_z)  return;

    points_2D[idx2D] = idx;  
}


void run_mask_projection_cuda(
    at::Tensor points,
    at::Tensor depth,
    at::Tensor extr,
    at::Tensor intr,
    float depth_thresh,
    at::Tensor points_2D,
    at::Tensor computed_depth,
    at::Tensor mask
)
{
    float3* d_points = reinterpret_cast<float3*>(points.data_ptr<float>());
    float* d_depth = depth.data_ptr<float>();
    float* d_extr = extr.data_ptr<float>();
    float* d_intr = intr.data_ptr<float>();

    //output datastruct
    float* d_points_2D = points_2D.data_ptr<float>();
    float* d_computed_depth = computed_depth.data_ptr<float>();
    bool* d_mask = mask.data_ptr<bool>();


    int N = points.size(0);
    int H = depth.size(0);
    int W = depth.size(1);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    pcd_mask_to_zbuffer_kernel<<<blocks, threads>>>(
        d_points, N, d_depth, H, W, d_extr, d_intr, depth_thresh, d_points_2D, d_computed_depth, d_mask);
    cudaDeviceSynchronize();
}