#ifndef PCD_2D_H_INCLUDED
#define PCD_2D_H_INCLUDED

#include <torch/extension.h> 
#include <cuda_runtime.h> 

void run_projection_cuda(
    at::Tensor points,   
    at::Tensor depth,
    at::Tensor extr,
    at::Tensor intr,
    float depth_thresh,
    at::Tensor points_2D,
    at::Tensor computed_depth);
  

#endif