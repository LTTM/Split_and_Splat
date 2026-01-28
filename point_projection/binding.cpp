
#include "pcd_2D.h"
#include "pcd_2D_mask.h"
#include <torch/extension.h>

void run_projection_cuda_wrapper(
    at::Tensor points,
    at::Tensor depth,
    at::Tensor extr,
    at::Tensor intr,
    float depth_thresh,
    at::Tensor points_2D,
    at::Tensor computed_depth
    )
{
    run_projection_cuda(
        points,
        depth,
        extr,
        intr,
        depth_thresh,
        points_2D,
        computed_depth
    );
}

void run_mask_projection_cuda_wrapper(
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
    run_mask_projection_cuda(
        points,
        depth,
        extr,
        intr,
        depth_thresh,
        points_2D,
        computed_depth,
        mask
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pcd2D_mask", &run_mask_projection_cuda_wrapper, "Project 2D mask points to 3D (CUDA)");

    m.def("pcd2D", &run_projection_cuda_wrapper, "Project 3D points to 2D (CUDA)");
}