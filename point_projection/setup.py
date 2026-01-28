from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

setup(
    name="point_projection_cuda",
    ext_modules=[
        CUDAExtension(
            name="point_projection_cuda",
            sources=[ 
            "pcd_2D.cu",
            "pcd_2D_mask.cu",
            "binding.cpp"
            ],
            include_dirs=["."],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
