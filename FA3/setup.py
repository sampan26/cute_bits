# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Point this to your CUTLASS/CuTe checkout (or vendor them in a subdir).
# Example: export CUTLASS_PATH=/path/to/cutlass
cutlass_path = os.environ.get("CUTLASS_PATH", "third_party/cutlass")

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--use_fast_math",
    # Hopper/SM90a. Adjust if you target different GPUs.
    "-gencode=arch=compute_90,code=sm_90a",
    # If you ship headers that use half/bfloat16 intrinsics:
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
]

cxx_flags = [
    "-O3",
    "-std=c++17",
]

setup(
    name="flash_attn_ext",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_ext",
            sources=[
                "binding.cpp",
                "flash_attn_kernel.cu",
            ],
            include_dirs=[
                cutlass_path,                 # has cutlass/ and cute/ headers
                os.path.join(cutlass_path, "include"),
                os.path.join(cutlass_path, "tools", "util", "include"),
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
