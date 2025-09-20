import subprocess
import os
from packaging.version import parse, Version
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

PACKAGE_NAME = "cutlass-kernels"

ext_modules = []
generator_flag = []
cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90a,code=sm_90a")

this_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules.append(
    CUDAExtension(
        # package name for import
        name="kernels_cuda.gemm._C",
        sources=[
            "csrc/gemm/gemm_api.cpp",
            "csrc/gemm/matmul.cu",
            "csrc/gemm/matmul_bias.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # add nvcc compile flags
            "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-lineinfo",
                    "--ptxas-options=-v",
                    "--ptxas-options=-O2",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",

                ]
                + generator_flag
                + cc_flag,
        },
        include_dirs=[
            Path(this_dir) / "csrc" / "gemm",
            Path(this_dir) / "third_party/cutlass/include",
            Path(this_dir) / "third_party/cutlass/tools/utils/include" ,
            Path(this_dir) / "third_party/cutlass/examples/common" ,
        ],
    )
)

setup(
    name=PACKAGE_NAME,
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="cutlass kernels",
    ext_modules=ext_modules,
    cmdclass={ "build_ext": BuildExtension},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
)