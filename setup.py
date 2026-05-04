import os
import subprocess
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# CUTLASS version to use
CUTLASS_VERSION = "v4.4.1"
CUTLASS_REPO = "https://github.com/NVIDIA/cutlass.git"

# Use relative path for CUTLASS directory (setuptools requires relative paths)
CUTLASS_REL_DIR = "/volume/code/jjcheng/cutlass"


def ensure_cutlass():
    """Download CUTLASS if not already present."""
    abs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CUTLASS_REL_DIR)
    if os.path.isfile(os.path.join(abs_dir, "include", "cutlass", "cutlass.h")):
        return  # Already available

    print(f"[siblas] Downloading CUTLASS {CUTLASS_VERSION} ...")
    os.makedirs(os.path.dirname(abs_dir), exist_ok=True)

    # Shallow clone with single branch for speed
    subprocess.check_call([
        "git", "clone",
        "--depth", "1",
        "--branch", CUTLASS_VERSION,
        CUTLASS_REPO,
        abs_dir,
    ])
    print(f"[siblas] CUTLASS downloaded to {abs_dir}")


def get_cuda_extensions():
    ensure_cutlass()

    # include_dirs must be absolute paths for nvcc to find headers correctly
    abs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CUTLASS_REL_DIR)
    cutlass_include = os.path.join(abs_dir, "include")
    cutlass_util_include = os.path.join(abs_dir, "tools", "util", "include")
    sources = [
        os.path.join("csrc", "siblas.cu"),
        os.path.join("csrc", "torch_binding.cu"),
    ]

    ext = CUDAExtension(
        name="siblas._C",
        sources=sources,
        include_dirs=[
            cutlass_include,
            cutlass_util_include,
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--threads=4",
                "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
            ],
        },
        libraries=["cublas", "cublasLt"],
    )
    return [ext]


setup(
    name="siblas",
    version="0.2.0",
    description="GEMM interface for linear layers with CUTLASS BF16x6 emulated FP32 (Blackwell SM100)",
    packages=["siblas"],
    ext_modules=get_cuda_extensions(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
