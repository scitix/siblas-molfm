import os
import subprocess
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Optimized CUTLASS from gemm_bf16x9 (persistent-B BF16x6 kernel)
CUTLASS_DIR = "/volume/code/chengjiajun/gemm_bf16x9/third_party/cutlass"


def get_cuda_extensions():
    cutlass_include = os.path.join(CUTLASS_DIR, "include")
    cutlass_util_include = os.path.join(CUTLASS_DIR, "tools", "util", "include")

    if not os.path.isfile(os.path.join(cutlass_include, "cutlass", "cutlass.h")):
        raise RuntimeError(
            f"CUTLASS not found at {cutlass_include}. "
            "Expected the gemm_bf16x9 third_party/cutlass tree."
        )

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
            os.path.abspath("csrc"),  # for sm100_mma_warpspecialized_emulated_optimized.hpp
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
                "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                "--expt-relaxed-constexpr",
            ],
        },
        libraries=["cublas"],
    )
    return [ext]


setup(
    name="siblas",
    version="0.2.0",
    description="GEMM interface for linear layers with CUTLASS BF16x6 optimized persistent-B kernel (Blackwell SM100)",
    packages=["siblas"],
    ext_modules=get_cuda_extensions(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
