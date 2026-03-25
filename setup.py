import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_extensions():
    sources = [
        os.path.join("csrc", "siblas.cu"),
        os.path.join("csrc", "torch_binding.cu"),
    ]

    ext = CUDAExtension(
        name="siblas._C",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--use_fast_math",
            ],
        },
        libraries=["cublas"],
    )
    return [ext]


setup(
    name="siblas",
    version="0.1.0",
    description="GEMM interface for linear layers (N=256, K=256) with cuBLAS TF32",
    packages=["siblas"],
    ext_modules=get_cuda_extensions(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
