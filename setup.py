"""
Setup script for Accelerated-ViT with conditional CUDA compilation.

This script automatically detects the build environment:
- PACE ICE Cluster (nvcc available): Builds custom CUDA kernels
- macOS/CPU Development: Skips CUDA build, enables CPU fallback
"""

import os
import shutil
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def build_cuda_extension():
    """
    Conditionally build the CUDA extension based on environment detection.

    Returns:
        list: CUDAExtension objects if nvcc is available, empty list otherwise
    """
    # Check 1: Is nvcc (NVIDIA CUDA Compiler) available?
    nvcc_available = shutil.which("nvcc") is not None

    if not nvcc_available:
        print("=" * 70)
        print("⚠️  CUDA COMPILER (nvcc) NOT FOUND")
        print("=" * 70)
        print("Building in CPU-ONLY mode.")
        print("The custom CUDA kernels will be skipped.")
        print("The model will use standard PyTorch operations as fallback.")
        print("=" * 70)
        return []

    # Check 2: Verify CUDA toolkit is properly configured
    try:
        import torch
        if not torch.cuda.is_available():
            print("=" * 70)
            print("⚠️  CUDA RUNTIME NOT AVAILABLE")
            print("=" * 70)
            print("PyTorch was compiled without CUDA support.")
            print("Skipping custom kernel build.")
            print("=" * 70)
            return []
    except ImportError:
        print("⚠️  PyTorch not installed. Cannot verify CUDA availability.")
        print("Attempting to build CUDA extension anyway...")

    # CUDA is available - proceed with compilation
    print("=" * 70)
    print("✅ CUDA COMPILER DETECTED")
    print("=" * 70)
    print(f"nvcc found at: {shutil.which('nvcc')}")
    print("Building custom CUDA kernels for vit_ops...")
    print("=" * 70)

    # Define the CUDA extension
    ext_modules = [
        CUDAExtension(
            name="vit_ops",
            sources=[
                "csrc/activation.cpp",
                "csrc/activation_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",  # For profiling
                ],
            },
        )
    ]

    return ext_modules


# Main setup configuration
setup(
    name="accelerated-vit",
    version="0.1.0",
    author="Georgia Tech Student",
    description="High-Performance Vision Transformer with Custom CUDA Kernels for Tiny ImageNet",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=build_cuda_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    },
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "hydra-core>=1.3.0",
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
    ],
)
