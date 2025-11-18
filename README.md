# ViT-Accelerated: High-Performance Vision Transformer with Custom CUDA Kernels

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Project Overview
This project implements a custom **Vision Transformer (ViT)** architecture from scratch, optimized for low-latency inference on the Fashion-MNIST dataset.

Unlike standard tutorials that rely on high-level abstractions, this project bridges the gap between **Deep Learning Research** and **High-Performance Computing (HPC)**. It features a custom C++/CUDA extension to fuse activation operations, reducing memory bandwidth bottlenecks, and utilizes NVIDIA TensorRT for post-training quantization.

## Key Features

### 1. Architecture: ViT-Lite
A specialized Vision Transformer designed for small-scale spatial data (28x28 resolution).
- **Patch Embeddings:** Custom convolutional projection for 4x4 patches.
- **Learnable Position Embeddings:** Standard BERT-style absolute positioning.
- **MHA (Multi-Head Attention):** Hand-tuned attention heads for efficiency.

### 2. The "Metal" Layer: Custom CUDA Extensions
To demonstrate proficiency in hardware-aware programming, this project replaces standard PyTorch `nn.GELU` activations with a custom **Fused Bias-GELU Kernel**.
- Written in **C++** and **CUDA**.
- Reduces global memory reads/writes by fusing the bias addition and activation function into a single kernel launch.
- Compiled via `torch.utils.cpp_extension`.

### 3. Tech Stack
- **PyTorch Lightning:** For organized, scalable training loops and checkpointing.
- **Hydra:** For configuration management and ablation studies.
- **WandB:** For experiment tracking and loss visualization.
- **TensorRT:** For compiling the final model to INT8 for maximum inference throughput.
