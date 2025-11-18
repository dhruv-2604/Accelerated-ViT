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



# Accelerated-ViT: HPC-Ready Vision Transformer for Tiny ImageNet

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)
![Dataset](https://img.shields.io/badge/Dataset-Tiny_ImageNet-red)

## 🚀 Project Overview
This project implements a high-performance **Vision Transformer (ViT)** system designed for the **Tiny ImageNet** dataset ($64 \times 64$, 200 Classes).

It bridges the gap between Deep Learning research and Systems Engineering. The system is architected for **Hybrid Development**:
1.  **Local Prototyping:** CPU-based debugging on local machines (MacOS/Apple Silicon).
2.  **HPC Deployment:** JIT-compiled, high-performance training on the **Georgia Tech PACE ICE Cluster** (NVIDIA A100/V100 Nodes).

## ⚡ Key Engineering Features

### 1. Custom CUDA Kernels ("The Metal Layer")
Optimized for the PACE ICE compute nodes, I implemented a custom **Fused Bias-GELU** operator in C++ and CUDA.
- **Bottleneck:** Standard `nn.Linear` + `nn.GELU` incurs expensive global memory round-trips.
- **Solution:** A custom kernel fuses the bias addition and activation, processing data in registers to maximize arithmetic intensity on data-center grade GPUs.

### 2. Data Engineering Pipeline
- **Custom DataModule:** Implements a robust pipeline to download and restructure the raw Tiny ImageNet dataset.
- **Validation Parsing:** Automatically parses `val_annotations.txt` to reorganize the flat validation directory into class-specific subfolders compatible with `ImageFolder`.

### 3. Architecture: Scalable ViT
- **Input:** $64 \times 64$ RGB Images.
- **Tokenization:** Processes 256 patches (Sequence Length) per image.
- **Position Embeddings:** Learnable absolute position embeddings.

### 4. Production-Grade MLOps
- **PyTorch Lightning:** Distributed Data Parallel (DDP) training.
- **Hydra:** Hierarchical configuration management.
- **WandB:** Real-time experiment tracking.


## Usage

### Cluster Deployment (GT PACE ICE)
The `setup.py` script automatically detects the NVIDIA environment on PACE nodes and compiles the C++ extensions.

```bash
# 1. Start an interactive session on a GPU node
salloc -A <your-account> -p gpus -N 1 --gres=gpu:1 --time=1:00:00

# 2. Load CUDA modules (PACE specific)
module load pytorch/2.0.1
module load cuda/11.8

# 3. Install & Train
python setup.py install
python train.py trainer.accelerator=gpu trainer.devices=auto