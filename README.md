# Accelerated-ViT: HPC-Ready Vision Transformer for Tiny ImageNet

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Cluster](https://img.shields.io/badge/Infrastructure-GT_PACE_ICE-gold)
![Dataset](https://img.shields.io/badge/Dataset-Tiny_ImageNet-red)

## Project Overview
This project implements a high-performance **Vision Transformer (ViT)** system designed for the **Tiny ImageNet** dataset ($64 \times 64$, 200 Classes).

1.  **Local Prototyping:** CPU-based debugging on local machines (MacOS/Apple Silicon).
It bridges the gap between Deep Learning research and Systems Engineering. The system is architected for **Hybrid Development**:
2.  **HPC Deployment:** JIT-compiled, high-performance training on the **Georgia Tech PACE ICE Cluster** (NVIDIA A100/V100 Nodes).

## ⚡ Key Engineering Features

### 1. Custom CUDA Kernels ("The Metal Layer")
Optimized for the PACE ICE compute nodes, I implemented a custom **Fused Bias-GELU** operator in C++ and CUDA.
- **Bottleneck:** Standard `nn.Linear` + `nn.GELU` incurs expensive global memory round-trips.
- **Solution:** A custom kernel fuses the bias addition and activation, processing data in registers to maximize arithmetic intensity on data-center grade GPUs.

### 2. Data Engineering Pipeline
- **Custom DataModule:** Implements a robust pipeline to download and restructure the raw Tiny ImageNet dataset.
- **Validation Parsing:** Automatically parses `val_annotations.txt` to reorganize the flat validation directory into class-specific subfolders compatible with `ImageFolder`.
- **HPC Storage Optimization:** Scripts automatically stage data from `$SCRATCH` (slow, large storage) to `$TMPDIR` (fast, local NVMe) at runtime.

### 3. Architecture: Scalable ViT
- **Input:** $64 \times 64$ RGB Images.
- **Tokenization:** Processes 256 patches (Sequence Length) per image.
- **Position Embeddings:** Learnable absolute position embeddings.

### 4. Production-Grade MLOps
- **PyTorch Lightning:** Distributed Data Parallel (DDP) training.
- **Hydra:** Hierarchical configuration management.
- **WandB:** Real-time experiment tracking.

## Usage

### Quick Start (GT PACE ICE Cluster)

This project uses a **Batch Submission** workflow to ensure reproducibility and proper resource usage.

**1. One-Time Data Prep (Run on Login Node)**
First, download and format the Tiny ImageNet dataset into your high-speed scratch storage.
```bash
# Loads Anaconda and runs the preparation script
bash scripts/prepare_data.sh