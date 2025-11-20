#!/bin/bash
# ============================================================================
# Tiny ImageNet Data Preparation Script for PACE ICE
# ============================================================================
# Purpose: Downloads and reformats the Tiny ImageNet dataset
# Run this ONCE on the LOGIN NODE (not on a compute node)
#
# Usage:
#   bash scripts/prepare_data.sh
#
# This script:
# 1. Loads the Anaconda module
# 2. Activates your virtual environment
# 3. Downloads Tiny ImageNet to ~/scratch (fast storage)
# 4. Reformats the validation set for PyTorch ImageFolder
# ============================================================================

set -e  # Exit on any error

echo "=========================================================================="
echo "Tiny ImageNet Data Preparation for PACE ICE"
echo "=========================================================================="

# ----------------------------------------------------------------------------
# Step 1: Load required modules
# ----------------------------------------------------------------------------
echo "Loading Anaconda module..."
module load anaconda3

# ----------------------------------------------------------------------------
# Step 2: Activate virtual environment
# ----------------------------------------------------------------------------
# IMPORTANT: Replace 'vit' with your actual conda environment name
ENV_NAME="vit"

echo "Activating conda environment: $ENV_NAME"
conda activate $ENV_NAME

# ----------------------------------------------------------------------------
# Step 3: Define data directory
# ----------------------------------------------------------------------------
# The ~/scratch symlink points to high-performance storage
# This is where we store datasets on PACE
DATA_DIR="$HOME/scratch/tiny_imagenet_data"

echo "Data will be prepared in: $DATA_DIR"
echo ""

# ----------------------------------------------------------------------------
# Step 4: Run the Python preparation script
# ----------------------------------------------------------------------------
echo "Running data preparation script..."
echo "This will:"
echo "  1. Download Tiny ImageNet (~237 MB)"
echo "  2. Extract the zip file"
echo "  3. Reorganize validation set (10,000 images)"
echo ""

python src/data/download_and_format.py --output_dir "$DATA_DIR"

# ----------------------------------------------------------------------------
# Step 5: Completion message
# ----------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo "✅ Data preparation complete!"
echo "=========================================================================="
echo "Dataset location: $DATA_DIR/tiny-imagenet-200"
echo ""
echo "Next steps:"
echo "  1. Verify the data looks correct:"
echo "     ls -lh $DATA_DIR/tiny-imagenet-200"
echo ""
echo "  2. Submit a training job:"
echo "     sbatch scripts/train.sh"
echo "=========================================================================="
