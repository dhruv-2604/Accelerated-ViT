#!/bin/bash
# =============================================================================
# SLURM Job Script for ViT Training on PACE ICE
# =============================================================================
#
# Usage:
#   sbatch scripts/train.sh
#
# This script:
#   1. Requests GPU resources from SLURM
#   2. Loads required modules (Anaconda, CUDA)
#   3. Copies data to fast local disk ($TMPDIR)
#   4. Runs training with our ViT model
#
# =============================================================================

# -----------------------------------------------------------------------------
# SLURM Configuration
# -----------------------------------------------------------------------------

#SBATCH -J vit-train                    # Job name
#SBATCH -A YOUR_ACCOUNT                 # Charge account (e.g., GT-gburdell3)
#SBATCH -p biggpu                       # Partition (for A100/V100 nodes)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH -N 1                            # Number of nodes
#SBATCH --ntasks-per-node=1             # Tasks per node
#SBATCH --cpus-per-task=4               # CPU cores (for DataLoader workers)
#SBATCH --mem-per-cpu=8G                # Memory per CPU (32GB total)
#SBATCH --tmp=20G                       # Local disk space for $TMPDIR
#SBATCH -t 02:00:00                     # Time limit (2 hours)
#SBATCH -o logs/train-%j.out            # Output file (%j = job ID)
#SBATCH -e logs/train-%j.err            # Error file

# -----------------------------------------------------------------------------
# Step 1: Environment Setup
# -----------------------------------------------------------------------------

echo "=========================================================================="
echo "Starting ViT Training Job"
echo "=========================================================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${HOSTNAME}"
echo "Start Time: $(date)"
echo "=========================================================================="

# Change to the directory where sbatch was called
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module purge
module load anaconda3

# Activate conda environment (change 'vit' to your environment name)
conda activate vit

# Set WandB API key (REPLACE WITH YOUR KEY or set in ~/.bashrc)
export WANDB_API_KEY="YOUR_WANDB_API_KEY"

# Print environment info
echo ""
echo "Environment:"
echo "  Python:    $(which python)"
echo "  PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:      $(python -c 'import torch; print(torch.version.cuda)')"
echo "  GPU:       $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Data Staging (Copy to Fast Local Disk)
# -----------------------------------------------------------------------------

echo "=========================================================================="
echo "Staging data to local disk ($TMPDIR)..."
echo "=========================================================================="

# Source: Your scratch storage (persistent)
DATA_SOURCE="$HOME/scratch/tiny_imagenet_data/tiny-imagenet-200"

# Destination: Local SSD on compute node (fast, temporary)
DATA_DEST="$TMPDIR/tiny-imagenet-200"

# Verify source exists
if [ ! -d "$DATA_SOURCE" ]; then
    echo "ERROR: Data not found at $DATA_SOURCE"
    echo "Did you run: bash scripts/prepare_data.sh"
    exit 1
fi

# Copy data (this takes ~30 seconds but saves hours of slow I/O)
echo "Copying from: $DATA_SOURCE"
echo "Copying to:   $DATA_DEST"
cp -r "$DATA_SOURCE" "$DATA_DEST"

echo "Data staging complete!"
echo "  Size: $(du -sh $DATA_DEST | cut -f1)"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Run Training
# -----------------------------------------------------------------------------

echo "=========================================================================="
echo "Starting Training..."
echo "=========================================================================="

# Run training with Hydra config
python train.py \
    experiment=tiny_vit_cuda \
    data.data_dir=$DATA_DEST \
    trainer.accelerator=gpu \
    trainer.devices=1

# Capture exit code
EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Step 4: Cleanup and Summary
# -----------------------------------------------------------------------------

echo ""
echo "=========================================================================="
echo "Job Complete!"
echo "=========================================================================="
echo "End Time:   $(date)"
echo "Exit Code:  $EXIT_CODE"
echo "=========================================================================="

# $TMPDIR is automatically cleaned up by SLURM

exit $EXIT_CODE
