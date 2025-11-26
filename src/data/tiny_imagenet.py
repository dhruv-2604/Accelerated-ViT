"""
TinyImageNetDataModule - PyTorch Lightning data loading wrapper

This module handles ALL data operations:
- Assumes data is already downloaded and formatted (via prepare_data.sh)
- Defines train/val data augmentation
- Creates DataLoaders with proper batching and workers
"""

import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


# ============================================================================
# TINY IMAGENET STATISTICS
# ============================================================================
# These are computed from the entire Tiny ImageNet dataset
# Used for normalization (makes training more stable)

TINY_IMAGENET_MEAN = [0.4802, 0.4481, 0.3975]  # RGB means
TINY_IMAGENET_STD = [0.2302, 0.2265, 0.2262]   # RGB standard deviations


# ============================================================================
# DATA MODULE
# ============================================================================

class TinyImageNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Tiny ImageNet.

    What is a DataModule?
    - It's a wrapper that organizes all data-related code
    - Keeps data loading separate from model code (clean!)
    - Handles train/val splits, transforms, and dataloaders

    Args:
        data_dir (str): Path to tiny-imagenet-200 directory
                        On PACE: This will be $TMPDIR/tiny-imagenet-200
        batch_size (int): Number of images per batch (128)
        num_workers (int): Number of CPU workers for data loading (4)
        pin_memory (bool): Pin memory for faster GPU transfer
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()

        # Store configuration
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # These will be set in setup()
        self.train_dataset = None
        self.val_dataset = None

        # ====================================================================
        # Define transforms (data augmentation)
        # ====================================================================
        # We define these here so they're created once, not every epoch

        # TRAINING transforms: Add augmentation for better generalization
        self.train_transform = transforms.Compose([
            # 1. Random crop with padding (simulates slight translations)
            #    - Pad image by 4 pixels on each side → 72x72
            #    - Then randomly crop back to 64x64
            transforms.RandomCrop(64, padding=4),

            # 2. Random horizontal flip (50% chance)
            #    - "dog facing left" vs "dog facing right" are both valid
            transforms.RandomHorizontalFlip(),

            # 3. Convert PIL Image to Tensor
            #    - Changes from [H, W, C] to [C, H, W]
            #    - Scales from [0, 255] to [0.0, 1.0]
            transforms.ToTensor(),

            # 4. Normalize using dataset statistics
            #    - Centers values around 0 (mean subtraction)
            #    - Scales to unit variance (std division)
            transforms.Normalize(mean=TINY_IMAGENET_MEAN, std=TINY_IMAGENET_STD),
        ])

        # VALIDATION transforms: NO augmentation (we want consistent evaluation)
        self.val_transform = transforms.Compose([
            # Just convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=TINY_IMAGENET_MEAN, std=TINY_IMAGENET_STD),
        ])

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets.

        This is called ONCE per process (important for distributed training!)

        Args:
            stage: 'fit' for training, 'test' for testing, None for both
        """
        # ====================================================================
        # Construct paths to train/val directories
        # ====================================================================
        # Expected structure (after prepare_data.sh):
        #   data_dir/
        #       train/
        #           n01443537/images/*.JPEG
        #           n01629819/images/*.JPEG
        #           ...
        #       val/
        #           n01443537/*.JPEG  (reorganized!)
        #           n01629819/*.JPEG
        #           ...

        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"

        # Verify directories exist
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found: {train_dir}\n"
                f"Did you run prepare_data.sh?"
            )
        if not val_dir.exists():
            raise FileNotFoundError(
                f"Validation directory not found: {val_dir}\n"
                f"Did you run prepare_data.sh?"
            )

        # ====================================================================
        # Create datasets using ImageFolder
        # ====================================================================
        # ImageFolder automatically:
        # - Scans subdirectories as classes
        # - Assigns labels based on alphabetical order
        # - Loads images as PIL Images

        if stage == "fit" or stage is None:
            # Training dataset with augmentation
            self.train_dataset = ImageFolder(
                root=train_dir,
                transform=self.train_transform,
            )

            # Validation dataset without augmentation
            self.val_dataset = ImageFolder(
                root=val_dir,
                transform=self.val_transform,
            )

            # Print dataset info (helpful for debugging)
            print(f"✅ Loaded Tiny ImageNet:")
            print(f"   Train: {len(self.train_dataset)} images")
            print(f"   Val:   {len(self.val_dataset)} images")
            print(f"   Classes: {len(self.train_dataset.classes)}")

    def train_dataloader(self):
        """
        Create training dataloader.

        Returns:
            DataLoader configured for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for training (randomizes order)
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # Faster GPU transfer
            persistent_workers=True if self.num_workers > 0 else False,  # Keep workers alive between epochs
        )

    def val_dataloader(self):
        """
        Create validation dataloader.

        Returns:
            DataLoader configured for validation
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation (consistent evaluation)
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )
