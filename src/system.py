"""
ViT Lightning Module - Training System

This module wraps our ViT model with PyTorch Lightning to handle:
- Forward pass and loss computation
- Optimizer and learning rate scheduling
- Logging metrics to WandB
- Validation loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy

from src.model.layers import ViT


class ViTLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Vision Transformer.

    What does a LightningModule do?
    - Organizes training code into standard methods
    - Handles GPU/CPU placement automatically
    - Manages logging, checkpointing, etc.

    Key methods we implement:
    - __init__: Create the model
    - forward: Run the model (inference)
    - training_step: One batch of training
    - validation_step: One batch of validation
    - configure_optimizers: Setup optimizer and scheduler

    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): FFN hidden dim ratio
        dropout (float): Dropout probability
        learning_rate (float): Initial learning rate
        weight_decay (float): Weight decay for AdamW
        warmup_epochs (int): Number of warmup epochs
        max_epochs (int): Total training epochs
    """

    def __init__(
        self,
        # Model architecture
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 200,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        # Training hyperparameters
        learning_rate: float = 5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
    ):
        super().__init__()

        # Save all hyperparameters (makes them available in checkpoints)
        self.save_hyperparameters()

        # ====================================================================
        # Create the ViT model
        # ====================================================================
        self.model = ViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # ====================================================================
        # Loss function
        # ====================================================================
        # CrossEntropyLoss for classification
        # It combines LogSoftmax + NLLLoss
        self.criterion = nn.CrossEntropyLoss()

        # ====================================================================
        # Metrics
        # ====================================================================
        # Accuracy metric from torchmetrics (handles batching automatically)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (just calls the model).

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Logits [B, num_classes]
        """
        return self.model(x)
