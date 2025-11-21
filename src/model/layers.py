"""
Building blocks for Vision Transformer architecture.

This file contains the individual components that make up a ViT model.
We'll build them one at a time to understand each piece.
"""

import torch
import torch.nn as nn


# ============================================================================
# COMPONENT 1: Patch Embedding Layer
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.

    Vision Transformers don't process images pixel-by-pixel. Instead:
    1. Divide image into non-overlapping patches (like tiles)
    2. Flatten each patch into a vector
    3. Project to embedding dimension

    Example with Tiny ImageNet:
        Input:  [B, 3, 64, 64]      (batch of RGB images)
        Output: [B, 256, 256]       (batch of 256 patches, each 256-dim)

    How it works:
        - Image is 64×64
        - Patch size is 4×4
        - Number of patches = (64/4) × (64/4) = 16 × 16 = 256 patches

    Args:
        img_size (int): Size of input image (assumes square images)
        patch_size (int): Size of each patch (assumes square patches)
        in_channels (int): Number of input channels (3 for RGB)
        embed_dim (int): Dimension of patch embeddings
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()

        # Store configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Calculate number of patches
        # For 64×64 image with 4×4 patches: 16 × 16 = 256 patches
        self.num_patches = (img_size // patch_size) ** 2

        # ====================================================================
        # The "Trick": Use Conv2d to Extract Patches
        # ====================================================================
        # Instead of manually slicing the image into patches, we use a
        # convolution with:
        #   - kernel_size = patch_size (extracts one patch at a time)
        #   - stride = patch_size (no overlap between patches)
        #
        # This is mathematically equivalent to:
        #   1. Cutting image into patches
        #   2. Flattening each patch
        #   3. Applying a linear projection
        #
        # But MUCH faster because it's done in a single GPU operation!

        self.projection = nn.Conv2d(
            in_channels=in_channels,      # 3 (RGB)
            out_channels=embed_dim,       # 256 (our embedding dimension)
            kernel_size=patch_size,       # 4×4 kernel
            stride=patch_size,            # Move 4 pixels at a time (no overlap)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image batch to patch embeddings.

        Args:
            x: Input images of shape [B, C, H, W]
               Example: [128, 3, 64, 64]

        Returns:
            Patch embeddings of shape [B, num_patches, embed_dim]
            Example: [128, 256, 256]
        """
        B, C, H, W = x.shape

        # Sanity check: Ensure input dimensions match expected size
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}×{W}) doesn't match model ({self.img_size}×{self.img_size})"

        # ====================================================================
        # Step 1: Apply convolution to extract patches
        # ====================================================================
        # Input:  [B, 3, 64, 64]
        # Output: [B, 256, 16, 16]
        #          └─ embed_dim
        #                └─ 16×16 grid of patches
        x = self.projection(x)

        # ====================================================================
        # Step 2: Flatten spatial dimensions
        # ====================================================================
        # We need to convert from [B, embed_dim, grid_h, grid_w]
        #                      to [B, num_patches, embed_dim]
        #
        # Current shape: [B, 256, 16, 16]
        # Flatten:       [B, 256, 256]  (16×16 = 256 patches)

        x = x.flatten(2)  # Flatten dimensions 2 and 3 (the spatial dims)
        x = x.transpose(1, 2)  # Swap to [B, num_patches, embed_dim]

        # Final shape: [B, 256, 256]
        return x

    def extra_repr(self) -> str:
        """Debug representation when printing the model."""
        return (
            f"img_size={self.img_size}, patch_size={self.patch_size}, "
            f"num_patches={self.num_patches}, embed_dim={self.embed_dim}"
        )
