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


# ============================================================================
# COMPONENT 2: Positional Embedding
# ============================================================================

class PositionalEmbedding(nn.Module):
    """
    Adds learnable positional information to patch embeddings.

    THE PROBLEM:
        After patch embedding, we have 256 vectors. But they have no sense
        of "where" they came from in the original image!

        Patch from top-left and patch from bottom-right look the same
        to the Transformer - just vectors of numbers.

    THE SOLUTION:
        Add a unique "position vector" to each patch. The model learns
        what these position vectors should be during training.

    Example:
        Input:  [B, 256, 256]  (patch embeddings)
        Output: [B, 256, 256]  (patch embeddings WITH position info)

    How it works:
        We create a learnable parameter of shape [1, num_patches, embed_dim]
        and simply ADD it to the patch embeddings:

        output = patch_embeddings + positional_embeddings

    Args:
        num_patches (int): Number of patches (256 for our 64×64 images)
        embed_dim (int): Embedding dimension (256)
        dropout (float): Dropout probability for regularization
    """

    def __init__(
        self,
        num_patches: int = 256,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ====================================================================
        # Create the learnable positional embeddings
        # ====================================================================
        # Shape: [1, num_patches, embed_dim]
        #         └─ Batch dim of 1 (will broadcast to any batch size)
        #             └─ One position vector per patch
        #                  └─ Same dimension as patch embeddings

        # Initialize to zeros - the model will learn the right values
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Standard practice: Initialize with small random values instead of zeros
        # This helps training start faster (breaks symmetry)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ====================================================================
        # Optional: Dropout for regularization
        # ====================================================================
        # Randomly zero out some dimensions during training
        # This prevents overfitting
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to patch embeddings.

        Args:
            x: Patch embeddings of shape [B, num_patches, embed_dim]
               Example: [128, 256, 256]

        Returns:
            Embeddings with position information, same shape as input
            Example: [128, 256, 256]
        """
        # ====================================================================
        # Add positional embeddings
        # ====================================================================
        # self.pos_embed is [1, 256, 256]
        # x is [B, 256, 256]
        # PyTorch automatically broadcasts the batch dimension!

        x = x + self.pos_embed

        # ====================================================================
        # Apply dropout
        # ====================================================================
        x = self.dropout(x)

        return x

    def extra_repr(self) -> str:
        """Debug representation."""
        return f"num_patches={self.pos_embed.size(1)}, embed_dim={self.pos_embed.size(2)}"


# ============================================================================
# COMPONENT 3: Multi-Head Self-Attention
# ============================================================================

class Attention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    THE CORE IDEA:
        Each patch asks: "Which other patches are relevant to me?"

        This is done through three learnable projections:
        - Query (Q): "What am I looking for?"
        - Key (K):   "What do I have to offer?"
        - Value (V): "What information can I share?"

    HOW IT WORKS (Step by Step):
        1. Project input into Q, K, V
        2. Compute attention scores: Q @ K^T (who matches with whom?)
        3. Apply softmax (normalize scores to probabilities)
        4. Weighted sum of Values: scores @ V

    MULTI-HEAD:
        Instead of one big attention, we do several smaller ones in parallel.
        - Each "head" can focus on different relationships
        - Head 1 might focus on color similarity
        - Head 2 might focus on spatial proximity
        - etc.

    Args:
        embed_dim (int): Input embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ====================================================================
        # Configuration
        # ====================================================================
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Each head gets a portion of the embedding dimension
        # 256 dims / 8 heads = 32 dims per head
        self.head_dim = embed_dim // num_heads

        # Sanity check: embedding must be divisible by number of heads
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # Scaling factor for attention scores (prevents huge values)
        # We'll explain this more when we use it
        self.scale = self.head_dim ** -0.5  # = 1 / sqrt(head_dim)

        # ====================================================================
        # Q, K, V Projections
        # ====================================================================
        # These are learnable linear transformations
        # They convert our patch embeddings into Query, Key, Value vectors

        # Option 1: Three separate Linear layers
        # self.q_proj = nn.Linear(embed_dim, embed_dim)
        # self.k_proj = nn.Linear(embed_dim, embed_dim)
        # self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Option 2 (more efficient): One big Linear that we split
        # This computes Q, K, V in one matrix multiplication!
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # Output projection (combines heads back together)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head self-attention.

        Args:
            x: Input of shape [B, num_patches, embed_dim]
               Example: [128, 256, 256]

        Returns:
            Output of shape [B, num_patches, embed_dim]
            Example: [128, 256, 256]
        """
        B, N, C = x.shape  # Batch, Num patches, Channels (embed_dim)

        # ====================================================================
        # Step 1: Compute Q, K, V in one operation
        # ====================================================================
        # Input x:     [B, N, C]        = [128, 256, 256]
        # After qkv:   [B, N, 3*C]      = [128, 256, 768]

        qkv = self.qkv(x)

        # ====================================================================
        # Step 2: Reshape to separate Q, K, V and heads
        # ====================================================================
        # Current:  [B, N, 3*C]         = [128, 256, 768]
        # Reshape:  [B, N, 3, num_heads, head_dim]  = [128, 256, 3, 8, 32]
        # Permute:  [3, B, num_heads, N, head_dim]  = [3, 128, 8, 256, 32]

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Now split into Q, K, V (each has shape [B, num_heads, N, head_dim])
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v each: [128, 8, 256, 32]

        # ====================================================================
        # Step 3: Compute attention scores
        # ====================================================================
        # Q @ K^T: "How much does each query match each key?"
        #
        # q:      [B, heads, N, head_dim] = [128, 8, 256, 32]
        # k^T:    [B, heads, head_dim, N] = [128, 8, 32, 256]
        # Result: [B, heads, N, N]        = [128, 8, 256, 256]
        #
        # The [256, 256] matrix is: "attention from patch i to patch j"

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # ====================================================================
        # Step 4: Softmax to get probabilities
        # ====================================================================
        # Convert scores to probabilities (each row sums to 1)
        attn = attn.softmax(dim=-1)

        # Optional dropout on attention weights
        attn = self.attn_dropout(attn)

        # ====================================================================
        # Step 5: Weighted sum of values
        # ====================================================================
        # attn: [B, heads, N, N]       = [128, 8, 256, 256]
        # v:    [B, heads, N, head_dim] = [128, 8, 256, 32]
        # out:  [B, heads, N, head_dim] = [128, 8, 256, 32]

        out = attn @ v

        # ====================================================================
        # Step 6: Concatenate heads and project
        # ====================================================================
        # Reshape: [B, N, num_heads * head_dim] = [128, 256, 256]
        out = out.transpose(1, 2).reshape(B, N, C)

        # Final projection
        out = self.out_proj(out)
        out = self.out_dropout(out)

        return out

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}"


# ============================================================================
# COMPONENT 4: Feed-Forward Network (FFN)
# ============================================================================

class FeedForward(nn.Module):
    """
    Simple feed-forward network applied to each patch independently.

    Structure:
        Linear (expand) → GELU → Dropout → Linear (contract) → Dropout

    Why expand then contract?
        - Expand to higher dimension (4x) gives more "thinking space"
        - Non-linearity (GELU) allows learning complex patterns
        - Contract back to original dimension

    THIS IS WHERE OUR CUSTOM CUDA KERNEL IS USED!
        - The FusedBiasGELU layer replaces: Linear bias + GELU activation
        - On GPU: Uses our fast CUDA kernel
        - On CPU: Falls back to PyTorch

    Args:
        embed_dim (int): Input/output dimension
        hidden_dim (int): Hidden dimension (typically 4x embed_dim)
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Default hidden dim is 4x the embedding dim
        if hidden_dim is None:
            hidden_dim = embed_dim * 4

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # ====================================================================
        # Layer 1: Expand dimension
        # ====================================================================
        # Input:  [B, N, 256]
        # Output: [B, N, 1024]
        #
        # NOTE: We set bias=False here because FusedBiasGELU has its own bias!
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=False)

        # ====================================================================
        # Activation: Our custom fused layer!
        # ====================================================================
        # This combines: bias addition + GELU activation
        # Import from our custom layers module
        from src.layers.fused_gelu import FusedBiasGELU
        self.act = FusedBiasGELU(hidden_dim)

        # ====================================================================
        # Layer 2: Contract back to original dimension
        # ====================================================================
        # Input:  [B, N, 1024]
        # Output: [B, N, 256]
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        Args:
            x: Input of shape [B, num_patches, embed_dim]
               Example: [128, 256, 256]

        Returns:
            Output of same shape [B, num_patches, embed_dim]
        """
        # Expand: [B, N, 256] → [B, N, 1024]
        x = self.fc1(x)

        # Activation: FusedBiasGELU (uses CUDA kernel if available!)
        x = self.act(x)

        # Dropout
        x = self.dropout(x)

        # Contract: [B, N, 1024] → [B, N, 256]
        x = self.fc2(x)

        # Final dropout
        x = self.dropout(x)

        return x

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}"


# ============================================================================
# COMPONENT 5: Transformer Block (combines Attention + FFN)
# ============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer block.

    Structure:
        x → LayerNorm → Attention → + (residual) → LayerNorm → FFN → + (residual)

    TWO KEY TRICKS:

    1. LAYER NORMALIZATION:
       - Normalizes values to prevent them from exploding/vanishing
       - Applied BEFORE each sub-layer (this is called "Pre-Norm")

    2. RESIDUAL CONNECTIONS:
       - The "+" means we ADD the input back to the output
       - This helps gradients flow during training
       - Formula: output = input + sublayer(input)

    Visual:
        ┌─────────────────────────────────────────┐
        │  Input                                  │
        │    │                                    │
        │    ├──────────────┐                     │
        │    ▼              │                     │
        │  LayerNorm        │                     │
        │    ▼              │                     │
        │  Attention        │ (residual)          │
        │    ▼              │                     │
        │    + ◄────────────┘                     │
        │    │                                    │
        │    ├──────────────┐                     │
        │    ▼              │                     │
        │  LayerNorm        │                     │
        │    ▼              │                     │
        │  FeedForward      │ (residual)          │
        │    ▼              │                     │
        │    + ◄────────────┘                     │
        │    │                                    │
        │  Output                                 │
        └─────────────────────────────────────────┘

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of hidden dim to embed dim in FFN
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ====================================================================
        # Layer Norm 1 (before attention)
        # ====================================================================
        self.norm1 = nn.LayerNorm(embed_dim)

        # ====================================================================
        # Attention
        # ====================================================================
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ====================================================================
        # Layer Norm 2 (before FFN)
        # ====================================================================
        self.norm2 = nn.LayerNorm(embed_dim)

        # ====================================================================
        # Feed-Forward Network
        # ====================================================================
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input of shape [B, num_patches, embed_dim]

        Returns:
            Output of same shape
        """
        # ====================================================================
        # Attention block with residual connection
        # ====================================================================
        # Residual: x + attention(norm(x))
        x = x + self.attn(self.norm1(x))

        # ====================================================================
        # FFN block with residual connection
        # ====================================================================
        # Residual: x + ffn(norm(x))
        x = x + self.ffn(self.norm2(x))

        return x


# ============================================================================
# COMPONENT 6: The Complete ViT Model
# ============================================================================

class ViT(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    This combines everything we've built:
        1. PatchEmbedding: Image → Patches
        2. PositionalEmbedding: Add position info
        3. TransformerBlocks: Process patches (x8 blocks)
        4. Classification Head: Patches → Class prediction

    Full pipeline:
        Image [3, 64, 64]
            │
            ▼
        PatchEmbedding → [256, 256]  (256 patches, 256 dims)
            │
            ▼
        PositionalEmbedding → [256, 256]  (add position info)
            │
            ▼
        TransformerBlock 1 → [256, 256]
            │
            ▼
        TransformerBlock 2 → [256, 256]
            │
            ...
            ▼
        TransformerBlock 8 → [256, 256]
            │
            ▼
        Global Average Pool → [256]  (average all patches)
            │
            ▼
        Classification Head → [200]  (200 Tiny ImageNet classes)

    Args:
        img_size (int): Input image size (64 for Tiny ImageNet)
        patch_size (int): Size of each patch (4)
        in_channels (int): Number of input channels (3 for RGB)
        num_classes (int): Number of output classes (200 for Tiny ImageNet)
        embed_dim (int): Embedding dimension (256)
        depth (int): Number of transformer blocks (8)
        num_heads (int): Number of attention heads (8)
        mlp_ratio (float): FFN hidden dim ratio (4.0)
        dropout (float): Dropout probability (0.1)
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 200,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # ====================================================================
        # Step 1: Patch Embedding
        # ====================================================================
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches  # 256

        # ====================================================================
        # Step 2: Positional Embedding
        # ====================================================================
        self.pos_embed = PositionalEmbedding(
            num_patches=num_patches,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # ====================================================================
        # Step 3: Stack of Transformer Blocks
        # ====================================================================
        # We use nn.ModuleList to create 'depth' number of blocks
        # Each block is identical in structure but has its own weights
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # ====================================================================
        # Step 4: Final Layer Norm
        # ====================================================================
        # Applied after all transformer blocks, before classification
        self.norm = nn.LayerNorm(embed_dim)

        # ====================================================================
        # Step 5: Classification Head
        # ====================================================================
        # Takes the pooled features and predicts class probabilities
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire ViT.

        Args:
            x: Input images of shape [B, C, H, W]
               Example: [128, 3, 64, 64]

        Returns:
            Class logits of shape [B, num_classes]
            Example: [128, 200]
        """
        # ====================================================================
        # Step 1: Convert image to patches
        # ====================================================================
        # [B, 3, 64, 64] → [B, 256, 256]
        x = self.patch_embed(x)

        # ====================================================================
        # Step 2: Add positional information
        # ====================================================================
        # [B, 256, 256] → [B, 256, 256] (same shape, but now has position info)
        x = self.pos_embed(x)

        # ====================================================================
        # Step 3: Pass through all Transformer blocks
        # ====================================================================
        # Each block: [B, 256, 256] → [B, 256, 256]
        for block in self.blocks:
            x = block(x)

        # ====================================================================
        # Step 4: Final normalization
        # ====================================================================
        x = self.norm(x)

        # ====================================================================
        # Step 5: Global Average Pooling
        # ====================================================================
        # Average across all patches to get one vector per image
        # [B, 256, 256] → [B, 256]
        #
        # dim=1 means "average across the patches dimension"
        x = x.mean(dim=1)

        # ====================================================================
        # Step 6: Classification
        # ====================================================================
        # [B, 256] → [B, 200]
        x = self.head(x)

        return x
