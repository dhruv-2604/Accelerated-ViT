"""
Fused Bias-GELU Layer with Automatic Fallback

This module demonstrates a key engineering pattern in ML systems:
"Optimize when possible, degrade gracefully when not"

The layer will:
1. Try to use custom CUDA kernel (fast path)
2. Fall back to PyTorch operations (compatibility path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# STEP 1: Try to import the custom CUDA extension
# ============================================================================
# This happens at module import time (when Python first loads this file)

HAS_CUSTOM_OPS = False
try:
    import vit_ops
    HAS_CUSTOM_OPS = True
    print("✅ Custom CUDA kernels (vit_ops) loaded successfully")
except ImportError:
    print("⚠️  Custom CUDA kernels not available - using PyTorch fallback")
    print("    This is expected on CPU-only machines (e.g., macOS without NVIDIA GPU)")


# ============================================================================
# STEP 2: Define the PyTorch Layer
# ============================================================================

class FusedBiasGELU(nn.Module):
    """
    Fused Bias + GELU activation layer.

    Mathematical Operation:
        output = GELU(input + bias)

    Where:
        - input: shape [..., embedding_dim]
        - bias: shape [embedding_dim] (learnable parameter)
        - GELU: Gaussian Error Linear Unit activation

    Performance Notes:
        - Custom CUDA: ~1.5-2.5x faster than PyTorch (measured via benchmark.py)
        - Fallback: Same speed as nn.Linear + nn.GELU (no performance penalty)

    Args:
        embedding_dim (int): Size of the last dimension of input tensors
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Learnable bias parameter (initialized to zeros)
        # This is what nn.Linear would create internally
        self.bias = nn.Parameter(torch.zeros(embedding_dim))

        # Store whether custom ops are available (for logging/debugging)
        self.using_custom_ops = HAS_CUSTOM_OPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic backend selection.

        The decision tree:
        1. Is vit_ops available? (Was CUDA extension compiled?)
        2. Is input tensor on GPU? (x.is_cuda)

        If BOTH are true → use custom kernel
        Otherwise → use PyTorch fallback

        Args:
            x: Input tensor of shape [batch, seq_len, embedding_dim]
               (or any shape ending in embedding_dim)

        Returns:
            Output tensor (same shape as input) with GELU(x + bias) applied
        """

        # ====================================================================
        # PATH 1: Custom CUDA Kernel (Fast Path)
        # ====================================================================
        if self.using_custom_ops and x.is_cuda:
            # CRITICAL: Ensure tensor is contiguous in memory
            # Non-contiguous tensors will crash the CUDA kernel!
            # This call is cheap if already contiguous (no copy)
            x = x.contiguous()

            # Call our custom C++/CUDA function
            # This maps to: csrc/activation.cpp -> fused_bias_gelu_cuda()
            return vit_ops.fused_bias_gelu(x, self.bias)

        # ====================================================================
        # PATH 2: PyTorch Fallback (Compatibility Path)
        # ====================================================================
        # This path executes when:
        # - Running on CPU (Mac development)
        # - CUDA extension wasn't compiled
        # - Input tensor is on CPU for some reason

        return F.gelu(x + self.bias)

    def extra_repr(self) -> str:
        """
        Provides helpful debug information when you print the model.

        Example output:
            FusedBiasGELU(embedding_dim=256, backend=CUDA)
        """
        backend = "CUDA" if self.using_custom_ops else "PyTorch"
        return f"embedding_dim={self.embedding_dim}, backend={backend}"
