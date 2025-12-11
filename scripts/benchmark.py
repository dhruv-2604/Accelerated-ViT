"""
Benchmark: Custom CUDA Kernel vs Standard PyTorch

This script measures the speedup of our fused Bias+GELU kernel.

Usage:
    python scripts/benchmark.py

Run on a GPU node (PACE ICE) for accurate results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Run this on a GPU node.")
    print("On PACE: salloc --gres=gpu:1 -t 00:30:00")
    exit(1)

# Try to import custom ops
try:
    import vit_ops
    HAS_CUSTOM_OPS = True
    print("✅ Custom CUDA kernel loaded")
except ImportError:
    HAS_CUSTOM_OPS = False
    print("❌ Custom CUDA kernel not available")
    print("   Run: pip install -e .")
    exit(1)


# =============================================================================
# Benchmark Configuration
# =============================================================================

DEVICE = torch.device("cuda")
BATCH_SIZES = [32, 64, 128, 256]
SEQ_LEN = 256      # Number of patches in ViT
EMBED_DIM = 256    # Embedding dimension
NUM_WARMUP = 10    # Warmup iterations
NUM_TRIALS = 100   # Timed iterations


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_pytorch(x, bias):
    """Standard PyTorch: separate bias + GELU."""
    return F.gelu(x + bias)


def benchmark_custom(x, bias):
    """Our fused CUDA kernel."""
    return vit_ops.fused_bias_gelu(x, bias)


def run_benchmark(func, x, bias, num_warmup, num_trials):
    """
    Run benchmark with CUDA events for accurate timing.

    CUDA events are more accurate than Python time.time()
    because they measure GPU time, not CPU time.
    """
    # Warmup (let GPU reach steady state)
    for _ in range(num_warmup):
        _ = func(x, bias)

    torch.cuda.synchronize()

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Timed runs
    start_event.record()
    for _ in range(num_trials):
        _ = func(x, bias)
    end_event.record()

    torch.cuda.synchronize()

    # Return average time in milliseconds
    total_time_ms = start_event.elapsed_time(end_event)
    return total_time_ms / num_trials


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("FusedBiasGELU Benchmark: Custom CUDA vs PyTorch")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Embedding Dim: {EMBED_DIM}")
    print(f"Warmup: {NUM_WARMUP} | Trials: {NUM_TRIALS}")
    print("=" * 70)

    # Table header
    print(f"\n{'Batch':<8} {'PyTorch (ms)':<15} {'Custom (ms)':<15} {'Speedup':<10}")
    print("-" * 50)

    results = []

    for batch_size in BATCH_SIZES:
        # Create input tensors
        x = torch.randn(batch_size, SEQ_LEN, EMBED_DIM, device=DEVICE)
        bias = torch.randn(EMBED_DIM, device=DEVICE)

        # Ensure contiguous (required for custom kernel)
        x = x.contiguous()

        # Benchmark PyTorch
        pytorch_time = run_benchmark(benchmark_pytorch, x, bias, NUM_WARMUP, NUM_TRIALS)

        # Benchmark Custom
        custom_time = run_benchmark(benchmark_custom, x, bias, NUM_WARMUP, NUM_TRIALS)

        # Calculate speedup
        speedup = pytorch_time / custom_time

        # Print results
        print(f"{batch_size:<8} {pytorch_time:<15.4f} {custom_time:<15.4f} {speedup:<10.2f}x")

        results.append({
            'batch_size': batch_size,
            'pytorch_ms': pytorch_time,
            'custom_ms': custom_time,
            'speedup': speedup,
        })

    # Summary
    print("-" * 50)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")

    # Verify correctness
    print("\n" + "=" * 70)
    print("Correctness Check")
    print("=" * 70)

    x = torch.randn(4, SEQ_LEN, EMBED_DIM, device=DEVICE)
    bias = torch.randn(EMBED_DIM, device=DEVICE)

    pytorch_out = benchmark_pytorch(x, bias)
    custom_out = benchmark_custom(x.contiguous(), bias)

    max_diff = (pytorch_out - custom_out).abs().max().item()
    print(f"Max absolute difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("✅ Results match!")
    else:
        print("⚠️  Results differ (may be due to floating point precision)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
