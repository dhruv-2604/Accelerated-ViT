/**
 * Fused Bias + GELU Activation CUDA Kernel
 *
 * This kernel fuses two operations into a single GPU kernel:
 * 1. Bias addition: x + bias
 * 2. GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * Performance Benefits:
 * - Reduces memory bandwidth (one read/write instead of two)
 * - Eliminates intermediate tensor allocation
 * - Improves L1/L2 cache utilization
 *
 * Expected Speedup: 1.5x - 2.5x over separate PyTorch operations
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU approximation constants
#define SQRT_2_OVER_PI 0.7978845608028654f  // sqrt(2/pi)
#define GELU_COEFF 0.044715f

/**
 * CUDA kernel for fused bias addition and GELU activation.
 *
 * @param output: Output tensor (same shape as input)
 * @param input: Input tensor of shape [..., embedding_dim]
 * @param bias: Bias vector of shape [embedding_dim]
 * @param total_elements: Total number of elements in input tensor
 * @param embedding_dim: Size of the last dimension (for bias broadcasting)
 */
__global__ void fused_bias_gelu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int total_elements,
    int embedding_dim
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // Broadcast bias: bias is 1D, input is multi-dimensional
        // We want bias[i % embedding_dim] to match the last dimension
        int bias_idx = idx % embedding_dim;

        // Step 1: Add bias
        float x = input[idx] + bias[bias_idx];

        // Step 2: Compute GELU using tanh approximation
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x_cubed = x * x * x;
        float tanh_arg = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);

        // Use fast math tanh (enabled by --use_fast_math compiler flag)
        float tanh_val = tanhf(tanh_arg);

        // Final GELU computation
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

/**
 * CUDA kernel launcher (C++ interface).
 * This function is called from the PyTorch C++ binding.
 */
void fused_bias_gelu_cuda_forward(
    float* output,
    const float* input,
    const float* bias,
    int total_elements,
    int embedding_dim
) {
    // Kernel launch configuration
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    fused_bias_gelu_kernel<<<num_blocks, threads_per_block>>>(
        output, input, bias, total_elements, embedding_dim
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in fused_bias_gelu_kernel: %s\n", cudaGetErrorString(err));
    }
}
