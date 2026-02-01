/**
 * Fused Bias + GELU Activation CUDA Kernel (OPTIMIZED)
 *
 * This kernel fuses two operations into a single GPU kernel:
 * 1. Bias addition: x + bias
 * 2. GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * OPTIMIZATIONS APPLIED:
 * 1. Vectorized memory access (float4) - 4x memory bandwidth
 * 2. Shared memory for bias - reduces global memory reads
 * 3. Loop unrolling - better instruction-level parallelism
 * 4. Increased threads per block (512) - better occupancy on H100
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>

// GELU approximation constants
#define SQRT_2_OVER_PI 0.7978845608028654f  // sqrt(2/pi)
#define GELU_COEFF 0.044715f

// Inline device function for GELU computation
__device__ __forceinline__ float gelu_forward(float x) {
    float x_cubed = x * x * x;
    float tanh_arg = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    float tanh_val = tanhf(tanh_arg);
    return 0.5f * x * (1.0f + tanh_val);
}

/**
 * OPTIMIZED: Vectorized kernel using float4 for 4x memory bandwidth
 * Each thread processes 4 consecutive elements
 */
__global__ void fused_bias_gelu_kernel_vec4(
    float4* __restrict__ output,
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    int total_vec4_elements,
    int embedding_dim
) {
    // Shared memory for bias (reused by all threads in block)
    extern __shared__ float shared_bias[];

    // Cooperatively load bias into shared memory
    for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
        shared_bias[i] = bias[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_vec4_elements) {
        // Load 4 floats at once
        float4 in = input[idx];

        // Calculate base bias index (each float4 spans 4 positions)
        int base_bias_idx = (idx * 4) % embedding_dim;

        // Process 4 elements with unrolled loop
        float4 out;

        // Element 0
        float x0 = in.x + shared_bias[base_bias_idx];
        out.x = gelu_forward(x0);

        // Element 1
        float x1 = in.y + shared_bias[(base_bias_idx + 1) % embedding_dim];
        out.y = gelu_forward(x1);

        // Element 2
        float x2 = in.z + shared_bias[(base_bias_idx + 2) % embedding_dim];
        out.z = gelu_forward(x2);

        // Element 3
        float x3 = in.w + shared_bias[(base_bias_idx + 3) % embedding_dim];
        out.w = gelu_forward(x3);

        // Store 4 floats at once
        output[idx] = out;
    }
}

/**
 * Original scalar kernel (fallback for non-aligned sizes)
 */
__global__ void fused_bias_gelu_kernel_scalar(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int total_elements,
    int embedding_dim,
    int offset  // Starting offset for remainder elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < total_elements) {
        int bias_idx = idx % embedding_dim;
        float x = input[idx] + bias[bias_idx];
        output[idx] = gelu_forward(x);
    }
}

/**
 * CUDA kernel launcher with automatic vectorization
 */
void fused_bias_gelu_cuda_forward(
    float* output,
    const float* input,
    const float* bias,
    int total_elements,
    int embedding_dim
) {
    // Use vectorized kernel for bulk of elements
    int vec4_elements = total_elements / 4;
    int remainder = total_elements % 4;

    if (vec4_elements > 0) {
        // Optimized config for H100
        const int threads_per_block = 512;
        const int num_blocks = (vec4_elements + threads_per_block - 1) / threads_per_block;
        const int shared_mem_size = embedding_dim * sizeof(float);

        fused_bias_gelu_kernel_vec4<<<num_blocks, threads_per_block, shared_mem_size>>>(
            reinterpret_cast<float4*>(output),
            reinterpret_cast<const float4*>(input),
            bias,
            vec4_elements,
            embedding_dim
        );
    }

    // Handle remainder with scalar kernel
    if (remainder > 0) {
        const int threads_per_block = 256;
        const int num_blocks = (remainder + threads_per_block - 1) / threads_per_block;
        int offset = vec4_elements * 4;

        fused_bias_gelu_kernel_scalar<<<num_blocks, threads_per_block>>>(
            output, input, bias, total_elements, embedding_dim, offset
        );
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in fused_bias_gelu_kernel: %s\n", cudaGetErrorString(err));
    }
}
