/**
 * PyTorch C++ Binding for Fused Bias-GELU CUDA Kernel
 *
 * This file provides the interface between Python/PyTorch and the CUDA kernel.
 * It handles:
 * - Tensor validation (contiguity, device placement)
 * - Memory allocation for output tensors
 * - Kernel invocation
 */

#include <torch/extension.h>
#include "utils.h"

// Forward declaration of the CUDA kernel launcher
void fused_bias_gelu_cuda_forward(
    float* output,
    const float* input,
    const float* bias,
    int total_elements,
    int embedding_dim
);

/**
 * PyTorch-facing function for fused bias-GELU operation.
 *
 * @param input: Input tensor of shape [batch, seq_len, embedding_dim]
 * @param bias: Bias tensor of shape [embedding_dim]
 * @return: Output tensor (same shape as input) with GELU(input + bias) applied
 */
torch::Tensor fused_bias_gelu_cuda(torch::Tensor input, torch::Tensor bias) {
    // ===== INPUT VALIDATION =====
    CHECK_INPUT(input);
    CHECK_INPUT(bias);

    // Ensure both tensors are float32 (our kernel only supports this)
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");

    // Validate dimensions
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1-dimensional");

    // Get embedding dimension (last dimension of input)
    int embedding_dim = input.size(-1);
    TORCH_CHECK(
        bias.size(0) == embedding_dim,
        "Bias size must match last dimension of input. Got bias size: ",
        bias.size(0), ", expected: ", embedding_dim
    );

    // ===== MEMORY ALLOCATION =====
    // Allocate output tensor with same shape as input
    auto output = torch::empty_like(input);

    // Calculate total number of elements
    int total_elements = input.numel();

    // ===== KERNEL LAUNCH =====
    fused_bias_gelu_cuda_forward(
        TENSOR_DATA_PTR(output, float),
        TENSOR_DATA_PTR(input, float),
        TENSOR_DATA_PTR(bias, float),
        total_elements,
        embedding_dim
    );

    // Synchronize to catch any kernel errors (only in debug builds)
    #ifdef DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif

    return output;
}

// ===== PYBIND11 MODULE DEFINITION =====
// This makes the function callable from Python as: import vit_ops; vit_ops.fused_bias_gelu(x, b)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_bias_gelu",
        &fused_bias_gelu_cuda,
        "Fused Bias + GELU activation (CUDA implementation)",
        py::arg("input"),
        py::arg("bias")
    );
}
