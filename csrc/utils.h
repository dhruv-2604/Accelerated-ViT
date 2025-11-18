/**
 * CUDA Utility Macros for Error Checking and Tensor Validation
 *
 * These macros are critical for debugging CUDA kernels and ensuring
 * PyTorch tensors meet the requirements for custom operations.
 */

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Tensor Validation Macros
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// Helper macro for getting tensor data pointer with type safety
#define TENSOR_DATA_PTR(tensor, type) tensor.data_ptr<type>()
