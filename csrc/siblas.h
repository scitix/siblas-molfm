#pragma once

#include <torch/torch.h>

// ===== CUTLASS BF16x6 emulated FP32 =====

// Forward: Y = X @ W^T + bias
// X: [M, K], W: [N, K], bias: [N] -> Y: [M, N]
torch::Tensor siblas_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias);

// Backward: given grad_output [M, N]
// grad_input  = grad_output @ W        -> [M, K]
// grad_weight = grad_output^T @ input   -> [N, K]
// grad_bias   = sum(grad_output, dim=0) -> [N]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> siblas_linear_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight);

// ===== cuBLAS TF32 =====

torch::Tensor siblas_cublas_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> siblas_cublas_linear_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight);
