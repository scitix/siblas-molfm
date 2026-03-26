import torch
import siblas._C  # noqa: F401


def linear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Custom linear forward: Y = X @ W^T + bias
    Uses CUTLASS BF16×6 emulated FP32 GEMM internally.

    Args:
        input:  [M, K] float32 CUDA tensor
        weight: [N, K] float32 CUDA tensor
        bias:   [N] float32 CUDA tensor (pass empty tensor to skip)

    Returns:
        output: [M, N] float32 CUDA tensor
    """
    return torch.ops.siblas.linear_forward(input, weight, bias)


def cublas_linear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Custom linear forward: Y = X @ W^T + bias
    Uses cuBLAS with TF32 Tensor Core acceleration internally.

    Args:
        input:  [M, K] float32 CUDA tensor
        weight: [N, K] float32 CUDA tensor
        bias:   [N] float32 CUDA tensor (pass empty tensor to skip)

    Returns:
        output: [M, N] float32 CUDA tensor
    """
    return torch.ops.siblas.cublas_linear_forward(input, weight, bias)
