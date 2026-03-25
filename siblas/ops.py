import torch
import siblas._C  # noqa: F401


def linear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Custom linear forward: Y = X @ W^T + bias
    Uses cuBLAS with TF32 acceleration internally.

    Args:
        input:  [M, 256] float32 CUDA tensor
        weight: [256, 256] float32 CUDA tensor
        bias:   [256] float32 CUDA tensor (pass empty tensor to skip)

    Returns:
        output: [M, 256] float32 CUDA tensor
    """
    return torch.ops.siblas.linear_forward(input, weight, bias)
