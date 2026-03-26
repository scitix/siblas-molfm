import math

import torch
import torch.nn as nn

from .ops import linear_forward, cublas_linear_forward


class _BaseLinear(nn.Module):
    """
    Shared base for SiblasLinear and CublasLinear.
    Interface is identical to ``nn.Linear(in_features, out_features, bias)``.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    # Subclasses must set this
    _forward_fn = None  # type: ignore[assignment]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": torch.float32}

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.size(-1) != self.in_features:
            raise ValueError(
                f"Expected input last dim = {self.in_features}, "
                f"got {input.size(-1)}"
            )

        # Flatten batch dimensions: [..., in_features] -> [M, in_features]
        orig_shape = input.shape
        flat_input = input.reshape(-1, self.in_features)

        bias = (
            self.bias
            if self.bias is not None
            else torch.empty(0, device=input.device, dtype=input.dtype)
        )
        output = self._forward_fn(flat_input, self.weight, bias)

        # Restore batch dimensions: [M, out_features] -> [..., out_features]
        output_shape = orig_shape[:-1] + (self.out_features,)
        return output.reshape(output_shape)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class SiblasLinear(_BaseLinear):
    """
    Drop-in replacement for :class:`torch.nn.Linear` that uses
    CUTLASS BF16×6 emulated FP32 GEMM on Blackwell (SM100) GPUs.

    Supports autograd (backward pass via CUTLASS as well).
    """

    _forward_fn = staticmethod(linear_forward)


class CublasLinear(_BaseLinear):
    """
    Drop-in replacement for :class:`torch.nn.Linear` that uses
    cuBLAS with TF32 Tensor Core acceleration.

    Supports autograd (backward pass via cuBLAS TF32 as well).
    """

    _forward_fn = staticmethod(cublas_linear_forward)
