import torch
import torch.nn as nn

from .ops import linear_forward


class SiblasLinear(nn.Module):
    """
    Linear layer that uses cuBLAS GEMM with TF32 acceleration.
    Fixed N=256, K=256 (in_features=256, out_features=256).

    Supports autograd (backward pass via cuBLAS as well).
    """

    N: int = 256
    K: int = 256

    def __init__(self, bias: bool = True, device=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": torch.float32}

        self.weight = nn.Parameter(
            torch.empty(self.N, self.K, **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.N, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.K
            bound = 1.0 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.size(-1) == self.K, (
            f"Expected input last dim = {self.K}, got {input.size(-1)}"
        )

        # Flatten batch dimensions: [..., K] -> [M, K]
        orig_shape = input.shape
        flat_input = input.reshape(-1, self.K)

        bias = self.bias if self.bias is not None else torch.empty(0, device=input.device, dtype=input.dtype)
        output = linear_forward(flat_input, self.weight, bias)

        # Restore batch dimensions: [M, N] -> [..., N]
        output_shape = orig_shape[:-1] + (self.N,)
        return output.reshape(output_shape)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.K}, out_features={self.N}, "
            f"bias={self.bias is not None}"
        )
