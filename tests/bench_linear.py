"""
Benchmark: SiblasLinear (CUTLASS BF16x6) vs CublasLinear (TF32) vs nn.Linear

Usage:
    python tests/bench_linear.py
"""

import torch
import torch.utils.benchmark as benchmark
from siblas import SiblasLinear, CublasLinear


def bench_forward(M, K, N, device="cuda"):
    """Benchmark forward pass for different backends."""
    x = torch.randn(M, K, device=device, dtype=torch.float32)

    # ---- SiblasLinear (CUTLASS BF16x6) ----
    siblas_layer = SiblasLinear(K, N, bias=True, device=device)

    # ---- CublasLinear (cuBLAS TF32) ----
    cublas_layer = CublasLinear(K, N, bias=True, device=device)
    with torch.no_grad():
        cublas_layer.weight.copy_(siblas_layer.weight)
        cublas_layer.bias.copy_(siblas_layer.bias)

    # ---- nn.Linear ----
    torch_layer = torch.nn.Linear(K, N, bias=True, device=device, dtype=torch.float32)
    with torch.no_grad():
        torch_layer.weight.copy_(siblas_layer.weight)
        torch_layer.bias.copy_(siblas_layer.bias)

    results = []

    # Siblas (BF16x6)
    t = benchmark.Timer(
        stmt="layer(x)",
        globals={"layer": siblas_layer, "x": x},
        label="Forward",
        sub_label=f"M={M}, K={K}, N={N}",
        description="siblas (BF16x6)",
    )
    results.append(t.blocked_autorange(min_run_time=1.0))

    # CublasLinear (TF32)
    t = benchmark.Timer(
        stmt="layer(x)",
        globals={"layer": cublas_layer, "x": x},
        label="Forward",
        sub_label=f"M={M}, K={K}, N={N}",
        description="cublas (TF32)",
    )
    results.append(t.blocked_autorange(min_run_time=1.0))

    # nn.Linear FP32 (TF32 off)
    torch.backends.cuda.matmul.allow_tf32 = False
    t = benchmark.Timer(
        stmt="layer(x)",
        globals={"layer": torch_layer, "x": x},
        label="Forward",
        sub_label=f"M={M}, K={K}, N={N}",
        description="nn.Linear (FP32)",
    )
    results.append(t.blocked_autorange(min_run_time=1.0))

    # nn.Linear TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    t = benchmark.Timer(
        stmt="layer(x)",
        globals={"layer": torch_layer, "x": x},
        label="Forward",
        sub_label=f"M={M}, K={K}, N={N}",
        description="nn.Linear (TF32)",
    )
    results.append(t.blocked_autorange(min_run_time=1.0))
    torch.backends.cuda.matmul.allow_tf32 = False

    return results


def bench_forward_backward(M, K, N, device="cuda"):
    """Benchmark forward + backward pass for different backends."""
    results = []

    # ---- SiblasLinear (CUTLASS BF16x6) ----
    siblas_layer = SiblasLinear(K, N, bias=True, device=device)

    # ---- CublasLinear (cuBLAS TF32) ----
    cublas_layer = CublasLinear(K, N, bias=True, device=device)
    with torch.no_grad():
        cublas_layer.weight.copy_(siblas_layer.weight)
        cublas_layer.bias.copy_(siblas_layer.bias)

    # ---- nn.Linear ----
    torch_layer = torch.nn.Linear(K, N, bias=True, device=device, dtype=torch.float32)
    with torch.no_grad():
        torch_layer.weight.copy_(siblas_layer.weight)
        torch_layer.bias.copy_(siblas_layer.bias)

    def fwd_bwd(layer, x):
        out = layer(x)
        out.sum().backward()

    # Siblas (BF16x6)
    t = benchmark.Timer(
        stmt="fwd_bwd(layer, torch.randn(M, K, device=device, dtype=torch.float32))",
        globals={"fwd_bwd": fwd_bwd, "layer": siblas_layer, "M": M, "K": K,
                 "device": device, "torch": torch},
        label="Fwd+Bwd",
        sub_label=f"M={M}, K={K}, N={N}",
        description="siblas (BF16x6)",
    )
    results.append(t.blocked_autorange(min_run_time=1.0))

    # CublasLinear (TF32)
    t = benchmark.Timer(
        stmt="fwd_bwd(layer, torch.randn(M, K, device=device, dtype=torch.float32))",
        globals={"fwd_bwd": fwd_bwd, "layer": cublas_layer, "M": M, "K": K,
                 "device": device, "torch": torch},
        label="Fwd+Bwd",
        sub_label=f"M={M}, K={K}, N={N}",
        description="cublas (TF32)",
    )
    results.append(t.blocked_autorange(min_run_time=1.0))

    # nn.Linear FP32
    torch.backends.cuda.matmul.allow_tf32 = False
    t = benchmark.Timer(
        stmt="fwd_bwd(layer, torch.randn(M, K, device=device, dtype=torch.float32))",
        globals={"fwd_bwd": fwd_bwd, "layer": torch_layer, "M": M, "K": K,
                 "device": device, "torch": torch},
        label="Fwd+Bwd",
        sub_label=f"M={M}, K={K}, N={N}",
        description="nn.Linear (FP32)",
    )
    results.append(t.blocked_autorange(min_run_time=1.0))

    # nn.Linear TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    t = benchmark.Timer(
        stmt="fwd_bwd(layer, torch.randn(M, K, device=device, dtype=torch.float32))",
        globals={"fwd_bwd": fwd_bwd, "layer": torch_layer, "M": M, "K": K,
                 "device": device, "torch": torch},
        label="Fwd+Bwd",
        sub_label=f"M={M}, K={K}, N={N}",
        description="nn.Linear (TF32)",
    )
    results.append(t.blocked_autorange(min_run_time=1.0))
    torch.backends.cuda.matmul.allow_tf32 = False

    return results


def main():
    print("=" * 80)
    print("Benchmark: SiblasLinear vs CublasLinear vs nn.Linear")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print()

    # Problem sizes: (M, K, N)
    # aligned M (multiples of 256)
    sizes = [
        (10000, 256, 768),
        (10000, 256, 512),
        (10000, 256, 256),
        (10000, 256, 128),
        (20000, 256, 768),
        (20000, 256, 512),
        (20000, 256, 256),
        (20000, 256, 128),
        (30000, 256, 768),
        (30000, 256, 512),
        (30000, 256, 256),
        (30000, 256, 128),
        (40000, 256, 768),
        (40000, 256, 512),
        (40000, 256, 256),
        (40000, 256, 128),
        # unaligned M (not a power-of-2 multiple)
        (10001, 256, 256),
        (10100, 256, 256),
        (10333, 256, 768),
        (20001, 256, 512),
        (20777, 256, 256),
        (30100, 256, 768),
        (39999, 256, 256),
    ]

    # ---- Forward benchmark ----
    fwd_results = []
    for M, K, N in sizes:
        print(f"  Benchmarking forward  M={M}, K={K}, N={N} ...")
        fwd_results.extend(bench_forward(M, K, N))

    print()
    compare = benchmark.Compare(fwd_results)
    compare.print()

    # ---- Forward + Backward benchmark ----
    print()
    bwd_results = []
    for M, K, N in sizes:
        print(f"  Benchmarking fwd+bwd  M={M}, K={K}, N={N} ...")
        bwd_results.extend(bench_forward_backward(M, K, N))

    print()
    compare = benchmark.Compare(bwd_results)
    compare.print()


if __name__ == "__main__":
    main()
