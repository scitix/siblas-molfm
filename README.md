# siblas

CUTLASS BF16x6 emulated FP32 GEMM for NVIDIA Blackwell (SM100 / B200).

The forward GEMM (`Y = X @ W^T + bias`) uses the `MainloopSm100TmaUmmaWarpSpecializedFastF32PersistentB` mainloop with `NumBands=3, MaxPBTiles=8`, which achieves higher accuracy than TF32 while remaining fully FP32 compatible from the PyTorch API perspective. Backward GEMMs (NT for `grad_input`, RR for `grad_weight`) use the standard `KernelTmaWarpSpecialized2SmFastFP32Sm100` mainloop.

## Requirements

- NVIDIA Blackwell GPU (SM100, e.g. B200)
- CUDA 12.8
- PyTorch >= 2.0 with CUDA 12.8
- CUTLASS v4.4.1 (at `/volume/code/chengjiajun/gemm_bf16x9/third_party/cutlass`)

## Build & Install

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8

pip install -e . --no-build-isolation --break-system-packages
```

> If your environment reports CUDA 13.x but PyTorch was built with CUDA 12.8, you **must** set the PATH above before building to avoid a version mismatch error.

## API

```python
from siblas import SiblasLinear, linear_forward

# Drop-in replacement for nn.Linear (float32 weights, BF16x6 forward kernel)
layer = SiblasLinear(in_features=256, out_features=256, bias=True, device="cuda")
y = layer(x)          # supports 2D and batched (≥3D) inputs
loss = y.sum()
loss.backward()       # gradients computed correctly

# Functional interface
y = linear_forward(x, weight, bias)   # bias may be empty: torch.empty(0)
```

## Precision Tests

Run all correctness tests:

```bash
python tests/test_linear.py
```

The test suite uses **FP64 ground truth** throughout (all reference computations are upcast to `float64`). Global tolerance: `ATOL=5e-3`, `RTOL=1e-3`.

| Test | Description | Problem size |
|------|-------------|--------------|
| 1 | Forward with bias vs FP64 GT | M=128, N=256, K=256 |
| 2 | Forward without bias vs FP64 GT | M=64, N=256, K=256 |
| 3 | Backward (`grad_input`, `grad_weight`, `grad_bias`) vs FP64 GT | M=32, N=256, K=256 |
| 4 | `SiblasLinear` module API + output shape check | N=K=256 |
| 5 | Batched 3D input `(4, 8, K)` → `(4, 8, N)` | N=K=256 |
| 6 | `SiblasLinear` vs `nn.Linear` with shared weights, FP64 reference | M=64, N=256, K=256 |
| 7 | Precision comparison: BF16x6 vs TF32 vs FP32, all vs FP64 GT (forward + backward) | M=128, N=256, K=256 |

Test 7 is informational (no hard assertion on TF32/FP32 rows) and prints a table like:

```
[Forward] max_abs / mean_abs vs FP64 ground truth:
  BF16x6 (siblas):  max=...  mean=...
  TF32   (PyTorch): max=...  mean=...
  FP32   (PyTorch): max=...  mean=...

[Backward] grad_input / grad_weight / grad_bias max_abs vs FP64:
  BF16x6 (siblas):
    grad_input : max=...  mean=...
    grad_weight: max=...  mean=...
    grad_bias  : max=...  mean=...
  ...
```

## Performance Benchmarks

Run the benchmark suite:

```bash
python tests/bench_linear.py
```

The benchmark compares four backends across a range of problem sizes:

| Backend | Description |
|---------|-------------|
| `siblas (BF16x6)` | `SiblasLinear` — CUTLASS BF16x6 emulated FP32, forward via `OptPolicy` |
| `cublas (TF32)` | `CublasLinear` — cuBLAS GemmEx with TF32 compute |
| `nn.Linear (FP32)` | PyTorch `nn.Linear` with TF32 disabled (`allow_tf32=False`) |
| `nn.Linear (TF32)` | PyTorch `nn.Linear` with TF32 enabled |

Problem sizes tested — aligned M (multiples of 256) and unaligned M:

```
# aligned
(M=10000, K=256, N=768)   (M=20000, K=256, N=768)   (M=30000, K=256, N=768)   (M=40000, K=256, N=768)
(M=10000, K=256, N=512)   (M=20000, K=256, N=512)   (M=30000, K=256, N=512)   (M=40000, K=256, N=512)
(M=10000, K=256, N=256)   (M=20000, K=256, N=256)   (M=30000, K=256, N=256)   (M=40000, K=256, N=256)
(M=10000, K=256, N=128)   (M=20000, K=256, N=128)   (M=30000, K=256, N=128)   (M=40000, K=256, N=128)

# unaligned (M not a power-of-2 multiple)
(M=10001, K=256, N=256)   (M=10100, K=256, N=256)   (M=10333, K=256, N=768)
(M=20001, K=256, N=512)   (M=20777, K=256, N=256)
(M=30100, K=256, N=768)   (M=39999, K=256, N=256)
```

Two benchmark phases are reported:

- **Forward only** — `layer(x)` latency
- **Forward + Backward** — `layer(x)` then `.sum().backward()` latency

Results are printed as a `torch.utils.benchmark.Compare` table with median latency per backend per size.

## Implementation Notes

- **Forward (NN layout)**: `A=RowMajor (X)`, `B=ColMajor (W^T)`, output `ColumnMajor` → transposed back to row-major before returning. Uses `LinCombPerColBias` epilogue (bias fused into GEMM).
- **Backward NT** (`grad_input = grad_output @ W`): `A=ColMajor`, `B=RowMajor`, standard TF32 mainloop.
- **Backward RR** (`grad_weight = grad_output^T @ X`): `A=RowMajor`, `B=RowMajor`, standard TF32 mainloop.
- The BF16x6 `OptPolicy` is only applicable to the NN (forward) layout combination; CUTLASS does not provide a `FastF32NoSmemWarpSpecialized2Sm` epilogue specialization for the NT/RR cases at the chosen tile shape `<256,64,32>`.
