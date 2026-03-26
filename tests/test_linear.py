"""
Test script for siblas linear layer.
Verifies:
  1. Forward correctness against torch.nn.functional.linear (FP64 Ground Truth)
  2. Backward correctness via manual comparison (FP64 Ground Truth)
  3. SiblasLinear module end-to-end
  4. Batched input support
"""

import torch
import torch.nn.functional as F
from siblas import SiblasLinear, linear_forward

# 全局关闭 TF32，确保基准测试的绝对精度
torch.backends.cuda.matmul.allow_tf32 = False

# 针对 BF16x6 (Emulated FP32) 近似计算设定的合理容差阈值
ATOL = 5e-3
RTOL = 1e-3

def test_forward_correctness():
    """Compare siblas forward (FP32) with PyTorch FP64 Ground Truth."""
    print("=" * 60)
    print("Test 1: Forward correctness (vs FP64 GT)")
    print("=" * 60)

    torch.manual_seed(42)
    M, N, K = 128, 256, 256

    # 1. 我们的算子必须接收真实的 FP32 数据
    X = torch.randn(M, K, device="cuda", dtype=torch.float32) - 0.5
    W = torch.randn(N, K, device="cuda", dtype=torch.float32) - 0.5
    b = torch.randn(N, device="cuda", dtype=torch.float32) - 0.5

    # 2. Reference: 升维到 FP64 计算 Ground Truth，再切回 FP32
    ref_fp64 = F.linear(X.double(), W.double(), b.double())
    ref = ref_fp64.float()

    # 3. Our implementation: 直接运行在 FP32 (底层 BF16x6)
    out = linear_forward(X, W, b)

    diff = (out - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    print(f"  Max absolute error: {max_err:.6e}")
    print(f"  Mean absolute error: {mean_err:.6e}")

    assert max_err < ATOL, f"Forward error too large: max_abs={max_err}"
    print("  PASSED\n")


def test_forward_no_bias():
    """Forward without bias compared to FP64 Ground Truth."""
    print("=" * 60)
    print("Test 2: Forward without bias (vs FP64 GT)")
    print("=" * 60)

    torch.manual_seed(42)
    M, N, K = 64, 256, 256

    X = torch.rand(M, K, device="cuda", dtype=torch.float32) - 0.5
    W = torch.rand(N, K, device="cuda", dtype=torch.float32) - 0.5
    empty_bias = torch.empty(0, device="cuda", dtype=torch.float32)

    # Reference in FP64
    ref_fp64 = F.linear(X.double(), W.double(), None)
    ref = ref_fp64.float()
    
    # Ours in FP32
    out = linear_forward(X, W, empty_bias)

    diff = (out - ref).abs()
    max_err = diff.max().item()
    print(f"  Max absolute error: {max_err:.6e}")
    assert max_err < ATOL, f"Forward (no bias) error too large: {max_err}"
    print("  PASSED\n")


def test_backward_correctness():
    """Check gradients via manual comparison against FP64 backward pass."""
    print("=" * 60)
    print("Test 3: Backward correctness (vs FP64 GT)")
    print("=" * 60)

    torch.manual_seed(42)
    M, N, K = 32, 256, 256

    X = (torch.rand(M, K, device="cuda", dtype=torch.float32) - 0.5).requires_grad_(True)
    W = (torch.rand(N, K, device="cuda", dtype=torch.float32) - 0.5).requires_grad_(True)
    b = (torch.rand(N, device="cuda", dtype=torch.float32) - 0.5).requires_grad_(True)

    # Reference Tensors (FP64)
    X_ref = X.detach().clone().double().requires_grad_(True)
    W_ref = W.detach().clone().double().requires_grad_(True)
    b_ref = b.detach().clone().double().requires_grad_(True)

    # Forward in FP64
    ref_out = F.linear(X_ref, W_ref, b_ref)
    
    # Generate same random grad_out in FP32, then clone to FP64
    grad_out_fp32 = torch.rand(M, N, device=X.device, dtype=torch.float32) - 0.5
    grad_out_fp64 = grad_out_fp32.double()
    
    # Backward in FP64
    ref_out.backward(grad_out_fp64)

    # Ours: Forward & Backward in FP32
    our_out = linear_forward(X, W, b)
    our_out.backward(grad_out_fp32)

    # Compare gradients (Ours vs Ref casted back to FP32)
    for name, ours, theirs_fp64 in [
        ("grad_input", X.grad, X_ref.grad),
        ("grad_weight", W.grad, W_ref.grad),
        ("grad_bias", b.grad, b_ref.grad),
    ]:
        theirs = theirs_fp64.float()
        diff = (ours - theirs).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        print(f"  {name}: max_abs={max_err:.6e}, mean_abs={mean_err:.6e}")
        assert max_err < ATOL, f"{name} gradient error too large: max_abs={max_err}"

    print("  PASSED\n")


def test_module():
    """Test SiblasLinear as an nn.Module (API & Shapes check)."""
    print("=" * 60)
    print("Test 4: SiblasLinear module API check")
    print("=" * 60)

    layer = SiblasLinear(bias=True, device="cuda")
    print(f"  Module: {layer}")

    X = torch.rand(16, 256, device="cuda", dtype=torch.float32) - 0.5
    Y = layer(X)
    print(f"  Input:  {X.shape}")
    print(f"  Output: {Y.shape}")
    assert Y.shape == (16, 256)

    # Test backward through module
    loss = Y.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert layer.weight.grad.shape == (256, 256)
    assert layer.bias.grad.shape == (256,)
    print("  Gradients computed successfully")
    print("  PASSED\n")


def test_batched():
    """Test with multi-dimensional batch input."""
    print("=" * 60)
    print("Test 5: Batched input (3D)")
    print("=" * 60)

    layer = SiblasLinear(bias=True, device="cuda")
    X = torch.rand(4, 8, 256, device="cuda", dtype=torch.float32) - 0.5
    Y = layer(X)
    assert Y.shape == (4, 8, 256)

    loss = Y.sum()
    loss.backward()
    assert layer.weight.grad is not None
    print(f"  Input:  {X.shape}")
    print(f"  Output: {Y.shape}")
    print("  PASSED\n")


def test_against_nn_linear():
    """Compare SiblasLinear against nn.Linear with shared weights (vs FP64)."""
    print("=" * 60)
    print("Test 6: SiblasLinear vs nn.Linear (FP64)")
    print("=" * 60)

    torch.manual_seed(42)
    M, N, K = 64, 256, 256

    # Create both layers
    # siblas in FP32, torch_layer in FP64
    siblas_layer = SiblasLinear(bias=True, device="cuda")
    torch_layer = torch.nn.Linear(K, N, bias=True, device="cuda", dtype=torch.float64)

    # Copy weights from siblas to torch layer (and cast to double)
    with torch.no_grad():
        torch_layer.weight.copy_(siblas_layer.weight.double())
        torch_layer.bias.copy_(siblas_layer.bias.double())

    # Create same input
    X_fp32 = torch.rand(M, K, device="cuda", dtype=torch.float32) - 0.5

    # Forward
    X_siblas = X_fp32.clone().requires_grad_(True)
    X_torch = X_fp32.clone().double().requires_grad_(True)

    out_siblas = siblas_layer(X_siblas)
    out_torch_fp64 = torch_layer(X_torch)
    out_torch = out_torch_fp64.float()

    # Compare forward
    fwd_diff = (out_siblas - out_torch).abs()
    fwd_max = fwd_diff.max().item()
    fwd_mean = fwd_diff.mean().item()
    print(f"  Forward  max_abs={fwd_max:.6e}, mean_abs={fwd_mean:.6e}")
    assert fwd_max < ATOL, f"Forward mismatch: {fwd_max}"

    # Backward with same grad
    grad_out_fp32 = torch.rand_like(out_siblas) - 0.5
    grad_out_fp64 = grad_out_fp32.double()
    
    out_siblas.backward(grad_out_fp32)
    out_torch_fp64.backward(grad_out_fp64)

    # Compare gradients
    for name, s_grad, t_grad_fp64 in [
        ("grad_input", X_siblas.grad, X_torch.grad),
        ("grad_weight", siblas_layer.weight.grad, torch_layer.weight.grad),
        ("grad_bias", siblas_layer.bias.grad, torch_layer.bias.grad),
    ]:
        t_grad = t_grad_fp64.float()
        diff = (s_grad - t_grad).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        print(f"  {name}: max_abs={max_err:.6e}, mean_abs={mean_err:.6e}")
        assert max_err < ATOL, f"{name} mismatch: {max_err}"

    print("  PASSED\n")


def test_precision_comparison():
    """Compare siblas (BF16x6) vs TF32 vs FP32, all against FP64 ground truth."""
    print("=" * 60)
    print("Test 7: Precision comparison (BF16x6 vs TF32 vs FP32) against FP64")
    print("=" * 60)

    torch.manual_seed(42)
    M, N, K = 128, 256, 256

    X = torch.randn(M, K, device="cuda", dtype=torch.float32)
    W = torch.randn(N, K, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)

    # ---- Ground truth: FP64 ----
    ref_fp64 = F.linear(X.double(), W.double(), b.double())
    ref = ref_fp64.float()

    # ---- siblas (BF16x6 emulated FP32) ----
    out_siblas = linear_forward(X, W, b)

    # ---- PyTorch FP32 (exact, TF32 disabled globally) ----
    out_fp32 = F.linear(X, W, b)

    # ---- PyTorch TF32 ----
    torch.backends.cuda.matmul.allow_tf32 = True
    out_tf32 = F.linear(X, W, b)
    torch.backends.cuda.matmul.allow_tf32 = False  # restore

    # ---- Forward comparison ----
    print("\n  [Forward] max_abs / mean_abs vs FP64 ground truth:")
    for label, out in [
        ("BF16x6 (siblas)", out_siblas),
        ("TF32   (PyTorch)", out_tf32),
        ("FP32   (PyTorch)", out_fp32),
    ]:
        diff = (out - ref).abs()
        print(f"    {label}:  max={diff.max().item():.6e}  mean={diff.mean().item():.6e}")

    # ---- Backward comparison ----
    print("\n  [Backward] grad_input / grad_weight / grad_bias max_abs vs FP64:")

    grad_out_fp32 = torch.randn(M, N, device="cuda", dtype=torch.float32)
    grad_out_fp64 = grad_out_fp32.double()

    # FP64 reference backward
    X_ref = X.double().requires_grad_(True)
    W_ref = W.double().requires_grad_(True)
    b_ref = b.double().requires_grad_(True)
    F.linear(X_ref, W_ref, b_ref).backward(grad_out_fp64)

    def run_backward(label, forward_fn):
        x = X.clone().requires_grad_(True)
        w = W.clone().requires_grad_(True)
        bb = b.clone().requires_grad_(True)
        out = forward_fn(x, w, bb)
        out.backward(grad_out_fp32)
        print(f"    {label}:")
        for gname, ours, theirs_fp64 in [
            ("grad_input ", x.grad, X_ref.grad),
            ("grad_weight", w.grad, W_ref.grad),
            ("grad_bias  ", bb.grad, b_ref.grad),
        ]:
            diff = (ours - theirs_fp64.float()).abs()
            print(f"      {gname}:  max={diff.max().item():.6e}  mean={diff.mean().item():.6e}")

    # siblas backward
    run_backward("BF16x6 (siblas)", lambda x, w, bb: linear_forward(x, w, bb))

    # TF32 backward
    def tf32_linear(x, w, bb):
        torch.backends.cuda.matmul.allow_tf32 = True
        out = F.linear(x, w, bb)
        torch.backends.cuda.matmul.allow_tf32 = False
        return out

    run_backward("TF32   (PyTorch)", tf32_linear)

    # FP32 backward
    run_backward("FP32   (PyTorch)", lambda x, w, bb: F.linear(x, w, bb))

    print("\n  DONE\n")


if __name__ == "__main__":
    print("siblas test suite (FP64 Ground Truth Mode)")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.get_device_name(0)}")
    print(f"TF32:    {torch.backends.cuda.matmul.allow_tf32} (Forced Disabled)")
    print()

    test_forward_correctness()
    test_forward_no_bias()
    test_backward_correctness()
    test_module()
    test_batched()
    test_against_nn_linear()
    test_precision_comparison()

    print("All tests passed!")