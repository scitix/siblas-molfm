"""
Test script for siblas linear layer.
Verifies:
  1. Forward correctness against torch.nn.functional.linear
  2. Backward correctness via manual comparison
  3. SiblasLinear module end-to-end
  4. Batched input support
"""

import torch
import torch.nn.functional as F
from siblas import SiblasLinear, linear_forward


def test_forward_correctness():
    """Compare siblas forward with torch.nn.functional.linear (TF32 enabled)."""
    print("=" * 60)
    print("Test 1: Forward correctness")
    print("=" * 60)

    # Enable TF32 in PyTorch for fair comparison
    torch.backends.cuda.matmul.allow_tf32 = True

    torch.manual_seed(42)
    M, N, K = 128, 256, 256

    X = torch.randn(M, K, device="cuda", dtype=torch.float32)
    W = torch.randn(N, K, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)

    # Reference (PyTorch with TF32)
    ref = F.linear(X, W, b)

    # Our implementation
    out = linear_forward(X, W, b)

    # Both use TF32, should match closely
    diff = (out - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    print(f"  Max absolute error: {max_err:.6e}")
    print(f"  Mean absolute error: {mean_err:.6e}")

    assert max_err < 1e-3, f"Forward error too large: max_abs={max_err}"
    print("  PASSED\n")


def test_forward_no_bias():
    """Forward without bias."""
    print("=" * 60)
    print("Test 2: Forward without bias")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.manual_seed(42)
    M, N, K = 64, 256, 256

    X = torch.randn(M, K, device="cuda", dtype=torch.float32)
    W = torch.randn(N, K, device="cuda", dtype=torch.float32)
    empty_bias = torch.empty(0, device="cuda", dtype=torch.float32)

    ref = F.linear(X, W, None)
    out = linear_forward(X, W, empty_bias)

    diff = (out - ref).abs()
    max_err = diff.max().item()
    print(f"  Max absolute error: {max_err:.6e}")
    assert max_err < 1e-3, f"Forward (no bias) error too large: {max_err}"
    print("  PASSED\n")


def test_backward_correctness():
    """Check gradients via manual comparison."""
    print("=" * 60)
    print("Test 3: Backward correctness")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.manual_seed(42)
    M, N, K = 32, 256, 256

    X = torch.randn(M, K, device="cuda", dtype=torch.float32, requires_grad=True)
    W = torch.randn(N, K, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)

    # Reference
    X_ref = X.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    ref_out = F.linear(X_ref, W_ref, b_ref)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)

    # Ours
    our_out = linear_forward(X, W, b)
    our_out.backward(grad_out)

    # Compare gradients (both TF32, should match closely)
    for name, ours, theirs in [
        ("grad_input", X.grad, X_ref.grad),
        ("grad_weight", W.grad, W_ref.grad),
        ("grad_bias", b.grad, b_ref.grad),
    ]:
        diff = (ours - theirs).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        print(f"  {name}: max_abs={max_err:.6e}, mean_abs={mean_err:.6e}")
        assert max_err < 1e-3, f"{name} gradient error too large: max_abs={max_err}"

    print("  PASSED\n")


def test_module():
    """Test SiblasLinear as an nn.Module."""
    print("=" * 60)
    print("Test 4: SiblasLinear module")
    print("=" * 60)

    layer = SiblasLinear(bias=True, device="cuda")
    print(f"  Module: {layer}")

    X = torch.randn(16, 256, device="cuda", dtype=torch.float32)
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
    X = torch.randn(4, 8, 256, device="cuda", dtype=torch.float32)
    Y = layer(X)
    assert Y.shape == (4, 8, 256)

    loss = Y.sum()
    loss.backward()
    assert layer.weight.grad is not None
    print(f"  Input:  {X.shape}")
    print(f"  Output: {Y.shape}")
    print("  PASSED\n")


def test_against_nn_linear():
    """Compare SiblasLinear against nn.Linear with shared weights."""
    print("=" * 60)
    print("Test 6: SiblasLinear vs nn.Linear")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)

    M, N, K = 64, 256, 256

    # Create both layers
    siblas_layer = SiblasLinear(bias=True, device="cuda")
    torch_layer = torch.nn.Linear(K, N, bias=True, device="cuda", dtype=torch.float32)

    # Copy weights from siblas to torch layer
    with torch.no_grad():
        torch_layer.weight.copy_(siblas_layer.weight)
        torch_layer.bias.copy_(siblas_layer.bias)

    # Same input
    X = torch.randn(M, K, device="cuda", dtype=torch.float32)

    # Forward
    X_siblas = X.clone().requires_grad_(True)
    X_torch = X.clone().requires_grad_(True)

    out_siblas = siblas_layer(X_siblas)
    out_torch = torch_layer(X_torch)

    # Compare forward
    fwd_diff = (out_siblas - out_torch).abs()
    fwd_max = fwd_diff.max().item()
    fwd_mean = fwd_diff.mean().item()
    print(f"  Forward  max_abs={fwd_max:.6e}, mean_abs={fwd_mean:.6e}")
    assert fwd_max < 1e-3, f"Forward mismatch: {fwd_max}"

    # Backward with same grad
    grad_out = torch.randn_like(out_siblas)
    out_siblas.backward(grad_out)
    out_torch.backward(grad_out)

    # Compare gradients
    for name, s_grad, t_grad in [
        ("grad_input", X_siblas.grad, X_torch.grad),
        ("grad_weight", siblas_layer.weight.grad, torch_layer.weight.grad),
        ("grad_bias", siblas_layer.bias.grad, torch_layer.bias.grad),
    ]:
        diff = (s_grad - t_grad).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        print(f"  {name}: max_abs={max_err:.6e}, mean_abs={mean_err:.6e}")
        assert max_err < 1e-3, f"{name} mismatch: {max_err}"

    print("  PASSED\n")


if __name__ == "__main__":
    print("siblas test suite")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.get_device_name(0)}")
    print()

    test_forward_correctness()
    test_forward_no_bias()
    test_backward_correctness()
    test_module()
    test_batched()
    test_against_nn_linear()

    print("All tests passed!")
