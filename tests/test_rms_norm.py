import pytest
import torch

from fused_ml import rms_norm

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="RMSNorm Triton kernel requires CUDA"
)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Mixed 1D/2D/3D shapes, a non-power-of-two hidden dim, and the BLOCK_H boundary.
SHAPES = [
    (128,),
    (4, 128),
    (2, 8, 128),
    (3, 127),
    (2, 1024),
    (2, 2048),
]

def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Naive FP32 reference implementation."""
    x_f = x.float()
    rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rstd * weight.float()).to(x.dtype)

def _tolerance(dtype: torch.dtype) -> dict[str, float]:
    if dtype is torch.float32:
        return {"rtol": 1e-4, "atol": 1e-5}
    if dtype is torch.float16:
        return {"rtol": 2e-3, "atol": 2e-3}
    return {"rtol": 2e-2, "atol": 2e-2}  # bfloat16

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@requires_cuda
def test_forward_matches_reference(shape, dtype):
    torch.manual_seed(0)
    hidden = shape[-1]
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)

    y = rms_norm(x, weight)

    assert y.shape == x.shape
    assert y.dtype == dtype
    assert y.device == x.device
    torch.testing.assert_close(y, _reference(x, weight), **_tolerance(dtype))

@pytest.mark.parametrize("dtype", DTYPES)
@requires_cuda
def test_forward_zero_input_is_finite(dtype):
    torch.manual_seed(0)
    x = torch.zeros((4, 256), device="cuda", dtype=dtype)
    weight = torch.randn(256, device="cuda", dtype=dtype)

    y = rms_norm(x, weight)

    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.zeros_like(y))

@pytest.mark.parametrize("dtype", DTYPES)
@requires_cuda
def test_backward_matches_reference(dtype):
    torch.manual_seed(0)
    batch, hidden = 8, 256
    x = torch.randn(batch, hidden, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden, device="cuda", dtype=dtype, requires_grad=True)
    grad_out = torch.randn_like(x)

    y = rms_norm(x, weight)
    y.backward(grad_out)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    _reference(x_ref, weight_ref).backward(grad_out)

    assert x.grad is not None and weight.grad is not None
    assert x.grad.dtype == dtype
    assert weight.grad.dtype == dtype

    tol = _tolerance(dtype)
    torch.testing.assert_close(x.grad, x_ref.grad, **tol)
    torch.testing.assert_close(weight.grad, weight_ref.grad, **tol)

@pytest.mark.parametrize("dtype", DTYPES)
@requires_cuda
def test_backward_non_contiguous_input(dtype):
    torch.manual_seed(0)
    batch, seq, hidden = 4, 8, 128
    # [seq, batch, hidden] -> [batch, seq, hidden], non-contiguous.
    x = torch.randn(seq, batch, hidden, device="cuda", dtype=dtype).transpose(0, 1)
    assert not x.is_contiguous()
    x.requires_grad_(True)
    weight = torch.randn(hidden, device="cuda", dtype=dtype, requires_grad=True)
    grad_out = torch.randn_like(x)

    y = rms_norm(x, weight)
    y.backward(grad_out)

    x_ref = x.detach().contiguous().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    _reference(x_ref, weight_ref).backward(grad_out.contiguous())

    tol = _tolerance(dtype)
    torch.testing.assert_close(x.grad, x_ref.grad, **tol)
    torch.testing.assert_close(weight.grad, weight_ref.grad, **tol)

@requires_cuda
def test_float64_input_matches_reference():
    """The kernel accumulates in FP32 and stores back in the input dtype."""
    x = torch.randn((4, 128), device="cuda", dtype=torch.float64)
    weight = torch.randn(128, device="cuda", dtype=torch.float64)

    y = rms_norm(x, weight)

    assert y.dtype == torch.float64
    torch.testing.assert_close(y, _reference(x, weight), rtol=1e-4, atol=1e-5)

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@requires_cuda
def test_backward_large_hidden_dim_multiple_tiles(dtype):
    """H larger than BLOCK_H exercises the tiled dx/dw paths."""
    torch.manual_seed(0)
    batch, hidden = 8, 4096
    x = torch.randn(batch, hidden, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden, device="cuda", dtype=dtype, requires_grad=True)
    grad_out = torch.randn_like(x)

    y = rms_norm(x, weight)
    y.backward(grad_out)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    _reference(x_ref, weight_ref).backward(grad_out)

    tol = _tolerance(dtype)
    torch.testing.assert_close(x.grad, x_ref.grad, **tol)
    torch.testing.assert_close(weight.grad, weight_ref.grad, **tol)


@requires_cuda
def test_backward_many_rows_reduces_partials():
    """B > 256 gives more than one row per dW program and several partials."""
    torch.manual_seed(0)
    batch, hidden = 1024, 256
    x = torch.randn(batch, hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn_like(x)

    y = rms_norm(x, weight)
    y.backward(grad_out)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    _reference(x_ref, weight_ref).backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_backward_hidden_dim_at_fused_limit():
    """H=8192 still goes through the fused single-pass backward."""
    torch.manual_seed(0)
    batch, hidden = 4, 8192
    x = torch.randn(batch, hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn_like(x)

    y = rms_norm(x, weight)
    y.backward(grad_out)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    _reference(x_ref, weight_ref).backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_backward_hidden_dim_above_fused_limit():
    """H>8192 falls back to the tiled dx + dw kernels."""
    torch.manual_seed(0)
    batch, hidden = 2, 16384
    x = torch.randn(batch, hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn_like(x)

    y = rms_norm(x, weight)
    y.backward(grad_out)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    _reference(x_ref, weight_ref).backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_backward_3d_input():
    torch.manual_seed(0)
    seq, batch, hidden = 4, 8, 512
    x = torch.randn(seq, batch, hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn_like(x)

    y = rms_norm(x, weight)
    y.backward(grad_out)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    _reference(x_ref, weight_ref).backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_backward_skips_weight_grad_when_not_needed():
    """dW is not computed when the gain is frozen."""
    torch.manual_seed(0)
    x = torch.randn((16, 256), device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(256, device="cuda", dtype=torch.float32, requires_grad=False)
    grad_out = torch.randn_like(x)

    y = rms_norm(x, weight)
    y.backward(grad_out)

    assert weight.grad is None
    x_ref = x.detach().clone().requires_grad_(True)
    _reference(x_ref, weight).backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)


def test_rejects_invalid_input_rank():
    x = torch.randn((2, 2, 2, 2))
    weight = torch.randn(2)
    with pytest.raises(ValueError, match="1D-3D"):
        rms_norm(x, weight)

def test_rejects_non_vector_weight():
    x = torch.randn((4, 8))
    weight = torch.randn((8, 1))
    with pytest.raises(ValueError, match="weight must be 1D"):
        rms_norm(x, weight)

def test_rejects_hidden_dim_mismatch():
    x = torch.randn((4, 8))
    weight = torch.randn(16)
    with pytest.raises(ValueError, match="hidden dim mismatch"):
        rms_norm(x, weight)
