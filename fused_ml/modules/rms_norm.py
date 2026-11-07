"""RMSNorm module: host-side launch code and autograd wiring.

The Triton kernel itself lives in ``fused_ml.ops.rms_norm``.
"""

import torch
import triton

from fused_ml.ops import _rms_norm_fwd_kernel

def _to_2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Flatten a 1D/2D/3D tensor to 2D ``[B, H]``.

    Args:
        x: Input tensor of shape ``[H]``, ``[B, H]`` or ``[B, S, H]``.

    Returns:
        tuple[torch.Tensor, torch.Size]: A contiguous 2D view of shape
            ``[B, H]`` (``B = 1`` for 1D input, ``B = B * S`` for 3D input) and
            the original input shape.

    Raises:
        ValueError: If ``x`` is not 1D, 2D or 3D.
    """
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim != 2:
        raise ValueError(f"input must be 1D-3D, got {x.ndim}D")
    return x.contiguous(), orig_shape

def _rms_norm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Launch the RMSNorm forward kernel.

    Args:
        x: Input tensor of shape ``[H]``, ``[B, H]`` or ``[B, S, H]``.
        weight: Per-feature gain of shape ``[H]``.
        eps: Epsilon added to the mean square before the reciprocal square root.

    Returns:
        torch.Tensor: Normalized tensor with the same shape and dtype as ``x``.

    Raises:
        ValueError: If ``weight`` is not 1D or its length does not match the
            input hidden dimension.
    """
    x2d, orig_shape = _to_2d(x)

    if weight.ndim != 1:
        raise ValueError(f"weight must be 1D, got {tuple(weight.shape)}")
    if weight.shape[0] != x2d.shape[1]:
        raise ValueError(
            f"hidden dim mismatch: input H={x2d.shape[1]}, weight H={weight.shape[0]}"
        )

    weight = weight.contiguous()
    B, H = x2d.shape
    y = torch.empty_like(x2d)

    BLOCK_H = min(1024, triton.next_power_of_2(H))
    _rms_norm_fwd_kernel[(B,)](
        x2d,
        weight,
        y,
        stride_x_b=x2d.stride(0),
        stride_x_h=x2d.stride(1),
        stride_y_b=y.stride(0),
        stride_y_h=y.stride(1),
        H=H,
        eps=eps,
        BLOCK_H=BLOCK_H,
        num_warps=8 if H >= 4096 else 4,
        num_stages=2,
    )

    return y.reshape(orig_shape)

class _RMSNormFunction(torch.autograd.Function):
    """Autograd wrapper for :func:`rms_norm`."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Compute the RMSNorm forward pass and cache inputs for backward.

        Args:
            ctx: Autograd context used to stash tensors and ``eps``.
            x: Input tensor of shape ``[H]``, ``[B, H]`` or ``[B, S, H]``.
            weight: Per-feature gain of shape ``[H]``.
            eps: Epsilon added to the mean square before the reciprocal square root.

        Returns:
            torch.Tensor: Normalized tensor with the same shape and dtype as ``x``.
        """
        y = _rms_norm_forward(x, weight, eps)
        x2d, _ = _to_2d(x)
        ctx.save_for_backward(x2d, weight)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Compute gradients for the input and the gain.

        Args:
            ctx: Autograd context holding the cached ``x``, ``weight`` and ``eps``.
            grad_out: Upstream gradient with the same shape as the forward output.

        Returns:
            tuple[torch.Tensor, torch.Tensor, None]: Gradients for ``x``,
            ``weight`` and ``eps`` (which has no gradient).
        """
        x, weight = ctx.saved_tensors
        g, _ = _to_2d(grad_out)
        H = x.shape[-1]

        x_f = x.float()
        g_f = g.float()
        w_f = weight.float()

        rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + ctx.eps)
        # dx = rstd * w * dy - (rstd^3 / H) * x * sum(dy * w * x)
        gx = g_f * rstd * w_f
        proj = (g_f * w_f * x_f).sum(dim=-1, keepdim=True)
        gx = gx - (rstd.pow(3) / H) * x_f * proj
        gw = (g_f * x_f * rstd).sum(dim=0)
        return gx.to(grad_out.dtype).reshape_as(grad_out), gw.to(weight.dtype), None

def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Root-mean-square normalization with a learned per-feature gain.

    Args:
        x: Input tensor of shape ``[H]``, ``[B, H]`` or ``[B, S, H]``. A 3D
            input is flattened to ``[B*S, H]`` and reshaped back before returning.
        weight: Per-feature gain of shape ``[H]``.
        eps: Epsilon added to the mean square before the reciprocal square root.
            Defaults to ``1e-6``.

    Returns:
        torch.Tensor: Normalized tensor with the same shape and dtype as ``x``.
    """
    return _RMSNormFunction.apply(x, weight, eps)
