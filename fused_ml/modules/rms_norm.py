"""RMSNorm module: host-side launch code and autograd wiring.

The Triton kernels themselves live in ``fused_ml.ops.rms_norm``.
"""

import torch
import triton

from fused_ml.ops import (
    _rms_norm_bwd_dw_kernel,
    _rms_norm_bwd_dx_kernel,
    _rms_norm_bwd_fused_kernel,
    _rms_norm_fwd_kernel,
)

_MAX_PARTIAL_BLOCKS = 256

# The fused backward keeps one full row of x/dy plus the dW accumulator in
# registers, so it is only used while the row still fits comfortably.
_MAX_FUSED_H = 8192

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the RMSNorm forward kernel.

    Args:
        x: Input tensor of shape ``[H]``, ``[B, H]`` or ``[B, S, H]``.
        weight: Per-feature gain of shape ``[H]``.
        eps: Epsilon added to the mean square before the reciprocal square root.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The normalized tensor with the same
        shape and dtype as ``x``, and the FP32 reciprocal std of shape ``[B]``.

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
    rstd = torch.empty((B,), device=x2d.device, dtype=torch.float32)

    BLOCK_H = min(1024, triton.next_power_of_2(H))
    SINGLE_PASS = H <= BLOCK_H
    _rms_norm_fwd_kernel[(B,)](
        x2d,
        weight,
        y,
        rstd,
        stride_x_b=x2d.stride(0),
        stride_x_h=x2d.stride(1),
        stride_y_b=y.stride(0),
        stride_y_h=y.stride(1),
        H=H,
        eps=eps,
        BLOCK_H=BLOCK_H,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=8 if H >= 4096 else 4,
        num_stages=1 if SINGLE_PASS else 2,
    )

    return y.reshape(orig_shape), rstd

class _RMSNormFunction(torch.autograd.Function):
    """Autograd wrapper for :func:`rms_norm`.

    Backward is implemented with Triton kernels for both the input and the gain
    gradient. Double backward is not supported.
    """

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
        y, rstd = _rms_norm_forward(x, weight, eps)
        x2d, _ = _to_2d(x)
        # rstd is computed once here, so backward never re-reads x for the
        # reduction. weight is detached: it is only needed as a multiply factor.
        ctx.save_for_backward(x2d, weight.contiguous(), rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Compute gradients for the input and the gain with Triton kernels.

        Args:
            ctx: Autograd context holding the cached ``x``, ``weight`` and ``rstd``.
            grad_out: Upstream gradient with the same shape as the forward output.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, None]: Gradients for ``x``,
            ``weight`` (``None`` when the gain does not require grad) and ``eps``
            (which has no gradient).
        """
        x, weight, rstd = ctx.saved_tensors
        g, _ = _to_2d(grad_out)
        B, H = x.shape

        dx = torch.empty_like(x)
        needs_weight_grad = ctx.needs_input_grad[1]
        grad_weight = None

        # Fast path: one kernel produces dx and the dW partials from a single
        # read of x/dy. The whole row has to fit in one tile for that.
        if needs_weight_grad and H <= _MAX_FUSED_H:
            BLOCK_H = triton.next_power_of_2(H)
            num_warps = 4 if BLOCK_H <= 4096 else 8
            rows_per_program = max(1, triton.cdiv(B, _MAX_PARTIAL_BLOCKS))
            num_row_blocks = triton.cdiv(B, rows_per_program)

            partials = torch.empty(
                (num_row_blocks, H), device=x.device, dtype=torch.float32
            )
            _rms_norm_bwd_fused_kernel[(num_row_blocks,)](
                x,
                weight,
                g,
                dx,
                rstd,
                partials,
                stride_x_b=x.stride(0),
                stride_x_h=x.stride(1),
                stride_dy_b=g.stride(0),
                stride_dy_h=g.stride(1),
                stride_dx_b=dx.stride(0),
                stride_dx_h=dx.stride(1),
                stride_dw_b=partials.stride(0),
                stride_dw_h=partials.stride(1),
                B=B,
                H=H,
                ROWS_PER_PROGRAM=rows_per_program,
                BLOCK_H=BLOCK_H,
                num_warps=num_warps,
                num_stages=1,
            )
            grad_weight = partials.sum(dim=0).to(weight.dtype)
        else:
            # Fallback: large rows stream in tiles, and a frozen gain skips dW.
            BLOCK_H = min(1024, triton.next_power_of_2(H))
            SINGLE_PASS = H <= BLOCK_H
            num_warps = 8 if H >= 4096 else 4

            _rms_norm_bwd_dx_kernel[(B,)](
                x,
                weight,
                g,
                dx,
                rstd,
                stride_x_b=x.stride(0),
                stride_x_h=x.stride(1),
                stride_dy_b=g.stride(0),
                stride_dy_h=g.stride(1),
                stride_dx_b=dx.stride(0),
                stride_dx_h=dx.stride(1),
                H=H,
                BLOCK_H=BLOCK_H,
                SINGLE_PASS=SINGLE_PASS,
                num_warps=num_warps,
                num_stages=1 if SINGLE_PASS else 2,
            )

            if needs_weight_grad:
                rows_per_program = max(1, triton.cdiv(B, _MAX_PARTIAL_BLOCKS))
                num_row_blocks = triton.cdiv(B, rows_per_program)
                num_col_blocks = triton.cdiv(H, BLOCK_H)

                partials = torch.empty(
                    (num_row_blocks, H), device=x.device, dtype=torch.float32
                )
                _rms_norm_bwd_dw_kernel[(num_col_blocks, num_row_blocks)](
                    x,
                    g,
                    rstd,
                    partials,
                    stride_x_b=x.stride(0),
                    stride_x_h=x.stride(1),
                    stride_dy_b=g.stride(0),
                    stride_dy_h=g.stride(1),
                    stride_dw_b=partials.stride(0),
                    stride_dw_h=partials.stride(1),
                    B=B,
                    H=H,
                    ROWS_PER_PROGRAM=rows_per_program,
                    BLOCK_H=BLOCK_H,
                    num_warps=num_warps,
                    num_stages=1,
                )
                grad_weight = partials.sum(dim=0).to(weight.dtype)

        return (
            dx.to(grad_out.dtype).reshape_as(grad_out),
            grad_weight,
            None,
        )

def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Root-mean-square normalization with a learned per-feature gain.

    Forward and backward are both Triton kernels. The input gradient and the
    gain gradient are computed in two separate kernels, with FP32 accumulation.

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
