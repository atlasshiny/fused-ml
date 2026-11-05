"""Standalone RMSNorm Triton kernel.

The host-side launch code lives in ``fused_ml.modules.rms_norm``.
"""

import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_kernel(
    x_ptr,           # Pointer to input tensor [B, H]
    w_ptr,           # Pointer to RMSNorm gain vector [H]
    y_ptr,           # Pointer to normalized output [B, H]
    stride_x_b,      # Batch stride for input
    stride_x_h,      # Hidden dimension stride for input
    stride_y_b,      # Batch stride for output
    stride_y_h,      # Hidden dimension stride for output
    H,               # Hidden dimension size
    eps,             # Epsilon added to the mean square before rsqrt
    BLOCK_H: tl.constexpr,  # Tile size along H
):
    """RMSNorm forward: ``y = x * rsqrt(mean(x^2) + eps) * w``.

    One program per batch row. The sum of squares is accumulated in FP32
    registers over tiled passes of ``H``, then the scaled output is written in
    a second tiled pass.

    Args:
        x_ptr: Pointer to the input tensor of shape ``[B, H]``.
        w_ptr: Pointer to the gain vector of shape ``[H]``.
        y_ptr: Pointer to the output tensor of shape ``[B, H]``.
        stride_x_b: Stride between consecutive batch rows of ``x``.
        stride_x_h: Stride between consecutive elements along ``H`` of ``x``.
        stride_y_b: Stride between consecutive batch rows of ``y``.
        stride_y_h: Stride between consecutive elements along ``H`` of ``y``.
        H: Hidden dimension size. Passed at runtime to avoid recompilation
            across architectures.
        eps: Epsilon added to the mean square before the reciprocal square root.
        BLOCK_H: Number of hidden-dimension elements processed per tile.
    """
    bid = tl.program_id(axis=0)

    x_ptr_batch = x_ptr + bid * stride_x_b
    y_ptr_batch = y_ptr + bid * stride_y_b

    # First pass: accumulate sum of squares in FP32.
    var_acc = tl.zeros((), dtype=tl.float32)
    for h_off in range(0, H, BLOCK_H):
        cols = h_off + tl.arange(0, BLOCK_H)
        mask_h = cols < H
        x = tl.load(x_ptr_batch + cols * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)
        var_acc += tl.sum(x * x, axis=0)

    rstd = tl.rsqrt((var_acc / H) + eps)

    # Second pass: apply normalization and gain, then store.
    for h_off in range(0, H, BLOCK_H):
        cols = h_off + tl.arange(0, BLOCK_H)
        mask_h = cols < H
        x = tl.load(x_ptr_batch + cols * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask_h, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(
            y_ptr_batch + cols * stride_y_h,
            y.to(y_ptr.dtype.element_ty),
            mask=mask_h,
        )
