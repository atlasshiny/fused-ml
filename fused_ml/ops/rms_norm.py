"""Standalone RMSNorm Triton kernel.

The host-side launch code lives in ``fused_ml.modules.rms_norm``.
"""

import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_kernel(
    x_ptr, # Pointer to input tensor [B, H]
    w_ptr, # Pointer to RMSNorm gain vector [H]
    y_ptr, # Pointer to normalized output [B, H]
    rstd_ptr, # Pointer to per-row reciprocal std, FP32 [B]
    stride_x_b, # Batch stride for input
    stride_x_h, # Hidden dimension stride for input
    stride_y_b, # Batch stride for output
    stride_y_h, # Hidden dimension stride for output
    H, # Hidden dimension size
    eps, # Epsilon added to the mean square before rsqrt
    BLOCK_H: tl.constexpr, # Tile size along H
    SINGLE_PASS: tl.constexpr, # True when the whole row fits in one tile
):
    """RMSNorm forward: ``y = x * rstd * w``, ``rstd = rsqrt(mean(x^2) + eps)``.

    One program per batch row. ``rstd`` is written out in FP32 so the backward
    pass does not have to recompute the sum of squares.

    Args:
        x_ptr: Pointer to the input tensor of shape ``[B, H]``.
        w_ptr: Pointer to the gain vector of shape ``[H]``.
        y_ptr: Pointer to the output tensor of shape ``[B, H]``.
        rstd_ptr: Pointer to the FP32 reciprocal-std buffer of shape ``[B]``.
        stride_x_b: Stride between consecutive batch rows of ``x``.
        stride_x_h: Stride between consecutive elements along ``H`` of ``x``.
        stride_y_b: Stride between consecutive batch rows of ``y``.
        stride_y_h: Stride between consecutive elements along ``H`` of ``y``.
        H: Hidden dimension size. Passed at runtime to avoid recompilation
            across architectures.
        eps: Epsilon added to the mean square before the reciprocal square root.
        BLOCK_H: Number of hidden-dimension elements processed per tile.
        SINGLE_PASS: When true the row fits in ``BLOCK_H`` and each element is
            loaded/stored once instead of being streamed twice.
    """
    bid = tl.program_id(axis=0)

    x_ptr_batch = x_ptr + bid * stride_x_b
    y_ptr_batch = y_ptr + bid * stride_y_b

    if SINGLE_PASS:
        # Fast path: the entire row fits in registers, so load it once and
        # reuse it for both the reduction and the scaled store.
        cols = tl.arange(0, BLOCK_H)
        mask_h = cols < H
        x = tl.load(x_ptr_batch + cols * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask_h, other=0.0).to(tl.float32)

        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
        tl.store(rstd_ptr + bid, rstd)

        y = x * rstd * w
        tl.store(
            y_ptr_batch + cols * stride_y_h,
            y.to(y_ptr.dtype.element_ty),
            mask=mask_h,
        )
    else:
        # Large rows: stream twice so register pressure stays bounded.
        var_acc = tl.zeros((), dtype=tl.float32)
        for h_off in range(0, H, BLOCK_H):
            cols = h_off + tl.arange(0, BLOCK_H)
            mask_h = cols < H
            x = tl.load(x_ptr_batch + cols * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)
            var_acc += tl.sum(x * x, axis=0)

        rstd = tl.rsqrt((var_acc / H) + eps)
        tl.store(rstd_ptr + bid, rstd)

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


@triton.jit
def _rms_norm_bwd_dx_kernel(
    x_ptr, # Pointer to input tensor [B, H]
    w_ptr, # Pointer to gain vector [H]
    dy_ptr, # Pointer to upstream gradient [B, H]
    dx_ptr, # Pointer to input gradient [B, H]
    rstd_ptr, # Pointer to cached FP32 reciprocal std [B]
    stride_x_b,
    stride_x_h,
    stride_dy_b,
    stride_dy_h,
    stride_dx_b,
    stride_dx_h,
    H,
    BLOCK_H: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    """Input gradient: ``dx = rstd*dy*w - (rstd^3/H) * x * sum(dy*w*x)``.

    One program per batch row. The row-wise dot product ``sum(dy*w*x)`` is
    accumulated in FP32 over tiled passes, then a second pass writes ``dx``.

    Args:
        x_ptr: Pointer to the forward input of shape ``[B, H]``.
        w_ptr: Pointer to the gain vector of shape ``[H]``.
        dy_ptr: Pointer to the upstream gradient of shape ``[B, H]``.
        dx_ptr: Pointer to the input gradient buffer of shape ``[B, H]``.
        rstd_ptr: Pointer to the cached FP32 reciprocal std of shape ``[B]``.
        stride_x_b: Batch stride of ``x``.
        stride_x_h: Hidden stride of ``x``.
        stride_dy_b: Batch stride of ``dy``.
        stride_dy_h: Hidden stride of ``dy``.
        stride_dx_b: Batch stride of ``dx``.
        stride_dx_h: Hidden stride of ``dx``.
        H: Hidden dimension size.
        BLOCK_H: Number of hidden-dimension elements processed per tile.
        SINGLE_PASS: When true the row fits in ``BLOCK_H`` and is loaded once.
    """
    bid = tl.program_id(axis=0)

    x_ptr_batch = x_ptr + bid * stride_x_b
    dy_ptr_batch = dy_ptr + bid * stride_dy_b
    dx_ptr_batch = dx_ptr + bid * stride_dx_b
    rstd = tl.load(rstd_ptr + bid)

    if SINGLE_PASS:
        cols = tl.arange(0, BLOCK_H)
        mask_h = cols < H
        x = tl.load(x_ptr_batch + cols * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask_h, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr_batch + cols * stride_dy_h, mask=mask_h, other=0.0).to(tl.float32)

        proj = tl.sum(dy * w * x, axis=0)
        dx = rstd * dy * w - (rstd * rstd * rstd / H) * x * proj
        tl.store(
            dx_ptr_batch + cols * stride_dx_h,
            dx.to(dx_ptr.dtype.element_ty),
            mask=mask_h,
        )
    else:
        proj_acc = tl.zeros((), dtype=tl.float32)
        for h_off in range(0, H, BLOCK_H):
            cols = h_off + tl.arange(0, BLOCK_H)
            mask_h = cols < H
            x = tl.load(x_ptr_batch + cols * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=mask_h, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr_batch + cols * stride_dy_h, mask=mask_h, other=0.0).to(tl.float32)
            proj_acc += tl.sum(dy * w * x, axis=0)

        scale = rstd * rstd * rstd / H
        for h_off in range(0, H, BLOCK_H):
            cols = h_off + tl.arange(0, BLOCK_H)
            mask_h = cols < H
            x = tl.load(x_ptr_batch + cols * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=mask_h, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr_batch + cols * stride_dy_h, mask=mask_h, other=0.0).to(tl.float32)
            dx = rstd * dy * w - scale * x * proj_acc
            tl.store(
                dx_ptr_batch + cols * stride_dx_h,
                dx.to(dx_ptr.dtype.element_ty),
                mask=mask_h,
            )

@triton.jit
def _rms_norm_bwd_dw_kernel(
    x_ptr, # Pointer to input tensor [B, H]
    dy_ptr, # Pointer to upstream gradient [B, H]
    rstd_ptr, # Pointer to cached FP32 reciprocal std [B]
    dw_ptr, # Pointer to partial weight gradient [N_BLOCKS, H], FP32
    stride_x_b,
    stride_x_h,
    stride_dy_b,
    stride_dy_h,
    stride_dw_b,
    stride_dw_h,
    B, # Number of rows
    H,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Partial weight gradient: ``dw[cols] = sum_rows(dy * x * rstd)``.

    Grid is ``(num_col_blocks, num_row_blocks)``. Each program owns one column
    tile and a contiguous block of rows, accumulating in FP32 registers and
    writing one partial row. The host reduces the partials deterministically.

    Args:
        x_ptr: Pointer to the forward input of shape ``[B, H]``.
        dy_ptr: Pointer to the upstream gradient of shape ``[B, H]``.
        rstd_ptr: Pointer to the cached FP32 reciprocal std of shape ``[B]``.
        dw_ptr: Pointer to the FP32 partial buffer of shape ``[N_BLOCKS, H]``.
        stride_x_b: Batch stride of ``x``.
        stride_x_h: Hidden stride of ``x``.
        stride_dy_b: Batch stride of ``dy``.
        stride_dy_h: Hidden stride of ``dy``.
        stride_dw_b: Row stride of the partial buffer.
        stride_dw_h: Column stride of the partial buffer.
        B: Number of rows in the flattened input.
        H: Hidden dimension size.
        ROWS_PER_PROGRAM: Rows accumulated per program.
        BLOCK_H: Number of hidden-dimension elements handled per program.
    """
    pid_h = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    row_start = pid_b * ROWS_PER_PROGRAM
    row_end = tl.minimum(row_start + ROWS_PER_PROGRAM, B)

    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cols < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for row in range(row_start, row_end):
        row64 = row.to(tl.int64)
        x = tl.load(
            x_ptr + row64 * stride_x_b + cols * stride_x_h, mask=mask_h, other=0.0
        ).to(tl.float32)
        dy = tl.load(
            dy_ptr + row64 * stride_dy_b + cols * stride_dy_h, mask=mask_h, other=0.0
        ).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)
        acc += dy * x * rstd

    tl.store(
        dw_ptr + pid_b * stride_dw_b + cols * stride_dw_h,
        acc,
        mask=mask_h,
    )


@triton.jit
def _rms_norm_bwd_fused_kernel(
    x_ptr, # Pointer to input tensor [B, H]
    w_ptr, # Pointer to gain vector [H]
    dy_ptr, # Pointer to upstream gradient [B, H]
    dx_ptr, # Pointer to input gradient [B, H]
    rstd_ptr, # Pointer to cached FP32 reciprocal std [B]
    dw_ptr, # Pointer to partial weight gradient [N_BLOCKS, H], FP32
    stride_x_b,
    stride_x_h,
    stride_dy_b,
    stride_dy_h,
    stride_dx_b,
    stride_dx_h,
    stride_dw_b,
    stride_dw_h,
    B, # Number of rows
    H,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused input + partial weight gradient in a single pass over the row.

    Each program owns one contiguous block of rows and the full hidden
    dimension. For every row it loads ``x`` and ``dy`` once, computes the
    row-wise dot product, writes ``dx``, and accumulates ``dw`` in FP32
    registers. Requires the whole row to fit in one tile.

    Args:
        x_ptr: Pointer to the forward input of shape ``[B, H]``.
        w_ptr: Pointer to the gain vector of shape ``[H]``.
        dy_ptr: Pointer to the upstream gradient of shape ``[B, H]``.
        dx_ptr: Pointer to the input gradient buffer of shape ``[B, H]``.
        rstd_ptr: Pointer to the cached FP32 reciprocal std of shape ``[B]``.
        dw_ptr: Pointer to the FP32 partial buffer of shape ``[N_BLOCKS, H]``.
        stride_x_b: Batch stride of ``x``.
        stride_x_h: Hidden stride of ``x``.
        stride_dy_b: Batch stride of ``dy``.
        stride_dy_h: Hidden stride of ``dy``.
        stride_dx_b: Batch stride of ``dx``.
        stride_dx_h: Hidden stride of ``dx``.
        stride_dw_b: Row stride of the partial buffer.
        stride_dw_h: Column stride of the partial buffer.
        B: Number of rows in the flattened input.
        H: Hidden dimension size.
        ROWS_PER_PROGRAM: Rows accumulated per program.
        BLOCK_H: Tile size along ``H`` (``>= H`` for the fused path).
    """
    pid = tl.program_id(axis=0)

    row_start = pid * ROWS_PER_PROGRAM
    row_end = tl.minimum(row_start + ROWS_PER_PROGRAM, B)

    cols = tl.arange(0, BLOCK_H)
    mask_h = cols < H
    w = tl.load(w_ptr + cols, mask=mask_h, other=0.0).to(tl.float32)

    dw_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for row in range(row_start, row_end):
        row64 = row.to(tl.int64)
        x = tl.load(
            x_ptr + row64 * stride_x_b + cols * stride_x_h, mask=mask_h, other=0.0
        ).to(tl.float32)
        dy = tl.load(
            dy_ptr + row64 * stride_dy_b + cols * stride_dy_h, mask=mask_h, other=0.0
        ).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)

        proj = tl.sum(dy * w * x, axis=0)
        dx = rstd * dy * w - (rstd * rstd * rstd / H) * x * proj
        tl.store(
            dx_ptr + row64 * stride_dx_b + cols * stride_dx_h,
            dx.to(dx_ptr.dtype.element_ty),
            mask=mask_h,
        )

        dw_acc += dy * x * rstd

    tl.store(
        dw_ptr + pid * stride_dw_b + cols * stride_dw_h,
        dw_acc,
        mask=mask_h,
    )
