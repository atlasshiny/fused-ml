# fused-ml

Fused Triton RMSNorm with a hand-written forward **and** backward.

## What it is

A single operation, implemented end to end:

- `_rms_norm_fwd_kernel` — one program per row, writes the output and the FP32
  `rstd` cache, with a single-pass path when the row fits in one tile.
- `_rms_norm_bwd_dx_kernel` + `_rms_norm_bwd_dw_kernel` — tiled fallback for
  rows too large for the fused path, or when the gain is frozen.
- `_rms_norm_bwd_fused_kernel` — the fast path: loads `x` and `dy` once per row,
  computes the row dot product, writes `dx`, and accumulates `dW` in FP32
  registers. No atomics, so `dW` is deterministic.

## Results

RTX 5060 (8 GB, `sm_120`), driver 621.14, Python 3.13.15, `torch==2.15.0+cu132`,
`triton==3.9.0`. Lower is better; times are forward + backward in ms.

| Shape | dtype | fused | Liger 0.8.5 | `torch.nn.functional` |
| --- | --- | --- | --- | --- |
| 4096×2048 | bf16 | **0.441** | 0.493 | 0.426 |
| 16384×2048 | bf16 | **1.452** | 1.620 | 1.804 |
| 8192×4096 | bf16 | **1.452** | 1.570 | 1.794 |
| 16384×4096 | bf16 | **2.873** | 3.174 | 3.598 |
| 4096×8192 | bf16 | **1.476** | 1.587 | 1.923 |
| 16384×4096 | fp32 | **5.754** | 6.323 | 7.581 |

- The fused backward is 1.08–1.32× faster than the two-kernel version it
  replaced (1.22–1.25× at the larger shapes).
- Forward runs at 363 GB/s effective vs. 367 GB/s for a pure `copy_` on the same
  device — i.e. at the memory-copy ceiling. The fused backward moves ~407 MB in
  ~1.05 ms at 16384×4096 bf16 (~383 GB/s effective, read-heavy 2:1).
- `dW` matches PyTorch's FP32 accumulation exactly and is more accurate than
  Liger's default `llama` casting mode (bf16 relative error 1.7e-3 vs 3.0e-3);
  Liger's `gemma` mode matches it.
- Verified against a float64 reference across fp16/bf16/fp32, 1D/2D/3D inputs,
  non-contiguous inputs, `H=127`, frozen gain, and the fused/fallback boundary.

These numbers are from one consumer GPU. They do not transfer to a 5090 or a
B200 without re-tuning `num_warps`, `ROWS_PER_PROGRAM`, and `_MAX_FUSED_H`.

## Setup

```bash
uv sync # installs the pinned CUDA stack and the package
uv run pytest # 38 tests; CUDA tests skip without a GPU
uv run python benchmarks/bench_rms_norm.py
```

The benchmark compares against Liger only if it is installed; run it isolated
without touching the lockfile:

```bash
uv run --with liger-kernel python benchmarks/bench_rms_norm.py
```

## Usage

```python
import torch
from fused_ml import rms_norm

x = torch.randn(2, 8, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = rms_norm(x, w)
```

## Layout

```text
fused_ml/ops/rms_norm.py       Triton kernels (forward, dx, dw, fused backward)
fused_ml/modules/rms_norm.py   Host launches, rstd caching, autograd wiring
tests/test_rms_norm.py         Accuracy and edge-case coverage
benchmarks/bench_rms_norm.py   Shape/dtype sweep vs native and Liger
CMakeLists.txt, Makefile       Unused C++/CUDA scaffolding (does not build)
```

## What was evaluated and rejected

- **Porting to CUDA / Blackwell intrinsics.** Forward is at 99% of copy
  bandwidth and the backward is at the streaming limit, so there is no headroom
  for intrinsics to recover. The datacenter-only Blackwell features (`tcgen05`,
  TMEM, 2-SM clusters) are also unavailable on consumer `sm_120`. The port would
  cost weeks of CMake/pybind/ABI work for no measurable gain.
- **Fusing `add + RMSNorm`.** Worth 1.20–1.25× over eager, but `torch.compile`
  already fuses it (`fused` ≈ `compiled`, sometimes slower at `H=4096`).
- **Whole-pipeline fusion.** GEMMs dominate a training step and cannot be fused
  with pointwise work without losing tensor cores; autograd and the optimizer
  impose global dependencies. See `torch.compile` and CUDA graphs instead.
- **More standalone ops.** A library competing with Liger needs a fusion it
  cannot express, not a reimplementation of ops it already ships.

## Limitations

- Double backward is not supported.
- fp64 inputs accumulate in FP32 in backward (same policy as the forward).
- A custom `autograd.Function` graph-breaks `torch.compile`; wrapping in
  `torch.library.custom_op` with a fake/meta implementation would be required
  for compiled training.
- The fused backward is only used for `H <= 8192`; larger rows take the tiled
  path. `num_warps` and `ROWS_PER_PROGRAM` were tuned on an RTX 5060.

## License

MIT. See [LICENSE](LICENSE).

## When to use something else

Use `torch.nn.functional.rms_norm` for zero dependencies, `torch.compile` for
fused pointwise chains, and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
for fused linear cross-entropy, MoE, and loss kernels.
