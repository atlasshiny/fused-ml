"""Benchmark RMSNorm forward and forward+backward against native torch and Liger.

Run from the repository root:

    uv run python benchmarks/bench_rms_norm.py

Liger is optional; the comparison is skipped when it is not installed.
"""

import torch
import torch.nn.functional as F

from fused_ml import rms_norm

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
except ImportError:  # pragma: no cover - optional dependency
    LigerRMSNorm = None

SHAPES = [(4096, 2048), (16384, 2048), (8192, 4096), (16384, 4096), (4096, 8192)]
DTYPES = [torch.bfloat16, torch.float32]


def bench(fn, iters: int = 50) -> float:
    """Return the mean wall time of ``fn`` in milliseconds."""
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("RMSNorm benchmarks require a CUDA device")

    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}, {props.total_memory / 1e9:.1f} GB, torch {torch.__version__}")
    header = f"{'shape':>13} {'dtype':>8} | {'fwd':>7} {'fwd+bwd':>8}"
    if LigerRMSNorm is not None:
        header += f" | {'liger':>7} | {'torch':>7}"
    print(header)

    for batch, hidden in SHAPES:
        for dtype in DTYPES:
            x = torch.randn(batch, hidden, device="cuda", dtype=dtype, requires_grad=True)
            weight = torch.randn(hidden, device="cuda", dtype=dtype, requires_grad=True)
            grad_out = torch.randn_like(x)

            fwd_ms = bench(lambda: rms_norm(x, weight))
            full_ms = bench(lambda: rms_norm(x, weight).backward(grad_out, retain_graph=True))

            native_ms = bench(
                lambda: F.rms_norm(x, (hidden,), weight, 1e-6).backward(
                    grad_out, retain_graph=True
                )
            )
            row = f"{batch:>6}x{hidden:<6} {str(dtype).split('.')[-1]:>8} | {fwd_ms:>7.3f} {full_ms:>8.3f}"
            if LigerRMSNorm is not None:
                liger = LigerRMSNorm(
                    hidden, eps=1e-6, casting_mode="gemma", in_place=False
                ).cuda().to(dtype)
                liger_ms = bench(
                    lambda: liger(x).backward(grad_out, retain_graph=True)
                )
                row += f" | {liger_ms:>7.3f} | {native_ms:>7.3f}"
            print(row)


if __name__ == "__main__":
    main()
