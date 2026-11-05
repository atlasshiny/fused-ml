"""Standalone Triton kernels.

This package holds device-side kernel definitions only. Host-side launch code
and autograd wiring live in :mod:`fused_ml.modules`.
"""

from .rms_norm import _rms_norm_fwd_kernel

__all__ = ["_rms_norm_fwd_kernel"]
