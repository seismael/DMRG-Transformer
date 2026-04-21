"""QR with mandatory float64 cast (NUMERICAL_STABILITY.md §2).

This module is the single authorized call site for ``torch.linalg.qr``.
"""
from __future__ import annotations

import torch

from dmrg_transformer.core.precision import to_f32, to_f64


def qr_f64(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Q, R with float64 internal precision; Q returned as float32.

    Per NUMERICAL_STABILITY §2: orthogonality (Q^T Q = I) degrades rapidly in
    float32 across deep networks, so QR is always upcast.
    """
    if matrix.ndim != 2:
        raise ValueError(f"qr_f64 requires a 2D matrix, got shape {tuple(matrix.shape)}")
    Q64, R64 = torch.linalg.qr(to_f64(matrix), mode="reduced")
    return to_f32(Q64), to_f32(R64)


def qr_f64_strict(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as :func:`qr_f64` but returns float64 Q and R for orthogonality tests."""
    if matrix.ndim != 2:
        raise ValueError(f"qr_f64_strict requires a 2D matrix, got shape {tuple(matrix.shape)}")
    return torch.linalg.qr(to_f64(matrix), mode="reduced")
