"""Mixed-precision casting helpers (NUMERICAL_STABILITY.md §2)."""
from __future__ import annotations

import torch

CONDITION_UPCAST_THRESHOLD: float = 1.0e4


def to_f64(t: torch.Tensor) -> torch.Tensor:
    """Upcast to float64 for numerically sensitive ops (QR / ill-conditioned SVD)."""
    return t.to(dtype=torch.float64) if t.dtype != torch.float64 else t


def to_f32(t: torch.Tensor) -> torch.Tensor:
    """Downcast back to float32 for Tensor-Core-bound contractions."""
    return t.to(dtype=torch.float32) if t.dtype != torch.float32 else t


def condition_number(matrix: torch.Tensor) -> float:
    """Spectral condition number used to gate dynamic upcasting."""
    if matrix.numel() == 0:
        return 0.0
    s = torch.linalg.svdvals(to_f64(matrix))
    s_max = float(s[0])
    s_min = float(s[-1])
    if s_min <= 0.0:
        return float("inf")
    return s_max / s_min


def needs_f64_upcast(matrix: torch.Tensor) -> bool:
    """Return True if SVD/inverse should be done in float64 per NUMERICAL_STABILITY §2."""
    return condition_number(matrix) > CONDITION_UPCAST_THRESHOLD
