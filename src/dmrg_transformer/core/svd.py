"""Robust SVD with the four-tier fallback hierarchy (NUMERICAL_STABILITY.md §4).

This module is the **only** authorized call site for ``torch.linalg.svd`` and
``scipy.linalg.svd`` in the entire package. AGENTS.md prime directive: the
implementation is rejected if a raw SVD is invoked elsewhere. See
``tests/test_constraints.py`` for the AST scan that enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import torch

from dmrg_transformer.core.precision import needs_f64_upcast, to_f32, to_f64

_NOISE_SCALE: float = 1.0e-7


class SVDDivergenceError(RuntimeError):
    """Raised only after Tier 4 (noise + retry) of the fallback hierarchy fails."""


@dataclass(frozen=True)
class SVDResult:
    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor
    tier: int  # 1: GPU, 2: gesdd, 3: gesvd, 4: noise+retry


def _svd_torch(matrix: torch.Tensor) -> SVDResult:
    out_dtype = matrix.dtype
    work = to_f64(matrix) if needs_f64_upcast(matrix) else matrix
    U, S, Vh = torch.linalg.svd(work, full_matrices=False)
    if out_dtype == torch.float64:
        return SVDResult(U=to_f64(U), S=to_f64(S), Vh=to_f64(Vh), tier=1)
    return SVDResult(U=to_f32(U), S=to_f32(S), Vh=to_f32(Vh), tier=1)


def _svd_scipy(matrix: torch.Tensor, driver: str, tier: int) -> SVDResult:
    out_dtype = matrix.dtype
    np_in = matrix.detach().cpu().numpy().astype(np.float64, copy=False)
    U_np, S_np, Vh_np = scipy.linalg.svd(
        np_in, full_matrices=False, lapack_driver=driver, check_finite=True
    )
    device = matrix.device
    return SVDResult(
        U=torch.from_numpy(U_np).to(device=device, dtype=out_dtype),
        S=torch.from_numpy(S_np).to(device=device, dtype=out_dtype),
        Vh=torch.from_numpy(Vh_np).to(device=device, dtype=out_dtype),
        tier=tier,
    )


def robust_svd(matrix: torch.Tensor) -> SVDResult:
    """Single authoritative SVD call site. Implements the 4-tier fallback ladder.

    Tier 1: GPU/native ``torch.linalg.svd``.
    Tier 2: SciPy LAPACK ``gesdd`` on CPU (float64).
    Tier 3: SciPy LAPACK ``gesvd`` on CPU (float64) — slower but unconditionally stable.
    Tier 4: Add Gaussian noise of σ=1e-7 to break symmetry, retry Tier 1.

    Raises:
        SVDDivergenceError: if all four tiers fail.
    """
    if matrix.ndim != 2:
        raise ValueError(f"robust_svd requires a 2D matrix, got shape {tuple(matrix.shape)}")

    try:
        return _svd_torch(matrix)
    except (torch._C._LinAlgError, RuntimeError):  # type: ignore[attr-defined]
        pass

    try:
        return _svd_scipy(matrix, driver="gesdd", tier=2)
    except (np.linalg.LinAlgError, ValueError):
        pass

    try:
        return _svd_scipy(matrix, driver="gesvd", tier=3)
    except (np.linalg.LinAlgError, ValueError):
        pass

    noise = torch.randn_like(matrix) * _NOISE_SCALE
    try:
        result = _svd_torch(matrix + noise)
        return SVDResult(U=result.U, S=result.S, Vh=result.Vh, tier=4)
    except (torch._C._LinAlgError, RuntimeError) as exc:  # type: ignore[attr-defined]
        raise SVDDivergenceError(
            f"All four SVD fallback tiers failed for matrix of shape {tuple(matrix.shape)}"
        ) from exc


def truncate(result: SVDResult, max_rank: int) -> SVDResult:
    """Eckart–Young–Mirsky truncation (TENSOR_TOPOLOGY.md §6, step 2)."""
    r = min(max_rank, result.S.shape[0])
    return SVDResult(
        U=result.U[:, :r].contiguous(),
        S=result.S[:r].contiguous(),
        Vh=result.Vh[:r, :].contiguous(),
        tier=result.tier,
    )


def discarded_energy(full_S: torch.Tensor, kept_rank: int) -> float:
    """Theoretical Frobenius truncation error: sqrt(Σ_{i>r} σ_i²)."""
    if kept_rank >= full_S.shape[0]:
        return 0.0
    tail = full_S[kept_rank:]
    return float(torch.sqrt(torch.sum(tail.to(torch.float64) ** 2)).item())
