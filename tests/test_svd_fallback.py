"""SVD fallback hierarchy tests (NUMERICAL_STABILITY.md §4)."""
from __future__ import annotations

import torch

from dmrg_transformer.core.svd import SVDDivergenceError, robust_svd, truncate


def test_svd_tier1_on_well_conditioned_matrix() -> None:
    torch.manual_seed(0)
    A = torch.randn(16, 12, dtype=torch.float64)
    res = robust_svd(A)
    assert res.tier == 1
    # Reconstruction check.
    recon = res.U @ torch.diag(res.S) @ res.Vh
    err = float(torch.linalg.norm(A - recon).item()) / float(torch.linalg.norm(A).item())
    assert err < 1.0e-10


def test_svd_handles_rank_deficient_matrix() -> None:
    """Rank-deficient matrix must still decompose via one of the four tiers."""
    torch.manual_seed(1)
    u = torch.randn(10, 1, dtype=torch.float64)
    v = torch.randn(1, 8, dtype=torch.float64)
    A = u @ v  # rank 1
    res = robust_svd(A)
    assert res.tier in {1, 2, 3, 4}
    assert res.S.shape[0] == min(A.shape)


def test_svd_truncation_respects_max_rank() -> None:
    torch.manual_seed(2)
    A = torch.randn(20, 15, dtype=torch.float64)
    res = robust_svd(A)
    trunc = truncate(res, max_rank=5)
    assert trunc.U.shape == (20, 5)
    assert trunc.S.shape == (5,)
    assert trunc.Vh.shape == (5, 15)


def test_svd_raises_only_on_non_2d() -> None:
    """Non-2D inputs raise ValueError; 2D inputs with finite entries never raise."""
    import pytest
    with pytest.raises(ValueError):
        robust_svd(torch.zeros(3, 3, 3))
    # A matrix of zeros is degenerate but not a hardware failure — it must succeed.
    _ = SVDDivergenceError  # ensure the symbol is exported
    zero_mat = torch.zeros(4, 4, dtype=torch.float64)
    res = robust_svd(zero_mat)
    assert float(res.S.max().item()) == 0.0
