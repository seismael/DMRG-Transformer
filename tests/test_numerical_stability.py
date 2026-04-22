"""Numerical-stability edge case tests (NUMERICAL_STABILITY.md §3 + §4).

Covers paths that the existing happy-path tests do not exercise:

* Tikhonov λ escalation when the local solver produces NaNs.
* SVD fallback ladder Tier 2 (CPU gesdd), Tier 3 (CPU gesvd) reachability.
* Tier 4 (noise + retry) divergence error.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from dmrg_transformer.core.svd import (
    SVDDivergenceError,
    _svd_scipy,
    robust_svd,
)
from dmrg_transformer.optim.local_solver import solve_local_core
from dmrg_transformer.tt import TensorTrain


def test_tikhonov_lambda_escalates_on_nan_then_recovers() -> None:
    """When the local solve yields NaN, λ MUST escalate by 10× up to 6 times."""
    torch.manual_seed(0)
    W = torch.randn(16, 16, dtype=torch.float64)
    tt, _ = TensorTrain.from_dense(W, [4, 4], [4, 4], max_rank=4)
    X = torch.randn(32, 16, dtype=torch.float64)
    Y = X @ W

    # Force the FIRST solve to return NaN; second solve uses escalated λ.
    real_solve = torch.linalg.solve
    call_count = {"n": 0}

    def fake_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return torch.full_like(b, float("nan"))
        return real_solve(A, b)

    with patch("torch.linalg.solve", side_effect=fake_solve):
        result = solve_local_core(
            tt, X, Y, k=0,
            max_rank=4, lam=1.0e-8, direction="left", clamp_target=False,
        )
    assert call_count["n"] >= 2, "λ escalation path was not triggered"
    assert result.lam_used >= 1.0e-7, (
        f"λ should have escalated at least once; got {result.lam_used:.3e}"
    )
    assert torch.isfinite(result.U).all() and torch.isfinite(result.S).all()


def test_tikhonov_raises_after_6_escalations() -> None:
    """If NaN persists past 6 escalations the solver must raise."""
    torch.manual_seed(1)
    W = torch.randn(8, 8, dtype=torch.float64)
    tt, _ = TensorTrain.from_dense(W, [2, 4], [4, 2], max_rank=2)
    X = torch.randn(16, 8, dtype=torch.float64)
    Y = X @ W

    def always_nan(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.full_like(b, float("nan"))

    def always_nan_pinv(A: torch.Tensor) -> torch.Tensor:
        return torch.full_like(A, float("nan"))

    with patch("torch.linalg.solve", side_effect=always_nan), \
         patch("torch.linalg.pinv", side_effect=always_nan_pinv):
        with pytest.raises(RuntimeError, match="NaN persisted after 6"):
            solve_local_core(
                tt, X, Y, k=0,
                max_rank=2, lam=0.0, direction="left", clamp_target=False,
            )


def test_svd_tier2_reachable_when_torch_svd_fails() -> None:
    """Tier-2 SciPy gesdd MUST produce a valid SVD when Tier 1 raises."""
    torch.manual_seed(2)
    A = torch.randn(8, 6, dtype=torch.float64)

    def torch_svd_fails(*args, **kwargs):
        raise torch._C._LinAlgError("forced Tier-1 failure")  # type: ignore[attr-defined]

    with patch("torch.linalg.svd", side_effect=torch_svd_fails):
        result = robust_svd(A)
    assert result.tier == 2, f"expected Tier 2 (gesdd), got tier {result.tier}"
    # Reconstruction sanity.
    recon = result.U * result.S.unsqueeze(0) @ result.Vh
    rel = float(torch.linalg.norm(recon - A) / torch.linalg.norm(A))
    assert rel < 1.0e-10


def test_svd_tier3_reachable_when_gesdd_fails() -> None:
    """Tier-3 SciPy gesvd MUST be reached when both torch SVD and gesdd fail."""
    torch.manual_seed(3)
    A = torch.randn(7, 5, dtype=torch.float64)

    def torch_svd_fails(*args, **kwargs):
        raise torch._C._LinAlgError("forced Tier-1 failure")  # type: ignore[attr-defined]

    real_scipy = _svd_scipy

    def scipy_fail_then_succeed(matrix, driver, tier):
        if driver == "gesdd":
            import numpy as np
            raise np.linalg.LinAlgError("forced Tier-2 failure")
        return real_scipy(matrix, driver, tier)

    with patch("torch.linalg.svd", side_effect=torch_svd_fails), \
         patch("dmrg_transformer.core.svd._svd_scipy",
               side_effect=scipy_fail_then_succeed):
        result = robust_svd(A)
    assert result.tier == 3, f"expected Tier 3 (gesvd), got tier {result.tier}"


def test_svd_tier4_noise_retry() -> None:
    """Tier-4 (add ε noise, retry torch SVD) MUST be the final fallback."""
    torch.manual_seed(4)
    A = torch.randn(6, 4, dtype=torch.float64)

    fail_scipy_count = {"n": 0}
    real_torch_svd = torch.linalg.svd

    def torch_svd_first_fails(matrix, *args, **kwargs):
        # First call (Tier 1) fails; second call (Tier 4 with noise) succeeds.
        if not torch.equal(matrix, A):
            return real_torch_svd(matrix, *args, **kwargs)
        raise torch._C._LinAlgError("forced Tier-1 failure")  # type: ignore[attr-defined]

    def scipy_always_fails(*args, **kwargs):
        import numpy as np
        fail_scipy_count["n"] += 1
        raise np.linalg.LinAlgError("forced SciPy failure")

    with patch("torch.linalg.svd", side_effect=torch_svd_first_fails), \
         patch("dmrg_transformer.core.svd._svd_scipy",
               side_effect=scipy_always_fails):
        result = robust_svd(A)
    assert result.tier == 4
    assert fail_scipy_count["n"] == 2  # gesdd + gesvd both attempted


def test_svd_divergence_error_when_all_tiers_fail() -> None:
    """SVDDivergenceError MUST be raised after all 4 tiers fail."""
    torch.manual_seed(5)
    A = torch.randn(5, 3, dtype=torch.float64)

    def torch_svd_always_fails(*args, **kwargs):
        raise torch._C._LinAlgError("forced failure")  # type: ignore[attr-defined]

    def scipy_always_fails(*args, **kwargs):
        import numpy as np
        raise np.linalg.LinAlgError("forced failure")

    with patch("torch.linalg.svd", side_effect=torch_svd_always_fails), \
         patch("dmrg_transformer.core.svd._svd_scipy",
               side_effect=scipy_always_fails):
        with pytest.raises(SVDDivergenceError):
            robust_svd(A)
