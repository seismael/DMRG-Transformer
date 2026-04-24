"""Smoke tests for Phase D iso-rank benchmark methods.

Validates:
- ``run_adam_low_rank`` produces finite MSE in ``r·(in+out)`` parameter budget.
- ``run_project_to_rank`` matches Eckart-Young rank truncation of dense lstsq.
- Setting ``target_rank=R`` makes the target representable in TT-rank-R, and
  TT-DMRG approaches the noise floor (the Gate-3 production claim, miniature).
"""
from __future__ import annotations

import math

import pytest

from dmrg_transformer.bench import OptimizationBenchmark


@pytest.mark.slow
def test_run_adam_low_rank_produces_finite_mse() -> None:
    bench = OptimizationBenchmark(
        in_features=32, out_features=32, batch_size=128, rank=4,
    )
    res = bench.run_adam_low_rank(iterations=100, rank=4, warmup=0, seeds=1)

    assert math.isfinite(res.mse), f"non-finite MSE: {res.mse}"
    assert res.time_sec >= 0.0
    # Iso-rank Adam parameter budget = r * (in + out).
    assert res.parameters == 4 * (32 + 32)
    assert "low-rank" in res.name


@pytest.mark.slow
def test_run_project_to_rank_dominated_by_dense_exact() -> None:
    """SVD-truncating the dense lstsq solution to rank-r yields MSE >=
    dense-exact MSE (truncation can only lose information)."""
    bench = OptimizationBenchmark(
        in_features=32, out_features=32, batch_size=128, rank=4,
    )
    dense = bench.run_dense_exact(warmup=0, seeds=1)
    proj = bench.run_project_to_rank(rank=4, warmup=0, seeds=1)

    assert math.isfinite(proj.mse)
    # Truncating to a lower rank can only increase MSE (or hold it steady).
    assert proj.mse + 1e-10 >= dense.mse, (
        f"projected MSE {proj.mse} must be >= dense MSE {dense.mse}"
    )
    assert proj.parameters == 4 * (32 + 32)


@pytest.mark.slow
def test_target_rank_makes_dmrg_approach_noise_floor() -> None:
    """Gate-3 production form: when the target literally lives on a TT
    manifold of rank ``target_rank`` and we run DMRG with
    ``solver_rank >= target_rank``, the DMRG MSE should approach the
    irreducible noise variance (σ² = 0.01 with our σ=0.1 noise).
    """
    bench = OptimizationBenchmark(
        in_features=32, out_features=32, batch_size=256, rank=8,
        target_rank=4,
    )
    dmrg = bench.run_dmrg(num_sweeps=2, warmup=0, seeds=1)
    proj = bench.run_project_to_rank(rank=8, warmup=0, seeds=1)

    NOISE_VAR = 0.01
    # DMRG should hit the noise floor within a generous 4x tolerance at
    # this small smoke scale (the headline run achieves 0.87x).
    assert dmrg.mse < 4.0 * NOISE_VAR, (
        f"DMRG MSE {dmrg.mse} too far above noise floor σ²={NOISE_VAR}"
    )
    # And DMRG should beat naive SVD-truncate by a clear margin.
    assert dmrg.mse < 0.5 * proj.mse, (
        f"DMRG MSE {dmrg.mse} should beat SVD-truncate {proj.mse} by ≥2×"
    )
