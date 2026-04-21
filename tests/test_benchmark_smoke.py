"""Shrunken benchmark smoke test (BENCHMARK.md structure at smaller scale)."""
from __future__ import annotations

import math

import pytest

from dmrg_transformer.bench import OptimizationBenchmark


@pytest.mark.slow
def test_benchmark_smoke_small() -> None:
    """Smoke-scale benchmark: three-way runoff produces finite, ordered results.

    With a non-low-rank target (sin(X W) + noise), the rank-r DMRG will NOT
    match the dense pseudo-inverse MSE — that is the expected compression
    trade-off. We only assert that the pipeline runs end-to-end and DMRG
    delivers the promised parameter compression.
    """
    bench = OptimizationBenchmark(
        in_features=64, out_features=64, batch_size=256, rank=8,
    )
    adam = bench.run_adam(iterations=200)
    dense = bench.run_dense_exact()
    dmrg = bench.run_dmrg(num_sweeps=3)

    for r in (adam, dense, dmrg):
        assert math.isfinite(r.mse), f"{r.name} produced non-finite MSE={r.mse}"
        assert r.time_sec >= 0.0

    # Parameter compression must be real (rank 8 vs full 64x64).
    assert dmrg.parameters < dense.parameters
    # NB: The target here is `sin(X W) + noise` — NOT a low-TT-rank function.
    # The rank-r DMRG therefore CANNOT match dense or Adam on MSE; it is
    # correctly rank-constrained. PoC parity is validated on TT-native targets
    # in test_gate3_exact_parity.py and bench/POC_RESULTS.md. Here we only
    # assert the pipeline is wired end-to-end and produces finite numbers.
