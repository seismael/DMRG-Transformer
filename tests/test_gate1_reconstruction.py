"""Validation Gate 1 (AGENTS.md §3, Phase I).

Decompose a 1024×1024 random matrix into a TT with rank r=32, then verify the
reconstruction Frobenius error matches the theoretical Eckart–Young–Mirsky bound
derived from the discarded singular values inside ``TensorTrain.from_dense``.
"""
from __future__ import annotations

import torch

from dmrg_transformer.tt import TensorTrain


def test_gate1_tt_svd_reconstruction_matches_theoretical_bound() -> None:
    torch.manual_seed(0)
    N, M, max_rank = 1024, 1024, 32
    # Factor 1024 = 32 * 32 → d=2 cores, p_k = 32*32 = 1024.
    input_dims = [32, 32]
    output_dims = [32, 32]

    W = torch.randn(N, M, dtype=torch.float32)

    tt, report = TensorTrain.from_dense(W, input_dims, output_dims, max_rank=max_rank)

    # Reconstruct and measure Frobenius error.
    W_hat = tt.to_dense()
    measured = float(torch.linalg.norm(W - W_hat).item())

    # Eckart–Young–Mirsky upper bound (sum-of-cuts).
    theoretical_bound = report.total_frobenius_bound()

    assert theoretical_bound > 0.0, "test is vacuous: nothing was truncated"
    # Reconstruction error must not exceed the theoretical bound.
    assert measured <= theoretical_bound * (1.0 + 1.0e-4), (
        f"reconstruction error {measured:.6e} exceeds bound {theoretical_bound:.6e}"
    )
    # And it must be tight: classical TT-SVD saturates the bound up to numerical noise.
    rel_gap = abs(measured - theoretical_bound) / theoretical_bound
    assert rel_gap < 5.0e-3, (
        f"measured={measured:.6e} bound={theoretical_bound:.6e} rel_gap={rel_gap:.3e}"
    )


def test_gate1_full_rank_round_trip_is_exact() -> None:
    """Decomposing at full TT-rank (no truncation) must reconstruct to machine precision."""
    torch.manual_seed(1)
    W = torch.randn(64, 64, dtype=torch.float32)
    # Max achievable TT-rank for [8,8]x[8,8] interleaved layout is p_1 = 8*8 = 64.
    tt, report = TensorTrain.from_dense(W, [8, 8], [8, 8], max_rank=64)
    # Nothing was truncated → theoretical bound must be 0.
    assert report.total_frobenius_bound() < 1.0e-5
    W_hat = tt.to_dense()
    err = float(torch.linalg.norm(W - W_hat).item()) / float(torch.linalg.norm(W).item())
    assert err < 1.0e-4, f"full-rank round-trip rel error {err:.3e}"


def test_gate1_invariants_enforced() -> None:
    """Boundary ranks r_0 = r_d = 1 must be enforced."""
    torch.manual_seed(2)
    W = torch.randn(16, 16)
    tt, _ = TensorTrain.from_dense(W, [4, 4], [4, 4], max_rank=4)
    assert tt.ranks[0] == 1
    assert tt.ranks[-1] == 1
