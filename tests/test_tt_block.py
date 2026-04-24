"""Tests for ``TTBlock`` (Pre-LN Transformer encoder block)."""
from __future__ import annotations

import math

import torch

from dmrg_transformer.nn.tt_block import TTBlock


def _make_block(rank: int = 4, seed: int = 0) -> TTBlock:
    torch.manual_seed(seed)
    return TTBlock(
        embed_dim=12, num_heads=2, hidden_dim=12,
        embed_dims=[3, 4], hidden_dims=[3, 4],
        rank=rank, propagator_lam=1.0e-2, dtype=torch.float64,
    )


def test_ttblock_forward_shape_and_finite() -> None:
    block = _make_block()
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    y = block(x)
    assert y.shape == (2, 5, 12)
    assert torch.isfinite(y).all()


def test_ttblock_dmrg_step_reduces_global_mse() -> None:
    """Single block, rank-feasible target: one DMRG step must reduce MSE."""
    block_gt = _make_block(rank=4, seed=1)
    block = _make_block(rank=4, seed=11)

    torch.manual_seed(7)
    X = torch.randn(8, 4, 12, dtype=torch.float64)
    Y_target = block_gt(X)

    initial_mse = float(torch.mean((block(X) - Y_target) ** 2).item())

    for _ in range(3):
        report = block.dmrg_step(X, Y_target, lam=1.0e-5, target_blend=0.5)

    final_mse = float(torch.mean((block(X) - Y_target) ** 2).item())

    assert final_mse < initial_mse, (
        f"TTBlock.dmrg_step failed to reduce MSE: "
        f"initial={initial_mse:.4e} final={final_mse:.4e}"
    )
    # Q/K/V now update under a trust-region rule; non-convex steps may be
    # rejected, so the per-step reduction is bounded but monotone.
    # We require a measurable improvement (≥ 10%) without claiming parity.
    assert final_mse < 0.9 * initial_mse, (
        f"weak reduction (Q/K freeze gap suspected): "
        f"initial={initial_mse:.4e} final={final_mse:.4e}"
    )

    # Sweep report contract.
    assert "ffn" in report and "attn" in report
    assert {"fc1", "fc2"} <= set(report["ffn"].keys())
    assert {"Q", "K", "V", "W_out"} <= set(report["attn"].keys())
    assert isinstance(report["attn"]["accepted"], bool)
    diag = report["attn"]["diagnostics"]
    # Current TTBlock.dmrg_step contract: Q/K and V each carry an
    # independent trust-region accept/revert flag, and the diagnostics
    # report the global MSE before the joint Q/K/V step plus the global
    # MSE after the V update attempt.
    assert {"qk_accepted", "v_accepted", "mse_before", "mse_after_v"} <= set(diag.keys())
    assert isinstance(diag["qk_accepted"], bool)
    assert isinstance(diag["v_accepted"], bool)
    for key in ("mse_before", "mse_after_v"):
        assert math.isfinite(diag[key])
        assert diag[key] >= 0.0
    assert "global_mse_before" in report and "global_mse_after" in report
