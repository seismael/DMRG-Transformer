"""Tests for ``TTFeedForward`` — DMRG sweeps must reduce a rank-feasible target MSE."""
from __future__ import annotations

import torch

from dmrg_transformer.nn.tt_ffn import TTFeedForward


def _make_ffn(rank: int = 4, seed: int = 0) -> TTFeedForward:
    torch.manual_seed(seed)
    return TTFeedForward(
        embed_dim=12, hidden_dim=12,
        embed_dims=[3, 4], hidden_dims=[3, 4],
        rank=rank, propagator_lam=1.0e-2, dtype=torch.float64,
    )


def test_ttffn_forward_shape() -> None:
    ffn = _make_ffn()
    x = torch.randn(7, 12, dtype=torch.float64)
    y = ffn(x)
    assert y.shape == (7, 12)
    assert torch.isfinite(y).all()


def test_ttffn_forward_handles_3d_input() -> None:
    """[batch, seq, embed] inputs must be accepted (Transformer use-case)."""
    ffn = _make_ffn()
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    y = ffn(x)
    assert y.shape == (2, 5, 12)


def test_ttffn_dmrg_step_reduces_global_mse() -> None:
    """3 outer rounds of DMRG against a rank-feasible target must reduce MSE."""
    ffn_gt = _make_ffn(rank=4, seed=1)
    ffn = _make_ffn(rank=4, seed=11)

    torch.manual_seed(123)
    X = torch.randn(64, 12, dtype=torch.float64)
    Y_target = ffn_gt(X)

    initial_mse = float(torch.mean((ffn(X) - Y_target) ** 2).item())

    for _ in range(3):
        ffn.dmrg_step(X, Y_target, lam=1.0e-5, target_blend=0.5)

    final_mse = float(torch.mean((ffn(X) - Y_target) ** 2).item())

    assert final_mse < initial_mse, (
        f"TTFeedForward.dmrg_step failed to reduce MSE: "
        f"initial={initial_mse:.4e} final={final_mse:.4e}"
    )
    # Loose factor (GELU non-linearity + active-mask approx limit the rate).
    assert final_mse < 0.7 * initial_mse, (
        f"weak reduction: initial={initial_mse:.4e} final={final_mse:.4e}"
    )


def test_ttffn_dmrg_step_returns_per_sublayer_reports() -> None:
    ffn = _make_ffn()
    X = torch.randn(32, 12, dtype=torch.float64)
    Y = torch.randn(32, 12, dtype=torch.float64)
    reports = ffn.dmrg_step(X, Y, lam=1.0e-5)
    assert set(reports.keys()) == {"fc1", "fc2"}
    assert reports["fc1"].final_mse <= reports["fc1"].initial_mse + 1e-9
    assert reports["fc2"].final_mse <= reports["fc2"].initial_mse + 1e-9
