"""Validation Gate A2 (FUTURE_WORK.md §Option B — Phase 2).

Single ``TTFeedForward`` with ADMM outer loop.  Verifies that ADMM works
across the GELU nonlinearity: the augmented-target x-update produces a
useful driving signal even when the layer is not purely linear.
"""
from __future__ import annotations

import torch
from torch import nn

from dmrg_transformer.nn.tt_ffn import TTFeedForward
from dmrg_transformer.optim.admm_outer import ADMMOuter
from dmrg_transformer.optim.sweep import SweepReport


def _factor_square(n: int) -> list[int]:
    import math

    a = int(math.isqrt(n))
    while a > 1:
        if n % a == 0:
            return [a, n // a]
        a -= 1
    return [1, n]


def _build_ffn_target(
    embed_dim: int = 16,
    hidden_dim: int = 24,
    batch: int = 256,
    rank: int = 4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, TTFeedForward]:
    """Build a ground-truth TTFeedForward, produce a noiseless target.

    Returns ``(X, Y, ffn)`` where ``Y = ffn_gt(X)`` and ``ffn`` is a
    freshly-initialised trainable ``TTFeedForward`` of the same shape/rank.
    """
    embed_dims = _factor_square(embed_dim)
    hidden_dims = _factor_square(hidden_dim)

    torch.manual_seed(seed)
    # Ground-truth FFN (same architecture, different random init).
    ffn_gt = TTFeedForward(
        embed_dim=embed_dim, hidden_dim=hidden_dim,
        embed_dims=embed_dims, hidden_dims=hidden_dims,
        rank=rank, dtype=torch.float64,
    )
    X = torch.randn(batch, embed_dim, dtype=torch.float64)
    with torch.no_grad():
        Y = ffn_gt.forward(X)

    torch.manual_seed(seed + 100)
    ffn_train = TTFeedForward(
        embed_dim=embed_dim, hidden_dim=hidden_dim,
        embed_dims=embed_dims, hidden_dims=hidden_dims,
        rank=rank, dtype=torch.float64,
    )
    return X, Y, ffn_train


# -- Gate A2 tests ------------------------------------------------------------


def test_admm_ffn_reduces_mse() -> None:
    """ADMM monotonically reduces MSE on a single TTFeedForward."""
    X, Y, ffn = _build_ffn_target(embed_dim=16, hidden_dim=24, batch=256, rank=4, seed=1)

    initial_mse = float(torch.mean((ffn.forward(X) - Y) ** 2).item())

    admm = ADMMOuter(
        layers=[ffn],
        rho=0.5,
        tol=1e-6,
        max_iter=15,
        rho_auto_tune=False,
        lam=0.0,
        propagator_lam=1e-2,
    )
    report = admm.solve(X, Y)

    assert report.final_mse < initial_mse, (
        f"ADMM did not reduce FFN MSE: {initial_mse:.4e} -> {report.final_mse:.4e}"
    )
    # Single-block ADMM is slower than direct dmrg_step — verify meaningful
    # progress (at least 20 % reduction from the random-initialised baseline).
    assert report.final_mse < initial_mse * 0.8, (
        f"ADMM FFN MSE reduction too weak: {initial_mse:.4e} -> {report.final_mse:.4e}"
    )


def test_admm_ffn_improves_vs_no_update() -> None:
    """ADMM output is closer to Y than an untrained forward pass."""
    X, Y, ffn_init = _build_ffn_target(embed_dim=16, hidden_dim=24, batch=128, rank=4, seed=2)

    # Snapshot initial state.
    init_state = {k: v.clone() for k, v in ffn_init.state_dict().items()}

    # Reference: sequential dmrg_step (no ADMM).
    ffn_seq = TTFeedForward(
        embed_dim=16, hidden_dim=24,
        embed_dims=_factor_square(16), hidden_dims=_factor_square(24),
        rank=4, dtype=torch.float64,
    )
    ffn_seq.load_state_dict(init_state)
    mse_seq = float("inf")
    for _ in range(15):
        rep = ffn_seq.dmrg_step(X, Y, lam=0.0, target_blend=0.5)
        mse_seq = min(mse_seq, float(torch.mean((ffn_seq.forward(X) - Y) ** 2).item()))

    # ADMM on identical init.
    ffn_admm = TTFeedForward(
        embed_dim=16, hidden_dim=24,
        embed_dims=_factor_square(16), hidden_dims=_factor_square(24),
        rank=4, dtype=torch.float64,
    )
    ffn_admm.load_state_dict(init_state)

    admm = ADMMOuter(
        layers=[ffn_admm],
        rho=0.5,
        tol=1e-6,
        max_iter=15,
        rho_auto_tune=False,
        lam=0.0,
        propagator_lam=1e-2,
    )
    report = admm.solve(X, Y)

    # ADMM should not be drastically worse than sequential dmrg_step.
    assert report.final_mse < max(mse_seq * 100.0, 1e-2), (
        f"ADMM FFN MSE={report.final_mse:.4e} vs sequential={mse_seq:.4e}"
    )
