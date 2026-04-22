"""Adaptive-rank wiring through ``DMRGOptimizer.sweep`` (plan §C5 closure).

Validates that:

1. ``DMRGOptimizer(adaptive_threshold=...)`` propagates the threshold all the
   way to the local SVD truncation.
2. With a strict threshold (e.g. ``1e-12``), the achieved MSE matches the
   fixed-``max_rank`` baseline (no information lost — the discarded mass is
   below threshold so all modes are kept up to ``max_rank``).
3. With a loose threshold and an easy target (low-true-rank), the achieved
   ranks are *strictly* below ``max_rank`` — i.e. the rule actually prunes.
"""
from __future__ import annotations

import torch

from dmrg_transformer.optim.sweep import DMRGOptimizer
from dmrg_transformer.tt.tensor_train import TensorTrain


def _make_low_rank_target(
    in_features: int, out_features: int, true_rank: int, batch: int, *, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    A = torch.randn(in_features, true_rank, dtype=torch.float64)
    B = torch.randn(true_rank, out_features, dtype=torch.float64)
    W_true = A @ B
    X = torch.randn(batch, in_features, dtype=torch.float64)
    Y = X @ W_true
    return X, Y, W_true


def _init_tt(in_features: int, out_features: int, max_rank: int) -> TensorTrain:
    """Random small-init TT in the requested factorization."""
    return TensorTrain(
        cores=[
            torch.randn(1, 4 * 4, max_rank, dtype=torch.float64) * 0.01,
            torch.randn(max_rank, 4 * 4, max_rank, dtype=torch.float64) * 0.01,
            torch.randn(max_rank, 4 * 4, 1, dtype=torch.float64) * 0.01,
        ],
        input_dims=[4, 4, 4],
        output_dims=[4, 4, 4],
    )


def test_adaptive_threshold_matches_fixed_when_strict() -> None:
    """A strict (≈0) threshold must keep every mode up to max_rank — same MSE
    as the non-adaptive baseline."""
    X, Y, _ = _make_low_rank_target(64, 64, true_rank=32, batch=128, seed=11)
    tt_a = _init_tt(64, 64, max_rank=8)
    tt_b = _init_tt(64, 64, max_rank=8)
    # Same initialization (tt_b mirrors tt_a).
    for k in range(tt_a.num_cores):
        tt_b.update_core(k, tt_a.get_core(k).clone())

    opt_fixed = DMRGOptimizer(max_rank=8, lam=1.0e-8)
    opt_adapt = DMRGOptimizer(max_rank=8, lam=1.0e-8, adaptive_threshold=0.0)
    rep_fixed = opt_fixed.sweep(tt_a, X, Y)
    rep_adapt = opt_adapt.sweep(tt_b, X, Y)

    # The threshold==0 branch must take the same code path as fixed-rank
    # (every mode kept up to max_rank). Final MSEs must agree to FP noise.
    rel = abs(rep_fixed.final_mse - rep_adapt.final_mse) / max(rep_fixed.final_mse, 1e-30)
    assert rel < 1.0e-6, (
        f"strict adaptive threshold should match fixed-rank baseline: "
        f"fixed={rep_fixed.final_mse:.4e} adapt={rep_adapt.final_mse:.4e}"
    )


def test_adaptive_threshold_prunes_on_easy_target() -> None:
    """With a loose threshold the rule must cap the achieved rank at well
    below ``max_rank`` somewhere in the sweep — i.e. the spec §C5 pruning
    branch actually fires."""
    X, Y, _ = _make_low_rank_target(64, 64, true_rank=2, batch=128, seed=23)
    tt = _init_tt(64, 64, max_rank=8)
    # Threshold of 0.5 means "keep modes only until 50% of the spectral mass
    # is retained" — this is aggressive enough to prune any non-degenerate
    # local block down to rank 1 or 2.
    opt = DMRGOptimizer(max_rank=8, lam=1.0e-8, adaptive_threshold=0.5)
    rep = opt.sweep(tt, X, Y)
    achieved_ranks = [tt.get_core(k).shape[-1] for k in range(tt.num_cores - 1)]
    assert any(r < 8 for r in achieved_ranks), (
        f"adaptive_threshold=0.5 failed to prune at all: ranks={achieved_ranks}"
    )
    # Sanity: the sweep still ran a full pass (we have a finite final MSE).
    assert rep.final_mse >= 0.0
