"""End-to-end TargetPropagator cascade through stacked TT layers (plan §C1).

This is the first test that exercises ``TargetPropagator.project_through_linear``
in a multi-layer setting with DMRG sweeps on each layer. It verifies that
target propagation produces per-layer targets which, when each layer is then
fitted via DMRG, drive the *global* end-to-end MSE down monotonically.
"""
from __future__ import annotations

import torch

from dmrg_transformer.optim.sweep import DMRGOptimizer
from dmrg_transformer.propagation.target_propagator import TargetPropagator
from dmrg_transformer.tt import TensorTrain


def _fresh_tt(
    N: int, in_dims: list[int], out_dims: list[int], rank: int, seed: int,
) -> TensorTrain:
    torch.manual_seed(seed)
    W = torch.randn(N, N, dtype=torch.float64) * 0.1
    tt, _ = TensorTrain.from_dense(W, in_dims, out_dims, max_rank=rank)
    return tt


def test_target_propagator_cascade_reduces_global_mse() -> None:
    """Stack 3 TT layers, propagate a global target backward, sweep each layer,
    and verify the end-to-end MSE strictly decreases vs the un-fitted stack.

    The global target is constructed from a *known* 3-layer rank-bounded
    cascade so the rank-`r` stack is mathematically capable of fitting it
    closely; this isolates the propagator/sweep logic from rank-deficiency
    artefacts.
    """
    torch.manual_seed(0)
    N = 12
    in_dims = [3, 4]
    out_dims = [4, 3]
    rank = 4

    # Ground-truth: build target as the output of a rank-r 3-layer TT cascade
    # so a perfect fit is achievable in principle.
    tt_gt_1 = _fresh_tt(N, in_dims, out_dims, rank, seed=1)
    tt_gt_2 = _fresh_tt(N, in_dims, out_dims, rank, seed=2)
    tt_gt_3 = _fresh_tt(N, in_dims, out_dims, rank, seed=3)
    W_gt_1 = tt_gt_1.to_dense()
    W_gt_2 = tt_gt_2.to_dense()
    W_gt_3 = tt_gt_3.to_dense()

    # Different seeds for the trainable stack so it starts elsewhere.
    tt1 = _fresh_tt(N, in_dims, out_dims, rank, seed=11)
    tt2 = _fresh_tt(N, in_dims, out_dims, rank, seed=12)
    tt3 = _fresh_tt(N, in_dims, out_dims, rank, seed=13)

    X = torch.randn(128, N, dtype=torch.float64)
    Y_target = X @ W_gt_1 @ W_gt_2 @ W_gt_3

    def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = X @ tt1.to_dense()
        h2 = h1 @ tt2.to_dense()
        h3 = h2 @ tt3.to_dense()
        return h1, h2, h3

    _, _, h3_init = forward()
    initial_mse = float(torch.mean((h3_init - Y_target) ** 2).item())

    # Larger λ: the back-projection must remain numerically stable when the
    # downstream TT is rank-deficient (TT-rank `r` of N×N gives Gram rank ≤ r).
    propagator = TargetPropagator(lam=1.0e-2)
    opt = DMRGOptimizer(max_rank=rank, lam=1.0e-6, clamp_target=False)

    for _ in range(3):
        h1, h2, _h3 = forward()
        opt.sweep(tt3, h2, Y_target)
        target_h2 = propagator.project_through_linear(tt3.to_dense(), Y_target)
        opt.sweep(tt2, h1, target_h2)
        target_h1 = propagator.project_through_linear(tt2.to_dense(), target_h2)
        opt.sweep(tt1, X, target_h1)

    _, _, h3_final = forward()
    final_mse = float(torch.mean((h3_final - Y_target) ** 2).item())

    assert final_mse < initial_mse, (
        f"target propagation cascade failed to reduce global MSE: "
        f"initial={initial_mse:.4e}, final={final_mse:.4e}"
    )
    # Sanity: at least 2× reduction on this rank-feasible cascade.
    assert final_mse < initial_mse * 0.5, (
        f"weak reduction: initial={initial_mse:.4e}, final={final_mse:.4e}"
    )
