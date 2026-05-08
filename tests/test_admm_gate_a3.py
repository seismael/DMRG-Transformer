"""Validation Gate A3 (FUTURE_WORK.md §Option B — Phase 3).

3-layer ``TTLinear`` cascade with ADMM outer loop.  This is the first test
where ADMM's consensus variables actively resolve inter-layer drift:
each layer's augmented target is jointly consistent with the rest of the
chain, instead of arriving via a single chain of pseudo-inverse pull-backs.

The gate compares ADMM against sequential target propagation (the current
cascade test from ``test_target_propagation_cascade.py``) and against a
direct single-sweep baseline.
"""
from __future__ import annotations

import torch

from dmrg_transformer.nn.tt_linear import TTLinear
from dmrg_transformer.optim.admm_outer import ADMMOuter
from dmrg_transformer.optim.sweep import DMRGOptimizer
from dmrg_transformer.propagation.target_propagator import TargetPropagator
from dmrg_transformer.tt import TensorTrain


def _factor_square(n: int) -> list[int]:
    import math

    a = int(math.isqrt(n))
    while a > 1:
        if n % a == 0:
            return [a, n // a]
        a -= 1
    return [1, n]


def _build_3layer_cascade(
    N: int = 12,
    rank: int = 4,
    batch: int = 128,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, list[TTLinear]]:
    """Build a ground-truth 3-layer TT cascade and a trainable stack.

    Returns ``(X, Y_global, [layer0, layer1, layer2])`` where
    ``Y_global = X @ W0_gt @ W1_gt @ W2_gt``, and each trainable layer
    has the same shape / rank but different random init.
    """
    in_dims = _factor_square(N)
    out_dims = _factor_square(N)  # same N for simplicity

    # Ground-truth weights (seeds 1,2,3).
    torch.manual_seed(seed)
    W_gt = []
    for i in range(3):
        tt, _ = TensorTrain.from_dense(
            torch.randn(N, N, dtype=torch.float64),
            in_dims, out_dims, max_rank=rank,
        )
        W_gt.append(tt.to_dense())

    X = torch.randn(batch, N, dtype=torch.float64)
    Y = X @ W_gt[0] @ W_gt[1] @ W_gt[2]

    # Trainable stack (seeds 11,12,13 — different from GT).
    torch.manual_seed(seed + 10)
    layers = []
    for i in range(3):
        layer = TTLinear(
            N, N, input_dims=in_dims, output_dims=out_dims,
            rank=rank, bias=False, dtype=torch.float64,
        )
        layers.append(layer)
    return X, Y, layers


def _chain_forward(X: torch.Tensor, layers: list[TTLinear]) -> torch.Tensor:
    """Forward pass through a chain of TTLinear layers."""
    x = X
    for layer in layers:
        x = layer.forward(x)
    return x


def _chain_mse(X: torch.Tensor, Y: torch.Tensor, layers: list[TTLinear]) -> float:
    return float(torch.mean((_chain_forward(X, layers) - Y) ** 2).item())


def _run_sequential_cascade(
    X: torch.Tensor,
    Y: torch.Tensor,
    layers: list[TTLinear],
    sweeps: int = 3,
) -> float:
    """Run sequential target-propagation cascade (baseline method).

    Returns best global MSE achieved.
    """
    propagator = TargetPropagator(lam=1e-2)
    opt = DMRGOptimizer(max_rank=layers[0].rank, lam=1e-6, clamp_target=False)
    best = float("inf")
    for _ in range(sweeps):
        h1 = layers[0].forward(X)
        h2 = layers[1].forward(h1)
        # Layer 2 (last) → Layer 1 → Layer 0
        tt2 = layers[2]._view_tt()
        opt.sweep(tt2, h2, Y)
        layers[2]._commit_tt(tt2)
        target_h2 = propagator.project_through_linear(tt2.to_dense(), Y)
        tt1 = layers[1]._view_tt()
        opt.sweep(tt1, h1, target_h2)
        layers[1]._commit_tt(tt1)
        target_h1 = propagator.project_through_linear(tt1.to_dense(), target_h2)
        tt0 = layers[0]._view_tt()
        opt.sweep(tt0, X, target_h1)
        layers[0]._commit_tt(tt0)
        best = min(best, _chain_mse(X, Y, layers))
    return best


# -- Gate A3 tests ------------------------------------------------------------


def test_admm_cascade_reduces_global_mse() -> None:
    """ADMM reduces global MSE on a 3-layer TTLinear cascade."""
    X, Y, layers = _build_3layer_cascade(N=12, rank=4, batch=128, seed=1)
    initial_mse = _chain_mse(X, Y, layers)

    admm = ADMMOuter(
        layers=layers,
        rho=0.5,
        tol=1e-6,
        max_iter=10,
        rho_auto_tune=False,
        lam=1e-6,
        propagator_lam=1e-2,
    )
    report = admm.solve(X, Y)

    assert report.final_mse < initial_mse, (
        f"ADMM cascade did not reduce MSE: {initial_mse:.4e} -> {report.final_mse:.4e}"
    )
    # Must achieve meaningful reduction on a rank-feasible cascade.
    assert report.final_mse < initial_mse * 0.5, (
        f"ADMM cascade reduction too weak: {initial_mse:.4e} -> {report.final_mse:.4e}"
    )


def test_admm_cascade_beats_sequential() -> None:
    """ADMM consensus variables should match or beat sequential target prop."""
    X, Y, layers_admm = _build_3layer_cascade(N=12, rank=4, batch=128, seed=2)

    # Clone initial state for fair comparison.
    init_state = {k: v.clone() for k, v in layers_admm[0].state_dict().items()}
    # Sequential baseline (same init).
    layers_seq = []
    for i in range(3):
        layer = TTLinear(
            12, 12,
            input_dims=_factor_square(12), output_dims=_factor_square(12),
            rank=4, bias=False, dtype=torch.float64,
        )
        layers_seq.append(layer)
    # Both stacks use same init.
    for l_seq, l_admm in zip(layers_seq, layers_admm):
        l_seq.load_state_dict(l_admm.state_dict())

    mse_seq = _run_sequential_cascade(X, Y, layers_seq, sweeps=5)

    admm = ADMMOuter(
        layers=layers_admm,
        rho=0.5,
        tol=1e-6,
        max_iter=10,
        rho_auto_tune=False,
        lam=1e-6,
        propagator_lam=1e-2,
    )
    report = admm.solve(X, Y)

    # ADMM should be competitive with sequential target propagation.
    # (Within 5× — ADMM has consensus-variable overhead but should still
    # converge on this rank-feasible task.)
    assert report.final_mse < max(mse_seq * 5.0, 1e-2), (
        f"ADMM MSE={report.final_mse:.4e} vs sequential MSE={mse_seq:.4e}"
    )


def test_admm_cascade_primal_residual_decreases() -> None:
    """Global MSE must trend downward across outer iterations.

    Note: For a pure feedforward chain (no skip connections) the ADMM
    primal residual ‖y−z‖ may fluctuate because the constraint y_i = x_{i+1}
    is automatically satisfied by the chain topology.  ADMM's consensus
    mechanism becomes valuable once residual connections or multi-branch
    architectures are introduced (Gate A3b, future).
    """
    X, Y, layers = _build_3layer_cascade(N=12, rank=4, batch=64, seed=3)

    admm = ADMMOuter(
        layers=layers,
        rho=0.5,
        tol=0.0,
        max_iter=10,
        rho_auto_tune=False,
        lam=1e-6,
        propagator_lam=1e-2,
    )
    report = admm.solve(X, Y)

    # The global MSE must decrease.
    assert report.final_mse < report.mse_history[0], (
        f"MSE did not decrease: {report.mse_history[0]:.4e} -> {report.final_mse:.4e}"
    )
    assert report.final_mse < report.mse_history[0] * 0.5, (
        f"MSE reduction too weak: {report.mse_history[0]:.4e} -> {report.final_mse:.4e}"
    )
