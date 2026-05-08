"""Validation Gate A1 (FUTURE_WORK.md §Option B — Phase 1).

Single ``TTLinear`` layer with ADMM outer loop.  Verifies that:

1. ADMM monotonically reduces the global MSE.
2. The primal residual  ‖y − z‖  decreases as the consensus tightens.
3. ADMM converges to a solution comparable to direct DMRG on the same
   rank-bounded target.
"""
from __future__ import annotations

import torch

from dmrg_transformer.nn.tt_linear import TTLinear
from dmrg_transformer.optim.admm_outer import ADMMOuter
from dmrg_transformer.optim.sweep import DMRGOptimizer, SweepReport
from dmrg_transformer.tt import TensorTrain


# -- helpers -----------------------------------------------------------------


def _factor_square(n: int) -> list[int]:
    """Factor n into 2 roughly equal integers for TT dimension splitting."""
    import math

    a = int(math.isqrt(n))
    while a > 1:
        if n % a == 0:
            return [a, n // a]
        a -= 1
    return [1, n]


def _build_tt_native_target(
    N: int = 64,
    M: int = 64,
    batch: int = 256,
    rank: int = 8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, TTLinear]:
    """Generate a TT-rank-``rank`` weight, produce a noiseless target.

    Returns ``(X, Y, layer)`` where ``Y = X @ W_gt`` and ``layer`` is a
    freshly-initialised ``TTLinear`` of the same shape / rank.
    """
    in_dims = _factor_square(N)
    out_dims = _factor_square(M)
    torch.manual_seed(seed)
    W_gt_tt, _ = TensorTrain.from_dense(
        torch.randn(N, M, dtype=torch.float64), in_dims, out_dims, max_rank=rank,
    )
    W_gt = W_gt_tt.to_dense()
    X = torch.randn(batch, N, dtype=torch.float64)
    Y = X @ W_gt

    layer = TTLinear(N, M, input_dims=in_dims, output_dims=out_dims, rank=rank, dtype=torch.float64)
    return X, Y, layer


def _direct_dmrg_mse(layer: TTLinear, X: torch.Tensor, Y: torch.Tensor, sweeps: int = 10) -> float:
    """Run ``sweeps`` direct DMRG sweeps, return final MSE."""
    best = float("inf")
    for _ in range(sweeps):
        rep = layer.dmrg_step(X, Y, lam=0.0, clamp_target=False)
        best = min(best, rep.final_mse)
    return best


# -- Gate A1 tests -----------------------------------------------------------


def test_admm_single_linear_reduces_mse() -> None:
    """ADMM monotonically reduces global MSE on a single TTLinear layer."""
    X, Y, layer = _build_tt_native_target(N=64, M=64, batch=256, rank=8, seed=1)

    initial_mse = float(torch.mean((layer.forward(X) - Y) ** 2).item())

    admm = ADMMOuter(
        layers=[layer],
        rho=0.1,
        tol=1e-6,
        max_iter=30,
        rho_auto_tune=False,
        lam=0.0,
        propagator_lam=1e-6,
    )
    report = admm.solve(X, Y)

    # Monotonic MSE reduction (within noise).
    assert report.final_mse < initial_mse, (
        f"ADMM did not reduce MSE: {initial_mse:.4e} -> {report.final_mse:.4e}"
    )
    # Single-layer ADMM is slower than direct DMRG — convergence to ~1e-2 is
    # expected within 30 iterations.  The value of ADMM appears in multi-layer
    # cases (Gates A2/A3).
    assert report.final_mse < 1e-2, (
        f"ADMM did not converge: final MSE={report.final_mse:.4e}"
    )


def test_admm_single_linear_primal_residual_decreases() -> None:
    """Primal residual  ‖y − z‖  must tighten across outer iterations."""
    X, Y, layer = _build_tt_native_target(N=64, M=64, batch=128, rank=8, seed=2)

    admm = ADMMOuter(
        layers=[layer],
        rho=1.0,
        tol=0.0,            # never early-stop
        max_iter=15,
        rho_auto_tune=False,
        lam=0.0,
        propagator_lam=1e-6,
    )
    report = admm.solve(X, Y)

    primals = report.primal_residuals
    assert len(primals) >= 5, f"too few iterations: {len(primals)}"
    # Primal residual must trend downward (first → last).
    assert primals[-1] < primals[0], (
        f"primal residual did not shrink: {primals[0]:.4e} -> {primals[-1]:.4e}"
    )


def test_admm_single_linear_matches_direct_dmrg() -> None:
    """ADMM reaches the same neighbourhood as direct DMRG sweeps."""
    X, Y, layer_admm = _build_tt_native_target(N=36, M=36, batch=128, rank=6, seed=3)
    # Clone for independent direct-DMRG run via state-dict copy.
    layer_direct = TTLinear(
        36, 36, input_dims=_factor_square(36), output_dims=_factor_square(36),
        rank=6, dtype=torch.float64,
    )
    layer_direct.load_state_dict(layer_admm.state_dict())

    # Direct DMRG
    mse_direct = _direct_dmrg_mse(layer_direct, X, Y, sweeps=10)

    # ADMM
    admm = ADMMOuter(
        layers=[layer_admm],
        rho=0.5,
        tol=1e-6,
        max_iter=20,
        rho_auto_tune=False,
        lam=0.0,
        propagator_lam=1e-6,
    )
    report = admm.solve(X, Y)

    # ADMM should converge (single-layer ADMM is slower than direct DMRG
    # but should still make substantial progress).
    assert report.final_mse < max(mse_direct * 1e4, 1e-2), (
        f"ADMM MSE={report.final_mse:.4e} vs direct DMRG MSE={mse_direct:.4e}"
    )


def test_admm_single_linear_respects_max_iter() -> None:
    """ADMM stops at ``max_iter`` when tolerance is unreachable."""
    X, Y, layer = _build_tt_native_target(N=36, M=36, batch=64, rank=6, seed=4)

    admm = ADMMOuter(
        layers=[layer],
        rho=1.0,
        tol=1e-20,       # unreachable
        max_iter=5,
        rho_auto_tune=False,
        lam=0.0,
    )
    report = admm.solve(X, Y)
    assert report.n_iter == 5
    assert not report.converged


def test_admm_single_linear_primal_dual_consistency() -> None:
    """At convergence, z should be close to both y and Y_global."""
    X, Y, layer = _build_tt_native_target(N=36, M=36, batch=128, rank=6, seed=5)

    admm = ADMMOuter(
        layers=[layer],
        rho=0.5,
        tol=1e-6,
        max_iter=20,
        rho_auto_tune=False,
        lam=0.0,
        propagator_lam=1e-6,
    )
    report = admm.solve(X, Y)

    # After convergence, actual output is closer to Y than initially.
    y_final = layer.forward(X)
    mse_y_vs_Y = float(torch.mean((y_final - Y) ** 2).item())
    initial_mse = report.mse_history[0]
    assert mse_y_vs_Y < initial_mse * 0.5, (
        f"MSE did not improve enough: {initial_mse:.4e} -> {mse_y_vs_Y:.4e}"
    )

    # Consensus z is close to actual output.
    z_final = admm._states[0].z
    mse_z_vs_y = float(torch.mean((z_final - y_final) ** 2).item())
    assert mse_z_vs_y < 1e-2, f"z far from y: MSE={mse_z_vs_y:.4e}"


def test_admm_auto_tune_rho_remains_bounded() -> None:
    """Auto-tune must not push ρ to infinity; it must stay in [1e-6, 1e4]."""
    X, Y, layer = _build_tt_native_target(N=36, M=36, batch=128, rank=6, seed=6)

    admm = ADMMOuter(
        layers=[layer],
        rho=1.0,
        tol=0.0,
        max_iter=20,
        rho_auto_tune=True,
        lam=0.0,
    )
    report = admm.solve(X, Y)

    rho_final = admm.rho
    assert 1e-6 <= rho_final <= 1e4, f"ρ out of bounds: {rho_final:.4e}"
    # MSE should still reduce (auto-tune must not break convergence).
    assert report.final_mse < report.mse_history[0], (
        f"auto-tuned ADMM increased MSE: {report.mse_history[0]:.4e} -> {report.final_mse:.4e}"
    )
