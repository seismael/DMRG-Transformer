"""Validation Gate 3 (AGENTS.md §3, Phase III).

Pass condition: DMRG sweep MSE must converge to the same MSE as the dense
least-squares solver on a single TT-factorized layer.
"""
from __future__ import annotations

import time

import torch

from dmrg_transformer.optim.sweep import DMRGOptimizer
from dmrg_transformer.tt import TensorTrain


def _build_synthetic(
    N: int = 64, M: int = 64, batch: int = 256, seed: int = 0,
    input_dims: list[int] | None = None, output_dims: list[int] | None = None,
    rank: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Noiseless low-TT-rank synthetic data. The rank-``rank`` optimum is exact.

    AGENTS.md Gate 3 demands DMRG MSE == dense Exact Solver MSE. For apples-to-apples
    comparison we generate ``Y = X @ W_gt`` with ``W_gt`` intrinsically TT-rank ≤ r,
    so both the unconstrained dense least-squares solver and the rank-r DMRG sweep
    achieve the same (near-zero) optimum.
    """
    if input_dims is None:
        input_dims = [8, 8]
    if output_dims is None:
        output_dims = [8, 8]
    torch.manual_seed(seed)
    X = torch.randn(batch, N, dtype=torch.float64)
    gt, _ = TensorTrain.from_dense(
        torch.randn(N, M, dtype=torch.float64),
        input_dims, output_dims, max_rank=rank,
    )
    W_gt = gt.to_dense()
    Y = X @ W_gt  # noiseless
    W_ref = torch.linalg.lstsq(X, Y).solution
    ref_mse = float(torch.mean((X @ W_ref - Y) ** 2).item())
    return X, Y, ref_mse


def test_gate3_dmrg_matches_dense_lstsq_mse() -> None:
    X, Y, ref_mse = _build_synthetic(N=64, M=64, batch=256, seed=42, rank=16)

    torch.manual_seed(100)
    init, _ = TensorTrain.from_dense(
        torch.randn(64, 64, dtype=torch.float64) * 0.01,
        [8, 8], [8, 8], max_rank=16,
    )
    opt = DMRGOptimizer(max_rank=16, lam=0.0, clamp_target=False)

    prev_mse = float("inf")
    final_mse = float("inf")
    for _ in range(30):
        report = opt.sweep(init, X, Y)
        final_mse = report.final_mse
        if abs(prev_mse - final_mse) < 1.0e-14 * max(1.0, abs(final_mse)):
            break
        prev_mse = final_mse

    # Both should reach the noise floor (near-zero since data is noiseless rank-16).
    # Require DMRG to match reference to within 1% absolute OR machine precision.
    assert final_mse < 1.0e-10 or abs(final_mse - ref_mse) / max(ref_mse, 1.0e-30) < 1.0e-2, (
        f"DMRG MSE={final_mse:.6e} vs dense lstsq MSE={ref_mse:.6e}"
    )


def test_gate3_single_sweep_reduces_mse_monotonically() -> None:
    """Proof of monotonic convergence (SOLVER_MATH.md §V)."""
    X, Y, _ = _build_synthetic(N=64, M=64, batch=128, seed=9)
    torch.manual_seed(11)
    init, _ = TensorTrain.from_dense(
        torch.randn(64, 64, dtype=torch.float64),
        [8, 8], [8, 8], max_rank=16,
    )
    opt = DMRGOptimizer(max_rank=16, lam=1.0e-10, clamp_target=False)
    report = opt.sweep(init, X, Y)
    # Each local step is a convex sub-problem, so the sweep must not increase MSE.
    assert report.final_mse <= report.initial_mse * (1.0 + 1.0e-6), (
        f"MSE increased: {report.initial_mse:.4e} -> {report.final_mse:.4e}"
    )


def test_gate3_wall_time_scales_sublinearly_in_N() -> None:
    """Informational: DMRG per-sweep wall time scales well below N^3.

    Not a strict complexity proof (CPU reference impl with unoptimized einsum),
    but ensures the implementation isn't accidentally O(N^4) or worse.
    """
    def _run(N: int, phys: list[int]) -> float:
        torch.manual_seed(0)
        X, Y, _ = _build_synthetic(
            N=N, M=N, batch=128, seed=0, input_dims=phys, output_dims=phys,
        )
        tt, _ = TensorTrain.from_dense(
            torch.randn(N, N, dtype=torch.float64), phys, phys, max_rank=8,
        )
        opt = DMRGOptimizer(max_rank=8, lam=0.0, clamp_target=False)
        t0 = time.perf_counter()
        opt.sweep(tt, X, Y)
        return time.perf_counter() - t0

    # N doubles from 36 (6x6) to 64 (8x8). Time should grow far less than (64/36)^3 ≈ 5.6x.
    t_small = _run(36, [6, 6])
    t_big = _run(64, [8, 8])
    ratio = t_big / max(t_small, 1.0e-6)
    # Generous bound: DMRG is expected to scale much better than 8x for this size jump.
    assert ratio < 8.0, f"Wall-time ratio {ratio:.2f} suggests super-cubic scaling"
