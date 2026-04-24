"""Matrix-free local solver parity and memory regression tests.

Guards REVIEW.md Issues C + D: the ``_solve_core_cg`` path must (a) produce
bit-comparable cores to the dense ``_build_block_normal_equations`` +
``torch.linalg.solve`` path on small problems, and (b) fit within a bounded
memory budget at BENCHMARK.md-adjacent scale where the dense H would OOM.
"""
from __future__ import annotations

import torch

from dmrg_transformer.core.device import cuda_available, require_cuda
from dmrg_transformer.optim.local_solver import (
    _apply_J,
    _apply_JT,
    _build_block_normal_equations,
    _should_use_matrix_free,
    _solve_core_cg,
    solve_local_core,
)
from dmrg_transformer.tt import TensorTrain
from dmrg_transformer.tt.environments import left_state_through, right_pure_product
from math import prod


def _build_views(tt: TensorTrain, X: torch.Tensor, Y: torch.Tensor, k: int):
    G_k = tt.get_core(k)
    r_l, p_k, r_r = G_k.shape
    i_k = tt.input_dims[k]
    j_k = tt.output_dims[k]
    batch = X.shape[0]
    J_pre = prod(tt.output_dims[:k]) if k > 0 else 1
    I_suf = prod(tt.input_dims[k + 1 :]) if k + 1 < tt.num_cores else 1
    J_suf = prod(tt.output_dims[k + 1 :]) if k + 1 < tt.num_cores else 1
    L = left_state_through(tt, X, k_stop=k)
    R = right_pure_product(tt, k_start=k + 1)
    L_view = L.reshape(batch, J_pre, r_l, i_k, I_suf)
    R_view = R.reshape(r_r, I_suf, J_suf)
    Y_view = Y.reshape(batch, J_pre, j_k, J_suf)
    return L_view, R_view, Y_view, (r_l, i_k, j_k, r_r)


def test_jacobian_adjoint_symmetry() -> None:
    """``<J v, y> == <v, J^T y>`` for random v, y — the fundamental adjoint identity."""
    dev = require_cuda()
    torch.manual_seed(0)
    N = 16
    W = torch.randn(N, N, dtype=torch.float64, device=dev)
    X = torch.randn(32, N, dtype=torch.float64, device=dev)
    Y = X @ W
    tt, _ = TensorTrain.from_dense(W, [4, 4], [4, 4], max_rank=4)
    L_view, R_view, _, (r_l, i_k, j_k, r_r) = _build_views(tt, X, Y, k=0)
    v = torch.randn(r_l, i_k, j_k, r_r, dtype=torch.float64, device=dev)
    y = torch.randn(L_view.shape[0], L_view.shape[1], j_k, R_view.shape[-1],
                    dtype=torch.float64, device=dev)
    Jv = _apply_J(v, L_view, R_view)
    JTy = _apply_JT(y, L_view, R_view)
    lhs = (Jv * y).sum()
    rhs = (v * JTy).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-10, atol=1e-10), (
        f"adjoint identity violated: <Jv,y>={lhs.item():.3e} vs "
        f"<v,J^Ty>={rhs.item():.3e}"
    )


def test_cg_matches_dense_solve_on_small_problem() -> None:
    """CG must recover the dense-solve core to tight tolerance on small problems."""
    dev = require_cuda()
    torch.manual_seed(1)
    N = 16
    W = torch.randn(N, N, dtype=torch.float64, device=dev)
    X = torch.randn(64, N, dtype=torch.float64, device=dev)
    Y = X @ W
    tt, _ = TensorTrain.from_dense(W, [4, 4], [4, 4], max_rank=4)
    lam = 1e-6
    for k in (0, 1):
        L_view, R_view, Y_view, shape = _build_views(tt, X, Y, k)
        r_l, i_k, j_k, r_r = shape
        P_block = r_l * i_k * r_r

        # Dense path
        H, RHS = _build_block_normal_equations(tt, X, Y, k)
        H_reg = H + lam * torch.eye(P_block, dtype=H.dtype, device=H.device)
        X_dense = torch.linalg.solve(H_reg, RHS)  # (P_block, j_k)
        G_dense = X_dense.reshape(r_l, i_k, r_r, j_k).permute(0, 1, 3, 2)

        # CG path
        G_cg = _solve_core_cg(L_view, R_view, Y_view, shape=shape,
                              lam=lam, tol=1e-12, max_iter=2 * P_block)

        # Compare via residual MSE: both should produce identical Y_pred up
        # to CG tolerance — the direct G comparison is also tight because
        # the system is well-conditioned at lam=1e-6.
        assert torch.allclose(G_cg, G_dense, rtol=1e-6, atol=1e-8), (
            f"k={k}: CG vs dense max abs diff = "
            f"{(G_cg - G_dense).abs().max().item():.3e}"
        )


def test_should_use_matrix_free_threshold() -> None:
    """Threshold picks dense for small P_block and matrix-free at scale."""
    # TTBlock-scale (rank=8, i_k=4) → P_block=256 → H=0.5 MB float64 → dense.
    assert _should_use_matrix_free(8, 4, 8, torch.float64) is False
    # 1024x1024 headline (rank=32, i_k=32) → P_block=32768 → H=8 GiB → CG.
    assert _should_use_matrix_free(32, 32, 32, torch.float64) is True
    # Borderline: tune threshold so mid-sized problems still pick dense.
    assert _should_use_matrix_free(16, 16, 16, torch.float64) is False


def test_solve_local_core_dense_and_cg_paths_agree_on_residual() -> None:
    """End-to-end parity: dense and matrix-free paths must produce the same
    post-truncation residual MSE on the same problem.

    This is the primary regression guard that the CG path is a drop-in
    replacement for the dense path inside ``solve_local_core``.
    """
    import copy

    import dmrg_transformer.optim.local_solver as ls

    dev = require_cuda()
    torch.manual_seed(2)
    N = 32
    W = torch.randn(N, N, dtype=torch.float64, device=dev)
    X = torch.randn(128, N, dtype=torch.float64, device=dev)
    Y = X @ W
    tt_base, _ = TensorTrain.from_dense(W, [4, 8], [8, 4], max_rank=4)

    # Dense path (natural — small P_block stays under the budget).
    tt_dense = copy.deepcopy(tt_base)
    rep_dense = solve_local_core(tt_dense, X, Y, k=0, max_rank=4, lam=1.0e-6)

    # Matrix-free path (forced via monkey-patched predicate).
    tt_cg = copy.deepcopy(tt_base)
    orig = ls._should_use_matrix_free
    ls._should_use_matrix_free = lambda *a, **kw: True
    try:
        rep_cg = solve_local_core(tt_cg, X, Y, k=0, max_rank=4, lam=1.0e-6)
    finally:
        ls._should_use_matrix_free = orig

    # Both paths solve the same strictly convex system to CG tolerance, so
    # the post-truncation residuals must agree tightly.
    assert abs(rep_dense.residual_mse - rep_cg.residual_mse) < 1.0e-8, (
        f"dense residual {rep_dense.residual_mse:.6e} vs "
        f"CG residual {rep_cg.residual_mse:.6e} — paths diverged."
    )
    # And the updated cores must match (gauge is identical since SVD is
    # deterministic on identical inputs).
    for k in range(tt_dense.num_cores):
        diff = (tt_dense.get_core(k) - tt_cg.get_core(k)).abs().max().item()
        assert diff < 1.0e-6, (
            f"core {k} diverged between dense and CG paths: max|Δ|={diff:.3e}"
        )


def test_matrix_free_memory_budget_at_moderate_scale() -> None:
    """At N=256, rank=24 the matrix-free path must stay under 256 MiB peak GPU.

    This is the direct regression guard for REVIEW Issues C + D: the prior
    dense path materialised LL of shape (r_l*i_k*I_suf)^2 which OOMs here.
    """
    dev = require_cuda()
    if not cuda_available():
        return  # CPU has no per-op peak tracker; skip hard assertion.
    torch.manual_seed(4)
    N = 256
    batch = 512
    rank = 24
    W = torch.randn(N, N, dtype=torch.float64, device=dev)
    X = torch.randn(batch, N, dtype=torch.float64, device=dev)
    Y = X @ W
    tt, _ = TensorTrain.from_dense(W, [16, 16], [16, 16], max_rank=rank)

    # Force matrix-free path.
    import dmrg_transformer.optim.local_solver as ls
    orig = ls._should_use_matrix_free
    ls._should_use_matrix_free = lambda *a, **kw: True
    try:
        torch.cuda.reset_peak_memory_stats(dev)
        solve_local_core(tt, X, Y, k=0, max_rank=rank, lam=1e-6)
        peak_mib = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
    finally:
        ls._should_use_matrix_free = orig
    assert peak_mib < 256.0, (
        f"matrix-free solver peak GPU memory {peak_mib:.1f} MiB exceeds 256 MiB "
        "budget at N=256 rank=24 — REVIEW.md Issue C/D may have regressed."
    )
