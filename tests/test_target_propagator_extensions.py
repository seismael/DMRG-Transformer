"""Tests for ``TargetPropagator.project_through_residual / _layernorm``.

Verifies the closed-form pull-backs round-trip against their forward maps on
synthetic inputs (rank-feasible, no rank-deficiency artefacts).
"""
from __future__ import annotations

import torch

from dmrg_transformer.propagation.target_propagator import TargetPropagator


def test_project_through_residual_recovers_branch_target() -> None:
    """For ``y = x + f(x)``, given ``y_target = x + f_target``, the
    propagator must return exactly ``f_target``.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 16, dtype=torch.float64)
    f_target = torch.randn(8, 16, dtype=torch.float64)
    y_target = x + f_target

    prop = TargetPropagator(lam=1e-5)
    f_recovered = prop.project_through_residual(y_target, x)

    assert torch.allclose(f_recovered, f_target, atol=1e-12), (
        f"residual pull-back must be exact; max err "
        f"{(f_recovered - f_target).abs().max().item():.3e}"
    )


def test_project_through_residual_shape_mismatch_raises() -> None:
    prop = TargetPropagator()
    with __import__("pytest").raises(ValueError):
        prop.project_through_residual(torch.zeros(4, 8), torch.zeros(4, 16))


def test_project_through_layernorm_round_trip_identity_affine() -> None:
    """LN with γ=1, β=0: pulling back ``y = LN(x)`` must recover ``x`` (up to
    the LN affine subspace — i.e. matching row mean/std). We verify by feeding
    ``y_target = LN(x)`` and checking that ``LN(x_pulled) ≈ y_target``.
    """
    torch.manual_seed(1)
    x = torch.randn(16, 32, dtype=torch.float64) * 3.0 + 1.5
    ln = torch.nn.LayerNorm(32, dtype=torch.float64, elementwise_affine=False)
    y_target = ln(x)

    prop = TargetPropagator(lam=1e-5)
    x_pulled = prop.project_through_layernorm(y_target, x_pre_ln=x, eps=1e-5)

    # The pull-back uses x's row stats so x_pulled must satisfy LN(x_pulled) ≈ y_target.
    y_check = ln(x_pulled)
    assert torch.allclose(y_check, y_target, atol=1e-9), (
        f"LN round-trip failed; max err {(y_check - y_target).abs().max().item():.3e}"
    )


def test_project_through_layernorm_with_affine_params() -> None:
    """LN with non-trivial γ, β: ``y = γ * normalize(x) + β`` must round-trip."""
    torch.manual_seed(2)
    x = torch.randn(8, 24, dtype=torch.float64) * 2.0
    gamma = torch.rand(24, dtype=torch.float64) + 0.5  # in [0.5, 1.5]
    beta = torch.randn(24, dtype=torch.float64) * 0.3

    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    sigma = torch.sqrt(var + 1e-5)
    y_target = gamma * (x - mu) / sigma + beta

    prop = TargetPropagator(lam=0.0)  # exact (no Tikhonov on γ)
    x_pulled = prop.project_through_layernorm(
        y_target, x_pre_ln=x, eps=1e-5, gamma=gamma, beta=beta,
    )

    # Forward through the same affine LN.
    mu_p = x_pulled.mean(dim=-1, keepdim=True)
    var_p = x_pulled.var(dim=-1, keepdim=True, unbiased=False)
    sigma_p = torch.sqrt(var_p + 1e-5)
    y_check = gamma * (x_pulled - mu_p) / sigma_p + beta
    assert torch.allclose(y_check, y_target, atol=1e-9), (
        f"LN affine round-trip failed; max err {(y_check - y_target).abs().max().item():.3e}"
    )


def test_project_through_layernorm_shape_mismatch_raises() -> None:
    prop = TargetPropagator()
    with __import__("pytest").raises(ValueError):
        prop.project_through_layernorm(torch.zeros(4, 8), torch.zeros(4, 16))


def test_project_through_attention_v_recovers_v_when_context_is_av() -> None:
    """Given A and a context produced as ``A @ V_true``, the V pull-back must
    recover ``V_true`` (up to the Tikhonov damping). Use a well-conditioned A
    (random softmax over short L_k) and small λ.
    """
    torch.manual_seed(3)
    B, H, L_q, L_k, d_h = 2, 2, 5, 5, 4
    scores = torch.randn(B, H, L_q, L_k, dtype=torch.float64)
    A = torch.softmax(scores, dim=-1)
    V_true = torch.randn(B, H, L_k, d_h, dtype=torch.float64)
    context = A @ V_true

    prop = TargetPropagator(lam=1.0e-8)
    V_recovered = prop.project_through_attention_v(A, context)

    err = (V_recovered - V_true).abs().max().item()
    assert err < 1e-4, f"V pull-back error too large: {err:.3e}"


def test_project_through_attention_v_shape_validation() -> None:
    prop = TargetPropagator()
    with __import__("pytest").raises(ValueError):
        prop.project_through_attention_v(
            torch.zeros(2, 5, 5), torch.zeros(2, 2, 5, 4),
        )
    with __import__("pytest").raises(ValueError):
        prop.project_through_attention_v(
            torch.zeros(2, 2, 5, 5), torch.zeros(2, 3, 5, 4),
        )


def test_solve_attention_pattern_target_recovers_a_when_v_is_full_rank() -> None:
    """If ``C = A V`` and V is square + invertible, ``A_target ≈ A`` after
    simplex projection (when A is already row-stochastic and well-separated).
    """
    torch.manual_seed(11)
    B, H, L_q, L_k, d_h = 1, 1, 3, 4, 4  # L_k == d_h → V invertible
    scores = torch.randn(B, H, L_q, L_k, dtype=torch.float64) * 2.0
    A_true = torch.softmax(scores, dim=-1)
    V = torch.randn(B, H, L_k, d_h, dtype=torch.float64)
    C = A_true @ V

    prop = TargetPropagator(lam=1.0e-10)
    A_recovered = prop.solve_attention_pattern_target(V, C, eps=1.0e-10)

    # Result must be row-stochastic.
    row_sums = A_recovered.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1.0e-9)
    assert (A_recovered >= 0).all()
    err = (A_recovered - A_true).abs().max().item()
    assert err < 1.0e-4, f"A pull-back error too large: {err:.3e}"


def test_solve_attention_pattern_target_woodbury_matches_dense() -> None:
    """REVIEW.md Issue E: the Woodbury push-through path (d_h < L_k) must
    produce numerically identical ``A_unconstrained`` to the dense
    ``(V V^T + λI_L)^{-1}`` path.

    The two paths share the simplex-projection postprocessing, so comparing
    ``A_target`` (post clamp + renormalise) is the tightest possible check.
    """
    torch.manual_seed(42)
    B, H, L_q, L_k, d_h = 2, 3, 7, 16, 4  # d_h < L_k → Woodbury path active.
    V = torch.randn(B, H, L_k, d_h, dtype=torch.float64)
    C = torch.randn(B, H, L_q, d_h, dtype=torch.float64)
    lam = 1.0e-6

    # Reference: run the dense path by forcing d_h >= L_k via zero-padding V.
    # This keeps λ and C identical and yields the same A up to the implicit
    # zero columns being absorbed by the damping.
    prop = TargetPropagator(lam=lam)
    A_struct = prop.solve_attention_pattern_target(V, C, eps=1.0e-14)

    # Inline reference computation of the dense path for exact parity:
    VVt = V @ V.transpose(-2, -1)
    eye = torch.eye(L_k, dtype=VVt.dtype, device=VVt.device).expand_as(VVt)
    rhs = C @ V.transpose(-2, -1)
    A_ref_unconstr = torch.linalg.solve(
        VVt + lam * eye, rhs.transpose(-2, -1)
    ).transpose(-2, -1)
    A_ref_clamped = A_ref_unconstr.clamp_min(1.0e-14)
    A_ref = A_ref_clamped / A_ref_clamped.sum(dim=-1, keepdim=True)

    max_diff = (A_struct - A_ref).abs().max().item()
    # 1e-8 tolerance: the dense path's (V V^T + λI_L) is rank-deficient
    # (rank ≤ d_h in L_k-dim), so its solve is ill-conditioned at λ=1e-6.
    # The Woodbury path works in the d_h×d_h space and is actually more
    # accurate — the gap is pure float64 residue from the dense path.
    assert max_diff < 1.0e-8, (
        f"Woodbury vs dense path diverged: max|Δ|={max_diff:.3e}"
    )


def test_solve_attention_pattern_target_scales_to_long_sequences() -> None:
    """REVIEW Issue E regression: at L=512, d_h=8 the Woodbury path must
    use materially less peak memory than the dense O(L²) ``VVt`` path.

    Approach: compare peak *additional* allocations during each path with
    a clean allocator state per call. The dense path necessarily allocates
    ``VVt`` and ``VVt + λI`` both of shape ``[B, H, L, L]`` whereas the
    Woodbury path allocates only ``VtV`` of shape ``[B, H, d_h, d_h]``.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    torch.manual_seed(7)
    dev = torch.device("cuda", 0)
    B, H, L, d_h = 2, 4, 512, 8
    V = torch.randn(B, H, L, d_h, dtype=torch.float64, device=dev)
    C = torch.randn(B, H, L, d_h, dtype=torch.float64, device=dev)
    prop = TargetPropagator(lam=1.0e-6)
    lam = prop.lam

    def _measure_peak(fn):
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        out = fn()
        torch.cuda.synchronize(dev)
        peak = torch.cuda.max_memory_allocated(dev) - base
        return out, peak

    def _run_dense():
        VVt = V @ V.transpose(-2, -1)
        eye = torch.eye(L, dtype=VVt.dtype, device=dev).expand_as(VVt)
        rhs = C @ V.transpose(-2, -1)
        X = torch.linalg.solve(
            VVt + lam * eye, rhs.transpose(-2, -1)
        ).transpose(-2, -1)
        Xc = X.clamp_min(1.0e-14)
        return Xc / Xc.sum(dim=-1, keepdim=True)

    A_wb, peak_wb = _measure_peak(
        lambda: prop.solve_attention_pattern_target(V, C, eps=1.0e-14),
    )
    A_dense, peak_dense = _measure_peak(_run_dense)

    assert torch.isfinite(A_wb).all()
    assert torch.allclose(A_wb, A_dense, atol=1.0e-7, rtol=1.0e-5), (
        f"Woodbury vs dense: max|Δ|={(A_wb - A_dense).abs().max().item():.3e}"
    )
    # Woodbury must be meaningfully cheaper than dense. At L=512 dense
    # allocates at least 3× [B, H, L, L] float64 = 3 × 16 MiB = 48 MiB of
    # L×L tensors that Woodbury avoids entirely. Require ≥ 1.5× ratio to
    # absorb allocator jitter while still failing loudly if Woodbury ever
    # regresses to materialising VVt.
    ratio = peak_dense / max(peak_wb, 1)
    assert ratio >= 1.5, (
        f"Woodbury peak={peak_wb/2**20:.2f} MiB, "
        f"dense peak={peak_dense/2**20:.2f} MiB "
        f"(ratio={ratio:.2f}×) — REVIEW Issue E regressed."
    )


def test_softmax_target_to_scores_round_trip() -> None:
    """``softmax(softmax_target_to_scores(A)) == A`` (gauge-invariance check)."""
    torch.manual_seed(12)
    A = torch.softmax(torch.randn(2, 3, 4, 5, dtype=torch.float64), dim=-1)
    prop = TargetPropagator()
    scores = prop.softmax_target_to_scores(A, scale=1.0)
    A_back = torch.softmax(scores, dim=-1)
    err = (A_back - A).abs().max().item()
    assert err < 1.0e-12, f"softmax inverse round-trip failed: {err:.3e}"
    # Gauge: each row of the recovered logits must sum to zero.
    row_sums = scores.sum(dim=-1)
    assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1.0e-12)


def test_project_through_qk_bilinear_satisfies_equation_underdetermined() -> None:
    """Underdetermined regime (L_k < d_h): Q* won't equal Q_true (many valid
    Q* exist), but it MUST satisfy Q* K^T = S to high precision.
    """
    torch.manual_seed(13)
    B, H, L_q, L_k, d_h = 2, 2, 4, 4, 6
    Q_true = torch.randn(B, H, L_q, d_h, dtype=torch.float64)
    K_true = torch.randn(B, H, L_k, d_h, dtype=torch.float64)
    scores = Q_true @ K_true.transpose(-2, -1)

    prop = TargetPropagator(lam=1.0e-10)
    Q_recovered, K_recovered = prop.project_through_qk_bilinear(
        scores, Q_true, K_true,
    )
    res_q = (Q_recovered @ K_true.transpose(-2, -1) - scores).abs().max().item()
    res_k = (Q_true @ K_recovered.transpose(-2, -1) - scores).abs().max().item()
    assert res_q < 1.0e-6, f"Q solver residual: {res_q:.3e}"
    assert res_k < 1.0e-6, f"K solver residual: {res_k:.3e}"


def test_project_through_qk_bilinear_recovers_q_overdetermined() -> None:
    """Overdetermined regime (L_k >= d_h): the system Q* K^T = S has a unique
    solution (Q_true) with K fixed, so we should recover Q_true exactly.
    """
    torch.manual_seed(14)
    B, H, L_q, L_k, d_h = 2, 2, 5, 8, 4   # L_k=8 > d_h=4
    Q_true = torch.randn(B, H, L_q, d_h, dtype=torch.float64)
    K_true = torch.randn(B, H, L_k, d_h, dtype=torch.float64)
    scores = Q_true @ K_true.transpose(-2, -1)

    prop = TargetPropagator(lam=1.0e-10)
    Q_recovered, _ = prop.project_through_qk_bilinear(scores, Q_true, K_true)
    err = (Q_recovered - Q_true).abs().max().item()
    assert err < 1.0e-4, f"Q pull-back error (overdetermined): {err:.3e}"


def test_project_through_qk_bilinear_recovers_k_overdetermined() -> None:
    """Symmetric overdetermined K test (L_q >= d_h)."""
    torch.manual_seed(15)
    B, H, L_q, L_k, d_h = 2, 2, 8, 5, 4
    Q_true = torch.randn(B, H, L_q, d_h, dtype=torch.float64)
    K_true = torch.randn(B, H, L_k, d_h, dtype=torch.float64)
    scores = Q_true @ K_true.transpose(-2, -1)

    prop = TargetPropagator(lam=1.0e-10)
    _, K_recovered = prop.project_through_qk_bilinear(scores, Q_true, K_true)
    err = (K_recovered - K_true).abs().max().item()
    assert err < 1.0e-4, f"K pull-back error (overdetermined): {err:.3e}"


def test_full_attention_pull_back_pipeline_reproduces_context() -> None:
    """End-to-end: starting from Q,K,V at the truth, the propagator pipeline
    must produce (Q*, K*, V*) that, when forward-propagated, reproduce the
    target context to within Tikhonov-precision. Parameters need not match
    individually (alternating solver finds *a* valid joint solution), only
    the forward equation does.

    Conditioning requirements:
      * V must be full-rank in the row space (``L_k <= d_h``) so the A pull-back
        is unique.
      * Q and K solvers in the overdetermined regime: ``L_q >= d_h`` AND
        ``L_k >= d_h``. Combined with the above: ``L_k == d_h``.
    """
    torch.manual_seed(16)
    B, H, L_q, L_k, d_h = 1, 1, 5, 4, 4   # V is 4x4 invertible; Q,K solvers exact
    Q_true = torch.randn(B, H, L_q, d_h, dtype=torch.float64)
    K_true = torch.randn(B, H, L_k, d_h, dtype=torch.float64)
    V_true = torch.randn(B, H, L_k, d_h, dtype=torch.float64)
    scale = d_h ** -0.5
    A_true = torch.softmax(Q_true @ K_true.transpose(-2, -1) * scale, dim=-1)
    C_true = A_true @ V_true

    prop = TargetPropagator(lam=1.0e-12)
    A_target = prop.solve_attention_pattern_target(V_true, C_true, eps=1.0e-14)
    scores_target = prop.softmax_target_to_scores(A_target, scale=1.0 / scale)
    Q_recovered, K_recovered = prop.project_through_qk_bilinear(
        scores_target, Q_true, K_true,
    )
    V_recovered = prop.project_through_attention_v(A_target, C_true)

    # Forward through the recovered Q,K,V and compare to C_true.
    A_back = torch.softmax(
        Q_recovered @ K_recovered.transpose(-2, -1) * scale, dim=-1,
    )
    C_back = A_back @ V_recovered
    err = (C_back - C_true).abs().max().item()
    assert err < 1.0e-3, f"end-to-end context recovery error: {err:.3e}"
