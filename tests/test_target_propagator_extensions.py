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
