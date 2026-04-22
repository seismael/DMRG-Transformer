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
