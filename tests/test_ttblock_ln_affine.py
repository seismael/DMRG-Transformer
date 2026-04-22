"""LN affine (γ, β) OLS update tests for ``TTBlock`` (compliance closure)."""
from __future__ import annotations

import torch

from dmrg_transformer.nn.tt_block import TTBlock, _AffineLN


def test_affine_ln_init_is_identity_with_frozen_layernorm() -> None:
    """At init γ=1, β=0 — must be bit-exact with elementwise_affine=False LN."""
    torch.manual_seed(0)
    f = 12
    affine = _AffineLN(features=f, eps=1.0e-5, dtype=torch.float64)
    ref = torch.nn.LayerNorm(
        f, eps=1.0e-5, elementwise_affine=False, dtype=torch.float64,
    )
    x = torch.randn(8, f, dtype=torch.float64)
    assert torch.allclose(affine(x), ref(x), atol=1.0e-12)


def test_affine_ln_ols_recovers_known_target() -> None:
    """If y_target = γ*·z + β* exactly, the OLS update must recover (γ*, β*)."""
    torch.manual_seed(1)
    f = 8
    affine = _AffineLN(features=f, eps=1.0e-5, dtype=torch.float64)
    x = torch.randn(64, f, dtype=torch.float64)
    z = affine._standardize(x)
    gamma_star = torch.linspace(0.5, 2.0, f, dtype=torch.float64)
    beta_star = torch.linspace(-1.0, 1.0, f, dtype=torch.float64)
    y_target = z * gamma_star + beta_star
    mse_before, mse_after = affine.update_affine_lsq(x, y_target, ridge=0.0)
    assert mse_after < mse_before
    assert torch.allclose(affine.gamma, gamma_star, atol=1.0e-9)
    assert torch.allclose(affine.beta, beta_star, atol=1.0e-9)
    assert mse_after < 1.0e-18


def test_ttblock_enable_ln_affine_off_is_unchanged_at_init() -> None:
    """`enable_ln_affine=False` (default) must be forward-equivalent with the
    legacy frozen-LN block at init (γ=1, β=0)."""
    torch.manual_seed(2)
    block = TTBlock(
        embed_dim=12, num_heads=2, hidden_dim=12,
        embed_dims=[3, 4], hidden_dims=[3, 4],
        rank=4, propagator_lam=1.0e-2, dtype=torch.float64,
    )
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    y = block(x)
    # Compare against a manual identity-LN forward (no affine).
    ln_ref = torch.nn.LayerNorm(
        12, eps=1.0e-5, elementwise_affine=False, dtype=torch.float64,
    )
    expected_x_ln1 = ln_ref(x)
    actual_x_ln1 = block.ln1(x)
    assert torch.allclose(actual_x_ln1, expected_x_ln1, atol=1.0e-12)
    assert torch.isfinite(y).all()


def test_ttblock_with_ln_affine_does_not_increase_mse() -> None:
    """Trust-region accept/revert must guarantee global MSE never increases
    when LN affine update is enabled."""
    torch.manual_seed(3)
    block_gt = TTBlock(
        embed_dim=12, num_heads=2, hidden_dim=12,
        embed_dims=[3, 4], hidden_dims=[3, 4],
        rank=4, propagator_lam=1.0e-2, dtype=torch.float64,
    )
    block = TTBlock(
        embed_dim=12, num_heads=2, hidden_dim=12,
        embed_dims=[3, 4], hidden_dims=[3, 4],
        rank=4, propagator_lam=1.0e-2, dtype=torch.float64,
        enable_ln_affine=True,
    )

    torch.manual_seed(13)
    X = torch.randn(8, 4, 12, dtype=torch.float64)
    Y_target = block_gt(X)

    initial_mse = float(torch.mean((block(X) - Y_target) ** 2).item())
    for _ in range(3):
        report = block.dmrg_step(X, Y_target, lam=1.0e-5, target_blend=0.5)
    final_mse = float(torch.mean((block(X) - Y_target) ** 2).item())

    # Strict non-increase contract from the trust-region accept/revert.
    assert final_mse <= initial_mse, (
        f"LN-affine-enabled block must not regress global MSE: "
        f"initial={initial_mse:.4e} final={final_mse:.4e}"
    )
    assert "ln_accepted" in report
    assert {"ln1", "ln2"} <= set(report["ln_accepted"].keys())


def test_ttblock_ln_affine_buffers_are_not_parameters() -> None:
    """AGENTS Constraint 1: γ, β must be **buffers** (no autograd)."""
    block = TTBlock(
        embed_dim=8, num_heads=2, hidden_dim=8,
        embed_dims=[2, 4], hidden_dims=[2, 4],
        rank=2, dtype=torch.float64, enable_ln_affine=True,
    )
    param_names = {n for n, _ in block.named_parameters()}
    assert "ln1.gamma" not in param_names
    assert "ln1.beta" not in param_names
    assert "ln2.gamma" not in param_names
    assert "ln2.beta" not in param_names
