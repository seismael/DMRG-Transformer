"""Integration tests for TTLinear (drop-in Linear replacement)."""
from __future__ import annotations

import torch
from torch import nn

from dmrg_transformer.nn import TTLinear


def test_ttlinear_shape_parity_with_nn_linear() -> None:
    torch.manual_seed(0)
    layer = TTLinear(
        in_features=64, out_features=32,
        input_dims=[8, 8], output_dims=[4, 8], rank=8,
        bias=True, dtype=torch.float64,
    )
    x = torch.randn(5, 64, dtype=torch.float64)
    y = layer(x)
    assert y.shape == (5, 32)


def test_ttlinear_forward_deterministic() -> None:
    torch.manual_seed(0)
    layer = TTLinear(
        in_features=16, out_features=16,
        input_dims=[4, 4], output_dims=[4, 4], rank=4, dtype=torch.float64,
    )
    x = torch.randn(3, 16, dtype=torch.float64)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2)


def test_ttlinear_dmrg_step_reduces_mse() -> None:
    """A single DMRG step must (weakly) reduce the training MSE (SOLVER_MATH §V)."""
    torch.manual_seed(42)
    layer = TTLinear(
        in_features=32, out_features=32,
        input_dims=[4, 8], output_dims=[4, 8], rank=8,
        bias=False, dtype=torch.float64,
    )
    X = torch.randn(128, 32, dtype=torch.float64)
    # Ground-truth rank-8 TT target.
    target_layer = TTLinear(
        in_features=32, out_features=32,
        input_dims=[4, 8], output_dims=[4, 8], rank=8,
        bias=False, dtype=torch.float64,
    )
    Y = target_layer(X)

    mse_before = float(torch.mean((layer(X) - Y) ** 2).item())
    report = layer.dmrg_step(X, Y, lam=0.0, clamp_target=False)
    mse_after = float(torch.mean((layer(X) - Y) ** 2).item())
    assert mse_after <= mse_before * (1.0 + 1.0e-6), (
        f"DMRG step did not reduce MSE: {mse_before:.3e} -> {mse_after:.3e}"
    )
    assert report.final_mse == mse_after or abs(report.final_mse - mse_after) < 1.0e-8


def test_ttlinear_parameter_count_matches_tt_formula() -> None:
    layer = TTLinear(
        in_features=64, out_features=64,
        input_dims=[8, 8], output_dims=[8, 8], rank=16,
        bias=False, dtype=torch.float32,
    )
    # Sum over d cores of r_{k-1} * p_k * r_k.
    expected = (1 * 64 * 16) + (16 * 64 * 1)
    assert layer.num_parameters == expected


def test_ttlinear_can_coexist_with_nn_linear_pipeline() -> None:
    """Smoke test: compose TTLinear with standard nn modules."""
    torch.manual_seed(7)
    layer = TTLinear(
        16, 16, input_dims=[4, 4], output_dims=[4, 4], rank=4, dtype=torch.float64,
    )
    pipeline = nn.Sequential(layer, nn.Identity())
    x = torch.randn(2, 16, dtype=torch.float64)
    y = pipeline(x)
    assert y.shape == (2, 16)
