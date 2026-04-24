"""Integration tests for TTMultiHeadAttention."""
from __future__ import annotations

import torch

from dmrg_transformer.nn import TTMultiHeadAttention


def test_ttmha_shape_parity() -> None:
    torch.manual_seed(0)
    mha = TTMultiHeadAttention(
        embed_dim=32, num_heads=4,
        input_dims=[4, 8], output_dims=[4, 8], rank=8,
        dtype=torch.float64,
    )
    x = torch.randn(2, 5, 32, dtype=torch.float64)
    y = mha(x)
    assert y.shape == (2, 5, 32)


def test_ttmha_self_attention_is_finite() -> None:
    torch.manual_seed(1)
    mha = TTMultiHeadAttention(
        embed_dim=16, num_heads=2,
        input_dims=[4, 4], output_dims=[4, 4], rank=4,
        dtype=torch.float64,
    )
    x = torch.randn(1, 3, 16, dtype=torch.float64)
    y = mha(x)
    assert torch.isfinite(y).all()


def test_ttmha_dmrg_step_reduces_projection_mse() -> None:
    """Each per-projection DMRG step must weakly reduce its MSE."""
    torch.manual_seed(5)
    mha = TTMultiHeadAttention(
        embed_dim=16, num_heads=2,
        input_dims=[4, 4], output_dims=[4, 4], rank=4,
        dtype=torch.float64,
    )
    # Flat pseudo-batch of token embeddings.
    X = torch.randn(64, 16, dtype=torch.float64)
    # Targets: just pass through a second (random) set of TT projections to get valid
    # rank-matched targets.
    gt = TTMultiHeadAttention(
        embed_dim=16, num_heads=2,
        input_dims=[4, 4], output_dims=[4, 4], rank=4,
        dtype=torch.float64,
    )
    Y_Q = gt.W_Q(X)
    Y_K = gt.W_K(X)
    Y_V = gt.W_V(X)

    mse_before = float(torch.mean((mha.W_Q(X) - Y_Q) ** 2).item())
    results = mha.dmrg_step_projections(X, Y_Q, Y_K, Y_V, lam=0.0)
    mse_after = float(torch.mean((mha.W_Q(X) - Y_Q) ** 2).item())
    assert mse_after <= mse_before * (1.0 + 1.0e-6)
    assert set(results.keys()) == {"Q", "K", "V"}


def test_ttmha_dmrg_step_can_skip_untouched_projections() -> None:
    torch.manual_seed(9)
    mha = TTMultiHeadAttention(
        embed_dim=16, num_heads=2,
        input_dims=[4, 4], output_dims=[4, 4], rank=4,
        dtype=torch.float64,
    )
    gt = TTMultiHeadAttention(
        embed_dim=16, num_heads=2,
        input_dims=[4, 4], output_dims=[4, 4], rank=4,
        dtype=torch.float64,
    )
    X = torch.randn(32, 16, dtype=torch.float64)
    Y_K = gt.W_K(X)

    mse_before = float(torch.mean((mha.W_K(X) - Y_K) ** 2).item())
    results = mha.dmrg_step_projections(X, None, Y_K, None, lam=0.0)
    mse_after = float(torch.mean((mha.W_K(X) - Y_K) ** 2).item())

    assert set(results.keys()) == {"K"}
    assert mse_after <= mse_before * (1.0 + 1.0e-6)
