"""Forward-parity tests for TTLinearAttention.

We verify two properties:
1. The TT-projected forward equals a hand-rolled reference linear attention
   that uses dense ``W_Q, W_K, W_V, W_out`` extracted from the TT cores via
   ``to_dense_weight()``. This is the closed-form correctness check.
2. ``φ(Q)·(φ(K)ᵀ V)`` rearrangement matches an explicit per-token
   ``Σ_k φ(Q_q)·φ(K_k) V_k / Σ_k φ(Q_q)·φ(K_k)`` ground-truth — sanity check
   that our einsum factoring did not flip an axis.
"""
from __future__ import annotations

import pytest
import torch

from dmrg_transformer.core.device import require_cuda
from dmrg_transformer.nn.tt_linear_attention import TTLinearAttention, elu_plus_one


@pytest.fixture(scope="module")
def device() -> torch.device:
    return require_cuda()


def _reference_linear_attn(
    X: torch.Tensor,
    W_Q: torch.Tensor, b_Q: torch.Tensor,
    W_K: torch.Tensor, b_K: torch.Tensor,
    W_V: torch.Tensor, b_V: torch.Tensor,
    W_out: torch.Tensor, b_out: torch.Tensor,
    num_heads: int,
    eps: float,
) -> torch.Tensor:
    """Per-token reference: out_q = (Σ_k φ(Q_q)·φ(K_k) V_k) / (Σ_k φ(Q_q)·φ(K_k))."""
    B, L, D = X.shape
    H = num_heads
    d_h = D // H
    Q = (X @ W_Q + b_Q).reshape(B, L, H, d_h).transpose(1, 2)  # [B,H,L,d_h]
    K = (X @ W_K + b_K).reshape(B, L, H, d_h).transpose(1, 2)
    V = (X @ W_V + b_V).reshape(B, L, H, d_h).transpose(1, 2)

    phiQ = elu_plus_one(Q)                                         # [B,H,L,d_h]
    phiK = elu_plus_one(K)                                         # [B,H,L,d_h]

    out = torch.zeros_like(V)                                      # [B,H,L,d_h]
    for q in range(L):
        # numerator over k
        num = torch.zeros(B, H, d_h, dtype=X.dtype, device=X.device)
        denom = torch.zeros(B, H, dtype=X.dtype, device=X.device)
        for k in range(L):
            wt = (phiQ[:, :, q, :] * phiK[:, :, k, :]).sum(dim=-1)  # [B,H]
            num = num + wt.unsqueeze(-1) * V[:, :, k, :]
            denom = denom + wt
        out[:, :, q, :] = num / (denom.unsqueeze(-1) + eps)

    out = out.transpose(1, 2).reshape(B, L, D)
    return out @ W_out + b_out


def test_forward_matches_dense_reference(device: torch.device) -> None:
    embed_dim, num_heads, rank = 8, 2, 4
    input_dims = output_dims = [4, 2]
    eps = 1.0e-6

    torch.manual_seed(0)
    mod = TTLinearAttention(
        embed_dim, num_heads,
        input_dims=input_dims, output_dims=output_dims,
        rank=rank, dtype=torch.float64, eps=eps,
    )

    B, L = 4, 5
    X = torch.randn(B, L, embed_dim, dtype=torch.float64, device=device)

    # Extract dense weights / biases from the TT modules.
    W_Q = mod.W_Q.to_dense_weight();  b_Q = mod.W_Q._bias
    W_K = mod.W_K.to_dense_weight();  b_K = mod.W_K._bias
    W_V = mod.W_V.to_dense_weight();  b_V = mod.W_V._bias
    W_O = mod.W_out.to_dense_weight(); b_O = mod.W_out._bias

    ref = _reference_linear_attn(
        X, W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O,
        num_heads=num_heads, eps=eps,
    )
    got = mod(X)

    assert ref.shape == got.shape
    # float64 + multiple einsums: tight but not machine precision.
    err = float((ref - got).abs().max().item())
    assert err < 1.0e-9, f"max abs error {err} exceeds tolerance"


def test_forward_invariant_under_permutation_in_kv_axis(device: torch.device) -> None:
    """Linear attention is permutation-equivariant on the key/value axis modulo
    the corresponding query rebind. As a quick sanity check, swapping K and V
    rows together must permute the rows of the output the same way the queries
    are permuted (here we check K=Q so a permutation of all three permutes
    the output rows)."""
    embed_dim, num_heads, rank = 8, 2, 4
    input_dims = output_dims = [4, 2]

    torch.manual_seed(1)
    mod = TTLinearAttention(
        embed_dim, num_heads,
        input_dims=input_dims, output_dims=output_dims,
        rank=rank, dtype=torch.float64,
    )
    B, L = 2, 4
    X = torch.randn(B, L, embed_dim, dtype=torch.float64, device=device)
    perm = torch.tensor([2, 0, 3, 1], device=device)

    y_orig = mod(X)
    y_perm = mod(X[:, perm, :])

    # Permuting all three (Q==K==V==X) must permute the output rows identically.
    assert torch.allclose(y_perm, y_orig[:, perm, :], atol=1.0e-10)
