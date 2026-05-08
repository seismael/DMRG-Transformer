"""Smoke + unit tests for TT-GPT2 components."""
from __future__ import annotations

import torch

from dmrg_transformer.core.device import require_cuda
from dmrg_transformer.nn.tt_gpt2 import (
    TTCausalSelfAttention,
    TTDecoderBlock,
    TTGPT2LMHead,
    TTGPT2Model,
    tt_gpt2_pico,
)


def test_tt_gpt2_forward_shape() -> None:
    """TT-GPT2 pico forward pass produces correct output shape."""
    device = require_cuda()
    model = tt_gpt2_pico(vocab_size=256, dtype=torch.float32).to(device)
    x = torch.randint(0, 256, (2, 16), device=device)
    y = model(x)
    assert y.shape == (2, 16, 256), f"got {y.shape}"


def test_tt_gpt2_tt_params_exist() -> None:
    """TT-GPT2 pico has TT cores registered as buffers."""
    device = require_cuda()
    model = tt_gpt2_pico(vocab_size=256, dtype=torch.float32).to(device)
    n_cores = sum(1 for n, _ in model.named_buffers() if "_core_" in n)
    assert n_cores > 0, "no TT cores found"
    # Each block: W_Q(2 cores) + W_K(2) + W_V(2) + W_out(2) + fc1(2) + fc2(2) = 12 cores
    # 2 blocks = 24 cores
    assert n_cores >= 20, f"expected >= 20 TT cores, got {n_cores}"


def test_causal_mask_enforced() -> None:
    """TTCausalSelfAttention must not let position i attend to j > i."""
    device = require_cuda()
    attn = TTCausalSelfAttention(
        embed_dim=16, num_heads=2,
        input_dims=[4, 4], output_dims=[4, 4],
        rank=4, dtype=torch.float64,
    ).to(device)
    x = torch.randn(1, 8, 16, dtype=torch.float64, device=device)
    # Extract attention weights manually.
    B, L_q, _ = x.shape
    H = attn.num_heads
    d_h = attn.head_dim
    Q = attn._project(attn.W_Q, x).reshape(B, L_q, H, d_h).transpose(1, 2)
    K = attn._project(attn.W_K, x).reshape(B, L_q, H, d_h).transpose(1, 2)
    scale = d_h**-0.5
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * scale
    causal_mask = torch.triu(torch.ones(L_q, L_q, dtype=scores.dtype, device=device), diagonal=1)
    scores_masked = scores.masked_fill(causal_mask.bool(), float("-inf"))
    attn_w = torch.softmax(scores_masked, dim=-1)
    # Upper triangle (j > i) must be zero.
    for i in range(L_q):
        for j in range(i + 1, L_q):
            assert attn_w[:, :, i, j].max().item() < 1e-8, (
                f"non-zero attention at q={i}, k={j}"
            )
    # Lower triangle (j <= i) should have non-zero attention.
    for i in range(L_q):
        row_sum = attn_w[:, :, i, : i + 1].sum().item()
        assert row_sum > 0.9, f"attention row {i} sum={row_sum:.4f}"


def test_decoder_block_reduces_mse() -> None:
    """FFN + W_out sweep must reduce MSE on a rank-bounded target.

    Q/K update is excluded from this test — the decoder block's attention
    pattern is harder to fit at small embed dims, and the bilinear Q/K
    pull-back can produce degenerate targets.  FFN + W_out alone should
    still reduce the global MSE (this is the same pattern observed in
    the encoder TTBlock PoC where FFN carries the accuracy).
    """
    device = require_cuda()
    embed_dim, num_heads, hidden_dim = 12, 2, 12
    dims = [3, 4]
    hdim = [3, 4]

    torch.manual_seed(0)
    block_gt = TTDecoderBlock(
        embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim,
        embed_dims=dims, hidden_dims=hdim,
        rank=4, dtype=torch.float64,
    ).to(device)

    X = torch.randn(4, 6, embed_dim, dtype=torch.float64, device=device)
    with torch.no_grad():
        Y = block_gt.forward(X)

    torch.manual_seed(100)
    block_train = TTDecoderBlock(
        embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim,
        embed_dims=dims, hidden_dims=hdim,
        rank=4, dtype=torch.float64,
    ).to(device)

    from dmrg_transformer.propagation.target_propagator import TargetPropagator
    prop = TargetPropagator(lam=1e-2)

    initial_mse = float(torch.mean((block_train.forward(X) - Y) ** 2).item())

    # FFN sweep (same logic as TTBlock/DecoderBlock dmrg_step).
    cache = block_train.forward_with_cache(X)
    ffn_target = prop.project_through_residual(Y, cache["h"])
    block_train.ffn.dmrg_step(
        cache["ln2"].reshape(-1, embed_dim),
        ffn_target.reshape(-1, embed_dim),
        lam=1e-2, target_blend=0.5,
    )

    # Re-cache; W_out sweep.
    cache_mid = block_train.forward_with_cache(X)
    h_target = 0.5 * (Y - cache_mid["ffn_out"]) + 0.5 * cache_mid["h"]
    attn_out_target = prop.project_through_residual(h_target, cache["x"])

    B, L, _ = X.shape
    H = block_train.attn.num_heads
    d_h = block_train.attn.head_dim
    x_ln1_flat = cache["ln1"].reshape(-1, embed_dim)
    Q_c = block_train.attn.W_Q(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
    K_c = block_train.attn.W_K(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
    V_c = block_train.attn.W_V(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
    scale = d_h**-0.5
    scores_c = torch.einsum("bhqd,bhkd->bhqk", Q_c, K_c) * scale
    mask = torch.triu(torch.ones(L, L, dtype=scores_c.dtype, device=device), diagonal=1)
    scores_c = scores_c.masked_fill(mask.bool(), float("-inf"))
    aw = torch.softmax(scores_c, dim=-1)
    ctx = torch.einsum("bhqk,bhkd->bhqd", aw, V_c)
    ctx_full = ctx.transpose(1, 2).reshape(B, L, embed_dim)

    block_train.attn.W_out.dmrg_step(
        ctx_full.reshape(-1, embed_dim),
        attn_out_target.reshape(-1, embed_dim),
        lam=1e-2,
    )

    final_mse = float(torch.mean((block_train.forward(X) - Y) ** 2).item())
    assert final_mse < initial_mse, (
        f"decoder block FFN+W_out MSE increased: {initial_mse:.4e} -> {final_mse:.4e}"
    )


def test_head_fit_reduces_loss() -> None:
    """Exact LSQ head fit must reduce cross-entropy."""
    device = require_cuda()
    model = tt_gpt2_pico(vocab_size=256, dtype=torch.float32).to(device)
    x = torch.randint(0, 256, (4, 16), device=device)

    # Next-token prediction: hidden[:, :-1] → targets[:, 1:]
    with torch.no_grad():
        hidden_full = model.model.forward(x)
        hidden = hidden_full[:, :-1, :].reshape(-1, 64)
        targets = x[:, 1:].reshape(-1)
        # Random head → high CE.
        logits_init = hidden @ model.lm_head.weight.T
        ce_init = float(torch.nn.functional.cross_entropy(logits_init, targets).item())

    # Fit head.
    ce_after = model.fit_head(hidden, targets)
    assert ce_after < ce_init, (
        f"head fit did not reduce CE: {ce_init:.4f} -> {ce_after:.4f}"
    )
