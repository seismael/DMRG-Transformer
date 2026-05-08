"""TT-GPT2 — GPT-2-style decoder-only Transformer with TT-factorized weights.

Components
---------
* ``TTCausalSelfAttention`` — ``TTMultiHeadAttention`` + causal mask
* ``TTDecoderBlock`` — Pre-LN decoder block (causal attn + FFN)
* ``TTGPT2Model`` — stack of decoder blocks + embedding + positional encoding
* ``TTGPT2LMHead`` — model + TT-factorized language-modelling head

All linear projections (Q/K/V/W_out in attention, fc1/fc2 in FFN, LM head)
use :class:`TTLinear` — gradient-free, exact-solver trainable via
:meth:`dmrg_step`.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dmrg_transformer.nn.embeddings import PositionalEncoding
from dmrg_transformer.nn.tt_block import TTBlock
from dmrg_transformer.nn.tt_ffn import TTFeedForward
from dmrg_transformer.nn.tt_linear import TTLinear
from dmrg_transformer.nn.tt_mha import TTMultiHeadAttention
from dmrg_transformer.optim.sweep import SweepReport


# ── Causal Self-Attention ────────────────────────────────────────────────────


class TTCausalSelfAttention(TTMultiHeadAttention):
    """Multi-head self-attention with causal (lower-triangular) masking.

    Identical to :class:`TTMultiHeadAttention` except :meth:`forward` applies
    a causal mask so each position can only attend to earlier positions.
    """

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L_q, _ = x.shape

        Q = self._project(self.W_Q, x)
        K = self._project(self.W_K, x)
        V = self._project(self.W_V, x)

        Q = Q.reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim**-0.5
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * scale  # [B, H, L, L]

        # Causal mask: zero out positions j > i.
        causal_mask = torch.triu(
            torch.ones(L_q, L_q, dtype=scores.dtype, device=scores.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.bool(), float("-inf"))

        attn_w = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhqk,bhkd->bhqd", attn_w, V)
        context = context.transpose(1, 2).reshape(B, L_q, self.embed_dim)
        return self._project(self.W_out, context)


# ── Decoder Block ────────────────────────────────────────────────────────────


class TTDecoderBlock(nn.Module):
    """Pre-LN GPT-2 decoder block.

    ::

        h = x + causal_attn(ln1(x))
        y = h + ffn(ln2(h))

    Uses :class:`TTCausalSelfAttention` for masked self-attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        *,
        embed_dims: list[int],
        hidden_dims: list[int],
        rank: int,
        propagator_lam: float = 1.0e-2,
        ln_eps: float = 1.0e-5,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ln1 = nn.LayerNorm(embed_dim, eps=ln_eps, dtype=dtype)
        self.ln2 = nn.LayerNorm(embed_dim, eps=ln_eps, dtype=dtype)
        self.attn = TTCausalSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            input_dims=embed_dims, output_dims=embed_dims,
            rank=rank, dtype=dtype,
        )
        self.ffn = TTFeedForward(
            embed_dim=embed_dim, hidden_dim=hidden_dim,
            embed_dims=embed_dims, hidden_dims=hidden_dims,
            rank=rank, propagator_lam=propagator_lam, dtype=dtype,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln1(x))
        return h + self.ffn(self.ln2(h))

    @torch.no_grad()
    def forward_with_cache(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        ln1 = self.ln1(x)
        attn_out = self.attn(ln1)
        h = x + attn_out
        ln2 = self.ln2(h)
        ffn_out = self.ffn(ln2)
        y = h + ffn_out
        return {"x": x, "ln1": ln1, "attn_out": attn_out, "h": h,
                "ln2": ln2, "ffn_out": ffn_out, "y": y}

    @torch.no_grad()
    def dmrg_step(
        self,
        X: torch.Tensor,
        Y_target: torch.Tensor,
        *,
        lam: float = 1.0e-5,
        target_blend: float = 0.5,
        adaptive_threshold: float | None = None,
    ) -> dict[str, object]:
        """Exact-solver update for one decoder block.

        Follows the same 10-step pipeline as :class:`TTBlock.dmrg_step`,
        but uses causal self-attention.
        """
        # Delegate to a temporary TTBlock with the same architecture.
        # This is a pragmatic shortcut — a dedicated decoder-block solver
        # would mirror tt_block.py's 10-step pipeline with causal masking.
        cache = self.forward_with_cache(X)
        global_mse_before = float(torch.mean((cache["y"] - Y_target) ** 2).item())

        # FFN sweep (same as TTBlock).
        from dmrg_transformer.propagation.target_propagator import TargetPropagator
        prop = TargetPropagator(lam=lam if lam > 0 else 1e-5)

        ffn_target = prop.project_through_residual(Y_target, cache["h"])
        ffn_reports = self.ffn.dmrg_step(
            cache["ln2"].reshape(-1, self.embed_dim),
            ffn_target.reshape(-1, self.embed_dim),
            lam=lam, target_blend=target_blend,
            adaptive_threshold=adaptive_threshold,
        )

        # Re-cache; derive attn_out target.
        cache_mid = self.forward_with_cache(X)
        h_target_full = Y_target - cache_mid["ffn_out"]
        h_target = target_blend * h_target_full + (1.0 - target_blend) * cache_mid["h"]
        attn_out_target = prop.project_through_residual(h_target, cache["x"])

        # Q/K/V/W_out via causal attention — reuse existing Q/K/V steps.
        B, L, _ = X.shape
        H = self.attn.num_heads
        d_h = self.attn.head_dim
        x_ln1_flat = cache["ln1"].reshape(-1, self.embed_dim)

        Q_curr = self.attn.W_Q(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        K_curr = self.attn.W_K(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        V_curr = self.attn.W_V(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        scale = d_h**-0.5
        scores_curr = torch.einsum("bhqd,bhkd->bhqk", Q_curr, K_curr) * scale
        causal_mask = torch.triu(torch.ones(L, L, dtype=scores_curr.dtype, device=scores_curr.device), diagonal=1)
        scores_curr = scores_curr.masked_fill(causal_mask.bool(), float("-inf"))
        attn_w_curr = torch.softmax(scores_curr, dim=-1)
        context_curr = torch.einsum("bhqk,bhkd->bhqd", attn_w_curr, V_curr)
        context_full_curr = context_curr.transpose(1, 2).reshape(B, L, self.embed_dim)

        # W_out sweep.
        rep_Wout = self.attn.W_out.dmrg_step(
            context_full_curr.reshape(-1, self.embed_dim),
            attn_out_target.reshape(-1, self.embed_dim),
            lam=lam, adaptive_threshold=adaptive_threshold,
        )

        # Pull context target through updated W_out.
        Wd = self.attn.W_out.to_dense_weight()
        t = attn_out_target.reshape(-1, self.embed_dim)
        if self.attn.W_out._has_bias:
            t = t - self.attn.W_out._bias
        context_target_full = prop.project_through_linear(Wd, t).reshape(B, L, self.embed_dim)
        context_target_heads = context_target_full.reshape(B, L, H, d_h).transpose(1, 2)

        # Q/K update with trust-region (simplified — same as TTBlock).
        mse_before_qk = float(torch.mean((self.forward_with_cache(X)["y"] - Y_target) ** 2).item())
        snap_Q = {k: v.detach().clone() for k, v in self.attn.W_Q.state_dict().items()}
        snap_K = {k: v.detach().clone() for k, v in self.attn.W_K.state_dict().items()}

        A_target = prop.solve_attention_pattern_target(V_curr, context_target_heads, eps=1e-12)
        attn_blend = 0.05
        A_blended = attn_blend * A_target + (1.0 - attn_blend) * attn_w_curr
        scores_target = prop.softmax_target_to_scores(
            A_blended, A_curr=attn_w_curr, S_curr=scores_curr, scale=1.0 / scale,
        )
        Q_tgt, _ = prop.project_through_qk_bilinear(scores_target, Q_curr, K_curr)
        _, K_tgt = prop.project_through_qk_bilinear(scores_target, Q_tgt, K_curr)

        self.attn.dmrg_step_projections(
            cache["ln1"],
            Q_tgt.transpose(1, 2).reshape(B, L, self.embed_dim),
            K_tgt.transpose(1, 2).reshape(B, L, self.embed_dim),
            None, lam=lam, adaptive_threshold=adaptive_threshold,
        )

        mse_after_qk = float(torch.mean((self.forward_with_cache(X)["y"] - Y_target) ** 2).item())
        qk_accepted = mse_after_qk <= 1.01 * mse_before_qk + 1e-4
        if not qk_accepted:
            self.attn.W_Q.load_state_dict(snap_Q)
            self.attn.W_K.load_state_dict(snap_K)
            mse_after_qk = mse_before_qk

        # V update with trust-region.
        snap_V = {k: v.detach().clone() for k, v in self.attn.W_V.state_dict().items()}
        Q_now = self.attn.W_Q(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        K_now = self.attn.W_K(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        A_now = torch.softmax(
            torch.einsum("bhqd,bhkd->bhqk", Q_now, K_now) * scale, dim=-1,
        )
        A_now = A_now.masked_fill(causal_mask.bool(), 0.0)
        V_tgt_heads = prop.project_through_attention_v(A_now, context_target_heads)
        Y_V_tgt = V_tgt_heads.transpose(1, 2).reshape(B, L, self.embed_dim)
        self.attn.dmrg_step_projections(
            cache["ln1"], None, None, Y_V_tgt, lam=lam,
            adaptive_threshold=adaptive_threshold,
        )
        mse_after_v = float(torch.mean((self.forward_with_cache(X)["y"] - Y_target) ** 2).item())
        v_accepted = mse_after_v <= 1.01 * max(mse_after_qk, 1e-10)
        if not v_accepted:
            self.attn.W_V.load_state_dict(snap_V)

        cache_after = self.forward_with_cache(X)
        global_mse_after = float(torch.mean((cache_after["y"] - Y_target) ** 2).item())

        return {
            "ffn": ffn_reports,
            "attn": {
                "W_out": rep_Wout.final_mse,
                "accepted": qk_accepted or v_accepted,
                "diagnostics": {"qk_accepted": qk_accepted, "v_accepted": v_accepted,
                                "mse_before": global_mse_before, "mse_after_v": mse_after_v},
            },
            "global_mse_before": global_mse_before,
            "global_mse_after": global_mse_after,
        }


# ── GPT-2 Model ──────────────────────────────────────────────────────────────


class TTGPT2Model(nn.Module):
    """GPT-2-style decoder-only Transformer with TT-factorized weights.

    Args:
        vocab_size: size of the token vocabulary.
        embed_dim: model dimension.
        num_heads: attention heads per block.
        hidden_dim: FFN inner dimension.
        num_layers: number of decoder blocks.
        max_seq_len: maximum sequence length (for positional encoding).
        embed_dims: TT factorisation of ``embed_dim``.
        hidden_dims: TT factorisation of ``hidden_dim``.
        rank: TT-rank bound for all linear projections.
        dtype: storage dtype.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        max_seq_len: int,
        *,
        embed_dims: list[int],
        hidden_dims: list[int],
        rank: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, dtype=dtype)
        self.positional = PositionalEncoding(embed_dim, max_len=max_seq_len, dtype=dtype)
        self.blocks = nn.ModuleList([
            TTDecoderBlock(
                embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim,
                embed_dims=embed_dims, hidden_dims=hidden_dims,
                rank=rank, dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim, dtype=dtype)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` token ids → ``[B, L, embed_dim]`` hidden states."""
        x = self.token_embedding(input_ids)
        x = self.positional(x)
        for block in self.blocks:
            x = block.forward(x)
        return self.ln_f(x)


class TTGPT2LMHead(nn.Module):
    """TT-GPT2 with a dense language-modelling head.

    The LM head is a standard :class:`nn.Linear` (dense, not TT-factorized)
    because the ``embed_dim × vocab_size`` matrix is already small for pico/
    nano-scale models.  The head is fitted via exact least-squares in the
    training loop — no gradient required.
    """

    def __init__(self, model: TTGPT2Model) -> None:
        super().__init__()
        self.model = model
        vocab_size = model.token_embedding.num_embeddings
        embed_dim = model.embed_dim
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, dtype=model.token_embedding.weight.dtype)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` → ``[B, L, vocab_size]`` logits."""
        hidden = self.model.forward(input_ids)
        return hidden @ self.lm_head.weight.T  # [B, L, V]

    @torch.no_grad()
    def fit_head(self, hidden: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Fit the LM head via exact least-squares.

        Args:
            hidden: ``[B*L, D]`` flattened hidden states.
            target_ids: ``[B*L]`` target token ids.

        Returns:
            Cross-entropy loss after fitting.
        """
        V = self.lm_head.weight.shape[0]  # vocab_size
        Y = torch.zeros(hidden.shape[0], V, dtype=hidden.dtype, device=hidden.device)
        target_ids = target_ids.to(device=hidden.device)
        Y.scatter_(1, target_ids.unsqueeze(-1), 1.0)
        # Exact LSQ: W = (H^T H + λI)^-1 H^T Y
        lam = 1e-5
        D = hidden.shape[-1]
        gram = hidden.T @ hidden + lam * torch.eye(D, dtype=hidden.dtype, device=hidden.device)
        self.lm_head.weight.data = torch.linalg.solve(gram, hidden.T @ Y).T
        # Return CE for monitoring.
        logits = hidden @ self.lm_head.weight.T
        ce = torch.nn.functional.cross_entropy(logits, target_ids)
        return float(ce.item())


# ── Helpers ──────────────────────────────────────────────────────────────────


def tt_gpt2_pico(
    vocab_size: int = 1000,
    dtype: torch.dtype = torch.float64,
) -> TTGPT2LMHead:
    """Create a TT-GPT2 *pico* model (~100 k TT params, trainable on 2 GiB GPU).

    2 layers, embed=64, heads=2, hidden=256, max_seq_len=128.
    """
    embed_dims = [8, 8]
    hidden_dims = [16, 16]
    model = TTGPT2Model(
        vocab_size=vocab_size, embed_dim=64, num_heads=2, hidden_dim=256,
        num_layers=2, max_seq_len=128,
        embed_dims=embed_dims, hidden_dims=hidden_dims,
        rank=4, dtype=dtype,
    )
    return TTGPT2LMHead(model)
