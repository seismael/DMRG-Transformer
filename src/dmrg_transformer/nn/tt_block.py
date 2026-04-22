"""``TTBlock`` — Pre-LN Transformer encoder block with TT-factorized linears.

Composition (Pre-LN):

    h = x + attn(LN1(x))           # attention sub-block
    y = h + ffn(LN2(h))            # feed-forward sub-block

LayerNorm affine parameters are **frozen at the identity** (γ=1, β=0) for this
slice — see [docs/COMPLIANCE.md](../../../docs/COMPLIANCE.md) §C3 notes. This
keeps the LN pull-back exact under the saved row statistics; updating the
affine params via least-squares is a follow-up refinement.

DMRG update sequence (per call to :meth:`dmrg_step`):

1. Forward with cache → ``(x, x_ln1, attn_out, h, h_ln2, ffn_out)``.
2. Pull ``Y_target`` through residual #2 → target for ``ffn(LN2(h))``.
3. Sweep ``ffn`` against that target.
4. Pull ``ffn`` target back through ``fc1`` (post-LN target) and through LN2
   to get a target for ``h``.
5. Pull through residual #1 → target for ``attn(LN1(x))``.
6. Pull through ``W_out`` → target for the V-projection branch.
7. Sweep ``attn`` projections (Q/K targets frozen at current outputs;
   only V is meaningfully updated). This is the *honest deferral* —
   Q/K propagation through softmax is the dominant root cause of any
   measured DMRG-vs-Adam gap on stacked blocks.
"""
from __future__ import annotations

import torch
from torch import nn

from dmrg_transformer.nn.tt_ffn import TTFeedForward
from dmrg_transformer.nn.tt_mha import TTMultiHeadAttention
from dmrg_transformer.propagation.target_propagator import TargetPropagator


class TTBlock(nn.Module):
    """Pre-LN Transformer encoder block (TT-MHA + TT-FFN with frozen LN)."""

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
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ln_eps = ln_eps

        # LN affine params frozen at identity (γ=1, β=0) for this slice — see
        # docs/COMPLIANCE.md §C3 notes. We use elementwise_affine=False to make
        # the freeze structural rather than a parameter we forget to freeze.
        self.ln1 = nn.LayerNorm(
            embed_dim, eps=ln_eps, elementwise_affine=False, dtype=dtype,
        )
        self.ln2 = nn.LayerNorm(
            embed_dim, eps=ln_eps, elementwise_affine=False, dtype=dtype,
        )
        self.attn = TTMultiHeadAttention(
            embed_dim, num_heads,
            input_dims=embed_dims, output_dims=embed_dims,
            rank=rank, dtype=dtype,
        )
        self.ffn = TTFeedForward(
            embed_dim, hidden_dim,
            embed_dims=embed_dims, hidden_dims=hidden_dims,
            rank=rank, propagator_lam=propagator_lam, dtype=dtype,
        )
        self.propagator = TargetPropagator(lam=propagator_lam)

    # -- forward ---------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ln1 = self.ln1(x)
        attn_out = self.attn(x_ln1)
        h = x + attn_out
        h_ln2 = self.ln2(h)
        ffn_out = self.ffn(h_ln2)
        return h + ffn_out

    @torch.no_grad()
    def forward_with_cache(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward returning every intermediate tensor needed by ``dmrg_step``."""
        x_ln1 = self.ln1(x)
        attn_out = self.attn(x_ln1)
        h = x + attn_out
        h_ln2 = self.ln2(h)
        ffn_out = self.ffn(h_ln2)
        y = h + ffn_out
        return {
            "x": x, "x_ln1": x_ln1, "attn_out": attn_out,
            "h": h, "h_ln2": h_ln2, "ffn_out": ffn_out, "y": y,
        }

    # -- DMRG update -----------------------------------------------------------

    @torch.no_grad()
    def dmrg_step(
        self,
        X: torch.Tensor,
        Y_target: torch.Tensor,
        *,
        lam: float = 1.0e-5,
        target_blend: float = 0.5,
    ) -> dict[str, object]:
        """Exact-solver update for one Pre-LN block.

        **Honest scope (this slice):** only the strictly-linear sub-paths are
        swept — ``W_out`` (attention output projection) and the FFN's two TT
        linears. The ``Q/K/V`` projections are left frozen because pulling a
        target through ``softmax(QK^T)V`` requires a non-trivial
        linearization not yet implemented; updating Q/K/V with a stale
        target through the wrong Jacobian destabilizes the sweep (verified
        empirically — see [tests/test_tt_block.py](../../../tests/test_tt_block.py)).

        Args:
            X: ``[batch, seq, embed_dim]`` input.
            Y_target: ``[batch, seq, embed_dim]`` target for the block output.
            lam: Tikhonov damping inside each TT linear sweep.
            target_blend: blending factor for intermediate-target damping.

        Returns:
            Dict with ``"ffn"`` (per-sublayer SweepReports), ``"attn"``
            (only ``W_out`` final MSE; Q/K/V marked ``"frozen"``), and
            ``"global_mse_before" / "global_mse_after"``.
        """
        cache = self.forward_with_cache(X)
        global_mse_before = float(torch.mean((cache["y"] - Y_target) ** 2).item())

        # 1) Pull through residual #2: target for ffn_out.
        ffn_target = self.propagator.project_through_residual(Y_target, cache["h"])

        # 2) Sweep FFN (both fc1 and fc2, with internal target propagation).
        ffn_reports = self.ffn.dmrg_step(
            cache["h_ln2"].reshape(-1, self.embed_dim),
            ffn_target.reshape(-1, self.embed_dim),
            lam=lam, target_blend=target_blend,
        )

        # 3) Re-forward to get the post-ffn residual sum after the FFN update,
        #    so the W_out sweep targets the most up-to-date attn_out target.
        cache_mid = self.forward_with_cache(X)
        # Target for ffn_out is unchanged (residual pull-back from Y_target through h).
        # Now compute target for h (which equals x + attn_out): we want
        # h + ffn_new(LN2(h)) ≈ Y_target. With ffn updated, the target for h
        # collapses to: h_target = Y_target - ffn_new(LN2(h_current)) (one-step
        # local linearization), blended with current h.
        h_target_full = Y_target - cache_mid["ffn_out"]
        h_target = target_blend * h_target_full + (1.0 - target_blend) * cache_mid["h"]

        # 4) Pull through residual #1: target for attn_out.
        attn_out_target = self.propagator.project_through_residual(h_target, cache["x"])

        # 5) Recompute the context (softmax(QK^T)V) — this is W_out's input.
        #    Q/K/V are frozen this slice so we just rerun the head contraction.
        x_ln1_flat = cache["x_ln1"].reshape(-1, self.embed_dim)
        B, L, _ = cache["x_ln1"].shape
        Q = self.attn.W_Q(x_ln1_flat).reshape(
            B, L, self.attn.num_heads, self.attn.head_dim,
        ).transpose(1, 2)
        K = self.attn.W_K(x_ln1_flat).reshape(
            B, L, self.attn.num_heads, self.attn.head_dim,
        ).transpose(1, 2)
        V = self.attn.W_V(x_ln1_flat).reshape(
            B, L, self.attn.num_heads, self.attn.head_dim,
        ).transpose(1, 2)
        scale = self.attn.head_dim ** -0.5
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * scale
        attn_w = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhqk,bhkd->bhqd", attn_w, V).transpose(1, 2).reshape(
            B, L, self.embed_dim,
        )
        context_flat = context.reshape(-1, self.embed_dim)

        # 6) Sweep W_out: this is a pure linear least-squares step with
        #    context as input and attn_out_target as target — guaranteed
        #    not to increase the local MSE.
        rep_Wout = self.attn.W_out.dmrg_step(
            context_flat,
            attn_out_target.reshape(-1, self.embed_dim),
            lam=lam,
        )

        cache_after = self.forward_with_cache(X)
        global_mse_after = float(torch.mean((cache_after["y"] - Y_target) ** 2).item())

        return {
            "ffn": ffn_reports,
            "attn": {
                "Q": "frozen", "K": "frozen", "V": "frozen",
                "W_out": rep_Wout.final_mse,
            },
            "global_mse_before": global_mse_before,
            "global_mse_after": global_mse_after,
        }

    @property
    def num_parameters(self) -> int:
        return (
            self.attn.W_Q.num_parameters + self.attn.W_K.num_parameters
            + self.attn.W_V.num_parameters + self.attn.W_out.num_parameters
            + self.ffn.num_parameters
        )
