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
        attn_target_blend: float | None = None,
    ) -> dict[str, object]:
        """Exact-solver update for one Pre-LN block (full Q/K/V/W_out + FFN).

        Update sequence:

        1. Forward with cache.
        2. Pull ``Y_target`` through residual #2 → FFN target. Sweep FFN.
        3. Re-cache; pull ``Y_target - ffn_out_new`` through residual #1 →
           target for ``attn_out``.
        4. Pull ``attn_out_target`` back through ``W_out`` (Tikhonov pseudo-
           inverse) to get a per-head ``context_target``.
        5. Pull ``context_target`` back to ``A_target`` (target attention
           pattern) using current ``V`` via
           :meth:`TargetPropagator.solve_attention_pattern_target`.
        6. Invert softmax to get ``scores_target = √d_h · log(A_target)``
           via :meth:`TargetPropagator.softmax_target_to_scores`.
        7. Pull ``scores_target`` back to ``Q_target / K_target`` via the
           bilinear solver
           :meth:`TargetPropagator.project_through_qk_bilinear`.
        8. Pull ``context_target`` back to ``V_target`` through the *target*
           attention pattern via
           :meth:`TargetPropagator.project_through_attention_v`.
        9. Sweep ``W_out`` (input = current context).
        10. Sweep ``W_Q, W_K, W_V`` against their targets (parallel streams
            on GPU per ``MEMORY_ARENA §5``).

        All targets are blended with the current value at ``target_blend``
        for damping (the joint Q,K problem is non-convex; per-step blending
        prevents oscillation).

        Args:
            X: ``[batch, seq, embed_dim]`` input.
            Y_target: ``[batch, seq, embed_dim]`` target for the block output.
            lam: Tikhonov damping for each TT linear sweep.
            target_blend: blending factor for intermediate targets ``∈ (0,1]``.

        Returns:
            Dict with ``"ffn"`` (per-sublayer SweepReports), ``"attn"`` (per-
            projection final MSEs for Q, K, V, W_out), and
            ``"global_mse_before" / "global_mse_after"``.
        """
        cache = self.forward_with_cache(X)
        global_mse_before = float(torch.mean((cache["y"] - Y_target) ** 2).item())

        # Step 1-2: FFN sweep.
        ffn_target = self.propagator.project_through_residual(Y_target, cache["h"])
        ffn_reports = self.ffn.dmrg_step(
            cache["h_ln2"].reshape(-1, self.embed_dim),
            ffn_target.reshape(-1, self.embed_dim),
            lam=lam, target_blend=target_blend,
        )

        # Step 3: re-cache with updated FFN; derive attn_out target.
        cache_mid = self.forward_with_cache(X)
        h_target_full = Y_target - cache_mid["ffn_out"]
        h_target = target_blend * h_target_full + (1.0 - target_blend) * cache_mid["h"]
        attn_out_target = self.propagator.project_through_residual(h_target, cache["x"])

        # Step 4: pull attn_out_target back through W_out → context_target.
        W_out_dense = self.attn.W_out.to_dense_weight()
        attn_out_minus_b = attn_out_target.reshape(-1, self.embed_dim)
        if self.attn.W_out._has_bias:
            attn_out_minus_b = attn_out_minus_b - self.attn.W_out._bias
        context_target_full = self.propagator.project_through_linear(
            W_out_dense, attn_out_minus_b,
        ).reshape(*cache["x_ln1"].shape)

        # Step 5-7: pull through softmax(QK^T)V → A_target → scores_target →
        # Q_target, K_target. Use current Q, K, V.
        x_ln1_flat = cache["x_ln1"].reshape(-1, self.embed_dim)
        B, L, _ = cache["x_ln1"].shape
        H = self.attn.num_heads
        d_h = self.attn.head_dim
        Q_curr = self.attn.W_Q(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        K_curr = self.attn.W_K(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        V_curr = self.attn.W_V(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        scale = d_h ** -0.5
        scores_curr = torch.einsum("bhqd,bhkd->bhqk", Q_curr, K_curr) * scale
        attn_w_curr = torch.softmax(scores_curr, dim=-1)
        context_curr = torch.einsum(
            "bhqk,bhkd->bhqd", attn_w_curr, V_curr,
        )                                                       # [B, H, L, d_h]
        # Per-head shaping of the context target.
        context_target_heads = context_target_full.reshape(B, L, H, d_h).transpose(1, 2)

        # 5. A target via current V.
        A_target = self.propagator.solve_attention_pattern_target(
            V_curr, context_target_heads, eps=1.0e-12,
        )                                                       # [B, H, L, L]

        # Mirror-descent step on the probability simplex: blend the target
        # attention pattern with the current pattern *before* inverting
        # softmax. This avoids the inverse-softmax pathology where rows of
        # A_target with near-zero entries produce huge logits → score
        # targets far from current scores → Q/K solutions far from current
        # parameters → joint update overshoots and the global MSE explodes.
        # The convex combination of two row-stochastic matrices stays on the
        # simplex, so no re-projection is needed.
        attn_blend = (
            attn_target_blend if attn_target_blend is not None
            else 0.5 * target_blend
        )
        A_blended = attn_blend * A_target + (1.0 - attn_blend) * attn_w_curr

        # 6. Invert softmax (with √d_h scaling so scores_target = Q K^T target).
        scores_target = self.propagator.softmax_target_to_scores(
            A_blended, scale=1.0 / scale,
        )                                                       # [B, H, L, L]

        # 7. Bilinear pull-back to Q, K targets — Gauss-Seidel ordering:
        #    solve Q* with K_curr fixed, then solve K* with the *new* Q*
        #    fixed. This is the standard alternating least-squares fix for
        #    the joint (non-convex) bilinear problem (SOLVER_MATH §4.3).
        Q_target_heads, _ = self.propagator.project_through_qk_bilinear(
            scores_target, Q_curr, K_curr,
        )
        _, K_target_heads = self.propagator.project_through_qk_bilinear(
            scores_target, Q_target_heads, K_curr,
        )

        # 8. V target via the *blended* attention pattern (so the equation
        #    A_blended · V = C_target is satisfied exactly at the target,
        #    consistent with the same simplex-step used for Q,K).
        V_target_heads = self.propagator.project_through_attention_v(
            A_blended, context_target_heads,
        )

        # No additional parameter-level damping for Q/K — the simplex blend
        # at the A level is the principled damping. V target is also
        # consistent with the blended A. (Damping at the parameter level
        # *after* damping at the A level was found to over-damp empirically.)

        # Reshape back to [B, L, embed].
        Y_Q_target = Q_target_heads.transpose(1, 2).reshape(B, L, self.embed_dim)
        Y_K_target = K_target_heads.transpose(1, 2).reshape(B, L, self.embed_dim)
        Y_V_target = V_target_heads.transpose(1, 2).reshape(B, L, self.embed_dim)

        # Step 9: sweep W_out (input = current context).
        context_full_curr = context_curr.transpose(1, 2).reshape(B, L, self.embed_dim)
        rep_Wout = self.attn.W_out.dmrg_step(
            context_full_curr.reshape(-1, self.embed_dim),
            attn_out_target.reshape(-1, self.embed_dim),
            lam=lam,
        )

        # Step 10: sweep Q, K, V projections under a trust-region accept
        # rule. The Q,K bilinear update is non-convex, so even with mirror-
        # descent damping a step can occasionally increase the global MSE.
        # Snapshot W_Q/W_K/W_V state, run the sweep, then revert if the
        # global MSE worsened.
        snap_Q = {k: v.detach().clone() for k, v in self.attn.W_Q.state_dict().items()}
        snap_K = {k: v.detach().clone() for k, v in self.attn.W_K.state_dict().items()}
        snap_V = {k: v.detach().clone() for k, v in self.attn.W_V.state_dict().items()}
        mse_before_attn = float(
            torch.mean((self.forward_with_cache(X)["y"] - Y_target) ** 2).item()
        )
        attn_results = self.attn.dmrg_step_projections(
            cache["x_ln1"], Y_Q_target, Y_K_target, Y_V_target, lam=lam,
        )
        mse_after_attn = float(
            torch.mean((self.forward_with_cache(X)["y"] - Y_target) ** 2).item()
        )
        attn_accepted = mse_after_attn <= mse_before_attn
        if not attn_accepted:
            self.attn.W_Q.load_state_dict(snap_Q)
            self.attn.W_K.load_state_dict(snap_K)
            self.attn.W_V.load_state_dict(snap_V)

        cache_after = self.forward_with_cache(X)
        global_mse_after = float(torch.mean((cache_after["y"] - Y_target) ** 2).item())

        return {
            "ffn": ffn_reports,
            "attn": {
                "Q": attn_results["Q"],
                "K": attn_results["K"],
                "V": attn_results["V"],
                "W_out": rep_Wout.final_mse,
                "accepted": attn_accepted,
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
