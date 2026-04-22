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


class _AffineLN(nn.Module):
    """LayerNorm with affine ``(γ, β)`` stored as **buffers** (not nn.Parameter).

    Buffers are used (matching :class:`TTLinear`'s core storage convention) to
    guarantee no autograd engagement under AGENTS Constraint 1. Initial values
    are ``γ = 1, β = 0`` so the module is bit-exact with
    ``nn.LayerNorm(elementwise_affine=False)`` until :meth:`update_affine_lsq`
    is called.

    The update rule is a per-feature 2-variable least-squares fit: given
    ``x_pre_ln`` and a target ``y_target`` with shape ``[..., features]``,
    we standardize per-row to ``z = (x - μ_row) / σ_row`` and solve, for each
    feature ``f`` independently,

        min_{γ_f, β_f}  Σ_n (γ_f · z[n, f] + β_f - y_target[n, f])²

    via a closed-form 2×2 normal-equation solve. This is the canonical
    Eckart–Young / OLS update for the affine head of LayerNorm.
    """

    def __init__(self, features: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.features = int(features)
        self.eps = float(eps)
        self.register_buffer("gamma", torch.ones(features, dtype=dtype))
        self.register_buffer("beta", torch.zeros(features, dtype=dtype))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        z = (x - mu) / torch.sqrt(var + self.eps)
        return z * self.gamma + self.beta

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps)

    @torch.no_grad()
    def update_affine_lsq(
        self,
        x_pre_ln: torch.Tensor,
        y_target_post_ln: torch.Tensor,
        *,
        ridge: float = 1.0e-8,
    ) -> tuple[float, float]:
        """Per-feature OLS fit ``y ≈ γ·z + β``. Returns (mse_before, mse_after)."""
        if x_pre_ln.shape != y_target_post_ln.shape:
            raise ValueError(
                f"shape mismatch: x_pre_ln {tuple(x_pre_ln.shape)} vs "
                f"y_target {tuple(y_target_post_ln.shape)}"
            )
        z = self._standardize(x_pre_ln).reshape(-1, self.features)        # [N, F]
        y = y_target_post_ln.reshape(-1, self.features)                    # [N, F]
        N = z.shape[0]
        y_pred_before = z * self.gamma + self.beta
        mse_before = float(torch.mean((y_pred_before - y) ** 2).item())

        # Per-feature 2×2 normal equations:
        #   [Σ z²  Σ z ] [γ]   [Σ z·y]
        #   [Σ z   N   ] [β] = [Σ y  ]
        sum_zz = (z * z).sum(dim=0)                    # [F]
        sum_z = z.sum(dim=0)                           # [F]
        sum_zy = (z * y).sum(dim=0)                    # [F]
        sum_y = y.sum(dim=0)                           # [F]
        # Build per-feature 2×2 systems and solve via stacked linalg.solve.
        A = torch.stack([
            torch.stack([sum_zz + ridge, sum_z], dim=-1),
            torch.stack([sum_z, torch.full_like(sum_z, float(N)) + ridge], dim=-1),
        ], dim=-2)                                                          # [F, 2, 2]
        rhs = torch.stack([sum_zy, sum_y], dim=-1).unsqueeze(-1)            # [F, 2, 1]
        sol = torch.linalg.solve(A, rhs).squeeze(-1)                        # [F, 2]
        gamma_new = sol[:, 0]
        beta_new = sol[:, 1]
        if not torch.isfinite(gamma_new).all() or not torch.isfinite(beta_new).all():
            return mse_before, mse_before
        self.gamma.copy_(gamma_new)
        self.beta.copy_(beta_new)
        y_pred_after = z * self.gamma + self.beta
        mse_after = float(torch.mean((y_pred_after - y) ** 2).item())
        return mse_before, mse_after


class TTBlock(nn.Module):
    """Pre-LN Transformer encoder block (TT-MHA + TT-FFN with optional LN affine).

    The ``enable_ln_affine`` flag controls whether the LayerNorm γ, β
    parameters participate in the DMRG update:

    * ``False`` (default): LN affine is frozen at γ=1, β=0 — bit-exact with
      ``nn.LayerNorm(elementwise_affine=False)``. This preserves all existing
      benchmarks and regression tests.
    * ``True``: γ, β are updated by per-feature OLS at the end of each
      :meth:`dmrg_step` under a trust-region accept/revert rule
      (snapshot → fit → if global MSE worsens, revert).
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
        enable_ln_affine: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ln_eps = ln_eps
        self.enable_ln_affine = bool(enable_ln_affine)

        # LN modules: always use the buffer-based _AffineLN so behaviour is
        # bit-exact with frozen LN at init (γ=1, β=0). The OLS update is
        # only invoked if `enable_ln_affine=True`.
        self.ln1 = _AffineLN(embed_dim, eps=ln_eps, dtype=dtype)
        self.ln2 = _AffineLN(embed_dim, eps=ln_eps, dtype=dtype)
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

        # Step 11 (opt-in): LN affine OLS refit under trust-region. We refit
        # γ2,β2 first (LN2 → FFN), then γ1,β1 (LN1 → attn). Each fit is
        # accepted only if the global block MSE strictly improves.
        ln_accepted = {"ln1": False, "ln2": False}
        if self.enable_ln_affine:
            cache_pre_ln = self.forward_with_cache(X)
            mse_pre_ln = float(
                torch.mean((cache_pre_ln["y"] - Y_target) ** 2).item()
            )

            # LN2 target: pull `ffn_target` (target for FFN output, given by
            # the residual #2 pull-back at the *current* x) back through the
            # FFN to get a target for `h_ln2`.
            ffn_target_now = self.propagator.project_through_residual(
                Y_target, cache_pre_ln["h"],
            )
            h_ln2_target = self._pullback_ffn_to_input(
                cache_pre_ln["h_ln2"], ffn_target_now,
            )
            snap_g2 = self.ln2.gamma.detach().clone()
            snap_b2 = self.ln2.beta.detach().clone()
            self.ln2.update_affine_lsq(cache_pre_ln["h"], h_ln2_target)
            mse_after_ln2 = float(
                torch.mean(
                    (self.forward_with_cache(X)["y"] - Y_target) ** 2
                ).item()
            )
            if mse_after_ln2 <= mse_pre_ln:
                ln_accepted["ln2"] = True
                mse_pre_ln = mse_after_ln2
            else:
                self.ln2.gamma.copy_(snap_g2)
                self.ln2.beta.copy_(snap_b2)

            # LN1 target: pull combined Q/K/V targets back through stacked
            # W_Q/W_K/W_V (Tikhonov ridge) to get a target for `x_ln1`.
            x_ln1_target = self._pullback_qkv_to_input(
                Y_Q_target, Y_K_target, Y_V_target,
            )
            snap_g1 = self.ln1.gamma.detach().clone()
            snap_b1 = self.ln1.beta.detach().clone()
            self.ln1.update_affine_lsq(cache_pre_ln["x"], x_ln1_target)
            mse_after_ln1 = float(
                torch.mean(
                    (self.forward_with_cache(X)["y"] - Y_target) ** 2
                ).item()
            )
            if mse_after_ln1 <= mse_pre_ln:
                ln_accepted["ln1"] = True
            else:
                self.ln1.gamma.copy_(snap_g1)
                self.ln1.beta.copy_(snap_b1)

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
            "ln_accepted": ln_accepted,
            "global_mse_before": global_mse_before,
            "global_mse_after": global_mse_after,
        }

    # -- LN-affine target derivation helpers ------------------------------------

    @torch.no_grad()
    def _pullback_ffn_to_input(
        self,
        h_ln2_current: torch.Tensor,
        ffn_target: torch.Tensor,
    ) -> torch.Tensor:
        """Three-step pull-back of an FFN output target to an input target.

        ``ffn_target`` (post-fc2) → ``h1_target`` (post-GELU, via fc2 pinv) →
        ``z1_target`` (pre-GELU, via active mask) → ``h_ln2_target`` (via
        fc1 pinv). Mirrors :meth:`TTFeedForward.dmrg_step`'s internal
        propagation.
        """
        flat = h_ln2_current.reshape(-1, self.embed_dim)
        ffn_target_flat = ffn_target.reshape(-1, self.embed_dim)
        ffn = self.ffn
        # Step 1: pull through fc2 (post-GELU target).
        W2 = ffn.fc2.to_dense_weight()
        y_minus_b = ffn_target_flat - ffn.fc2._bias if ffn.fc2._has_bias else ffn_target_flat
        h1_target = self.propagator.project_through_linear(W2, y_minus_b)
        # Step 2: GELU active mask (positive-derivative proxy).
        z1 = ffn.fc1(flat)
        active = z1 > 0
        z1_target = torch.where(active, h1_target, z1)
        # Step 3: pull through fc1.
        W1 = ffn.fc1.to_dense_weight()
        z1_minus_b = z1_target - ffn.fc1._bias if ffn.fc1._has_bias else z1_target
        h_ln2_target_flat = self.propagator.project_through_linear(W1, z1_minus_b)
        return h_ln2_target_flat.reshape(*ffn_target.shape)

    @torch.no_grad()
    def _pullback_qkv_to_input(
        self,
        Y_Q_target: torch.Tensor,
        Y_K_target: torch.Tensor,
        Y_V_target: torch.Tensor,
    ) -> torch.Tensor:
        """Stacked-LSQ pull-back of (Q, K, V) targets to a single x_ln1 target.

        Solves ``[W_Q; W_K; W_V] · x_ln1 ≈ [Y_Q; Y_K; Y_V]`` (Tikhonov-
        damped) for the per-token LN1-output target. Bias terms are absorbed
        into the targets first.
        """
        WQ = self.attn.W_Q.to_dense_weight()
        WK = self.attn.W_K.to_dense_weight()
        WV = self.attn.W_V.to_dense_weight()
        YQ = Y_Q_target.reshape(-1, self.embed_dim)
        YK = Y_K_target.reshape(-1, self.embed_dim)
        YV = Y_V_target.reshape(-1, self.embed_dim)
        if self.attn.W_Q._has_bias:
            YQ = YQ - self.attn.W_Q._bias
        if self.attn.W_K._has_bias:
            YK = YK - self.attn.W_K._bias
        if self.attn.W_V._has_bias:
            YV = YV - self.attn.W_V._bias
        # Stack along the *output* dim of each linear: W_stack: [embed, 3*embed].
        # `project_through_linear(W, target)` expects W of shape [in, out] and
        # target of shape [..., out], and returns [..., in]. So we cat along
        # dim=1 (out axis) and the target along the last axis.
        W_stack = torch.cat([WQ, WK, WV], dim=1)
        Y_stack = torch.cat([YQ, YK, YV], dim=-1)
        x_ln1_target_flat = self.propagator.project_through_linear(W_stack, Y_stack)
        return x_ln1_target_flat.reshape(*Y_Q_target.shape)

    @property
    def num_parameters(self) -> int:
        return (
            self.attn.W_Q.num_parameters + self.attn.W_K.num_parameters
            + self.attn.W_V.num_parameters + self.attn.W_out.num_parameters
            + self.ffn.num_parameters
        )
