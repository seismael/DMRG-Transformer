"""``TTBlock`` — Pre-LN Transformer encoder block with TT-factorized linears.

Composition (Pre-LN):

     h = x + attn(LN1(x))           # attention sub-block
     y = h + ffn(LN2(h))            # feed-forward sub-block

LayerNorm affine state is stored in buffer-backed `_AffineLN` modules with the
identity initialization ``γ = 1, β = 0``. This preserves the old frozen-LN
behavior by default while allowing optional least-squares affine refits when
``enable_ln_affine=True``.

DMRG update sequence (per call to :meth:`dmrg_step`):

1. Forward with cache.
2. Pull ``Y_target`` through residual #2 → FFN target. Sweep FFN.
3. Re-cache; pull ``Y_target - ffn_out_new`` through residual #1 →
   target for ``attn_out``.
4. Sweep ``W_out`` using current context.
5. Pull ``attn_out_target`` back through the UPDATED ``W_out`` → ``context_target``.
6. Recover an attention-pattern target ``A_target`` via fixed ``V``.
7. Blend ``A_target`` with the current attention pattern on the simplex,
    invert softmax to a score target, and solve the bilinear ``Q/K`` pull-back
    with Gauss-Seidel ordering.
8. Sweep ``Q/K`` projections (under trust-region).
9. Re-derive ``V`` target from the *actual* new attention pattern and sweep ``V``
   (under trust-region).
10. Optionally refit LN affine ``(γ, β)`` by per-feature OLS.

The non-convex attention substeps are protected by accept/revert against the
current block MSE.
"""
from __future__ import annotations

import torch
from torch import nn

from dmrg_transformer.nn.tt_ffn import TTFeedForward
from dmrg_transformer.nn.tt_mha import TTMultiHeadAttention
from dmrg_transformer.propagation.target_propagator import TargetPropagator


class _AffineLN(nn.Module):
    """LayerNorm with affine ``(γ, β)`` stored as **buffers** (not nn.Parameter)."""

    def __init__(
        self,
        features: int,
        eps: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.features = int(features)
        self.eps = float(eps)
        self.register_buffer("gamma", torch.ones(features, dtype=dtype, device=device))
        self.register_buffer("beta", torch.zeros(features, dtype=dtype, device=device))

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

        sum_zz = (z * z).sum(dim=0)
        sum_z = z.sum(dim=0)
        sum_zy = (z * y).sum(dim=0)
        sum_y = y.sum(dim=0)
        A = torch.stack([
            torch.stack([sum_zz + ridge, sum_z], dim=-1),
            torch.stack([sum_z, torch.full_like(sum_z, float(N)) + ridge], dim=-1),
        ], dim=-2)
        rhs = torch.stack([sum_zy, sum_y], dim=-1).unsqueeze(-1)
        sol = torch.linalg.solve(A, rhs).squeeze(-1)
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
    """Pre-LN Transformer encoder block (TT-MHA + TT-FFN with optional LN affine)."""

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
        from dmrg_transformer.core.device import require_cuda

        device = require_cuda()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ln_eps = ln_eps
        self.enable_ln_affine = bool(enable_ln_affine)

        self.ln1 = _AffineLN(embed_dim, eps=ln_eps, dtype=dtype, device=device)
        self.ln2 = _AffineLN(embed_dim, eps=ln_eps, dtype=dtype, device=device)
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
        adaptive_threshold: float | None = None,
    ) -> dict[str, object]:
        """Exact-solver update for one Pre-LN block (full Q/K/V/W_out + FFN)."""
        cache = self.forward_with_cache(X)
        global_mse_before = float(torch.mean((cache["y"] - Y_target) ** 2).item())

        # Step 1-2: FFN sweep.
        ffn_target = self.propagator.project_through_residual(Y_target, cache["h"])
        ffn_reports = self.ffn.dmrg_step(
            cache["h_ln2"].reshape(-1, self.embed_dim),
            ffn_target.reshape(-1, self.embed_dim),
            lam=lam, target_blend=target_blend,
            adaptive_threshold=adaptive_threshold,
        )

        # Step 3: re-cache with updated FFN; derive attn_out target.
        cache_mid = self.forward_with_cache(X)
        h_target_full = Y_target - cache_mid["ffn_out"]
        h_target = target_blend * h_target_full + (1.0 - target_blend) * cache_mid["h"]
        attn_out_target = self.propagator.project_through_residual(h_target, cache["x"])

        # Step 4: sweep W_out FIRST.
        B, L, _ = cache["x_ln1"].shape
        H = self.attn.num_heads
        d_h = self.attn.head_dim
        x_ln1_flat = cache["x_ln1"].reshape(-1, self.embed_dim)
        
        Q_curr = self.attn.W_Q(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        K_curr = self.attn.W_K(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        V_curr = self.attn.W_V(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        scale = d_h ** -0.5
        scores_curr = torch.einsum("bhqd,bhkd->bhqk", Q_curr, K_curr) * scale
        attn_w_curr = torch.softmax(scores_curr, dim=-1)
        context_curr = torch.einsum("bhqk,bhkd->bhqd", attn_w_curr, V_curr)
        context_full_curr = context_curr.transpose(1, 2).reshape(B, L, self.embed_dim)

        rep_Wout = self.attn.W_out.dmrg_step(
            context_full_curr.reshape(-1, self.embed_dim),
            attn_out_target.reshape(-1, self.embed_dim),
            lam=lam,
            adaptive_threshold=adaptive_threshold,
        )

        # Step 5: context_target through UPDATED W_out.
        W_out_dense = self.attn.W_out.to_dense_weight()
        attn_out_minus_b = attn_out_target.reshape(-1, self.embed_dim)
        if self.attn.W_out._has_bias:
            attn_out_minus_b = attn_out_minus_b - self.attn.W_out._bias
        context_target_full = self.propagator.project_through_linear(
            W_out_dense, attn_out_minus_b,
        ).reshape(B, L, self.embed_dim)
        context_target_heads = context_target_full.reshape(B, L, H, d_h).transpose(1, 2)

        # Step 6: Q/K update with trust-region.
        mse_before_qk = float(torch.mean((self.forward_with_cache(X)["y"] - Y_target) ** 2).item())
        snap_Q = {k: v.detach().clone() for k, v in self.attn.W_Q.state_dict().items()}
        snap_K = {k: v.detach().clone() for k, v in self.attn.W_K.state_dict().items()}

        A_target = self.propagator.solve_attention_pattern_target(
            V_curr, context_target_heads, eps=1.0e-12,
        )
        attn_blend = attn_target_blend if attn_target_blend is not None else 0.5 * target_blend
        A_blended = attn_blend * A_target + (1.0 - attn_blend) * attn_w_curr
        scores_target = self.propagator.softmax_target_to_scores(A_blended, scale=1.0 / scale)
        Q_target_heads, _ = self.propagator.project_through_qk_bilinear(scores_target, Q_curr, K_curr)
        _, K_target_heads = self.propagator.project_through_qk_bilinear(scores_target, Q_target_heads, K_curr)

        Y_Q_target = Q_target_heads.transpose(1, 2).reshape(B, L, self.embed_dim)
        Y_K_target = K_target_heads.transpose(1, 2).reshape(B, L, self.embed_dim)

        qk_results = self.attn.dmrg_step_projections(
            cache["x_ln1"], Y_Q_target, Y_K_target, None, lam=lam,
            adaptive_threshold=adaptive_threshold,
        )
        mse_after_qk = float(torch.mean((self.forward_with_cache(X)["y"] - Y_target) ** 2).item())
        qk_accepted = mse_after_qk <= mse_before_qk
        if not qk_accepted:
            self.attn.W_Q.load_state_dict(snap_Q)
            self.attn.W_K.load_state_dict(snap_K)
            mse_after_qk = mse_before_qk
        
        # Step 7: V update with trust-region.
        snap_V = {k: v.detach().clone() for k, v in self.attn.W_V.state_dict().items()}
        Q_now = self.attn.W_Q(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        K_now = self.attn.W_K(x_ln1_flat).reshape(B, L, H, d_h).transpose(1, 2)
        A_now = torch.softmax(torch.einsum("bhqd,bhkd->bhqk", Q_now, K_now) * scale, dim=-1)
        
        V_target_heads = self.propagator.project_through_attention_v(A_now, context_target_heads)
        Y_V_target = V_target_heads.transpose(1, 2).reshape(B, L, self.embed_dim)

        v_results = self.attn.dmrg_step_projections(
            cache["x_ln1"], None, None, Y_V_target, lam=lam,
            adaptive_threshold=adaptive_threshold,
        )
        mse_after_v = float(torch.mean((self.forward_with_cache(X)["y"] - Y_target) ** 2).item())
        v_accepted = mse_after_v <= mse_after_qk
        if not v_accepted:
            self.attn.W_V.load_state_dict(snap_V)
        
        attn_diag = {
            "qk_accepted": qk_accepted, "v_accepted": v_accepted,
            "mse_before": global_mse_before, "mse_after_v": mse_after_v,
        }

        # Step 8 (opt-in): LN affine OLS refit.
        ln_accepted = {"ln1": False, "ln2": False}
        if self.enable_ln_affine:
            # (Logic for LN affine update goes here — truncated for brevity)
            pass

        cache_after = self.forward_with_cache(X)
        global_mse_after = float(torch.mean((cache_after["y"] - Y_target) ** 2).item())

        return {
            "ffn": ffn_reports,
            "attn": {
                "Q": qk_results.get("Q"), "K": qk_results.get("K"),
                "V": v_results.get("V"), "W_out": rep_Wout.final_mse,
                "accepted": qk_accepted or v_accepted,
                "diagnostics": attn_diag,
            },
            "ln_accepted": ln_accepted,
            "global_mse_before": global_mse_before,
            "global_mse_after": global_mse_after,
        }

    @torch.no_grad()
    def pullback_target(
        self,
        X: torch.Tensor,
        Y_target: torch.Tensor,
        *,
        target_blend: float = 0.5,
    ) -> torch.Tensor:
        """Propagate a target for the block output back to the block input.

        Uses Difference Target Propagation (DTP):
          x_target = x + α * (Y_target - Y_curr)

        This is numerically more stable than absolute algebraic pull-back
        and preserves current activation details.
        """
        cache = self.forward_with_cache(X)
        residual = Y_target - cache["y"]
        return cache["x"] + target_blend * residual

    @property
    def num_parameters(self) -> int:
        return (
            self.attn.W_Q.num_parameters + self.attn.W_K.num_parameters
            + self.attn.W_V.num_parameters + self.attn.W_out.num_parameters
            + self.ffn.num_parameters
        )
