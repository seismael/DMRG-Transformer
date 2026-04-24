"""``TTLinearAttentionBlock`` — Pre-LN block with TT linear attention.

Composition (Pre-LN):

     h = x + lin_attn(LN1(x))
     y = h + ffn(LN2(h))

Why a separate block class
--------------------------
The DMRG step for linear attention is structurally simpler than for softmax:
- No bilinear Q/K pull-back (no softmax to invert).
- The V update is a single closed-form LSQ that minimizes the *global* loss
  modulo the W_out pinv pull-back. No alternating-minimization drift.
- First-cut strategy: V-only attention update. Q and K stay at their random
  init (Random-Features-Attention regime). This is sufficient to test the
  multilinear hypothesis. Q/K updates are a follow-up if needed.

Update sequence (per call to :meth:`dmrg_step`):

1. Forward with cache.
2. ``ffn_target = Y_target − h`` (residual #2 pull-back). Sweep FFN.
3. Re-cache; ``attn_out_target = Y_target − ffn_out_new − x``.
4. Sweep ``W_out`` with current context as input.
5. Pull ``attn_out_target`` back through the *updated* ``W_out`` to get
   ``context_target`` (per-token).
6. Solve for the per-(B,H,L_k) value matrix ``V`` via per-batch-head LSQ:
       w[b,h,q,k] = φ(Q)[b,h,q,:] · φ(K)[b,h,k,:]
       target_per_query[b,h,q,j] = denom[b,h,q] · context_target[b,h,q,j]
       V_target[b,h] = (wᵀ w + λI)⁻¹ wᵀ target_per_query[b,h]
   Reshape to ``[B, L_k, embed_dim]`` and run ``W_V.dmrg_step`` against it.
7. Trust-region accept/revert against block global MSE (defensive — by
   construction this should always accept on monotonic problems).
"""
from __future__ import annotations

import torch
from torch import nn

from dmrg_transformer.nn.tt_block import _AffineLN
from dmrg_transformer.nn.tt_ffn import TTFeedForward
from dmrg_transformer.nn.tt_linear_attention import TTLinearAttention, elu_plus_one
from dmrg_transformer.propagation.target_propagator import TargetPropagator


class TTLinearAttentionBlock(nn.Module):
    """Pre-LN Transformer block with TT linear attention + TT-FFN."""

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
        attn_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        from dmrg_transformer.core.device import require_cuda

        device = require_cuda()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ln_eps = ln_eps
        self.enable_ln_affine = bool(enable_ln_affine)
        self.attn_eps = float(attn_eps)

        self.ln1 = _AffineLN(embed_dim, eps=ln_eps, dtype=dtype, device=device)
        self.ln2 = _AffineLN(embed_dim, eps=ln_eps, dtype=dtype, device=device)
        self.attn = TTLinearAttention(
            embed_dim, num_heads,
            input_dims=embed_dims, output_dims=embed_dims,
            rank=rank, dtype=dtype, eps=attn_eps,
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

    # -- linear-attention internals -------------------------------------------

    def _compute_attn_internals(
        self, x_ln1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return per-head Q, K, V, phiQ, phiK, w (= phiQ·phiKᵀ), denom, context."""
        B, L, _ = x_ln1.shape
        H, d_h = self.attn.num_heads, self.attn.head_dim
        x_flat = x_ln1.reshape(-1, self.embed_dim)
        Q = self.attn.W_Q(x_flat).reshape(B, L, H, d_h).transpose(1, 2)
        K = self.attn.W_K(x_flat).reshape(B, L, H, d_h).transpose(1, 2)
        V = self.attn.W_V(x_flat).reshape(B, L, H, d_h).transpose(1, 2)
        phiQ = elu_plus_one(Q)
        phiK = elu_plus_one(K)
        w = torch.einsum("bhqi,bhki->bhqk", phiQ, phiK)               # [B,H,L_q,L_k]
        denom = w.sum(dim=-1)                                          # [B,H,L_q]
        # context = (w @ V) / denom
        num = torch.einsum("bhqk,bhkj->bhqj", w, V)
        context = num / (denom.unsqueeze(-1) + self.attn_eps)          # [B,H,L_q,d_h]
        return {
            "Q": Q, "K": K, "V": V, "phiQ": phiQ, "phiK": phiK,
            "w": w, "denom": denom, "context": context,
        }

    def _solve_v_target(
        self,
        w: torch.Tensor,                     # [B,H,L_q,L_k]
        denom: torch.Tensor,                 # [B,H,L_q]
        context_target: torch.Tensor,        # [B,H,L_q,d_h]
        *,
        lam_rel: float = 1.0e-3,
    ) -> torch.Tensor:
        """Per-(B,H) ridge LSQ for V given fixed w (i.e. fixed phiQ, phiK).

        ``w`` is typically rank-deficient at random init (cond(wᵀw) >> 1e10
        observed empirically), so absolute Tikhonov regularization is unstable
        — a tiny λ amplifies V_target by 4+ orders of magnitude. We therefore
        scale λ relative to ``trace(wᵀw)/L_k`` per (B,H), guaranteeing the
        regularizer's contribution is on the same order as the data term's
        diagonal.
        """
        # The target context_q satisfies  Σ_k w_{q,k} V_k = denom_q · context_target_q.
        target = denom.unsqueeze(-1) * context_target                  # [B,H,L_q,d_h]
        # Per (B,H) regularized normal equations:  V = (wᵀw + λI)⁻¹ wᵀ target
        wTw = torch.einsum("bhqk,bhqK->bhkK", w, w)                    # [B,H,L_k,L_k]
        L_k = wTw.shape[-1]
        # Per-(B,H) λ scaled to the average diagonal entry of wᵀw.
        diag_avg = torch.diagonal(wTw, dim1=-2, dim2=-1).mean(dim=-1)  # [B,H]
        lam_eff = lam_rel * diag_avg                                    # [B,H]
        eye = torch.eye(L_k, dtype=wTw.dtype, device=wTw.device)
        reg = lam_eff[..., None, None] * eye                            # [B,H,L_k,L_k]
        wTt = torch.einsum("bhqk,bhqj->bhkj", w, target)                # [B,H,L_k,d_h]
        V_target = torch.linalg.solve(wTw + reg, wTt)                   # [B,H,L_k,d_h]
        return V_target

    # -- DMRG update -----------------------------------------------------------

    @torch.no_grad()
    def dmrg_step(
        self,
        X: torch.Tensor,
        Y_target: torch.Tensor,
        *,
        lam: float = 1.0e-5,
        adaptive_threshold: float | None = None,
        v_solver_lam_rel: float = 1.0e-3,
        v_line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02),
        inner_iters: int = 1,
    ) -> dict[str, object]:
        """Exact-solver update for one Pre-LN linear-attention block.

        Sequence (attention first, FFN last):
          1. Cache forward.  attn_out_target := Y_target − ffn_out_cur − x
             (ffn is held fixed as the current contribution; it absorbs the
             residual on its own pass after attention has moved.)
          2. Perform ``inner_iters`` micro-sweeps between W_out and V.
             This allows the attention projection and the value-mixing to
             settle into a joint optimum per epoch (Pathway 1.6).
          3. Re-cache; ffn_target := Y_target − h_new. Sweep FFN.
        """
        cache = self.forward_with_cache(X)
        global_mse_before = float(torch.mean((cache["y"] - Y_target) ** 2).item())
        B, L, _ = X.shape
        H, d_h = self.attn.num_heads, self.attn.head_dim

        inner_reports = []
        for i_it in range(inner_iters):
            # Step 1: derive attn_out target with current FFN output as fixed background.
            curr_cache = self.forward_with_cache(X)
            attn_out_target = Y_target - curr_cache["ffn_out"] - curr_cache["x"]    # [B,L,D]
            mse_loop_start = float(torch.mean((curr_cache["y"] - Y_target) ** 2).item())

            # Step 2: W_out sweep on current context, with trust-region revert.
            internals = self._compute_attn_internals(curr_cache["x_ln1"])
            context_full = internals["context"].transpose(1, 2).reshape(B, L, self.embed_dim)
            snap_Wout: dict[str, torch.Tensor] = {
                n: b.detach().clone() for n, b in self.attn.W_out.named_buffers()
            }
            rep_Wout = self.attn.W_out.dmrg_step(
                context_full.reshape(-1, self.embed_dim),
                attn_out_target.reshape(-1, self.embed_dim),
                lam=lam, adaptive_threshold=adaptive_threshold,
            )
            mse_after_wout_check = float(torch.mean((self.forward_with_cache(X)["y"] - Y_target) ** 2).item())
            wout_accepted = mse_after_wout_check <= mse_loop_start
            if not wout_accepted:
                for n, b in snap_Wout.items():
                    self.attn.W_out.get_buffer(n).copy_(b)
                mse_after_wout = mse_loop_start
            else:
                mse_after_wout = mse_after_wout_check

            # Step 3: pull attn_out_target back through updated W_out.
            W_out_dense = self.attn.W_out.to_dense_weight()
            attn_out_minus_b = attn_out_target.reshape(-1, self.embed_dim)
            if self.attn.W_out._has_bias:
                attn_out_minus_b = attn_out_minus_b - self.attn.W_out._bias
            context_target_full = self.propagator.project_through_linear(
                W_out_dense, attn_out_minus_b,
            ).reshape(B, L, self.embed_dim)
            context_target_heads = context_target_full.reshape(B, L, H, d_h).transpose(1, 2)

            # Step 4: closed-form V LSQ given current phiQ, phiK.
            V_target_heads = self._solve_v_target(
                internals["w"], internals["denom"], context_target_heads, lam_rel=v_solver_lam_rel,
            )
            V_cur_heads = internals["V"]

            # Step 5: line search over the V damping factor α.
            snap_V = {k: v.detach().clone() for k, v in self.attn.W_V.state_dict().items()}
            x_ln1_flat = curr_cache["x_ln1"].reshape(-1, self.embed_dim)

            best_alpha = 0.0
            best_mse = mse_after_wout
            best_state: dict[str, torch.Tensor] | None = None
            v_final_mse = float("nan")
            trial_mses = []
            for alpha in v_line_search_alphas:
                self.attn.W_V.load_state_dict(snap_V)
                V_blend_heads = alpha * V_target_heads + (1.0 - alpha) * V_cur_heads
                Y_V_blend = V_blend_heads.transpose(1, 2).reshape(B, L, self.embed_dim)
                v_rep = self.attn.W_V.dmrg_step(
                    x_ln1_flat, Y_V_blend.reshape(-1, self.embed_dim),
                    lam=lam, adaptive_threshold=adaptive_threshold,
                )
                mse_trial = float(torch.mean((self.forward_with_cache(X)["y"] - Y_target) ** 2).item())
                trial_mses.append((alpha, mse_trial))
                if mse_trial < best_mse:
                    best_mse = mse_trial
                    best_alpha = alpha
                    best_state = {k: v.detach().clone() for k, v in self.attn.W_V.state_dict().items()}
                    v_final_mse = float(v_rep.final_mse)
            
            if best_state is not None:
                self.attn.W_V.load_state_dict(best_state)
            else:
                self.attn.W_V.load_state_dict(snap_V)
            
            v_accepted = best_state is not None
            mse_after_v = best_mse
            inner_reports.append({
                "iter": i_it,
                "wout_accepted": wout_accepted,
                "v_accepted": v_accepted,
                "v_alpha": best_alpha,
                "mse_start": mse_loop_start,
                "mse_after_v": mse_after_v,
            })

        # Step 6: re-sweep FFN to absorb whatever the attention update left over.
        # Trust-region: if the FFN re-sweep makes the global MSE worse than it
        # was after the attention pass (this happens when FFN was already near
        # optimal for the teacher and the new attention residual is just noise
        # that FFN over-fits), revert FFN to its prior state.
        cache_after_attn = self.forward_with_cache(X)
        mse_after_attn = float(torch.mean((cache_after_attn["y"] - Y_target) ** 2).item())
        ffn_target = Y_target - cache_after_attn["h"]
        snap_ffn_buffers: dict[str, torch.Tensor] = {
            n: b.detach().clone() for n, b in self.ffn.named_buffers()
        }
        ffn_reports = self.ffn.dmrg_step(
            cache_after_attn["h_ln2"].reshape(-1, self.embed_dim),
            ffn_target.reshape(-1, self.embed_dim),
            lam=lam, target_blend=1.0, adaptive_threshold=adaptive_threshold,
        )
        cache_post_ffn = self.forward_with_cache(X)
        mse_post_ffn = float(torch.mean((cache_post_ffn["y"] - Y_target) ** 2).item())
        ffn_accepted = mse_post_ffn <= mse_after_attn
        if not ffn_accepted:
            for n, b in snap_ffn_buffers.items():
                self.ffn.get_buffer(n).copy_(b)

        cache_final = self.forward_with_cache(X)
        global_mse_after = float(torch.mean((cache_final["y"] - Y_target) ** 2).item())

        attn_diag = {
            "v_accepted": v_accepted,
            "v_alpha": best_alpha,
            "mse_before_v": mse_after_wout,
            "mse_after_v": mse_after_v,
            "v_final_mse_local": v_final_mse,
            "v_trial_mses": trial_mses,
            "ffn_accepted": ffn_accepted,
            "wout_accepted": wout_accepted,
            "qk_accepted": True,        # frozen at init in v-only first cut
        }
        return {
            "ffn": ffn_reports,
            "attn": {
                "V": v_final_mse if v_accepted else None,
                "W_out": rep_Wout.final_mse,
                "accepted": v_accepted,
                "diagnostics": attn_diag,
            },
            "global_mse_before": global_mse_before,
            "global_mse_after": global_mse_after,
        }

    @torch.no_grad()
    def pullback_target(
        self,
        X: torch.Tensor,
        Y_target: torch.Tensor,
        *,
        target_blend: float = 1.0,
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
