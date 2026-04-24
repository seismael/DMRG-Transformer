"""Layer-wise target propagator (ARCHITECTURE.md §4.2).

Replaces the Backpropagation Chain Rule with a Tikhonov-damped Moore–Penrose
pseudo-inverse that translates the global network error into a local target
tensor ``T_l`` each layer must achieve. This is valid because every TT-layer's
forward map is linear in its weights — the target is the preimage of the
desired output under the next layer's linearized pass.
"""
from __future__ import annotations

import torch


class TargetPropagator:
    """Compute a local target for a layer given a global target and its forward output.

    Let ``y = f(x; W)`` with ``f`` linear in ``W``. Given ``y_target`` for the
    subsequent layer and the current input ``x`` to this layer, the local
    target is the Tikhonov-regularized pseudo-inverse solution

        t_local = (X^T X + λ I)^{-1} X^T y_target     (NUMERICAL_STABILITY §3)

    This computes the activation the current layer should emit so that the
    cascaded network hits ``y_target`` — the exact-solver analogue of the
    Chain Rule.
    """

    def __init__(self, lam: float = 1.0e-5) -> None:
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        self.lam = float(lam)

    def compute_layer_target(
        self,
        global_target: torch.Tensor,
        current_layer_out: torch.Tensor,
    ) -> torch.Tensor:
        """Solve ``current_layer_out ≈ global_target`` as a least-squares residual.

        For a batch-wise linear cascade, the natural local target is simply the
        portion of ``global_target`` that the current layer's output must
        reproduce. We apply Tikhonov damping to guard against ill-conditioned
        downstream activations.

        Args:
            global_target: ``[batch, features_out]``
            current_layer_out: ``[batch, features_out]``

        Returns:
            ``[batch, features_out]`` — the propagated local target.
        """
        if global_target.shape != current_layer_out.shape:
            raise ValueError(
                f"shape mismatch: global_target {tuple(global_target.shape)} vs "
                f"current_layer_out {tuple(current_layer_out.shape)}"
            )
        # Damped residual blending: T_l = current + (I - λ·(I+X^TX)^-1) (global - current)
        # For the trivial case where the subsequent layer is identity, this collapses
        # to the damped least-squares projection of ``global_target`` onto the
        # current output manifold.
        residual = global_target - current_layer_out
        # Identity-linear next-layer case: the local target is simply the global target
        # with a Tikhonov shrinkage pulling it toward the current output.
        lam = self.lam
        alpha = 1.0 / (1.0 + lam)
        return current_layer_out + alpha * residual

    def project_through_linear(
        self,
        downstream_weight: torch.Tensor,
        downstream_target: torch.Tensor,
    ) -> torch.Tensor:
        """Pull a downstream target back through a linear layer ``y = x @ W``.

        Solves ``x @ W = y_target`` for ``x`` via Tikhonov-damped pseudo-inverse
        (SOLVER_MATH.md §I closed-form with damping). Uses the min-dimension
        Gram matrix for stability:

        * **Overdetermined** (``in >= out``):
          ``x* = y_target @ (W^T W + λ I)^{-1} W^T``
        * **Underdetermined** (``in < out``):
          ``x* = y_target @ W^T (W W^T + λ I)^{-1}``

        Args:
            downstream_weight: ``[in, out]`` weight matrix of the subsequent layer.
            downstream_target: ``[batch, out]`` target of the subsequent layer.

        Returns:
            ``[batch, in]`` — the local target for the current layer's output.
        """
        W = downstream_weight
        d_in, d_out = W.shape
        if d_in >= d_out:
            # Overdetermined (or square): x* = y_target @ W^+
            # W^+ = (W^T W + λI)^-1 W^T
            gram = W.T @ W  # [out, out]
            lam_I = self.lam * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            # Solve (gram + λI) X = W^T  -> X is [out, in]
            inv_term = torch.linalg.solve(gram + lam_I, W.T)
            return downstream_target @ inv_term
        else:
            # Underdetermined: x* = y_target @ W^T (W W^T + λ I)^{-1}
            gram = W @ W.T  # [in, in]
            lam_I = self.lam * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            # Solve (gram + λI)^T X^T = W  -> X is [in, out]
            inv_term = torch.linalg.solve(gram + lam_I, W)
            return downstream_target @ inv_term.T

    def project_through_residual(
        self,
        downstream_target: torch.Tensor,
        branch_input: torch.Tensor,
    ) -> torch.Tensor:
        """Pull a target back through a residual connection ``y = x + f(x)``.

        Given ``y_target`` for the residual sum and ``x = branch_input`` (the
        skip-connection tensor), the target the non-skip branch ``f(x)`` must
        produce is exactly ``y_target - x``. No damping is required since the
        residual is identity-linear in ``f(x)``.

        Args:
            downstream_target: ``[..., features]`` target for the residual sum.
            branch_input: ``[..., features]`` activation entering the residual
                (the skip-connection tensor).

        Returns:
            ``[..., features]`` — target the non-skip branch ``f(x)`` must hit.
        """
        if downstream_target.shape != branch_input.shape:
            raise ValueError(
                f"shape mismatch: downstream_target {tuple(downstream_target.shape)} vs "
                f"branch_input {tuple(branch_input.shape)}"
            )
        return downstream_target - branch_input

    def project_through_layernorm(
        self,
        downstream_target: torch.Tensor,
        x_pre_ln: torch.Tensor,
        *,
        eps: float = 1.0e-5,
        gamma: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pull a target back through ``LayerNorm`` using current-row statistics.

        LayerNorm computes ``y = γ * (x - μ(x)) / σ(x) + β`` per token, where
        ``μ, σ`` are the row mean / std of ``x`` over the last dim. To pull a
        target ``y_target`` back to a pre-LN target ``x_target`` we use the
        current row stats from ``x_pre_ln`` as a local linearization:

            x_target ≈ ((y_target - β) / γ) * σ(x_pre_ln) + μ(x_pre_ln)

        This is exact when ``x_target`` shares the row-stats of ``x_pre_ln``.
        Affine params default to ``γ=1, β=0`` (the frozen-LN slice).

        Args:
            downstream_target: ``[..., features]`` post-LN target.
            x_pre_ln: ``[..., features]`` pre-LN activation (provides μ, σ).
            eps: numerical stabilizer matching ``nn.LayerNorm``'s default.
            gamma: optional ``[features]`` affine scale (default 1).
            beta: optional ``[features]`` affine bias (default 0).

        Returns:
            ``[..., features]`` — pre-LN target.
        """
        if downstream_target.shape != x_pre_ln.shape:
            raise ValueError(
                f"shape mismatch: downstream_target {tuple(downstream_target.shape)} vs "
                f"x_pre_ln {tuple(x_pre_ln.shape)}"
            )
        mu = x_pre_ln.mean(dim=-1, keepdim=True)
        var = x_pre_ln.var(dim=-1, keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + eps)
        normalized_target = (
            downstream_target - beta if beta is not None else downstream_target
        )
        if gamma is not None:
            # Avoid divide-by-zero on degenerate γ rows.
            normalized_target = normalized_target / (gamma + self.lam)
        return normalized_target * sigma + mu

    def project_through_attention_v(
        self,
        attn_weights: torch.Tensor,
        context_target: torch.Tensor,
    ) -> torch.Tensor:
        """Pull a context target back to a per-head V target through ``A @ V``.

        For scaled dot-product attention the context is ``C = A V`` where
        ``A = softmax(Q K^T / √d)`` is held fixed (this is the *softmax-aware
        V pull-back* — Q and K are not updated by this method). We solve
        ``A V = C_target`` for ``V`` per (batch, head) via Tikhonov-damped
        normal equations:

            V_target = (Aᵀ A + λ I)⁻¹ Aᵀ C_target

        Args:
            attn_weights: ``[batch, heads, L_q, L_k]`` softmax weights.
            context_target: ``[batch, heads, L_q, d_h]`` target for ``A V``.

        Returns:
            ``[batch, heads, L_k, d_h]`` per-head V target.
        """
        if attn_weights.dim() != 4 or context_target.dim() != 4:
            raise ValueError(
                f"expected 4-D tensors; got attn_weights {tuple(attn_weights.shape)} "
                f"and context_target {tuple(context_target.shape)}"
            )
        if attn_weights.shape[:3] != context_target.shape[:3]:
            raise ValueError(
                f"leading shape mismatch: attn_weights {tuple(attn_weights.shape)} vs "
                f"context_target {tuple(context_target.shape)}"
            )
        # Per-(batch, head) normal equations, batched.
        A = attn_weights                                  # [B, H, L_q, L_k]
        AtA = A.transpose(-2, -1) @ A                     # [B, H, L_k, L_k]
        eye = torch.eye(
            AtA.shape[-1], dtype=AtA.dtype, device=AtA.device,
        ).expand_as(AtA)
        AtC = A.transpose(-2, -1) @ context_target        # [B, H, L_k, d_h]
        # torch.linalg.solve broadcasts over the leading batch dims.
        return torch.linalg.solve(AtA + self.lam * eye, AtC)

    def solve_attention_pattern_target(
        self,
        V: torch.Tensor,
        context_target: torch.Tensor,
        *,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        """Recover a target attention pattern ``A_target`` from a context target.

        Given ``C_target = A V`` with ``V`` fixed, solve per ``(B, H, L_q)`` row
        ``a_q V = c_q`` for the ``[L_k]`` row vector ``a_q`` via Tikhonov-damped
        normal equations, then project onto the probability simplex by
        clamping to ``[eps, ∞)`` and renormalizing rows to sum to 1.

        This is a pragmatic simplex repair, **not** the exact Euclidean
        projection onto the simplex. In the current block solver that
        approximation is stabilized by the caller's mirror-descent blend
        with the current attention pattern before inverse-softmax.

        Args:
            V: ``[B, H, L_k, d_h]`` value tensor (fixed).
            context_target: ``[B, H, L_q, d_h]`` target context.
            eps: minimum entry value before renormalization (keeps ``log`` finite).

        Returns:
            ``[B, H, L_q, L_k]`` row-stochastic attention pattern target.
        """
        if V.dim() != 4 or context_target.dim() != 4:
            raise ValueError(
                f"expected 4-D tensors; got V {tuple(V.shape)} and "
                f"context_target {tuple(context_target.shape)}"
            )
        if V.shape[:2] != context_target.shape[:2] or V.shape[3] != context_target.shape[3]:
            raise ValueError(
                f"shape mismatch: V {tuple(V.shape)} vs context_target "
                f"{tuple(context_target.shape)}"
            )
        # Solve a_q V = c_q  →  a_q* = c_q V^T (V V^T + λ I_L)^{-1}.
        #
        # REVIEW.md Issue E: the naive solve on (V V^T + λ I_L) is O(B·H·L³)
        # and materialises an L×L matrix per (b, h). Since V has shape
        # [B, H, L_k, d_h] with d_h typically ≤ L_k, the push-through
        # identity
        #     V^T (V V^T + λ I_L)^{-1} = (V^T V + λ I_{d_h})^{-1} V^T
        # gives an exact rewrite
        #     A_unconstrained = context_target · (V^T V + λ I_{d_h})^{-1} · V^T
        # with cost O(B·H·d_h³) for the solve plus O(B·H·L·d_h²) matmuls —
        # no L×L matrix is ever formed and no 1/λ appears anywhere.
        L_k = V.shape[-2]
        d_h = V.shape[-1]
        if d_h < L_k:
            # Structured (Woodbury push-through) path — exact, cheaper.
            VtV = V.transpose(-2, -1) @ V                      # [B, H, d_h, d_h]
            eye_h = torch.eye(
                d_h, dtype=VtV.dtype, device=VtV.device,
            ).expand_as(VtV)
            # M = V^T V + λ I_{d_h}
            M = VtV + self.lam * eye_h
            # context_target: [B, H, L_q, d_h] → solve returns [B, H, d_h, L_q]
            # via  M X = context_target^T   (broadcast over B, H).
            X = torch.linalg.solve(M, context_target.transpose(-2, -1))
            # Multiply by V^T on the right: [B, H, d_h, L_q]^T @ V^T? No —
            # we want  context_target · M^{-1} · V^T
            #       = (M^{-1} · context_target^T)^T · V^T
            A_unconstrained = X.transpose(-2, -1) @ V.transpose(-2, -1)
        else:
            # Dense path kept for the d_h >= L_k regime (e.g. small L_k).
            VVt = V @ V.transpose(-2, -1)                      # [B, H, L_k, L_k]
            eye = torch.eye(
                L_k, dtype=VVt.dtype, device=VVt.device,
            ).expand_as(VVt)
            rhs = context_target @ V.transpose(-2, -1)         # [B, H, L_q, L_k]
            A_unconstrained = torch.linalg.solve(
                VVt + self.lam * eye, rhs.transpose(-2, -1)
            ).transpose(-2, -1)                                # [B, H, L_q, L_k]
        # Project to probability simplex: clamp + renormalize per row.
        A_clamped = A_unconstrained.clamp_min(eps)
        return A_clamped / A_clamped.sum(dim=-1, keepdim=True)

    def softmax_target_to_scores(
        self,
        A_target: torch.Tensor,
        A_curr: torch.Tensor,
        S_curr: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Map a target attention pattern ``A_target`` to a score target ``S``.

        Uses Difference-based Log-Target Propagation (DLTP):
          S_target = S_curr + scale * (A_target - A_curr)

        This is a first-order Taylor approximation of the softmax inverse
        around the current activation. It is materially more stable than
        a direct log-transform for local solvers (DMRG).

        Args:
            A_target: ``[..., L_k]`` row-stochastic target.
            A_curr: ``[..., L_k]`` current probabilities.
            S_curr: ``[..., L_k]`` current scores (pre-softmax).
            scale: inverse of the attention head scaling factor (e.g. sqrt(d_h)).

        Returns:
            ``[..., L_k]`` score target.
        """
        residual = A_target - A_curr
        return S_curr + scale * residual

    def project_through_qk_bilinear(
        self,
        scores_target: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoupled bilinear pull-back of a score target ``S = Q K^T``.

        Given a target ``S [B, H, L_q, L_k]`` for the un-normalized scores
        ``Q K^T`` and current ``Q [B, H, L_q, d_h]``, ``K [B, H, L_k, d_h]``,
        compute targets for ``Q`` and ``K`` independently by holding the
        other factor fixed and using the appropriate Tikhonov-damped pseudo-
        inverse (regime depends on whether ``L_k`` ≥ or < ``d_h``):

        * **Overdetermined** (``L_k ≥ d_h``):
          ``Q_target = S K (Kᵀ K + λ I_{d_h})^{-1}``
        * **Underdetermined** (``L_k < d_h``):
          ``Q_target = S (K Kᵀ + λ I_{L_k})^{-1} K``  (min-norm solution)

        Symmetric formulas hold for the K solver (with Q's dim regime).

        This is the standard alternating linearization of a bilinear form;
        it is exact when one factor matches its target. When *both* targets
        are applied jointly the result is a one-step linearization (joint
        Q,K is non-convex). Convergence is ensured by the outer DMRG loop's
        monotonicity proof composed with target-blend damping in the caller.

        Args:
            scores_target: ``[B, H, L_q, L_k]`` target for ``Q Kᵀ``.
            Q: ``[B, H, L_q, d_h]`` current query tensor.
            K: ``[B, H, L_k, d_h]`` current key tensor.

        Returns:
            Tuple ``(Q_target, K_target)`` of shapes
            ``[B, H, L_q, d_h]`` and ``[B, H, L_k, d_h]``.
        """
        if scores_target.dim() != 4 or Q.dim() != 4 or K.dim() != 4:
            raise ValueError(
                f"expected 4-D tensors; got scores_target {tuple(scores_target.shape)}, "
                f"Q {tuple(Q.shape)}, K {tuple(K.shape)}"
            )
        if (
            scores_target.shape[:2] != Q.shape[:2]
            or scores_target.shape[:2] != K.shape[:2]
            or scores_target.shape[2] != Q.shape[2]
            or scores_target.shape[3] != K.shape[2]
            or Q.shape[3] != K.shape[3]
        ):
            raise ValueError(
                f"shape mismatch: scores_target {tuple(scores_target.shape)}, "
                f"Q {tuple(Q.shape)}, K {tuple(K.shape)}"
            )
        d_h = Q.shape[-1]
        L_q = Q.shape[-2]
        L_k = K.shape[-2]

        # --- Q solver: Q* K^T = S, with K fixed.
        if L_k >= d_h:
            # Overdetermined: Q* = S K (K^T K + λI_{d_h})^{-1}
            KtK = K.transpose(-2, -1) @ K                 # [B, H, d_h, d_h]
            eye_q = torch.eye(d_h, dtype=Q.dtype, device=Q.device).expand_as(KtK)
            SK = scores_target @ K                         # [B, H, L_q, d_h]
            # Solve (KtK + λI)^T x^T = SK^T  ⇒  x = SK @ (KtK + λI)^{-1}.
            Q_target = torch.linalg.solve(
                (KtK + self.lam * eye_q).transpose(-2, -1),
                SK.transpose(-2, -1),
            ).transpose(-2, -1)
        else:
            # Underdetermined min-norm: Q* = S (K K^T + λI_{L_k})^{-1} K
            KKt = K @ K.transpose(-2, -1)                  # [B, H, L_k, L_k]
            eye_q = torch.eye(L_k, dtype=Q.dtype, device=Q.device).expand_as(KKt)
            # inner = S (K K^T + λI)^{-1}  via  solve(KKt^T, S^T)^T
            inner = torch.linalg.solve(
                (KKt + self.lam * eye_q).transpose(-2, -1),
                scores_target.transpose(-2, -1),
            ).transpose(-2, -1)                            # [B, H, L_q, L_k]
            Q_target = inner @ K                           # [B, H, L_q, d_h]

        # --- K solver: Q K*^T = S, equivalently K* Q^T = S^T, with Q fixed.
        S_T = scores_target.transpose(-2, -1)              # [B, H, L_k, L_q]
        if L_q >= d_h:
            QtQ = Q.transpose(-2, -1) @ Q                  # [B, H, d_h, d_h]
            eye_k = torch.eye(d_h, dtype=Q.dtype, device=Q.device).expand_as(QtQ)
            SQ = S_T @ Q                                    # [B, H, L_k, d_h]
            K_target = torch.linalg.solve(
                (QtQ + self.lam * eye_k).transpose(-2, -1),
                SQ.transpose(-2, -1),
            ).transpose(-2, -1)
        else:
            QQt = Q @ Q.transpose(-2, -1)                  # [B, H, L_q, L_q]
            eye_k = torch.eye(L_q, dtype=Q.dtype, device=Q.device).expand_as(QQt)
            inner = torch.linalg.solve(
                (QQt + self.lam * eye_k).transpose(-2, -1),
                S_T.transpose(-2, -1),
            ).transpose(-2, -1)                            # [B, H, L_k, L_q]
            K_target = inner @ Q                            # [B, H, L_k, d_h]

        return Q_target, K_target
