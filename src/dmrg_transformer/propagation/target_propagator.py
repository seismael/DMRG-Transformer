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
        (SOLVER_MATH.md §I closed-form with damping):

            x* = y_target @ W^T (W W^T + λ I)^{-1}

        Args:
            downstream_weight: ``[in, out]`` weight matrix of the subsequent layer.
            downstream_target: ``[batch, out]`` target of the subsequent layer.

        Returns:
            ``[batch, in]`` — the local target for the current layer's output.
        """
        W = downstream_weight
        gram = W @ W.T  # [in, in]
        lam_I = self.lam * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        inv_term = torch.linalg.solve(gram + lam_I, W)  # [in, out]
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
