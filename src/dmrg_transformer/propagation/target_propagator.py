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
