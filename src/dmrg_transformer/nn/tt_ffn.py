"""``TTFeedForward`` — position-wise feed-forward block with TT-factorized linears.

Composition: ``TTLinear(embed → hidden) → GELU → TTLinear(hidden → embed)``.

The DMRG update follows the same target-propagation pattern as the
[scripts/train_real_world_classifier.py](../../../scripts/train_real_world_classifier.py)
two-layer MLP loop:

1. Sweep ``fc2`` against the externally-supplied output target ``Y``.
2. Pull ``Y`` back through ``fc2`` (Tikhonov-damped pseudo-inverse) to obtain a
   post-GELU target for ``fc1``'s output.
3. Convert post-GELU target to a pre-GELU target via an active-mask blend
   (positive-derivative proxy — analogous to ReLU active-mask trick).
4. Sweep ``fc1`` against the pre-GELU target.

Parallels: this is the FFN sub-block of a Transformer encoder layer, the
direct counterpart to ``nn.Sequential(nn.Linear, nn.GELU, nn.Linear)``.
"""
from __future__ import annotations

import torch
from torch import nn

from dmrg_transformer.nn.tt_linear import TTLinear
from dmrg_transformer.optim.sweep import SweepReport
from dmrg_transformer.propagation.target_propagator import TargetPropagator


class TTFeedForward(nn.Module):
    """Position-wise feed-forward block with TT-factorized weight matrices.

    Args:
        embed_dim: input/output dimension.
        hidden_dim: width of the inner activation.
        embed_dims: factorization of ``embed_dim`` for the TT cores.
        hidden_dims: factorization of ``hidden_dim`` for the TT cores.
        rank: TT-rank bound for both linears.
        propagator_lam: Tikhonov damping for the fc2 → fc1 pull-back.
        dtype: storage dtype for TT cores.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        *,
        embed_dims: list[int],
        hidden_dims: list[int],
        rank: int,
        propagator_lam: float = 1.0e-2,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.fc1 = TTLinear(
            in_features=embed_dim, out_features=hidden_dim,
            input_dims=embed_dims, output_dims=hidden_dims,
            rank=rank, bias=True, dtype=dtype,
        )
        self.fc2 = TTLinear(
            in_features=hidden_dim, out_features=embed_dim,
            input_dims=hidden_dims, output_dims=embed_dims,
            rank=rank, bias=True, dtype=dtype,
        )
        self.propagator = TargetPropagator(lam=propagator_lam)

    # -- forward ---------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        z1 = self.fc1(flat)
        h1 = torch.nn.functional.gelu(z1)
        z2 = self.fc2(h1)
        return z2.reshape(*x.shape[:-1], z2.shape[-1])

    @torch.no_grad()
    def forward_with_cache(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning ``(y, z1_pre_gelu, h1_post_gelu)`` over flat tokens."""
        flat = x.reshape(-1, x.shape[-1])
        z1 = self.fc1(flat)
        h1 = torch.nn.functional.gelu(z1)
        z2 = self.fc2(h1)
        return z2, z1, h1

    # -- DMRG update -----------------------------------------------------------

    @torch.no_grad()
    def dmrg_step(
        self,
        X: torch.Tensor,
        Y_target: torch.Tensor,
        *,
        lam: float = 1.0e-5,
        target_blend: float = 0.5,
    ) -> dict[str, SweepReport]:
        """Exact-solver update: fc2 sweep → propagate back → fc1 sweep.

        Args:
            X: ``[batch, embed_dim]`` flat input tokens.
            Y_target: ``[batch, embed_dim]`` target for the FFN output.
            lam: Tikhonov damping inside each :meth:`TTLinear.dmrg_step`.
            target_blend: ``∈ [0, 1]`` blending factor for the pre-GELU target
                (1.0 = greedy, 0.0 = no update). Mirrors the outer-loop
                "learning rate" used in :mod:`scripts.train_real_world_classifier`.

        Returns:
            ``{"fc1": SweepReport, "fc2": SweepReport}``.
        """
        X_flat = X.reshape(-1, X.shape[-1])
        Y_flat = Y_target.reshape(-1, Y_target.shape[-1])

        # 1) Forward to expose intermediate activations.
        _, z1, h1 = self.forward_with_cache(X_flat)

        # 2) Sweep fc2 against the supplied output target.
        rep_fc2 = self.fc2.dmrg_step(h1, Y_flat, lam=lam)

        # 3) Pull the output target back through fc2 to a post-GELU target.
        W2_dense = self.fc2.to_dense_weight()  # [hidden, embed]
        Y_minus_b = Y_flat - self.fc2._bias if self.fc2._has_bias else Y_flat
        h1_target = self.propagator.project_through_linear(W2_dense, Y_minus_b)

        # 4) Convert post-GELU target to pre-GELU target via active-mask blend.
        #    GELU's derivative is positive almost everywhere except small
        #    negative inputs; we use the same active-mask blending pattern as
        #    the ReLU loop in train_real_world_classifier.py for a clean,
        #    documented approximation.
        active = z1 > 0
        z1_target_active = target_blend * h1_target + (1.0 - target_blend) * z1
        z1_target = torch.where(active, z1_target_active, z1)

        # 5) Sweep fc1 against the pre-GELU target.
        rep_fc1 = self.fc1.dmrg_step(X_flat, z1_target, lam=lam)

        return {"fc1": rep_fc1, "fc2": rep_fc2}

    @property
    def num_parameters(self) -> int:
        return self.fc1.num_parameters + self.fc2.num_parameters
