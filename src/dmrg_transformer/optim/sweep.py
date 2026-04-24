"""DMRG optimizer implementing ``IDMRGOptimizer`` (ARCHITECTURE.md §4.3).

The :class:`DMRGOptimizer` orchestrates alternating left-to-right and
right-to-left sweeps of the local exact solver, with SVD truncation enforcing
the maximum TT-rank bound (AGENTS.md Constraint 3). It holds **no gradient
state** and exposes no learning rate (Constraints 1–2).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from dmrg_transformer.optim.local_solver import LocalSolveResult, solve_local_core
from dmrg_transformer.tt.environments import EnvironmentCache
from dmrg_transformer.tt.tensor_train import TensorTrain


@dataclass
class SweepReport:
    """Bookkeeping for a full left-to-right + right-to-left sweep."""

    initial_mse: float
    final_mse: float
    local_steps: int
    tiers: list[int]  # SVD tier used at each step (1=GPU, 2=gesdd, 3=gesvd, 4=noise)


class DMRGOptimizer:
    """Exact-solver replacement for SGD/Adam.

    Args:
        max_rank: TT-rank bound enforced after each local SVD truncation.
        lam: Tikhonov damping factor (NUMERICAL_STABILITY §3). Default ``1e-5``.
        clamp_target: apply Huber ±5σ clamp to targets (NUMERICAL_STABILITY §5).
        adaptive_threshold: when set (e.g. ``1e-4``), each local SVD picks the
            smallest rank ``r ≤ max_rank`` whose discarded squared-singular-
            value mass is at most ``adaptive_threshold`` of the total
            (plan §C5 / discarded-mass rule). When ``None`` (default), every
            local truncation uses the fixed ``max_rank`` for backward
            compatibility with all existing benchmarks and tests.
    """

    def __init__(
        self,
        max_rank: int,
        *,
        lam: float = 1.0e-5,
        clamp_target: bool = True,
        adaptive_threshold: float | None = None,
    ) -> None:
        if max_rank <= 0:
            raise ValueError(f"max_rank must be positive, got {max_rank}")
        self.max_rank = int(max_rank)
        self.lam = float(lam)
        self.clamp_target = bool(clamp_target)
        self.adaptive_threshold = (
            None if adaptive_threshold is None else float(adaptive_threshold)
        )

    # -- IDMRGOptimizer ---------------------------------------------------------

    def solve_local_core(
        self,
        tt: TensorTrain,
        X: torch.Tensor,
        Y: torch.Tensor,
        k: int,
        direction: str = "left",
        L: torch.Tensor | None = None,
        R: torch.Tensor | None = None,
    ) -> LocalSolveResult:
        return solve_local_core(
            tt, X, Y, k,
            max_rank=self.max_rank,
            lam=self.lam,
            direction=direction,
            clamp_target=self.clamp_target,
            adaptive_threshold=self.adaptive_threshold,
            L=L, R=R,
        )

    def truncate_svd(self, exact_core: torch.Tensor, max_rank: int) -> torch.Tensor:
        """Expose Eckart–Young–Mirsky truncation as a standalone primitive."""
        from dmrg_transformer.core.svd import robust_svd, truncate

        if exact_core.ndim != 3:
            raise ValueError(f"core must be 3D, got {tuple(exact_core.shape)}")
        r_l, p_k, r_r = exact_core.shape
        C = exact_core.reshape(r_l * p_k, r_r)
        trunc = truncate(robust_svd(C), max_rank)
        new_left = trunc.U.reshape(r_l, p_k, trunc.S.shape[0]).to(dtype=exact_core.dtype)
        return new_left

    def sweep(
        self,
        tt: TensorTrain,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> SweepReport:
        """Full bidirectional sweep: L→R then R→L.

        Ensures the TT is properly gauged before each direction via
        :mod:`dmrg_transformer.tt.gauge`.
        """
        from dmrg_transformer.tt.gauge import (
            orthogonalize_left_to,
            orthogonalize_right_to,
        )

        initial_pred = X @ tt.to_dense()
        initial_mse = float(torch.mean((initial_pred - Y) ** 2).item())

        tiers: list[int] = []
        steps = 0
        d = tt.num_cores

        # Prepare for L→R sweep: right-orthogonalize everything so gauge center is at 0.
        orthogonalize_right_to(tt, 0)
        cache = EnvironmentCache(tt, X)

        # L→R Sweep
        for k in range(d):
            L_block = cache.get_left(k)
            R_block = cache.get_right(k + 1)

            res = self.solve_local_core(tt, X, Y, k, direction="left", L=L_block, R=R_block)

            # After updating core k and k+1 (gauge shift), invalidate both directions
            # for any block that might use them.
            cache.invalidate_left(k + 1)
            cache.invalidate_right(k + 1)

            steps += 1
            if hasattr(res, "tier"):
                tiers.append(int(res.tier))

        # R→L Sweep
        orthogonalize_left_to(tt, d - 1)
        cache = EnvironmentCache(tt, X)

        for k in range(d - 1, -1, -1):
            L_block = cache.get_left(k)
            R_block = cache.get_right(k + 1)

            res = self.solve_local_core(tt, X, Y, k, direction="right", L=L_block, R=R_block)

            cache.invalidate_left(k)
            cache.invalidate_right(k)

            steps += 1
            if hasattr(res, "tier"):
                tiers.append(int(res.tier))

        final_pred = X @ tt.to_dense()
        final_mse = float(torch.mean((final_pred - Y) ** 2).item())

        return SweepReport(
            initial_mse=initial_mse, final_mse=final_mse,
            local_steps=steps, tiers=tiers,
        )
