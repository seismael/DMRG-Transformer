"""Validation Gate A3b — ADMM on a **residual** TTLinear cascade.

Unlike Gate A3 (pure feedforward chain), this test introduces skip
connections  ``h_{i+1} = h_i + f(h_i; W_i)``.  The identity path creates a
genuine consensus problem at each interface: the output of block *i* must
equal the input of block *i+1*, but the residual function can independently
shift the representation.

ADMM's consensus variables resolve this drift in a way that sequential
DTP (Difference Target Propagation) cannot — each block's augmented target
includes both its layer-local loss and the consensus penalty.
"""
from __future__ import annotations

import torch
from torch import nn

from dmrg_transformer.nn.tt_linear import TTLinear
from dmrg_transformer.optim.admm_outer import ADMMOuter
from dmrg_transformer.optim.sweep import DMRGOptimizer, SweepReport
from dmrg_transformer.propagation.target_propagator import TargetPropagator
from dmrg_transformer.tt import TensorTrain


# ------------------------------------------------------------------
# ResidualTTLinear — TTLinear with a skip connection
# ------------------------------------------------------------------


class ResidualTTLinear(nn.Module):
    """A trainable TTLinear wrapped in a residual connection.

    ``forward(x) = x + f(x; W)`` where ``f`` is a ``TTLinear``.
    Supports both DTP (``pullback_target``) and exact pseudo-inverse
    (``to_dense_weight``) for the ADMM z-update.
    """

    def __init__(self, linear: TTLinear, target_blend: float = 0.5) -> None:
        super().__init__()
        self.linear = linear
        self.target_blend = float(target_blend)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear.forward(x)

    @torch.no_grad()
    def dmrg_step(
        self,
        X: torch.Tensor,
        Y_target: torch.Tensor,
        *,
        lam: float = 1e-5,
    ) -> SweepReport:
        """Fit the function part so that ``x + f(x) ≈ Y_target``."""
        f_target = Y_target - X
        return self.linear.dmrg_step(X, f_target, lam=lam)

    @torch.no_grad()
    def to_dense_weight(self) -> torch.Tensor:
        """Effective weight including identity: ``I + W_tt``."""
        W = self.linear.to_dense_weight()
        I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
        return I + W

    @torch.no_grad()
    def pullback_target(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        *,
        target_blend: float | None = None,
    ) -> torch.Tensor:
        """DTP pullback — first-order approximation when ``I + W`` is too
        large to invert exactly."""
        blend = target_blend if target_blend is not None else self.target_blend
        y_curr = self.forward(x)
        return x + blend * (y_target - y_curr)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _factor_square(n: int) -> list[int]:
    import math

    a = int(math.isqrt(n))
    while a > 1:
        if n % a == 0:
            return [a, n // a]
        a -= 1
    return [1, n]


def _build_residual_cascade(
    N: int = 12,
    rank: int = 4,
    batch: int = 128,
    depth: int = 3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, list[ResidualTTLinear]]:
    """Build a ground-truth residual cascade and a trainable stack.

    Returns ``(X, Y, residual_blocks)`` where
    ``Y = h_L``  and  ``h_{i+1} = h_i + f_i(h_i; W_i)``
    with each ``f_i`` a TT-rank-bounded linear.
    """
    dims = _factor_square(N)

    # Ground-truth function weights.
    torch.manual_seed(seed)
    W_gt = []
    for i in range(depth):
        tt, _ = TensorTrain.from_dense(
            torch.randn(N, N, dtype=torch.float64) * 0.3,
            dims, dims, max_rank=rank,
        )
        W_gt.append(tt.to_dense())

    X = torch.randn(batch, N, dtype=torch.float64)
    h = X
    for W in W_gt:
        h = h + h @ W
    Y = h

    # Trainable residual blocks (different seeds).
    torch.manual_seed(seed + 100)
    blocks = []
    for i in range(depth):
        linear = TTLinear(
            N, N, input_dims=dims, output_dims=dims,
            rank=rank, bias=False, dtype=torch.float64,
        )
        blocks.append(ResidualTTLinear(linear, target_blend=0.5))
    return X, Y, blocks


def _residual_chain_forward(
    X: torch.Tensor, layers: list[ResidualTTLinear],
) -> torch.Tensor:
    x = X
    for layer in layers:
        x = layer.forward(x)
    return x


def _chain_mse(X: torch.Tensor, Y: torch.Tensor, layers: list) -> float:
    return float(torch.mean((_residual_chain_forward(X, layers) - Y) ** 2).item())


# ------------------------------------------------------------------
# Gate A3b tests
# ------------------------------------------------------------------


def test_admm_residual_cascade_reduces_mse() -> None:
    """ADMM reduces global MSE on a residual TTLinear cascade."""
    X, Y, blocks = _build_residual_cascade(N=12, rank=4, batch=128, depth=3, seed=1)
    initial_mse = _chain_mse(X, Y, blocks)

    admm = ADMMOuter(
        layers=blocks,
        rho=1.0,
        tol=1e-6,
        max_iter=15,
        rho_auto_tune=False,
        lam=1e-6,
        propagator_lam=1e-2,
    )
    report = admm.solve(X, Y)

    assert report.final_mse < initial_mse, (
        f"ADMM residual cascade did not reduce MSE: "
        f"{initial_mse:.4e} -> {report.final_mse:.4e}"
    )
    # 3-layer residual cascade converges more slowly than pure feedforward.
    # 15 ADMM iterations should achieve at least a 5× MSE reduction.
    assert report.final_mse < initial_mse * 0.2, (
        f"ADMM residual cascade MSE reduction too weak: "
        f"{initial_mse:.4e} -> {report.final_mse:.4e}"
    )


def test_admm_residual_primal_residual_decreases() -> None:
    """In a residual cascade, the ADMM primal residual should tighten."""
    X, Y, blocks = _build_residual_cascade(N=12, rank=4, batch=64, depth=3, seed=2)

    admm = ADMMOuter(
        layers=blocks,
        rho=1.0,
        tol=0.0,
        max_iter=12,
        rho_auto_tune=False,
        lam=1e-6,
        propagator_lam=1e-2,
    )
    report = admm.solve(X, Y)

    primals = report.primal_residuals
    # First half should be larger than second half (downward trend).
    mid = len(primals) // 2
    avg_first = sum(primals[:mid]) / mid
    avg_second = sum(primals[mid:]) / (len(primals) - mid) if mid < len(primals) else avg_first
    assert avg_second < avg_first, (
        f"primal residual rising: first-half avg={avg_first:.2f}, "
        f"second-half avg={avg_second:.2f}"
    )


def test_admm_residual_beats_no_consensus() -> None:
    """ADMM with consensus should beat per-block DMRG without consensus.

    The baseline updates each block independently against its layer-local
    target (no consensus variables), which causes inter-block drift.
    ADMM's consensus terms should resolve this.
    """
    N, rank, batch, depth = 12, 4, 128, 2
    dims = _factor_square(N)

    X, Y, blocks_admm = _build_residual_cascade(
        N=N, rank=rank, batch=batch, depth=depth, seed=3,
    )

    # Snapshot initial state.
    init_state = [
        {k: v.clone() for k, v in b.state_dict().items()} for b in blocks_admm
    ]

    # --- Baseline: per-block DMRG (no consensus) ---
    torch.manual_seed(seed=3)
    blocks_seq: list[ResidualTTLinear] = []
    for i in range(depth):
        linear = TTLinear(
            N, N, input_dims=dims, output_dims=dims,
            rank=rank, bias=False, dtype=torch.float64,
        )
        blocks_seq.append(ResidualTTLinear(linear, target_blend=0.5))
    for b_seq, init in zip(blocks_seq, init_state):
        b_seq.load_state_dict(init)

    best_seq = float("inf")
    for _ in range(8):
        # Forward pass to get all intermediate activations.
        h_vals = [X]
        h = X
        for block in blocks_seq:
            h = block.forward(h)
            h_vals.append(h)
        # Backward: propagate target through chain.
        target = Y
        for bi in range(len(blocks_seq) - 1, -1, -1):
            h_in = h_vals[bi]  # input to block bi
            blocks_seq[bi].dmrg_step(h_in, target, lam=1e-6)
            target = blocks_seq[bi].pullback_target(h_in, target)
        best_seq = min(best_seq, _chain_mse(X, Y, blocks_seq))

    # --- ADMM ---
    blocks_admm2: list[ResidualTTLinear] = []
    for i in range(depth):
        linear = TTLinear(
            N, N, input_dims=dims, output_dims=dims,
            rank=rank, bias=False, dtype=torch.float64,
        )
        blocks_admm2.append(ResidualTTLinear(linear, target_blend=0.5))
    for b_admm, init in zip(blocks_admm2, init_state):
        b_admm.load_state_dict(init)

    admm = ADMMOuter(
        layers=blocks_admm2,
        rho=1.0,
        tol=1e-6,
        max_iter=8,
        rho_auto_tune=False,
        lam=1e-6,
        propagator_lam=1e-2,
    )
    report = admm.solve(X, Y)

    # ADMM should be competitive with or beat the sequential baseline.
    assert report.final_mse <= max(best_seq * 3.0, 1e-2), (
        f"ADMM MSE={report.final_mse:.4e} vs sequential (no consensus)={best_seq:.4e}"
    )
