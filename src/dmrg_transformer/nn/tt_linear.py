"""``TTLinear`` — drop-in replacement for ``nn.Linear`` using a TensorTrain.

Per AGENTS.md §2 mapping:
    ``nn.Linear(in, out)`` → ``TensorTrain(cores=[G_1, ..., G_d], ranks=r)``
"""
from __future__ import annotations

from math import prod

import torch
from torch import nn

from dmrg_transformer.optim.sweep import DMRGOptimizer, SweepReport
from dmrg_transformer.tt.tensor_train import TensorTrain


class TTLinear(nn.Module):
    """Linear layer whose weight matrix is stored as a Tensor-Train.

    The module exposes a standard ``forward`` (inference contraction) and an
    exact-solver update ``dmrg_step(X, Y)`` that replaces ``optimizer.step()``.

    Args:
        in_features: input dimension ``N``.
        out_features: output dimension ``M``.
        input_dims: factorization of ``in_features``. ``prod(input_dims) == N``.
        output_dims: factorization of ``out_features``. ``prod(output_dims) == M``.
        rank: maximum TT-rank (bond dimension).
        bias: whether to include an additive bias term.
        dtype: storage dtype for the TT cores. Defaults to ``torch.float64`` so
            that the DMRG solver can reach Gate-3 precision; downcast to
            ``float32`` for deployment once training converges.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        input_dims: list[int],
        output_dims: list[int],
        rank: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if prod(input_dims) != in_features:
            raise ValueError(f"prod(input_dims)={prod(input_dims)} != in_features={in_features}")
        if prod(output_dims) != out_features:
            raise ValueError(
                f"prod(output_dims)={prod(output_dims)} != out_features={out_features}"
            )
        if len(input_dims) != len(output_dims):
            raise ValueError("input_dims and output_dims must have the same length")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)

        # Initialize TT from a small-magnitude random dense matrix on the
        # project's authorized device (GPU by default).
        from dmrg_transformer.core.device import require_cuda

        device = require_cuda()
        scale = 1.0 / (in_features**0.5)
        W_init = torch.randn(in_features, out_features, dtype=dtype, device=device) * scale
        tt, _ = TensorTrain.from_dense(W_init, input_dims, output_dims, max_rank=rank)
        # Store cores as buffers (not nn.Parameter — AGENTS Constraint 1 forbids gradient
        # tracking on weights updated by the DMRG solver).
        for k, core in enumerate(tt.cores):
            self.register_buffer(f"_core_{k}", core.clone())
        self._num_cores = tt.num_cores
        if bias:
            self.register_buffer("_bias", torch.zeros(out_features, dtype=dtype, device=device))
            self._has_bias = True
        else:
            self._has_bias = False

    # -- TT view helpers --------------------------------------------------------

    def _view_tt(self) -> TensorTrain:
        cores = [getattr(self, f"_core_{k}") for k in range(self._num_cores)]
        return TensorTrain(cores=cores, input_dims=self.input_dims, output_dims=self.output_dims)

    def _commit_tt(self, tt: TensorTrain) -> None:
        for k, core in enumerate(tt.cores):
            # Shape may have shifted due to truncation/absorption — re-register.
            self.register_buffer(f"_core_{k}", core.contiguous())

    # -- public API --------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tt = self._view_tt()
        # Preserve input dtype where possible; cores are in self.dtype.
        x_cast = x.to(dtype=tt.get_core(0).dtype)
        y = tt.contract_forward(x_cast)
        if self._has_bias:
            y = y + self._bias
        return y.to(dtype=x.dtype)

    @torch.no_grad()
    def dmrg_step(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        lam: float = 1.0e-5,
        clamp_target: bool = True,
        adaptive_threshold: float | None = None,
    ) -> SweepReport:
        """Exact-solver weight update. Returns the sweep report (initial/final MSE)."""
        tt = self._view_tt()
        opt = DMRGOptimizer(
            max_rank=self.rank, lam=lam, clamp_target=clamp_target,
            adaptive_threshold=adaptive_threshold,
        )
        X_cast = X.to(dtype=tt.get_core(0).dtype)
        # Subtract bias from target so the solver focuses only on the TT part.
        if self._has_bias:
            Y_cast = (Y - self._bias).to(dtype=tt.get_core(0).dtype)
        else:
            Y_cast = Y.to(dtype=tt.get_core(0).dtype)
        report = opt.sweep(tt, X_cast, Y_cast)
        self._commit_tt(tt)
        # Refresh bias as the mean residual (closed-form optimum).
        if self._has_bias:
            residual = Y_cast - tt.contract_forward(X_cast)
            self._bias = residual.mean(dim=0).to(dtype=self._bias.dtype)
        return report

    def to_dense_weight(self) -> torch.Tensor:
        """Return the equivalent dense weight matrix (expensive — diagnostic use)."""
        return self._view_tt().to_dense()

    @property
    def num_parameters(self) -> int:
        """Parameter count across all TT cores (plus bias if present)."""
        p = sum(c.numel() for c in self._view_tt().cores)
        if self._has_bias:
            p += self._bias.numel()
        return p
