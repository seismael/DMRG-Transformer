"""Tensor Train (TT) factorized weight space.

Implements ``ITensorTrain`` from ARCHITECTURE.md §4.1 using the simplified
3D core convention from TENSOR_TOPOLOGY.md §2:

    Core G_k has shape ``[r_{k-1}, p_k, r_k]`` with ``p_k = i_k * j_k``.

Boundary conditions ``r_0 = r_d = 1`` are enforced as invariants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

import torch

from dmrg_transformer.core.svd import discarded_energy, robust_svd, truncate


@dataclass(frozen=True)
class TruncationReport:
    """Per-step record of the TT-SVD decomposition (for Gate 1 verification)."""

    discarded_singular_values: list[torch.Tensor] = field(default_factory=list)

    def total_frobenius_bound(self) -> float:
        """Sqrt of the sum-of-squares of all discarded singular values across all cuts.

        This is the exact Eckart–Young–Mirsky upper bound on the reconstruction error
        of the TT-SVD (a sum-of-cuts upper bound; reconstruction error must be ≤ this).
        """
        total = 0.0
        for s in self.discarded_singular_values:
            total += float(torch.sum(s.to(torch.float64) ** 2).item())
        return total**0.5


class TensorTrain:
    """A factorized weight space ``W ∈ R^{N×M}`` stored as TT-cores.

    Args:
        cores: list of 3D tensors with shapes ``[r_{k-1}, p_k, r_k]``.
        input_dims:  ``[i_1, ..., i_d]``  with ``prod(input_dims) == N``.
        output_dims: ``[j_1, ..., j_d]``  with ``prod(output_dims) == M``.

    Invariants:
        * ``cores[0].shape[0] == 1`` and ``cores[-1].shape[-1] == 1`` (TENSOR_TOPOLOGY §2).
        * ``cores[k].shape[1] == input_dims[k] * output_dims[k]``.
        * Adjacent ranks match: ``cores[k].shape[-1] == cores[k+1].shape[0]``.
    """

    def __init__(
        self,
        cores: list[torch.Tensor],
        input_dims: list[int],
        output_dims: list[int],
    ) -> None:
        if len(cores) != len(input_dims) or len(cores) != len(output_dims):
            raise ValueError("cores, input_dims, output_dims must have equal length")
        if cores[0].shape[0] != 1:
            raise ValueError(f"r_0 must be 1, got {cores[0].shape[0]}")
        if cores[-1].shape[-1] != 1:
            raise ValueError(f"r_d must be 1, got {cores[-1].shape[-1]}")
        for k, G in enumerate(cores):
            if G.ndim != 3:
                raise ValueError(f"core {k} must be 3D, got shape {tuple(G.shape)}")
            expected_p = input_dims[k] * output_dims[k]
            if G.shape[1] != expected_p:
                raise ValueError(
                    f"core {k} physical dim is {G.shape[1]}, expected i_k*j_k={expected_p}"
                )
            if k > 0 and cores[k - 1].shape[-1] != G.shape[0]:
                raise ValueError(
                    f"rank mismatch at cut {k}: r_left={cores[k-1].shape[-1]} vs {G.shape[0]}"
                )
        self._cores: list[torch.Tensor] = [c.contiguous() for c in cores]
        self.input_dims: list[int] = list(input_dims)
        self.output_dims: list[int] = list(output_dims)
        self._orth_center: int | None = None  # gauge-center index, or None if unspecified

    # -- ITensorTrain ------------------------------------------------------------

    def get_core(self, index: int) -> torch.Tensor:
        return self._cores[index]

    def update_core(self, index: int, new_core: torch.Tensor) -> None:
        if new_core.ndim != 3:
            raise ValueError(f"new core must be 3D, got shape {tuple(new_core.shape)}")
        expected_p = self.input_dims[index] * self.output_dims[index]
        if new_core.shape[1] != expected_p:
            raise ValueError(
                f"new core physical dim {new_core.shape[1]} != i_k*j_k={expected_p}"
            )
        self._cores[index] = new_core.contiguous()

    def orthogonalize_left(self, core_index: int) -> None:
        # Imported here to avoid a circular import (gauge depends on TensorTrain).
        from dmrg_transformer.tt.gauge import orthogonalize_left_to

        orthogonalize_left_to(self, core_index)

    def orthogonalize_right(self, core_index: int) -> None:
        from dmrg_transformer.tt.gauge import orthogonalize_right_to

        orthogonalize_right_to(self, core_index)

    def contract_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward inference: ``y = X · W`` where W is the TT.

        Args:
            x: shape ``[batch, N]`` with ``N = prod(input_dims)``.

        Returns:
            ``[batch, M]`` with ``M = prod(output_dims)``.
        """
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"input must be [batch, {self.in_features}], got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        # V starts as [batch, r_left=1, i_1, i_2, ..., i_d]
        V = x.reshape(batch, 1, *self.input_dims)
        out_consumed: list[int] = []
        for k, G in enumerate(self._cores):
            i_k = self.input_dims[k]
            j_k = self.output_dims[k]
            r_left = G.shape[0]
            r_right = G.shape[2]
            G4 = G.reshape(r_left, i_k, j_k, r_right)
            J_size = prod(out_consumed) if out_consumed else 1
            I_rest = self.input_dims[k + 1 :]
            I_rest_size = prod(I_rest) if I_rest else 1
            V_flat = V.reshape(batch, J_size, r_left, i_k, I_rest_size)
            # 'b J r i I, r i j R -> b J j R I'
            new_V = torch.einsum("bJriI,rijR->bJjRI", V_flat, G4)
            out_consumed.append(j_k)
            V = new_V.reshape(batch, *out_consumed, r_right, *I_rest)
        # Final V: [batch, j_1, ..., j_d, r_d=1]; squeeze trailing rank.
        return V.reshape(batch, self.out_features)

    # -- helpers / properties ----------------------------------------------------

    @property
    def cores(self) -> list[torch.Tensor]:
        return self._cores

    @property
    def num_cores(self) -> int:
        return len(self._cores)

    @property
    def ranks(self) -> list[int]:
        """Bond dimensions ``[r_0, r_1, ..., r_d]`` (with ``r_0 = r_d = 1``)."""
        out = [self._cores[0].shape[0]]
        for G in self._cores:
            out.append(G.shape[-1])
        return out

    @property
    def in_features(self) -> int:
        return prod(self.input_dims)

    @property
    def out_features(self) -> int:
        return prod(self.output_dims)

    @property
    def orthogonality_center(self) -> int | None:
        return self._orth_center

    def _set_orth_center(self, k: int | None) -> None:
        self._orth_center = k

    # -- construction ------------------------------------------------------------

    def to_dense(self) -> torch.Tensor:
        """Reconstruct the full ``[N, M]`` matrix by sequential core contraction."""
        # Start with the first core flattened: shape (1, p_1, r_1) -> (p_1, r_1)
        T = self._cores[0].reshape(self.input_dims[0] * self.output_dims[0], -1)
        for k in range(1, self.num_cores):
            G = self._cores[k]  # (r_{k-1}, p_k, r_k)
            r_left, p_k, r_right = G.shape
            # T: (P_prev, r_left); fold core in
            T = T @ G.reshape(r_left, p_k * r_right)
            # T: (P_prev, p_k * r_right) -> reshape to (P_prev*p_k, r_right)
            T = T.reshape(-1, r_right)
        # T: (prod(p_k), 1)
        T = T.reshape(*self.input_dims, *self.output_dims)
        # Permute interleaved (i_1, j_1, i_2, j_2, ...) layout back to (i..., j...).
        # Currently order is (i_1, j_1, i_2, j_2, ...) since we stored p_k = i_k*j_k.
        # Wait: we stored core with p_k = i_k*j_k, so when reshaping prod(p_k) to
        # (*input_dims, *output_dims) we need to interleave correctly.
        return self._interleaved_to_matrix(T)

    def _interleaved_to_matrix(self, T: torch.Tensor) -> torch.Tensor:
        """Convert tensor with axes ``(p_1, ..., p_d)`` (each p_k = i_k*j_k) to ``[N,M]``."""
        d = self.num_cores
        # Reshape each p_k -> (i_k, j_k): final shape (i_1, j_1, i_2, j_2, ..., i_d, j_d)
        shape_pairs: list[int] = []
        for k in range(d):
            shape_pairs.extend([self.input_dims[k], self.output_dims[k]])
        T = T.reshape(*shape_pairs)
        # Permute to (i_1, ..., i_d, j_1, ..., j_d)
        i_axes = [2 * k for k in range(d)]
        j_axes = [2 * k + 1 for k in range(d)]
        T = T.permute(*i_axes, *j_axes).contiguous()
        return T.reshape(self.in_features, self.out_features)

    @classmethod
    def from_dense(
        cls,
        W: torch.Tensor,
        input_dims: list[int],
        output_dims: list[int],
        max_rank: int,
    ) -> tuple[TensorTrain, TruncationReport]:
        """Classical TT-SVD decomposition of a dense matrix.

        Implements the algorithm from SOLVER_MATH.md §II / TENSOR_TOPOLOGY.md §6.
        Returns the ``TensorTrain`` and a :class:`TruncationReport` containing the
        discarded singular values at each cut so Validation Gate 1 can verify the
        reconstruction error matches the theoretical Eckart–Young–Mirsky bound.
        """
        N = prod(input_dims)
        M = prod(output_dims)
        if W.shape != (N, M):
            raise ValueError(f"W shape {tuple(W.shape)} != ({N}, {M})")
        d = len(input_dims)
        if d != len(output_dims):
            raise ValueError("input_dims and output_dims must be the same length")

        # Reshape W -> (i_1, ..., i_d, j_1, ..., j_d) -> interleave to (p_1, ..., p_d).
        T = W.reshape(*input_dims, *output_dims)
        perm: list[int] = []
        for k in range(d):
            perm.append(k)
            perm.append(d + k)
        T = T.permute(*perm).contiguous()
        p_dims = [input_dims[k] * output_dims[k] for k in range(d)]
        T = T.reshape(*p_dims)

        cores: list[torch.Tensor] = []
        report = TruncationReport()
        r_left = 1
        # Sequential left-to-right SVD.
        for k in range(d - 1):
            p_k = p_dims[k]
            rest = prod(p_dims[k + 1 :])
            M_k = T.reshape(r_left * p_k, rest)
            full = robust_svd(M_k)
            r_new = min(max_rank, full.S.shape[0])
            report.discarded_singular_values.append(full.S[r_new:].detach().clone())
            trunc = truncate(full, r_new)
            core = trunc.U.reshape(r_left, p_k, r_new)
            cores.append(core)
            # Carry S·Vh forward.
            T = (trunc.S.unsqueeze(1) * trunc.Vh).reshape(r_new, *p_dims[k + 1 :])
            r_left = r_new
        # Last core: T has shape (r_{d-1}, p_d), pad trailing rank=1.
        last = T.reshape(r_left, p_dims[-1], 1)
        cores.append(last)

        return cls(cores=cores, input_dims=input_dims, output_dims=output_dims), report
