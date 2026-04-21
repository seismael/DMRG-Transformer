"""Strict OOD interfaces (Protocols) mirroring ARCHITECTURE.md Rust traits.

These Protocols enforce the SOLID interface segregation defined in
``docs/ARCHITECTURE.md`` §4. Concrete implementations live under
``dmrg_transformer.tt``, ``dmrg_transformer.optim``, and
``dmrg_transformer.propagation``.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

Tensor3D = torch.Tensor  # shape [r_{k-1}, p_k, r_k]
Tensor = torch.Tensor


@runtime_checkable
class ITensorTrain(Protocol):
    """Geometric encapsulation of a factorized weight space.

    Mirrors ``ITensorTrain`` from ARCHITECTURE.md §4.1.
    """

    def orthogonalize_left(self, core_index: int) -> None: ...
    def orthogonalize_right(self, core_index: int) -> None: ...
    def get_core(self, index: int) -> Tensor3D: ...
    def update_core(self, index: int, new_core: Tensor3D) -> None: ...
    def contract_forward(self, input: Tensor) -> Tensor: ...


@runtime_checkable
class ITargetPropagator(Protocol):
    """Replacement for the Backpropagation Chain Rule (ARCHITECTURE.md §4.2)."""

    def compute_layer_target(
        self, global_target: Tensor, current_layer_out: Tensor
    ) -> Tensor: ...


@runtime_checkable
class IDMRGOptimizer(Protocol):
    """Exact-solver replacement for SGD/Adam (ARCHITECTURE.md §4.3)."""

    def sweep(self, tt: ITensorTrain, target: Tensor, max_rank: int) -> float: ...
    def solve_local_core(
        self, left_block: Tensor, right_block: Tensor, target: Tensor
    ) -> Tensor: ...
    def truncate_svd(self, exact_core: Tensor, max_rank: int) -> Tensor3D: ...
