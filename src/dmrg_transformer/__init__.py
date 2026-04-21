"""DMRG-Transformer: post-Gradient-Descent neural optimization via Tensor Trains.

Public API surface (stable):

    from dmrg_transformer import TensorTrain, DMRGOptimizer
    from dmrg_transformer import TTLinear, TTMultiHeadAttention, TargetPropagator

See ``docs/AGENTS.md`` and ``docs/SOLVER_MATH.md`` for the mathematical contract.
Submodules are imported lazily so partial installations are possible during
phased development without provoking ``ImportError`` at top-level import.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = [
    "DMRGOptimizer",
    "TTLinear",
    "TTMultiHeadAttention",
    "TargetPropagator",
    "TensorTrain",
    "TruncationReport",
]

if TYPE_CHECKING:
    from dmrg_transformer.nn.tt_linear import TTLinear
    from dmrg_transformer.nn.tt_mha import TTMultiHeadAttention
    from dmrg_transformer.optim.sweep import DMRGOptimizer
    from dmrg_transformer.propagation.target_propagator import TargetPropagator
    from dmrg_transformer.tt.tensor_train import TensorTrain, TruncationReport


_LAZY_MAP = {
    "TensorTrain": ("dmrg_transformer.tt.tensor_train", "TensorTrain"),
    "TruncationReport": ("dmrg_transformer.tt.tensor_train", "TruncationReport"),
    "DMRGOptimizer": ("dmrg_transformer.optim.sweep", "DMRGOptimizer"),
    "TTLinear": ("dmrg_transformer.nn.tt_linear", "TTLinear"),
    "TTMultiHeadAttention": ("dmrg_transformer.nn.tt_mha", "TTMultiHeadAttention"),
    "TargetPropagator": (
        "dmrg_transformer.propagation.target_propagator",
        "TargetPropagator",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        import importlib

        module_path, attr = _LAZY_MAP[name]
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
