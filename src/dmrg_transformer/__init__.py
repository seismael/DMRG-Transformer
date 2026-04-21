"""DMRG-Transformer: post-Gradient-Descent neural optimization via Tensor Trains.

Public API surface (stable):

    from dmrg_transformer.nn import TTLinear, TTMultiHeadAttention
    from dmrg_transformer.optim import DMRGOptimizer
    from dmrg_transformer.tt import TensorTrain
    from dmrg_transformer.propagation import TargetPropagator

See ``docs/AGENTS.md`` and ``docs/SOLVER_MATH.md`` for the mathematical contract.
"""
from dmrg_transformer.nn.tt_linear import TTLinear
from dmrg_transformer.nn.tt_mha import TTMultiHeadAttention
from dmrg_transformer.optim.sweep import DMRGOptimizer
from dmrg_transformer.propagation.target_propagator import TargetPropagator
from dmrg_transformer.tt.tensor_train import TensorTrain, TruncationReport

__all__ = [
    "DMRGOptimizer",
    "TTLinear",
    "TTMultiHeadAttention",
    "TargetPropagator",
    "TensorTrain",
    "TruncationReport",
]

__version__ = "0.1.0"
