"""Validation Gate 2 (AGENTS.md §3, Phase II).

After a left-orthogonalization sweep to core ``k``, the merged left environment
block ``L_{<k}`` must satisfy ``L^T L = I`` to machine precision in float64
(per NUMERICAL_STABILITY.md §2).
"""
from __future__ import annotations

import torch

from dmrg_transformer.tt import TensorTrain
from dmrg_transformer.tt.gauge import (
    merged_left_block,
    merged_right_block,
    orthogonalize_left_to,
    orthogonalize_right_to,
)


def _make_random_tt(seed: int = 42, dtype: torch.dtype = torch.float64) -> TensorTrain:
    """Construct a random TT directly in the requested dtype (bypasses SVD/QR cascade).

    Gate 2 tests numerical orthogonality ``< 1e-7``, which is only physically
    achievable when the cores themselves are stored in ``float64`` (see
    NUMERICAL_STABILITY.md §2 — float32 QR followed by float32 storage has
    ~6e-8 relative roundoff per operation, which compounds across cores).
    """
    torch.manual_seed(seed)
    input_dims = [8, 8]
    output_dims = [8, 8]
    ranks = [1, 32, 1]
    cores: list[torch.Tensor] = []
    for k in range(len(input_dims)):
        r_l, p_k, r_r = ranks[k], input_dims[k] * output_dims[k], ranks[k + 1]
        cores.append(torch.randn(r_l, p_k, r_r, dtype=dtype))
    return TensorTrain(cores=cores, input_dims=input_dims, output_dims=output_dims)


def test_gate2_left_orthogonalization_yields_identity() -> None:
    tt = _make_random_tt(seed=123)
    target = tt.num_cores - 1  # left-orthogonalize everything except the last core
    orthogonalize_left_to(tt, target)
    L = merged_left_block(tt, target).to(torch.float64)
    gram = L.T @ L
    I = torch.eye(gram.shape[0], dtype=torch.float64)
    err = float(torch.linalg.norm(gram - I, ord=float("inf")).item())
    assert err < 1.0e-7, f"||L^T L - I||_inf = {err:.3e} (expected < 1e-7)"


def test_gate2_right_orthogonalization_yields_identity() -> None:
    tt = _make_random_tt(seed=321)
    orthogonalize_right_to(tt, 0)
    R = merged_right_block(tt, 0).to(torch.float64)
    gram = R @ R.T
    I = torch.eye(gram.shape[0], dtype=torch.float64)
    err = float(torch.linalg.norm(gram - I, ord=float("inf")).item())
    assert err < 1.0e-7, f"||R R^T - I||_inf = {err:.3e} (expected < 1e-7)"


def test_gate2_left_ortho_preserves_dense_matrix() -> None:
    """Orthogonalization must be gauge-preserving: W = L·G·... must be unchanged."""
    tt = _make_random_tt(seed=7)
    W_before = tt.to_dense().clone()
    orthogonalize_left_to(tt, tt.num_cores - 1)
    W_after = tt.to_dense()
    err = float(torch.linalg.norm(W_before - W_after).item()) / float(
        torch.linalg.norm(W_before).item()
    )
    assert err < 1.0e-4, f"gauge transform changed the represented matrix: rel err {err:.3e}"
