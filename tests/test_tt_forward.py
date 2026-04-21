"""Forward-pass consistency: TT.contract_forward must match dense X @ W."""
from __future__ import annotations

import torch

from dmrg_transformer.tt import TensorTrain


def test_forward_matches_dense() -> None:
    torch.manual_seed(3)
    W = torch.randn(64, 48, dtype=torch.float32)
    tt, _ = TensorTrain.from_dense(W, [8, 8], [6, 8], max_rank=64)

    X = torch.randn(5, 64, dtype=torch.float32)
    y_tt = tt.contract_forward(X)
    y_dense = X @ tt.to_dense()
    assert y_tt.shape == (5, 48)
    err = float(torch.linalg.norm(y_tt - y_dense).item()) / float(
        torch.linalg.norm(y_dense).item()
    )
    assert err < 1.0e-4, f"forward contraction mismatch rel err {err:.3e}"


def test_forward_matches_original_dense() -> None:
    """Forward through the TT must approximate X @ W_original up to truncation error."""
    torch.manual_seed(4)
    W = torch.randn(64, 64, dtype=torch.float32)
    tt, _ = TensorTrain.from_dense(W, [8, 8], [8, 8], max_rank=64)  # full rank
    X = torch.randn(3, 64)
    y_tt = tt.contract_forward(X)
    y_exact = X @ W
    err = float(torch.linalg.norm(y_tt - y_exact).item()) / float(
        torch.linalg.norm(y_exact).item()
    )
    assert err < 1.0e-4, f"forward rel err {err:.3e}"
