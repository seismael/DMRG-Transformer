"""Unit sanity check: J @ vec(G_k) must equal the TT forward output."""
from __future__ import annotations

import torch

from dmrg_transformer.optim.local_solver import _build_jacobian
from dmrg_transformer.tt import TensorTrain


def test_jacobian_reconstructs_forward() -> None:
    torch.manual_seed(0)
    W = torch.randn(64, 64, dtype=torch.float64)
    tt, _ = TensorTrain.from_dense(W, [8, 8], [8, 8], max_rank=16)
    X = torch.randn(7, 64, dtype=torch.float64)
    Y_tt = tt.contract_forward(X)
    for k in range(tt.num_cores):
        J = _build_jacobian(tt, X, k)  # [batch, M, P]
        vec = tt.get_core(k).reshape(-1)
        Y_from_J = torch.einsum("bmp,p->bm", J, vec)
        err = float(torch.linalg.norm(Y_from_J - Y_tt).item()) / float(
            torch.linalg.norm(Y_tt).item()
        )
        assert err < 1.0e-8, f"core {k}: jac*vec != forward (rel err {err:.3e})"
