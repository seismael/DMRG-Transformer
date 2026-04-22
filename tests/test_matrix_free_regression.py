"""Regression test: matrix-free local solver must fit BENCHMARK.md headline scale.

This is the gate that prevents future changes from re-introducing the O(r⁴·p²)
JᵀJ materialisation that previously blocked 1024×1024 on a 2 GiB GPU.
"""
from __future__ import annotations

import torch

from dmrg_transformer.core.device import cuda_available, require_cuda
from dmrg_transformer.optim.sweep import DMRGOptimizer
from dmrg_transformer.tt import TensorTrain


def test_dmrg_sweep_at_512x512_under_memory_budget() -> None:
    """A single sweep at N=512, batch=1024, rank=16 must use < 1 GiB peak GPU."""
    dev = require_cuda()
    if not cuda_available():
        # CPU fallback: skip the GPU-memory assertion but still run.
        torch.manual_seed(0)
    torch.manual_seed(0)
    N = 512
    batch = 1024
    rank = 16
    W = torch.randn(N, N, dtype=torch.float64, device=dev)
    X = torch.randn(batch, N, dtype=torch.float64, device=dev)
    Y = X @ W
    tt, _ = TensorTrain.from_dense(W, [16, 32], [32, 16], max_rank=rank)
    opt = DMRGOptimizer(max_rank=rank)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    rep = opt.sweep(tt, X, Y)
    assert rep.final_mse <= rep.initial_mse, "DMRG sweep must not increase MSE"
    if dev.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(dev) / 1e9
        assert peak_gb < 1.0, (
            f"matrix-free solver regression: peak GPU memory {peak_gb:.3f} GB "
            "exceeds 1 GiB budget at N=512 — the block-diagonal JᵀJ collapse "
            "may have regressed (see local_solver._build_block_normal_equations)."
        )
