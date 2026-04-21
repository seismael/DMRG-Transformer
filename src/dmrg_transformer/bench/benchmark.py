"""Reproduction of ``docs/BENCHMARK.md`` — the three-way optimizer runoff.

Invocation::

    python -m dmrg_transformer.bench

Compares:
    1. Adam (iterative approximation, 500 steps).
    2. Dense Exact Solver (pseudo-inverse, O(N^3)).
    3. TT-DMRG (this project) with a single bidirectional sweep.

Output format mirrors BENCHMARK.md exactly. The DMRG path uses the
:mod:`dmrg_transformer.optim` exact solver; no gradients are used anywhere.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from dmrg_transformer.optim import DMRGOptimizer
from dmrg_transformer.tt import TensorTrain


def _factor_pair(n: int) -> list[int]:
    """Return [a, b] with a*b == n and a as close to sqrt(n) as possible."""
    a = int(round(n**0.5))
    while a > 1 and n % a != 0:
        a -= 1
    if a < 1:
        a = 1
    return [a, n // a]


@dataclass
class BenchmarkResult:
    name: str
    mse: float
    time_sec: float
    parameters: int


class OptimizationBenchmark:
    """Faithful reproduction of BENCHMARK.md with identical dimensions."""

    def __init__(
        self,
        in_features: int = 1024,
        out_features: int = 1024,
        batch_size: int = 2048,
        rank: int = 32,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.rank = rank

        self.total_dense_params = in_features * out_features
        self.total_tt_params = (in_features * rank) + (rank * out_features)

        log = logging.getLogger(__name__)
        log.info("=" * 60)
        log.info("INITIALIZING BENCHMARK ENVIRONMENT")
        log.info("=" * 60)
        log.info(f"Layer Dimensions : {in_features} -> {out_features}")
        log.info(f"Dense Parameters : {self.total_dense_params:,}")
        log.info(f"TT-DMRG Parameters: {self.total_tt_params:,} (Rank={rank})")
        log.info(f"Compression Ratio: {self.total_dense_params / self.total_tt_params:.1f}x")

        np.random.seed(42)
        self.X_np = np.random.randn(batch_size, in_features).astype(np.float64)
        true_W = np.random.randn(in_features, out_features).astype(np.float64) / np.sqrt(
            in_features
        )
        noise = np.random.randn(batch_size, out_features).astype(np.float64) * 0.1
        self.Y_np = np.sin(self.X_np @ true_W) + noise

    # -- 1. Adam ---------------------------------------------------------------

    def run_adam(self, iterations: int = 500, lr: float = 0.01) -> BenchmarkResult:
        rng = np.random.default_rng(0)
        W = rng.standard_normal((self.in_features, self.out_features)) * 0.01
        m, v = np.zeros_like(W), np.zeros_like(W)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        t0 = time.perf_counter()
        for t in range(1, iterations + 1):
            pred = self.X_np @ W
            error = pred - self.Y_np
            grad = self.X_np.T @ error / self.batch_size
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            W -= lr * m_hat / (np.sqrt(v_hat) + eps)
        elapsed = time.perf_counter() - t0
        mse = float(np.mean((self.X_np @ W - self.Y_np) ** 2))
        return BenchmarkResult("Gradient Descent (Adam)", mse, elapsed, self.total_dense_params)

    # -- 2. Dense Exact Solver -------------------------------------------------

    def run_dense_exact(self) -> BenchmarkResult:
        t0 = time.perf_counter()
        X_pinv = np.linalg.pinv(self.X_np)
        W = X_pinv @ self.Y_np
        elapsed = time.perf_counter() - t0
        mse = float(np.mean((self.X_np @ W - self.Y_np) ** 2))
        return BenchmarkResult(
            "Dense Exact Solver (O(N^3))", mse, elapsed, self.total_dense_params
        )

    # -- 3. TT-DMRG ------------------------------------------------------------

    def run_dmrg(self, num_sweeps: int = 2) -> BenchmarkResult:
        input_dims = _factor_pair(self.in_features)
        output_dims = _factor_pair(self.out_features)

        X = torch.from_numpy(self.X_np)
        Y = torch.from_numpy(self.Y_np)

        t0 = time.perf_counter()
        torch.manual_seed(0)
        W_init = torch.randn(self.in_features, self.out_features, dtype=torch.float64) * 0.01
        tt, _ = TensorTrain.from_dense(W_init, input_dims, output_dims, max_rank=self.rank)
        opt = DMRGOptimizer(max_rank=self.rank, lam=1.0e-6, clamp_target=False)
        for _ in range(num_sweeps):
            opt.sweep(tt, X, Y)
        elapsed = time.perf_counter() - t0

        Y_pred = X @ tt.to_dense()
        mse = float(torch.mean((Y_pred - Y) ** 2).item())
        return BenchmarkResult("TT-DMRG Exact Sweep", mse, elapsed, self.total_tt_params)


def execute(
    in_features: int = 1024,
    out_features: int = 1024,
    batch_size: int = 2048,
    rank: int = 32,
) -> list[BenchmarkResult]:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    log = logging.getLogger(__name__)

    bench = OptimizationBenchmark(in_features, out_features, batch_size, rank)
    results = [
        bench.run_adam(iterations=500),
        bench.run_dense_exact(),
        bench.run_dmrg(num_sweeps=2),
    ]

    log.info("\n" + "=" * 60)
    log.info(f"{'Algorithm':<30} | {'MSE Error':<12} | {'Time (sec)':<10}")
    log.info("-" * 60)
    for res in results:
        log.info(f"{res.name:<30} | {res.mse:<12.6f} | {res.time_sec:<10.4f}")
    log.info("=" * 60)
    return results


if __name__ == "__main__":
    execute()
