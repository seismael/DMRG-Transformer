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

import torch

from dmrg_transformer.core.device import describe_device, require_cuda
from dmrg_transformer.optim import DMRGOptimizer
from dmrg_transformer.tt import TensorTrain


def _factor_for_tt(n: int) -> list[int]:
    """Choose a TT factorization adapted to the size of ``n``.

    For small ``n`` (<256) a 2-core TT keeps each physical dim ~ sqrt(n),
    which is the most expressive rank schedule. For large ``n`` we drop to a
    4-core TT so per-core normal-equation matrices remain tractable
    (size r²·p² ≲ 10⁴).
    """
    if n < 256:
        return _factor_pair(n)
    # Greedy 4-core: pick the largest divisor <= n^{1/4} three times.
    target = int(round(n ** 0.25))
    dims: list[int] = []
    rem = n
    for _ in range(3):
        d = target
        while d > 1 and rem % d != 0:
            d -= 1
        dims.append(max(d, 1))
        rem //= dims[-1]
    dims.append(rem)
    return dims


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
        *,
        device: torch.device | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.rank = rank
        self.device = device or require_cuda()

        # Adaptive TT: 2 cores for small n (expressive), 4 cores for large n
        # (tractable normal-equation matrices at 1024 scale).
        self.input_dims = _factor_for_tt(in_features)
        self.output_dims = _factor_for_tt(out_features)

        self.total_dense_params = in_features * out_features
        # Compute actual TT param count from a probe decomposition.
        probe, _ = TensorTrain.from_dense(
            torch.zeros(in_features, out_features, dtype=torch.float64, device=self.device),
            self.input_dims, self.output_dims, max_rank=rank,
        )
        self.total_tt_params = sum(c.numel() for c in probe.cores)

        log = logging.getLogger(__name__)
        log.info("=" * 60)
        log.info("INITIALIZING BENCHMARK ENVIRONMENT")
        log.info("=" * 60)
        log.info(describe_device())
        log.info(f"Layer Dimensions : {in_features} -> {out_features}")
        log.info(f"TT factorization : {self.input_dims} -> {self.output_dims}")
        log.info(f"Dense Parameters : {self.total_dense_params:,}")
        log.info(f"TT-DMRG Parameters: {self.total_tt_params:,} (Rank={rank})")
        log.info(f"Compression Ratio: {self.total_dense_params / self.total_tt_params:.1f}x")

        torch.manual_seed(42)
        self.X = torch.randn(
            batch_size, in_features, dtype=torch.float64, device=self.device,
        )
        true_W = torch.randn(
            in_features, out_features, dtype=torch.float64, device=self.device,
        ) / (in_features ** 0.5)
        noise = torch.randn(
            batch_size, out_features, dtype=torch.float64, device=self.device,
        ) * 0.1
        self.Y = torch.sin(self.X @ true_W) + noise

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # -- 1. Adam (on GPU) ------------------------------------------------------

    def run_adam(self, iterations: int = 500, lr: float = 0.01) -> BenchmarkResult:
        torch.manual_seed(0)
        W = torch.randn(
            self.in_features, self.out_features,
            dtype=torch.float64, device=self.device,
        ) * 0.01
        m = torch.zeros_like(W)
        v = torch.zeros_like(W)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        self._sync()
        t0 = time.perf_counter()
        for t in range(1, iterations + 1):
            pred = self.X @ W
            error = pred - self.Y
            grad = self.X.T @ error / self.batch_size
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            W.sub_(lr * m_hat / (v_hat.sqrt() + eps))
        self._sync()
        elapsed = time.perf_counter() - t0
        mse = float(torch.mean((self.X @ W - self.Y) ** 2).item())
        return BenchmarkResult("Gradient Descent (Adam)", mse, elapsed, self.total_dense_params)

    # -- 2. Dense Exact Solver (on GPU) ----------------------------------------

    def run_dense_exact(self) -> BenchmarkResult:
        self._sync()
        t0 = time.perf_counter()
        W = torch.linalg.lstsq(self.X, self.Y).solution
        self._sync()
        elapsed = time.perf_counter() - t0
        mse = float(torch.mean((self.X @ W - self.Y) ** 2).item())
        return BenchmarkResult(
            "Dense Exact Solver (O(N^3))", mse, elapsed, self.total_dense_params
        )

    # -- 3. TT-DMRG (on GPU) ---------------------------------------------------

    def run_dmrg(self, num_sweeps: int = 2) -> BenchmarkResult:
        torch.manual_seed(0)
        W_init = torch.randn(
            self.in_features, self.out_features,
            dtype=torch.float64, device=self.device,
        ) * 0.01

        self._sync()
        t0 = time.perf_counter()
        tt, _ = TensorTrain.from_dense(
            W_init, self.input_dims, self.output_dims, max_rank=self.rank,
        )
        opt = DMRGOptimizer(max_rank=self.rank, lam=1.0e-6, clamp_target=False)
        for _ in range(num_sweeps):
            opt.sweep(tt, self.X, self.Y)
        self._sync()
        elapsed = time.perf_counter() - t0

        Y_pred = self.X @ tt.to_dense()
        mse = float(torch.mean((Y_pred - self.Y) ** 2).item())
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
