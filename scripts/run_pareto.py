"""Rank/MSE Pareto curve for TT-DMRG vs dense baseline (Phase B4 deliverable).

Sweeps DMRG rank ∈ {2, 4, 8, 16, 32, 64} at fixed layer size to expose the
honest rank-vs-MSE trade-off on a non-TT target (`sin(X·W) + noise`). Dense
exact solver is reported as the unconstrained reference.

Writes ``bench/PARETO.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmrg_transformer.bench import OptimizationBenchmark  # noqa: E402
from dmrg_transformer.core.device import describe_device, require_cuda  # noqa: E402

N = 256
BATCH = 1024
RANKS = [2, 4, 8, 16, 32, 64]
SEEDS = 2
WARMUP = 1
ADAM_ITERS = 500


def main() -> None:
    device = require_cuda()
    print(f"Pareto sweep on {describe_device()}: N={N}, batch={BATCH}, ranks={RANKS}")

    # Dense baseline (rank-independent).
    bench0 = OptimizationBenchmark(
        in_features=N, out_features=N, batch_size=BATCH, rank=2, device=device,
    )
    dense = bench0.run_dense_exact(warmup=WARMUP, seeds=SEEDS)
    adam = bench0.run_adam(iterations=ADAM_ITERS, warmup=WARMUP, seeds=SEEDS)

    rows: list[tuple[int, float, float, float, int]] = []
    for r in RANKS:
        bench = OptimizationBenchmark(
            in_features=N, out_features=N, batch_size=BATCH, rank=r, device=device,
        )
        dmrg = bench.run_dmrg(num_sweeps=2, warmup=WARMUP, seeds=SEEDS)
        compression = bench.total_dense_params / bench.total_tt_params
        rows.append((r, dmrg.mse, dmrg.time_sec, compression, bench.total_tt_params))
        print(
            f"  rank={r:>3}: MSE={dmrg.mse:.4e}  "
            f"time={dmrg.time_sec:.3f}s  compression={compression:.1f}x"
        )

    out_path = ROOT / "bench" / "PARETO.md"
    lines: list[str] = []
    lines.append("# DMRG-Transformer — Rank/MSE Pareto Curve\n\n")
    lines.append(f"**Device:** `{describe_device()}`  \n")
    lines.append(
        f"**Configuration:** {N}×{N} layer, batch={BATCH}, "
        f"target = `sin(X·W) + 0.1·η`, {SEEDS}-seed mean.  \n\n"
    )
    lines.append(
        "Reference baselines (rank-independent):\n\n"
        f"* **Dense Exact (lstsq):** MSE = `{dense.mse:.4e}`  "
        f"(time = {dense.time_sec:.3f}s, {dense.parameters:,} params)\n"
        f"* **Adam ({ADAM_ITERS} iters, lr=0.01):** MSE = `{adam.mse:.4e}`  "
        f"(time = {adam.time_sec:.3f}s, {adam.parameters:,} params)\n\n"
    )
    lines.append("## TT-DMRG rank sweep\n\n")
    lines.append(
        "| Rank | TT params | Compression | DMRG MSE | DMRG time (s) | "
        "MSE / Dense MSE |\n"
        "| ---: | --------: | ----------: | -------: | ------------: | --------------: |\n"
    )
    for r, mse, dt, comp, params in rows:
        ratio = mse / dense.mse if dense.mse > 0 else float("inf")
        lines.append(
            f"| {r} | {params:,} | {comp:.1f}× | "
            f"{mse:.4e} | {dt:.3f} | {ratio:.2f}× |\n"
        )

    lines.append(
        "\n## Reading the curve\n\n"
        "The MSE column shows how close DMRG gets to the dense optimum at each "
        "rank budget. The last column is the multiplicative gap. As `r` grows "
        "the rank-constrained DMRG converges to the dense exact solution; the "
        "compression column shows the parameter cost. This is the honest "
        "Pareto trade-off for a *non-TT-rank* target — when the target lives "
        "on a TT manifold of rank `r₀` (Gate 3 setup), DMRG matches the dense "
        "MSE at any `r ≥ r₀` (see `bench/GATE3_PROOF.md`).\n"
    )

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
