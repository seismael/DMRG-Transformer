"""Headline benchmark — BENCHMARK.md spec at 1024x1024 (Phase A1+B1+B3 deliverable).

Runs the canonical three-way runoff (Adam vs Dense Exact vs TT-DMRG) at the
spec-headline scale: in=out=1024, batch=2048, rank=32, target = sin(X·W)+0.1·η.

* Each method runs with **1 warmup pass** (discarded) plus **3 seeded measurement
  passes**; mean ± std reported.
* Peak GPU memory captured per method via ``torch.cuda.max_memory_allocated``.
* Forward-pass FLOPs reported (analytic estimate; see benchmark.py).

Writes ``bench/HEADLINE.md``.

Requires: matrix-free local solver (Phase A1) — without it, DMRG OOMs at this
scale on a 2 GiB GPU.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmrg_transformer.bench import OptimizationBenchmark  # noqa: E402
from dmrg_transformer.core.device import describe_device, require_cuda  # noqa: E402

# BENCHMARK.md headline configuration (verbatim).
IN_FEATURES = 1024
OUT_FEATURES = 1024
BATCH_SIZE = 2048
RANK = 32
ADAM_ITERS = 500
SWEEPS = 2
SEEDS = 3
WARMUP = 1


def main() -> None:
    device = require_cuda()
    out_dir = ROOT / "bench"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "HEADLINE.md"

    print(f"Running headline benchmark on {describe_device()}")
    print(
        f"  shape={IN_FEATURES}x{OUT_FEATURES}, batch={BATCH_SIZE}, "
        f"rank={RANK}, seeds={SEEDS}, warmup={WARMUP}"
    )

    t_total = time.perf_counter()
    bench = OptimizationBenchmark(
        in_features=IN_FEATURES, out_features=OUT_FEATURES,
        batch_size=BATCH_SIZE, rank=RANK, device=device,
    )

    print("  [1/3] Adam ...")
    adam = bench.run_adam(iterations=ADAM_ITERS, warmup=WARMUP, seeds=SEEDS)
    print(
        f"        MSE={adam.mse:.4e}±{adam.mse_std:.1e}  "
        f"time={adam.time_sec:.3f}±{adam.time_std:.3f}s  peak={adam.peak_mem_gb:.3f} GB"
    )

    print("  [2/3] Dense Exact ...")
    dense = bench.run_dense_exact(warmup=WARMUP, seeds=SEEDS)
    print(
        f"        MSE={dense.mse:.4e}±{dense.mse_std:.1e}  "
        f"time={dense.time_sec:.3f}±{dense.time_std:.3f}s  peak={dense.peak_mem_gb:.3f} GB"
    )

    print("  [3/3] TT-DMRG ...")
    dmrg = bench.run_dmrg(num_sweeps=SWEEPS, warmup=WARMUP, seeds=SEEDS)
    print(
        f"        MSE={dmrg.mse:.4e}±{dmrg.mse_std:.1e}  "
        f"time={dmrg.time_sec:.3f}±{dmrg.time_std:.3f}s  peak={dmrg.peak_mem_gb:.3f} GB"
    )

    elapsed_total = time.perf_counter() - t_total
    compression = bench.total_dense_params / bench.total_tt_params

    lines: list[str] = []
    lines.append("# DMRG-Transformer — Headline Benchmark (BENCHMARK.md spec)\n\n")
    lines.append(f"**Device:** `{describe_device()}`  \n")
    lines.append(
        f"**Configuration:** {IN_FEATURES}×{OUT_FEATURES} layer, "
        f"batch={BATCH_SIZE}, rank={RANK}, "
        f"target = `sin(X·W) + 0.1·η`  \n"
    )
    lines.append(
        f"**Methodology:** {WARMUP} warmup pass(es) discarded; "
        f"{SEEDS} seeded measurement passes; mean ± population std reported. "
        "All wall-times include `torch.cuda.synchronize()`. Peak memory via "
        "`torch.cuda.max_memory_allocated()` (reset after warmup).  \n"
    )
    lines.append(f"**Total wall-time:** {elapsed_total:.1f}s  \n\n")

    lines.append(
        "| Method | MSE (mean ± std) | Time (s, mean ± std) | "
        "Peak GPU mem | Params | FLOPs/call |\n"
        "| :----- | ---: | ---: | ---: | ---: | ---: |\n"
    )
    for res in (adam, dense, dmrg):
        lines.append(
            f"| {res.name} | "
            f"{res.mse:.4e} ± {res.mse_std:.1e} | "
            f"{res.time_sec:.3f} ± {res.time_std:.3f} | "
            f"{res.peak_mem_gb:.3f} GB | "
            f"{res.parameters:,} | "
            f"{res.flops:.2e} |\n"
        )

    lines.append(
        f"\n**Compression (dense → TT):** {compression:.1f}× "
        f"({bench.total_dense_params:,} → {bench.total_tt_params:,} parameters)\n\n"
    )

    lines.append("## Interpretation\n\n")
    lines.append(
        "* **Headline target** (`sin(X·W)+noise`) is a full-rank target — DMRG "
        "is constrained to a TT-rank-`r` manifold and will not match the dense "
        "exact MSE in general; see `bench/PARETO.md` for the rank/MSE trade-off.\n"
        "* **Adam** uses lr=0.01, 500 iters; for fairness see "
        "`docs/BENCHMARK.md` reconciliation notes.\n"
        "* **Dense Exact** is `torch.linalg.lstsq` (cuSOLVER) — the unconstrained "
        "least-squares optimum.\n"
        "* **TT-DMRG** runs 2 bidirectional sweeps from a TT initialised by "
        "TT-SVD of a random matrix; no learning rate, no iteration budget. "
        "Its MSE matches the dense optimum *only* on TT-rank-bounded targets "
        "(see `bench/GATE3_PROOF.md`).\n"
    )
    lines.append(
        "\n## Honest assessment vs. BENCHMARK.md spec language\n\n"
        "The original spec headline (\"DMRG matches dense MSE in ~0.05s with "
        "15.6× compression\") holds **only when the data lives on a TT manifold "
        "of the chosen rank**. On the spec's own non-TT target "
        "(`sin(X·W)+noise`), DMRG produces the rank-`r` Pareto-optimal point — "
        "lower MSE than its parameter budget allows for a dense baseline of "
        "the same parameter count, but generally higher MSE than an "
        "unconstrained dense lstsq fit. See `docs/BENCHMARK.md` reconciliation.\n"
    )

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
