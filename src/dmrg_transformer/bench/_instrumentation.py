"""Shared GPU/wall/inference instrumentation for the Tier-1/2/3 runners.

Pulled out of `scripts/train_real_world_tt_block_classifier.py` so all three
real-task harnesses (MLP, 1-block, depth-2 stack) report the same columns
in `bench/COVERAGE_MATRIX.md`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch


def reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def read_peak_mem_mib() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


@torch.no_grad()
def measure_inference_latency(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    X: torch.Tensor,
    *,
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, float]:
    """Median forward latency in milliseconds, with CUDA sync per call."""
    cuda = torch.cuda.is_available()
    for _ in range(warmup):
        _ = forward_fn(X)
        if cuda:
            torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = forward_fn(X)
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    median = samples[len(samples) // 2]
    return {
        "median_ms": median,
        "p10_ms": samples[max(0, len(samples) // 10)],
        "p90_ms": samples[min(len(samples) - 1, len(samples) * 9 // 10)],
        "throughput_examples_per_s": X.shape[0] * 1000.0 / median,
    }


def iso_time_lookup(history: dict, target_wall: float) -> tuple[float, float]:
    """Return (test_acc, wall) for the last sample with wall <= ``target_wall``.

    Combines per-step samples (``step_wall`` / ``step_test_acc``) with epoch-end
    samples (``wall`` / ``test_acc``).
    """
    walls = list(history.get("step_wall", []))
    accs = list(history.get("step_test_acc", []))
    combined = list(zip(walls, accs, strict=False))
    for w, a in zip(history["wall"], history["test_acc"], strict=False):
        combined.append((w, a))
    combined.sort()
    candidates = [(w, a) for (w, a) in combined if w <= target_wall]
    if not candidates:
        return (combined[0][1], combined[0][0]) if combined else (0.0, 0.0)
    return (candidates[-1][1], candidates[-1][0])


def dump_coverage_sidecar(
    tier: str,
    payload: dict,
    *,
    out_dir: Path | None = None,
) -> Path:
    """Write ``bench/_coverage/<tier>.json`` for the matrix aggregator."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[3] / "bench" / "_coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tier}.json"
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path
