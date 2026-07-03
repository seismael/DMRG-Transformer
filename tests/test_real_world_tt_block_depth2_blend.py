"""Unit tests for the depth-2 TTBlock blend harness helpers.

The original ``train_real_world_tt_block_depth2_blend`` script was removed.
The helper functions it exposed are inlined here so the tests remain valid.
"""
from __future__ import annotations

from typing import Any


def _aggregate_attn_reports(
    reports: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate attention diagnostics across DMRG steps (copied from the
    removed ``train_real_world_tt_block_depth2_blend`` script)."""
    n = len(reports)
    rejects = sum(1 for r in reports if not r["attn"]["accepted"])
    scores_max = max(
        r["attn"]["diagnostics"]["scores_target_abs_max"] for r in reports
    )
    delta_max = max(
        r["attn"]["diagnostics"]["scores_delta_abs_max"] for r in reports
    )
    mse_ratios = [
        r["attn"]["diagnostics"]["mse_after_attempt"]
        / max(r["attn"]["diagnostics"]["mse_before"], 1e-30)
        for r in reports
    ]
    return {
        "attn_steps": float(n),
        "attn_reject_rate": rejects / max(n, 1),
        "scores_target_abs_max": scores_max,
        "scores_delta_abs_max": delta_max,
        "mse_ratio_max": max(mse_ratios),
    }


def test_aggregate_attn_reports_tracks_rejects_and_maxima() -> None:
    reports = [
        {
            "attn": {
                "accepted": True,
                "diagnostics": {
                    "scores_target_abs_max": 7.0,
                    "scores_delta_abs_max": 3.0,
                    "mse_before": 2.0,
                    "mse_after_attempt": 1.0,
                },
            },
        },
        {
            "attn": {
                "accepted": False,
                "diagnostics": {
                    "scores_target_abs_max": 9.5,
                    "scores_delta_abs_max": 4.25,
                    "mse_before": 1.5,
                    "mse_after_attempt": 3.0,
                },
            },
        },
    ]

    summary = _aggregate_attn_reports(reports)

    assert summary["attn_steps"] == 2.0
    assert summary["attn_reject_rate"] == 0.5
    assert summary["scores_target_abs_max"] == 9.5
    assert summary["scores_delta_abs_max"] == 4.25
    assert summary["mse_ratio_max"] == 2.0
