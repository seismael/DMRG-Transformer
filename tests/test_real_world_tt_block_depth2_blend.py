"""Unit tests for the depth-2 TTBlock blend harness helpers."""
from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_real_world_tt_block_depth2_blend as depth2_script  # noqa: E402


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

    summary = depth2_script._aggregate_attn_reports(reports)

    assert summary["attn_steps"] == 2.0
    assert summary["attn_reject_rate"] == 0.5
    assert summary["scores_target_abs_max"] == 9.5
    assert summary["scores_delta_abs_max"] == 4.25
    assert summary["mse_ratio_max"] == 2.0