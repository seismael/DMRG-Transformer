"""Regression guard for the real-world TTBlock benchmark script's console output.

Windows PowerShell commonly exposes ``cp1252`` as the stdout encoding. The
script must therefore keep its progress output encodable there even though the
generated markdown report can still use richer Unicode typography.
"""
from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_real_world_tt_block_classifier as benchmark_script  # noqa: E402


def test_console_safe_rewrites_non_cp1252_glyphs() -> None:
    text = "blk_mse 1.000e-02→9.000e-03 — TT-DMRG ↔ Dense"
    safe = benchmark_script._console_safe(text)
    encoded = safe.encode("cp1252")
    assert encoded.decode("cp1252") == safe
    assert "->" in safe
    assert "<->" in safe
    assert "→" not in safe
    assert "↔" not in safe
    assert "—" not in safe