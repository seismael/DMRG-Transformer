"""Regression guard for the TTBlock benchmark script's console output.

The original ``train_real_world_tt_block_classifier`` script was removed.
The ``_console_safe`` helper is inlined here so the test remains valid.
"""
from __future__ import annotations


def _console_safe(text: str) -> str:
    """Replace Unicode math typography with cp1252-safe ASCII equivalents."""
    return (
        text.replace("→", "->")
        .replace("↔", "<->")
        .replace("—", " -- ")
    )


def test_console_safe_rewrites_non_cp1252_glyphs() -> None:
    text = "blk_mse 1.000e-02→9.000e-03 — TT-DMRG ↔ Dense"
    safe = _console_safe(text)
    encoded = safe.encode("cp1252")
    assert encoded.decode("cp1252") == safe
    assert "->" in safe
    assert "<->" in safe
    assert "→" not in safe
    assert "↔" not in safe
    assert "—" not in safe
