"""Adaptive rank scheduler tests (plan §C5)."""
from __future__ import annotations

import torch

from dmrg_transformer.core.svd import adaptive_rank


def test_adaptive_rank_keeps_dominant_modes() -> None:
    # Spectrum with a sharp drop at index 4: 4 dominant + 12 noise modes.
    S = torch.tensor(
        [10.0, 9.0, 8.0, 7.0] + [1.0e-3] * 12, dtype=torch.float64
    )
    r = adaptive_rank(S, rel_threshold=1.0e-4, min_rank=1, max_rank=16)
    assert r == 4, f"expected dominant-4 truncation, got rank {r}"


def test_adaptive_rank_returns_full_when_threshold_strict() -> None:
    S = torch.linspace(1.0, 0.5, 8, dtype=torch.float64)
    r = adaptive_rank(S, rel_threshold=0.0, min_rank=1, max_rank=8)
    assert r == 8


def test_adaptive_rank_respects_min() -> None:
    S = torch.tensor([1.0, 1.0e-12], dtype=torch.float64)
    r = adaptive_rank(S, rel_threshold=0.5, min_rank=2, max_rank=2)
    assert r == 2


def test_adaptive_rank_respects_max() -> None:
    S = torch.linspace(1.0, 0.99, 50, dtype=torch.float64)
    # Strict threshold would want full rank; max_rank caps it.
    r = adaptive_rank(S, rel_threshold=1.0e-12, min_rank=1, max_rank=4)
    assert r == 4
