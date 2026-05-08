"""Unit tests for decision-boundary-aware target propagation (Option E).

Validates the three new TargetPropagator methods:
- ``compute_logit_target`` — logit-space margin maximisation
- ``compute_pool_constrained_target`` — only constrains pool mean
- ``compute_margin_aware_block_target`` — combined convenience method
"""
from __future__ import annotations

import torch

from dmrg_transformer.propagation.target_propagator import TargetPropagator


def test_logit_target_preserves_class_decision_dimensions() -> None:
    """Logit-space target should move the correct class logit upward."""
    torch.manual_seed(0)
    B, D, C = 8, 4, 3
    pooled = torch.randn(B, D, dtype=torch.float64) * 0.5
    W_head = torch.randn(D, C, dtype=torch.float64) * 0.3
    b_head = torch.randn(C, dtype=torch.float64) * 0.1
    Y = torch.eye(C, dtype=torch.float64)[torch.randint(0, C, (B,))]

    prop = TargetPropagator(lam=1e-4)
    pooled_target = prop.compute_logit_target(
        pooled, Y, W_head, b_head, margin_scale=1.0,
    )

    # Re-project to logit space and check that the correct-class logit
    # is larger than it was before.
    logits_before = pooled @ W_head + b_head
    logits_after = pooled_target @ W_head + b_head
    correct_before = (logits_before * Y).sum(dim=-1)
    correct_after = (logits_after * Y).sum(dim=-1)
    # The target should increase correct-class confidence.
    assert (correct_after > correct_before - 1e-8).all(), (
        "logit target decreased correct-class logit"
    )


def test_logit_target_shape_and_finite() -> None:
    """Logit target must have correct shape and no NaN/inf."""
    torch.manual_seed(1)
    B, D, C = 16, 6, 4
    pooled = torch.randn(B, D, dtype=torch.float64)
    W_head = torch.randn(D, C, dtype=torch.float64) * 0.5
    b_head = torch.zeros(C, dtype=torch.float64)
    Y = torch.eye(C, dtype=torch.float64)[torch.randint(0, C, (B,))]

    prop = TargetPropagator(lam=1e-4)
    target = prop.compute_logit_target(pooled, Y, W_head, b_head, margin_scale=2.0)

    assert target.shape == (B, D)
    assert torch.isfinite(target).all()


def test_pool_constrained_target_preserves_token_deviation() -> None:
    """Only the pool mean shifts; per-token deviations from mean stay intact."""
    torch.manual_seed(2)
    B, L, D = 4, 6, 8
    acts = torch.randn(B, L, D, dtype=torch.float64)
    pooled_target = torch.randn(B, D, dtype=torch.float64)

    prop = TargetPropagator(lam=1e-4)
    result = prop.compute_pool_constrained_target(acts, pooled_target, target_blend=0.5)

    # Check: the deviation of each token from its sequence mean must be unchanged.
    mean_before = acts.mean(dim=1, keepdim=True)
    dev_before = acts - mean_before
    mean_after = result.mean(dim=1, keepdim=True)
    dev_after = result - mean_after
    assert torch.allclose(dev_before, dev_after, atol=1e-10), (
        "pool-constrained target altered per-token deviations"
    )


def test_pool_constrained_target_mean_shifts() -> None:
    """With target_blend=1.0, the pool mean must exactly match pooled_target."""
    torch.manual_seed(3)
    B, L, D = 4, 6, 8
    acts = torch.randn(B, L, D, dtype=torch.float64)
    pooled_target = torch.randn(B, D, dtype=torch.float64)

    prop = TargetPropagator(lam=1e-4)
    result = prop.compute_pool_constrained_target(acts, pooled_target, target_blend=1.0)

    mean_after = result.mean(dim=1)
    assert torch.allclose(mean_after, pooled_target, atol=1e-10)


def test_margin_aware_block_target_produces_valid_shape() -> None:
    """Combined method returns correct per-token shape."""
    torch.manual_seed(4)
    B, L, D, C = 4, 6, 12, 5
    block_out = torch.randn(B, L, D, dtype=torch.float64)
    W_head = torch.randn(D, C, dtype=torch.float64)
    b_head = torch.zeros(C, dtype=torch.float64)
    Y = torch.eye(C, dtype=torch.float64)[torch.randint(0, C, (B,))]

    prop = TargetPropagator(lam=1e-4)
    result = prop.compute_margin_aware_block_target(
        block_out, Y, W_head, b_head, target_blend=0.5, margin_scale=1.0,
    )

    assert result.shape == (B, L, D)
    assert torch.isfinite(result).all()


def test_margin_aware_blend_zero_is_noop() -> None:
    """With target_blend=0.0, the target equals the current block output."""
    torch.manual_seed(5)
    B, L, D, C = 4, 5, 8, 3
    block_out = torch.randn(B, L, D, dtype=torch.float64)
    W_head = torch.randn(D, C, dtype=torch.float64)
    b_head = torch.zeros(C, dtype=torch.float64)
    Y = torch.eye(C, dtype=torch.float64)[torch.randint(0, C, (B,))]

    prop = TargetPropagator(lam=1e-4)
    result = prop.compute_margin_aware_block_target(
        block_out, Y, W_head, b_head, target_blend=0.0, margin_scale=1.0,
    )

    # target_blend=0 means the pool-constrained step is a no-op → result = block_out
    assert torch.allclose(result, block_out, atol=1e-10)
