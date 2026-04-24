"""Phase 1.3 — TTLinearAttentionBlock.dmrg_step convergence tests.

Two complementary tests:

1. ``test_dmrg_step_descends_global_mse`` — full random teacher/student.
   The architecture has enough FFN slack to absorb most of the residual on
   its own pass, so V acceptance is *not* the right correctness criterion.
   What we require is **monotonic global descent** plus a meaningful overall
   reduction by step 5.

2. ``test_v_update_accepts_when_only_v_differs`` — focused V-only test.
   Teacher and student are constructed so that all components (Q, K, W_out,
   FFN, LN) are identical and only V differs. Now FFN cannot mask the V
   error; the closed-form V solver must fire and the V update must accept.
"""
from __future__ import annotations

import pytest
import torch

from dmrg_transformer.core.device import require_cuda
from dmrg_transformer.nn.tt_linear_attention_block import TTLinearAttentionBlock


@pytest.fixture(scope="module")
def device() -> torch.device:
    return require_cuda()


def _build_block(seed: int) -> TTLinearAttentionBlock:
    torch.manual_seed(seed)
    return TTLinearAttentionBlock(
        embed_dim=16, num_heads=2, hidden_dim=32,
        embed_dims=[4, 4], hidden_dims=[4, 8], rank=4,
        dtype=torch.float64,
    )


def test_dmrg_step_descends_global_mse(device: torch.device) -> None:
    torch.manual_seed(7)
    X = torch.randn(64, 8, 16, dtype=torch.float64, device=device)
    teacher = _build_block(seed=11)
    student = _build_block(seed=23)
    Y_target = teacher.forward(X)

    n_steps = 5
    mses: list[float] = []
    for _ in range(n_steps):
        diag = student.dmrg_step(X, Y_target, lam=1.0e-5)
        mses.append(float(diag["global_mse_after"]))                      # type: ignore[arg-type]

    print(f"\n[lin-attn block, full random] MSE traj = {mses}")
    for i in range(min(4, n_steps - 1)):
        assert mses[i + 1] <= mses[i] + 1.0e-12, (
            f"non-monotonic at step {i + 1}: {mses[i]} -> {mses[i + 1]}"
        )
    assert mses[-1] < 0.7 * mses[0], (
        f"insufficient descent: {mses[0]} -> {mses[-1]}"
    )


def test_v_update_accepts_when_only_v_differs(device: torch.device) -> None:
    """When the teacher only differs in V, V update must fire and accept."""
    torch.manual_seed(7)
    X = torch.randn(64, 8, 16, dtype=torch.float64, device=device)

    teacher = _build_block(seed=11)
    student = _build_block(seed=11)  # Identical init...
    # ...then perturb only V.
    torch.manual_seed(99)
    for k in range(student.attn.W_V._num_cores):
        core = getattr(student.attn.W_V, f"_core_{k}")
        getattr(student.attn.W_V, f"_core_{k}").copy_(core + 0.1 * torch.randn_like(core))

    Y_target = teacher.forward(X)
    mse0 = float(torch.mean((student.forward(X) - Y_target) ** 2).item())

    n_steps = 3
    accepts: list[bool] = []
    mses: list[float] = []
    for _ in range(n_steps):
        diag = student.dmrg_step(X, Y_target, lam=1.0e-5)
        accepts.append(bool(diag["attn"]["diagnostics"]["v_accepted"]))   # type: ignore[index]
        mses.append(float(diag["global_mse_after"]))                      # type: ignore[arg-type]

    print(f"\n[lin-attn block, V-only differ] init MSE={mse0:.6e} traj={mses}")
    print(f"  accepts: {accepts}")
    assert any(accepts), (
        f"V update never accepted when only V differs (accepts={accepts})"
    )
    assert mses[-1] < 0.5 * mse0, (
        f"V-only test: insufficient descent {mse0} -> {mses[-1]}"
    )
