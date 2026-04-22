"""Regression test for the end-to-end real-world classifier (DMRG-trained TT-MLP).

Asserts the DMRG-trained TT-MLP from
``scripts/train_real_world_classifier.py`` actually learns a real
classification task — not just fits noise. We use a small slice of the
sklearn digits dataset so the test stays fast (<10 s on the reference MX150).

Pass condition: TT-DMRG test accuracy ≥ 0.80 (versus 0.10 for random) within
the budgeted sweeps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmrg_transformer.core.device import require_cuda  # noqa: E402
from dmrg_transformer.nn import TTLinear  # noqa: E402
from dmrg_transformer.propagation.target_propagator import TargetPropagator  # noqa: E402

DTYPE = torch.float64
SEED = 42


@pytest.mark.slow
def test_real_world_classifier_learns_above_chance() -> None:
    """DMRG + target propagation on sklearn digits must beat 80% test accuracy."""
    device = require_cuda()
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    n_classes = int(y.max()) + 1
    Y_tr = np.eye(n_classes, dtype=np.float64)[y_tr]

    X_tr_t = torch.from_numpy(X_tr).to(device=device, dtype=DTYPE)
    X_te_t = torch.from_numpy(X_te).to(device=device, dtype=DTYPE)
    y_te_t = torch.from_numpy(y_te).to(device=device)
    Y_tr_t = torch.from_numpy(Y_tr).to(device=device, dtype=DTYPE)

    torch.manual_seed(SEED)
    layer1 = TTLinear(64, 32, input_dims=[8, 8], output_dims=[8, 4],
                      rank=8, bias=True, dtype=DTYPE)
    layer2 = TTLinear(32, n_classes, input_dims=[8, 4], output_dims=[5, 2],
                      rank=8, bias=True, dtype=DTYPE)
    propagator = TargetPropagator(lam=1e-2)

    epochs = 6
    blend = 0.5
    lam = 1e-2
    for _ in range(epochs):
        with torch.no_grad():
            z1 = layer1(X_tr_t)
            h1 = torch.relu(z1)
            layer2.dmrg_step(h1, Y_tr_t, lam=lam)
            W2 = layer2.to_dense_weight()
            Y_minus_b2 = Y_tr_t - layer2._bias if layer2._has_bias else Y_tr_t
            h1_target = propagator.project_through_linear(W2, Y_minus_b2)
            z1_now = layer1(X_tr_t)
            active = z1_now > 0
            z1_target = torch.where(
                active, blend * h1_target + (1 - blend) * z1_now, z1_now,
            )
            layer1.dmrg_step(X_tr_t, z1_target, lam=lam)

    with torch.no_grad():
        logits = layer2(torch.relu(layer1(X_te_t)))
        test_acc = float((logits.argmax(dim=1) == y_te_t).float().mean().item())

    assert test_acc >= 0.80, (
        f"DMRG-trained TT-MLP failed to learn the digits task: "
        f"test_acc={test_acc:.4f} (expected >= 0.80)"
    )
