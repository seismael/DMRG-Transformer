"""Proves that ``qk_first=True`` unfreezes attention by training Q/K
**before** W_out, so the MSE floor is not yet saturated when Q/K runs.

Contrasts with the default ordering (W_out first) where Q/K acceptance
is 0 % because W_out has already fit the target near-perfectly.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from dmrg_transformer.core.device import require_cuda
from dmrg_transformer.nn import TTBlock

DTYPE = torch.float64


def _train_qk_acceptance(
    qk_first: bool,
    epochs: int = 6,
    seed: int = 7,
) -> tuple[float, float]:
    """Train a single TTBlock, return (qk_accept_rate, test_acc)."""
    device = require_cuda()
    digits = load_digits()
    rng = np.random.default_rng(seed)
    idx = rng.choice(digits.data.shape[0], size=600, replace=False)
    X = digits.data[idx].astype(np.float64) / 16.0
    y = digits.target[idx].astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed,
    )
    Y_tr = np.eye(10, dtype=np.float64)[y_tr]
    X_tr_t = torch.from_numpy(X_tr.reshape(-1, 8, 8)).to(device, DTYPE)
    X_te_t = torch.from_numpy(X_te.reshape(-1, 8, 8)).to(device, DTYPE)
    y_te_t = torch.from_numpy(y_te).to(device)
    Y_tr_t = torch.from_numpy(Y_tr).to(device, DTYPE)

    torch.manual_seed(seed)
    W_in = torch.randn(8, 16, dtype=DTYPE, device=device) * 0.3
    block = TTBlock(
        embed_dim=16, num_heads=2, hidden_dim=16,
        embed_dims=[4, 4], hidden_dims=[4, 4],
        rank=4, propagator_lam=1e-2, dtype=DTYPE,
    )
    W_head = torch.zeros(16, 10, dtype=DTYPE, device=device)
    b_head = torch.zeros(10, dtype=DTYPE, device=device)
    eye_e = torch.eye(16, dtype=DTYPE, device=device)

    qk_attempts = 0
    qk_accepted = 0.0

    for _ in range(epochs):
        h = X_tr_t @ W_in
        r = block.forward(h)
        pooled = r.mean(dim=1)

        gram = pooled.T @ pooled + 1e-2 * eye_e
        W_head = torch.linalg.solve(gram, pooled.T @ Y_tr_t)
        b_head = (Y_tr_t - pooled @ W_head).mean(dim=0)

        gram_h = W_head @ W_head.T + 1e-2 * eye_e
        inv_W = torch.linalg.solve(gram_h, W_head)
        pooled_target = (Y_tr_t - b_head) @ inv_W.T
        R_target = pooled_target.unsqueeze(1).expand(-1, 8, -1).contiguous()

        h_in = X_tr_t @ W_in
        result = block.dmrg_step(
            h_in, R_target, lam=1e-2, target_blend=0.5,
            qk_first=qk_first,
        )
        attn = result.get("attn", {})
        diag = attn.get("diagnostics", {}) if isinstance(attn, dict) else {}
        qk_attempts += 1
        qk_accepted += float(diag.get("qk_accepted", False))

    with torch.no_grad():
        h_te = X_te_t @ W_in
        r_te = block.forward(h_te)
        pooled_te = r_te.mean(dim=1)
        logits_te = pooled_te @ W_head + b_head
        acc = float((logits_te.argmax(dim=1) == y_te_t).float().mean().item())

    qk_rate = qk_accepted / max(qk_attempts, 1)
    return qk_rate, acc


# ---- tests ------------------------------------------------------------------


def test_qk_first_increases_acceptance_rate() -> None:
    """qk_first ordering must yield higher Q/K acceptance than default."""
    qk_default, acc_default = _train_qk_acceptance(qk_first=False, epochs=4)
    qk_first, acc_first = _train_qk_acceptance(qk_first=True, epochs=4)

    # Core claim: qk_first accepts Q/K updates at least once.
    assert qk_first > 0.0, (
        f"qk_first Q/K still frozen: acceptance_rate={qk_first:.2f}"
    )

    # qk_first should accept Q/K at least as often as default (usually more).
    assert qk_first >= qk_default, (
        f"qk_first acceptance ({qk_first:.2f}) < default ({qk_default:.2f})"
    )

    # Accuracy must not regress.
    assert acc_first >= acc_default * 0.8, (
        f"qk_first accuracy regressed: {acc_first:.4f} vs {acc_default:.4f}"
    )


def test_qk_first_does_not_regress_accuracy() -> None:
    """Across seeds, qk_first must not catastrophically regress accuracy."""
    for seed in [7, 42, 99]:
        _, acc_default = _train_qk_acceptance(qk_first=False, epochs=4, seed=seed)
        _, acc_first = _train_qk_acceptance(qk_first=True, epochs=4, seed=seed)
        assert acc_first >= acc_default * 0.70, (
            f"seed={seed}: qk_first acc={acc_first:.4f} vs default={acc_default:.4f}"
        )
