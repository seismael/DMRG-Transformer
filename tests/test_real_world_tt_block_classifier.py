"""Regression guard for the stacked-TTBlock real-task classifier.

Runs a tiny version of [scripts/train_real_world_tt_block_classifier.py]
(../scripts/train_real_world_tt_block_classifier.py) and asserts the TT-DMRG
trainee reaches at least 50% test accuracy on a sklearn-digits subset. The
bar is intentionally loose because Q/K projections are frozen this slice
(see `docs/COMPLIANCE.md` §C3) — the goal is to catch regressions that
break learning entirely, not to chase accuracy parity with Adam.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from dmrg_transformer.core.device import require_cuda
from dmrg_transformer.nn import TTBlock

DTYPE = torch.float64
SEED = 7
SEQ_LEN = 8
TOKEN_DIM = 8
EMBED_DIM = 16
HIDDEN_DIM = 16
NUM_HEADS = 2
RANK = 4
EPOCHS = 4
DMRG_LAM = 1.0e-2
PROP_LAM = 1.0e-2
TARGET_BLEND = 0.5


def test_tt_block_classifier_learns_above_random() -> None:
    device = require_cuda()
    digits = load_digits()
    # Subsample for speed.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(digits.data.shape[0], size=600, replace=False)
    X = digits.data[idx].astype(np.float64) / 16.0
    y = digits.target[idx].astype(np.int64)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED,
    )
    n_classes = int(y.max()) + 1
    Y_tr = np.eye(n_classes, dtype=np.float64)[y_tr]

    X_tr_t = torch.from_numpy(X_tr.reshape(-1, SEQ_LEN, TOKEN_DIM)).to(device, DTYPE)
    X_te_t = torch.from_numpy(X_te.reshape(-1, SEQ_LEN, TOKEN_DIM)).to(device, DTYPE)
    y_te_t = torch.from_numpy(y_te).to(device)
    Y_tr_t = torch.from_numpy(Y_tr).to(device, DTYPE)

    torch.manual_seed(SEED)
    W_in = torch.randn(TOKEN_DIM, EMBED_DIM, dtype=DTYPE, device=device) * 0.3
    block = TTBlock(
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM,
        embed_dims=[4, 4], hidden_dims=[4, 4],
        rank=RANK, propagator_lam=PROP_LAM, dtype=DTYPE,
    )
    W_head = torch.zeros(EMBED_DIM, n_classes, dtype=DTYPE, device=device)
    b_head = torch.zeros(n_classes, dtype=DTYPE, device=device)

    eye_e = torch.eye(EMBED_DIM, dtype=DTYPE, device=device)

    @torch.no_grad()
    def forward(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = X @ W_in
        r = block(h)
        pooled = r.mean(dim=1)
        return pooled, pooled @ W_head + b_head

    @torch.no_grad()
    def fit_head(pooled: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gram = pooled.T @ pooled + DMRG_LAM * eye_e
        W = torch.linalg.solve(gram, pooled.T @ Y)
        b = (Y - pooled @ W).mean(dim=0)
        return W, b

    for _ in range(EPOCHS):
        pooled, _ = forward(X_tr_t)
        W_head, b_head = fit_head(pooled, Y_tr_t)
        gram_h = W_head @ W_head.T + PROP_LAM * eye_e
        inv_W = torch.linalg.solve(gram_h, W_head)
        pooled_target = (Y_tr_t - b_head) @ inv_W.T
        R_target = pooled_target.unsqueeze(1).expand(-1, SEQ_LEN, -1).contiguous()
        h = X_tr_t @ W_in
        block.dmrg_step(h, R_target, lam=DMRG_LAM, target_blend=TARGET_BLEND)

    with torch.no_grad():
        _, logits_te = forward(X_te_t)
        acc = float((logits_te.argmax(dim=1) == y_te_t).float().mean().item())

    # Loose bar: chance is 10%, simple linear-on-pixels baseline ~70-80%.
    # We require 50% to confirm the block + propagation pipeline is *learning*.
    assert acc >= 0.50, f"TTBlock classifier collapsed: test_acc={acc:.4f}"
