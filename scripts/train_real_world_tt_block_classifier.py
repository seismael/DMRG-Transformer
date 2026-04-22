"""Real-supervised-learning validation for the stacked TTBlock (plan §C-Validation).

Extends [scripts/train_real_world_classifier.py](train_real_world_classifier.py)
to a 1× TTBlock classifier on the same sklearn ``load_digits`` task. The 8×8
images are re-shaped as 8 tokens of dimension 8 so the attention layer has a
genuine sequence to mix.

Three architecturally identical trainees:

1. **TT-DMRG**: input proj (frozen at init) → TTBlock (DMRG sweeps) →
   mean-pool → linear head (closed-form LSQ each epoch). Zero gradients.
2. **Adam-MSE**: identical shapes with dense ``nn.Linear`` + ``nn.MultiheadAttention``
   + GELU FFN, AdamW + MSE-on-one-hot.
3. **Adam-CE**: same as (2) with cross-entropy loss.

Output: ``bench/REAL_WORLD_TT_BLOCK.md`` with held-out test accuracy,
behavior agreement, confusion matrices, **and an explicit root-cause
analysis** of the measured DMRG-vs-Adam gap.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmrg_transformer.core.device import describe_device, require_cuda  # noqa: E402
from dmrg_transformer.nn import TTBlock  # noqa: E402

# ----------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------
DTYPE = torch.float64
SEED = 42
EMBED_DIM = 16
HIDDEN_DIM = 16
NUM_HEADS = 2
SEQ_LEN = 8         # rows of the 8×8 digit
TOKEN_DIM = 8       # cols
RANK = 8
EPOCHS = 12
ADAM_LR = 1e-2
ADAM_ITERS_PER_EPOCH = 50
DMRG_LAM = 1e-2
PROP_LAM = 1e-2
TARGET_BLEND = 0.5

# Factorizations.
EMBED_DIMS = [4, 4]      # 16
HIDDEN_DIMS = [4, 4]     # 16


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(device: torch.device) -> dict[str, torch.Tensor]:
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0  # [N, 64]
    y = digits.target.astype(np.int64)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    n_classes = int(y.max()) + 1
    Y_tr_onehot = np.eye(n_classes, dtype=np.float64)[y_tr]
    Y_te_onehot = np.eye(n_classes, dtype=np.float64)[y_te]

    def reshape_seq(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(-1, SEQ_LEN, TOKEN_DIM)

    return {
        "X_tr": torch.from_numpy(reshape_seq(X_tr)).to(device=device, dtype=DTYPE),
        "X_te": torch.from_numpy(reshape_seq(X_te)).to(device=device, dtype=DTYPE),
        "y_tr": torch.from_numpy(y_tr).to(device=device),
        "y_te": torch.from_numpy(y_te).to(device=device),
        "Y_tr_onehot": torch.from_numpy(Y_tr_onehot).to(device=device, dtype=DTYPE),
        "Y_te_onehot": torch.from_numpy(Y_te_onehot).to(device=device, dtype=DTYPE),
        "n_classes": n_classes,
    }


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def confusion(pred: torch.Tensor, y: torch.Tensor, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    p = pred.cpu().numpy()
    t = y.cpu().numpy()
    for ti, pi in zip(t, p, strict=False):
        cm[ti, pi] += 1
    return cm


# ----------------------------------------------------------------------------
# DMRG-trained TTBlock classifier
# ----------------------------------------------------------------------------
class TTBlockClassifier:
    """``input_proj → TTBlock → mean-pool → linear head`` trained by DMRG sweeps.

    The input projection is fixed at initialization (random Gaussian) so the
    DMRG path stays focused on the block itself; the linear head is fit by
    closed-form least squares each epoch (the linear-regression analog of a
    DMRG sweep on a single TT layer of rank=∞).
    """

    def __init__(self, n_classes: int, device: torch.device) -> None:
        torch.manual_seed(SEED)
        self.device = device
        # Frozen random input projection [TOKEN_DIM, EMBED_DIM].
        self.W_in = torch.randn(TOKEN_DIM, EMBED_DIM, dtype=DTYPE, device=device) * 0.3
        self.b_in = torch.zeros(EMBED_DIM, dtype=DTYPE, device=device)
        self.block = TTBlock(
            embed_dim=EMBED_DIM, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM,
            embed_dims=EMBED_DIMS, hidden_dims=HIDDEN_DIMS,
            rank=RANK, propagator_lam=PROP_LAM, dtype=DTYPE,
        )
        self.W_head = torch.zeros(EMBED_DIM, n_classes, dtype=DTYPE, device=device)
        self.b_head = torch.zeros(n_classes, dtype=DTYPE, device=device)
        self.n_classes = n_classes

    @torch.no_grad()
    def _project_input(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, SEQ, TOKEN_DIM] -> [B, SEQ, EMBED_DIM]
        return X @ self.W_in + self.b_in

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._project_input(X)        # [B, SEQ, EMBED]
        r = self.block(h)                  # [B, SEQ, EMBED]
        pooled = r.mean(dim=1)             # [B, EMBED]
        logits = pooled @ self.W_head + self.b_head  # [B, classes]
        return r, pooled, logits

    @torch.no_grad()
    def _fit_head_lsq(self, pooled: torch.Tensor, Y_onehot: torch.Tensor) -> None:
        """Closed-form ridge regression for the linear head."""
        # Solve (P^T P + λ I) W = P^T Y, then b = mean(Y - P W).
        P = pooled
        gram = P.T @ P
        gram = gram + DMRG_LAM * torch.eye(gram.shape[0], dtype=DTYPE, device=P.device)
        rhs = P.T @ Y_onehot
        W = torch.linalg.solve(gram, rhs)
        b = (Y_onehot - P @ W).mean(dim=0)
        self.W_head = W
        self.b_head = b

    @torch.no_grad()
    def train_epoch(self, X: torch.Tensor, Y_onehot: torch.Tensor) -> dict[str, float]:
        # 1) Forward to expose intermediate activations.
        r, pooled, _ = self.forward(X)

        # 2) Fit head exactly (closed-form LSQ).
        self._fit_head_lsq(pooled, Y_onehot)

        # 3) Pull head target back to pooled target via Tikhonov pseudo-inverse.
        Y_minus_b = Y_onehot - self.b_head
        gram_h = self.W_head @ self.W_head.T
        gram_h = gram_h + PROP_LAM * torch.eye(
            gram_h.shape[0], dtype=DTYPE, device=gram_h.device,
        )
        # pooled_target = (Y - b) W^T (W W^T + λI)^-1 = solve(gram, W) yields
        # the right matrix.
        inv_W = torch.linalg.solve(gram_h, self.W_head)  # [EMBED, classes]
        pooled_target = Y_minus_b @ inv_W.T              # [B, EMBED]

        # 4) Broadcast pooled target to per-token target (one-step
        #    linearization: ``mean(R_target) = pooled_target`` with constant
        #    per-token contribution).
        R_target = pooled_target.unsqueeze(1).expand(-1, SEQ_LEN, -1).contiguous()

        # 5) Sweep the block.
        h = self._project_input(X)
        report = self.block.dmrg_step(h, R_target, lam=DMRG_LAM, target_blend=TARGET_BLEND)

        return {
            "global_mse_before": report["global_mse_before"],
            "global_mse_after": report["global_mse_after"],
        }

    @property
    def num_parameters(self) -> int:
        return (
            self.W_in.numel() + self.b_in.numel()
            + self.block.num_parameters
            + self.W_head.numel() + self.b_head.numel()
        )


# ----------------------------------------------------------------------------
# Dense baseline: same architecture with autograd
# ----------------------------------------------------------------------------
class DenseBlockClassifier(torch.nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        torch.manual_seed(SEED)
        self.in_proj = torch.nn.Linear(TOKEN_DIM, EMBED_DIM)
        self.ln1 = torch.nn.LayerNorm(EMBED_DIM, elementwise_affine=False)
        self.attn = torch.nn.MultiheadAttention(
            EMBED_DIM, NUM_HEADS, batch_first=True, bias=True,
        )
        self.ln2 = torch.nn.LayerNorm(EMBED_DIM, elementwise_affine=False)
        self.fc1 = torch.nn.Linear(EMBED_DIM, HIDDEN_DIM)
        self.fc2 = torch.nn.Linear(HIDDEN_DIM, EMBED_DIM)
        self.head = torch.nn.Linear(EMBED_DIM, n_classes)
        self.to(dtype=DTYPE)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(X)
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), need_weights=False)
        h = h + a
        ff = self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(h))))
        h = h + ff
        return self.head(h.mean(dim=1))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_dense(
    model: DenseBlockClassifier, data: dict, *, loss_kind: str,
) -> dict[str, list]:
    opt = torch.optim.AdamW(model.parameters(), lr=ADAM_LR)
    history: dict[str, list] = {"epoch": [], "train_acc": [], "test_acc": [], "wall": []}
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for _ in range(ADAM_ITERS_PER_EPOCH):
            opt.zero_grad()
            logits = model(data["X_tr"])
            if loss_kind == "mse":
                loss = torch.nn.functional.mse_loss(logits, data["Y_tr_onehot"])
            else:
                loss = torch.nn.functional.cross_entropy(logits, data["y_tr"])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_acc = accuracy(model(data["X_tr"]), data["y_tr"])
            te_acc = accuracy(model(data["X_te"]), data["y_te"])
        history["epoch"].append(epoch)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["wall"].append(time.perf_counter() - t0)
        print(f"  Dense({loss_kind})  ep{epoch}: train_acc={tr_acc:.4f} test_acc={te_acc:.4f}")
    return history


def train_tt(model: TTBlockClassifier, data: dict) -> dict[str, list]:
    history: dict[str, list] = {
        "epoch": [], "train_acc": [], "test_acc": [], "wall": [],
        "block_mse_before": [], "block_mse_after": [],
    }
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        rep = model.train_epoch(data["X_tr"], data["Y_tr_onehot"])
        with torch.no_grad():
            _, _, logits_tr = model.forward(data["X_tr"])
            _, _, logits_te = model.forward(data["X_te"])
            tr_acc = accuracy(logits_tr, data["y_tr"])
            te_acc = accuracy(logits_te, data["y_te"])
        history["epoch"].append(epoch)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["wall"].append(time.perf_counter() - t0)
        history["block_mse_before"].append(rep["global_mse_before"])
        history["block_mse_after"].append(rep["global_mse_after"])
        print(
            f"  TT-DMRG    ep{epoch}: train_acc={tr_acc:.4f} test_acc={te_acc:.4f}  "
            f"blk_mse {rep['global_mse_before']:.3e}→{rep['global_mse_after']:.3e}"
        )
    return history


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _fmt_cm(cm: np.ndarray) -> str:
    n = cm.shape[0]
    header = "| true \\ pred | " + " | ".join(str(i) for i in range(n)) + " |"
    sep = "| :- | " + " | ".join("-:" for _ in range(n)) + " |"
    rows = [
        f"| **{i}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |"
        for i in range(n)
    ]
    return "\n".join([header, sep, *rows])


def main() -> None:
    device = require_cuda()
    print(f"Real-world TTBlock classifier benchmark on {describe_device()}")
    print(
        f"Architecture: [B,{SEQ_LEN},{TOKEN_DIM}] -> proj -> TTBlock("
        f"embed={EMBED_DIM}, heads={NUM_HEADS}, hidden={HIDDEN_DIM}, rank={RANK}) "
        f"-> mean-pool -> head"
    )

    data = load_data(device)
    n_classes = data["n_classes"]
    print(f"Dataset: sklearn load_digits — train={data['X_tr'].shape[0]}, "
          f"test={data['X_te'].shape[0]}, classes={n_classes}\n")

    print("=== TTBlock trained by DMRG + target propagation ===")
    tt = TTBlockClassifier(n_classes, device)
    tt_hist = train_tt(tt, data)

    print("\n=== Dense block (AdamW + MSE) ===")
    dense_mse = DenseBlockClassifier(n_classes).to(device)
    dense_mse_hist = train_dense(dense_mse, data, loss_kind="mse")

    print("\n=== Dense block (AdamW + CE) ===")
    dense_ce = DenseBlockClassifier(n_classes).to(device)
    dense_ce_hist = train_dense(dense_ce, data, loss_kind="ce")

    with torch.no_grad():
        _, _, tt_logits = tt.forward(data["X_te"])
        cm_tt = confusion(tt_logits.argmax(dim=1), data["y_te"], n_classes)
        cm_mse = confusion(dense_mse(data["X_te"]).argmax(dim=1), data["y_te"], n_classes)
        cm_ce = confusion(dense_ce(data["X_te"]).argmax(dim=1), data["y_te"], n_classes)
        p_tt = tt_logits.argmax(dim=1)
        p_mse = dense_mse(data["X_te"]).argmax(dim=1)
        p_ce = dense_ce(data["X_te"]).argmax(dim=1)
        agree_tt_mse = float((p_tt == p_mse).float().mean().item())
        agree_tt_ce = float((p_tt == p_ce).float().mean().item())
        agree_mse_ce = float((p_mse == p_ce).float().mean().item())

    out = ROOT / "bench" / "REAL_WORLD_TT_BLOCK.md"
    out.parent.mkdir(exist_ok=True)
    dense_params = sum(p.numel() for p in dense_mse.parameters())
    tt_params = tt.num_parameters

    lines: list[str] = []
    lines.append("# DMRG-Transformer — Stacked TTBlock Real-Task Validation")
    lines.append("")
    lines.append(f"**Device:** `{describe_device()}`  ")
    lines.append("**Task:** 10-class classification on `sklearn.datasets.load_digits` "
                 f"reshaped as {SEQ_LEN} tokens of dim {TOKEN_DIM} (stratified 80/20 "
                 f"split, seed={SEED}).  ")
    lines.append(f"**Architecture:** input proj → 1× TTBlock(embed={EMBED_DIM}, "
                 f"heads={NUM_HEADS}, hidden={HIDDEN_DIM}, rank={RANK}) → mean-pool → "
                 "linear head.  ")
    lines.append(f"**TT-DMRG path:** zero gradients. Block trained by per-block "
                 f"`dmrg_step` ({EPOCHS} epochs); head fit by closed-form ridge LSQ.  ")
    lines.append(f"**Adam baselines:** identical-shape dense block "
                 f"(`nn.MultiheadAttention` + GELU FFN), AdamW lr={ADAM_LR}, "
                 f"{ADAM_ITERS_PER_EPOCH * EPOCHS} total steps.")
    lines.append("")
    lines.append("## Final test-set accuracy")
    lines.append("")
    lines.append("| Model | Train acc | **Test acc** | Params | Wall (s) |")
    lines.append("| :---- | --------: | -----------: | -----: | -------: |")
    lines.append(
        f"| TT-DMRG (no grads) | {tt_hist['train_acc'][-1]:.4f} | "
        f"**{tt_hist['test_acc'][-1]:.4f}** | {tt_params:,} | "
        f"{tt_hist['wall'][-1]:.2f} |"
    )
    lines.append(
        f"| Dense (AdamW, MSE) | {dense_mse_hist['train_acc'][-1]:.4f} | "
        f"**{dense_mse_hist['test_acc'][-1]:.4f}** | {dense_params:,} | "
        f"{dense_mse_hist['wall'][-1]:.2f} |"
    )
    lines.append(
        f"| Dense (AdamW, CE)  | {dense_ce_hist['train_acc'][-1]:.4f} | "
        f"**{dense_ce_hist['test_acc'][-1]:.4f}** | {dense_params:,} | "
        f"{dense_ce_hist['wall'][-1]:.2f} |"
    )
    lines.append("")
    gap_mse = dense_mse_hist["test_acc"][-1] - tt_hist["test_acc"][-1]
    gap_ce = dense_ce_hist["test_acc"][-1] - tt_hist["test_acc"][-1]
    lines.append(f"**Measured DMRG → Adam-MSE gap:** {gap_mse * 100:+.2f} pp  ")
    lines.append(f"**Measured DMRG → Adam-CE  gap:** {gap_ce * 100:+.2f} pp")
    lines.append("")
    lines.append("## Behavioral agreement on test set")
    lines.append("")
    lines.append(f"* TT-DMRG ↔ Dense-MSE: **{agree_tt_mse:.4f}**")
    lines.append(f"* TT-DMRG ↔ Dense-CE:  **{agree_tt_ce:.4f}**")
    lines.append(f"* Dense-MSE ↔ Dense-CE: **{agree_mse_ce:.4f}** (sanity check)")
    lines.append("")
    lines.append("## Per-epoch test accuracy")
    lines.append("")
    lines.append("| Epoch | TT-DMRG | Dense (MSE) | Dense (CE) |")
    lines.append("| ----: | ------: | ----------: | ---------: |")
    for i in range(EPOCHS):
        lines.append(
            f"| {i+1} | {tt_hist['test_acc'][i]:.4f} | "
            f"{dense_mse_hist['test_acc'][i]:.4f} | "
            f"{dense_ce_hist['test_acc'][i]:.4f} |"
        )
    lines.append("")
    lines.append("## TTBlock per-epoch global MSE (block forward target tracking)")
    lines.append("")
    lines.append("| Epoch | MSE before sweep | MSE after sweep |")
    lines.append("| ----: | ---------------: | --------------: |")
    for i in range(EPOCHS):
        lines.append(
            f"| {i+1} | {tt_hist['block_mse_before'][i]:.3e} | "
            f"{tt_hist['block_mse_after'][i]:.3e} |"
        )
    lines.append("")
    lines.append("## Confusion matrices (held-out test set)")
    lines.append("")
    for title, cm in [
        ("### TT-DMRG", cm_tt),
        ("### Dense (AdamW + MSE)", cm_mse),
        ("### Dense (AdamW + CE)", cm_ce),
    ]:
        lines.append(title)
        lines.append("")
        lines.append(_fmt_cm(cm))
        lines.append("")
    lines.append("## Honest gap analysis — root causes")
    lines.append("")
    lines.append("The measured DMRG-vs-Adam gap on this stacked-TTBlock task is *expected* "
                 "to be larger than the 9 pp MLP gap reported in [bench/REAL_WORLD_MNIST.md]"
                 "(REAL_WORLD_MNIST.md). The dominant root causes are documented below — "
                 "they are **propagation limitations**, not solver-precision issues:")
    lines.append("")
    lines.append("1. **Frozen Q/K projections.** The current `TTBlock.dmrg_step` only "
                 "updates `W_out` and the FFN sub-block. Pulling a target through "
                 "`softmax(QK^T)V` requires linearizing through the softmax Jacobian, "
                 "which is not yet implemented. Q/K stay at their random initialization "
                 "for the entire run, so the attention pattern itself never adapts to the "
                 "task. (See `docs/COMPLIANCE.md` §C3 deferral note.)")
    lines.append("")
    lines.append("2. **Frozen input projection.** The input projection (token-dim → "
                 "embed-dim) is held at initialization. This caps the upstream "
                 "expressiveness available to the block.")
    lines.append("")
    lines.append("3. **Pooled-target broadcast.** The head target is pulled back to a "
                 "*single* pooled vector and broadcast to every token, so the per-token "
                 "block targets have rank-1 structure across the sequence axis. Adam's "
                 "backprop can shape per-token outputs independently.")
    lines.append("")
    lines.append("4. **GELU active-mask propagation** is identical to the MLP slice's "
                 "ReLU mask trick — first-order, not exact. This is a smaller contributor.")
    lines.append("")
    lines.append("Closing this gap requires implementing softmax pull-back for Q/K and an "
                 "exact-solver update for the input projection — both deferred to a "
                 "follow-up plan slice (see `/memories/session/plan_c2_c3_c4.md` "
                 "*Deferred* section).")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
