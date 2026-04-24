"""Real-supervised-learning validation for the stacked TTBlock (plan §C-Validation).

Extends [scripts/train_real_world_classifier.py](train_real_world_classifier.py)
to a 1× TTBlock classifier on the same sklearn ``load_digits`` task. The 8×8
images are re-shaped as 8 tokens of dimension 8 so the attention layer has a
genuine sequence to mix.

Trainees:

1. **TT-DMRG**: input proj (exact LSQ) → TTBlock (DMRG sweeps) →
   mean-pool → linear head (closed-form LSQ each epoch). Zero gradients.
2. **Dense Adam-MSE**: identical shapes with TT model.
3. **Dense Adam-CE**: identical shapes with TT model.
4. **Large Dense Adam-CE**: ~2x parameters (matches Tier-1 compression story).

Output: ``bench/REAL_WORLD_TT_BLOCK.md``
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
from dmrg_transformer.nn.embeddings import PositionalEncoding  # noqa: E402

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
INNER_ITERS = 3


def _console_safe(text: str) -> str:
    """Return a console-safe rendering for Windows cp1252 terminals."""
    return (
        text.replace("→", "->")
        .replace("↔", "<->")
        .replace("—", "-")
        .replace("–", "-")
    )


def _console_print(text: str) -> None:
    print(_console_safe(text))

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
    def __init__(self, n_classes: int, device: torch.device) -> None:
        torch.manual_seed(SEED)
        self.device = device
        self.W_in = torch.randn(TOKEN_DIM, EMBED_DIM, dtype=DTYPE, device=device) * 0.3
        self.b_in = torch.zeros(EMBED_DIM, dtype=DTYPE, device=device)
        self.block = TTBlock(
            embed_dim=EMBED_DIM, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM,
            embed_dims=EMBED_DIMS, hidden_dims=HIDDEN_DIMS,
            rank=RANK, propagator_lam=PROP_LAM, dtype=DTYPE,
        )
        self.pos_enc = PositionalEncoding(EMBED_DIM, max_len=SEQ_LEN, dtype=DTYPE).to(device)
        self.W_head = torch.zeros(EMBED_DIM, n_classes, dtype=DTYPE, device=device)
        self.b_head = torch.zeros(n_classes, dtype=DTYPE, device=device)
        self.n_classes = n_classes

    @torch.no_grad()
    def _project_input(self, X: torch.Tensor) -> torch.Tensor:
        h = X @ self.W_in + self.b_in
        return self.pos_enc(h)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._project_input(X)
        r = self.block(h)
        pooled = r.mean(dim=1)
        logits = pooled @ self.W_head + self.b_head
        return r, pooled, logits

    @torch.no_grad()
    def _fit_head_lsq(self, pooled: torch.Tensor, Y_onehot: torch.Tensor) -> None:
        P = pooled
        gram = P.T @ P + DMRG_LAM * torch.eye(P.shape[1], dtype=DTYPE, device=P.device)
        self.W_head = torch.linalg.solve(gram, P.T @ Y_onehot)
        self.b_head = (Y_onehot - P @ self.W_head).mean(dim=0)

    @torch.no_grad()
    def train_epoch(
        self,
        X: torch.Tensor,
        Y_onehot: torch.Tensor,
        *,
        attn_target_blend: float | None = None,
    ) -> dict[str, object]:
        _, pooled, _ = self.forward(X)
        self._fit_head_lsq(pooled, Y_onehot)

        def compute_R_target() -> torch.Tensor:
            R_curr, _, logits_curr = self.forward(X)
            residual = Y_onehot - logits_curr
            W = self.W_head
            if W.shape[0] >= W.shape[1]:
                inv_term = torch.linalg.solve(W.T @ W + PROP_LAM * torch.eye(W.shape[1], dtype=DTYPE, device=W.device), W.T)
                delta_pooled = residual @ inv_term
            else:
                inv_term = torch.linalg.solve(W @ W.T + PROP_LAM * torch.eye(W.shape[0], dtype=DTYPE, device=W.device), W)
                delta_pooled = residual @ inv_term.T
            return R_curr + delta_pooled.unsqueeze(1)

        snap_W_in, snap_b_in = self.W_in.clone(), self.b_in.clone()
        loss_before = self._mse_to_targets(X, Y_onehot)
        h_target = self.block.pullback_target(self._project_input(X), compute_R_target(), target_blend=TARGET_BLEND)
        X_f, H_f = X.reshape(-1, TOKEN_DIM), h_target.reshape(-1, EMBED_DIM)
        gram_in = X_f.T @ X_f + DMRG_LAM * torch.eye(TOKEN_DIM, dtype=DTYPE, device=X.device)
        W_in_new = torch.linalg.solve(gram_in, X_f.T @ H_f)
        b_in_new = (H_f - X_f @ W_in_new).mean(dim=0)
        self.W_in = 0.5 * self.W_in + 0.5 * W_in_new
        self.b_in = 0.5 * self.b_in + 0.5 * b_in_new
        self._fit_head_lsq(self.forward(X)[1], Y_onehot)
        if self._mse_to_targets(X, Y_onehot) > 1.01 * loss_before:
            self.W_in, self.b_in = snap_W_in, snap_b_in
            proj_acc = False
        else:
            proj_acc = True

        last_mse = self._mse_to_targets(X, Y_onehot)
        total_rep = None
        for _ in range(INNER_ITERS):
            total_rep = self.block.dmrg_step(self._project_input(X), compute_R_target(), lam=DMRG_LAM, target_blend=TARGET_BLEND, attn_target_blend=attn_target_blend)
            self._fit_head_lsq(self.forward(X)[1], Y_onehot)

        return {
            "global_mse_before": last_mse, "global_mse_after": self._mse_to_targets(X, Y_onehot),
            "input_proj_accepted": proj_acc, "input_proj_alpha": 0.5 if proj_acc else 0.0,
            "attn_accepted": bool(total_rep["attn"]["accepted"]) if total_rep else False,
            "attn_diagnostics": total_rep["attn"]["diagnostics"] if total_rep else {},
        }

    @torch.no_grad()
    def _mse_to_targets(self, X: torch.Tensor, Y_onehot: torch.Tensor) -> float:
        return float(torch.mean((self.forward(X)[2] - Y_onehot) ** 2).item())

    @property
    def num_parameters(self) -> int:
        return self.W_in.numel() + self.b_in.numel() + self.block.num_parameters + self.W_head.numel() + self.b_head.numel()


# ----------------------------------------------------------------------------
# Dense baselines
# ----------------------------------------------------------------------------
class DenseBlockClassifier(torch.nn.Module):
    def __init__(self, n_classes: int, embed: int = EMBED_DIM, hidden: int = HIDDEN_DIM) -> None:
        super().__init__()
        torch.manual_seed(SEED)
        self.in_proj = torch.nn.Linear(TOKEN_DIM, embed)
        self.pos_enc = PositionalEncoding(embed, max_len=SEQ_LEN, dtype=DTYPE)
        self.ln1 = torch.nn.LayerNorm(embed, elementwise_affine=False)
        self.attn = torch.nn.MultiheadAttention(embed, NUM_HEADS, batch_first=True, bias=True)
        self.ln2 = torch.nn.LayerNorm(embed, elementwise_affine=False)
        self.fc1 = torch.nn.Linear(embed, hidden)
        self.fc2 = torch.nn.Linear(hidden, embed)
        self.head = torch.nn.Linear(embed, n_classes)
        self.to(dtype=DTYPE)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.in_proj(X))
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), need_weights=False)
        h = h + a
        ff = self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(h))))
        return self.head((h + ff).mean(dim=1))


def train_dense(model: torch.nn.Module, data: dict, *, loss_kind: str) -> dict[str, list]:
    opt = torch.optim.AdamW(model.parameters(), lr=ADAM_LR)
    history = {"epoch": [], "train_acc": [], "test_acc": [], "wall": [], "step_wall": [], "step_test_acc": []}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step_idx = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for _ in range(ADAM_ITERS_PER_EPOCH):
            opt.zero_grad()
            logits = model(data["X_tr"])
            loss = torch.nn.functional.mse_loss(logits, data["Y_tr_onehot"]) if loss_kind == "mse" else torch.nn.functional.cross_entropy(logits, data["y_tr"])
            loss.backward()
            opt.step()
            step_idx += 1
            if step_idx % 10 == 0:
                model.eval()
                with torch.no_grad():
                    history["step_wall"].append(time.perf_counter() - t0)
                    history["step_test_acc"].append(accuracy(model(data["X_te"]), data["y_te"]))
                model.train()
        model.eval()
        with torch.no_grad():
            history["epoch"].append(epoch)
            history["train_acc"].append(accuracy(model(data["X_tr"]), data["y_tr"]))
            history["test_acc"].append(accuracy(model(data["X_te"]), data["y_te"]))
            history["wall"].append(time.perf_counter() - t0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        history["peak_mem_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return history


def train_tt(model: TTBlockClassifier, data: dict) -> dict[str, list]:
    history = {"epoch": [], "train_acc": [], "test_acc": [], "wall": [], "block_mse_before": [], "block_mse_after": [], "input_proj_accepted": [], "attn_accepted": []}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        rep = model.train_epoch(data["X_tr"], data["Y_tr_onehot"])
        with torch.no_grad():
            history["epoch"].append(epoch)
            history["train_acc"].append(accuracy(model.forward(data["X_tr"])[2], data["y_tr"]))
            history["test_acc"].append(accuracy(model.forward(data["X_te"])[2], data["y_te"]))
            history["wall"].append(time.perf_counter() - t0)
            history["block_mse_before"].append(rep["global_mse_before"])
            history["block_mse_after"].append(rep["global_mse_after"])
            history["input_proj_accepted"].append(rep["input_proj_accepted"])
            history["attn_accepted"].append(rep["attn_accepted"])
            _console_print(f"  TT-DMRG    ep{epoch}: train={history['train_acc'][-1]:.4f} test={history['test_acc'][-1]:.4f}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        history["peak_mem_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return history

def measure_inference_latency(forward_fn, X: torch.Tensor, *, warmup: int = 5, repeats: int = 20) -> dict[str, float]:
    cuda = torch.cuda.is_available()
    for _ in range(warmup): forward_fn(X)
    if cuda: torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        forward_fn(X)
        if cuda: torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {"median_ms": samples[len(samples) // 2], "throughput": X.shape[0] * 1000.0 / samples[len(samples) // 2]}

def main() -> None:
    device = require_cuda()
    data = load_data(device)
    n_classes = data["n_classes"]

    _console_print("=== TTBlock trained by DMRG ===")
    tt = TTBlockClassifier(n_classes, device)
    tt_hist = train_tt(tt, data)

    _console_print("\n=== Dense block (AdamW + MSE) ===")
    dense_mse = DenseBlockClassifier(n_classes).to(device)
    dense_mse_hist = train_dense(dense_mse, data, loss_kind="mse")

    _console_print("\n=== Dense block (AdamW + CE) ===")
    dense_ce = DenseBlockClassifier(n_classes).to(device)
    dense_ce_hist = train_dense(dense_ce, data, loss_kind="ce")

    _console_print("\n=== Large Dense block (AdamW + CE) ===")
    large_dense = DenseBlockClassifier(n_classes, embed=24, hidden=24).to(device)
    large_dense_hist = train_dense(large_dense, data, loss_kind="ce")

    def _acc_at(hist, t):
        combined = sorted(list(zip(hist["step_wall"], hist["step_test_acc"])) + list(zip(hist["wall"], hist["test_acc"])))
        c = [a for w, a in combined if w <= t]
        return c[-1] if c else combined[0][1]

    t_limit = tt_hist["wall"][-1]
    iso_mse, iso_ce, iso_large = _acc_at(dense_mse_hist, t_limit), _acc_at(dense_ce_hist, t_limit), _acc_at(large_dense_hist, t_limit)

    out = ROOT / "bench" / "REAL_WORLD_TT_BLOCK.md"
    lines = ["# Softmax-Attention TTBlock — Real-Task Validation", "", f"**Device:** `{describe_device()}`", ""]
    lines += ["## Final test-set accuracy", "", "| Model | Train acc | **Test acc** | Params | Wall (s) | Peak GPU (MiB) |", "| :---- | --------: | -----------: | -----: | -------: | -------------: |"]
    lines.append(f"| TT-DMRG (no grads) | {tt_hist['train_acc'][-1]:.4f} | **{tt_hist['test_acc'][-1]:.4f}** | {tt.num_parameters:,} | {tt_hist['wall'][-1]:.2f} | {tt_hist['peak_mem_mib']:.1f} |")
    lines.append(f"| Dense (AdamW, MSE) | {dense_mse_hist['train_acc'][-1]:.4f} | **{dense_mse_hist['test_acc'][-1]:.4f}** | {sum(p.numel() for p in dense_mse.parameters()):,} | {dense_mse_hist['wall'][-1]:.2f} | {dense_mse_hist['peak_mem_mib']:.1f} |")
    lines.append(f"| Dense (AdamW, CE)  | {dense_ce_hist['train_acc'][-1]:.4f} | **{dense_ce_hist['test_acc'][-1]:.4f}** | {sum(p.numel() for p in dense_ce.parameters()):,} | {dense_ce_hist['wall'][-1]:.2f} | {dense_ce_hist['peak_mem_mib']:.1f} |")
    lines.append(f"| Large Dense (CE)   | {large_dense_hist['train_acc'][-1]:.4f} | **{large_dense_hist['test_acc'][-1]:.4f}** | {sum(p.numel() for p in large_dense.parameters()):,} | {large_dense_hist['wall'][-1]:.2f} | {large_dense_hist['peak_mem_mib']:.1f} |")
    lines += ["", "## Iso-time fairness check", "", "| Comparison | Wall budget (s) | Test acc at budget | Final test acc |", "| :--------- | --------------: | -----------------: | -------------: |"]
    lines.append(f"| TT-DMRG (reference) | {t_limit:.2f} | **{tt_hist['test_acc'][-1]:.4f}** | {tt_hist['test_acc'][-1]:.4f} |")
    lines.append(f"| Dense Adam-MSE      | {t_limit:.2f} | **{iso_mse:.4f}** | {dense_mse_hist['test_acc'][-1]:.4f} |")
    lines.append(f"| Dense Adam-CE       | {t_limit:.2f} | **{iso_ce:.4f}** | {dense_ce_hist['test_acc'][-1]:.4f} |")
    lines.append(f"| Large Dense-CE      | {t_limit:.2f} | **{iso_large:.4f}** | {large_dense_hist['test_acc'][-1]:.4f} |")
    out.write_text("\n".join(lines), encoding="utf-8")
    _console_print(f"\nWrote {out.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
