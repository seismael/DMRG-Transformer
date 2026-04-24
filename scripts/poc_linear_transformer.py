"""Tier-2 (1× block) classifier — linear-attention variant.

Drop-in counterpart to ``train_real_world_tt_block_classifier.py`` that
swaps the softmax ``TTBlock`` for the multilinear ``TTLinearAttentionBlock``
(see [bench/PHASE0_DIAGNOSTIC.md](../bench/PHASE0_DIAGNOSTIC.md) for the
motivation).

To minimize divergence and review cost we monkey-patch the existing runner
module's ``TTBlock`` reference and a couple of call sites that pass
softmax-only kwargs (``target_blend``, ``attn_target_blend``).

Outputs:
- Markdown report: ``bench/REAL_WORLD_LIN_TT_BLOCK.md``
- Coverage sidecar (Phase E aggregator): ``bench/_coverage/tier2_one_block_linear.json``
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # make ``scripts`` importable

# Import the softmax runner module so we can reuse all of its plumbing.
import scripts.poc_softmax_transformer as softmax_runner  # noqa: E402
from dmrg_transformer.nn.tt_linear_attention_block import TTLinearAttentionBlock  # noqa: E402


# --- Monkey patches: swap TTBlock for TTLinearAttentionBlock and adapt the
#     few call sites that pass softmax-only kwargs. -------------------------

softmax_runner.TTBlock = TTLinearAttentionBlock  # type: ignore[assignment]


# Capture the original train_epoch so we can re-implement its DMRG call without
# the softmax-only ``attn_target_blend`` / ``target_blend`` parameters.
_OriginalTTBlockClassifier = softmax_runner.TTBlockClassifier


class _LinearTTBlockClassifier(_OriginalTTBlockClassifier):
    """Subclass that replaces the softmax-flavored block sweep call."""

    def train_epoch(self, X, Y_onehot, *, attn_target_blend=None):
        # Re-implement train_epoch from softmax_runner.TTBlockClassifier but:
        # (a) construct ``self.block`` as a TTLinearAttentionBlock (already true
        #     because softmax_runner.TTBlock is now patched); and
        # (b) call ``block.dmrg_step`` without the softmax-only kwargs.
        import torch  # local import keeps top of file slim

        rcls = softmax_runner
        # 1) Forward & exact-LSQ head.
        _, pooled, _ = self.forward(X)
        self._fit_head_lsq(pooled, Y_onehot)

        def compute_R_target() -> torch.Tensor:
            """Difference Target Propagation (DTP) for linear head."""
            R_curr, pooled_curr, logits_curr = self.forward(X)
            residual = Y_onehot - logits_curr
            W = self.W_head
            d_in, d_out = W.shape
            if d_in >= d_out:
                gram = W.T @ W
                lam_I = rcls.PROP_LAM * torch.eye(gram.shape[0], dtype=rcls.DTYPE, device=W.device)
                inv_term = torch.linalg.solve(gram + lam_I, W.T)
                delta_pooled = residual @ inv_term
            else:
                gram = W @ W.T
                lam_I = rcls.PROP_LAM * torch.eye(gram.shape[0], dtype=rcls.DTYPE, device=W.device)
                inv_term = torch.linalg.solve(gram + lam_I, W)
                delta_pooled = residual @ inv_term.T
            return R_curr + delta_pooled.unsqueeze(1)

        # 2) W_in trust-region step (identical to the softmax flavour).
        snap_W_in = self.W_in.clone()
        snap_b_in = self.b_in.clone()
        snap_W_head = self.W_head.clone()
        snap_b_head = self.b_head.clone()
        loss_before = self._mse_to_targets(X, Y_onehot)

        R_target_pre = compute_R_target()
        h_curr = self._project_input(X)
        # Linear-attn block uses target_blend=1.0 (no blending in linear regime).
        h_target = self.block.pullback_target(h_curr, R_target_pre, target_blend=1.0)

        X_flat = X.reshape(-1, rcls.TOKEN_DIM)
        H_target_flat = h_target.reshape(-1, rcls.EMBED_DIM)
        gram_in = X_flat.T @ X_flat
        gram_in = gram_in + rcls.DMRG_LAM * torch.eye(
            rcls.TOKEN_DIM, dtype=rcls.DTYPE, device=gram_in.device,
        )
        rhs_in = X_flat.T @ H_target_flat
        self.W_in = torch.linalg.solve(gram_in, rhs_in)
        self.b_in = (H_target_flat - X_flat @ self.W_in).mean(dim=0)
        _, pooled_new, _ = self.forward(X)
        self._fit_head_lsq(pooled_new, Y_onehot)
        loss_after_proj = self._mse_to_targets(X, Y_onehot)
        if loss_after_proj > loss_before:
            self.W_in = snap_W_in
            self.b_in = snap_b_in
            self.W_head = snap_W_head
            self.b_head = snap_b_head
            input_proj_accepted = False
            input_proj_alpha = 0.0
        else:
            input_proj_accepted = True
            input_proj_alpha = 1.0

        # 3) Block DMRG sweep — linear-attn signature has no target_blend.
        R_target = compute_R_target()
        h = self._project_input(X)
        report = self.block.dmrg_step(h, R_target, lam=rcls.DMRG_LAM)

        # 4) Final head re-fit.
        _, pooled_final, _ = self.forward(X)
        self._fit_head_lsq(pooled_final, Y_onehot)

        attn_diag = report["attn"]["diagnostics"]
        return {
            "global_mse_before": report["global_mse_before"],
            "global_mse_after": report["global_mse_after"],
            "input_proj_accepted": input_proj_accepted,
            "input_proj_alpha": float(input_proj_alpha),
            "attn_accepted": bool(report["attn"]["accepted"]),
            "attn_diagnostics": attn_diag,
        }


softmax_runner.TTBlockClassifier = _LinearTTBlockClassifier  # type: ignore[assignment]


# Override the report + sidecar paths so we don't clobber the softmax outputs.
def _patched_main() -> None:
    # Save originals
    import dmrg_transformer.bench._instrumentation as instrumentation_mod

    original_dump = instrumentation_mod.dump_coverage_sidecar
    original_write = Path.write_text

    def patched_dump(name, payload):
        return original_dump("tier2_one_block_linear", payload)

    instrumentation_mod.dump_coverage_sidecar = patched_dump  # type: ignore

    # Redirect the markdown report path. The softmax runner hard-codes
    # ``ROOT / "bench" / "REAL_WORLD_TT_BLOCK.md"``. Patch Path.write_text to
    # rewrite that one path on-the-fly.
    target_softmax = ROOT / "bench" / "REAL_WORLD_TT_BLOCK.md"
    target_linear = ROOT / "bench" / "REAL_WORLD_LIN_TT_BLOCK.md"

    def patched_write(self, *args, **kwargs):
        if self.resolve() == target_softmax.resolve():
            return original_write(target_linear, *args, **kwargs)
        return original_write(self, *args, **kwargs)

    Path.write_text = patched_write  # type: ignore[method-assign]

    try:
        softmax_runner.main()
    finally:
        instrumentation_mod.dump_coverage_sidecar = original_dump  # type: ignore
        Path.write_text = original_write  # type: ignore[method-assign]


if __name__ == "__main__":
    _patched_main()
