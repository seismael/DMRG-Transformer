"""B3: End-to-end TT-GPT2 LM training with frozen-head + block DMRG.

Key insight from B2: alternating head+blocks fitting creates a Nash
equilibrium.  Solution: freeze the LM head during block training, only
re-fit between passes.
"""
from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset

from dmrg_transformer.core.device import require_cuda
from dmrg_transformer.nn.tt_gpt2 import TTDecoderBlock, tt_gpt2_pico
from dmrg_transformer.propagation.target_propagator import TargetPropagator


class _SynthLMData(Dataset):
    """Random data with hidden linear rule for next-token prediction."""
    def __init__(self, seq_len=32, n_samples=300, vocab_size=16):
        self.seq_len = seq_len; self.vocab_size = vocab_size
        torch.manual_seed(42)
        W = torch.randn(vocab_size, vocab_size) * 0.5
        data = [torch.randint(1, vocab_size, (1,)).item()]
        for _ in range(n_samples * seq_len):
            logits = W[data[-1]]
            data.append(max(1, min(vocab_size - 1, logits.argmax().item())))
        self.data = torch.tensor(data, dtype=torch.long)
    def __len__(self): return max(0, (len(self.data) - 1) // self.seq_len)
    def __getitem__(self, idx):
        s = idx * self.seq_len
        return self.data[s:s+self.seq_len], self.data[s+1:s+self.seq_len+1]


def _perplexity(model, dl, device, V):
    total_loss = 0.0; total_tokens = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        hidden = model.model.forward(x)
        logits = hidden.reshape(-1, hidden.shape[-1]) @ model.lm_head.weight.T
        total_loss += torch.nn.functional.cross_entropy(
            logits, y.reshape(-1).long(), reduction="sum",
        ).item()
        total_tokens += y.numel()
    return math.exp(total_loss / max(total_tokens, 1))


def test_frozen_head_training_reduces_perplexity() -> None:
    """Frozen-head + block DMRG training must reduce LM perplexity."""
    device = require_cuda()
    ds_train = _SynthLMData(seq_len=24, n_samples=300, vocab_size=16)
    ds_val = _SynthLMData(seq_len=24, n_samples=80, vocab_size=16)
    dl_train = DataLoader(ds_train, batch_size=4, shuffle=False, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=4, shuffle=False, drop_last=True)
    V = ds_train.vocab_size

    model = tt_gpt2_pico(vocab_size=V, dtype=torch.float32).to(device)
    prop = TargetPropagator(lam=1e-2)
    D = model.model.embed_dim
    L = len(model.model.blocks)
    ln_f = model.model.ln_f

    # ── Initial perplexity (random head, random blocks) ─────────────────
    ppl_init = _perplexity(model, dl_val, device, V)

    # ── Outer loop: head fit → block training → repeat ─────────────────
    for _pass in range(3):
        # 1. Fit head on all training data.
        all_h, all_t = [], []
        for x, y in dl_train:
            x = x.to(device)
            all_h.append(model.model.forward(x).reshape(-1, D))
            all_t.append(y.reshape(-1))
        model.fit_head(torch.cat(all_h).to(device), torch.cat(all_t).to(device))
        head_W = model.lm_head.weight.T.clone()  # [D, V] — frozen copy

        # 2. Train blocks with frozen head.
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            hidden = model.model.forward(x)

            # One-hot target in logit space.
            Y_oh = torch.zeros(x.numel(), V, dtype=hidden.dtype, device=device)
            Y_oh.scatter_(1, y.reshape(-1).unsqueeze(-1), 1.0)
            # Pull through frozen head → hidden target.
            hidden_target_flat = prop.project_through_linear(head_W, Y_oh)
            hidden_target = hidden_target_flat.reshape(x.shape[0], x.shape[1], D)

            # Invert final LayerNorm.
            mu = hidden.mean(dim=-1, keepdim=True)
            sigma = torch.sqrt(hidden.var(dim=-1, keepdim=True, unbiased=False) + ln_f.eps)
            block_target = hidden_target * sigma + mu

            # Intermediate activations.
            emb = model.model.token_embedding(x)
            pos = model.model.positional(emb)
            h_vals = [emb + pos]
            h_curr = emb + pos
            for blk in model.model.blocks:
                h_curr = blk.forward(h_curr)
                h_vals.append(h_curr)

            # Backward block DMRG (last → first).
            target = block_target
            for i in range(L - 1, -1, -1):
                blk = model.model.blocks[i]
                h_in = h_vals[i]
                B, S, _ = h_in.shape

                # FFN sweep.
                cache = blk.forward_with_cache(h_in)
                ffn_target = prop.project_through_residual(target, cache["h"])
                blk.ffn.dmrg_step(
                    cache["ln2"].reshape(-1, D), ffn_target.reshape(-1, D),
                    lam=1e-2, target_blend=0.5,
                )
                # W_out sweep.
                cache_mid = blk.forward_with_cache(h_in)
                h_tgt = 0.5 * (target - cache_mid["ffn_out"]) + 0.5 * cache_mid["h"]
                attn_tgt = prop.project_through_residual(h_tgt, cache_mid["x"])

                H2 = blk.attn.num_heads; dh = blk.attn.head_dim
                xln = cache_mid["ln1"].reshape(-1, D)
                Qc = blk.attn.W_Q(xln).reshape(B, S, H2, dh).transpose(1, 2)
                Kc = blk.attn.W_K(xln).reshape(B, S, H2, dh).transpose(1, 2)
                Vc = blk.attn.W_V(xln).reshape(B, S, H2, dh).transpose(1, 2)
                sc = torch.einsum("bhqd,bhkd->bhqk", Qc, Kc) * (dh**-0.5)
                mask = torch.triu(torch.ones(S, S, dtype=sc.dtype, device=device), diagonal=1)
                sc = sc.masked_fill(mask.bool(), float("-inf"))
                ctx = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(sc, -1), Vc)
                blk.attn.W_out.dmrg_step(
                    ctx.transpose(1, 2).reshape(B, S, D).reshape(-1, D),
                    attn_tgt.reshape(-1, D), lam=1e-2,
                )
                if i > 0:
                    target = h_in + 0.5 * (target - blk.forward(h_in))

    # ── Final evaluation (re-fit head for fair comparison) ──────────────
    all_h, all_t = [], []
    for x, y in dl_train:
        x = x.to(device)
        all_h.append(model.model.forward(x).reshape(-1, D))
        all_t.append(y.reshape(-1))
    model.fit_head(torch.cat(all_h).to(device), torch.cat(all_t).to(device))
    ppl_final = _perplexity(model, dl_val, device, V)

    assert ppl_final < ppl_init, (
        f"Frozen-head training did not reduce perplexity: "
        f"{ppl_init:.2f} -> {ppl_final:.2f}"
    )
