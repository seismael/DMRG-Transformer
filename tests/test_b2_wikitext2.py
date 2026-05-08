"""B2 integration test: TT-GPT2 decoder block DMRG reduces MSE on LM data."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from dmrg_transformer.core.device import require_cuda
from dmrg_transformer.nn.tt_gpt2 import TTDecoderBlock, tt_gpt2_pico
from dmrg_transformer.propagation.target_propagator import TargetPropagator


class _SynthLMData(Dataset):
    def __init__(self, seq_len: int = 32, n_samples: int = 200, vocab_size: int = 16):
        self.seq_len = seq_len; self.vocab_size = vocab_size
        torch.manual_seed(42)
        data = [torch.randint(1, vocab_size, (1,)).item()]
        W = torch.randn(vocab_size, vocab_size) * 0.5
        for _ in range(n_samples * seq_len):
            logits = W[data[-1]]
            data.append(max(1, min(vocab_size - 1, logits.argmax().item())))
        self.data = torch.tensor(data, dtype=torch.long)
    def __len__(self): return max(0, (len(self.data) - 1) // self.seq_len)
    def __getitem__(self, idx):
        s = idx * self.seq_len
        return self.data[s:s+self.seq_len], self.data[s+1:s+self.seq_len+1]


def test_decoder_block_dmrg_reduces_mse_on_lm_data() -> None:
    """One dmrg_step on a decoder block reduces MSE vs a ground-truth block."""
    device = require_cuda()
    ds = _SynthLMData(seq_len=16, n_samples=100, vocab_size=12)
    dl = DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
    x, _ = next(iter(dl))
    x = x.to(device)
    B, L = x.shape

    embed_dim, num_heads, hidden_dim = 12, 2, 12
    dims = [3, 4]; hdim = [3, 4]

    # Ground-truth block.
    torch.manual_seed(0)
    gt = TTDecoderBlock(
        embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim,
        embed_dims=dims, hidden_dims=hdim, rank=4, dtype=torch.float64,
    ).to(device)
    # Simple embedding for the synthetic tokens.
    emb_gt = torch.nn.Embedding(ds.vocab_size, embed_dim, dtype=torch.float64).to(device)
    with torch.no_grad():
        emb = emb_gt(x)
        Y_gt = gt.forward(emb)

    # Trainable block + embedding.
    torch.manual_seed(100)
    train = TTDecoderBlock(
        embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim,
        embed_dims=dims, hidden_dims=hdim, rank=4, dtype=torch.float64,
    ).to(device)
    emb_tr = torch.nn.Embedding(ds.vocab_size, embed_dim, dtype=torch.float64).to(device)
    emb_tr.load_state_dict(emb_gt.state_dict())  # same embedding

    emb = emb_tr(x)
    init_mse = float(torch.mean((train.forward(emb) - Y_gt) ** 2).item())

    # FFN + W_out sweep (robust path; Q/K skipped due to bilinear pullback fragility).
    prop = TargetPropagator(lam=1e-2)
    cache = train.forward_with_cache(emb)
    ffn_target = prop.project_through_residual(Y_gt, cache["h"])
    train.ffn.dmrg_step(
        cache["ln2"].reshape(-1, embed_dim),
        ffn_target.reshape(-1, embed_dim),
        lam=1e-2, target_blend=0.5,
    )
    cache_mid = train.forward_with_cache(emb)
    h_target = 0.5 * (Y_gt - cache_mid["ffn_out"]) + 0.5 * cache_mid["h"]
    attn_target = prop.project_through_residual(h_target, cache_mid["x"])

    H2 = train.attn.num_heads; d_h2 = train.attn.head_dim
    x_ln1 = cache_mid["ln1"].reshape(-1, embed_dim)
    Qc = train.attn.W_Q(x_ln1).reshape(B, L, H2, d_h2).transpose(1, 2)
    Kc = train.attn.W_K(x_ln1).reshape(B, L, H2, d_h2).transpose(1, 2)
    Vc = train.attn.W_V(x_ln1).reshape(B, L, H2, d_h2).transpose(1, 2)
    sc = torch.einsum("bhqd,bhkd->bhqk", Qc, Kc) * (d_h2**-0.5)
    mask = torch.triu(torch.ones(L, L, dtype=sc.dtype, device=device), diagonal=1)
    sc = sc.masked_fill(mask.bool(), float("-inf"))
    aw = torch.softmax(sc, dim=-1)
    ctx = torch.einsum("bhqk,bhkd->bhqd", aw, Vc)
    ctx_full = ctx.transpose(1, 2).reshape(B, L, embed_dim)
    train.attn.W_out.dmrg_step(
        ctx_full.reshape(-1, embed_dim), attn_target.reshape(-1, embed_dim), lam=1e-2,
    )

    final_mse = float(torch.mean((train.forward(emb) - Y_gt) ** 2).item())
    assert final_mse < init_mse, (
        f"Decoder block FFN+W_out did not reduce MSE: {init_mse:.4e} -> {final_mse:.4e}"
    )


def test_tt_gpt2_pico_forward_shape() -> None:
    """Full pico model produces correct logit shape on LM data."""
    device = require_cuda()
    ds = _SynthLMData(seq_len=16, n_samples=50, vocab_size=12)
    dl = DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
    x, _ = next(iter(dl))
    x = x.to(device)

    model = tt_gpt2_pico(vocab_size=ds.vocab_size, dtype=torch.float32).to(device)
    logits = model.forward(x)
    assert logits.shape == (x.shape[0], x.shape[1], ds.vocab_size)


def test_head_fit_reduces_ce() -> None:
    """Exact LSQ head fitting must reduce cross-entropy on LM data."""
    device = require_cuda()
    ds = _SynthLMData(seq_len=16, n_samples=100, vocab_size=12)
    dl = DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
    x, y = next(iter(dl))
    x, y = x.to(device), y.to(device)

    model = tt_gpt2_pico(vocab_size=ds.vocab_size, dtype=torch.float32).to(device)
    D = model.model.embed_dim; V = ds.vocab_size

    hidden = model.model.forward(x).reshape(-1, D)
    ce_before = float(torch.nn.functional.cross_entropy(
        (hidden @ model.lm_head.weight.T), y.reshape(-1),
    ).item())

    model.fit_head(hidden, y.reshape(-1))
    ce_after = float(torch.nn.functional.cross_entropy(
        (model.model.forward(x).reshape(-1, D) @ model.lm_head.weight.T), y.reshape(-1),
    ).item())

    assert ce_after < ce_before, (
        f"Head fit did not reduce CE: {ce_before:.4f} -> {ce_after:.4f}"
    )
