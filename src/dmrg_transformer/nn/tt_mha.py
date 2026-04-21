"""``TTMultiHeadAttention`` — MHA with TT-factorized projection matrices.

Each of ``W_Q``, ``W_K``, ``W_V``, ``W_out`` is an independent :class:`TTLinear`.
Per ARCHITECTURE.md §6 / MEMORY_ARENA.md §5, individual heads are mathematically
independent and their DMRG sweeps are dispatched to separate CUDA streams when
available (the Python analog of the Rust microkernel's per-head stream pool).
"""
from __future__ import annotations

from math import prod

import torch
from torch import nn

from dmrg_transformer.nn.tt_linear import TTLinear


class TTMultiHeadAttention(nn.Module):
    """Multi-Head Attention with TT-factorized projection weights.

    Args:
        embed_dim: model dimension (``d_model``).
        num_heads: number of attention heads. ``embed_dim`` must be divisible by it.
        input_dims: factorization of ``embed_dim`` for the TT cores.
        output_dims: factorization of ``embed_dim`` for the TT cores.
        rank: TT-rank bound for every projection.
        dtype: storage dtype for TT cores.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        input_dims: list[int],
        output_dims: list[int],
        rank: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        if prod(input_dims) != embed_dim:
            raise ValueError(f"prod(input_dims)={prod(input_dims)} != embed_dim={embed_dim}")
        if prod(output_dims) != embed_dim:
            raise ValueError(f"prod(output_dims)={prod(output_dims)} != embed_dim={embed_dim}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rank = rank

        def make_proj() -> TTLinear:
            return TTLinear(
                embed_dim, embed_dim,
                input_dims=input_dims, output_dims=output_dims,
                rank=rank, bias=True, dtype=dtype,
            )

        self.W_Q = make_proj()
        self.W_K = make_proj()
        self.W_V = make_proj()
        self.W_out = make_proj()

    # -- forward -----------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scaled dot-product attention.

        Shapes:
            query, key, value: ``[batch, seq, embed_dim]``. ``key`` and ``value``
            default to ``query`` (self-attention).
        """
        if key is None:
            key = query
        if value is None:
            value = query
        B, L_q, _ = query.shape
        L_k = key.shape[1]

        Q = self._project(self.W_Q, query).reshape(B, L_q, self.num_heads, self.head_dim)
        K = self._project(self.W_K, key).reshape(B, L_k, self.num_heads, self.head_dim)
        V = self._project(self.W_V, value).reshape(B, L_k, self.num_heads, self.head_dim)
        # Transpose to [B, H, L, d_h].
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        scale = self.head_dim**-0.5
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) * scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhqk,bhkd->bhqd", attn, V)
        context = context.transpose(1, 2).reshape(B, L_q, self.embed_dim)
        return self._project(self.W_out, context)

    @staticmethod
    def _project(proj: TTLinear, x: torch.Tensor) -> torch.Tensor:
        """Apply a TTLinear over ``[..., embed_dim]`` inputs."""
        flat = x.reshape(-1, x.shape[-1])
        y = proj(flat)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    # -- DMRG update --------------------------------------------------------------

    @torch.no_grad()
    def dmrg_step_projections(
        self,
        X: torch.Tensor,
        Y_Q: torch.Tensor,
        Y_K: torch.Tensor,
        Y_V: torch.Tensor,
        *,
        lam: float = 1.0e-5,
    ) -> dict[str, float]:
        """Run per-projection exact-solver updates given layer-local targets.

        The four projections are mathematically independent and are dispatched
        to distinct CUDA streams when a GPU is available (MEMORY_ARENA §5
        Python analog).
        """
        use_streams = X.is_cuda and torch.cuda.is_available()
        results: dict[str, float] = {}

        def _run(name: str, proj: TTLinear, target: torch.Tensor) -> None:
            flat_in = X.reshape(-1, X.shape[-1])
            flat_tgt = target.reshape(-1, target.shape[-1])
            report = proj.dmrg_step(flat_in, flat_tgt, lam=lam)
            results[name] = report.final_mse

        if use_streams:
            streams = [torch.cuda.Stream() for _ in range(3)]
            for stream, name, proj, tgt in zip(
                streams, ("Q", "K", "V"), (self.W_Q, self.W_K, self.W_V),
                (Y_Q, Y_K, Y_V), strict=True,
            ):
                with torch.cuda.stream(stream):
                    _run(name, proj, tgt)
            torch.cuda.synchronize()
        else:
            _run("Q", self.W_Q, Y_Q)
            _run("K", self.W_K, Y_K)
            _run("V", self.W_V, Y_V)
        return results
