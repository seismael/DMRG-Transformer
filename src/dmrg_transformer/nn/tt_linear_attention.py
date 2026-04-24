"""``TTLinearAttention`` — multi-head linear attention with TT-factorized projections.

Replaces the softmax kernel ``softmax(Q Kᵀ/√d) V`` with a feature-map kernel:

        φ(x) = elu(x) + 1                              (Katharopoulos et al., 2020)

        out_q = ( φ(Q_q) · Σ_k φ(K_k) V_kᵀ ) / ( φ(Q_q) · Σ_k φ(K_k) + ε )

Why this matters for DMRG
-------------------------
In softmax attention the V-update target arrives via three successive
pinv pull-backs (W_out → residual → softmax-aware A·V inversion). Each
pinv injects approximation error and the V LSQ objective stops being
aligned with the global MSE — see ``bench/PHASE0_DIAGNOSTIC.md``.

In linear attention the attention-V composition is *multilinear* in the
projection cores once φ is fixed: numerator = ``φ(Q)·Sᵀ`` with the
per-batch state ``S = Σ_k φ(K_k) V_kᵀ`` of shape ``[d_h, d_h]``. The V
update therefore reduces to a closed-form LSQ on the global loss directly
(modulo the fixed denominator), and the Target-Propagation Drift hole
identified in REVIEW.md / Phase 0 disappears at the V step.

This module deliberately mirrors :class:`dmrg_transformer.nn.tt_mha.TTMultiHeadAttention`
in shape and constructor signature so it is a drop-in replacement at the
TTBlock level. The DMRG step itself is implemented later in
``TTLinearAttentionBlock``; this module only owns the forward pass and
the per-projection sweeps.
"""
from __future__ import annotations

from math import prod

import torch
from torch import nn

from dmrg_transformer.nn.tt_linear import TTLinear


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Positive feature map ``φ(x) = elu(x) + 1``. Element-wise, range (0, ∞)."""
    return torch.nn.functional.elu(x) + 1.0


class TTLinearAttention(nn.Module):
    """Multi-head linear attention with TT-factorized Q/K/V/W_out projections.

    Args:
        embed_dim: model dimension (``d_model``).
        num_heads: number of attention heads. ``embed_dim`` must be divisible by it.
        input_dims: factorization of ``embed_dim`` for the TT cores.
        output_dims: factorization of ``embed_dim`` for the TT cores.
        rank: TT-rank bound for every projection.
        dtype: storage dtype for TT cores.
        eps: denominator stabilizer.
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
        eps: float = 1.0e-6,
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
        self.eps = float(eps)

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

    # -- forward ---------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Linear-attention forward pass.

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
        H, d_h = self.num_heads, self.head_dim

        Q = self._project(self.W_Q, query).reshape(B, L_q, H, d_h).transpose(1, 2)  # [B,H,L_q,d_h]
        K = self._project(self.W_K, key).reshape(B, L_k, H, d_h).transpose(1, 2)
        V = self._project(self.W_V, value).reshape(B, L_k, H, d_h).transpose(1, 2)

        phiQ = elu_plus_one(Q)
        phiK = elu_plus_one(K)

        # KV state: S[b,h,i,j] = Σ_k phiK[b,h,k,i] * V[b,h,k,j]   shape [B,H,d_h,d_h]
        S = torch.einsum("bhki,bhkj->bhij", phiK, V)
        # Z[b,h,i] = Σ_k phiK[b,h,k,i]                            shape [B,H,d_h]
        Z = phiK.sum(dim=-2)

        # numerator[b,h,q,j] = Σ_i phiQ[b,h,q,i] * S[b,h,i,j]
        num = torch.einsum("bhqi,bhij->bhqj", phiQ, S)
        # denom[b,h,q] = Σ_i phiQ[b,h,q,i] * Z[b,h,i]
        denom = torch.einsum("bhqi,bhi->bhq", phiQ, Z).unsqueeze(-1) + self.eps

        context = num / denom                                              # [B,H,L_q,d_h]
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
        Y_Q: torch.Tensor | None,
        Y_K: torch.Tensor | None,
        Y_V: torch.Tensor | None,
        *,
        lam: float = 1.0e-5,
        adaptive_threshold: float | None = None,
    ) -> dict[str, float]:
        """Per-projection exact-solver updates given layer-local targets.

        Identical contract to :meth:`TTMultiHeadAttention.dmrg_step_projections`
        so block-level orchestration code can dispatch either flavor uniformly.
        Q, K and V projections are each independent TTLinear LSQ problems
        once their targets are fixed; CUDA stream dispatch is reused.
        """
        use_streams = X.is_cuda and torch.cuda.is_available()
        results: dict[str, float] = {}
        jobs: list[tuple[str, TTLinear, torch.Tensor]] = []
        if Y_Q is not None:
            jobs.append(("Q", self.W_Q, Y_Q))
        if Y_K is not None:
            jobs.append(("K", self.W_K, Y_K))
        if Y_V is not None:
            jobs.append(("V", self.W_V, Y_V))
        if not jobs:
            raise ValueError("at least one of Y_Q, Y_K, Y_V must be provided")

        def _run(name: str, proj: TTLinear, target: torch.Tensor) -> None:
            flat_in = X.reshape(-1, X.shape[-1])
            flat_tgt = target.reshape(-1, target.shape[-1])
            report = proj.dmrg_step(
                flat_in, flat_tgt, lam=lam, adaptive_threshold=adaptive_threshold,
            )
            results[name] = report.final_mse

        if use_streams:
            streams = [torch.cuda.Stream() for _ in range(len(jobs))]
            producer = torch.cuda.current_stream()
            for stream, (name, proj, tgt) in zip(streams, jobs, strict=True):
                stream.wait_stream(producer)
                with torch.cuda.stream(stream):
                    _run(name, proj, tgt)
            for stream in streams:
                producer.wait_stream(stream)
            torch.cuda.synchronize()
        else:
            for name, proj, tgt in jobs:
                _run(name, proj, tgt)
        return results
