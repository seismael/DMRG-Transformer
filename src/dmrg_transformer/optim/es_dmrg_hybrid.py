"""ES/DMRG Hybrid — Evolution Strategies for attention Q/K training.

EGGROLL (arXiv 2511.16652) shows that low-rank Gaussian perturbations
achieve 91 % of inference throughput for billion-parameter ES.  We adapt
this to DMRG-Transformer by using ES *only* for the attention Q/K
sub-block — the component where DMRG's Frobenius-minimising target
propagation fails (Exactness Paradox, REVIEW.md §3).

ES evaluates the global fitness (classification accuracy or block MSE)
directly, bypassing the Frobenius→DLTP→bilinear-pullback chain that
produces trust-region-rejected Q/K updates.

Phase 1: DMRG sweeps on FFN + W_out (exact, robust, monotonic).
Phase 2: ES perturbation on Q/K (global fitness, no Frobenius targets).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from dmrg_transformer.propagation.target_propagator import TargetPropagator

if TYPE_CHECKING:
    from dmrg_transformer.nn.tt_linear import TTLinear


@dataclass
class ESRoundReport:
    """Bookkeeping for one ES perturbation round."""
    fitness_before: float
    fitness_after: float
    accepted: bool
    sigma: float
    population_size: int
    best_perturbation_fitness: float


class ESDMRGHybrid:
    """Hybrid trainer: DMRG for FFN/W_out, ES for Q/K.

    Args:
        block: a ``TTBlock`` or ``TTDecoderBlock`` to train.
        population_size: number of random perturbations per ES round.
        sigma: standard deviation of Gaussian perturbations.
        lr: learning rate (step size) for the weighted ES update.
        lam: Tikhonov damping for DMRG sweeps.
    """

    def __init__(
        self,
        block: nn.Module,
        *,
        population_size: int = 20,
        sigma: float = 0.01,
        lr: float = 0.1,
        lam: float = 1e-2,
    ) -> None:
        self.block = block
        self.population_size = int(population_size)
        self.sigma = float(sigma)
        self.lr = float(lr)
        self.lam = float(lam)
        self.propagator = TargetPropagator(lam=lam)

    # ------------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------------

    def _fitness(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Negative MSE: higher fitness = better fit."""
        with torch.no_grad():
            y = self.block.forward(X)
            mse = float(torch.mean((y - Y) ** 2).item())
        return -mse

    # ------------------------------------------------------------------
    # ES perturbation on Q/K
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _es_round_qk(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> ESRoundReport:
        """One round of ES on the Q and K projections only."""
        attn = self.block.attn  # TTMultiHeadAttention or TTCausalSelfAttention
        fitness_before = self._fitness(X, Y)

        # Sample perturbations and evaluate.
        noises_Q: list[dict[str, torch.Tensor]] = []
        noises_K: list[dict[str, torch.Tensor]] = []
        fitnesses: list[float] = []

        for _ in range(self.population_size):
            nQ = _sample_tt_noise(attn.W_Q, self.sigma)
            nK = _sample_tt_noise(attn.W_K, self.sigma)
            noises_Q.append(nQ)
            noises_K.append(nK)

            # Apply perturbation.
            _apply_tt_noise(attn.W_Q, nQ)
            _apply_tt_noise(attn.W_K, nK)

            fitnesses.append(self._fitness(X, Y))

            # Revert.
            _revert_tt_noise(attn.W_Q, nQ)
            _revert_tt_noise(attn.W_K, nK)

        # Weighted ES update: Δθ = (lr / (N * σ²)) Σ f_i · ε_i
        fitness_t = torch.tensor(fitnesses, dtype=torch.float64, device=X.device)
        # Rank-based transform for robustness (natural gradient approximation).
        _, ranks = fitness_t.sort()
        weights = (ranks.float() - (self.population_size - 1) / 2) / (self.population_size - 1)
        weights = weights / (weights.abs().sum() + 1e-8)

        scale = self.lr / (self.sigma * self.sigma)
        for nQ, nK, w in zip(noises_Q, noises_K, weights):
            w_val = float(w.item()) * scale
            _apply_tt_noise(attn.W_Q, nQ, scale=w_val)
            _apply_tt_noise(attn.W_K, nK, scale=w_val)

        fitness_after = self._fitness(X, Y)
        accepted = fitness_after > fitness_before
        best = max(fitnesses)

        return ESRoundReport(
            fitness_before=fitness_before,
            fitness_after=fitness_after,
            accepted=accepted,
            sigma=self.sigma,
            population_size=self.population_size,
            best_perturbation_fitness=best,
        )

    # ------------------------------------------------------------------
    # Full hybrid step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        X: torch.Tensor,
        Y_target: torch.Tensor,
        *,
        es_rounds: int = 1,
    ) -> dict[str, object]:
        """One hybrid training step.

        1. DMRG sweep on FFN (exact, robust).
        2. DMRG sweep on W_out (exact, robust).
        3. ES round(s) on Q/K (global fitness optimisation).
        """
        B, L, D = X.shape
        cache = self.block.forward_with_cache(X)

        # ── DMRG: FFN ─────────────────────────────────────────────────
        ffn_target = self.propagator.project_through_residual(Y_target, cache["h"])
        ffn_reports = self.block.ffn.dmrg_step(
            cache["h_ln2"].reshape(-1, D), ffn_target.reshape(-1, D),
            lam=self.lam, target_blend=0.5,
        )

        # ── DMRG: W_out ───────────────────────────────────────────────
        cache_mid = self.block.forward_with_cache(X)
        h_target = 0.5 * (Y_target - cache_mid["ffn_out"]) + 0.5 * cache_mid["h"]
        attn_target = self.propagator.project_through_residual(h_target, cache_mid["x"])

        # Use cached pre-LN activations (available in TTBlock, not TTMultiHeadAttention).
        if "x_ln1" in cache_mid:
            x_ln1 = cache_mid["x_ln1"].reshape(-1, D)
        else:
            x_ln1 = self.block.ln1(X).reshape(-1, D)

        attn = self.block.attn
        H = attn.num_heads; dh = attn.head_dim
        Qc = attn.W_Q(x_ln1).reshape(B, L, H, dh).transpose(1, 2)
        Kc = attn.W_K(x_ln1).reshape(B, L, H, dh).transpose(1, 2)
        Vc = attn.W_V(x_ln1).reshape(B, L, H, dh).transpose(1, 2)
        sc = torch.einsum("bhqd,bhkd->bhqk", Qc, Kc) * (dh**-0.5)

        # Causal mask for decoder blocks.
        if hasattr(attn, "_causal") or type(attn).__name__ == "TTCausalSelfAttention":
            mask = torch.triu(torch.ones(L, L, dtype=sc.dtype, device=X.device), diagonal=1)
            sc = sc.masked_fill(mask.bool(), float("-inf"))

        aw = torch.softmax(sc, dim=-1)
        ctx = torch.einsum("bhqk,bhkd->bhqd", aw, Vc)
        ctx_full = ctx.transpose(1, 2).reshape(B, L, D)
        wout_rep = attn.W_out.dmrg_step(
            ctx_full.reshape(-1, D), attn_target.reshape(-1, D), lam=self.lam,
        )

        # ── ES: Q/K ───────────────────────────────────────────────────
        es_reports = []
        for _ in range(es_rounds):
            es_reports.append(self._es_round_qk(X, Y_target))

        return {
            "ffn": ffn_reports,
            "wout": wout_rep.final_mse,
            "es": es_reports,
            "es_accepted": any(r.accepted for r in es_reports),
        }


# ═══════════════════════════════════════════════════════════════════════════
# TT noise utilities (low-rank Gaussian perturbations on TT cores)
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def _sample_tt_noise(
    tt_linear: TTLinear, sigma: float,
) -> dict[str, torch.Tensor]:
    """Sample independent Gaussian noise for each TT core.

    Returns a dict ``{core_name: noise_tensor}`` for later application.
    """
    noise = {}
    for k in range(tt_linear._num_cores):
        core = getattr(tt_linear, f"_core_{k}")
        noise[f"_core_{k}"] = sigma * torch.randn_like(core)
    return noise


@torch.no_grad()
def _apply_tt_noise(
    tt_linear: TTLinear,
    noise: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> None:
    """Add scaled noise to each TT core in-place."""
    for name, n in noise.items():
        core = getattr(tt_linear, name)
        core.add_(scale * n)


@torch.no_grad()
def _revert_tt_noise(
    tt_linear: TTLinear,
    noise: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> None:
    """Subtract scaled noise from each TT core in-place."""
    for name, n in noise.items():
        core = getattr(tt_linear, name)
        core.sub_(scale * n)
