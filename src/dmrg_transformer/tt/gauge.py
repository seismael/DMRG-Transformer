"""Gauge management: left/right orthogonalization sweeps (AGENTS.md Phase II).

Each core ``G_k`` of shape ``[r_{k-1}, p_k, r_k]`` is matricized to
``M ∈ R^{(r_{k-1}·p_k) × r_k}``; a QR factorization ``M = Q R`` produces a
left-orthogonal core ``Q`` and a "remnant" ``R`` that is absorbed into the next
core ``G_{k+1}``. Operations mutate the ``TensorTrain`` in place (AGENTS.md
Constraint 4 — no massive reallocation during sweeps).

Per NUMERICAL_STABILITY.md §2, QR is executed in float64.
"""
from __future__ import annotations

import torch

from dmrg_transformer.core.qr import qr_f64, qr_f64_strict
from dmrg_transformer.tt.tensor_train import TensorTrain


def _left_orthogonalize_core(tt: TensorTrain, k: int, *, strict_f64: bool = False) -> None:
    """Left-orthogonalize core ``k`` and push the remnant into core ``k+1``.

    Requires ``k < num_cores - 1``.
    """
    if k < 0 or k >= tt.num_cores - 1:
        raise IndexError(f"cannot left-orthogonalize core {k} (num_cores={tt.num_cores})")
    G_k = tt.get_core(k)
    r_left, p_k, r_right = G_k.shape
    M = G_k.reshape(r_left * p_k, r_right)
    if strict_f64:
        Q, R = qr_f64_strict(M)  # returns float64 tensors
    else:
        Q, R = qr_f64(M)
    new_r = Q.shape[1]
    # In-place replace core k (same memory layout policy; PyTorch tensors are not
    # strictly in-place, but we reuse the TT's list-slot per MEMORY_ARENA §4 intent).
    new_G_k = Q.reshape(r_left, p_k, new_r).to(dtype=G_k.dtype)
    tt.update_core(k, new_G_k)
    # Absorb R into the next core: G_{k+1} <- einsum('rs, s p t -> r p t', R, G_{k+1})
    G_next = tt.get_core(k + 1)
    R_cast = R.to(dtype=G_next.dtype)
    new_G_next = torch.einsum("rs,spt->rpt", R_cast, G_next)
    tt.update_core(k + 1, new_G_next)


def _right_orthogonalize_core(tt: TensorTrain, k: int, *, strict_f64: bool = False) -> None:
    """Right-orthogonalize core ``k`` and push the remnant into core ``k-1``.

    Uses the LQ decomposition implemented as a QR of the transposed matricization.
    Requires ``k > 0``.
    """
    if k <= 0 or k >= tt.num_cores:
        raise IndexError(f"cannot right-orthogonalize core {k} (num_cores={tt.num_cores})")
    G_k = tt.get_core(k)
    r_left, p_k, r_right = G_k.shape
    # Matricize along the left index: M ∈ R^{r_left × (p_k·r_right)}. LQ via QR of M^T.
    M = G_k.reshape(r_left, p_k * r_right)
    if strict_f64:
        Q_T, R_T = qr_f64_strict(M.T)
    else:
        Q_T, R_T = qr_f64(M.T)
    # M = R_T^T · Q_T^T, so L = R_T^T and Q_right = Q_T^T (rows orthonormal).
    L = R_T.T  # (r_left, new_r)
    Q_right = Q_T.T  # (new_r, p_k·r_right)
    new_r = Q_right.shape[0]
    new_G_k = Q_right.reshape(new_r, p_k, r_right).to(dtype=G_k.dtype)
    tt.update_core(k, new_G_k)
    G_prev = tt.get_core(k - 1)
    L_cast = L.to(dtype=G_prev.dtype)
    new_G_prev = torch.einsum("rpq,qs->rps", G_prev, L_cast)
    tt.update_core(k - 1, new_G_prev)


def orthogonalize_left_to(tt: TensorTrain, target: int) -> None:
    """Left-orthogonalize cores ``0..target-1`` so the orthogonality center is at ``target``.

    After this call the merged left environment block is column-orthonormal.
    """
    if target < 0 or target >= tt.num_cores:
        raise IndexError(f"target {target} out of range for {tt.num_cores} cores")
    for k in range(target):
        _left_orthogonalize_core(tt, k)
    tt._set_orth_center(target)  # noqa: SLF001 — intentional internal hook


def orthogonalize_right_to(tt: TensorTrain, target: int) -> None:
    """Right-orthogonalize cores ``target+1..d-1`` so the gauge center is at ``target``."""
    if target < 0 or target >= tt.num_cores:
        raise IndexError(f"target {target} out of range for {tt.num_cores} cores")
    for k in range(tt.num_cores - 1, target, -1):
        _right_orthogonalize_core(tt, k)
    tt._set_orth_center(target)  # noqa: SLF001


def merged_left_block(tt: TensorTrain, k: int) -> torch.Tensor:
    """Return the merged left environment matrix of shape ``(prod(p_0..p_{k-1}), r_{k-1})``.

    For a TT that has been left-orthogonalized up to core ``k``, this matrix
    must satisfy ``L^T L = I_{r_{k-1}}`` (see Validation Gate 2).
    """
    if k <= 0:
        # The trivial boundary block is just the scalar 1.
        G0 = tt.get_core(0)
        return torch.ones((1, 1), dtype=G0.dtype, device=G0.device)
    L = tt.get_core(0).reshape(tt.get_core(0).shape[1], tt.get_core(0).shape[2])
    for idx in range(1, k):
        G = tt.get_core(idx)
        r_left, p_k, r_right = G.shape
        # L: (P_prev, r_left); contract with G: (r_left, p_k, r_right) → (P_prev, p_k, r_right).
        L = L @ G.reshape(r_left, p_k * r_right)
        L = L.reshape(-1, r_right)
    return L


def merged_right_block(tt: TensorTrain, k: int) -> torch.Tensor:
    """Return the merged right environment matrix of shape ``(r_k, prod(p_{k+1}..p_{d-1}))``.

    After right-orthogonalization above ``k``, ``R R^T = I_{r_k}``.
    """
    d = tt.num_cores
    if k >= d - 1:
        G_last = tt.get_core(-1)
        return torch.ones((1, 1), dtype=G_last.dtype, device=G_last.device)
    G_last = tt.get_core(k + 1)
    r_left, p, r_right = G_last.shape
    R = G_last.reshape(r_left, p * r_right)
    for idx in range(k + 2, d):
        G = tt.get_core(idx)
        r_l, p_k, r_r = G.shape
        # R: (r_left, P_prev·r_l). Need to contract r_l with next core's left rank.
        R = R.reshape(-1, r_l) @ G.reshape(r_l, p_k * r_r)
        # R: (r_left·P_prev, p_k·r_r). Reshape to keep r_left on the far left.
    # Return shape (r_k, prod(p_{k+1..d-1}))
    r_k = tt.get_core(k).shape[-1]
    return R.reshape(r_k, -1)
