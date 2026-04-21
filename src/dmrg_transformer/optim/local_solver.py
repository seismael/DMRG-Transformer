"""Exact local least-squares solver for a single TT-core (AGENTS.md Phase III).

Given a TT representing ``W : R^N -> R^M``, activation ``X : [batch, N]`` and
target ``Y : [batch, M]``, this module computes the core ``G_k*`` that
minimizes the strictly convex sub-problem

    L(G_k) = ½ || Y - X · W(G_k) ||_F² + λ/2 || G_k ||_F²    (Tikhonov, NUMERICAL_STABILITY §3)

via explicit construction of the environment-derived Jacobian and solution of
the resulting linear system. The regularization term ``λ`` is NUMERICAL_STABILITY
§3's Tikhonov damping. After solving, the updated core is truncated via SVD
(Eckart–Young–Mirsky) per TENSOR_TOPOLOGY.md §6.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from dmrg_transformer.core.svd import robust_svd, truncate
from dmrg_transformer.tt.environments import left_state_through, right_pure_product
from dmrg_transformer.tt.tensor_train import TensorTrain

_HUBER_CLAMP_SIGMAS: float = 5.0


@dataclass(frozen=True)
class LocalSolveResult:
    """Output of a single local-core update."""

    U: torch.Tensor      # new left-orthogonal left half of the updated core
    S: torch.Tensor      # singular values (post-truncation)
    Vh: torch.Tensor     # right factor to be absorbed into core k+1 (gauge shift)
    residual_mse: float  # post-update MSE on the provided batch
    lam_used: float      # Tikhonov λ actually applied (may be auto-escalated on NaN)


def _huber_clamp(target: torch.Tensor) -> torch.Tensor:
    """NUMERICAL_STABILITY §5: clamp target entries to ±5σ from per-batch mean.

    This is the mathematical substitute for gradient clipping when the exact
    solver is used (there is no learning-rate to dampen outliers).
    """
    mean = target.mean(dim=0, keepdim=True)
    std = target.std(dim=0, keepdim=True).clamp_min(1.0e-12)
    return torch.clamp(target, min=mean - _HUBER_CLAMP_SIGMAS * std,
                       max=mean + _HUBER_CLAMP_SIGMAS * std)


def _build_jacobian(tt: TensorTrain, X: torch.Tensor, k: int) -> torch.Tensor:
    """Build the Jacobian ``J`` such that ``Y_pred.flatten(start_dim=1) = J @ vec(G_k)``.

    Returns a tensor of shape ``[batch, M, r_left * p_k * r_right]``.
    """
    d = tt.num_cores
    G_k = tt.get_core(k)
    r_l, p_k, r_r = G_k.shape
    i_k = tt.input_dims[k]
    j_k = tt.output_dims[k]

    # Left state: contract cores 0..k-1 with X.
    # Shape: [batch, *out_consumed (j_0..j_{k-1}), r_{k-1}, i_k, i_{k+1}..i_{d-1}]
    left = left_state_through(tt, X, k_stop=k)

    # Right pure product of cores k+1..d-1 (no X).
    # Shape: [r_k, i_{k+1}..i_{d-1}, j_{k+1}..j_{d-1}]  (empty for k = d-1).
    right = right_pure_product(tt, k_start=k + 1)

    batch = X.shape[0]
    J_pre = prod(tt.output_dims[:k]) if k > 0 else 1
    I_suf = prod(tt.input_dims[k + 1 : d]) if k + 1 < d else 1
    J_suf = prod(tt.output_dims[k + 1 : d]) if k + 1 < d else 1

    # Flatten left to [batch, J_pre, r_l, i_k, I_suf] and right to [r_r, I_suf, J_suf].
    left_flat = left.reshape(batch, J_pre, r_l, i_k, I_suf)
    right_flat = right.reshape(r_r, I_suf, J_suf)

    # Jacobian entry: Y_pred[b, j_pre, j_k_out, j_suf] = Σ_{r_l, r_r, i_k, I_suf}
    #   left[b, j_pre, r_l, i_k, I_suf] * G_k[r_l, i_k, j_k_out, r_r] * right[r_r, I_suf, j_suf]
    # ∂ / ∂ G_k[a, i_k', j_k', b']:
    #   δ(j_k_out, j_k') · Σ_{I_suf} left[b, j_pre, a, i_k', I_suf] · right[b', I_suf, j_suf]
    # Build  jac_core[b, j_pre, a, i_k', j_suf, b'] =
    #   Σ_{I_suf} left[b, j_pre, a, i_k', I_suf] · right[b', I_suf, j_suf]
    jac_core = torch.einsum("bPaiI,RIQ->bPaiQR", left_flat, right_flat)
    # Now insert the δ(j_k_out, j_k') by constructing an explicit identity.
    # Final Jacobian shape: [batch, J_pre, j_k_out, j_suf, r_l, p_k=(i_k·j_k'), r_r]
    # where p_k is organized as (i_k', j_k') row-major.
    eye_j = torch.eye(j_k, dtype=jac_core.dtype, device=jac_core.device)
    # jac: [b, j_pre, j_k_out, j_suf, r_l, i_k', j_k', r_r]
    #  = jac_core[b, j_pre, r_l, i_k', j_suf, r_r] * eye_j[j_k_out, j_k']
    jac = torch.einsum("bPaiQR,JK->bPJQaiKR", jac_core, eye_j)
    # Reshape: Y axes (j_pre, j_k_out, j_suf) into M; G_k axes (r_l, i_k', j_k', r_r) into r_l*p_k*r_r.
    batch_dim = jac.shape[0]
    # Current shape: (b, J_pre, j_k_out, J_suf, r_l, i_k', j_k', r_r)
    # Collapse (J_pre, j_k_out, J_suf) -> M ; (r_l, i_k', j_k', r_r) -> P
    M = tt.out_features
    P = r_l * p_k * r_r
    jac = jac.reshape(batch_dim, M, P)
    return jac


def solve_local_core(
    tt: TensorTrain,
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int,
    *,
    max_rank: int,
    lam: float = 0.0,
    direction: str = "left",
    clamp_target: bool = True,
) -> LocalSolveResult:
    """Exact least-squares + SVD-truncated update for core ``k``.

    Args:
        tt: the tensor train (mutated in place).
        X, Y: activation and target of shape ``[batch, N]`` and ``[batch, M]``.
        k: index of the focused core.
        max_rank: Eckart–Young–Mirsky truncation rank.
        lam: Tikhonov damping factor (NUMERICAL_STABILITY §3).
        direction: ``"left"`` (L→R sweep, absorb Vh into core k+1) or ``"right"``.
        clamp_target: apply Huber clamp to ``Y`` (NUMERICAL_STABILITY §5).

    Returns:
        :class:`LocalSolveResult` describing the SVD factors produced for this
        core; the caller handles gauge shifting to the next core.
    """
    if direction not in ("left", "right"):
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

    G_k = tt.get_core(k)
    r_l, p_k, r_r = G_k.shape

    target = _huber_clamp(Y) if clamp_target else Y

    # Build Jacobian and flatten: [(batch*M), P].
    jac = _build_jacobian(tt, X, k)                        # [batch, M, P]
    batch = jac.shape[0]
    J = jac.reshape(batch * tt.out_features, r_l * p_k * r_r)
    rhs = target.reshape(batch * tt.out_features)

    # Tikhonov-damped normal equations: (J^T J + λ I) vec(G_k) = J^T rhs.
    # NaN-escalation loop per NUMERICAL_STABILITY §3.
    current_lam = float(lam)
    attempts = 0
    while True:
        JtJ = J.T @ J
        if current_lam > 0.0:
            JtJ = JtJ + current_lam * torch.eye(JtJ.shape[0], dtype=JtJ.dtype, device=JtJ.device)
        Jty = J.T @ rhs
        # Cholesky/solve; fall back to pinv if the system is numerically singular.
        try:
            vec_new = torch.linalg.solve(JtJ, Jty)
        except (torch._C._LinAlgError, RuntimeError):  # type: ignore[attr-defined]
            vec_new = torch.linalg.pinv(JtJ) @ Jty
        if torch.isfinite(vec_new).all():
            break
        attempts += 1
        if attempts > 6:
            raise RuntimeError(
                f"solve_local_core: NaN persisted after 6 λ escalations (λ={current_lam})"
            )
        current_lam = max(current_lam, 1.0e-12) * 10.0

    G_new = vec_new.reshape(r_l, p_k, r_r)

    # SVD truncation per TENSOR_TOPOLOGY §6: matricize, SVD, truncate.
    if direction == "left":
        # Matricize along the left index: M = [r_l * p_k, r_r].
        C = G_new.reshape(r_l * p_k, r_r)
        full = robust_svd(C)
        trunc = truncate(full, max_rank)
        if k + 1 < tt.num_cores:
            # Gauge shift: keep U as the new left-orthogonal core, push S·Vh right.
            U = trunc.U.reshape(r_l, p_k, trunc.S.shape[0])
            tt.update_core(k, U.to(dtype=G_k.dtype))
            SVh = (trunc.S.unsqueeze(1) * trunc.Vh).to(dtype=G_k.dtype)
            G_next = tt.get_core(k + 1)
            new_G_next = torch.einsum("rs,spt->rpt", SVh, G_next)
            tt.update_core(k + 1, new_G_next)
        else:
            # Terminal core: no neighbour to absorb the remnant → store U·S·Vh
            # (rank-truncated reconstruction) directly. Eckart–Young still holds.
            reconstructed = trunc.U * trunc.S.unsqueeze(0)
            reconstructed = reconstructed @ trunc.Vh  # (r_l*p_k, r_r)
            new_core = reconstructed.reshape(r_l, p_k, r_r).to(dtype=G_k.dtype)
            tt.update_core(k, new_core)
        result_U = trunc.U
        result_S = trunc.S
        result_Vh = trunc.Vh
    else:  # direction == "right"
        # Matricize along the right index: M = [r_l, p_k * r_r].
        C = G_new.reshape(r_l, p_k * r_r)
        full = robust_svd(C)
        trunc = truncate(full, max_rank)
        if k - 1 >= 0:
            V_core = trunc.Vh.reshape(trunc.S.shape[0], p_k, r_r)
            tt.update_core(k, V_core.to(dtype=G_k.dtype))
            U_S = (trunc.U * trunc.S.unsqueeze(0)).to(dtype=G_k.dtype)
            G_prev = tt.get_core(k - 1)
            new_G_prev = torch.einsum("rpq,qs->rps", G_prev, U_S)
            tt.update_core(k - 1, new_G_prev)
        else:
            reconstructed = trunc.U * trunc.S.unsqueeze(0)
            reconstructed = reconstructed @ trunc.Vh  # (r_l, p_k*r_r)
            new_core = reconstructed.reshape(r_l, p_k, r_r).to(dtype=G_k.dtype)
            tt.update_core(k, new_core)
        result_U = trunc.U
        result_S = trunc.S
        result_Vh = trunc.Vh

    # Measure residual MSE after update.
    W_hat = tt.to_dense()
    Y_pred = X @ W_hat
    residual_mse = float(torch.mean((Y_pred - Y) ** 2).item())

    return LocalSolveResult(
        U=result_U, S=result_S, Vh=result_Vh,
        residual_mse=residual_mse, lam_used=current_lam,
    )
