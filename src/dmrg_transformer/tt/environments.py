"""Environment block computations for the DMRG sweep.

Given a TT factorization of the linear map ``W : R^N -> R^M`` with input
activations ``X : [batch, N]``, we define the "left state" through the first
``k`` cores and the "right-pure" product of the last ``d-k-1`` cores. Combining
them with the focused core ``G_k`` reconstructs ``Y_pred = X @ W``. These blocks
are the operands of the exact local least-squares solver in
``dmrg_transformer.optim.local_solver``.
"""
from __future__ import annotations

from math import prod

import torch

from dmrg_transformer.tt.tensor_train import TensorTrain


def left_state_through(tt: TensorTrain, X: torch.Tensor, k_stop: int) -> torch.Tensor:
    """Contract ``X`` through cores ``0..k_stop-1``.

    Returns a tensor of shape
    ``[batch, J_prefix, r_{k_stop}, i_{k_stop}, i_{k_stop+1}, ..., i_{d-1}]``
    (``J_prefix = prod(output_dims[:k_stop])``). When ``k_stop == 0`` this is
    simply the reshaped input with an inserted boundary rank.
    """
    batch = X.shape[0]
    V = X.reshape(batch, 1, *tt.input_dims)  # [batch, r_0=1, i_0..i_{d-1}]
    out_consumed: list[int] = []
    for k in range(k_stop):
        G = tt.get_core(k)
        i_k = tt.input_dims[k]
        j_k = tt.output_dims[k]
        r_l, p_k, r_r = G.shape
        G4 = G.reshape(r_l, i_k, j_k, r_r)
        J_size = prod(out_consumed) if out_consumed else 1
        I_rest = tt.input_dims[k + 1 :]
        I_rest_size = prod(I_rest) if I_rest else 1
        V_flat = V.reshape(batch, J_size, r_l, i_k, I_rest_size)
        new_V = torch.einsum("bJriI,rijR->bJjRI", V_flat, G4)
        out_consumed.append(j_k)
        V = new_V.reshape(batch, *out_consumed, r_r, *I_rest)
    return V


def right_pure_product(tt: TensorTrain, k_start: int) -> torch.Tensor:
    """Contract cores ``k_start..d-1`` without X.

    Returns a tensor of shape ``[r_{k_start-1}, i_{k_start}, i_{k_start+1}, ...,
    i_{d-1}, j_{k_start}, j_{k_start+1}, ..., j_{d-1}]``. If ``k_start == d``
    returns a tensor ``[r_d=1]`` of ones (the empty product).
    """
    d = tt.num_cores
    G0 = tt.get_core(0)
    dtype = G0.dtype
    device = G0.device
    if k_start >= d:
        return torch.ones((1,), dtype=dtype, device=device)
    
    # Start from the end.
    last = tt.get_core(d - 1)
    r_l, p, _ = last.shape
    i_last = tt.input_dims[d - 1]
    j_last = tt.output_dims[d - 1]
    T = last.reshape(r_l, i_last, j_last)
    
    for k in range(d - 2, k_start - 1, -1):
        G = tt.get_core(k)
        r_l_k, p_k, r_r_k = G.shape
        i_k = tt.input_dims[k]
        j_k = tt.output_dims[k]
        G4 = G.reshape(r_l_k, i_k, j_k, r_r_k)
        
        I_rest_size = prod(tt.input_dims[k + 1 : d])
        J_rest_size = prod(tt.output_dims[k + 1 : d])
        T_flat = T.reshape(r_r_k, I_rest_size, J_rest_size)
        T_new = torch.einsum("aijr,rIJ->aiIjJ", G4, T_flat)
        
        I_rest_dims = tt.input_dims[k + 1 : d]
        J_rest_dims = tt.output_dims[k + 1 : d]
        T = T_new.reshape(r_l_k, i_k, *I_rest_dims, j_k, *J_rest_dims)
    return T


class EnvironmentCache:
    """Incremental cache for DMRG environment blocks (REVIEW.md Issue A)."""

    def __init__(self, tt: TensorTrain, X: torch.Tensor) -> None:
        self.tt = tt
        self.X = X
        self.d = tt.num_cores
        self.left_envs: list[torch.Tensor | None] = [None] * (self.d + 1)
        self.right_envs: list[torch.Tensor | None] = [None] * (self.d + 1)

    def get_left(self, k: int) -> torch.Tensor:
        """Get or compute the left environment through cores 0..k-1."""
        if self.left_envs[k] is not None:
            return self.left_envs[k]  # type: ignore
        if k == 0:
            batch = self.X.shape[0]
            val = self.X.reshape(batch, 1, *self.tt.input_dims)
            self.left_envs[0] = val
            return val
        
        # Recursive compute using get_left(k-1).
        L_prev = self.get_left(k - 1)
        G_prev = self.tt.get_core(k - 1)
        r_l, p_prev, r_r = G_prev.shape
        i_k_prev = self.tt.input_dims[k - 1]
        j_k_prev = self.tt.output_dims[k - 1]
        G4 = G_prev.reshape(r_l, i_k_prev, j_k_prev, r_r)

        batch = self.X.shape[0]
        J_pre = prod(self.tt.output_dims[: k - 1])
        I_rest_next_size = prod(self.tt.input_dims[k:])
        
        # L_prev shape: [batch, J_pre, r_l, i_k_prev, i_k...i_d]
        V_flat = L_prev.reshape(batch, J_pre, r_l, i_k_prev, I_rest_next_size)
        new_V = torch.einsum("bJriI,rijR->bJjRI", V_flat, G4)
        
        out_dims = self.tt.output_dims[:k]
        I_rest_dims = self.tt.input_dims[k:]
        val = new_V.reshape(batch, *out_dims, r_r, *I_rest_dims)
        self.left_envs[k] = val
        return val

    def get_right(self, k: int) -> torch.Tensor:
        """Get or compute the right-pure product of cores k..d-1."""
        if self.right_envs[k] is not None:
            return self.right_envs[k]  # type: ignore
        if k >= self.d:
            val = torch.ones((1,), dtype=self.tt.get_core(0).dtype,
                             device=self.tt.get_core(0).device)
            self.right_envs[k] = val
            return val
        
        if k == self.d - 1:
            last = self.tt.get_core(self.d - 1)
            r_l, p, _ = last.shape
            val = last.reshape(r_l, self.tt.input_dims[-1], self.tt.output_dims[-1])
            self.right_envs[k] = val
            return val
            
        # Recursive compute using get_right(k+1).
        R_next = self.get_right(k + 1)
        G_k = self.tt.get_core(k)
        r_l_k, p_k, r_r_k = G_k.shape
        i_k = self.tt.input_dims[k]
        j_k = self.tt.output_dims[k]
        G4 = G_k.reshape(r_l_k, i_k, j_k, r_r_k)

        I_rest_size = prod(self.tt.input_dims[k + 1:])
        J_rest_size = prod(self.tt.output_dims[k + 1:])
        
        T_flat = R_next.reshape(r_r_k, I_rest_size, J_rest_size)
        T_new = torch.einsum("aijr,rIJ->aiIjJ", G4, T_flat)
        
        I_rest_dims = self.tt.input_dims[k + 1:]
        J_rest_dims = self.tt.output_dims[k + 1:]
        val = T_new.reshape(r_l_k, i_k, *I_rest_dims, j_k, *J_rest_dims)
        self.right_envs[k] = val
        return val

    def invalidate_left(self, k: int) -> None:
        """Invalidate left cache from core k onwards."""
        for i in range(max(0, k), self.d + 1):
            self.left_envs[i] = None

    def invalidate_right(self, k: int) -> None:
        """Invalidate right cache from 0 up to core k."""
        for i in range(min(k + 1, self.d + 1)):
            self.right_envs[i] = None
