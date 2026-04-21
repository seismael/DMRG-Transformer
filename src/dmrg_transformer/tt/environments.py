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
    dtype = tt.get_core(0).dtype
    if k_start >= d:
        return torch.ones((1,), dtype=dtype)
    # Start with last core: shape (r_{d-1}, i_d, j_d, 1) flatten trailing rank-1.
    last = tt.get_core(d - 1)
    r_l, p, _ = last.shape
    i_last = tt.input_dims[d - 1]
    j_last = tt.output_dims[d - 1]
    # T: (r_{d-1}, i_d, j_d)
    T = last.reshape(r_l, i_last, j_last)
    # Iterate from d-2 down to k_start.
    for k in range(d - 2, k_start - 1, -1):
        G = tt.get_core(k)
        r_l_k, p_k, r_r_k = G.shape
        i_k = tt.input_dims[k]
        j_k = tt.output_dims[k]
        G4 = G.reshape(r_l_k, i_k, j_k, r_r_k)
        # T currently has shape (r_{k+1}, i_{k+1}..i_{d-1}, j_{k+1}..j_{d-1}).
        # Flatten to (r_{k+1}, I_rest_size, J_rest_size).
        # We want to combine with G4 (r_l_k, i_k, j_k, r_{k+1}) to produce:
        #   T' (r_l_k, i_k, i_{k+1}..i_{d-1}, j_k, j_{k+1}..j_{d-1})
        num_i_rest = (d - 1) - k  # number of i axes to the right of current k in T
        # But T's axes layout is (r_{k+1}, i's..., j's...). Let's compute flattened sizes.
        I_rest_size = prod(tt.input_dims[k + 1 : d])
        J_rest_size = prod(tt.output_dims[k + 1 : d])
        T_flat = T.reshape(r_r_k, I_rest_size, J_rest_size)
        # einsum: G4[a, i_k, j_k, r] · T_flat[r, Irest, Jrest] -> (a, i_k, Irest, j_k, Jrest)
        T_new = torch.einsum("aijr,rIJ->aiIjJ", G4, T_flat)
        # Reshape back to (r_l_k, i_k, i_{k+1}..i_{d-1}, j_k, j_{k+1}..j_{d-1}).
        I_rest_dims = tt.input_dims[k + 1 : d]
        J_rest_dims = tt.output_dims[k + 1 : d]
        T = T_new.reshape(r_l_k, i_k, *I_rest_dims, j_k, *J_rest_dims)
        _ = num_i_rest  # silence linters; kept for documentation.
    return T
