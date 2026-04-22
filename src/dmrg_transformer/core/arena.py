"""Python prototype of the GPU MemoryArena (MEMORY_ARENA.md §2-§4).

This is the **Python contract** that the future Rust microkernel
(:doc:`MEMORY_ARENA.md` §3) will implement byte-for-byte. It exists so we can:

1. Validate the ping-pong double-buffering protocol in pure PyTorch before
   committing to a Rust rewrite.
2. Instrument allocations and prove zero per-step `cudaMalloc` (Phase IV
   Gate 4 leak gate, partial form).
3. Give the existing :class:`DMRGOptimizer` a path to opt into pre-allocated
   environment buffers without changing call sites.

The arena pre-allocates the worst-case-sized `L_A`, `L_B`, `R_A`, `R_B`
buffers at construction time (sized from `max_rank` and the largest TT-core
physical dim). During a sweep the optimiser reads from one buffer and writes
to its pair, then swaps pointers — never allocating new device memory.

This is a **prototype**: it does not yet integrate into the sweep loop (that
is Phase A4 of the plan, gated behind a feature flag). It currently exposes
the API + alloc-counter test surface so the contract is locked in.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ArenaSpec:
    """Sizing parameters for the arena."""

    max_rank: int
    max_input_dim: int   # max i_k over cores
    max_output_dim: int  # max j_k over cores
    max_batch: int
    max_J_pre: int       # noqa: N815  - matches MEMORY_ARENA.md math notation
    max_I_suf: int       # noqa: N815
    max_J_suf: int       # noqa: N815
    dtype: torch.dtype = torch.float64


class MemoryArena:
    """Pre-allocated double-buffered storage for L/R environment blocks.

    Buffers are allocated once and reused for every local solve in every
    sweep. Use :meth:`take_left` / :meth:`take_right` to get the active
    buffer for read and the inactive buffer for write; call :meth:`swap_left`
    / :meth:`swap_right` after the write completes.
    """

    def __init__(self, spec: ArenaSpec, device: torch.device | None = None) -> None:
        self.spec = spec
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        # L block worst-case shape: (batch, J_pre, r, i, I_suf).
        l_shape = (
            spec.max_batch, spec.max_J_pre, spec.max_rank,
            spec.max_input_dim, spec.max_I_suf,
        )
        # R block worst-case shape: (r, I_suf, J_suf).
        r_shape = (spec.max_rank, spec.max_I_suf, spec.max_J_suf)

        self._l_a = torch.empty(l_shape, dtype=spec.dtype, device=self.device)
        self._l_b = torch.empty(l_shape, dtype=spec.dtype, device=self.device)
        self._r_a = torch.empty(r_shape, dtype=spec.dtype, device=self.device)
        self._r_b = torch.empty(r_shape, dtype=spec.dtype, device=self.device)

        # SVD workspace — sized for the largest H block that will be solved.
        # H is (P_block, P_block) with P_block = r·i·r.
        p_block_max = spec.max_rank * spec.max_input_dim * spec.max_rank
        self._svd_work = torch.empty(
            (p_block_max, p_block_max), dtype=spec.dtype, device=self.device,
        )

        # Pointers (which physical buffer is currently "active").
        self._l_active_is_a = True
        self._r_active_is_a = True

    # -- L block --------------------------------------------------------------

    def take_left(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(read_buf, write_buf)`` for the L environment.

        Caller reads from ``read_buf`` and writes its updated L block into
        ``write_buf`` (in place via ``read_buf.copy_(...)`` style is forbidden
        — the contract is no in-place updates of the read source while it's
        being read; that's the whole point of double-buffering, see
        MEMORY_ARENA.md §4).
        """
        if self._l_active_is_a:
            return self._l_a, self._l_b
        return self._l_b, self._l_a

    def swap_left(self) -> None:
        """Promote the write buffer to the new active buffer (constant-time)."""
        self._l_active_is_a = not self._l_active_is_a

    # -- R block --------------------------------------------------------------

    def take_right(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._r_active_is_a:
            return self._r_a, self._r_b
        return self._r_b, self._r_a

    def swap_right(self) -> None:
        self._r_active_is_a = not self._r_active_is_a

    # -- SVD workspace --------------------------------------------------------

    def svd_workspace(self) -> torch.Tensor:
        """Return the pre-allocated workspace for the next solve's H matrix."""
        return self._svd_work

    # -- Memory accounting ----------------------------------------------------

    def total_bytes(self) -> int:
        """Sum of bytes pinned by the arena (constant for the arena's lifetime)."""
        elem_bytes = torch.empty((), dtype=self.spec.dtype).element_size()
        return elem_bytes * sum(t.numel() for t in (
            self._l_a, self._l_b, self._r_a, self._r_b, self._svd_work,
        ))


__all__ = ["ArenaSpec", "MemoryArena"]
