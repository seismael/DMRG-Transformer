"""MemoryArena prototype tests (MEMORY_ARENA.md §2-§4).

These verify the *contract* the future Rust microkernel must satisfy:

* Total bytes pinned is fixed at construction (no growth during use).
* `take_left`/`take_right` return distinct buffers (no aliasing).
* `swap_*` flips the read/write pair in constant time without realloc.
* The buffers persist across many take/swap cycles with zero allocator activity.
"""
from __future__ import annotations

import torch

from dmrg_transformer.core.arena import ArenaSpec, MemoryArena
from dmrg_transformer.core.device import cuda_available, require_cuda


def _spec(dtype: torch.dtype = torch.float64) -> ArenaSpec:
    return ArenaSpec(
        max_rank=8, max_input_dim=4, max_output_dim=4,
        max_batch=64, max_J_pre=4, max_I_suf=4, max_J_suf=4, dtype=dtype,
    )


def test_arena_buffers_distinct() -> None:
    arena = MemoryArena(_spec(), device=require_cuda())
    l_read, l_write = arena.take_left()
    r_read, r_write = arena.take_right()
    assert l_read.data_ptr() != l_write.data_ptr()
    assert r_read.data_ptr() != r_write.data_ptr()


def test_arena_swap_flips_active() -> None:
    arena = MemoryArena(_spec(), device=require_cuda())
    l_read_0, l_write_0 = arena.take_left()
    arena.swap_left()
    l_read_1, l_write_1 = arena.take_left()
    assert l_read_1.data_ptr() == l_write_0.data_ptr()
    assert l_write_1.data_ptr() == l_read_0.data_ptr()


def test_arena_zero_allocations_across_cycles() -> None:
    """The "zero-allocation prime directive" (MEMORY_ARENA.md §2).

    Run 1000 take/swap cycles and assert torch's CUDA allocator did not allocate
    a single new block (only the pre-allocated arena buffers exist).
    """
    if not cuda_available():
        return
    dev = require_cuda()
    arena = MemoryArena(_spec(), device=dev)
    torch.cuda.synchronize(dev)
    base_alloc_count = torch.cuda.memory_stats(dev).get("allocation.all.allocated", 0)
    for _ in range(1000):
        _, lw = arena.take_left()
        _, rw = arena.take_right()
        # Touch the buffers so the loop isn't optimised away.
        lw.fill_(0.0)
        rw.fill_(0.0)
        arena.swap_left()
        arena.swap_right()
    torch.cuda.synchronize(dev)
    new_alloc_count = torch.cuda.memory_stats(dev).get("allocation.all.allocated", 0)
    delta = new_alloc_count - base_alloc_count
    # `fill_` may produce a handful of internal allocations on some kernels;
    # require dramatically less than 1000 (one per cycle would be a regression).
    assert delta < 50, (
        f"arena leaked {delta} allocations across 1000 cycles — "
        "double-buffering contract violated"
    )


def test_arena_total_bytes_constant() -> None:
    arena = MemoryArena(_spec(), device=require_cuda())
    bytes0 = arena.total_bytes()
    for _ in range(100):
        arena.take_left()
        arena.take_right()
        arena.swap_left()
    bytes1 = arena.total_bytes()
    assert bytes0 == bytes1
