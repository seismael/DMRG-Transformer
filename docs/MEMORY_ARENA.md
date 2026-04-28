# MEMORY_ARENA.md: Memory Lifecycle & Concurrency
**Version:** 1.1.0
**Domain:** VRAM Management, FFI Lifetimes, and Zero-Allocation Sweeps

## 1. Objective
The Density Matrix Renormalization Group (DMRG) sweep is a continuous, high-frequency loop of tensor contractions and Singular Value Decompositions. This document dictates the strict memory management protocols required to achieve zero-allocation execution during the optimization sweep.

**Current status:** Python prototype implemented (`src/dmrg_transformer/core/arena.py`). Full Rust+CUDA enforcement deferred to Phase IV (requires sm_70+ GPU with Tensor Cores).

## 2. The "Zero-Allocation" Prime Directive
**CRITICAL CONSTRAINT:** No allocating new heap memory (Host or Device) inside the `IDMRGOptimizer.sweep()` loop.

* **Pre-Allocation Phase:** Before the first forward pass, the system calculates the maximum possible size for environment blocks (L, R) and projection tensors based on the maximum allowed TT-Rank (r_max).
* **The Arena:** These max-sized buffers are allocated once into a `MemoryArena` struct.
* **In-Place Execution:** All tensor contractions and SVDs write their outputs directly into these pre-allocated device buffers using appropriate strides/offsets.

**Python prototype:** `MemoryArena` class pre-allocates L_A, L_B, R_A, R_B buffers at construction; `take_left()`/`take_right()` return (read, write) pairs; `swap_left()`/`swap_right()` toggle the active buffer pointer in constant time. The existing `DMRGOptimizer.sweep` uses `EnvironmentCache` for incremental construction — integration with the arena is pending.

## 3. Rust Ownership & CUDA FFI (Phase IV Only)
When the Rust microkernel is implemented, Rust's memory safety will enforce the physical state of the GPU. The agent must map Rust's lifetime and mutability rules directly to CUDA device pointers.

* **Immutable Reads:** Environment blocks being contracted against MUST be passed as immutable references `&DeviceBuffer`.
* **Mutable Writes:** The target buffer for a contraction or SVD MUST be passed as an exclusive mutable reference `&mut DeviceBuffer`.
* **FFI Safety:** When passing pointers to cuTensorNet, use `as_ptr()` for `&DeviceBuffer` and `as_mut_ptr()` for `&mut DeviceBuffer`. Rust's borrow checker will mathematically guarantee at compile-time that we are never reading and writing to the same CUDA memory block simultaneously.

## 4. The "Ping-Pong" Buffer Strategy (Double Buffering)
During a left-to-right sweep, the system must update the left environment block L_{<k} to L_{<k+1}. If we write the new block directly into the memory of the old block while the GPU is still computing, we trigger a data race.

* **Implementation:** The `MemoryArena` MUST contain two buffers for each environment: `L_buffer_A` and `L_buffer_B`.
* **Execution:**
  * Step 1: Read from `L_buffer_A`, compute contraction, write to `L_buffer_B`.
  * Step 2: Swap the pointers in the Rust state manager.
  * Step 3: Shift to the next core. Read from `L_buffer_B`, write to `L_buffer_A`.

**Python prototype:** Implements this contract exactly — `MemoryArena.take_left()` returns `(read_buf, write_buf)` and `swap_left()` flips the active indicator. Future Rust implementation must match this byte-for-byte.

## 5. CUDA Stream Concurrency (Multi-Head Parallelism)
To saturate the GPU Tensor Cores, the microkernel must exploit the independent nature of Transformer Multi-Head Attention (MHA).

* **Stream Allocation:** The `MemoryArena` must instantiate a dedicated CUDA stream (`cudaStream_t`) and a dedicated `cuTensorNet` handle for each attention head.
* **Dispatch:** When the microkernel triggers a sweep for an Attention layer, it dispatches the DMRG loops for Head 1, Head 2, …, Head H asynchronously across their respective streams.
* **Synchronization:** The microkernel must NOT call `cudaDeviceSynchronize()` inside the sweep. Synchronization across heads may only occur at the end of the layer's backward target propagation phase using CUDA events (`cudaEventRecord` / `cudaStreamWaitEvent`).

**Python prototype:** Partial implementation — `TTMultiHeadAttention.dmrg_step_projections` dispatches independent Q/K/V projection sweeps to three `torch.cuda.Stream`s, but this is projection-level (not per-head) and ends with an explicit synchronize. Full event-based per-head orchestration remains Phase IV.

## 6. SVD Workspace Memory
The cuSOLVER library requires a temporary "Workspace" buffer to compute the SVD.

* **Query First:** The agent MUST call `cusolverDnSgesvd_bufferSize` during the Pre-Allocation Phase to determine the required bytes.
* **Arena Storage:** This workspace must be permanently allocated in the `MemoryArena` alongside the tensor buffers. It must be reused for every SVD operation across the entire training lifecycle.

**Python prototype:** `MemoryArena.svd_workspace()` pre-allocates the workspace. The Python reference uses PyTorch's SVD wrapper which internally manages its own workspace; direct cuSOLVER binding remains Phase IV.
