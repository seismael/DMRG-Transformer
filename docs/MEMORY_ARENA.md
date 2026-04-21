# MEMORY_ARENA.md: Rust/CUDA Memory Lifecycle & Concurrency
**Version:** 1.0.0
**Domain:** VRAM Management, FFI Lifetimes, and Zero-Allocation Sweeps

## 1. Objective
The Density Matrix Renormalization Group (DMRG) sweep is a continuous, high-frequency loop of tensor contractions and Singular Value Decompositions. This document dictates the strict memory management protocols required to achieve zero-allocation execution during the optimization sweep, utilizing Rust's borrow checker to guarantee memory safety across the C/CUDA boundary.

## 2. The "Zero-Allocation" Prime Directive
**CRITICAL CONSTRAINT:** The agent is strictly forbidden from allocating new heap memory (Host or Device) inside the `IDMRGOptimizer.sweep()` loop. 

* **Pre-Allocation Phase:** Before the first forward pass, the system MUST calculate the maximum possible size for the environment blocks ($L$, $R$) and the projection tensors based on the maximum allowed TT-Rank ($r_{max}$).
* **The Arena:** These max-sized buffers are allocated once into a `MemoryArena` struct on the GPU.
* **In-Place Execution:** All `cuTensorNet` contractions and `cuSOLVER` SVDs MUST write their outputs directly into these pre-allocated device pointers using appropriate strides/offsets.

## 3. Rust Ownership & CUDA FFI Protocol
Rust's memory safety must enforce the physical state of the GPU. The agent must map Rust's lifetime and mutability rules directly to CUDA device pointers.

* **Immutable Reads:** Environment blocks that are being contracted against MUST be passed as immutable references `&DeviceBuffer`.
* **Mutable Writes:** The target buffer for a contraction or SVD MUST be passed as an exclusive mutable reference `&mut DeviceBuffer`.
* **FFI Safety:** When passing pointers to `cuTensorNet`, the agent MUST use `as_ptr()` for `&DeviceBuffer` and `as_mut_ptr()` for `&mut DeviceBuffer`. Rust's borrow checker will therefore mathematically guarantee at compile-time that we are never reading and writing to the same CUDA memory block simultaneously.

## 4. The "Ping-Pong" Buffer Strategy (Double Buffering)
During a left-to-right sweep, the system must update the left environment block $L_{<k}$ to $L_{<k+1}$. If we write the new block directly into the memory of the old block while the GPU is still computing, we trigger a data race.

* **Implementation:** The `MemoryArena` MUST contain two buffers for each environment: `L_buffer_A` and `L_buffer_B`.
* **Execution:**
  * Step 1: Read from `L_buffer_A`, compute contraction, write to `L_buffer_B`.
  * Step 2: Swap the pointers in the Rust state manager.
  * Step 3: Shift to the next core. Read from `L_buffer_B`, write to `L_buffer_A`.

## 5. CUDA Stream Concurrency (Multi-Head Parallelism)
To saturate the GPU Tensor Cores, the microkernel must exploit the independent nature of Transformer Multi-Head Attention (MHA).

* **Stream Allocation:** The `MemoryArena` must instantiate a dedicated CUDA stream (`cudaStream_t`) and a dedicated `cuTensorNet` handle for *each* attention head.
* **Dispatch:** When the microkernel triggers a sweep for an Attention layer, it MUST dispatch the DMRG loops for Head 1, Head 2, ..., Head $H$ asynchronously across their respective streams.
* **Synchronization:** The microkernel MUST NOT call `cudaDeviceSynchronize()` inside the sweep. Synchronization across heads may only occur at the end of the layer's backward target propagation phase using CUDA events (`cudaEventRecord` / `cudaStreamWaitEvent`).

## 6. SVD Workspace Memory
The `cuSOLVER` library requires a temporary "Workspace" buffer to compute the SVD.
* **Query First:** The agent MUST call `cusolverDnSgesvd_bufferSize` during the Pre-Allocation Phase to determine the required bytes.
* **Arena Storage:** This workspace must be permanently allocated in the `MemoryArena` alongside the tensor buffers. It must be reused for every SVD operation across the entire training lifecycle.