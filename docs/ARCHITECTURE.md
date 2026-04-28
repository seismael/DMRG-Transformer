# ARCHITECTURE.md: DMRG-Transformer
**Version:** 1.1.0
**Status:** Python reference implementation validated (Gates 1–3). Phase IV Rust/CUDA deferred.
**Domain:** Deep Learning Optimization / Systems Architecture

## 1. Executive Summary
The **DMRG-Transformer** is a post-Gradient Descent neural network architecture. It replaces the traditional Backpropagation optimization engine with a topological, exact-solver framework derived from quantum many-body physics: the **Density Matrix Renormalization Group (DMRG)** applied over a **Tensor Train (TT)** manifold.

The architecture achieves mathematically exact local weight optimization without iterative gradient steps, learning rates, or vanishing gradients, by solving per-core least-squares sub-problems followed by Eckart–Young–Mirsky SVD truncation.

**Current implementation:** Python 3.12 reference with PyTorch 2.5.1 + CUDA 12.1. Future Rust microkernel (Phase IV) is deferred pending hardware upgrade to sm_70+ (Volta Tensor Cores).

---

## 2. Architectural Paradigms & Constraints

To ensure modularity, extensibility, and hardware performance, the system architecture adheres to:

* **OOD Interface Contracts:** All mathematical operators, tensor representations, and optimization sweeps are encapsulated within strictly typed protocols (see `src/dmrg_transformer/core/interfaces.py`).
* **Separation of Concerns:** TT geometry (`tt/`), optimization engine (`optim/`), target propagation (`propagation/`), and neural network modules (`nn/`) are isolated layers.
* **Hardware Symbiosis:** Tensor contractions, SVD, and QR operations run on GPU via PyTorch's CUDA backend (cuSOLVER). Future Phase IV will use direct cuTensorNet bindings.

---

## 3. High-Level System Topology

The architecture is divided into three isolated domains:

### Layer 1: The Network Topology (Transformer Modularity)
Standard Transformer blocks (Multi-Head Attention, Feed-Forward) remain structurally identical to preserve compatibility with existing architectures. Dense weight matrices (W_Q, W_K, W_V, W_out) are stored as decoupled `TensorTrain` objects rather than dense 2D arrays.

**Implementation:** `src/dmrg_transformer/nn/tt_linear.py`, `tt_mha.py`, `tt_ffn.py`, `tt_block.py`

### Layer 2: The Orchestration Layer (Python)
Responsible for:
1.  Target Propagation — calculating layer-wise exact target states T_ℓ (`src/dmrg_transformer/propagation/`).
2.  Scheduling the Alternating Linear Scheme (ALS) sweeps (`src/dmrg_transformer/optim/`).
3.  Managing memory allocations for orthogonalized tensor blocks (L_{<k} and R_{>k}) via `EnvironmentCache`.

### Layer 3: The Mathematical Execution Engine (PyTorch/CUDA)
A stateless execution layer that accepts instructions to perform:
* Tensor Contractions (einsum operations via `torch.einsum`).
* Singular Value Decompositions (SVD) via `torch.linalg.svd` or SciPy fallback.
* QR Factorizations for left/right orthogonalization.

---

## 4. Component Breakdown & OOD Interfaces

### 4.1. `ITensorTrain`
Encapsulates the geometry of the factorized weight space.
* **Responsibilities:** Stores 3D core tensors (shape `[r_{k-1}, p_k, r_k]`), manages TT-ranks (r), and executes left/right orthogonalization.
* **State:** Array of localized core tensors.
* **Implementation:** `src/dmrg_transformer/tt/tensor_train.py`

```python
class ITensorTrain(Protocol):
    def orthogonalize_left(self, core_index: int) -> None: ...
    def orthogonalize_right(self, core_index: int) -> None: ...
    def get_core(self, index: int) -> Tensor3D: ...
    def update_core(self, index: int, new_core: Tensor3D) -> None: ...
    def contract_forward(self, input: Tensor) -> Tensor: ...
```

### 4.2. `ITargetPropagator`
Replaces the Backpropagation Chain Rule.
* **Responsibilities:** Translates the global network error into a specific, localized target tensor T_ℓ for a single layer ℓ.
* **Mechanics:** Uses Tikhonov-damped pseudo-inverses with regime-aware Gram matrix selection (overdetermined vs underdetermined).
* **Implementation:** `src/dmrg_transformer/propagation/target_propagator.py`

```python
class ITargetPropagator(Protocol):
    def compute_layer_target(
        self, global_target: Tensor, current_layer_out: Tensor
    ) -> Tensor: ...
```

### 4.3. `IDMRGOptimizer`
The exact-solver replacement for Gradient Descent (Adam/SGD).
* **Responsibilities:** Executes left-to-right and right-to-left topological sweeps over an `ITensorTrain`.
* **Mechanics:** Solves the convex sub-problem via block-diagonal normal equations and applies SVD truncation.
* **Implementation:** `src/dmrg_transformer/optim/sweep.py`, `local_solver.py`

```python
class IDMRGOptimizer(Protocol):
    def sweep(self, tt: ITensorTrain, target: Tensor, max_rank: int) -> float: ...
    def solve_local_core(self, left_block: Tensor, right_block: Tensor,
                         target: Tensor) -> Tensor: ...
    def truncate_svd(self, exact_core: Tensor, max_rank: int) -> Tensor3D: ...
```

---

## 5. Execution Pipeline: The DMRG Optimization Loop

This sequence replaces the standard backward pass and optimizer step.

1.  **Forward Pass:** The input X is contracted through the TT-layers. The final output Y_pred is generated.
2.  **Target Propagation:** The `TargetPropagator` calculates the ideal target tensor for each layer using DTP (Difference Target Propagation): `x_target = x_curr + α·(Y_target − Y_curr)`. This preserves internal sequence structure.
3.  **Sweep Initialization:** The `DMRGOptimizer` initializes left/right orthogonal environment blocks (L, R) via `EnvironmentCache`. Blocks are computed with optional memoization.
4.  **Local Exact Solve (Per-Core Loop):**
    * Isolate core k. Build environment views L and R via `left_state_through()` and `right_pure_product()`.
    * Construct block-diagonal normal equations `H = JᵀJ` where H has shape `(r_{k-1}·i_k·r_k)²`.
    * Solve `(H + λI) · vec(G_k) = JᵀY` via `torch.linalg.solve` (or matrix-free CG for large scales).
    * Execute SVD on the matricized core: `C = U Σ Vᵀ`.
    * Truncate singular values to enforce max TT-rank r.
    * Update core k with truncated factors. Absorb remnant into adjacent core (gauge shift).
5.  **Bidirectional Sweep:** Repeat steps 3–4 L→R then R→L across all cores. MSE decreases monotonically per sweep.

---

## 6. Key Design Decisions

### Memory Management
* Left/right environment blocks are memoized in `EnvironmentCache` with lazy invalidation on core updates.
* Matrix-free Conjugate Gradient path auto-activates when the dense block-diagonal H matrix would exceed the GPU memory budget (default 512 MiB threshold).
* Python `MemoryArena` prototype pre-allocates double-buffered L/R blocks and SVD workspace.

### Numerical Stability
* Float32 for forward pass/contractions; float64 for QR and SVD (condition number ≥ 10⁴ triggers float64).
* 4-tier SVD fallback: GPU native → SciPy gesdd CPU → SciPy gesvd CPU → Gaussian noise + retry.
* Tikhonov regularization with auto-escalation on NaN detection (λ starts at 1e-5, multiplies by 10).
* Huber clamping at ±5σ on targets to prevent exact solver from responding violently to outliers.

### Parallelization
* Individual attention head projections (Q, K, V, W_out) are mathematically independent and can be solved in parallel `torch.cuda.Stream`s.
* Current Python implementation dispatches at the projection level; full per-head CUDA stream orchestration is deferred to Phase IV Rust microkernel.

---

## 7. Implementation Status

### Phase I: The Mathematical Primitives — ✅ COMPLETE
TT-decomposition (`from_dense`), reconstruction (`to_dense`), forward contraction (`contract_forward`). Validated by Gate 1.

### Phase II: The Orthogonalization Engine — ✅ COMPLETE
Left/right orthogonalization sweeps via QR decomposition in float64. Validated by Gate 2.

### Phase III: The DMRG Local Solver — ✅ COMPLETE
Exact local least-squares solver with block-diagonal normal equations, matrix-free CG path, and SVD truncation. Validated by Gate 3.

### Phase IV: Systems Integration (Rust/CUDA) — ⏳ DEFERRED
Python prototype only. Full Rust microkernel with cuTensorNet + cuSOLVER direct bindings requires sm_70+ GPU. Current MX150 (sm_61, 2 GiB) cannot run this phase.

### Future: ADMM Outer Loop — 🧭 IN DEVELOPMENT
Wraps per-layer DMRG sweeps in an Alternating Direction Method of Multipliers outer loop to resolve inter-layer drift in stacked blocks. See `FUTURE_WORK.md` Option B.
