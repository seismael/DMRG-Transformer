# AGENTS.md: DMRG-Transformer Implementation Directives
**Version:** 1.1.0
**Status:** Validated Python reference implementation (Gates 1–3 passed). Phase IV (Rust/CUDA) deferred.
**Target Audience:** Autonomous Coding Agents (Cursor, Copilot, Devin, custom LLM wrappers).
**Role:** Lead Systems Architect and Quantum Optimization Engineer.

## 0. REQUIRED READING MATRIX (BLOCKING)
You are strictly forbidden from writing any code until you have read and parsed the following documentation files. They contain the exact mathematical and physical constraints for this project.
1. `docs/ARCHITECTURE.md` — System topology, OOD interfaces, execution pipeline
2. `docs/TENSOR_TOPOLOGY.md` — Einsum strings and strict rank boundaries
3. `docs/NUMERICAL_STABILITY.md` — SVD fallback hierarchy and Tikhonov regularization
4. `docs/MEMORY_ARENA.md` — Double-buffer contract (Python prototype; Rust Phase IV deferred)
5. `docs/SOLVER_MATH.md` — Formal proofs of exactness and O(d·n·r³) complexity
6. `docs/COMPLIANCE.md` — Spec-to-implementation traceability matrix
7. `README.md` — Quick start, measured results, project structure

## 1. Prime Directives & Absolute Constraints
You are building a post-Gradient Descent neural network. You must fundamentally disregard standard deep learning optimization paradigms.

* **CONSTRAINT 1 (NO GRADIENTS):** You are strictly forbidden from using `loss.backward()`, `tf.GradientTape`, or any auto-differentiation graph for weight updates. An AST scan (`tests/test_constraints.py`) enforces this.
* **CONSTRAINT 2 (NO ITERATIVE OPTIMIZERS):** You are strictly forbidden from implementing SGD, Adam, RMSprop, or any optimizer that relies on a "learning rate." The Adam implementation exists only in `scripts/` PoC comparisons and in `bench/benchmark.py` for baseline measurement — never in `src/`.
* **CONSTRAINT 3 (NO DENSE INVERSIONS):** You are strictly forbidden from inverting any matrix larger than the predefined TT-Rank bounds. O(N³) operations on the global parameter space will result in immediate failure. All solves use block-diagonal normal equations of size `(r·i_k·r)²` or the matrix-free CG path.
* **CONSTRAINT 4 (MEMORY MUTABILITY):** When orchestrating left/right orthogonal environment blocks (L, R), you must update them in-place via the `EnvironmentCache` and `MemoryArena` contracts. Do not allocate new massive tensors in memory during the Alternating Linear Scheme (ALS) sweep.

## 2. Core Architectural Mappings
When translating standard Transformer concepts into this framework, adhere to these exact mappings:

| Standard Deep Learning Concept | DMRG-Transformer Equivalent (MUST USE) |
| :--- | :--- |
| `nn.Linear(in, out)` | `TensorTrain(cores=[G_1, ..., G_d], ranks=r)` |
| `optimizer.step()` | `DMRGOptimizer.sweep_and_truncate(tt, target)` |
| Backpropagation (Chain Rule) | Layer-wise Target Propagation (pseudo-inverse) |
| Weight Update Calculation | Local SVD Projection (See `TENSOR_TOPOLOGY.md`) |
| Regularization / Weight Decay | Eckart-Young-Mirsky SVD Truncation (drop singular values > r) |

## 3. Implementation Phasing & Validation Gates
Do not proceed to a subsequent phase until the current phase passes its specific Validation Gate.

### PHASE I: The Mathematical Primitives (Tensor Train Core)
**Status:** ✅ COMPLETE. Implemented in `src/dmrg_transformer/tt/tensor_train.py`, `src/dmrg_transformer/core/svd.py`.
**Validation Gate 1:** Generate a dense 1024×1024 random matrix, decompose into TT with max rank r=32, reconstruct. The reconstruction error (Frobenius norm) must exactly match the theoretical truncation bound of the discarded singular values.
**Evidence:** `tests/test_gate1_reconstruction.py` — passes.

### PHASE II: The Orthogonalization Engine
**Status:** ✅ COMPLETE. Implemented in `src/dmrg_transformer/tt/gauge.py`, `src/dmrg_transformer/core/qr.py`.
**Validation Gate 2:** After a left-orthogonalization sweep, compute `LᵀL`. The result MUST be the Identity Matrix (I) to machine precision (< 1e-7 error).
**Evidence:** `tests/test_gate2_orthogonality.py` — passes.

### PHASE III: The DMRG Local Solver
**Status:** ✅ COMPLETE. Implemented in `src/dmrg_transformer/optim/sweep.py`, `src/dmrg_transformer/optim/local_solver.py`.
**Validation Gate 3:** Create a single TT-layer. Run dense least squares vs DMRG sweep. The MSE of the DMRG sweep must converge to the exact same MSE as the Dense Exact Solver, but execute in O(d·n·r³) time instead of O(N³).
**Evidence:** `tests/test_gate3_exact_parity.py` and `bench/GATE3_PROOF.md` — passes (DMRG MSE = 1.349e-29 vs Dense = 2.386e-30).

### PHASE IV: Systems Integration (Rust/CUDA)
**Status:** ⏳ DEFERRED. Python prototype only (`src/dmrg_transformer/core/arena.py`).
**Implementation needed:** Port the validated logic into a Rust microkernel with cuTensorNet for contractions and cuSOLVER for SVD/QR using double buffering. Requires sm_70+ GPU for Tensor Cores and cuTensorNet.
**Validation Gate 4:** Zero memory leaks across 1,000 sweep cycles. Profiling must show GPU Tensor Cores maintaining >80% utilization.
**Pre-requisites:** Hardware with Volta+ architecture (sm_70+). Not achievable on current MX150 (sm_61).

## 4. Authorized Single Call Sites
The following operations MUST only be invoked from their designated module:

| Operation | Authorized Module |
| :--- | :--- |
| `torch.linalg.svd` / `scipy.linalg.svd` | `src/dmrg_transformer/core/svd.py` (via `robust_svd()`) |
| `torch.linalg.qr` | `src/dmrg_transformer/core/qr.py` (via `qr_f64()`) |
| Direct TT-core modification | `src/dmrg_transformer/tt/tensor_train.py` (via `update_core()`) |
| Environment block computation | `src/dmrg_transformer/tt/environments.py` |
| Gauge/orthogonalization | `src/dmrg_transformer/tt/gauge.py` |

## 5. Agent Acknowledgment Protocol
Before writing any code or executing any shell commands, read all files listed in §0 REQUIRED READING MATRIX. When prompted to begin, respond with:
*"DMRG Optimization Protocol Acknowledged. Iterative Gradient Descent is disabled. All documentation files ingested."*
