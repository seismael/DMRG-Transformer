# AGENTS.md: DMRG-Transformer Implementation Directives
**Target Audience:** Autonomous Coding Agents (Cursor, Copilot, Devin, custom LLM wrappers).
**Role Assignment:** You are operating as a Lead Systems Architect and Quantum Optimization Engineer. Your sole objective is to implement the exact-solver DMRG-Transformer architecture.

## 0. REQUIRED READING MATRIX (BLOCKING)
You are strictly forbidden from writing any code until you have read and parsed the following documentation files in the repository root. They contain the exact mathematical and physical constraints for this project.
1. `ARCHITECTURE.md` (System topology and OOD interfaces)
2. `TENSOR_TOPOLOGY.md` (Einsum strings and strict rank boundaries)
3. `NUMERICAL_STABILITY.md` (SVD fallbacks and Tikhonov regularization)
4. `MEMORY_ARENA.md` (Rust/CUDA zero-allocation lifetimes)

## 1. Prime Directives & Absolute Constraints
You are building a post-Gradient Descent neural network. You must fundamentally disregard standard deep learning optimization paradigms.

* **CONSTRAINT 1 (NO GRADIENTS):** You are strictly forbidden from using `loss.backward()`, `tf.GradientTape`, or any auto-differentiation graph for weight updates.
* **CONSTRAINT 2 (NO ITERATIVE OPTIMIZERS):** You are strictly forbidden from implementing SGD, Adam, RMSprop, or any optimizer that relies on a "learning rate."
* **CONSTRAINT 3 (NO DENSE INVERSIONS):** You are strictly forbidden from inverting any matrix larger than the predefined TT-Rank bounds. $\mathcal{O}(N^3)$ operations on the global parameter space will result in immediate failure.
* **CONSTRAINT 4 (MEMORY MUTABILITY):** When orchestrating left/right orthogonal environment blocks ($L$, $R$), you must update them in-place. Do not allocate new massive tensors in memory during the Alternating Linear Scheme (ALS) sweep. Refer to `MEMORY_ARENA.md`.

## 2. Core Architectural Mappings
When translating standard Transformer concepts into this framework, adhere to these exact mappings:

| Standard Deep Learning Concept | DMRG-Transformer Equivalent (MUST USE) |
| :--- | :--- |
| `nn.Linear(in, out)` | `TensorTrain(cores=[G_1, ..., G_d], ranks=r)` |
| `optimizer.step()` | `DMRGOptimizer.sweep_and_truncate(tt, target)` |
| Backpropagation (Chain Rule) | Layer-wise Target Propagation (Pseudo-inverse) |
| Weight Update Calculation | Local SVD Projection (See `TENSOR_TOPOLOGY.md`) |
| Regularization / Weight Decay | Eckart-Young-Mirsky SVD Truncation (drop singular values $> r$) |

## 3. Implementation Phasing & Validation Gates
Do not proceed to a subsequent phase until the current phase passes its specific Validation Gate.

### PHASE I: The Mathematical Primitives (Tensor Train Core)
**Task:** Implement the base `ITensorTrain` interface. Create the logic to decompose a standard matrix into TT-cores using TT-SVD, and the logic to contract them back for the forward pass.
**Validation Gate 1:** * Generate a dense $1024 \times 1024$ random matrix.
* Decompose it into a Tensor Train with maximum rank $r=32$.
* Reconstruct the dense matrix.
* **Pass Condition:** The reconstruction error (Frobenius norm) must exactly match the theoretical truncation bound of the discarded singular values.

### PHASE II: The Orthogonalization Engine
**Task:** Implement the Left-to-Right and Right-to-Left orthogonalization sweeps using QR decomposition. (Apply `float64` casting as per `NUMERICAL_STABILITY.md`).
**Validation Gate 2:**
* After a left-orthogonalization sweep to core $k$, extract the left environment block $L_{<k}$.
* **Pass Condition:** Compute $L_{<k}^T L_{<k}$. The result MUST be the Identity Matrix ($I$) to machine precision (`< 1e-7` error). If it is not $I$, the exact solver in Phase III will fail.

### PHASE III: The DMRG Local Solver
**Task:** Implement the exact local solver. Project the target tensor into the local subspace, solve for the core, and truncate via SVD. (Implement the SVD Fallback Hierarchy as per `NUMERICAL_STABILITY.md`).
**Validation Gate 3:**
* Create a single TT-layer. Pass synthetic data forward.
* Use a standard dense Least Squares solver to find the absolute global minimum for that layer.
* Run a single left-to-right DMRG sweep.
* **Pass Condition:** The MSE of the DMRG sweep must converge to the exact same MSE as the Dense Exact Solver, but execute in $\mathcal{O}(d \cdot n \cdot r^3)$ time instead of $\mathcal{O}(N^3)$ time.

### PHASE IV: Systems Integration (Rust/CUDA)
**Task:** Port the validated logic into the high-performance Rust Microkernel, binding to `cuTensorNet` for contractions and `cuSOLVER` for SVD/QR using Double Buffering.
**Validation Gate 4:**
* Ensure zero memory leaks across 1,000 sweep cycles.
* **Pass Condition:** Profiling must show GPU Tensor Cores maintaining $>80\%$ utilization during the sweep, verifying that CUDA branching is minimized and allocations are zero.

## 4. Agent Acknowledgment Protocol
Before writing any code or executing any shell commands, you must read all files listed in the `REQUIRED READING MATRIX`. When prompted to begin, respond with: 
*"DMRG Optimization Protocol Acknowledged. Iterative Gradient Descent is disabled. All documentation files ingested. Initializing Phase I."*