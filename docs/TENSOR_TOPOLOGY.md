# TENSOR_TOPOLOGY.md: Dimension and Contraction Ledger
**Version:** 1.1.0
**Domain:** Tensor Network Geometry & CUDA Einsum Specifications

## 1. Objective
This document defines the strict dimensional boundaries, rank constraints, and exact Einstein Summation (einsum) strings required to execute the DMRG Alternating Linear Scheme. Any deviation from these dimensional bounds will result in catastrophic CUDA memory access violations.

**Implementation conformance:** The Python reference implementation (`src/dmrg_transformer/tt/`) follows this specification exactly. All einsum strings, rank boundaries, and the SVD reshape protocol are implemented as specified below.

## 2. Global Notation & Constraints
For a single Tensor Train (TT) layer mapping an input vector of size N to an output vector of size M:

* **Physical Input Dimensions:** i = [i₁, i₂, …, i_d] such that Π i_k = N.
* **Physical Output Dimensions:** j = [j₁, j₂, …, j_d] such that Π j_k = M.
* **TT-Ranks (Bond Dimensions):** r = [r₀, r₁, …, r_d].
* **Boundary Condition:** r₀ = 1 and r_d = 1 MUST be strictly enforced.
* **Core Tensor Shape:** The k-th core G_k has exactly 4 dimensions: `[r_{k-1}, i_k, j_k, r_k]`.

> **Implementation note:** For computation efficiency, physical dimensions i_k and j_k are flattened into a single physical dimension p_k = i_k × j_k, resulting in a 3D core: `[r_{k-1}, p_k, r_k]`. The 4D form is expanded on-the-fly during contractions. See `TensorTrain.contract_forward` in `src/dmrg_transformer/tt/tensor_train.py`.

## 3. Forward Pass (Inference Contraction)
When propagating an input vector X through the frozen TT-manifold.

* **Input State:** X reshaped to `[batch, i₁, i₂, ..., i_d]`
* **Core State:** G_k shaped `[r_{k-1}, i_k, j_k, r_k]`
* **Contraction Protocol (Left-to-Right):**
  Initialize a running tensor V with the input X.
  For each core k from 1 to d:
  `V = einsum('b i_k ..., a i_k j_k c -> b a j_k ... c', V, G_k)`

> **Note:** The Python implementation uses `torch.einsum` with the einsum strings specified above. Future Phase IV Rust implementation will use `cuTensorNet` for path-optimized contractions.

## 4. Environment Block Construction (L and R)
To isolate core k, we must build the orthogonal environment blocks from the surrounding cores and the input/target data.

### Left Environment Block (L_{<k})
Represents the contraction of all cores from 1 to k-1 with the Input X.
* **Shape:** `[batch, r_{k-1}]`
* **Recursive Update (if shifting left-to-right):**
  `L_new = einsum('b a, a p c -> b c', L_old, G_{k-1})`

### Right Environment Block (R_{>k})
Represents the contraction of all cores from d down to k+1.
* **Shape:** `[r_k, j_{k+1} … j_d]` (Simplified as `[r_k, output_remnant]`)

**Implementation:** See `src/dmrg_transformer/tt/environments.py` — `left_state_through()`, `right_pure_product()`, and `EnvironmentCache` for memoized incremental construction.

## 5. Local Core Projection (The Exact Solve)
To calculate the optimal weights for core k, the global target T must be projected into the local subspace using the environment blocks.

* **Target Tensor:** T shaped `[batch, output_space]`
* **Equation:** G̃_k = (L_{<k}^T ⊗ R_{>k}^T) T
* **Einsum Specification:** `G_tilde = einsum('b a, b p c, c -> a p c', L_left, Target_Local, R_right)`

> **Directive:** Ensure batch dimension `b` is correctly reduced during this contraction to yield the un-batched weight core.

## 6. SVD Reshape Protocol (CRITICAL)
cuSOLVER and LAPACK cannot perform Singular Value Decomposition on 3D tensors. The 3D core must be flattened, decomposed, and unflattened.

**Step 1: Flattening (Matricization)**
* Input Core: G̃_k shaped `[r_{k-1}, p_k, r_k]`
* Reshape to 2D Matrix C: `[r_{k-1} * p_k, r_k]` (left gauge) or `[r_{k-1}, p_k * r_k]` (right gauge)
* Command: `C = G_tilde.reshape(…)`

**Step 2: Execution & Truncation**
* Execute: `U, S, Vh = svd(C, full_matrices=False)`
* Truncate to maximum allowed rank r_max:
  `U_trunc = U[:, :r_max]`
  `S_trunc = diag(S[:r_max])`
  `Vh_trunc = Vh[:r_max, :]`

**Step 3: Unflattening & Gauge Shift**
* Left sweeps (k increasing): Keep U·S as new core G_k, absorb S·Vh into G_{k+1}.
* Right sweeps (k decreasing): Keep S·Vh as new core G_k, absorb U·S into G_{k-1}.

**Implementation:** See `solve_local_core()` in `src/dmrg_transformer/optim/local_solver.py` — the `direction` parameter controls the gauge shift direction. The 4-tier SVD fallback hierarchy (GPU → gesdd CPU → gesvd CPU → noise+retry) is in `src/dmrg_transformer/core/svd.py`.
