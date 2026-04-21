# TENSOR_TOPOLOGY.md: Dimension and Contraction Ledger
**Version:** 1.0.0
**Domain:** Tensor Network Geometry & CUDA Einsum Specifications

## 1. Objective
This document defines the strict dimensional boundaries, rank constraints, and exact Einstein Summation (einsum) strings required to execute the DMRG Alternating Linear Scheme. Any deviation from these dimensional bounds will result in catastrophic CUDA memory access violations.

## 2. Global Notation & Constraints
For a single Tensor Train (TT) layer mapping an input vector of size $N$ to an output vector of size $M$:
* **Physical Input Dimensions:** $\mathbf{i} = [i_1, i_2, \dots, i_d]$ such that $\prod i_k = N$.
* **Physical Output Dimensions:** $\mathbf{j} = [j_1, j_2, \dots, j_d]$ such that $\prod j_k = M$.
* **TT-Ranks (Bond Dimensions):** $\mathbf{r} = [r_0, r_1, \dots, r_d]$. 
* **Boundary Condition:** $r_0 = 1$ and $r_d = 1$ MUST be strictly enforced.
* **Core Tensor Shape:** The $k$-th core $\mathcal{G}_k$ has exactly 4 dimensions: `[r_{k-1}, i_k, j_k, r_k]`.

*Note: For implementation simplicity, physical dimensions $i_k$ and $j_k$ are often flattened into a single physical dimension $p_k = i_k \times j_k$, resulting in a 3D core: `[r_{k-1}, p_k, r_k]`.*

## 3. Forward Pass (Inference Contraction)
When propagating an input vector $X$ through the frozen TT-manifold.

* **Input State:** $X$ reshaped to `[batch, i_1, i_2, ..., i_d]`
* **Core State:** $\mathcal{G}_k$ shaped `[r_{k-1}, i_k, j_k, r_k]`
* **Contraction Protocol (Left-to-Right):**
  Initialize a running tensor $V$ with the input $X$.
  For each core $k$ from 1 to $d$:
  `V = einsum('b i_k ..., a i_k j_k c -> b a j_k ... c', V, G_k)`
  
*(Agent Directive: Use `cuTensorNet` path optimization for this contraction, as naive `einsum` execution order will cause exponential memory inflation).*

## 4. Environment Block Construction ($L$ and $R$)
To isolate core $k$, we must build the orthogonal environment blocks from the surrounding cores and the input/target data.

### Left Environment Block ($L_{<k}$)
Represents the contraction of all cores from $1$ to $k-1$ with the Input $X$.
* **Shape:** `[batch, r_{k-1}]`
* **Recursive Update (if shifting left-to-right):**
  `L_{new} = einsum('b a, a p c -> b c', L_{old}, G_{k-1})`

### Right Environment Block ($R_{>k}$)
Represents the contraction of all cores from $d$ down to $k+1$.
* **Shape:** `[r_k, j_{k+1} \dots j_d]` (Simplified as `[r_k, output\_remnant]`)

## 5. Local Core Projection (The Exact Solve)
To calculate the optimal weights for core $k$, the global target $\mathcal{T}$ must be projected into the local subspace using the environment blocks.

* **Target Tensor:** $\mathcal{T}$ shaped `[batch, output\_space]`
* **Equation:** $\tilde{\mathcal{G}}_k = (L_{<k}^T \otimes R_{>k}^T) \mathcal{T}$
* **Einsum Specification:** `G_tilde = einsum('b a, b p c, c -> a p c', L_left, Target_Local, R_right)`
  *(Agent Directive: Ensure batch dimension `b` is correctly reduced during this contraction to yield the un-batched weight core).*

## 6. SVD Reshape Protocol (CRITICAL)
`cuSOLVER` and LAPACK cannot perform Singular Value Decomposition on 3D tensors. The 3D core must be flattened, decomposed, and unflattened.

**Step 1: Flattening (Matricization)**
* Input Core: $\tilde{\mathcal{G}}_k$ shaped `[r_{k-1}, p_k, r_k]`
* Reshape to 2D Matrix $C$: `[r_{k-1} * p_k, r_k]`
* Command: `C = G_tilde.reshape(r_{k-1} * p_k, r_k)`

**Step 2: Execution & Truncation**
* Execute: `U, S, Vh = svd(C, full_matrices=False)`
* Truncate to maximum allowed rank $r_{max}$:
  `U_trunc = U[:, :r_max]`
  `S_trunc = diag(S[:r_max])`
  `Vh_trunc = Vh[:r_max, :]`

**Step 3: Unflattening (Tensorization)**
* Combine $U$ and $S$ into the new left-orthogonal core:
  `G_new_2D = U_trunc @ S_trunc`
* Reshape back to 3D:
  `G_k = G_new_2D.reshape(r_{k-1}, p_k, r_{max})`
* The $Vh\_trunc$ matrix is passed to the right to be absorbed by core $k+1$ (Gauge shifting).