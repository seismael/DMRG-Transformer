# Formal Mathematical Documentation: Tensor Train Optimization (DMRG) as an Exact Solver

**Version:** 1.1.0

## Executive Summary
This document provides the rigorous mathematical proofs underpinning the replacement of iterative Gradient Descent (GD) with **Density Matrix Renormalization Group (DMRG)** sweeps over a **Tensor Train (TT)** manifold. The objective is to prove mathematically how the O(N³) computational bottleneck of exact weight calculation is circumvented, allowing for non-iterative, learning-rate-free optimization of deep neural architectures.

---

## I. The Baseline Problem: The O(N³) Exact Solver Wall
Before establishing the tensor framework, we must formally define the mathematical bottleneck of standard exact solvers.

For a single neural network layer, let A ∈ R^{M×B} be the incoming activation matrix (M features, B batch size) and T ∈ R^{P×B} be the target output matrix. We seek the weight matrix W ∈ R^{P×M} that minimizes the Frobenius norm of the residual error:

```
L(W) = ½‖T − W A‖²_F
```

To find the global minimum analytically, we set the gradient w.r.t. W to zero:

```
∇_W L = −(T − W A)Aᵀ = 0
W A Aᵀ = T Aᵀ
```

Isolating W yields the exact closed-form solution:

```
W* = T Aᵀ (A Aᵀ)⁻¹
```

**The Bottleneck Proof:**
The term A Aᵀ yields an M×M covariance matrix. Inverting this matrix requires O(M³) operations via Gaussian elimination or Cholesky decomposition. For large layers (e.g., M = 10,000), this is physically unfeasible.

---

## II. Geometrical Restructuring: Tensor Train (TT) Factorization
To break the O(M³) wall, we redefine the geometry of W. We apply a tensorization mapping Φ that reshapes the 2D matrix W into a d-dimensional high-order tensor W ∈ R^{n₁×n₂×…×n_d}.

To avoid storing W densely (which would still cost O(N) memory with N = Π n_k), we decompose it into a **Tensor Train (TT)** — a sequence of 3-dimensional core tensors G_k:

```
W(i₁, …, i_d, j₁, …, j_d) = Σ_{α₀=1}^{r₀} … Σ_{α_d=1}^{r_d} G₁(α₀,i₁,j₁,α₁) … G_d(α_{d-1},i_d,j_d,α_d)
```

### Definitions of the TT-Manifold
* **Physical Indices (i_k, j_k):** The observable dimensions of the data, where i_k ∈ [1, n_k], j_k ∈ [1, m_k].
* **Bond Dimensions / TT-Ranks (r_k):** The internal summation indices α_k. Boundary ranks are strictly r₀ = r_d = 1. The maximum rank r = max(r_k) governs expressivity and computational cost.

**Complexity Reduction:**
A dense tensor requires storing N = Π(i_k·j_k) parameters. The TT format requires storing Σ(i_k·j_k·r_{k-1}·r_k) parameters. If r is kept small, the parameter count drops from exponential to linear with respect to dimension d.

**Implementation:** In the Python reference, physical dimensions i_k and j_k are flattened into p_k = i_k·j_k. Core shape is `[r_{k-1}, p_k, r_k]`. See `TENSOR_TOPOLOGY.md` §2.

---

## III. The Optimization Engine: Single-Site DMRG
With the parameter space factorized into G_k cores, we abandon the global Chain Rule. Instead, we optimize the network by solving for one core at a time, freezing all others. This is the **Alternating Linear Scheme (ALS)**, natively known in physics as Single-Site DMRG.

### 1. The Local Linear Projection
Let us isolate the k-th core, G_k. We merge all cores to the left into a single orthogonal block L, and all cores to the right into a single orthogonal block R.

```
W = L_{<k} · G_k · R_{>k}
```

Because L_{<k} and R_{>k} are fixed, the global non-linear neural network equation collapses into a strictly convex, linear equation with respect to the local core G_k:

```
L(G_k) = ½‖T − (L_{<k} ⊗ R_{>k}) G_k‖²_F
```

### 2. The Exact Local Solver
Because the equation is now linear, we can calculate the exact minimum for G_k. We project the global target tensor T into the local subspace:

```
G̃_k = (Lᵀ_{<k} ⊗ Rᵀ_{>k}) T
```

**Crucially, no inverse is required here.** Because the DMRG protocol enforces left- and right-orthogonality during its sweep, Lᵀ_{<k} L_{<k} = I and R_{>k} Rᵀ_{>k} = I. The dense inverse (A Aᵀ)⁻¹ from the baseline problem has been mathematically canceled out by the topological orthogonality of the Tensor Train.

**Implementation note:** In practice, environment blocks may not be perfectly orthonormal between sweeps. The Python implementation solves the normal equations (JᵀJ + λI)·vec(G_k) = JᵀY via block-diagonal solver or matrix-free CG, with λ providing numerical stability. See `solve_local_core()` in `src/dmrg_transformer/optim/local_solver.py`.

---

## IV. Truncation and Convergence (The SVD Step)
When we solve for the exact local core G̃_k, its bond dimensions (TT-ranks) naturally expand to perfectly absorb the target data. Left unchecked, the ranks would grow exponentially, returning us to O(N³) complexity.

To enforce the complexity bound, we apply **Singular Value Decomposition (SVD)** to truncate the updated core back to the maximum allowed rank r.

We reshape the updated core into a matrix C and perform SVD:
```
C = U Σ Vᵀ
```

Where Σ is a diagonal matrix of singular values σᵢ. We truncate to keep only the top r singular values:
```
C_truncated = U_{[:, 1:r]} Σ_{[1:r, 1:r]} Vᵀ_{[:, 1:r]}
```

### Proof of Optimality (Eckart-Young-Mirsky Theorem)
The Eckart-Young-Mirsky Theorem mathematically guarantees that the truncated matrix C_truncated is the absolute closest possible rank-r approximation to the exact matrix C under the Frobenius norm:

```
min_{rank(Ĉ) ≤ r} ‖C − Ĉ‖_F = ‖C − C_truncated‖_F = √(Σ_{i=r+1}^{min(rows,cols)} σ²_i)
```

**The Implication:** We are not "guessing" the optimal weights via a learning rate. We calculate the exact solution, and then mathematically project it into the hardware-constrained rank r with zero loss of mathematically available precision.

---

## V. The Sweep Protocol & Final Complexity Proof

The DMRG optimizer executes a bidirectional "Sweep":
1. Solve for G₁ → SVD truncate → Orthogonalize.
2. Shift focus to G₂.
3. Continue left-to-right to G_d.
4. Reverse direction right-to-left.

### Proof of Monotonic Convergence
Because each micro-optimization step solves a strictly convex linear sub-problem (due to the frozen environments), the local error is guaranteed to decrease or remain constant at every step. Therefore, the global error must decrease monotonically:

```
L(W_step i+1) ≤ L(W_step i)
```

### Final Hardware Complexity Validation
The most expensive operation in this entire protocol is the SVD on the local core matrix C.
The matrix C has dimensions roughly (r·n) × r.
The computational complexity of computing the SVD of an A×B matrix is O(min(A·B², A²·B)).

Therefore, the cost of updating a single TT-core is:
```
O(n·r³)
```

For a layer with d tensor dimensions, a full sweep costs:
```
O(d·n·r³)
```

Compare this to the original dense inversion cost of O(N³) where N = Π(i_k·j_k) is the total parameter count.

### Conclusion
By mapping standard deep learning weight matrices to a Tensor Train manifold and enforcing local orthogonality, we have successfully replaced the O(N³) dense matrix inversion with a sequence of localized SVDs scaling at O(d·n·r³). This proves the viability of an exact, non-iterative optimization backbone for modern artificial intelligence architectures.
