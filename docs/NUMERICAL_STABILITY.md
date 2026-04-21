# NUMERICAL_STABILITY.md: Floating-Point Physics & Error Mitigation
**Version:** 1.0.0
**Domain:** GPU Hardware Constraints, Numerical Analysis, and Exception Handling

## 1. Objective
Pure mathematics assumes infinite precision. NVIDIA GPUs operate in quantized floating-point spaces (`float32`, `float64`, `bfloat16`). The DMRG Alternating Linear Scheme—specifically QR orthogonalization and Singular Value Decomposition (SVD)—is hypersensitive to numerical underflow, overflow, and rank collapse. 

This document dictates the strict defensive programming protocols the agent MUST implement to prevent catastrophic mathematical divergence.

## 2. Precision Casting Protocol
To balance execution speed with mathematical stability, the architecture mandates a mixed-precision state machine.

* **Forward Pass / Contractions:** Execution MUST occur in `float32` (or TensorFloat-32 / `TF32` on Ampere+ architectures) to maximize Tensor Core utilization.
* **Orthogonalization (QR Decomposition):** When calculating the $Q$ and $R$ matrices for environment block shifting, the input matrices MUST be cast to `float64` before execution.
  * *Reasoning:* Orthogonality (where $Q^T Q = I$) degrades rapidly in `float32` across deep networks. Loss of strict orthogonality will corrupt the exact local solver.
  * *Directive:* Cast to `float64` -> Execute QR -> Cast $Q$ back to `float32`.
* **SVD Execution:** Matricized core tensors $C$ MUST be evaluated in `float32`. If the condition number exceeds $10^4$, automatically upcast to `float64` for the decomposition step.

## 3. Tikhonov Regularization (Target Damping)
When projecting the global target $\mathcal{T}$ into the local subspace, or when using pseudo-inverses, the environment matrices can become ill-conditioned (singular), resulting in division-by-zero or NaN explosions.

* **The Fix:** Implement Tikhonov Regularization (Damping) on all covariance matrices before inversion or Least Squares solving.
* **Equation:** Replace $(X^T X)^{-1}$ with $(X^T X + \lambda I)^{-1}$
* **Hyperparameter:** Set the default damping factor $\lambda = 1e-5$. 
* *Agent Directive:* Expose $\lambda$ as a configurable parameter in the `DMRGOptimizer` struct. If NaNs are detected during the forward pass, dynamically increase $\lambda$ by a factor of $10$ and trigger a warning.

## 4. The SVD Convergence Fallback Hierarchy
GPU-accelerated SVD via `cuSOLVER` (`torch.linalg.svd` or `cp.linalg.svd`) uses the Divide-and-Conquer algorithm (`gesvdj` or `gesdd`). Highly correlated data manifolds will frequently cause this algorithm to fail to converge on the GPU, throwing a hardware-level exception.

**CRITICAL DIRECTIVE:** The agent MUST wrap every SVD operation in a strict `try-catch` fallback hierarchy. If the agent writes a raw `U, S, V = svd(C)` without this hierarchy, the implementation is rejected.

* **Tier 1 (Hardware Native):** Attempt standard GPU SVD.
* **Tier 2 (CPU Fallback):** Catch the non-convergence exception. Transfer the matrix $C$ to Host RAM (CPU). Execute LAPACK SVD (`scipy.linalg.svd(..., lapack_driver='gesdd')`). Transfer the resulting $U, \Sigma, V^T$ back to VRAM.
* **Tier 3 (Robust CPU Fallback):** If `gesdd` fails, fallback to the slower but unconditionally stable `gesvd` driver on the CPU.
* **Tier 4 (Algebraic Bypass):** If all SVDs fail due to catastrophic matrix scaling, apply a tiny random Gaussian noise matrix $\epsilon \sim \mathcal{N}(0, 1e-7)$ to $C$ to break symmetry, and retry Tier 1.

## 5. Target Bounding (Gradient Clipping Equivalent)
Because the DMRG-Transformer does not use a learning rate to dampen updates, a massive outlier in the training data will cause the exact solver to calculate a massive weight update, violently shifting the Tensor Train manifold.

* **The Fix:** Implement a Huber-style clamping function on the local target tensor $\mathcal{T}_l$ before executing the exact solver.
* *Agent Directive:* Clamp all elements of the target projection such that values exceeding $\pm 5.0$ standard deviations from the local batch mean are clipped. This acts as the mathematical equivalent of Gradient Clipping, ensuring the exact solver operates within a bounded, stable geometrical subspace.