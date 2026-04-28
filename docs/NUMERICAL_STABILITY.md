# NUMERICAL_STABILITY.md: Floating-Point Physics & Error Mitigation
**Version:** 1.1.0
**Domain:** GPU Hardware Constraints, Numerical Analysis, and Exception Handling

## 1. Objective
Pure mathematics assumes infinite precision. NVIDIA GPUs operate in quantized floating-point spaces (float32, float64). The DMRG Alternating Linear Scheme — specifically QR orthogonalization and Singular Value Decomposition (SVD) — is hypersensitive to numerical underflow, overflow, and rank collapse.

This document dictates the strict defensive programming protocols implemented to prevent catastrophic mathematical divergence.

**Implementation status:** All protocols below are fully implemented in the Python reference. See `docs/COMPLIANCE.md` for traceability matrix.

## 2. Precision Casting Protocol
To balance execution speed with mathematical stability, the architecture mandates a mixed-precision state machine.

* **Forward Pass / Contractions:** Execution occurs in `float32` (or TensorFloat-32 / TF32 on Ampere+ architectures) to maximize Tensor Core utilization.
* **Orthogonalization (QR Decomposition):** Input matrices MUST be cast to `float64` before execution.
  * *Reasoning:* Orthogonality (where QᵀQ = I) degrades rapidly in float32 across deep networks. Loss of strict orthogonality corrupts the exact local solver.
  * *Directive:* Cast to float64 → Execute QR → Cast Q back to float32.
* **SVD Execution:** Matricized core tensors C are evaluated in float32. If the condition number exceeds 10⁴, automatically upcast to float64 for the decomposition step.

**Implementation:** `src/dmrg_transformer/core/qr.py` (qr_f64), `src/dmrg_transformer/core/precision.py` (needs_f64_upcast, to_f64, to_f32).

## 3. Tikhonov Regularization (Target Damping)
When projecting the global target T into the local subspace, or when using pseudo-inverses, the environment matrices can become ill-conditioned (singular), resulting in division-by-zero or NaN explosions.

* **The Fix:** Tikhonov Regularization (Damping) on all covariance matrices before inversion or least-squares solving.
* **Equation:** Replace (XᵀX)⁻¹ with (XᵀX + λI)⁻¹
* **Hyperparameter:** Default damping factor λ = 1e-5.
* **NaN escalation:** If NaNs are detected during the forward pass, dynamically increase λ by a factor of 10 and retry (up to 6 escalations).

**Implementation:** `DMRGOptimizer.lam`, `TargetPropagator.lam`, `_huber_clamp()` in `local_solver.py`, 6-step NaN escalation in `solve_local_core()`.

## 4. The SVD Convergence Fallback Hierarchy
GPU-accelerated SVD via cuSOLVER (`torch.linalg.svd`) uses the Divide-and-Conquer algorithm (`gesdd`). Highly correlated data manifolds will frequently cause this algorithm to fail to converge on the GPU.

**The implemented 4-tier fallback hierarchy:**

* **Tier 1 (Hardware Native):** Attempt standard GPU SVD (`torch.linalg.svd`).
* **Tier 2 (CPU Fallback — gesdd):** Catch non-convergence exception. Transfer matrix C to Host RAM (CPU). Execute LAPACK SVD via SciPy (`scipy.linalg.svd(..., lapack_driver='gesdd')`). Transfer U, Σ, Vᵀ back to VRAM.
* **Tier 3 (Robust CPU Fallback — gesvd):** If `gesdd` fails, fallback to the slower but unconditionally stable `gesvd` driver on the CPU.
* **Tier 4 (Algebraic Bypass):** If all SVDs fail due to catastrophic matrix scaling, apply Gaussian noise ε ~ N(0, 1e-7) to C to break symmetry, retry Tier 1.

**Implementation:** `src/dmrg_transformer/core/svd.py` — `robust_svd()` function. This is the single authorized call site for all SVD operations.

## 5. Target Bounding (Gradient Clipping Equivalent)
Because the DMRG-Transformer does not use a learning rate to dampen updates, a massive outlier in the training data will cause the exact solver to calculate a massive weight update, violently shifting the Tensor Train manifold.

* **The Fix:** Huber-style clamping on the local target tensor T_ℓ before executing the exact solver.
* **Directive:** Clamp all elements of the target projection such that values exceeding ±5.0 standard deviations from the local batch mean are clipped. This acts as the mathematical equivalent of Gradient Clipping.

**Implementation:** `_huber_clamp()` in `src/dmrg_transformer/optim/local_solver.py` — clamped to ±5σ per batch mean.
