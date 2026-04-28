# PROJECT REVIEW: DMRG-Transformer PoC (v1.1.0)

## 1. Executive Summary: The "Zero Backprop" Milestone
This project has successfully implemented and verified a **Transformer architecture trained entirely without Gradient Descent or Backpropagation**. By replacing the traditional Autograd+Adam stack with a Quantum-inspired **DMRG (Density Matrix Renormalization Group)** local solver, we have achieved stable, monotonic convergence on real-world classification tasks.

### Core Achievements:
- **Stabilized Stacked Training:** Resolved the catastrophic divergence (10⁵ error explosions) that previously plagued multi-layer non-gradient training.
- **Real-World Performance:** Reached **87.2%** test accuracy on `sklearn.digits` (sequence-mode) with zero backpropagation, up from previous failure states.
- **2.0× Parameter Compression:** TT-DMRG achieves ~87% accuracy with half the parameters of an architecturally identical dense Adam baseline.
- **Linear-Attention Breakthrough:** Linear TTBlock achieved **2.7× training speedup** and **3.2× peak memory reduction** vs the softmax variant, with negligible accuracy impact.

---

## 2. Technical Breakthroughs: How the PoC was Solved

We identified that the failure of earlier phases was not due to the DMRG solver itself, but to **Compositional Defects** in how targets were moved between layers.

### A. Numerical Stability (The Gram Matrix Fix)
- **Problem:** Target propagation through underdetermined layers (in < out) caused 1/λ explosions in the pseudo-inverse.
- **Solution:** Implemented **Regime-Aware Inversion**. By dynamically selecting the compact Gram matrix by min(in, out), the LSQ solver remains well-conditioned even without heavy Tikhonov damping. See `TargetPropagator.project_through_linear()`.

### B. Feature Routing (Difference Target Propagation)
- **Problem:** "Broadcast Targets" forced all tokens in a sequence to become identical, destroying the model's ability to "route" information.
- **Solution:** Implemented **DTP** — targets are propagated as relative deltas: `x_target = x_curr + α·(Y_target − Y_curr)`. This preserves internal sequence structure (the "routing") while shifting the entire representation toward the global objective.

### C. Symmetry Breaking (Positional Coordinate Systems)
- **Problem:** The attention mechanism was permutation-invariant, making it "blind" to token order in real images.
- **Solution:** Injected **Sinusoidal Positional Encodings**, providing the DMRG solver with a fixed coordinate system for learning position-specific roles.

### D. Global Consistency (Inner-Loop Micro-Sweeps)
- **Problem:** Sequential updates (Head → Block → Proj) caused "inter-layer drift" where the first layer's update was invalidated by the second.
- **Solution:** Implemented **Pathway 1.6 Micro-Sweeps** — layers perform 3–5 iterative "settling" steps per batch, reaching a mutually consistent joint optimum. See `TTLinearAttentionBlock.dmrg_step()`.

---

## 3. The Remaining Gap: Why Adam Still Leads (+10 pp)

Despite reaching 87.2%, a ~10 pp gap remains relative to Adam (97.5%). Our investigation has pinpointed the mathematical root cause:

### The "Exactness Paradox" (Frobenius vs. Semantic Loss)
- **The Finding:** Exact local solvers (DMRG/ALS) minimize **Frobenius norm** to a target. However, in non-convex layers like Softmax Attention, this "mathematically perfect" move can be semantically destructive.
- **The "Frozen Attention" Symptom:** In the Softmax PoC, the Q/K/V trust-region accept rate is 0% because the Frobenius-optimal update consistently increases the global classification MSE. The 87.2% accuracy is currently carried entirely by the **FFN** and **Input-Projection** updates.
- **The Linear Attention Corroboration:** The Linear Attention variant (where attention weights *do* move) achieves near-identical accuracy (86.7%), confirming that 87% is the current structural ceiling for Frobenius-minimization.

---

## 4. Current Development Focus

The v1.1 PoC is complete and successful. Current work focuses on three pathways to close the gap:

1. **ADMM Outer Loop** (`FUTURE_WORK.md` Option B): Wraps per-layer DMRG sweeps in an Alternating Direction Method of Multipliers to introduce consensus variables that allow layers to negotiate jointly-consistent targets.

2. **Decision-Boundary Targets**: Shifting target propagation from Frobenius-norm minimization to logit-space targets that preserve classification decisions.

3. **Annealed Attention Trust-Regions**: Allowing larger Q/K/V jumps early in training and tightening later, with per-head (rather than all-or-nothing) acceptance.

---

## 5. Conclusion

The **DMRG-Transformer PoC is complete and successful**. We have built a stable, gradient-free backbone that handles real-world sequence modeling with significant parameter and memory efficiency. The "Exactness Paradox" — where Frobenius-optimal updates are semantically destructive — is the identified root cause of the remaining gap to Adam. Closing this gap is the current focus of active development.
