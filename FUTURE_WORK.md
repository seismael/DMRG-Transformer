# Future Work — Pathways Beyond the V1.1 PoC

**Status:** Identified, analyzed, and prioritized. Option B (ADMM) is the current development target.
**Context:** [REVIEW.md](REVIEW.md), [AGENTS.md](AGENTS.md).

---

The v1.1 release of DMRG-Transformer ships a verified gradient-free solver for TT-parameterized neural layers (see `README.md` — Gate 3 parity, 8-12× compression, 87% real-task accuracy). The identified limiting factor is **per-layer local solvers cannot reconstruct the global loss landscape that backprop traverses end-to-end**. The following pathways are proposed for closing this gap.

---

## Option B — Pathway 1 + Pathway 3: ADMM Outer Loop (IN DEVELOPMENT)

### Idea
Wrap the per-layer DMRG sweeps in an **Alternating Direction Method of Multipliers (ADMM)** outer loop. Introduce auxiliary "consensus" variables `z_ℓ` for each layer's output and dual variables `u_ℓ` enforcing `y_ℓ ≈ z_ℓ`. The inner DMRG sweep solves for layer-ℓ cores with `z_ℓ + u_ℓ` as the local target — a target that is *jointly consistent* with the rest of the network instead of arriving via a single chain of pseudo-inverse pull-backs.

### Why it should work
ADMM converges for any closed convex problem (Boyd et al. 2011) and has been extended to bi-affine and weakly non-convex problems with provable local convergence. On the multilinear block, each ADMM x-subproblem is a single TT-Linear LSQ — already solved exactly by DMRG. The z-subproblem reduces to per-token activation projections (convex combinations, O(d_ℓ) per layer). The dual update is a single residual addition.

### Cost
- Reintroduces hyperparameters (ρ penalty parameter, primal/dual residual tolerances). ρ in ADMM is far less brittle than an Adam learning rate (typically auto-adapted via residual balancing).
- Doubles per-epoch work (one DMRG sweep + one z/u step per layer per outer iteration).
- Convergence proofs for stacked TT manifolds with nonlinear activations (GELU) are novel.

### Diagnostic Gate
On the sklearn-digits Tier 2 setup (1× TTBlock, embed=16, rank=8), ADMM must close the iso-time gap from 10.8 pp to ≤ 5 pp before scaling depth.

### Where to start
- Implement `dmrg_transformer/optim/admm_outer.py` with `ADMMOuter(layers, rho, tol)`.
- Use existing `TTLinear.dmrg_step()` / `TTLinearAttentionBlock.dmrg_step()` as x-subproblem. `Y_admm = z + u`.
- Test on single TTLinear layer first (Gate A1), then TTFeedForward (Gate A2), then stacked cascade (Gate A3).

---

## Option C — Pathway 2: Unified PEPS / Global Tensor Network

### Idea
Stop treating the network as a stack of independently-solved layers. Represent the **entire encoder** as a single 2-D tensor network — Projected Entangled Pair States (PEPS) or a high-order Tensor Train where the time/sequence axis and the depth axis are both tensor-network bonds. Run one **global** DMRG sweep across this 2-D network, with environment blocks that simultaneously summarize "everything to the left in depth, everything below in sequence."

### Why it should work
This is the **only** approach that fundamentally addresses the layer-coupling problem. There is no longer a "layer-ℓ local target" that can drift — there is one global cost and one global sweep. Phase III's monotonic-decrease guarantee extends to PEPS sweeps under standard regularity assumptions.

### Cost
- PEPS contractions are #P-hard in general; tractable approximations (boundary MPS, CTMRG, simple update) carry their own truncation error budget.
- Memory scales with bond dimension D as O(D¹⁰) for exact contractions — much worse than TT's O(r³).
- Multi-month research project. Requires the Phase IV Rust + cuTensorNet microkernel before it is even profileable on consumer hardware.
- Risk: the contraction error might dominate the optimization gain.

### Where to start
- Survey: Cirac, Pérez-García, Schuch, Verstraete, *Matrix product states and projected entangled pair states*, Rev. Mod. Phys. 93, 045003 (2021).
- Prototype: 2-block × 2-token PEPS contraction, validate against per-block DMRG on a TT-realizable target. If global PEPS sweep beats sequential per-block DMRG by ≥ 30% MSE on this trivial setup, the approach is worth scaling.

---

## Option D — Hybrid: DMRG Initializer, Adam Fine-Tune

### Idea
Pragmatic engineering option for users who want the compression benefit without the optimization risk.

- Use DMRG to fit each TT-Linear layer to its layer-wise target (verified, monotonic).
- Then unfreeze and fine-tune the full stacked network with a small number of Adam steps (e.g. 10–50, lr=1e-4) using standard backprop.
- Analogous to LoRA / QLoRA workflow: the heavy structural fit is gradient-free; only the inter-layer alignment uses gradients.

### Cost
- Violates AGENTS Constraint 1 ("no `loss.backward()`") in the fine-tune phase — but only there. The bulk of the parameters and wall time remain gradient-free.
- Diagnostic gate: should close the Phase 3.1 gap with < 100 Adam steps and < 5% of the total wall budget.

---

## Option E — Decision-Boundary-Aware Targets

### Idea
Shift target propagation from Frobenius-norm minimization (feature-space targets) to logit-space targets that preserve classification decisions. Instead of propagating `Y_target - Y_curr` through layers, propagate targets that maximize inter-class margins. This directly addresses the "Exactness Paradox" identified in REVIEW.md.

### Why it should work
The current Frobenius targets are semantically destructive because they treat all feature dimensions equally. Margin-aware targets would prioritize the dimensions that separate classes, naturally aligning the Frobenius-optimal solution with the semantic optimum.

### Cost
- Requires extending `TargetPropagator` with a logit-space propagation path.
- The softmax inverse is non-trivial — working approximations exist via temperature-scaled logits.

---

## What is **not** being proposed

- **Per-layer trust-region grid searches** — Exhausted in the v1.1 attempts. The attention trust-region rejection rate is 100% across all blend settings because Frobenius-optimal targets are structurally wrong, not because the threshold is too tight.
- **Larger ranks, more sweeps, more epochs** — The divergence at epoch 10 in the linear attention variant indicates the failure mode is qualitative, not a budget shortage.
- **Propagator damping schedules** — Do not address the structural cause of inter-layer drift.

---

## Reproducing Current Results

Before investing in any pathway, a contributor should confirm the current baseline on their hardware:

```powershell
# Verify DMRG parses correctly:
uv run python -m pytest tests --no-header -q

# Run PoC entries:
uv run python scripts/poc_tt_mlp.py              # TT-MLP (88 % test acc)
uv run python scripts/poc_softmax_transformer.py # Softmax Transformer (87 % test acc)
uv run python scripts/poc_linear_transformer.py  # Linear Transformer (87 % test acc)

# Run Gate 3 proof:
uv run python scripts/run_gate3_proof.py
```
