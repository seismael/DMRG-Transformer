# Future Work — Stacked-Block Pathways Beyond the v1 POC

**Status:** Out of scope for v1. Documented for follow-up research / contributions.  
**Context:** [REVIEW.md](REVIEW.md), [bench/PHASE0_DIAGNOSTIC.md](bench/PHASE0_DIAGNOSTIC.md), [bench/PHASE3_1_OUTCOME.md](bench/PHASE3_1_OUTCOME.md).

The v1 release of DMRG-Transformer ships a **verified gradient-free solver for a single TT-parameterized layer** (see `README.md` §3 — Gate 3 parity, 8–12× compression, real-task TT-MLP). Two systematic attempts to extend this to a **stacked-block Transformer encoder** were carried out in this codebase and produced clean negative results:

1. **Softmax attention block** ([bench/REAL_WORLD_TT_BLOCK.md](bench/REAL_WORLD_TT_BLOCK.md)): Q/K/V trust-region rejected 12/12 epochs across 5 blend settings; final 16 pp gap to Adam.
2. **Linear attention block** ([bench/PHASE3_1_OUTCOME.md](bench/PHASE3_1_OUTCOME.md)): synthetic-target unit tests pass, but real-task gap is +27 pp iso-time vs Adam-MSE with catastrophic divergence at ep10.

Both attempts converge on the same diagnosis: **per-layer local solvers (DMRG/ALS) cannot reconstruct the global loss landscape that backprop traverses end-to-end.** Two pathways from REVIEW.md remain mathematically viable for closing this gap; both require significant additional design and are listed here so a future contributor can pick them up.

---

## Option B — Pathway 1 + Pathway 3 (Linear Attention + ADMM Outer Loop)

### Idea
Keep the multilinear block from `nn/tt_linear_attention_block.py`. Wrap the per-layer DMRG sweeps in an **Alternating Direction Method of Multipliers (ADMM)** outer loop that introduces auxiliary "consensus" variables `z_ℓ` for each layer's output and dual variables `u_ℓ` enforcing `y_ℓ ≈ z_ℓ`. The inner DMRG sweep then solves for layer-`ℓ` cores with `z_ℓ + u_ℓ` as the local target — a target that is *jointly consistent* with the rest of the network instead of arriving via a single chain of pseudo-inverse pull-backs.

### Why it should work
ADMM converges for any closed convex problem (Boyd et al. 2011) and has been extended to bi-affine and weakly-non-convex problems with provable local convergence (Wang et al. 2019). On the multilinear block, each ADMM x-subproblem is a **single TT-Linear LSQ** — already solved exactly by DMRG (Phase D). The z-subproblem reduces to per-token activation projections (cheap). The dual update is a single residual addition. The outer loop gives the layers a way to **negotiate** instead of overwriting each other's targets.

### Cost
- Reintroduces hyperparameters (ρ, primal/dual residual tolerances). Partially weakens the "no learning rate" claim, though ρ in ADMM is far less brittle than an Adam learning rate (typically auto-adapted via residual balancing).
- Doubles per-epoch work (one DMRG sweep + one z/u step per layer per outer iteration).
- Convergence proofs for stacked TT manifolds are an open research question.

### Where to start
- Implement `dmrg_transformer/optim/admm_outer.py` with `ADMMOuter(layers, rho, tol)` and an `admm_step(X, Y)` that performs one full pass.
- Use `TTLinearAttentionBlock.dmrg_step(X, Y_admm)` as the x-subproblem; `Y_admm = z + u`.
- Diagnostic gate: on the Phase 3.1 sklearn-digits Tier 2 setup, ADMM must close the iso-time gap to ≤ 5 pp before scaling depth.

---

## Option C — Pathway 2 (Unified PEPS / Global Tensor Network)

### Idea
Stop treating the network as a stack of independently-solved layers. Represent the **entire encoder** as a single 2-D tensor network — Projected Entangled Pair States (PEPS) or a high-order Tensor Train where the time/sequence axis and the depth axis are both tensor-network bonds. Run one **global** DMRG sweep across this 2-D network, with environment blocks that simultaneously summarize "everything to the left in depth, everything below in sequence."

### Why it should work
This is the **only** approach that fundamentally addresses the layer-coupling problem. There is no longer a "layer-`ℓ` local target" that can drift — there is one global cost and one global sweep. Phase D's monotonic-decrease guarantee extends to PEPS sweeps under standard regularity assumptions.

### Cost
- PEPS contractions are `#P`-hard in general; tractable approximations (boundary MPS, CTMRG, simple update) carry their own truncation error budget.
- Memory scales with bond dimension `D` as `O(D^10)` for exact contractions — much worse than TT's `O(r^3)`.
- Multi-month research project. Likely requires the AGENTS Phase IV Rust + cuTensorNet microkernel before it is even profileable on consumer hardware.
- Risk: the contraction error might dominate the optimization gain, leaving us no better off than per-layer DMRG.

### Where to start
- Survey: Cirac, Pérez-García, Schuch, Verstraete, *Matrix product states and projected entangled pair states*, Rev. Mod. Phys. 93, 045003 (2021).
- Prototype: 2-block × 2-token PEPS contraction with `cuTensorNet`, validate against per-block DMRG on a TT-realizable target. If global PEPS sweep beats sequential per-block DMRG by ≥ 30 % MSE on this trivial setup, the approach is worth scaling.

---

## Option D — Hybrid: keep DMRG as a frozen-feature initializer, fine-tune with Adam

Pragmatic engineering option for users who want the compression benefit without the optimization risk. **Not** a research contribution, but worth noting:

- Use DMRG to fit each TT-Linear layer to its layer-wise target (current Phase D solver — verified, monotonic).
- Then unfreeze and fine-tune the full stacked network with a small number of Adam steps (e.g. 10–50, lr=1e-4) using standard backprop.
- This is analogous to the LoRA / QLoRA workflow: the heavy structural fit is gradient-free; only the inter-layer alignment uses gradients.
- Cost: violates AGENTS Constraint 1 ("no `loss.backward()`") in the fine-tune phase — but only there. The bulk of the parameters and the bulk of the wall time remain gradient-free.
- Diagnostic gate: should close the Phase 3.1 gap with < 100 Adam steps and < 5 % of the total wall budget.

---

## What is **not** being proposed as future work

- **Per-layer trust-region tuning, target-blend grid search, propagator damping schedules.** These were exhausted in the v1 attempts (see [bench/REAL_WORLD_TT_BLOCK_ATTN_ABLATION.md](bench/REAL_WORLD_TT_BLOCK_ATTN_ABLATION.md), [bench/REAL_WORLD_TT_BLOCK_DEPTH2_BLEND.md](bench/REAL_WORLD_TT_BLOCK_DEPTH2_BLEND.md)). They do not address the structural cause and will not close the gap.
- **Larger ranks, more sweeps, more epochs.** The Phase 3.1 divergence at ep10 indicates the failure mode is qualitative, not a budget shortage.

---

## Reproducing the negative results before starting

Before investing in Option B or C, a contributor should reproduce the v1 negative-result pipeline on their own hardware to confirm the failure mode is real and is consistent with what is documented:

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe scripts\train_real_world_tt_block_classifier.py
.venv\Scripts\python.exe scripts\train_real_world_linear_tt_block_classifier.py
```

Expected outputs:
- `bench/REAL_WORLD_TT_BLOCK.md` — softmax block, 16 pp gap, attn 0/12 acceptance
- `bench/REAL_WORLD_LIN_TT_BLOCK.md` — linear block, 27 pp iso-time gap, ep10 divergence
