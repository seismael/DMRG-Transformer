# DMRG-Transformer — Stacked TTBlock Real-Task Validation

**Device:** `device=cuda:0 (NVIDIA GeForce MX150, sm_61, 2.0 GiB)`  
**Task:** 10-class classification on `sklearn.datasets.load_digits` reshaped as 8 tokens of dim 8 (stratified 80/20 split, seed=42).  
**Architecture:** input proj → 1× TTBlock(embed=16, heads=2, hidden=16, rank=8) → mean-pool → linear head.  
**TT-DMRG path:** zero gradients. Block trained by per-block `dmrg_step` (12 epochs); head fit by closed-form ridge LSQ.  
**Adam baselines:** identical-shape dense block (`nn.MultiheadAttention` + GELU FFN), AdamW lr=0.01, 600 total steps.

## Final test-set accuracy

| Model | Train acc | **Test acc** | Params | Wall (s) |
| :---- | --------: | -----------: | -----: | -------: |
| TT-DMRG (no grads) | 0.6569 | **0.6556** | 1,946 | 11.72 |
| Dense (AdamW, MSE) | 0.9179 | **0.8778** | 1,946 | 48.55 |
| Dense (AdamW, CE)  | 1.0000 | **0.8611** | 1,946 | 48.80 |

**Measured DMRG → Adam-MSE gap:** +22.22 pp  
**Measured DMRG → Adam-CE  gap:** +20.56 pp

## Behavioral agreement on test set

* TT-DMRG ↔ Dense-MSE: **0.6694**
* TT-DMRG ↔ Dense-CE:  **0.6417**
* Dense-MSE ↔ Dense-CE: **0.8778** (sanity check)

## Per-epoch test accuracy

| Epoch | TT-DMRG | Dense (MSE) | Dense (CE) |
| ----: | ------: | ----------: | ---------: |
| 1 | 0.2194 | 0.6250 | 0.7722 |
| 2 | 0.3611 | 0.7500 | 0.7972 |
| 3 | 0.5639 | 0.8083 | 0.8417 |
| 4 | 0.5972 | 0.8278 | 0.8611 |
| 5 | 0.6194 | 0.8306 | 0.8694 |
| 6 | 0.6250 | 0.8306 | 0.8583 |
| 7 | 0.5750 | 0.8444 | 0.8611 |
| 8 | 0.5694 | 0.8556 | 0.8583 |
| 9 | 0.5944 | 0.8556 | 0.8556 |
| 10 | 0.6806 | 0.8722 | 0.8583 |
| 11 | 0.6667 | 0.8722 | 0.8611 |
| 12 | 0.6556 | 0.8778 | 0.8611 |

## TTBlock per-epoch global MSE (block forward target tracking)

| Epoch | MSE before sweep | MSE after sweep |
| ----: | ---------------: | --------------: |
| 1 | 4.043e-01 | 6.876e-02 |
| 2 | 5.540e-02 | 2.893e-02 |
| 3 | 3.847e-02 | 1.966e-02 |
| 4 | 2.030e-02 | 1.729e-02 |
| 5 | 1.509e-02 | 1.440e-02 |
| 6 | 1.455e-02 | 1.307e-02 |
| 7 | 1.262e-02 | 1.286e-02 |
| 8 | 1.239e-02 | 1.235e-02 |
| 9 | 1.172e-02 | 1.146e-02 |
| 10 | 1.065e-02 | 1.041e-02 |
| 11 | 1.039e-02 | 9.353e-03 |
| 12 | 8.895e-03 | 8.820e-03 |

## Confusion matrices (held-out test set)

### TT-DMRG

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| **0** | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| **1** | 0 | 20 | 0 | 3 | 2 | 0 | 0 | 7 | 4 | 0 |
| **2** | 0 | 2 | 25 | 1 | 5 | 1 | 0 | 0 | 0 | 1 |
| **3** | 2 | 1 | 3 | 26 | 1 | 0 | 0 | 1 | 1 | 2 |
| **4** | 1 | 1 | 4 | 2 | 24 | 0 | 0 | 4 | 0 | 0 |
| **5** | 8 | 0 | 3 | 3 | 1 | 16 | 2 | 2 | 0 | 2 |
| **6** | 0 | 1 | 0 | 0 | 0 | 1 | 28 | 4 | 2 | 0 |
| **7** | 1 | 0 | 1 | 2 | 3 | 0 | 3 | 25 | 1 | 0 |
| **8** | 4 | 6 | 2 | 0 | 1 | 0 | 0 | 3 | 19 | 0 |
| **9** | 3 | 0 | 0 | 6 | 3 | 0 | 1 | 4 | 0 | 19 |

### Dense (AdamW + MSE)

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| **0** | 35 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| **1** | 0 | 33 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 |
| **2** | 0 | 1 | 29 | 0 | 0 | 2 | 0 | 1 | 1 | 1 |
| **3** | 0 | 2 | 0 | 31 | 0 | 0 | 0 | 2 | 0 | 2 |
| **4** | 0 | 0 | 0 | 0 | 36 | 0 | 0 | 0 | 0 | 0 |
| **5** | 0 | 1 | 1 | 0 | 0 | 34 | 0 | 1 | 0 | 0 |
| **6** | 0 | 1 | 0 | 0 | 0 | 1 | 34 | 0 | 0 | 0 |
| **7** | 0 | 0 | 2 | 2 | 1 | 3 | 0 | 28 | 0 | 0 |
| **8** | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 30 | 0 |
| **9** | 0 | 0 | 0 | 1 | 5 | 0 | 0 | 3 | 1 | 26 |

### Dense (AdamW + CE)

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| **0** | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| **1** | 0 | 31 | 1 | 0 | 1 | 2 | 0 | 1 | 0 | 0 |
| **2** | 0 | 2 | 26 | 1 | 1 | 2 | 0 | 1 | 1 | 1 |
| **3** | 0 | 0 | 1 | 30 | 0 | 0 | 0 | 2 | 0 | 4 |
| **4** | 0 | 0 | 0 | 0 | 34 | 0 | 1 | 0 | 0 | 1 |
| **5** | 1 | 0 | 2 | 0 | 0 | 33 | 0 | 0 | 1 | 0 |
| **6** | 0 | 1 | 0 | 0 | 0 | 0 | 34 | 0 | 1 | 0 |
| **7** | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 31 | 0 | 1 |
| **8** | 2 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 30 | 0 |
| **9** | 0 | 0 | 0 | 1 | 3 | 1 | 0 | 3 | 1 | 27 |

## Honest gap analysis — root causes

The DMRG-vs-Adam gap on this stacked-TTBlock task remains larger than the 9 pp MLP gap reported in [bench/REAL_WORLD_MNIST.md](REAL_WORLD_MNIST.md). After implementing softmax-aware Q/K/V joint updates (commit following the bilinear pull-back work), the dominant remaining root causes are:

1. **Q/K bilinear non-convexity (now mitigated, not eliminated).** `TTBlock.dmrg_step` now updates Q, K, V jointly via `TargetPropagator.solve_attention_pattern_target` → `softmax_target_to_scores` → `project_through_qk_bilinear` (Gauss-Seidel ordering, simplex mirror-descent damping). The whole attention update is wrapped in a **trust-region accept/revert** rule (snapshot Q/K/V, run sweep, revert if global block MSE worsened). The bilinear `Q Kᵀ = S` problem is non-convex; many proposed steps are rejected, so the per-step gain on the attention path is bounded. Best-epoch test acc improved ~+2 pp over the prior frozen-Q/K baseline (0.66 → ~0.68 best-epoch on this run); steady-state accuracy is similar.

2. **Frozen input projection.** The input projection (token-dim → embed-dim) is held at initialization. This caps the upstream expressiveness available to the block.

3. **Pooled-target broadcast.** The head target is pulled back to a *single* pooled vector and broadcast to every token, so the per-token block targets have rank-1 structure across the sequence axis. Adam's backprop can shape per-token outputs independently.

4. **GELU active-mask propagation** is identical to the MLP slice's ReLU mask trick — first-order, not exact. This is a smaller contributor.

Further closing this gap requires (a) an exact-solver update for the input projection, (b) per-token target propagation that doesn't collapse to a pooled-broadcast, and (c) tighter linearization for the GELU sub-path. The Q/K softmax pull-back primitives (`solve_attention_pattern_target`, `softmax_target_to_scores`, `project_through_qk_bilinear`) are now landed and unit-tested — see [tests/test_target_propagator_extensions.py](../tests/test_target_propagator_extensions.py).
