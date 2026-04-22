# DMRG-Transformer — Stacked TTBlock Real-Task Validation

**Device:** `device=cuda:0 (NVIDIA GeForce MX150, sm_61, 2.0 GiB)`  
**Task:** 10-class classification on `sklearn.datasets.load_digits` reshaped as 8 tokens of dim 8 (stratified 80/20 split, seed=42).  
**Architecture:** input proj → 1× TTBlock(embed=16, heads=2, hidden=16, rank=8) → mean-pool → linear head.  
**TT-DMRG path:** zero gradients. Block trained by per-block `dmrg_step` (12 epochs); head fit by closed-form ridge LSQ.  
**Adam baselines:** identical-shape dense block (`nn.MultiheadAttention` + GELU FFN), AdamW lr=0.01, 600 total steps.

## Final test-set accuracy

| Model | Train acc | **Test acc** | Params | Wall (s) |
| :---- | --------: | -----------: | -----: | -------: |
| TT-DMRG (no grads) | 0.7209 | **0.7194** | 1,946 | 14.16 |
| Dense (AdamW, MSE) | 0.9179 | **0.8778** | 1,946 | 48.56 |
| Dense (AdamW, CE)  | 1.0000 | **0.8611** | 1,946 | 50.35 |

**Measured DMRG → Adam-MSE gap:** +15.83 pp  
**Measured DMRG → Adam-CE  gap:** +14.17 pp

## Behavioral agreement on test set

* TT-DMRG ↔ Dense-MSE: **0.7389**
* TT-DMRG ↔ Dense-CE:  **0.7167**
* Dense-MSE ↔ Dense-CE: **0.8778** (sanity check)

## Per-epoch test accuracy

| Epoch | TT-DMRG | Dense (MSE) | Dense (CE) |
| ----: | ------: | ----------: | ---------: |
| 1 | 0.6361 | 0.6250 | 0.7722 |
| 2 | 0.6611 | 0.7500 | 0.7972 |
| 3 | 0.6583 | 0.8083 | 0.8417 |
| 4 | 0.7000 | 0.8278 | 0.8611 |
| 5 | 0.7000 | 0.8306 | 0.8694 |
| 6 | 0.6889 | 0.8306 | 0.8583 |
| 7 | 0.6944 | 0.8444 | 0.8611 |
| 8 | 0.7028 | 0.8556 | 0.8583 |
| 9 | 0.7111 | 0.8556 | 0.8556 |
| 10 | 0.7167 | 0.8722 | 0.8583 |
| 11 | 0.7111 | 0.8722 | 0.8611 |
| 12 | 0.7194 | 0.8778 | 0.8611 |

## TTBlock per-epoch global MSE (block forward target tracking)

| Epoch | MSE before sweep | MSE after sweep |
| ----: | ---------------: | --------------: |
| 1 | 1.219e+00 | 9.093e-02 |
| 2 | 9.480e-02 | 3.349e-02 |
| 3 | 2.703e-02 | 2.098e-02 |
| 4 | 3.040e-02 | 1.864e-02 |
| 5 | 3.865e-02 | 1.722e-02 |
| 6 | 1.916e-02 | 1.610e-02 |
| 7 | 1.465e-02 | 1.513e-02 |
| 8 | 1.326e-02 | 1.201e-02 |
| 9 | 1.325e-02 | 9.650e-03 |
| 10 | 1.110e-02 | 1.080e-02 |
| 11 | 1.331e-02 | 9.989e-03 |
| 12 | 1.052e-02 | 9.575e-03 |

## Confusion matrices (held-out test set)

### TT-DMRG

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| **0** | 33 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 |
| **1** | 0 | 32 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
| **2** | 0 | 2 | 24 | 0 | 0 | 4 | 3 | 1 | 0 | 1 |
| **3** | 2 | 6 | 3 | 18 | 1 | 1 | 0 | 1 | 0 | 5 |
| **4** | 1 | 1 | 4 | 0 | 27 | 1 | 0 | 1 | 1 | 0 |
| **5** | 0 | 0 | 1 | 0 | 1 | 26 | 0 | 1 | 0 | 8 |
| **6** | 0 | 1 | 0 | 0 | 0 | 2 | 30 | 2 | 1 | 0 |
| **7** | 0 | 0 | 1 | 1 | 0 | 2 | 6 | 23 | 2 | 1 |
| **8** | 3 | 5 | 2 | 2 | 1 | 0 | 0 | 1 | 20 | 1 |
| **9** | 1 | 0 | 3 | 1 | 1 | 1 | 0 | 3 | 0 | 26 |

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

After landing (a) softmax-aware Q/K/V joint updates with trust-region accept/revert and (b) exact-LSQ input-projection updates (also trust-region wrapped), the residual DMRG-vs-Adam gap on this task is now dominated by the items below. Note: the dense Adam-MSE baseline still reaches **0.88 test acc**; the TT-DMRG path is at **~0.72**, narrowing the gap from the original 22 pp (frozen Q/K + frozen input proj) to ~16 pp.

1. **Pooled-target broadcast.** The head target is pulled back to a *single* pooled vector and broadcast to every token, so the per-token block targets have rank-1 structure across the sequence axis. Adam's backprop can shape per-token outputs independently. This is now the single largest information bottleneck.

2. **Trust-region rejections on Q,K bilinear path.** The Q,K joint update is non-convex (mirror-descent simplex damping + Gauss-Seidel ordering); the wrapper reverts steps that increase block MSE. Rejection rate climbs over training as the block approaches its local minimum, bounding per-step gain.

3. **Trust-region rejections on input projection.** Empirically the input-proj exact-LSQ step is accepted in epoch 1 (large gain) and then reverted in subsequent epochs — the local-identity linearization `h_target ≈ h_curr + (R_target − block(h_curr))` becomes inaccurate once the block has been swept. Sharper input-proj propagation requires an exact pull-back through the *current* block, which is the same open problem as per-token pooled-target inversion.

4. **GELU active-mask propagation** is identical to the MLP slice's ReLU mask trick — first-order, not exact. Smaller contributor.

Further closing this gap requires (a) per-token target propagation (invert `mean_pool` with explicit per-token degrees of freedom) and (b) iterating the input-proj / block sweep on the same epoch under a joint trust region. Both are scoped follow-ups; the propagator and block APIs are now in place.
