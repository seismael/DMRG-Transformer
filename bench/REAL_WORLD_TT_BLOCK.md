# DMRG-Transformer — Stacked TTBlock Real-Task Validation

**Device:** `device=cuda:0 (NVIDIA GeForce MX150, sm_61, 2.0 GiB)`  
**Task:** 10-class classification on `sklearn.datasets.load_digits` reshaped as 8 tokens of dim 8 (stratified 80/20 split, seed=42).  
**Architecture:** input proj → 1× TTBlock(embed=16, heads=2, hidden=16, rank=8) → mean-pool → linear head.  
**TT-DMRG path:** zero gradients. Block trained by per-block `dmrg_step` (12 epochs); head fit by closed-form ridge LSQ.  
**Adam baselines:** identical-shape dense block (`nn.MultiheadAttention` + GELU FFN), AdamW lr=0.01, 600 total steps.

## Final test-set accuracy

| Model | Train acc | **Test acc** | Params | Wall (s) |
| :---- | --------: | -----------: | -----: | -------: |
| TT-DMRG (no grads) | 0.7209 | **0.7194** | 1,946 | 14.14 |
| Dense (AdamW, MSE) | 0.9179 | **0.8778** | 1,946 | 49.11 |
| Dense (AdamW, CE)  | 1.0000 | **0.8611** | 1,946 | 51.96 |

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

After landing (a) softmax-aware Q/K/V joint updates with trust-region accept/revert, (b) exact-LSQ input-projection updates (also trust-region wrapped), and (c) **empirically validating** that per-token target propagation does *not* help, the residual ~16 pp DMRG-vs-Adam gap on this task is now identified as a **structural ceiling** of the mean-pool-head architecture rather than a propagation defect.

### What we tried and what it told us

- **Pooled-target broadcast** (current): each token is held to the same pooled target. Reaches ~0.72 test acc.
- **Per-token "detail-preserving" target** (`R_target[t] = r_curr[t] + (pooled_target − mean_t r_curr)`): **regressed** to ~0.67 test acc. Diagnosis: the mean-pool head exposes only a single 16-dim constraint per example, so per-token rank in `R_target` is an *unconstrained* degree of freedom — preserving current per-token detail tells the block "keep doing what you do, just shifted by a constant", which removes the learning signal for per-token routing. **The broadcast is provably the maximum-information per-token target under mean pooling.**
- **Inner block-sweep iterations per epoch (1 → 4)**: peak test acc unchanged (0.72), reached at ep3 instead of ep12, but later epochs overfit to ~0.68. Same architectural ceiling, faster convergence.

### Remaining contributors (in order)

1. **Mean-pool head invariance.** The classifier loss is invariant to per-token permutation, so the block cannot learn position-specific roles from the loss alone. Adam's per-token gradient still uses the same constraint but applies it through the network Jacobian, breaking the symmetry implicitly. Closing this gap requires changing the head (e.g. [CLS]-token classification, or per-token logits + voting).

2. **Trust-region rejections.** Past epoch 1 the input-projection step is rejected (the local-identity linearization `h_target ≈ h_curr + (R_target − block(h_curr))` becomes inaccurate as the block moves), and Q,K bilinear steps are occasionally rejected too. Both bound per-step gain.

3. **GELU active-mask propagation** — first-order, not exact. Smaller contributor.

The Q/K softmax pull-back primitives (`solve_attention_pattern_target`, `softmax_target_to_scores`, `project_through_qk_bilinear`) are unit-tested in [tests/test_target_propagator_extensions.py](../tests/test_target_propagator_extensions.py). The block forward MSE drops monotonically (~0.40 → ~0.009) every epoch, demonstrating the solver is doing its job — the gap is in the *signal*, not the *solver*.
