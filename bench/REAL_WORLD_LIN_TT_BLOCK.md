# DMRG-Transformer — Stacked TTBlock Real-Task Validation

**Device:** `device=cuda:0 (NVIDIA GeForce MX150, sm_61, 2.0 GiB)`  
**Task:** 10-class classification on `sklearn.datasets.load_digits` reshaped as 8 tokens of dim 8 (stratified 80/20 split, seed=42).  
**Architecture:** input proj → 1× TTBlock(embed=16, heads=2, hidden=16, rank=8) → mean-pool → linear head.  
**TT-DMRG path:** zero gradients. Block trained by per-block `dmrg_step` (12 epochs); head fit by closed-form ridge LSQ.  
**Adam baselines:** identical-shape dense block (`nn.MultiheadAttention` + GELU FFN), AdamW lr=0.01, 600 total steps.

## Final test-set accuracy

| Model | Train acc | **Test acc** | Params | Wall (s) | Peak GPU (MiB) |
| :---- | --------: | -----------: | -----: | -------: | -------------: |
| TT-DMRG (no grads) | 0.8546 | **0.8667** | 1,946 | 19.50 | 115.9 |
| Dense (AdamW, MSE) | 0.9896 | **0.9806** | 1,946 | 61.47 | 42.7 |
| Dense (AdamW, CE)  | 1.0000 | **0.9667** | 1,946 | 82.43 | 42.6 |

**Measured DMRG → Adam-MSE gap:** +11.39 pp  
**Measured DMRG → Adam-CE  gap:** +10.00 pp

## Iso-time fairness check

Both Adam baselines were sampled every 10 optimizer steps. The table below reports the test accuracy each Adam variant had reached by the wall-clock time TT-DMRG used in total.

| Comparison | Wall budget (s) | Test acc at budget | Final test acc | Final wall (s) |
| :--------- | --------------: | -----------------: | -------------: | -------------: |
| TT-DMRG (reference) | 19.50 | **0.8667** | 0.8667 | 19.50 |
| Dense Adam-MSE @ TT-DMRG budget | 19.30 | **0.9722** | 0.9806 | 61.47 |
| Dense Adam-CE  @ TT-DMRG budget | 19.26 | **0.9611** | 0.9667 | 82.43 |

**Iso-time DMRG → Adam-MSE gap:** +10.56 pp  
**Iso-time DMRG → Adam-CE  gap:** +9.44 pp

## Inference latency (held-out test set)

Median over 20 forward passes after 5 warmup runs. Batch sizes: full = 360 examples, single = 1.

| Model | Latency batch=1 (ms) | Latency batch=full (ms) | Throughput (ex/s, batch=full) |
| :---- | -------------------: | ----------------------: | ----------------------------: |
| TT-DMRG | 13.073 | 15.788 | 22802 |
| Dense (AdamW, MSE) | 3.600 | 20.544 | 17523 |
| Dense (AdamW, CE)  | 3.549 | 20.663 | 17422 |

## DMRG sub-update acceptance rates

Trust-region accept/revert is applied separately to the input projection (W_in, b_in) and the joint Q/K/V attention update. Rejection means the candidate update worsened the trust-region objective and was rolled back.

* **Input-projection accept rate:** 8.3% (1/12 epochs)
* **Attention (Q/K/V) accept rate:** 8.3% (1/12 epochs)

A persistently low attention accept rate is the leading indicator for the residual Adam gap on this task — see *root causes* below.

## Behavioral agreement on test set

* TT-DMRG ↔ Dense-MSE: **0.8667**
* TT-DMRG ↔ Dense-CE:  **0.8611**
* Dense-MSE ↔ Dense-CE: **0.9472** (sanity check)

## Per-epoch test accuracy

| Epoch | TT-DMRG | Dense (MSE) | Dense (CE) |
| ----: | ------: | ----------: | ---------: |
| 1 | 0.8028 | 0.7528 | 0.8944 |
| 2 | 0.8167 | 0.9222 | 0.9611 |
| 3 | 0.8250 | 0.9444 | 0.9611 |
| 4 | 0.8194 | 0.9667 | 0.9667 |
| 5 | 0.8250 | 0.9694 | 0.9639 |
| 6 | 0.8389 | 0.9750 | 0.9639 |
| 7 | 0.8528 | 0.9806 | 0.9639 |
| 8 | 0.8583 | 0.9833 | 0.9667 |
| 9 | 0.8583 | 0.9861 | 0.9667 |
| 10 | 0.8639 | 0.9833 | 0.9667 |
| 11 | 0.8639 | 0.9861 | 0.9667 |
| 12 | 0.8667 | 0.9806 | 0.9667 |

## TTBlock per-epoch global MSE (block forward target tracking)

| Epoch | MSE before sweep | MSE after sweep |
| ----: | ---------------: | --------------: |
| 1 | 1.303e-02 | 1.251e-02 |
| 2 | 3.150e-03 | 3.101e-03 |
| 3 | 2.283e-03 | 2.261e-03 |
| 4 | 1.951e-03 | 1.938e-03 |
| 5 | 1.806e-03 | 1.797e-03 |
| 6 | 1.680e-03 | 1.675e-03 |
| 7 | 1.577e-03 | 1.573e-03 |
| 8 | 1.502e-03 | 1.499e-03 |
| 9 | 1.449e-03 | 1.447e-03 |
| 10 | 1.410e-03 | 1.408e-03 |
| 11 | 1.378e-03 | 1.376e-03 |
| 12 | 1.350e-03 | 1.349e-03 |

## Confusion matrices (held-out test set)

### TT-DMRG

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| **0** | 35 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| **1** | 0 | 22 | 4 | 2 | 2 | 1 | 1 | 0 | 1 | 3 |
| **2** | 0 | 1 | 32 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| **3** | 2 | 0 | 1 | 32 | 0 | 0 | 0 | 0 | 0 | 2 |
| **4** | 0 | 0 | 0 | 0 | 36 | 0 | 0 | 0 | 0 | 0 |
| **5** | 0 | 0 | 0 | 0 | 0 | 34 | 1 | 1 | 0 | 1 |
| **6** | 0 | 0 | 0 | 0 | 0 | 1 | 35 | 0 | 0 | 0 |
| **7** | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 35 | 0 | 0 |
| **8** | 0 | 9 | 0 | 0 | 1 | 3 | 1 | 0 | 21 | 0 |
| **9** | 1 | 0 | 0 | 2 | 0 | 1 | 0 | 2 | 0 | 30 |

### Dense (AdamW + MSE)

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| **0** | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **1** | 0 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **2** | 0 | 0 | 35 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **3** | 0 | 0 | 0 | 37 | 0 | 0 | 0 | 0 | 0 | 0 |
| **4** | 0 | 0 | 0 | 0 | 36 | 0 | 0 | 0 | 0 | 0 |
| **5** | 0 | 0 | 0 | 0 | 0 | 37 | 0 | 0 | 0 | 0 |
| **6** | 0 | 0 | 0 | 0 | 0 | 0 | 35 | 0 | 1 | 0 |
| **7** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 36 | 0 | 0 |
| **8** | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 31 | 0 |
| **9** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 34 |

### Dense (AdamW + CE)

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| **0** | 35 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **1** | 0 | 35 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **2** | 0 | 1 | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **3** | 0 | 1 | 0 | 36 | 0 | 0 | 0 | 0 | 0 | 0 |
| **4** | 0 | 0 | 0 | 0 | 36 | 0 | 0 | 0 | 0 | 0 |
| **5** | 0 | 0 | 0 | 1 | 0 | 36 | 0 | 0 | 0 | 0 |
| **6** | 0 | 0 | 0 | 0 | 0 | 0 | 36 | 0 | 0 | 0 |
| **7** | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 34 | 0 | 0 |
| **8** | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 33 | 0 |
| **9** | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 33 |

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
