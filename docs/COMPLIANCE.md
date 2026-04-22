# COMPLIANCE.md — Specification ↔ Implementation Traceability

This document is the authoritative cross-reference between the architecture
specifications in `docs/` and the implemented code under `src/`, `tests/`,
`scripts/`, and `bench/`. It is regenerated whenever any specification or
gate criterion changes.

**Status legend:** ✅ PASS · 🟡 PARTIAL · ⏳ DEFERRED (Phase IV) · ❌ FAIL

## AGENTS.md prime constraints

| # | Constraint | Status | Evidence |
| - | ---------- | :----: | -------- |
| 1 | No gradients (`loss.backward()` banned) | ✅ | [tests/test_constraints.py](../tests/test_constraints.py) AST scan |
| 2 | No iterative optimisers (Adam/SGD banned in `src/`) | ✅ | [tests/test_constraints.py](../tests/test_constraints.py) |
| 3 | No dense inversions beyond TT-rank bounds | ✅ | [src/dmrg_transformer/optim/local_solver.py](../src/dmrg_transformer/optim/local_solver.py) — block-diagonal solver, `H` of size `(r·i_k·r)²` only |
| 4 | Memory mutability / in-place env-block updates | 🟡 | Python prototype: [src/dmrg_transformer/core/arena.py](../src/dmrg_transformer/core/arena.py); Rust+CUDA arena ⏳ |

## AGENTS.md gates

| Gate | Spec | Status | Evidence |
| :--- | :--- | :----: | -------- |
| 1 | TT-SVD reconstruction error matches Eckart–Young bound | ✅ | [tests/test_gate1_reconstruction.py](../tests/test_gate1_reconstruction.py) |
| 2 | Left/right orthogonalization yields `LᵀL = I` to ≤1e-7 | ✅ | [tests/test_gate2_orthogonality.py](../tests/test_gate2_orthogonality.py) |
| 3 | DMRG MSE matches dense lstsq on rank-bounded targets, in O(d·n·r³) | ✅ | [tests/test_gate3_exact_parity.py](../tests/test_gate3_exact_parity.py), [bench/GATE3_PROOF.md](../bench/GATE3_PROOF.md) |
| 4 | Rust microkernel: cuTensorNet+cuSOLVER+arena, 1000-sweep zero-leak, >80% TC util | ⏳ | Phase IV; arena Python prototype + zero-alloc test in [tests/test_memory_arena.py](../tests/test_memory_arena.py) is the partial gate |

## TENSOR_TOPOLOGY.md

| Spec section | Status | Evidence |
| :---         | :----: | -------- |
| §2 Core shape `[r_{k-1}, p_k, r_k]`, boundary ranks 1 | ✅ | [src/dmrg_transformer/tt/tensor_train.py](../src/dmrg_transformer/tt/tensor_train.py) |
| §3 Forward einsum | ✅ | `TensorTrain.contract_forward` |
| §4 L/R environment block construction | ✅ | [src/dmrg_transformer/tt/environments.py](../src/dmrg_transformer/tt/environments.py) |
| §5 Local-core normal-equation projection | ✅ | `_build_block_normal_equations` (block-diagonal in `j_k`) |
| §6 SVD reshape protocol (matricize → SVD → truncate → unfold) | ✅ | `solve_local_core` |

## NUMERICAL_STABILITY.md

| Spec section | Status | Evidence |
| :---         | :----: | -------- |
| §2 float32 forward / float64 QR/SVD upcast | ✅ | [src/dmrg_transformer/core/qr.py](../src/dmrg_transformer/core/qr.py), `precision.py` |
| §3 Tikhonov damping λ default + 6-step NaN escalation | ✅ | [tests/test_numerical_stability.py](../tests/test_numerical_stability.py) |
| §4 4-tier SVD fallback (GPU → gesdd → gesvd → noise+retry) | ✅ | All four tiers exercised in [tests/test_numerical_stability.py](../tests/test_numerical_stability.py) |
| §5 ±5σ Huber clamp on targets | ✅ | `_huber_clamp` in `local_solver.py` |

## MEMORY_ARENA.md

| Spec section | Status | Evidence |
| :---         | :----: | -------- |
| §2 Zero-allocation prime directive | 🟡 | Python prototype satisfies <50 allocations across 1000 cycles ([tests/test_memory_arena.py](../tests/test_memory_arena.py)); Rust+CUDA enforcement ⏳ |
| §3 Rust ownership / CUDA FFI | ⏳ | Phase IV |
| §4 Ping-pong double-buffering | 🟡 | Python `MemoryArena.swap_left/right` matches contract; integration into `DMRGOptimizer.sweep` pending |
| §5 Per-head CUDA stream concurrency | 🟡 | Python `torch.cuda.Stream` dispatch in `TTMultiHeadAttention.dmrg_step_projections`; Rust orchestration ⏳ |
| §6 cuSOLVER workspace pre-allocation | 🟡 | `MemoryArena.svd_workspace()` reserved in Python; cuSOLVER direct binding ⏳ |

## BENCHMARK.md

| Spec section | Status | Evidence |
| :---         | :----: | -------- |
| 1024×1024 / batch=2048 / rank=32 runnable | ✅ | [bench/HEADLINE.md](../bench/HEADLINE.md) — runs in 265s on MX150, peak 2.2 GB |
| 3-way comparison (Adam vs Dense vs DMRG) | ✅ | [src/dmrg_transformer/bench/benchmark.py](../src/dmrg_transformer/bench/benchmark.py) |
| Wall-time with `torch.cuda.synchronize()` | ✅ | `_sync()` in `benchmark.py` |
| Warmup + multi-seed mean ± std | ✅ | `warmup`/`seeds` parameters on each runner |
| Peak GPU memory tracked | ✅ | `_peak_mem_gb()` via `torch.cuda.max_memory_allocated` |
| FLOPs reported (analytic) | ✅ | `flops_per_call` field on `BenchmarkResult` |
| Rank/MSE Pareto sweep | ✅ | [scripts/run_pareto.py](../scripts/run_pareto.py) → `bench/PARETO.md` |
| Spec headline "DMRG matches dense MSE on `sin(XW)+noise` in 0.05s" | ❌ | **Reconciled** — see [docs/BENCHMARK.md](BENCHMARK.md) reconciliation note. Headline holds only on TT-rank-bounded targets ([bench/GATE3_PROOF.md](../bench/GATE3_PROOF.md)); on full-rank targets DMRG produces a Pareto-optimal rank-`r` point, not the dense optimum. |

## ARCHITECTURE.md / BLUEPRINT.md / SOLVER_MATH.md

| Item | Status | Evidence |
| :--- | :----: | -------- |
| `ITensorTrain`, `ITargetPropagator`, `IDMRGOptimizer` interfaces | ✅ | [src/dmrg_transformer/core/interfaces.py](../src/dmrg_transformer/core/interfaces.py) |
| Execution pipeline (forward → target → sweep → exact solve → truncate → shift) | ✅ | `DMRGOptimizer.sweep` |
| TargetPropagator usage in multi-layer cascade | ✅ | [tests/test_target_propagation_cascade.py](../tests/test_target_propagation_cascade.py) |
| TTMultiHeadAttention forward + per-projection DMRG | ✅ | [tests/test_mha_consistency.py](../tests/test_mha_consistency.py) |
| Adaptive rank scheduling | 🟡 | `adaptive_rank` selection rule + tests in [tests/test_adaptive_rank.py](../tests/test_adaptive_rank.py); integration into `DMRGOptimizer.sweep` deferred (single-flag wiring) |
| Multi-layer Transformer block (TTBlock = TTMHA + TTFFN + LN + residual) | 🟡 | C2 ✅ [src/dmrg_transformer/nn/tt_ffn.py](../src/dmrg_transformer/nn/tt_ffn.py) ([tests/test_tt_ffn.py](../tests/test_tt_ffn.py)); C3 ✅ [src/dmrg_transformer/nn/tt_block.py](../src/dmrg_transformer/nn/tt_block.py) Pre-LN with frozen LN affine and W_out + FFN sweeps ([tests/test_tt_block.py](../tests/test_tt_block.py)); C4 ✅ stacked end-to-end ([tests/test_tt_block_stacked_end_to_end.py](../tests/test_tt_block_stacked_end_to_end.py)). **Honest deferral:** Q/K projections frozen this slice — softmax pull-back deferred to follow-up. |
| TargetPropagator: residual + LayerNorm pull-back | ✅ | `project_through_residual` + `project_through_layernorm` in [src/dmrg_transformer/propagation/target_propagator.py](../src/dmrg_transformer/propagation/target_propagator.py); round-trip tests in [tests/test_target_propagator_extensions.py](../tests/test_target_propagator_extensions.py) |
| Stacked TTBlock real-task validation | ✅ | [scripts/train_real_world_tt_block_classifier.py](../scripts/train_real_world_tt_block_classifier.py) → [bench/REAL_WORLD_TT_BLOCK.md](../bench/REAL_WORLD_TT_BLOCK.md). TT-DMRG hits 65.6% test acc on sklearn-digits-as-sequence vs Adam ~87% (~22 pp gap, root-caused to frozen Q/K + frozen input projection — see bench file's *Honest gap analysis* section). Regression-guarded by [tests/test_real_world_tt_block_classifier.py](../tests/test_real_world_tt_block_classifier.py). |
| Language-modelling demo (WikiText-2) | ⏳ | Plan §D |
| **End-to-end real supervised training** (DMRG-trained 2-layer MLP on `sklearn.load_digits`, 80/20 stratified split, beats 80 % held-out test accuracy) | ✅ | [scripts/train_real_world_classifier.py](../scripts/train_real_world_classifier.py) → [bench/REAL_WORLD_MNIST.md](../bench/REAL_WORLD_MNIST.md); regression-guarded by [tests/test_real_world_classifier.py](../tests/test_real_world_classifier.py). Demonstrates real generalization (88.3 % test acc, 88-89 % prediction agreement with AdamW baselines), not synthetic curve-fitting. |

## What's deferred to Phase IV

Per session scope decisions captured in `/memories/session/plan.md`:

* **Rust crate** (`rust/dmrg-core/` with `maturin` + PyO3 + cuSOLVER + cuTensorNet) — MVP slice planned but not yet built. The Python `MemoryArena` is the contract the Rust impl must satisfy.
* **>80% Tensor Core utilisation claim** — requires `nvidia-smi`/`nsys` profiling in the Rust loop.
* **WikiText-2 head-to-head perplexity vs Adam** — requires Q/K softmax pull-back so the attention pattern actually adapts.
* **Q/K softmax pull-back** — currently `TTBlock.dmrg_step` only updates `W_out` + FFN; Q/K projections stay frozen at init. This is the dominant root cause of the measured DMRG-vs-Adam accuracy gap on stacked-block tasks. The **softmax-aware V pull-back primitive** is implemented and tested as `TargetPropagator.project_through_attention_v` (round-trip verified, see [tests/test_target_propagator_extensions.py](../tests/test_target_propagator_extensions.py)), but is *intentionally not wired* into `dmrg_step` until Q/K co-update lands — wiring V alone (with frozen random Q/K) destabilizes the block on the real-task validation (measured collapse 65.6% → 31.7%); the V update needs an adapting attention pattern to be useful.
* **TTBlock LN affine (γ, β) updates** — frozen at γ=1, β=0 this slice (LN inversion is then exact). LSQ update for affine params is a minor follow-up.

These are explicitly tracked in the session plan and the README §9 limitations section.
