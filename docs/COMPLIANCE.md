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
| §5 Per-head CUDA stream concurrency | 🟡 | Partial Python prototype only: `TTMultiHeadAttention.dmrg_step_projections` dispatches the independent `Q/K/V` projection sweeps to three `torch.cuda.Stream`s, but this is projection-level rather than per-head, omits `W_out`, and still ends with an explicit synchronize. Full event-based per-head orchestration remains Phase IV. |
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
| Adaptive rank scheduling | ✅ | `adaptive_rank` selection rule + tests in [tests/test_adaptive_rank.py](../tests/test_adaptive_rank.py); **wired through `DMRGOptimizer(adaptive_threshold=...)` → `solve_local_core(adaptive_threshold=...)` → local SVD truncation**, regression-guarded by [tests/test_adaptive_rank_wiring.py](../tests/test_adaptive_rank_wiring.py) (strict-threshold parity with fixed-rank baseline; loose-threshold pruning fires). Default behaviour (`adaptive_threshold=None`) is bit-exact with all prior benchmarks. |
| Multi-layer Transformer block (TTBlock = TTMHA + TTFFN + LN + residual) | 🟡 | C2 ✅ [src/dmrg_transformer/nn/tt_ffn.py](../src/dmrg_transformer/nn/tt_ffn.py) ([tests/test_tt_ffn.py](../tests/test_tt_ffn.py)); C3 ✅ [src/dmrg_transformer/nn/tt_block.py](../src/dmrg_transformer/nn/tt_block.py) Pre-LN with identity-initialized buffer-backed LN affine, **full Q/K/V/W_out + FFN update attempts**, optional LN affine OLS refits, and explicit attention-step diagnostics under per-substep trust-region accept/revert for the non-convex Q/K bilinear path ([tests/test_tt_block.py](../tests/test_tt_block.py), [tests/test_ttblock_ln_affine.py](../tests/test_ttblock_ln_affine.py)); C4 ✅ stacked end-to-end ([tests/test_tt_block_stacked_end_to_end.py](../tests/test_tt_block_stacked_end_to_end.py)). Q/K softmax pull-back primitives (`solve_attention_pattern_target`, `softmax_target_to_scores`, `project_through_qk_bilinear`) verified in [tests/test_target_propagator_extensions.py](../tests/test_target_propagator_extensions.py). |
| TargetPropagator: residual + LayerNorm pull-back | ✅ | `project_through_residual` + `project_through_layernorm` in [src/dmrg_transformer/propagation/target_propagator.py](../src/dmrg_transformer/propagation/target_propagator.py); round-trip tests in [tests/test_target_propagator_extensions.py](../tests/test_target_propagator_extensions.py) |
| Stacked TTBlock real-task validation | ✅ | [scripts/train_real_world_tt_block_classifier.py](../scripts/train_real_world_tt_block_classifier.py) → [bench/REAL_WORLD_TT_BLOCK.md](../bench/REAL_WORLD_TT_BLOCK.md). TT-DMRG reaches 0.7194 test acc on sklearn-digits-as-sequence vs Adam ~0.88 (~16 pp residual gap) with exact-LSQ input-projection updates plus the TTBlock exact-solver path. The dedicated ablation [scripts/analyze_tt_block_attention_rejections.py](../scripts/analyze_tt_block_attention_rejections.py) → [bench/REAL_WORLD_TT_BLOCK_ATTN_ABLATION.md](../bench/REAL_WORLD_TT_BLOCK_ATTN_ABLATION.md) shows that on this specific 1-block benchmark the Q/K/V substep is rejected `0/12` accepted epochs across `attn_target_blend ∈ {0.25, 0.10, 0.05, 0.02, 0.01}` while accuracy stays fixed at 0.7194. The follow-up probe [scripts/analyze_tt_block_attention_v_retarget.py](../scripts/analyze_tt_block_attention_v_retarget.py) → [bench/REAL_WORLD_TT_BLOCK_V_RETARGET.md](../bench/REAL_WORLD_TT_BLOCK_V_RETARGET.md) shows that re-deriving `V` from the **actual post-Q/K attention pattern** materially improves the epoch-1 local candidate (`2.365x → 1.023x` block-MSE ratio vs the pre-projection baseline) and is better than re-deriving from the idealized `Q/K` target pattern (`1.059x`), but still does not clear the trust-region. **Empirical finding**: per-token "detail-preserving" target propagation *regresses* the result — the gap is a **structural ceiling of the mean-pool-head architecture** (a single 16-dim constraint per example), not a propagator defect. Regression-guarded by [tests/test_real_world_tt_block_classifier.py](../tests/test_real_world_tt_block_classifier.py). |
| Language-modelling demo (WikiText-2) | ⏳ | Plan §D |
| **End-to-end real supervised training** (DMRG-trained 2-layer MLP on `sklearn.load_digits`, 80/20 stratified split, beats 80 % held-out test accuracy) | ✅ | [scripts/train_real_world_classifier.py](../scripts/train_real_world_classifier.py) → [bench/REAL_WORLD_MNIST.md](../bench/REAL_WORLD_MNIST.md); regression-guarded by [tests/test_real_world_classifier.py](../tests/test_real_world_classifier.py). Demonstrates real generalization (88.3 % test acc, 88-89 % prediction agreement with AdamW baselines), not synthetic curve-fitting. |

## What's deferred to Phase IV

Per session scope decisions captured in `/memories/session/plan.md`:

* **Rust crate** (`rust/dmrg-core/` with `maturin` + PyO3 + cuSOLVER + cuTensorNet) — MVP slice planned but not yet built. The Python `MemoryArena` is the contract the Rust impl must satisfy.
* **>80% Tensor Core utilisation claim** — requires `nvidia-smi`/`nsys` profiling in the Rust loop.
* **WikiText-2 head-to-head perplexity vs Adam** — requires Q/K softmax pull-back so the attention pattern actually adapts.
* **Whole-block rollback / deeper-stack damping validation** — current `TTBlock.dmrg_step` *does* run the full `Q/K/V/W_out + FFN` update attempt and optional LN affine OLS refits, but the reject/revert logic is still per non-convex or affine substep rather than a single whole-block snapshot. Synthetic depth-4 validation shows that lower damping (`target_blend≈0.3`) is materially more stable than the shallow-stack default `0.5`, but a small real-task depth-2 harness ([scripts/train_real_world_tt_block_depth2_blend.py](../scripts/train_real_world_tt_block_depth2_blend.py) → [bench/REAL_WORLD_TT_BLOCK_DEPTH2_BLEND.md](../bench/REAL_WORLD_TT_BLOCK_DEPTH2_BLEND.md)) does **not** transfer that win directly: `0.5` reaches 0.6667 best test acc vs 0.6083 for `0.3`, and both settings reject 100% of Q/K/V attempts. The 1-block ablation ([scripts/analyze_tt_block_attention_rejections.py](../scripts/analyze_tt_block_attention_rejections.py) → [bench/REAL_WORLD_TT_BLOCK_ATTN_ABLATION.md](../bench/REAL_WORLD_TT_BLOCK_ATTN_ABLATION.md)) shows the same pattern on the validated baseline: lowering `attn_target_blend` down to `0.01` leaves both rejection count (`0/12` accepted Q/K/V steps) and final accuracy (0.7194) unchanged. The new V-retarget probe ([scripts/analyze_tt_block_attention_v_retarget.py](../scripts/analyze_tt_block_attention_v_retarget.py) → [bench/REAL_WORLD_TT_BLOCK_V_RETARGET.md](../bench/REAL_WORLD_TT_BLOCK_V_RETARGET.md)) narrows the blocker further: re-deriving `V` from the **actual post-Q/K attention pattern** improves the local candidate sharply, but still stops just short of an accepted trust-region step. The next blocker is therefore still the real-task attention-path trust-region, not recent instrumentation alone. Full block-level rollback is still open work.
* **TTBlock LN affine (γ, β) updates** — ✅ closed. The block now stores LN γ, β as **buffers** (not `nn.Parameter`, preserving AGENTS Constraint 1) inside [src/dmrg_transformer/nn/tt_block.py](../src/dmrg_transformer/nn/tt_block.py) `_AffineLN` and exposes `enable_ln_affine=True` to opt in to a per-feature OLS update under a trust-region accept/revert rule (snapshot → fit → if global block MSE worsens, revert). Default (`enable_ln_affine=False`) is bit-exact with the previous frozen-LN behaviour, so existing benchmarks are unchanged. Regression-guarded by [tests/test_ttblock_ln_affine.py](../tests/test_ttblock_ln_affine.py) (init=identity LN, OLS recovers known (γ*, β*) to 1e-9, trust-region non-increase, buffers-not-parameters, and benchmark-style device placement for LN buffers).

These are explicitly tracked in the session plan and the README §9 limitations section.
