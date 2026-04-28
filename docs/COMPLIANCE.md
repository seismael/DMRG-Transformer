# COMPLIANCE.md — Specification ↔ Implementation Traceability

This document is the authoritative cross-reference between the architecture
specifications in `docs/` and the implemented code under `src/`, `tests/`,
`scripts/`, and `bench/`.

**Status legend:** ✅ PASS · 🟡 PARTIAL · ⏳ DEFERRED (Phase IV) · ❌ FAIL

## AGENTS.md Prime Constraints

| # | Constraint | Status | Evidence |
| - | ---------- | :---: | -------- |
| 1 | No gradients (`loss.backward()` banned) | ✅ | `tests/test_constraints.py` — AST scan across entire `src/` |
| 2 | No iterative optimisers (Adam/SGD banned in `src/`) | ✅ | `tests/test_constraints.py` — AST scan |
| 3 | No dense inversions beyond TT-rank bounds | ✅ | `src/dmrg_transformer/optim/local_solver.py` — block-diagonal solver, H of size `(r·i_k·r)²` only |
| 4 | Memory mutability / in-place env-block updates | 🟡 | Python prototype: `EnvironmentCache` + `MemoryArena`; Rust+CUDA arena ⏳ |

## AGENTS.md Gates

| Gate | Spec | Status | Evidence |
| :--- | :--- | :---: | -------- |
| 1 | TT-SVD reconstruction error matches Eckart–Young bound | ✅ | `tests/test_gate1_reconstruction.py` |
| 2 | Left/right orthogonalization yields `LᵀL = I` to ≤1e-7 | ✅ | `tests/test_gate2_orthogonality.py` |
| 3 | DMRG MSE matches dense lstsq on rank-bounded targets, in O(d·n·r³) | ✅ | `tests/test_gate3_exact_parity.py`, `bench/GATE3_PROOF.md` |
| 4 | Rust microkernel: cuTensorNet+cuSOLVER+arena, 1000-sweep zero-leak, >80% TC util | ⏳ | Phase IV; Python prototype + zero-alloc test in `tests/test_memory_arena.py` is partial |

## TENSOR_TOPOLOGY.md

| Spec Section | Status | Evidence |
| :--- | :---: | -------- |
| §2 Core shape `[r_{k-1}, p_k, r_k]`, boundary ranks r₀=r_d=1 | ✅ | `src/dmrg_transformer/tt/tensor_train.py` |
| §3 Forward einsum | ✅ | `TensorTrain.contract_forward` |
| §4 L/R environment block construction | ✅ | `src/dmrg_transformer/tt/environments.py` |
| §5 Local-core normal-equation projection | ✅ | `_build_block_normal_equations` (block-diagonal in j_k) |
| §6 SVD reshape protocol (matricize → SVD → truncate → unfold) | ✅ | `solve_local_core` |

## NUMERICAL_STABILITY.md

| Spec Section | Status | Evidence |
| :--- | :---: | -------- |
| §2 float32 forward / float64 QR/SVD upcast | ✅ | `src/dmrg_transformer/core/qr.py`, `precision.py` |
| §3 Tikhonov damping λ default + 6-step NaN escalation | ✅ | `tests/test_numerical_stability.py` |
| §4 4-tier SVD fallback (GPU → gesdd → gesvd → noise+retry) | ✅ | All four tiers exercised in `tests/test_numerical_stability.py` |
| §5 ±5σ Huber clamp on targets | ✅ | `_huber_clamp` in `local_solver.py` |

## MEMORY_ARENA.md

| Spec Section | Status | Evidence |
| :--- | :---: | -------- |
| §2 Zero-allocation prime directive | 🟡 | Python prototype: <50 allocations across 1000 cycles (`tests/test_memory_arena.py`); Rust+CUDA enforcement ⏳ |
| §3 Rust ownership / CUDA FFI | ⏳ | Phase IV |
| §4 Ping-pong double-buffering | 🟡 | Python `MemoryArena.swap_left/right` matches contract; integration into `DMRGOptimizer.sweep` pending |
| §5 Per-head CUDA stream concurrency | 🟡 | Python prototype dispatches Q/K/V projection sweeps to 3 `torch.cuda.Stream`s; full event-based per-head orchestration is Phase IV |
| §6 cuSOLVER workspace pre-allocation | 🟡 | `MemoryArena.svd_workspace()` reserved in Python; cuSOLVER direct binding ⏳ |

## BENCHMARK.md

| Spec Section | Status | Evidence |
| :--- | :---: | -------- |
| 1024×1024 / batch=2048 / rank=32 runnable on 2 GiB GPU | ✅ | Runs via `scripts/run_headline_benchmark.py` — 265s, peak 2.2 GiB |
| 3-way comparison (Adam vs Dense vs DMRG) | ✅ | `src/dmrg_transformer/bench/benchmark.py` |
| Wall-time with `torch.cuda.synchronize()` | ✅ | `_sync()` in `benchmark.py` |
| Warmup + multi-seed mean ± std | ✅ | `warmup`/`seeds` parameters on each runner |
| Peak GPU memory tracked | ✅ | `_peak_mem_gb()` via `torch.cuda.max_memory_allocated` |
| FLOPs reported (analytic) | ✅ | `flops_per_call` field on `BenchmarkResult` |
| Rank/MSE Pareto sweep | ✅ | `scripts/run_pareto.py` |
| DMRG matches dense MSE on TT-rank-bounded targets | ✅ | `bench/GATE3_PROOF.md` |
| DMRG matches dense MSE on full-rank targets | ❌ | **Not mathematically possible** — DMRG produces the Pareto-optimal rank-r approximation (see `docs/BENCHMARK.md` reconciliation note) |

## ARCHITECTURE.md / BLUEPRINT.md / SOLVER_MATH.md

| Item | Status | Evidence |
| :--- | :---: | -------- |
| `ITensorTrain`, `ITargetPropagator`, `IDMRGOptimizer` Protocol interfaces | ✅ | `src/dmrg_transformer/core/interfaces.py` |
| Execution pipeline (forward → target → sweep → exact solve → truncate → shift) | ✅ | `DMRGOptimizer.sweep` |
| TargetPropagator usage in multi-layer cascade | ✅ | `tests/test_target_propagation_cascade.py` |
| TTMultiHeadAttention forward + per-projection DMRG | ✅ | `tests/test_mha_consistency.py` |
| Adaptive rank scheduling | ✅ | `adaptive_rank` selection rule + tests in `tests/test_adaptive_rank.py`, `tests/test_adaptive_rank_wiring.py` |
| TTFeedForward DMRG update (fc2→pullback→fc1) | ✅ | `src/dmrg_transformer/nn/tt_ffn.py`, `tests/test_tt_ffn.py` |
| TTBlock Pre-LN DMRG update (Q/K/V/W_out + FFN) | 🟡 | Implemented in `src/dmrg_transformer/nn/tt_block.py` with trust-region accept/revert; Q/K/V attention substeps reject ~100% on softmax variant |
| TTLinearAttentionBlock DMRG update | ✅ | `src/dmrg_transformer/nn/tt_linear_attention_block.py`, `tests/test_tt_linear_attention_block_dmrg.py` |
| TargetPropagator: residual + LayerNorm + attention pull-backs | ✅ | `project_through_residual`, `project_through_layernorm`, `project_through_attention_v`, `solve_attention_pattern_target`, `project_through_qk_bilinear` all in `src/dmrg_transformer/propagation/target_propagator.py` |
| Stacked TTBlock end-to-end | ✅ | `tests/test_tt_block_stacked_end_to_end.py` |
| Stacked TTBlock real-task validation (sklearn digits) | ✅ | `scripts/poc_softmax_transformer.py` → `bench/REAL_WORLD_TT_BLOCK.md` (87.2% test acc) |
| Linear attention real-task validation | ✅ | `scripts/poc_linear_transformer.py` → `bench/REAL_WORLD_LIN_TT_BLOCK.md` (86.7% test acc) |
| End-to-end real supervised training (TT-MLP on sklearn digits) | ✅ | `scripts/poc_tt_mlp.py` → `bench/REAL_WORLD_MNIST.md` (88.3% test acc) |

## What's Deferred to Phase IV

* **Rust crate** (`rust/dmrg-core/` with `maturin` + PyO3 + cuSOLVER + cuTensorNet) — not yet built. Python `MemoryArena` is the contract.
* **>80% Tensor Core utilisation claim** — requires `nvidia-smi`/`nsys` profiling in the Rust loop.
* **WikiText-2 language modelling demo** — requires scaling beyond MX150 memory budget.

## What's In Development

* **ADMM Outer Loop** — wrapping per-layer DMRG sweeps in an ADMM consensus loop to resolve inter-layer drift. See `FUTURE_WORK.md` Option B for specification.
