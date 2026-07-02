# Graph Report - DMRG-Transformer  (2026-05-08)

## Corpus Check
- 87 files · ~2,120,878 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 977 nodes · 1523 edges · 107 communities (50 shown, 57 thin omitted)
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 379 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8dedc244`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 94|Community 94]]
- [[_COMMUNITY_Community 95|Community 95]]
- [[_COMMUNITY_Community 96|Community 96]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 100|Community 100]]
- [[_COMMUNITY_Community 101|Community 101]]
- [[_COMMUNITY_Community 102|Community 102]]
- [[_COMMUNITY_Community 103|Community 103]]
- [[_COMMUNITY_Community 104|Community 104]]
- [[_COMMUNITY_Community 105|Community 105]]
- [[_COMMUNITY_Community 106|Community 106]]

## God Nodes (most connected - your core abstractions)
1. `TargetPropagator` - 81 edges
2. `require_cuda()` - 56 edges
3. `TTLinear` - 33 edges
4. `TTBlock` - 29 edges
5. `ADMMOuter` - 27 edges
6. `OptimizationBenchmark` - 21 edges
7. `DMRGOptimizer` - 21 edges
8. `TTFeedForward` - 19 edges
9. `TensorTrain` - 19 edges
10. `robust_svd()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `pytest_collection_modifyitems()` --calls--> `cuda_available()`  [INFERRED]
  tests/conftest.py → src/dmrg_transformer/core/device.py
- `pytest_configure()` --calls--> `require_cuda()`  [INFERRED]
  tests/conftest.py → src/dmrg_transformer/core/device.py
- `device()` --calls--> `require_cuda()`  [INFERRED]
  tests/conftest.py → src/dmrg_transformer/core/device.py
- `device()` --calls--> `require_cuda()`  [INFERRED]
  tests/test_tt_linear_attention_block_dmrg.py → src/dmrg_transformer/core/device.py
- `device()` --calls--> `require_cuda()`  [INFERRED]
  tests/test_tt_linear_attention_forward.py → src/dmrg_transformer/core/device.py

## Communities (107 total, 57 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (66): Pre-LN Transformer encoder block (TT-MHA + TT-FFN with optional LN affine)., Pre-LN Transformer encoder block (TT-MHA + TT-FFN with optional LN affine)., TTBlock, ADMMOuter, Alternating Direction Method of Multipliers for a chain of layers.      Each lay, _build_tt_native_target(), _direct_dmrg_mse(), _factor_square() (+58 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (46): BenchmarkResult, execute(), _factor_for_tt(), _factor_pair(), OptimizationBenchmark, Reproduction of ``docs/BENCHMARK.md`` — the three-way optimizer runoff.  Invoc, Choose a TT factorization adapted to the size of ``n``.      For small ``n`` (, Adam on ``W = U @ V`` with ``U: (in, r), V: (r, out)``.          This is the * (+38 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (45): adaptive_rank(), Choose the smallest rank ``r`` whose discarded tail mass is below     ``rel_thr, _apply_J(), _apply_JT(), _build_block_normal_equations(), _build_jacobian(), _build_normal_equations(), _huber_clamp() (+37 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (30): PositionalEncoding, Embedding utilities for the DMRG-Transformer., Sinusoidal positional encoding (fixed).      PE(pos, 2i) = sin(pos / 10000^(2i, Add positional encoding to input.          Args:             x: [batch, seq_l, accuracy(), fit_head(), forward(), load_data() (+22 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (32): dmrg_step(), forward_with_cache(), pullback_target(), ``TTLinearAttentionBlock`` — Pre-LN block with TT linear attention.  Compositi, Return per-head Q, K, V, phiQ, phiK, w (= phiQ·phiKᵀ), denom, context., Per-(B,H) ridge LSQ for V given fixed w (i.e. fixed phiQ, phiK).          ``w`, Pre-LN Transformer block with TT linear attention + TT-FFN., TTLinearAttentionBlock (+24 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (30): DMRGOptimizer, DMRG optimizer implementing ``IDMRGOptimizer`` (ARCHITECTURE.md §4.3).  The :c, Full bidirectional sweep: L→R then R→L.          Ensures the TT is properly ga, Exact-solver replacement for SGD/Adam.      Args:         max_rank: TT-rank b, _init_tt(), _make_low_rank_target(), Adaptive-rank wiring through ``DMRGOptimizer.sweep`` (plan §C5 closure).  Vali, Random small-init TT in the requested factorization. (+22 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (30): qr_f64(), qr_f64_strict(), QR with mandatory float64 cast (NUMERICAL_STABILITY.md §2).  This module is th, Compute Q, R with float64 internal precision; result returned in ``matrix.dtype`, Same as :func:`qr_f64` but returns float64 Q and R for orthogonality tests., _make_random_tt(), Validation Gate 2 (AGENTS.md §3, Phase II).  After a left-orthogonalization sw, Construct a random TT directly in the requested dtype (bypasses SVD/QR cascade). (+22 more)

### Community 7 - "Community 7"
Cohesion: 0.08
Nodes (22): Dataset, eval_ppl(), load_wikitext2(), main(), Train TT-GPT2 on cached WikiText-2 (character-level, zero backprop).  Uses froze, WikiText2CharDataset, _perplexity(), B3: End-to-end TT-GPT2 LM training with frozen-head + block DMRG.  Key insight f (+14 more)

### Community 8 - "Community 8"
Cohesion: 0.1
Nodes (27): _apply_tt_noise(), _es_round_qk(), ESDMRGHybrid, ESRoundReport, ES/DMRG Hybrid — Evolution Strategies for attention Q/K training.  EGGROLL (arXi, Sample independent Gaussian noise for each TT core.      Returns a dict ``{core_, Add scaled noise to each TT core in-place., Subtract scaled noise from each TT core in-place. (+19 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (19): dump_coverage_sidecar(), iso_time_lookup(), measure_inference_latency(), Shared GPU/wall/inference instrumentation for the Tier-1/2/3 runners.  Pulled, Median forward latency in milliseconds, with CUDA sync per call., Return (test_acc, wall) for the last sample with wall <= ``target_wall``., Write ``bench/_coverage/<tier>.json`` for the matrix aggregator., read_peak_mem_mib() (+11 more)

### Community 10 - "Community 10"
Cohesion: 0.1
Nodes (11): Layer-wise target propagator (ARCHITECTURE.md §4.2).  Replaces the Backpropaga, Pull a target back through a residual connection ``y = x + f(x)``.          Gi, Pull a target back through ``LayerNorm`` using current-row statistics., Compute a local target for a layer given a global target and its forward output., Pull a context target back to a per-head V target through ``A @ V``., Map a target attention pattern ``A_target`` to a score target ``S``., Decoupled bilinear pull-back of a score target ``S = Q K^T``.          Given a, Solve ``current_layer_out ≈ global_target`` as a least-squares residual. (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (17): _build_residual_cascade(), _chain_mse(), dmrg_step(), _factor_square(), forward(), pullback_target(), Validation Gate A3b — ADMM on a **residual** TTLinear cascade.  Unlike Gate A3 (, Build a ground-truth residual cascade and a trainable stack.      Returns ``(X, (+9 more)

### Community 12 - "Community 12"
Cohesion: 0.13
Nodes (17): _cpu_allowed(), cuda_available(), default_device(), default_dtype(), GPU-only device policy for DMRG-Transformer.  AGENTS.md and ARCHITECTURE.md ma, Return whether a working CUDA device is visible to PyTorch., Return ``cuda:0`` or raise a descriptive ``RuntimeError``.      This is the pr, Alias for :func:`require_cuda` — kept for readability at call sites. (+9 more)

### Community 13 - "Community 13"
Cohesion: 0.18
Nodes (16): ADMMReport, ADMMState, _blend_target(), _global_mse(), _init_states(), _pullback_through(), ADMM outer loop for inter-layer consensus (FUTURE_WORK.md Option B).  Wraps per-, Adjust ρ when primal / dual residuals are out of balance. (+8 more)

### Community 14 - "Community 14"
Cohesion: 0.11
Nodes (18): Tests for ``TargetPropagator.project_through_residual / _layernorm``.  Verifie, REVIEW.md Issue E: the Woodbury push-through path (d_h < L_k) must     produce, ``softmax(softmax_target_to_scores(A)) == A`` (gauge-invariance check)., Underdetermined regime (L_k < d_h): Q* won't equal Q_true (many valid     Q* ex, Overdetermined regime (L_k >= d_h): the system Q* K^T = S has a unique     solu, Symmetric overdetermined K test (L_q >= d_h)., LN with non-trivial γ, β: ``y = γ * normalize(x) + β`` must round-trip., Given A and a context produced as ``A @ V_true``, the V pull-back must     reco (+10 more)

### Community 15 - "Community 15"
Cohesion: 0.12
Nodes (8): IDMRGOptimizer, ITargetPropagator, ITensorTrain, Strict OOD interfaces (Protocols) mirroring ARCHITECTURE.md Rust traits.  Thes, Geometric encapsulation of a factorized weight space.      Mirrors ``ITensorTr, Replacement for the Backpropagation Chain Rule (ARCHITECTURE.md §4.2)., Exact-solver replacement for SGD/Adam (ARCHITECTURE.md §4.3)., Protocol

### Community 16 - "Community 16"
Cohesion: 0.14
Nodes (12): _AffineLN, dmrg_step(), forward_with_cache(), pullback_target(), ``TTBlock`` — Pre-LN Transformer encoder block with TT-factorized linears.  Co, LayerNorm with affine ``(γ, β)`` stored as **buffers** (not nn.Parameter)., LayerNorm with affine ``(γ, β)`` stored as **buffers** (not nn.Parameter)., update_affine_lsq() (+4 more)

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (13): Position-wise feed-forward block with TT-factorized weight matrices.      Args, TTFeedForward, GPT-2-style decoder-only Transformer with TT-factorized weights.      Args:, TT-GPT2 with a dense language-modelling head.      The LM head is a standard :cl, Multi-head self-attention with causal (lower-triangular) masking.      Identical, Pre-LN GPT-2 decoder block.      ::          h = x + causal_attn(ln1(x)), TTCausalSelfAttention, TTDecoderBlock (+5 more)

### Community 18 - "Community 18"
Cohesion: 0.2
Nodes (15): _build_3layer_cascade(), _chain_forward(), _chain_mse(), _factor_square(), Validation Gate A3 (FUTURE_WORK.md §Option B — Phase 3).  3-layer ``TTLinear`` c, ADMM reduces global MSE on a 3-layer TTLinear cascade., ADMM consensus variables should match or beat sequential target prop., Global MSE must trend downward across outer iterations.      Note: For a pure fe (+7 more)

### Community 19 - "Community 19"
Cohesion: 0.13
Nodes (7): MemoryArena, Python prototype of the GPU MemoryArena (MEMORY_ARENA.md §2-§4).  This is the, Return the pre-allocated workspace for the next solve's H matrix., Sum of bytes pinned by the arena (constant for the arena's lifetime)., Pre-allocated double-buffered storage for L/R environment blocks.      Buffers, Return ``(read_buf, write_buf)`` for the L environment.          Caller reads, Promote the write buffer to the new active buffer (constant-time).

### Community 20 - "Community 20"
Cohesion: 0.15
Nodes (14): Create a TT-GPT2 *pico* model (~100 k TT params, trainable on 2 GiB GPU).      2, Create a TT-GPT2 *pico* model (~100 k TT params, trainable on 2 GiB GPU).      2, tt_gpt2_pico(), Smoke + unit tests for TT-GPT2 components., Exact LSQ head fit must reduce cross-entropy., TT-GPT2 pico forward pass produces correct output shape., TT-GPT2 pico has TT cores registered as buffers., TTCausalSelfAttention must not let position i attend to j > i. (+6 more)

### Community 21 - "Community 21"
Cohesion: 0.28
Nodes (13): accuracy(), _console_print(), _console_safe(), fit_head(), forward(), load_data(), main(), MultiBlockClassifier (+5 more)

### Community 22 - "Community 22"
Cohesion: 0.14
Nodes (13): Numerical-stability edge case tests (NUMERICAL_STABILITY.md §3 + §4).  Covers, Tier-4 (add ε noise, retry torch SVD) MUST be the final fallback., SVDDivergenceError MUST be raised after all 4 tiers fail., When the local solve yields NaN, λ MUST escalate by 10× up to 6 times., If NaN persists past 6 escalations the solver must raise., Tier-2 SciPy gesdd MUST produce a valid SVD when Tier 1 raises., Tier-3 SciPy gesvd MUST be reached when both torch SVD and gesdd fail., test_svd_divergence_error_when_all_tiers_fail() (+5 more)

### Community 23 - "Community 23"
Cohesion: 0.14
Nodes (13): Unit tests for decision-boundary-aware target propagation (Option E).  Validates, With target_blend=0.0, the target equals the current block output., Logit-space target should move the correct class logit upward., Logit target must have correct shape and no NaN/inf., Only the pool mean shifts; per-token deviations from mean stay intact., With target_blend=1.0, the pool mean must exactly match pooled_target., Combined method returns correct per-token shape., test_logit_target_preserves_class_decision_dimensions() (+5 more)

### Community 24 - "Community 24"
Cohesion: 0.17
Nodes (5): from_dense(), Tensor Train (TT) factorized weight space.  Implements ``ITensorTrain`` from A, Per-step record of the TT-SVD decomposition (for Gate 1 verification)., Sqrt of the sum-of-squares of all discarded singular values across all cuts., TruncationReport

### Community 25 - "Community 25"
Cohesion: 0.21
Nodes (8): B2 integration test: TT-GPT2 decoder block DMRG reduces MSE on LM data., Exact LSQ head fitting must reduce cross-entropy on LM data., One dmrg_step on a decoder block reduces MSE vs a ground-truth block., Full pico model produces correct logit shape on LM data., _SynthLMData, test_decoder_block_dmrg_reduces_mse_on_lm_data(), test_head_fit_reduces_ce(), test_tt_gpt2_pico_forward_shape()

### Community 26 - "Community 26"
Cohesion: 0.29
Nodes (10): Robust SVD with the four-tier fallback hierarchy (NUMERICAL_STABILITY.md §4)., Single authoritative SVD call site. Implements the 4-tier fallback ladder., Eckart–Young–Mirsky truncation (TENSOR_TOPOLOGY.md §6, step 2)., robust_svd(), _svd_scipy(), _svd_torch(), SVDResult, truncate() (+2 more)

### Community 27 - "Community 27"
Cohesion: 0.23
Nodes (10): Linear layer whose weight matrix is stored as a Tensor-Train.      The module, TTLinear, Integration tests for TTLinear (drop-in Linear replacement)., A single DMRG step must (weakly) reduce the training MSE (SOLVER_MATH §V)., Smoke test: compose TTLinear with standard nn modules., test_ttlinear_can_coexist_with_nn_linear_pipeline(), test_ttlinear_dmrg_step_reduces_mse(), test_ttlinear_forward_deterministic() (+2 more)

### Community 28 - "Community 28"
Cohesion: 0.27
Nodes (10): _load_data(), Validation: Decision-Boundary-Aware Targets (Option E).  Proves that margin-awar, Margin-aware targets improve classification even with frozen attention.      The, Margin-aware targets produce lower block MSE vs Frobenius (per-step).      This, Across multiple seeds, margin-aware must not regress accuracy., Train a single TTBlock and return (test_acc, qk_accept_rate, v_accept_rate)., test_margin_aware_improves_accuracy(), test_margin_aware_reduces_block_mse_faster() (+2 more)

### Community 29 - "Community 29"
Cohesion: 0.29
Nodes (9): ArenaSpec, Sizing parameters for the arena., MemoryArena prototype tests (MEMORY_ARENA.md §2-§4).  These verify the *contra, The "zero-allocation prime directive" (MEMORY_ARENA.md §2).      Run 1000 take, _spec(), test_arena_buffers_distinct(), test_arena_swap_flips_active(), test_arena_total_bytes_constant() (+1 more)

### Community 30 - "Community 30"
Cohesion: 0.24
Nodes (9): condition_number(), needs_f64_upcast(), Mixed-precision casting helpers (NUMERICAL_STABILITY.md §2)., Upcast to float64 for numerically sensitive ops (QR / ill-conditioned SVD)., Downcast back to float32 for Tensor-Core-bound contractions., Spectral condition number used to gate dynamic upcasting., Return True if SVD/inverse should be done in float64 per NUMERICAL_STABILITY §2., to_f32() (+1 more)

### Community 31 - "Community 31"
Cohesion: 0.27
Nodes (8): Multi-Head Attention with TT-factorized projection weights.      Args:, TTMultiHeadAttention, Integration tests for TTMultiHeadAttention., Each per-projection DMRG step must weakly reduce its MSE., test_ttmha_dmrg_step_can_skip_untouched_projections(), test_ttmha_dmrg_step_reduces_projection_mse(), test_ttmha_self_attention_is_finite(), test_ttmha_shape_parity()

### Community 32 - "Community 32"
Cohesion: 0.4
Nodes (9): _attr_chain(), _iter_sources(), _parse(), AGENTS.md constraint enforcement tests.  Asserts, via static AST scanning of t, Dotted name like ``torch.linalg.svd`` if ``node`` is an Attribute chain., test_no_backward_calls_anywhere(), test_no_iterative_optimizer_references(), test_qr_has_single_authorized_call_site() (+1 more)

### Community 33 - "Community 33"
Cohesion: 0.33
Nodes (8): _make_ffn(), Tests for ``TTFeedForward`` — DMRG sweeps must reduce a rank-feasible target MSE, [batch, seq, embed] inputs must be accepted (Transformer use-case)., 3 outer rounds of DMRG against a rank-feasible target must reduce MSE., test_ttffn_dmrg_step_reduces_global_mse(), test_ttffn_dmrg_step_returns_per_sublayer_reports(), test_ttffn_forward_handles_3d_input(), test_ttffn_forward_shape()

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (8): _build_ffn_target(), _factor_square(), Validation Gate A2 (FUTURE_WORK.md §Option B — Phase 2).  Single ``TTFeedForward, Build a ground-truth TTFeedForward, produce a noiseless target.      Returns ``(, ADMM monotonically reduces MSE on a single TTFeedForward., ADMM output is closer to Y than an untrained forward pass., test_admm_ffn_improves_vs_no_update(), test_admm_ffn_reduces_mse()

### Community 35 - "Community 35"
Cohesion: 0.28
Nodes (5): dmrg_step(), forward(), num_parameters(), ``TTLinear`` — drop-in replacement for ``nn.Linear`` using a TensorTrain.  Per, Return the equivalent dense weight matrix (expensive — diagnostic use).

### Community 36 - "Community 36"
Cohesion: 0.25
Nodes (6): dmrg_step(), fit_head(), forward_with_cache(), TT-GPT2 — GPT-2-style decoder-only Transformer with TT-factorized weights.  Comp, Regression guard for the stacked-TTBlock real-task classifier.  Runs a tiny ve, test_tt_block_classifier_learns_above_random()

### Community 37 - "Community 37"
Cohesion: 0.25
Nodes (4): Compute a pooled target that maximises the classification margin.          Ins, Create a per-token target that only constrains the **pool mean**.          Giv, Convenience: margin-aware + pool-constrained target for a block.          Chai, Pull a downstream target back through a linear layer ``y = x @ W``.          S

### Community 38 - "Community 38"
Cohesion: 0.32
Nodes (7): Proves that ``qk_first=True`` unfreezes attention by training Q/K **before** W_o, Across seeds, qk_first must not catastrophically regress accuracy., Train a single TTBlock, return (qk_accept_rate, test_acc)., qk_first ordering must yield higher Q/K acceptance than default., test_qk_first_does_not_regress_accuracy(), test_qk_first_increases_acceptance_rate(), _train_qk_acceptance()

### Community 39 - "Community 39"
Cohesion: 0.29
Nodes (6): SVD fallback hierarchy tests (NUMERICAL_STABILITY.md §4)., Rank-deficient matrix must still decompose via one of the four tiers., Non-2D inputs raise ValueError; 2D inputs with finite entries never raise., test_svd_handles_rank_deficient_matrix(), test_svd_raises_only_on_non_2d(), test_svd_tier1_on_well_conditioned_matrix()

### Community 40 - "Community 40"
Cohesion: 0.29
Nodes (5): Validation Gate 1 (AGENTS.md §3, Phase I).  Decompose a 1024×1024 random matri, Decomposing at full TT-rank (no truncation) must reconstruct to machine precisio, Boundary ranks r_0 = r_d = 1 must be enforced., test_gate1_full_rank_round_trip_is_exact(), test_gate1_invariants_enforced()

### Community 41 - "Community 41"
Cohesion: 0.52
Nodes (6): _forward_stack(), _make_block(), Stacked TTBlock end-to-end DMRG sweep test (plan §C4).  Verifies that two stac, _sweep_stack(), test_depth4_stack_prefers_lower_target_blend(), test_stacked_ttblocks_reduce_global_mse()

### Community 42 - "Community 42"
Cohesion: 0.53
Nodes (5): _fmt_num(), _fmt_pct(), _load(), main(), Phase E: aggregate `bench/_coverage/*.json` sidecars into COVERAGE_MATRIX.md.

### Community 43 - "Community 43"
Cohesion: 0.4
Nodes (3): dmrg_step(), forward_with_cache(), ``TTFeedForward`` — position-wise feed-forward block with TT-factorized linears.

### Community 44 - "Community 44"
Cohesion: 0.4
Nodes (4): device(), pytest_collection_modifyitems(), pytest_configure(), Global test fixtures.  AGENTS.md + NUMERICAL_STABILITY.md mandate GPU/CUDA exe

### Community 45 - "Community 45"
Cohesion: 0.5
Nodes (3): forward(), _project(), ``TTMultiHeadAttention`` — MHA with TT-factorized projection matrices.  Each o

### Community 46 - "Community 46"
Cohesion: 0.4
Nodes (3): Forward-pass consistency: TT.contract_forward must match dense X @ W., Forward through the TT must approximate X @ W_original up to truncation error., test_forward_matches_original_dense()

### Community 47 - "Community 47"
Cohesion: 0.5
Nodes (3): Regression test for the end-to-end real-world classifier (DMRG-trained TT-MLP)., DMRG + target propagation on sklearn digits must beat 80% test accuracy., test_real_world_classifier_learns_above_chance()

### Community 49 - "Community 49"
Cohesion: 0.67
Nodes (3): Raised only after Tier 4 (noise + retry) of the fallback hierarchy fails., SVDDivergenceError, RuntimeError

## Knowledge Gaps
- **373 isolated node(s):** `Verify the CUDA toolchain is wired up correctly.  Run with::      uv run pyt`, `Tier-2 (1× block) classifier — linear-attention variant.  Drop-in counterpart`, `Subclass that replaces the softmax-flavored block sweep call.`, `Real-supervised-learning validation for the stacked TTBlock (plan §C-Validation)`, `Return a console-safe rendering for Windows cp1252 terminals.` (+368 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **57 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `require_cuda()` connect `Community 12` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 7`, `Community 8`, `Community 9`, `Community 16`, `Community 20`, `Community 21`, `Community 25`, `Community 27`, `Community 28`, `Community 29`, `Community 36`, `Community 38`, `Community 44`, `Community 47`, `Community 49`?**
  _High betweenness centrality (0.278) - this node is a cross-community bridge._
- **Why does `TargetPropagator` connect `Community 10` to `Community 0`, `Community 3`, `Community 4`, `Community 5`, `Community 7`, `Community 8`, `Community 9`, `Community 11`, `Community 13`, `Community 14`, `Community 16`, `Community 17`, `Community 18`, `Community 20`, `Community 21`, `Community 23`, `Community 25`, `Community 28`, `Community 36`, `Community 37`, `Community 41`, `Community 47`, `Community 59`, `Community 60`, `Community 61`, `Community 62`, `Community 63`, `Community 64`?**
  _High betweenness centrality (0.260) - this node is a cross-community bridge._
- **Why does `TTLinear` connect `Community 27` to `Community 0`, `Community 35`, `Community 4`, `Community 5`, `Community 6`, `Community 8`, `Community 10`, `Community 11`, `Community 47`, `Community 17`, `Community 18`, `Community 31`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Are the 67 inferred relationships involving `TargetPropagator` (e.g. with `TTBlockClassifier` and `DenseBlockClassifier`) actually correct?**
  _`TargetPropagator` has 67 INFERRED edges - model-reasoned connections that need verification._
- **Are the 50 inferred relationships involving `require_cuda()` (e.g. with `main()` and `main()`) actually correct?**
  _`require_cuda()` has 50 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `TTLinear` (e.g. with `TTFeedForward` and `TTCausalSelfAttention`) actually correct?**
  _`TTLinear` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `TTBlock` (e.g. with `TTFeedForward` and `TTMultiHeadAttention`) actually correct?**
  _`TTBlock` has 25 INFERRED edges - model-reasoned connections that need verification._