# Graph Report - .  (2026-05-05)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 655 nodes · 941 edges · 47 communities (20 shown, 27 thin omitted)
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 220 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `326e2f02`
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

## God Nodes (most connected - your core abstractions)
1. `TargetPropagator` - 39 edges
2. `require_cuda()` - 37 edges
3. `OptimizationBenchmark` - 21 edges
4. `TTLinear` - 20 edges
5. `DMRGOptimizer` - 19 edges
6. `TensorTrain` - 19 edges
7. `robust_svd()` - 17 edges
8. `TTBlock` - 14 edges
9. `main()` - 13 edges
10. `MemoryArena` - 13 edges

## Surprising Connections (you probably didn't know these)
- `device()` --calls--> `require_cuda()`  [INFERRED]
  tests/test_tt_linear_attention_block_dmrg.py → src/dmrg_transformer/core/device.py
- `device()` --calls--> `require_cuda()`  [INFERRED]
  tests/test_tt_linear_attention_forward.py → src/dmrg_transformer/core/device.py
- `main()` --calls--> `cuda_available()`  [INFERRED]
  scripts/detect_cuda.py → src/dmrg_transformer/core/device.py
- `main()` --calls--> `require_cuda()`  [INFERRED]
  scripts/poc_softmax_transformer.py → src/dmrg_transformer/core/device.py
- `main()` --calls--> `describe_device()`  [INFERRED]
  scripts/poc_softmax_transformer.py → src/dmrg_transformer/core/device.py

## Communities (47 total, 27 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (46): _AffineLN, dmrg_step(), forward_with_cache(), pullback_target(), ``TTBlock`` — Pre-LN Transformer encoder block with TT-factorized linears.  Co, Pre-LN Transformer encoder block (TT-MHA + TT-FFN with optional LN affine)., LayerNorm with affine ``(γ, β)`` stored as **buffers** (not nn.Parameter)., TTBlock (+38 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (53): condition_number(), needs_f64_upcast(), Mixed-precision casting helpers (NUMERICAL_STABILITY.md §2)., Upcast to float64 for numerically sensitive ops (QR / ill-conditioned SVD)., Downcast back to float32 for Tensor-Core-bound contractions., Spectral condition number used to gate dynamic upcasting., Return True if SVD/inverse should be done in float64 per NUMERICAL_STABILITY §2., to_f32() (+45 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (39): Layer-wise target propagator (ARCHITECTURE.md §4.2).  Replaces the Backpropaga, Pull a target back through a residual connection ``y = x + f(x)``.          Gi, Pull a target back through ``LayerNorm`` using current-row statistics., Compute a local target for a layer given a global target and its forward output., Pull a context target back to a per-head V target through ``A @ V``., Recover a target attention pattern ``A_target`` from a context target., Map a target attention pattern ``A_target`` to a score target ``S``., Decoupled bilinear pull-back of a score target ``S = Q K^T``.          Given a (+31 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (37): ArenaSpec, MemoryArena, Python prototype of the GPU MemoryArena (MEMORY_ARENA.md §2-§4).  This is the, Return the pre-allocated workspace for the next solve's H matrix., Sum of bytes pinned by the arena (constant for the arena's lifetime)., Sizing parameters for the arena., Pre-allocated double-buffered storage for L/R environment blocks.      Buffers, Return ``(read_buf, write_buf)`` for the L environment.          Caller reads (+29 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (34): dmrg_step(), forward_with_cache(), ``TTFeedForward`` — position-wise feed-forward block with TT-factorized linears., Position-wise feed-forward block with TT-factorized weight matrices.      Args, TTFeedForward, dmrg_step(), forward(), num_parameters() (+26 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (32): dmrg_step(), forward_with_cache(), pullback_target(), ``TTLinearAttentionBlock`` — Pre-LN block with TT linear attention.  Compositi, Return per-head Q, K, V, phiQ, phiK, w (= phiQ·phiKᵀ), denom, context., Per-(B,H) ridge LSQ for V given fixed w (i.e. fixed phiQ, phiK).          ``w`, Pre-LN Transformer block with TT linear attention + TT-FFN., TTLinearAttentionBlock (+24 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (30): qr_f64(), qr_f64_strict(), QR with mandatory float64 cast (NUMERICAL_STABILITY.md §2).  This module is th, Compute Q, R with float64 internal precision; result returned in ``matrix.dtype`, Same as :func:`qr_f64` but returns float64 Q and R for orthogonality tests., _make_random_tt(), Validation Gate 2 (AGENTS.md §3, Phase II).  After a left-orthogonalization sw, Construct a random TT directly in the requested dtype (bypasses SVD/QR cascade). (+22 more)

### Community 7 - "Community 7"
Cohesion: 0.08
Nodes (25): BenchmarkResult, execute(), _factor_for_tt(), _factor_pair(), OptimizationBenchmark, Reproduction of ``docs/BENCHMARK.md`` — the three-way optimizer runoff.  Invoc, Choose a TT factorization adapted to the size of ``n``.      For small ``n`` (, Adam on ``W = U @ V`` with ``U: (in, r), V: (r, out)``.          This is the * (+17 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (36): _apply_J(), _apply_JT(), _build_block_normal_equations(), _build_jacobian(), _build_normal_equations(), _huber_clamp(), LocalSolveResult, Exact local least-squares solver for a single TT-core (AGENTS.md Phase III). (+28 more)

### Community 9 - "Community 9"
Cohesion: 0.06
Nodes (29): DMRGOptimizer, Full bidirectional sweep: L→R then R→L.          Ensures the TT is properly ga, Exact-solver replacement for SGD/Adam.      Args:         max_rank: TT-rank b, _init_tt(), _make_low_rank_target(), Adaptive-rank wiring through ``DMRGOptimizer.sweep`` (plan §C5 closure).  Vali, Random small-init TT in the requested factorization., A strict (≈0) threshold must keep every mode up to max_rank — same MSE     as t (+21 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (20): PositionalEncoding, Embedding utilities for the DMRG-Transformer., Sinusoidal positional encoding (fixed).      PE(pos, 2i) = sin(pos / 10000^(2i, Add positional encoding to input.          Args:             x: [batch, seq_l, accuracy(), _console_print(), _console_safe(), DenseBlockClassifier (+12 more)

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (21): dump_coverage_sidecar(), iso_time_lookup(), measure_inference_latency(), Shared GPU/wall/inference instrumentation for the Tier-1/2/3 runners.  Pulled, Median forward latency in milliseconds, with CUDA sync per call., Return (test_acc, wall) for the last sample with wall <= ``target_wall``., Write ``bench/_coverage/<tier>.json`` for the matrix aggregator., read_peak_mem_mib() (+13 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (21): describe_device(), Human-readable string describing the active device (for logs/reports)., main(), Verify the CUDA toolchain is wired up correctly.  Run with::      uv run pyt, main(), Gate-3 headline proof, rendered to ``bench/GATE3_PROOF.md`` (GPU-only)., main(), Headline benchmark — BENCHMARK.md spec at 1024x1024 (Phase A1+B1+B3 deliverable) (+13 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (8): IDMRGOptimizer, ITargetPropagator, ITensorTrain, Strict OOD interfaces (Protocols) mirroring ARCHITECTURE.md Rust traits.  Thes, Geometric encapsulation of a factorized weight space.      Mirrors ``ITensorTr, Replacement for the Backpropagation Chain Rule (ARCHITECTURE.md §4.2)., Exact-solver replacement for SGD/Adam (ARCHITECTURE.md §4.3)., Protocol

### Community 14 - "Community 14"
Cohesion: 0.17
Nodes (4): Tensor Train (TT) factorized weight space.  Implements ``ITensorTrain`` from A, Per-step record of the TT-SVD decomposition (for Gate 1 verification)., Sqrt of the sum-of-squares of all discarded singular values across all cuts., TruncationReport

### Community 15 - "Community 15"
Cohesion: 0.4
Nodes (9): _attr_chain(), _iter_sources(), _parse(), AGENTS.md constraint enforcement tests.  Asserts, via static AST scanning of t, Dotted name like ``torch.linalg.svd`` if ``node`` is an Attribute chain., test_no_backward_calls_anywhere(), test_no_iterative_optimizer_references(), test_qr_has_single_authorized_call_site() (+1 more)

### Community 16 - "Community 16"
Cohesion: 0.29
Nodes (5): Validation Gate 1 (AGENTS.md §3, Phase I).  Decompose a 1024×1024 random matri, Decomposing at full TT-rank (no truncation) must reconstruct to machine precisio, Boundary ranks r_0 = r_d = 1 must be enforced., test_gate1_full_rank_round_trip_is_exact(), test_gate1_invariants_enforced()

### Community 17 - "Community 17"
Cohesion: 0.53
Nodes (5): _fmt_num(), _fmt_pct(), _load(), main(), Phase E: aggregate `bench/_coverage/*.json` sidecars into COVERAGE_MATRIX.md.

### Community 18 - "Community 18"
Cohesion: 0.4
Nodes (3): Forward-pass consistency: TT.contract_forward must match dense X @ W., Forward through the TT must approximate X @ W_original up to truncation error., test_forward_matches_original_dense()

## Knowledge Gaps
- **239 isolated node(s):** `Verify the CUDA toolchain is wired up correctly.  Run with::      uv run pyt`, `Tier-2 (1× block) classifier — linear-attention variant.  Drop-in counterpart`, `Subclass that replaces the softmax-flavored block sweep call.`, `Real-supervised-learning validation for the stacked TTBlock (plan §C-Validation)`, `Return a console-safe rendering for Windows cp1252 terminals.` (+234 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **27 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `require_cuda()` connect `Community 3` to `Community 0`, `Community 1`, `Community 4`, `Community 5`, `Community 7`, `Community 8`, `Community 10`, `Community 11`, `Community 12`?**
  _High betweenness centrality (0.339) - this node is a cross-community bridge._
- **Why does `DMRGOptimizer` connect `Community 9` to `Community 1`, `Community 3`, `Community 4`, `Community 6`, `Community 8`, `Community 12`?**
  _High betweenness centrality (0.163) - this node is a cross-community bridge._
- **Why does `TargetPropagator` connect `Community 2` to `Community 0`, `Community 4`, `Community 5`, `Community 9`, `Community 11`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `TargetPropagator` (e.g. with `TTMlp` and `.__init__()`) actually correct?**
  _`TargetPropagator` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 31 inferred relationships involving `require_cuda()` (e.g. with `main()` and `main()`) actually correct?**
  _`require_cuda()` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `OptimizationBenchmark` (e.g. with `main()` and `main()`) actually correct?**
  _`OptimizationBenchmark` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `TTLinear` (e.g. with `.__init__()` and `TTFeedForward`) actually correct?**
  _`TTLinear` has 14 INFERRED edges - model-reasoned connections that need verification._