# DMRG-Transformer — Headline Benchmark (BENCHMARK.md spec)

**Device:** `device=cuda:0 (NVIDIA GeForce MX150, sm_61, 2.0 GiB)`  
**Configuration:** 1024×1024 layer, batch=2048, rank=32, target = `sin(X·W) + 0.1·η`  
**Methodology:** 1 warmup pass(es) discarded; 3 seeded measurement passes; mean ± population std reported. All wall-times include `torch.cuda.synchronize()`. Peak memory via `torch.cuda.max_memory_allocated()` (reset after warmup).  
**Total wall-time:** 1774.4s  

| Method | MSE (mean ± std) | Time (s, mean ± std) | Peak GPU mem | Params | FLOPs/call |
| :----- | ---: | ---: | ---: | ---: | ---: |
| Gradient Descent (Adam) | 3.7348e-02 ± 1.5e-07 | 196.557 ± 18.678 | 0.160 GB | 1,048,576 | 4.29e+12 |
| Dense Exact Solver (O(N^3)) | 3.7344e-02 ± 0.0e+00 | 0.776 ± 0.002 | 0.092 GB | 1,048,576 | 5.37e+09 |
| TT-DMRG Exact Sweep | 4.1531e-01 ± 6.7e-05 | 265.259 ± 7.649 | 2.215 GB | 33,024 | 8.60e+09 |

**Compression (dense → TT):** 31.8× (1,048,576 → 33,024 parameters)

## Interpretation

* **Headline target** (`sin(X·W)+noise`) is a full-rank target — DMRG is constrained to a TT-rank-`r` manifold and will not match the dense exact MSE in general; see `bench/PARETO.md` for the rank/MSE trade-off.
* **Adam** uses lr=0.01, 500 iters; for fairness see `docs/BENCHMARK.md` reconciliation notes.
* **Dense Exact** is `torch.linalg.lstsq` (cuSOLVER) — the unconstrained least-squares optimum.
* **TT-DMRG** runs 2 bidirectional sweeps from a TT initialised by TT-SVD of a random matrix; no learning rate, no iteration budget. Its MSE matches the dense optimum *only* on TT-rank-bounded targets (see `bench/GATE3_PROOF.md`).

## Honest assessment vs. BENCHMARK.md spec language

The original spec headline ("DMRG matches dense MSE in ~0.05s with 15.6× compression") holds **only when the data lives on a TT manifold of the chosen rank**. On the spec's own non-TT target (`sin(X·W)+noise`), DMRG produces the rank-`r` Pareto-optimal point — lower MSE than its parameter budget allows for a dense baseline of the same parameter count, but generally higher MSE than an unconstrained dense lstsq fit. See `docs/BENCHMARK.md` reconciliation.
