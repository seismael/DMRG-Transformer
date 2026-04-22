# DMRG-Transformer — Rank/MSE Pareto Curve

**Device:** `device=cuda:0 (NVIDIA GeForce MX150, sm_61, 2.0 GiB)`  
**Configuration:** 256×256 layer, batch=1024, target = `sin(X·W) + 0.1·η`, 2-seed mean.  

Reference baselines (rank-independent):

* **Dense Exact (lstsq):** MSE = `5.5159e-02`  (time = 0.038s, 65,536 params)
* **Adam (500 iters, lr=0.01):** MSE = `5.5159e-02`  (time = 4.781s, 65,536 params)

## TT-DMRG rank sweep

| Rank | TT params | Compression | DMRG MSE | DMRG time (s) | MSE / Dense MSE |
| ---: | --------: | ----------: | -------: | ------------: | --------------: |
| 2 | 192 | 341.3× | 4.3836e-01 | 0.406 | 7.95× |
| 4 | 640 | 102.4× | 4.3375e-01 | 0.587 | 7.86× |
| 8 | 2,304 | 28.4× | 4.1710e-01 | 1.091 | 7.56× |
| 16 | 8,704 | 7.5× | 3.6115e-01 | 3.305 | 6.55× |
| 32 | 16,896 | 3.9× | 2.9640e-01 | 6.231 | 5.37× |
| 64 | 33,280 | 2.0× | 2.0000e-01 | 21.849 | 3.63× |

## Reading the curve

The MSE column shows how close DMRG gets to the dense optimum at each rank budget. The last column is the multiplicative gap. As `r` grows the rank-constrained DMRG converges to the dense exact solution; the compression column shows the parameter cost. This is the honest Pareto trade-off for a *non-TT-rank* target — when the target lives on a TT manifold of rank `r₀` (Gate 3 setup), DMRG matches the dense MSE at any `r ≥ r₀` (see `bench/GATE3_PROOF.md`).
