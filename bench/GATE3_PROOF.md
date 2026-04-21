# Gate 3 Proof — TT-DMRG matches Dense Least-Squares

AGENTS.md §3 Validation Gate 3: *"The MSE of the DMRG sweep must converge to the exact same MSE as the Dense Exact Solver."*

**Setup:** N=64, M=64, TT-rank r=4, batch=512. Target `Y = X @ W_true` where `W_true` is generated to live on the rank-r TT manifold (reconstruction error = theoretical SVD bound).

## Results

| Estimator | MSE | Wall time (s) |
| :-------- | --: | ------------: |
| Dense `torch.linalg.lstsq` (O(N^3)) | 3.481e-30 | 0.0032 |
| TT-DMRG initial (random init) | 1.292e+01 | — |
| TT-DMRG after 1 sweep | 2.707e-02 | — |
| TT-DMRG after 20 sweeps | 1.007e-29 | 11.0153 |

**DMRG / Dense MSE ratio:** 2.892e+00

Both methods converge to the same minimum (to within float64 conditioning of `X`). DMRG achieves this with 512 TT parameters vs. 4096 dense parameters (8.0x compression).

See [`tests/test_gate3_exact_parity.py`](../tests/test_gate3_exact_parity.py) for the enforced assertion.
