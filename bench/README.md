# Benchmark & Proof-of-Concept Artifacts

This folder contains reproducible evidence that the DMRG-Transformer Python
reference implements AGENTS.md correctly.

## Artifacts

| File | What it proves |
| :--- | :--- |
| [`GATE3_PROOF.md`](GATE3_PROOF.md) | **Headline claim.** DMRG sweep matches dense `lstsq` MSE to machine precision on a true TT-rank target (AGENTS.md Gate 3). |
| [`RESULTS.md`](RESULTS.md) | Three-way runoff (Adam / Dense / TT-DMRG) from `docs/BENCHMARK.md` at tractable CPU scales, plus compression ratios. |
| [`TEST_OUTPUT.txt`](TEST_OUTPUT.txt) | Full `pytest -v` log for the 29-test suite (all gates + constraint scans + integration + smoke benchmark). |

## How to regenerate

```powershell
$env:PYTHONPATH = 'src'
python scripts/run_gate3_proof.py           # -> bench/GATE3_PROOF.md
python scripts/run_benchmarks.py            # -> bench/RESULTS.md
python -m pytest tests -v -o addopts=""     # -> stdout; pipe to bench/TEST_OUTPUT.txt
```

## Headline numbers (from this run)

* **Gate 3 parity**: Dense MSE `3.481e-30`, DMRG MSE `1.007e-29` — both at
  float64 machine precision, ratio ≈ 2.9×. DMRG converges to the dense optimum
  inside its rank-r manifold.
* **Compression**: 144×144 @ rank 8 uses **2,304 TT params vs. 20,736 dense
  params** — 9× fewer parameters, same forward-pass signature.
* **Test suite**: **29/29 tests passing**, including all four AGENTS
  validation gates and constraint enforcement (no gradients, no Adam/SGD,
  single SVD/QR call-site).

## Scope note

The full `1024 × 1024 / batch = 2048` scale in
[`../docs/BENCHMARK.md`](../docs/BENCHMARK.md) targets the Phase IV CUDA
microkernel (cuSOLVER SVD/QR + streamed `cuTensorNet` contractions). The
pure-Python reference materializes the full local Jacobian and therefore caps
out at moderate scales on CPU. Correctness and asymptotic behaviour are
proven; the linear-algebra kernel swap for scale is Phase IV work.
