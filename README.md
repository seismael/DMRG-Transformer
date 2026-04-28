# DMRG-Transformer

**A Post-Gradient-Descent Backbone for Neural Networks — Exact Solver on a Tensor Train Manifold (GPU / CUDA)**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76b900.svg)](https://developer.nvidia.com/cuda-12-1-0-download-archive)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/PoC-v1.1_Validated-blue.svg)](#)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#5-quick-start)
[![Gates](https://img.shields.io/badge/AGENTS_gates-1--3_validated-brightgreen.svg)](bench/GATE3_PROOF.md)

> **Status:** Python reference implementation validated at v1.1 PoC milestone (package version 0.1.0).
> This implementation proves that exact local solvers (DMRG) can train multi-layer Transformer
> architectures on real-world datasets using **zero backpropagation**.
> Current work: ADMM outer loop to resolve stacked-block inter-layer drift (see [#1](#8-next-steps)).

---

## 1. Overview

Modern neural networks are trained by iteratively nudging weights along the error gradient. This loop — backpropagation + a stochastic optimizer (Adam, SGD) with a hand-tuned learning rate — is the dominant cost of training every Transformer-based model today.

**DMRG-Transformer replaces that loop with an exact, gradient-free solver.**

- Weights are stored as a **Tensor Train (TT)**: a chain of small 3-D cores whose physical indices factor the dense matrix dimensions ([docs/TENSOR_TOPOLOGY.md](docs/TENSOR_TOPOLOGY.md)).
- The update rule is the **Density Matrix Renormalization Group (DMRG)** algorithm from quantum many-body physics: sweep left-to-right and right-to-left across cores, each step solving an exact least-squares sub-problem followed by Eckart–Young–Mirsky SVD truncation.
- No `loss.backward()`. No Adam/SGD. **Zero gradient-graph memory overhead.**

---

## 2. Core Architecture Mappings

| Standard Deep Learning | DMRG-Transformer Equivalent |
| :--- | :--- |
| `nn.Linear(in, out)` | `TensorTrain(cores=[G₁, ..., G_d], ranks=r)` |
| `optimizer.step()` | `DMRGOptimizer.sweep_and_truncate(tt, target)` |
| Backpropagation (Chain Rule) | Layer-wise Target Propagation (pseudo-inverse) |
| Weight Update Calculation | Local SVD Projection (Eckart–Young–Mirsky theorem) |
| Regularization / Weight Decay | SVD Truncation (drop singular values > r) |

**Four hard constraints (enforced by automated tests):**
- **No gradients** — `loss.backward()` banned across `src/`
- **No iterative optimizers** — Adam, SGD, RMSprop banned in `src/`
- **No dense inversions** — all operations bounded by `O(d·n·r³)`, never `O(N³)`
- **In-place environment updates** — left/right blocks reused, not reallocated

---

## 3. Goals

| # | Goal | Status |
| - | :--- | :---: |
| 1 | Faithfully implement the mathematics from `docs/` specifications | ✅ |
| 2 | Pass every AGENTS.md validation gate (Gates 1–3) | ✅ |
| 3 | Provide drop-in components: `TTLinear`, `TTMultiHeadAttention`, `DMRGOptimizer` | ✅ |
| 4 | Prove performance advantage over gradient descent on TT-native targets | ✅ |
| 5 | Run end-to-end on CUDA (cuSOLVER + PyTorch float64) | ✅ |
| 6 | 1024×1024 benchmark sweep on 2 GiB GPU (MX150) | ✅ |
| 7 | Real-world supervised classification (sklearn digits, zero backprop) | ✅ |
| 8 | Stacked-block Transformer PoC with target propagation | ✅ |
| 9 | ADMM outer loop for inter-layer consensus (in development) | 🧭 |
| 10 | Rust/CUDA microkernel for production scale (Phase IV) | 🧭 |

---

## 4. Achieved Results

All results reproduced on NVIDIA GeForce MX150 (sm_61, 2.0 GiB), CUDA 12.1, float64.

### 4.1 Machine-precision parity with the dense lower bound

AGENTS.md Gate 3: "The MSE of the DMRG sweep must converge to the exact same MSE as the Dense Exact Solver."

| Estimator | MSE |
| :--- | ---: |
| Dense `torch.linalg.lstsq` (O(N³), cuSOLVER) | `2.386e-30` |
| **TT-DMRG after 20 bidirectional sweeps** | **`1.349e-29`** |

### 4.2 Beats gradient descent on TT-native targets

Target is `Y = X @ W_true` with `W_true` drawn from a rank-r Tensor Train. Adam gets 5,000 iterations at `lr=0.01`. DMRG gets 3 sweeps.

| Layer | Method | Final MSE | Wall (s) | Params | vs Adam MSE | Compression |
| :--- | :--- | ---: | ---: | ---: | :---: | :---: |
| 64×64 r=4 | Adam (5000 iters) | `6.538e-07` | 3.56 | 4,096 | 1.0× | — |
| 64×64 r=4 | **TT-DMRG** (3 sweeps) | **`7.606e-09`** | **0.26** | **512** | **86× better** | **8.0×** |
| 144×144 r=6 | Adam (5000 iters) | `2.654e-06` | 7.51 | 20,736 | 1.0× | — |
| 144×144 r=6 | **TT-DMRG** (3 sweeps) | **`1.843e-07`** | **0.28** | **1,728** | **14× better** | **12.0×** |

### 4.3 Real-world supervised learning (TT-MLP)

10-class classification on `sklearn.digits`. From [bench/REAL_WORLD_MNIST.md](bench/REAL_WORLD_MNIST.md).

| Model | Train acc | **Test acc** | Params | Wall (s) | Compression |
| :--- | ---: | ---: | ---: | ---: | :---: |
| **TT-MLP (DMRG)** | 0.9026 | **0.8833** | 1,194 | 1.81 | **2.0×** |
| Dense MLP (AdamW, CE) | 1.0000 | **0.9694** | 2,410 | 1.89 | — |

### 4.4 Stacked-block Transformer (v1.1 PoC)

**Task:** 1× TTBlock (embed=16, heads=2, hidden=16, rank=8) on `sklearn.digits`.

| Model | Train acc | **Test acc** | Params | Wall (s) | Peak RAM |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Linear TT-DMRG (v1.1)** | 0.8546 | **0.8667** | 1,946 | **19.5** | **116 MiB** |
| **Softmax TT-DMRG (v1.1)** | 0.8622 | **0.8722** | 1,946 | 52.3 | 374 MiB |
| Dense Adam-CE (Large) | 1.0000 | **0.9750** | 3,922 | 85.1 | 305 MiB |

Key efficiency findings:
- **Linear attention** achieves **2.7× speedup** and **3.2× memory reduction** vs softmax variant
- **2.0× parameter compression** relative to identical-architecture dense Adam baseline
- Softmax attention Q/K/V substeps are rejected by trust-region (see [REVIEW.md](REVIEW.md))

### 4.5 1024×1024 Headline Benchmark

| Method | MSE | Time | Peak GPU mem |
| :--- | ---: | ---: | ---: |
| Adam (500 iters, lr=0.01) | 3.73e-02 | 196.6 s | 0.16 GiB |
| Dense Exact (`lstsq`) | 3.73e-02 | 0.78 s | 0.09 GiB |
| TT-DMRG (rank=32, 2 sweeps) | 4.15e-01 | 265.3 s | 2.22 GiB |

> DMRG produces the Pareto-optimal rank-32 approximation. It reaches machine-precision parity
> with the dense solver only on TT-rank-bounded targets (see [bench/GATE3_PROOF.md](bench/GATE3_PROOF.md)).
> At 1024×1024 scale on this hardware, the dense `lstsq` constant factors dominate the comparison.
> The Phase IV Rust/CUDA microkernel is the path to closing this gap.

---

## 5. Quick Start

Requires Python 3.12+, [uv](https://github.com/astral-sh/uv), and a CUDA 12.1 capable GPU.

```powershell
git clone https://github.com/seismael/DMRG-Transformer.git
cd DMRG-Transformer
uv sync --extra dev                            # installs torch+cu121, scipy, pytest
uv run python scripts/detect_cuda.py           # smoke test cuSOLVER + GPU
uv run python -m pytest tests --no-header -q   # run all tests

# Run the primary PoC entries:
uv run python scripts/poc_tt_mlp.py              # TT-MLP classification (88 %)
uv run python scripts/poc_softmax_transformer.py # Softmax Transformer PoC (87 %)
uv run python scripts/poc_linear_transformer.py  # Linear Transformer PoC (87 %)

# Run proofs and benchmarks:
uv run python scripts/run_gate3_proof.py         # Gate 3 parity proof
uv run python scripts/run_headline_benchmark.py  # 1024×1024 benchmark
```

Full spec-to-implementation traceability: [docs/COMPLIANCE.md](docs/COMPLIANCE.md).

---

## 6. Documentation Matrix

| Document | Purpose |
| :--- | :--- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System topology, OOD interfaces, execution pipeline |
| [docs/TENSOR_TOPOLOGY.md](docs/TENSOR_TOPOLOGY.md) | Einsum strings, rank boundaries, core shapes, SVD protocol |
| [docs/NUMERICAL_STABILITY.md](docs/NUMERICAL_STABILITY.md) | Float64 policies, 4-tier SVD fallback, Tikhonov, Huber clamp |
| [docs/MEMORY_ARENA.md](docs/MEMORY_ARENA.md) | GPU double-buffering, zero-allocation sweep contract |
| [docs/SOLVER_MATH.md](docs/SOLVER_MATH.md) | Formal proofs: DMRG exactness, O(d·n·r³) complexity bound |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | Benchmark specification and measured results |
| [docs/BLUEPRINT.md](docs/BLUEPRINT.md) | High-level architectural vision |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | Spec-to-implementation traceability matrix |
| [REVIEW.md](REVIEW.md) | Project review, breakthroughs, remaining gaps |
| [FUTURE_WORK.md](FUTURE_WORK.md) | Planned extensions (ADMM, PEPS, hybrid approaches) |

---

## 7. Project Structure

```
src/dmrg_transformer/
  core/            Mathematical primitives (SVD fallback, QR, precision, arena)
  tt/              Tensor Train geometry & gauge management
  optim/           DMRG exact-solver engine (sweep, local solver, CG)
  propagation/     Target propagation (Chain Rule replacement)
  nn/              Neural network modules (TTLinear, MHA, FFN, Blocks)
  bench/           Benchmark harness (3-way runoff, instrumentation)
tests/             Test suite (29 test files, Gates 1-3 validated)
scripts/           Runnable PoC scripts and benchmarks
bench/             Benchmark output reports (Markdown)
docs/              Architecture & math specifications
```

---

## 8. Next Steps

The current limiting factor is **inter-layer drift** in stacked blocks — per-layer DMRG solvers
cannot see the global loss landscape, producing a ~10 pp accuracy gap vs Adam on real tasks.

| Work Item | Status | See |
| :--- | :---: | :--- |
| ADMM Outer Loop | Planned | [FUTURE_WORK.md](FUTURE_WORK.md) Option B |
| Decision-Boundary Targets | Planned | [REVIEW.md](REVIEW.md) §3 |
| Rust/CUDA Microkernel (Phase IV) | Deferred | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §7 |
| Global PEPS Tensor Network | Research | [FUTURE_WORK.md](FUTURE_WORK.md) Option C |

---

## 9. References

- Oseledets, I. V. (2011). *Tensor-Train decomposition.* SIAM Journal on Scientific Computing, 33(5), 2295-2317.
- Schollwöck, U. (2011). *The density-matrix renormalization group in the age of matrix product states.* Annals of Physics, 326(1), 96-192.
- Eckart, C., & Young, G. (1936). *The approximation of one matrix by another of lower rank.* Psychometrika, 1(3), 211-218.
- Boyd, S., Parikh, N., & Chu, E. (2011). *Distributed optimization and statistical learning via the alternating direction method of multipliers.* Foundations and Trends in Machine Learning, 3(1), 1-122.
