# DMRG-Transformer

**A Post-Gradient-Descent Backbone for Neural Networks — Exact Solver on a Tensor Train Manifold (GPU / CUDA)**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76b900.svg)](https://developer.nvidia.com/cuda-12-1-0-download-archive)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/Status-Alpha_PoC-orange.svg)](#)
[![Tests](https://img.shields.io/badge/tests-29%2F29_GPU-brightgreen.svg)](#6-validation-gates)
[![Gates](https://img.shields.io/badge/AGENTS_gates-1%2F2%2F3_validated-brightgreen.svg)](bench/GATE3_PROOF.md)

> **Status: initial GPU-only proof-of-concept.** The reference implementation
> validates the mathematics, enforces every invariant from `AGENTS.md`, runs
> end-to-end on `cuda:0` (cuSOLVER + PyTorch), and demonstrates a clear
> advantage over gradient descent on the method's native domain (low-TT-rank
> targets). Scaling to production-size LLMs **requires the AGENTS Phase IV
> Rust + CUDA microkernel and community effort** — see
> [Limitations & Call for Collaborators](#9--limitations--call-for-collaborators).

---

## 1. Overview

Modern neural networks are trained by *iteratively* nudging weights along the
error gradient. This loop — backpropagation + a stochastic optimiser (Adam,
SGD) with a hand-tuned *learning rate* — is the dominant cost of training
every Transformer-based model today.

**DMRG-Transformer replaces that loop with an exact, gradient-free solver.**

- Weights are stored as a **Tensor Train (TT)**: a chain of small 3-D cores
  whose physical indices factor the dense matrix indices
  ([docs/TENSOR_TOPOLOGY.md](docs/TENSOR_TOPOLOGY.md) §2).
- The update rule is the **Density Matrix Renormalization Group (DMRG)**
  algorithm from quantum many-body physics: sweep left-to-right and
  right-to-left across cores, each step solving an *exact* least-squares
  sub-problem (matrix-free normal equations on the local subspace) followed
  by Eckart–Young–Mirsky SVD truncation.
- No `loss.backward()`. No Adam/SGD. **No learning rate to tune.**

The design follows the documentation in [docs/](docs/) literally, under
strict `AGENTS.md` constraints (no gradients; no iterative optimisers; no
dense matrix inversions; in-place environment updates; CUDA-native
execution).

---

## 2. Goals

| # | Goal                                                                              | Status |
| - | :-------------------------------------------------------------------------------- | :----: |
| 1 | Faithfully implement the mathematics from [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) + companions. |   ✅    |
| 2 | Pass every validation gate listed in [AGENTS.md](AGENTS.md) §3.                   |   ✅ (Gates 1–3) |
| 3 | Provide drop-in components: `TTLinear`, `TTMultiHeadAttention`, `DMRGOptimizer`. |   ✅    |
| 4 | Prove the **performance advantage over gradient descent** on a TT-native target.  |   ✅    |
| 5 | Run end-to-end on CUDA (cuSOLVER + PyTorch float64).                              |   ✅    |
| 6 | Ship the AGENTS Phase IV Rust + CUDA microkernel for production scale.            |   🧭 future work |
| 7 | Demonstrate a 1024×1024 [docs/BENCHMARK.md](docs/BENCHMARK.md) sweep + full LLM training. | 🧭 requires Phase IV |

---

## 3. Achieved Results (reproducible, CUDA 12.1, float64)

All numbers below come from one of three reproducible scripts that ship in
this repo:

| Script | Output |
| :----- | :----- |
| [scripts/run_gate3_proof.py](scripts/run_gate3_proof.py)         | [bench/GATE3_PROOF.md](bench/GATE3_PROOF.md) |
| [scripts/run_poc_benchmark.py](scripts/run_poc_benchmark.py)     | [bench/POC_RESULTS.md](bench/POC_RESULTS.md) |
| [scripts/run_benchmarks.py](scripts/run_benchmarks.py)           | [bench/RESULTS.md](bench/RESULTS.md) |

Reference hardware: NVIDIA GeForce MX150 (sm_61, 2.0 GiB), CUDA 12.1.

### 3.1 Machine-precision parity with the dense lower bound

AGENTS.md Gate 3: *"The MSE of the DMRG sweep must converge to the exact same
MSE as the Dense Exact Solver."* — see [bench/GATE3_PROOF.md](bench/GATE3_PROOF.md).

| Estimator                              |    MSE    |
| :------------------------------------- | --------: |
| Dense `torch.linalg.lstsq` (cuSOLVER, O(N³)) | `2.386e-30` |
| **TT-DMRG after 20 bidirectional sweeps**    | **`1.563e-29`** |

Both estimators bottom out at float64 numerical noise; DMRG reaches the
exact global optimum *inside its rank-r manifold* with ratio 6.55× of the
absolute lower bound — i.e. machine precision.

### 3.2 Beats gradient descent on the method's native domain

From [bench/POC_RESULTS.md](bench/POC_RESULTS.md). Target is `Y = X @ W_true`
with `W_true` drawn from a rank-r Tensor Train (the exact-solver's native
domain). Adam gets 5,000 iterations at `lr=0.01`. DMRG gets 3 sweeps —
**and no learning rate**.

| Layer       | Method              | Final MSE      | Wall (s) | Params    | vs Adam MSE     | Compression |
| :---------- | :------------------ | -------------: | -------: | --------: | :-------------: | :---------: |
| 64×64  r=4  | Adam (5000 iters)   | `6.538e-07`    |   3.46   |    4,096  |       1.0×      |      —      |
| 64×64  r=4  | Dense lstsq (cuSOLVER)| `2.484e-30`  |   0.04   |    4,096  |  ~10²⁴× better  |      —      |
| 64×64  r=4  | **TT-DMRG** (3 sw.) | **`7.606e-09`** | **0.28** |  **512**  | **86× better**  |  **8.0×**   |
| 144×144 r=6 | Adam (5000 iters)   | `2.654e-06`    |   8.26   |   20,736  |       1.0×      |      —      |
| 144×144 r=6 | Dense lstsq (cuSOLVER)| `2.210e-29`  |   0.02   |   20,736  |  ~10²³× better  |      —      |
| 144×144 r=6 | **TT-DMRG** (3 sw.) | **`1.843e-07`** | **0.59** | **1,728** | **14× better**  |  **12.0×**  |

**Headline:** on its native domain, DMRG reaches a **1–2 orders of magnitude
lower MSE than Adam**, with **8–12× fewer parameters**, in **5–14× less
wall time**, and **zero hyperparameter tuning**.

### 3.3 Compression vs. accuracy on a non-TT target

From [bench/RESULTS.md](bench/RESULTS.md). Target is `Y = sin(X @ W) + noise`,
which is *not* in the rank-r manifold; this is the BENCHMARK.md three-way
runoff and demonstrates the honest trade-off.

| Layer    | Compression | Adam (500 it) MSE | Dense Exact MSE | TT-DMRG MSE |
| :------- | :---------: | ----------------: | --------------: | ----------: |
| 64×64    |   4.0×      | `6.26e-02`        | `6.26e-02`      | `2.96e-01`  |
| 144×144  |   9.0×      | `5.31e-02`        | `5.31e-02`      | `3.67e-01`  |
| 256×256  |  28.4×      | `5.52e-02`        | `5.52e-02`      | `4.17e-01`  |

DMRG is rank-constrained; on a target outside the TT-rank-r manifold it
trades MSE for parameter compression — an Eckart–Young–Mirsky guarantee,
not a failure mode.

### 3.4 Quality gates

- **29 / 29 pytest tests passing on `cuda:0`** — see
  [tests/conftest.py](tests/conftest.py) and
  [scripts/check.ps1](scripts/check.ps1).
- AGENTS constraint AST scans enforce: **no `backward()`**, **no Adam/SGD**,
  single authorised SVD + QR call-sites — see
  [tests/test_constraints.py](tests/test_constraints.py).
- All four documentation files
  ([ARCHITECTURE.md](docs/ARCHITECTURE.md), [TENSOR_TOPOLOGY.md](docs/TENSOR_TOPOLOGY.md),
  [NUMERICAL_STABILITY.md](docs/NUMERICAL_STABILITY.md),
  [MEMORY_ARENA.md](docs/MEMORY_ARENA.md)) are implemented literally:
  interleaved TT reshape, `r_0 = r_d = 1` invariants, 4-tier SVD fallback,
  Tikhonov damping with NaN escalation, ±5σ Huber target clamp.

---

## 4. Quick start

Requires Python 3.12+, [uv](https://github.com/astral-sh/uv), and a CUDA 12.1
capable GPU. The repo pins `torch==2.5.1` from the explicit
`pytorch-cu121` index in [pyproject.toml](pyproject.toml).

```powershell
git clone https://github.com/seismael/DMRG-Transformer.git
cd DMRG-Transformer
uv sync --extra dev                          # installs torch+cu121, scipy, pytest
uv run python scripts/detect_cuda.py         # smoke test cuSOLVER + GPU
uv run python -m pytest tests --no-header -q # 29 tests, ~50 s
uv run python scripts/run_gate3_proof.py     # -> bench/GATE3_PROOF.md
uv run python scripts/run_poc_benchmark.py   # -> bench/POC_RESULTS.md
uv run python scripts/run_benchmarks.py      # -> bench/RESULTS.md
```

CPU execution is **not supported by default**: every entry point goes
through [src/dmrg_transformer/core/device.py](src/dmrg_transformer/core/device.py)
which calls `torch.cuda.is_available()` and raises if no GPU is present.
A development escape hatch (`DMRG_ALLOW_CPU=1`) exists for triaging only;
see [docs/NUMERICAL_STABILITY.md](docs/NUMERICAL_STABILITY.md) §4 for the
spec-mandated SVD CPU fallback that runs *inside* the GPU pipeline when
`gesdd` diverges.

### Drop-in replacement for `nn.Linear`

```python
import torch
from dmrg_transformer.core.device import require_cuda
from dmrg_transformer.nn import TTLinear
from dmrg_transformer.optim import DMRGOptimizer

device = require_cuda()
layer = TTLinear(in_dims=[32, 32], out_dims=[32, 32], max_rank=16, device=device)
X = torch.randn(2048, 1024, dtype=torch.float64, device=device)
Y = torch.randn(2048, 1024, dtype=torch.float64, device=device)

Y_hat = layer(X)                 # forward pass — NO autograd graph is built

optimizer = DMRGOptimizer(max_rank=16)
report = layer.dmrg_step(X, Y, optimizer)   # replaces optimizer.step()
print(report.mse_before, '->', report.mse_after)
```

### Multi-head attention

```python
from dmrg_transformer.nn import TTMultiHeadAttention

mha = TTMultiHeadAttention(
    d_model=512, n_heads=8,
    in_dims=[16, 32], out_dims=[16, 32], max_rank=16, device=device,
)
seq = torch.randn(4, 64, 512, dtype=torch.float64, device=device)
out = mha(seq)                                # (4, 64, 512)
mha.dmrg_step_projections(seq, seq)           # concurrent DMRG on W_Q/W_K/W_V
```

### Raw tensor train

```python
from dmrg_transformer.tt import TensorTrain
from dmrg_transformer.optim import DMRGOptimizer

tt, trunc_report = TensorTrain.from_dense(
    W_dense, in_dims=[32, 32], out_dims=[32, 32], max_rank=16,
)
opt = DMRGOptimizer(max_rank=16)
for _ in range(2):
    opt.sweep(tt, X, Y)          # bidirectional L->R then R->L
W_updated = tt.to_dense()
```

---

## 5. Architecture

| AGENTS layer                           | This repo                                         |
| :------------------------------------- | :------------------------------------------------ |
| **L1 Network topology** (Transformer)  | [src/dmrg_transformer/nn](src/dmrg_transformer/nn) — `TTLinear`, `TTMultiHeadAttention` |
| **L2 Orchestration microkernel**       | [src/dmrg_transformer/optim](src/dmrg_transformer/optim), [src/dmrg_transformer/propagation](src/dmrg_transformer/propagation), [src/dmrg_transformer/tt](src/dmrg_transformer/tt) |
| **L3 Mathematical execution engine**   | [src/dmrg_transformer/core](src/dmrg_transformer/core) — `robust_svd` (4-tier), `qr_f64`, precision policy, CUDA device guard |

AGENTS OOD interfaces (`ITensorTrain`, `ITargetPropagator`, `IDMRGOptimizer`)
are implemented as runtime-checkable `Protocol`s in
[src/dmrg_transformer/core/interfaces.py](src/dmrg_transformer/core/interfaces.py).

| Standard DL concept             | DMRG-Transformer equivalent                             |
| :------------------------------ | :------------------------------------------------------ |
| `nn.Linear(in, out)`            | `TTLinear(in_dims, out_dims, max_rank)`                 |
| `optimizer.step()`              | `DMRGOptimizer.sweep(tt, X, Y)`                         |
| Backpropagation (chain rule)    | Layer-wise `TargetPropagator` (pseudo-inverse)          |
| Weight update                   | Local SVD projection in the TT manifold                 |
| Weight decay / regularisation   | Eckart–Young–Mirsky SVD truncation at rank `r`          |

### Per-update complexity

| Optimisation paradigm | Mechanism             | Complexity                       |
| :-------------------- | :-------------------- | :------------------------------- |
| Gradient Descent      | Iterative chain rule  | $\mathcal{O}(N^2)$ per step, many steps |
| Dense exact solver    | Pseudo-inverse        | $\mathcal{O}(N^3)$ once                 |
| **DMRG-Transformer**  | SVD on TT-cores       | $\mathcal{O}(d \cdot n \cdot r^3)$      |

---

## 6. Validation gates

| Gate | Test module                                                                  | Asserts                                                   |
| :--- | :--------------------------------------------------------------------------- | :-------------------------------------------------------- |
| 1    | [tests/test_gate1_reconstruction.py](tests/test_gate1_reconstruction.py)     | TT-SVD reconstruction == Eckart–Young–Mirsky bound        |
| 2    | [tests/test_gate2_orthogonality.py](tests/test_gate2_orthogonality.py)       | $L_{<k}^\top L_{<k} = I$ to `< 1e-7`                      |
| 3    | [tests/test_gate3_exact_parity.py](tests/test_gate3_exact_parity.py)         | DMRG MSE == dense `lstsq` MSE on low-TT-rank data         |
| 4    | [docs/BENCHMARK.md](docs/BENCHMARK.md) (Phase IV)                            | GPU tensor-core utilisation > 80 % — **future work**      |
| —    | [tests/test_constraints.py](tests/test_constraints.py)                       | No gradients, no Adam/SGD, single SVD/QR call-site        |

Run the full quality gate (ruff + tests + benchmark smoke):

```powershell
pwsh -File scripts/check.ps1
```

---

## 7. Package layout

```
src/dmrg_transformer/
  core/          interfaces, precision policy, 4-tier SVD fallback, QR,
                 device guard (require_cuda)
  tt/            TensorTrain, gauge/orthogonalisation, environments
  optim/         local_solver (matrix-free normal equations + Tikhonov +
                 SVD truncation), DMRGOptimizer
  propagation/   layer-wise target propagation (replaces backprop)
  nn/            TTLinear, TTMultiHeadAttention
  bench/         BENCHMARK.md reproduction
scripts/
  detect_cuda.py            smoke-test cuSOLVER on cuda:0
  check.ps1                 lint + types + tests quality gate
  run_gate3_proof.py        -> bench/GATE3_PROOF.md
  run_poc_benchmark.py      -> bench/POC_RESULTS.md
  run_benchmarks.py         -> bench/RESULTS.md (three-way runoff)
tests/                      29 tests including all AGENTS gates;
                            conftest.py pins default device to cuda:0
docs/                       architecture specs — source of truth
```

---

## 8. Design invariants (do not violate)

- **GPU-only execution** through [core/device.py](src/dmrg_transformer/core/device.py)
  — every public entry point routes tensors through `require_cuda()`. The
  only authorised CPU code-path is the spec-mandated SVD Tier-2/3 fallback
  in [core/svd.py](src/dmrg_transformer/core/svd.py)
  ([NUMERICAL_STABILITY.md](docs/NUMERICAL_STABILITY.md) §4), which falls
  back to scipy `gesdd`/`gesvd` only when cuSOLVER returns a non-finite
  result.
- **`float64` storage** inside the DMRG solver; `float32` breaks Gate 2.
- [core/svd.py](src/dmrg_transformer/core/svd.py) is the only SVD call-site;
  [core/qr.py](src/dmrg_transformer/core/qr.py) is the only QR call-site.
  Enforced by [tests/test_constraints.py](tests/test_constraints.py).
- **Interleaved physical index**: `p_k = i_k · j_k`, reshape order
  `(i_1, j_1, i_2, j_2, …)` — see
  [docs/TENSOR_TOPOLOGY.md](docs/TENSOR_TOPOLOGY.md) §2.
- **Boundary ranks** `r_0 = r_d = 1` — checked on every construction.
- **Matrix-free local solve** — the per-core normal-equation tensor never
  materialises the `[batch · M, r² · p]` Jacobian; it is contracted from the
  left/right environment blocks in
  [optim/local_solver.py::_build_normal_equations](src/dmrg_transformer/optim/local_solver.py).
- **No `loss.backward()`, no Adam/SGD, no learning rate.** `TTLinear` cores
  are registered as **buffers**, not `nn.Parameter`, to guarantee no
  autograd engagement.

---

## 9. ⚠️ Limitations & Call for Collaborators

This repository is a **faithful initial proof-of-concept**. It proves the
mathematics, validates the math on a real GPU end-to-end, and demonstrates
the performance story on the method's native domain. It is **not yet a
production LLM backbone**. Honest limitations:

1. **Scale.** The pure-PyTorch local solver still allocates the
   `(r²·p) × (r²·p)` normal-equation tensor on-device per core. That caps
   the practical reference at roughly `144 × 144` to `256 × 256` layers on
   a 2 GiB GPU; at `1024 × 1024` it OOMs. The
   [docs/BENCHMARK.md](docs/BENCHMARK.md) `1024 × 1024` target **requires
   AGENTS Phase IV**: a Rust microkernel binding `cuSOLVER` (SVD/QR) and
   `cuTensorNet` (contractions), with double-buffered environment blocks
   per [docs/MEMORY_ARENA.md](docs/MEMORY_ARENA.md).
2. **Hardware.** Reference benchmarks were collected on an MX150 (2 GiB,
   sm_61) — the smallest CUDA-capable card we had access to. Tensor-core
   utilisation targets (Gate 4, `> 80 %`) need a modern Ampere/Hopper card
   (sm_80+) and `cuTensorNet` available. Access to H100, MI300, or TPU v5
   class hardware is required to demonstrate the full asymptotic advantage
   at LLM scale.
3. **End-to-end training.** Single-layer DMRG is validated. Full-network
   **target propagation** across stacked TT-layers in a real Transformer
   training loop is scaffolded
   ([propagation/target_propagator.py](src/dmrg_transformer/propagation/target_propagator.py))
   but not yet benchmarked on a language-modelling workload.
4. **Rank selection.** Adaptive rank schedules (growing `r` with residual
   energy) are not implemented; `max_rank` is currently a static
   hyperparameter.

**We welcome contributions** from the tensor-network, HPC, and ML
communities — especially:

- A Rust + CUDA microkernel implementing AGENTS Phase IV (PyO3 bindings,
  `cuSOLVER` + `cuTensorNet`, double-buffered arenas).
- GPU profiling on Ampere/Hopper hardware and validation of Gate 4.
- A full LLM training loop using `TTMultiHeadAttention` + target
  propagation on WikiText / The Pile.
- Adaptive-rank heuristics and layer-wise rank scheduling.

If you have access to modern accelerator hardware and want to push this to
real scale, please open an issue or PR. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 10. License

Apache 2.0 — see [LICENSE](LICENSE). Security disclosures via
[SECURITY.md](SECURITY.md).
