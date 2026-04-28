# BENCHMARK.md: Optimization Engine Benchmark Specification

**Version:** 1.1.0 | **Hardware:** NVIDIA GeForce MX150 (sm_61, 2.0 GiB)

## 1. Benchmark Design

The DMRG-Transformer replaces iterative Gradient Descent with an exact algebraic solver.
To validate this claim scientifically, we benchmark three optimization paradigms against
each other on the same synthetic regression target:

1. **Gradient Descent (Adam):** Iterative approximation — the industry standard.
2. **Dense Exact Solver (O(N³)):** `torch.linalg.lstsq` — the mathematical absolute minimum.
3. **TT-DMRG Exact Sweep (O(d·n·r³)):** The proposed innovation — block-diagonal normal equations + SVD truncation.

All timings include `torch.cuda.synchronize()`. Peak memory is measured via `torch.cuda.max_memory_allocated()`.

## 2. Headline Configuration: 1024×1024, rank=32, batch=2048

**Target:** `Y = sin(X·W_true) + 0.1·η` — a full-rank nonlinear target.

| Method | MSE | Time (s) | Peak GPU mem | Params |
| :--- | ---: | ---: | ---: | ---: |
| Adam (500 iters, lr=0.01) | 3.73e-02 | 196.6 | 0.16 GiB | 1,048,576 |
| Dense Exact (`torch.linalg.lstsq`) | 3.73e-02 | 0.78 | 0.09 GiB | 1,048,576 |
| **TT-DMRG** (rank=32, 2 sweeps) | 4.15e-01 | 265.3 | 2.22 GiB | 65,536 |

> **Important:** This target is full-rank. Rank-32 TT cannot represent it perfectly.
> DMRG produces the Pareto-optimal rank-32 point. For TT-rank-bounded targets, DMRG
> reaches machine-precision parity with the dense solver — see `bench/GATE3_PROOF.md`.

## 3. Gate 3 Configuration: 64×64, rank=4, batch=512

**Target:** `Y = X·W_true` where `W_true` is drawn from a rank-4 Tensor Train (TT-native target).

| Method | MSE | Time (s) | Params |
| :--- | ---: | ---: | ---: |
| Dense `torch.linalg.lstsq` | 2.386e-30 | 0.18 | 4,096 |
| **TT-DMRG** (2 sweeps) | 1.245e+01 → 6.379e-02 | — | 512 |
| **TT-DMRG** (20 sweeps) | **1.349e-29** | 0.63 | 512 |

DMRG converges to within 5.7× of the dense exact solver MSE — both at float64 machine precision.
Parameter compression: **8.0×** (512 vs 4,096).

## 4. PoC Configuration: 64×64 r=4 and 144×144 r=6, batch=512

**Target:** `Y = X·W_true` where `W_true` is drawn from a rank-r TT.
Adam: 5,000 iterations at lr=0.01. DMRG: 3 sweeps.

| Config | Method | Final MSE | Wall (s) | Params | vs Adam |
| :--- | :--- | ---: | ---: | ---: | :--- |
| 64×64 r=4 | Adam | 6.538e-07 | 3.57 | 4,096 | 1.0× |
| 64×64 r=4 | **TT-DMRG** | **7.606e-09** | **0.26** | **512** | **86× better** |
| 144×144 r=6 | Adam | 2.654e-06 | 7.52 | 20,736 | 1.0× |
| 144×144 r=6 | **TT-DMRG** | **1.843e-07** | **0.28** | **1,728** | **14× better** |

## 5. Real-World: sklearn-digits Classification

| Setup | Model | Test Acc | Params | Wall (s) |
| :--- | :--- | ---: | ---: | ---: |
| Gate 3.1 | TT-MLP (DMRG) | 0.8833 | 1,194 | 1.81 |
| Gate 3.1 | Dense MLP (AdamW, CE) | 0.9694 | 2,410 | 1.89 |
| Tier 2 | Softmax TTBlock (DMRG) | 0.8722 | 1,946 | 52.3 |
| Tier 2 | Linear TTBlock (DMRG) | 0.8667 | 1,946 | 19.5 |
| Tier 2 | Large Dense (AdamW, CE) | 0.9750 | 3,922 | 85.1 |

## 6. Running Benchmarks

```powershell
# Gate 3 exact parity proof (64×64, rank=4, 20 sweeps)
uv run python scripts/run_gate3_proof.py

# PoC benchmark (3-way runoff: Adam vs Dense vs DMRG)
uv run python scripts/run_poc_benchmark.py

# Headline 1024×1024 benchmark (rank 32, requires ~2.5 GiB VRAM)
uv run python scripts/run_headline_benchmark.py

# Pareto frontier sweep (varying rank vs MSE)
uv run python scripts/run_pareto.py

# Iso-rank fairness baseline
uv run python scripts/run_iso_rank_benchmark.py
```

## 7. Interpretation

1. **On TT-rank-bounded targets:** DMRG converges to the exact same MSE as the dense solver (within float64 roundoff) in O(d·n·r³) instead of O(N³). This is the core mathematical innovation — no learning rate, no hyperparameter tuning, no iteration limit.

2. **On full-rank targets:** DMRG produces the Pareto-optimal rank-r approximation. The asymptotic advantage over Adam appears when N ≫ r — at 1024×1024 with rank=32 on the MX150's Pascal architecture, the DMRG constant factors currently dominate the dense `lstsq` call.

3. **Memory:** DMRG's peak VRAM (2.22 GiB) approaches the MX150's 2.0 GiB limit at 1024×1024 scale due to environment block materialisation. The Phase IV Rust microkernel with zero-allocation double-buffering and direct cuTensorNet bindings is the path to closing both the speed and memory gaps.

4. **Real tasks:** DMRG achieves ~87% test accuracy on sklearn-digits with zero backpropagation and 2.0× parameter compression vs dense Adam baselines. The ~10 pp residual gap is attributed to the Exactness Paradox (Frobenius-minimization vs semantic loss mismatch) and inter-layer drift in stacked blocks — addressed by the planned ADMM outer loop.
