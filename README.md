# DMRG-Transformer

**A Post-Gradient Descent Paradigm Using Tensor Network Optimization**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)](#)

DMRG-Transformer is a research-grade framework designed to replace the traditional Backpropagation and Gradient Descent (GD) engine in neural networks with a topological, exact-solver framework derived from quantum many-body physics.

By redefining the neural network's weight space as a **Tensor Train (TT)** and employing the **Density Matrix Renormalization Group (DMRG)** algorithm, this project achieves mathematically exact local weight optimization. It eliminates the need for learning rates, iterative gradient steps, and the $O(N^3)$ computational bottleneck of dense matrix inversion.

## 🚀 Key Features

- **Gradient-Free Optimization:** No `loss.backward()`, no learning rates, and no vanishing gradients.
- **Tensor Train Weights:** Replaces dense matrices with compressed, high-dimensional tensor manifolds.
- **Exact Local Solver:** Uses Singular Value Decomposition (SVD) and QR Factorization to calculate exact mathematical minima for weights.
- **Transformer Compatible:** Natively integrates with Multi-Head Attention (MHA) and Feed-Forward Networks (FFN).
- **Hybrid Performance:** Orchestrated by a Rust-based microkernel with CUDA-accelerated tensor contractions via `cuTensorNet`.

## 🏗️ Architectural Overview

The DMRG-Transformer operates on three primary layers:

1.  **Network Topology:** Standard Transformer blocks where weights are injected as `TensorTrain` objects.
2.  **Orchestration Microkernel (Rust):** Manages Target Propagation, scheduling ALS sweeps, and memory-safe tensor graph operations.
3.  **Mathematical Execution Engine (GPU):** Stateless layer for high-performance SVD, QR, and tensor contractions.

For a deep dive into the math and systems design, see:
- [Architectural Blueprint](docs/BLUEPRINT.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Tensor Topology](docs/TENSOR_TOPOLOGY.md)

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Rust (latest stable)
- NVIDIA GPU with CUDA 12+ (for hardware acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/seismael/DMRG-Transformer.git
cd DMRG-Transformer

# Install development dependencies
pip install -e ".[dev]"
```

## 📖 Usage

Initializing a DMRG-optimized Transformer layer:

```python
from dmrg_transformer.core import TensorTrainLinear

# Create a layer compatible with 1024x1024 weights, compressed to TT-Rank 32
layer = TensorTrainLinear(in_features=1024, out_features=1024, tt_rank=32)

# Standard forward pass
output = layer(input_tensor)

# Exact optimization sweep (Replaces optimizer.step())
# target is generated via Target Propagation
layer.sweep(target_tensor)
```

## 📊 Performance Analysis

| Optimization Paradigm | Mechanism | Complexity (Per Layer Update) |
| :--- | :--- | :--- |
| **Gradient Descent** | Iterative Chain Rule | $\mathcal{O}(N^2)$ *per step* |
| **DMRG-Transformer** | SVD on TT-Cores | $\mathcal{O}(d \cdot n \cdot r^3)$ |

*Where $d$ is the number of tensor dimensions, $n$ is the dimension size, and $r$ is the TT-Rank.*

## 🤝 Contributing

We welcome contributions from the quantum computing and machine learning communities. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## 🛡️ Security

For security concerns, please refer to our [Security Policy](SECURITY.md).
