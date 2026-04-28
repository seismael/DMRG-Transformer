# Contributing to DMRG-Transformer

Thank you for considering contributing! This project welcomes contributions that advance the core mathematical framework and its implementations.

## How Can I Contribute?

### Reporting Bugs
- Use the [GitHub Issue Tracker](https://github.com/seismael/DMRG-Transformer/issues).
- Describe the bug and include steps to reproduce it.
- Mention your environment (OS, Python version, CUDA version, GPU model).

### Suggesting Enhancements
- Open an issue describing the feature you'd like to see.
- Explain why it would be useful and how it fits into the project's mathematical framework.
- Check [FUTURE_WORK.md](FUTURE_WORK.md) first — many extensions are already scoped.

### Pull Requests
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation (see `docs/` directory).
4. Run the quality gate: `uv run python -m pytest tests --no-header -q`
5. Follow the project coding standards below.

## Development Setup

```powershell
git clone https://github.com/seismael/DMRG-Transformer.git
cd DMRG-Transformer
uv sync --extra dev
uv run python scripts/detect_cuda.py
uv run python -m pytest tests --no-header -q
```

## Coding Standards
- **Ruff** for linting and formatting (`uv run ruff check src tests`).
- **Mypy** for strict type checking (`uv run mypy src`).
- Mathematical variable names (W, X, Y, U, S, Vh, L, R, G) are intentional and exempt from naming conventions.
- All weight updates in `src/dmrg_transformer/` must use the DMRG solver — no `loss.backward()` or autograd optimizers.
- SVD operations MUST go through `dmrg_transformer.core.svd.robust_svd()` — the single authorized call site.
- QR operations MUST go through `dmrg_transformer.core.qr.qr_f64()`.

## Architecture Constraints
Before contributing, read these files to understand the project's mathematical and architectural constraints:
1. [AGENTS.md](AGENTS.md) — Prime directives and constraints
2. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System topology
3. [docs/TENSOR_TOPOLOGY.md](docs/TENSOR_TOPOLOGY.md) — Rank boundaries and einsum strings
4. [docs/NUMERICAL_STABILITY.md](docs/NUMERICAL_STABILITY.md) — Float64 policies and SVD fallback
