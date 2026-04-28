# Security Policy

## Supported Versions
Only the latest version of DMRG-Transformer (current: 0.1.0) is supported for security updates.

## Reporting a Vulnerability

If you discover a security vulnerability within this project:

1. **Do NOT report it through public GitHub issues** if it is exploitable.
2. Report it directly to the project maintainers via GitHub's [private vulnerability reporting](https://github.com/seismael/DMRG-Transformer/security/advisories/new) or by email to the repository owner.
3. Include a clear description of the vulnerability, steps to reproduce, and potential impact.

We aim to respond to all security reports within 48 hours and provide a fix within 7 days of confirmation.

## Scope

This is a research implementation of a novel optimization algorithm. Security considerations include:

- **Numerical stability:** NaN/Inf propagation could crash training pipelines. All SVD operations go through the 4-tier fallback hierarchy (`src/dmrg_transformer/core/svd.py`).
- **CUDA memory safety:** The Python `MemoryArena` prototype pre-allocates buffers; future Rust implementation will use the borrow checker for compile-time safety guarantees.
- **Dependency integrity:** All dependencies are pinned via `pyproject.toml` and `uv.lock`. Dependencies with CUDA bindings (PyTorch) are sourced from the official PyTorch index.
