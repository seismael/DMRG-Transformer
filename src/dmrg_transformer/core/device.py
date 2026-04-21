"""GPU-only device policy for DMRG-Transformer.

AGENTS.md and ARCHITECTURE.md mandate CUDA execution. This module is the
**single authorized entry point** for device selection. Any code path that
silently falls back to CPU is a policy violation.

Usage::

    from dmrg_transformer.core.device import require_cuda, default_device

    device = require_cuda()  # raises RuntimeError if CUDA is unavailable
    x = torch.empty(1024, 1024, device=device, dtype=torch.float32)

The environment variable ``DMRG_ALLOW_CPU=1`` can be set to temporarily
downgrade to CPU (unit tests, CI without GPU). Production and benchmarks must
never set this flag.
"""
from __future__ import annotations

import os

import torch

_ALLOW_CPU_ENV = "DMRG_ALLOW_CPU"


def cuda_available() -> bool:
    """Return whether a working CUDA device is visible to PyTorch."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _cpu_allowed() -> bool:
    return os.environ.get(_ALLOW_CPU_ENV, "0") not in ("", "0", "false", "False")


def require_cuda() -> torch.device:
    """Return ``cuda:0`` or raise a descriptive ``RuntimeError``.

    This is the primary device accessor. The policy is hard: CPU fallback
    is only permitted when ``DMRG_ALLOW_CPU=1`` is set *explicitly*.
    """
    if cuda_available():
        return torch.device("cuda", 0)
    if _cpu_allowed():
        return torch.device("cpu")
    raise RuntimeError(
        "DMRG-Transformer requires CUDA. No CUDA device is visible to PyTorch.\n"
        "  - Ensure NVIDIA drivers + CUDA 12.1 runtime are installed.\n"
        "  - Install torch from the pytorch-cu121 index (see pyproject.toml).\n"
        "  - For CI without a GPU set DMRG_ALLOW_CPU=1 (NOT permitted in benchmarks)."
    )


def default_device() -> torch.device:
    """Alias for :func:`require_cuda` — kept for readability at call sites."""
    return require_cuda()


def default_dtype() -> torch.dtype:
    """Project default compute dtype (float32 on GPU tensor cores)."""
    return torch.float32


def solver_dtype() -> torch.dtype:
    """Dtype for the inner least-squares solve — float64 per NUMERICAL_STABILITY §2."""
    return torch.float64


def describe_device() -> str:
    """Human-readable string describing the active device (for logs/reports)."""
    dev = require_cuda()
    if dev.type != "cuda":
        return f"device={dev} (CPU fallback — DMRG_ALLOW_CPU set)"
    idx = dev.index or 0
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    return f"device=cuda:{idx} ({name}, sm_{cap[0]}{cap[1]}, {total_mem_gb:.1f} GiB)"
