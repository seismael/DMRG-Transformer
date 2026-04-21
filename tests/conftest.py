"""Global test fixtures.

AGENTS.md + NUMERICAL_STABILITY.md mandate GPU/CUDA execution. This conftest
sets the default torch device to ``cuda:0`` so any ``torch.randn(...)`` in a
test without an explicit ``device=`` argument allocates on the GPU.

The escape hatch ``DMRG_ALLOW_CPU=1`` (see :mod:`dmrg_transformer.core.device`)
is honoured only for CI without a GPU; production and benchmarks must never
set it.
"""
from __future__ import annotations

import pytest
import torch

from dmrg_transformer.core.device import cuda_available, require_cuda


def pytest_configure(config: pytest.Config) -> None:
    # Pin default device for the whole test session. On GPU-less CI with
    # DMRG_ALLOW_CPU=1 this falls back to CPU; otherwise cuda:0.
    dev = require_cuda()
    torch.set_default_device(dev)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return require_cuda()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if cuda_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
