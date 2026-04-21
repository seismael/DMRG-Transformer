"""Verify the CUDA toolchain is wired up correctly.

Run with::

    uv run python scripts/detect_cuda.py
"""
from __future__ import annotations

import sys

import torch

from dmrg_transformer.core.device import cuda_available, describe_device


def main() -> int:
    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__}")
    print(f"cuda build  : {torch.version.cuda}")
    print(f"cuda avail  : {cuda_available()}")
    if not cuda_available():
        print("ERROR: torch cannot see a CUDA device.")
        return 1
    print(describe_device())
    # Smoke-test the cuSOLVER path used by robust_svd.
    A = torch.randn(256, 256, device="cuda", dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    recon = (U * S) @ Vh
    err = (A - recon).norm().item() / A.norm().item()
    print(f"cuSOLVER svd reconstruction error: {err:.2e}")
    assert err < 1e-10, "cuSOLVER SVD failed basic sanity check"
    print("CUDA toolchain OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
