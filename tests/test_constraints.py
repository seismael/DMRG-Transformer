"""AGENTS.md constraint enforcement tests.

Asserts, via static AST scanning of the package sources, that:
  * No ``.backward`` call exists anywhere under ``dmrg_transformer`` (Constraint 1).
  * No import or reference to ``torch.optim`` / Adam / SGD / RMSprop (Constraint 2).
  * ``torch.linalg.svd`` / ``scipy.linalg.svd`` are only called from
    ``core/svd.py`` (the single authorized call site per NUMERICAL_STABILITY §4).
  * ``torch.linalg.qr`` is only called from ``core/qr.py``.
"""
from __future__ import annotations

import ast
import pathlib

PKG_ROOT = pathlib.Path(__file__).parent.parent / "src" / "dmrg_transformer"


def _iter_sources() -> list[pathlib.Path]:
    return sorted(p for p in PKG_ROOT.rglob("*.py") if p.is_file())


def _parse(path: pathlib.Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _attr_chain(node: ast.AST) -> str:
    """Dotted name like ``torch.linalg.svd`` if ``node`` is an Attribute chain."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return ""
    return ".".join(reversed(parts))


def test_no_backward_calls_anywhere() -> None:
    offenders: list[str] = []
    for path in _iter_sources():
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "backward":
                    offenders.append(f"{path.relative_to(PKG_ROOT)}:{node.lineno}")
    assert not offenders, f"AGENTS Constraint 1 violated — .backward calls: {offenders}"


def test_no_iterative_optimizer_references() -> None:
    banned_substrings = ("torch.optim", "optim.adam", "optim.sgd", "optim.rmsprop")
    offenders: list[str] = []
    for path in _iter_sources():
        text = path.read_text(encoding="utf-8").lower()
        for needle in banned_substrings:
            if needle in text:
                offenders.append(f"{path.relative_to(PKG_ROOT)}: found {needle!r}")
    assert not offenders, f"AGENTS Constraint 2 violated: {offenders}"


def test_svd_has_single_authorized_call_site() -> None:
    authorized = PKG_ROOT / "core" / "svd.py"
    offenders: list[str] = []
    for path in _iter_sources():
        if path.resolve() == authorized.resolve():
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _attr_chain(node.func)
                if name.endswith("linalg.svd") or name.endswith("scipy.linalg.svd"):
                    offenders.append(f"{path.relative_to(PKG_ROOT)}:{node.lineno} -> {name}")
    assert not offenders, (
        f"SVD must only be called from core/svd.py (NUMERICAL_STABILITY §4): {offenders}"
    )


def test_qr_has_single_authorized_call_site() -> None:
    authorized = PKG_ROOT / "core" / "qr.py"
    offenders: list[str] = []
    for path in _iter_sources():
        if path.resolve() == authorized.resolve():
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _attr_chain(node.func)
                if name.endswith("linalg.qr"):
                    offenders.append(f"{path.relative_to(PKG_ROOT)}:{node.lineno} -> {name}")
    assert not offenders, f"QR must only be called from core/qr.py: {offenders}"
