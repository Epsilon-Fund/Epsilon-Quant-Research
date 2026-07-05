"""
The library's one non-negotiable rule: it never imports the private research
stack. Package source may import only the stdlib, its declared third-party
deps, and itself. (See library/README.md — this is what makes the eventual
`git subtree split` a non-event.)
"""
from __future__ import annotations

import ast
from pathlib import Path

PKG_SRC = Path(__file__).resolve().parents[1] / "src"

EPSILON_INTERNAL = {
    "infrastructure", "polymarket", "live_trading", "topics", "midas", "brain", "tools",
}

ALLOWED_TOP_LEVEL = {
    # stdlib used by the package
    "__future__", "argparse", "json", "math", "os", "pathlib", "shutil", "sys",
    # declared deps (pyproject; matplotlib + sklearn are optional extras,
    # imported lazily inside functions)
    "numpy", "pandas", "matplotlib", "sklearn",
    # itself
    "lemma",
}


def _imports(path: Path):
    """Yield top-level module names actually imported by `path` (ast-based,
    so docstrings/comments can never false-positive). Relative imports have
    level > 0 and are internal by construction — skipped."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def _sources():
    files = sorted(PKG_SRC.rglob("*.py"))
    assert files, f"no package sources under {PKG_SRC}"
    return files


def test_no_epsilon_internal_imports():
    offenders = [
        f"{f.name}: {mod}"
        for f in _sources()
        for mod in _imports(f)
        if mod in EPSILON_INTERNAL
    ]
    assert not offenders, "library imports epsilon internals:\n" + "\n".join(offenders)


def test_only_declared_imports():
    offenders = sorted({
        f"{f.name}: {mod}"
        for f in _sources()
        for mod in _imports(f)
        if mod not in ALLOWED_TOP_LEVEL
    })
    assert not offenders, "undeclared imports (add dep or fix):\n" + "\n".join(offenders)
