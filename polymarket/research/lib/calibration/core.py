"""
core.py — calibration scoring layer (epsilon shim over rigorkit-calibrate).

SHIM since 2026-07-05: the engine was extracted to the decoupled skills library
(`library/calibrate/`, package `rigorkit-calibrate`) — see
brain/reflection/candidates.md RC-024 and library/README.md. Epsilon consumes
the library one-way; this module re-exports the identical public API and adds
back the one epsilon-specific piece the public package deliberately does not
carry: the book -> repo-relative ledger-path mapping ($SF_LEDGER_DIR still
wins, same as always).

If the import below fails, install the library package into this project's venv:

    crypto (repo root):        uv pip install -e "library/calibrate[dev]" --python .venv/bin/python
    polymarket (repo root):    uv pip install -e "library/calibrate" --python polymarket/research/.venv/bin/python

Epsilon-context notes (unchanged by the extraction):
  * This file stays byte-identical across the two project copies
    (infrastructure/calibration/ crypto · polymarket/research/lib/calibration/ PM)
    — both are now thin shims, and the two books still share no state.
  * Read-only consumer of the forked superforecasting ledger; the sf.py state
    machine remains the single writer (anti-post-hoc guarantee lives there).
  * Design + gate results: docs/calibration_scoring_layer_findings.md.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from rigorkit.calibrate.core import (  # noqa: F401
        brier_score,
        calibration_in_the_large,
        calibration_table,
        devig,
        ece,
        implied_prob_american,
        implied_prob_decimal,
        isotonic_recalibrate,
        log_loss,
        market_edge,
        mce,
        murphy_decomposition,
        platt_recalibrate,
        realized_edge,
        reliability_diagram,
        reliability_table,
        spiegelhalter_z,
    )
    from rigorkit.calibrate import core as _core
except ImportError as e:  # pragma: no cover - environment guard, not logic
    raise ImportError(
        "rigorkit-calibrate is not installed in this venv. From the repo root:\n"
        "  uv pip install -e \"library/calibrate[dev]\" --python <this-venv>/bin/python\n"
        "(see the module docstring for the per-project line)"
    ) from e

# ── the epsilon-specific piece: book -> repo-relative ledger path ─────────────

_BOOK_SUBPATHS = {
    "polymarket": ("polymarket", "research", "data", "superforecast"),
    "crypto": ("live_trading", "data", "superforecast"),
}


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk up from this file until a directory containing .git is found."""
    here = (start or Path(__file__)).resolve()
    for parent in [here] + list(here.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("could not locate repo root (.git) above " + str(here))


def _books() -> dict:
    root = _find_repo_root()
    return {name: root.joinpath(*sub) for name, sub in _BOOK_SUBPATHS.items()}


def resolve_ledger_dir(book: str | None = None) -> Path:
    """Same contract as sf.py: $SF_LEDGER_DIR wins, else the book (arg or
    $SF_BOOK) maps to its repo-relative ledger. Never guesses."""
    return _core.resolve_ledger_dir(book, books=_books())


def load_scored_forecasts(book: str | None = None):
    """One row per SCORED forecast from the book's append-only event log (read-only)."""
    return _core.load_scored_forecasts(book, books=_books())


def score_ledger(book: str | None = None, n_bins: int = 10) -> dict:
    """One-call scorecard over a book's scored forecasts."""
    return _core.score_ledger(book, n_bins=n_bins, books=_books())


# retained for callers/tests that referenced the env override directly
__all__ = [
    "brier_score", "calibration_in_the_large", "calibration_table", "devig",
    "ece", "implied_prob_american", "implied_prob_decimal",
    "isotonic_recalibrate", "load_scored_forecasts", "log_loss", "market_edge",
    "mce", "murphy_decomposition", "platt_recalibrate", "realized_edge",
    "reliability_diagram", "reliability_table", "resolve_ledger_dir",
    "score_ledger", "spiegelhalter_z",
]
