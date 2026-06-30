"""
calibration — project-agnostic calibration scoring layer.

Scores how good probabilistic forecasts are (Brier + Murphy decomposition,
log-loss, ECE/MCE, reliability diagrams, Spiegelhalter's Z, recalibration, and
a model-vs-market edge layer) on top of the forked superforecasting ledger.

Read-only consumer of the ledger: it never writes and never re-implements the
state machine. Duplicated byte-identical into each project's package; the two
books share no state — never cross-import.
"""

from __future__ import annotations

from .core import (
    brier_score,
    calibration_in_the_large,
    calibration_table,
    devig,
    ece,
    implied_prob_american,
    implied_prob_decimal,
    isotonic_recalibrate,
    load_scored_forecasts,
    log_loss,
    market_edge,
    mce,
    murphy_decomposition,
    platt_recalibrate,
    realized_edge,
    reliability_diagram,
    reliability_table,
    resolve_ledger_dir,
    score_ledger,
    spiegelhalter_z,
)

__all__ = [
    "brier_score",
    "calibration_in_the_large",
    "calibration_table",
    "devig",
    "ece",
    "implied_prob_american",
    "implied_prob_decimal",
    "isotonic_recalibrate",
    "load_scored_forecasts",
    "log_loss",
    "market_edge",
    "mce",
    "murphy_decomposition",
    "platt_recalibrate",
    "realized_edge",
    "reliability_diagram",
    "reliability_table",
    "resolve_ledger_dir",
    "score_ledger",
    "spiegelhalter_z",
]
