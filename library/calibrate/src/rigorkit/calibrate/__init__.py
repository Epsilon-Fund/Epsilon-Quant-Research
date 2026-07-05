"""
rigorkit.calibrate — proper-score calibration diagnostics for probabilistic
forecasts.

Answers "are these probabilities any good?" — not with an opinion, but with
proper scores, decompositions, and diagnostics: Brier + Murphy decomposition
(reliability − resolution + uncertainty), log-loss, ECE/MCE, reliability
tables/diagrams with Wilson 95% bands, Spiegelhalter's Z, isotonic/Platt
recalibration, and a markets layer (model-p vs de-vigged implied-p, realized
edge over resolved markets).

Public API:
    from rigorkit.calibrate import brier_score, murphy_decomposition, ece
    brier_score(prob, label)
    murphy_decomposition(prob, label, n_bins=None)   # exact identity form
    reliability_table(prob, label, n_bins=10)        # + Wilson bands
    isotonic_recalibrate(p_train, y_train, p_apply)

Ledger reader (optional, read-only): `load_scored_forecasts` / `score_ledger`
parse an append-only `forecasts/events.jsonl` (superforecasting-skill ledger
convention) — they never write it.
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

__version__ = "0.1.0"

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
