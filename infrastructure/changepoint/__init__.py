"""
infrastructure.changepoint — causal, lookahead-free structural-break detection.

A live first line (CUSUM / Page-Hinkley, O(1)/bar) plus a Bayesian run-length
posterior (BOCPD, Adams & MacKay 2007) that emits a real `change_prob`. It
COMPLEMENTS topics/regime-classifier/ (batch HMM→XGBoost, which cannot run live):
this runs causally, bar by bar.

Public API:
    from infrastructure.changepoint import run_detector, LiveDetector, make_detector
    s = run_detector(values, name="bocpd")              # per-bar DataFrame
    live = LiveDetector("cusum"); live.update(ts, x_t)  # thin real-time hook

Integration (see integration.py):
    changepoint_features(...)         -> causal feature for regime-classifier Stage 2
    fresh_break_gate(stream)          -> trend-entry allow-mask
    embargo_indices_from_breaks(...)  -> bar positions to purge in cpcv_engine

Crypto instance (infrastructure/). A Polymarket instance gets its OWN copy under
polymarket/research/ — never cross-import (brain/CODEX.md).
"""
from __future__ import annotations

from .detectors import (
    BOCPD,
    CUSUM,
    DETECTORS,
    PageHinkley,
    StepResult,
    make_detector,
)
from .evaluate import (
    benchmark_detector,
    cohens_kappa,
    detection_metrics,
    kappa_vs_transitions,
    match_breaks,
)
from .integration import (
    changepoint_features,
    embargo_indices_from_breaks,
    fresh_break_gate,
)
from .stream import (
    LiveDetector,
    append_changepoints,
    breaks_from_stream,
    causal_standardize,
    run_detector,
)

__all__ = [
    "CUSUM", "PageHinkley", "BOCPD", "StepResult", "DETECTORS", "make_detector",
    "run_detector", "LiveDetector", "append_changepoints", "breaks_from_stream",
    "causal_standardize", "detection_metrics", "match_breaks", "cohens_kappa",
    "kappa_vs_transitions", "benchmark_detector", "changepoint_features",
    "fresh_break_gate", "embargo_indices_from_breaks",
]
