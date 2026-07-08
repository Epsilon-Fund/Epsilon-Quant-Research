"""
infrastructure.changepoint — causal, lookahead-free structural-break detection.

SHIM since 2026-07-04: the engine was extracted to the decoupled skills library
(`library/changepoint/`, package `lemma-changepoint`) as the Phase-1 packaging
proof — see brain/reflection/candidates.md RC-001 and library/README.md. Epsilon
consumes the library one-way; this module re-exports the identical public API so
every existing import (`from infrastructure.changepoint import run_detector`)
and the SKILL_MAP invocation lines keep working unchanged.

If the import below fails, install the library package into the crypto venv:

    uv pip install -e "library/changepoint[dev]" --python .venv/bin/python

Epsilon-context notes (unchanged by the extraction):
  * This is the crypto instance. A Polymarket instance, if ever needed, gets its
    OWN install/copy under polymarket/research/ — never cross-import (brain/CODEX.md).
  * It COMPLEMENTS topics/regime-classifier/ (batch HMM→XGBoost, which cannot
    run live): this runs causally, bar by bar.
  * Integration targets: regime-classifier Stage-2 features, trend-entry gating,
    and cpcv_engine embargo (infrastructure/walkforward/cpcv_engine.py).
  * Findings note: topics/regime-classifier/changepoint_detector_findings.md.
"""
from __future__ import annotations

try:
    from lemma.changepoint import (  # noqa: F401
        BOCPD,
        CUSUM,
        DETECTORS,
        LiveDetector,
        PageHinkley,
        StepResult,
        append_changepoints,
        benchmark_detector,
        breaks_from_stream,
        causal_standardize,
        changepoint_features,
        cohens_kappa,
        detection_metrics,
        embargo_indices_from_breaks,
        fresh_break_gate,
        kappa_vs_transitions,
        make_detector,
        match_breaks,
        run_detector,
    )
    from lemma.changepoint import offline  # noqa: F401
except ImportError as e:  # pragma: no cover - environment guard, not logic
    raise ImportError(
        "infrastructure.changepoint is a shim over the extracted library package "
        "'lemma-changepoint' (library/changepoint/). Install it into the crypto "
        "venv first:  uv pip install -e 'library/changepoint[dev]' --python .venv/bin/python"
    ) from e

__all__ = [
    "CUSUM", "PageHinkley", "BOCPD", "StepResult", "DETECTORS", "make_detector",
    "run_detector", "LiveDetector", "append_changepoints", "breaks_from_stream",
    "causal_standardize", "detection_metrics", "match_breaks", "cohens_kappa",
    "kappa_vs_transitions", "benchmark_detector", "changepoint_features",
    "fresh_break_gate", "embargo_indices_from_breaks", "offline",
]
