"""mm_eval — the Task-4 validation / evaluation layer for the MM engine.

This package *uses* :mod:`mm_engine` (it never modifies it) to turn the engine's raw
fill/order/quote telemetry into a **decision**: per market, under the optimistic and
pessimistic queue models, does a fixed-spread maker plausibly survive?

It is the layer the build plan ([[2026-06-23_mm_engine_phase01_buildplan]]) calls "Task 4 —
validation gates / eval layer ... reads engine logs, no new engine code." The engine is
JOIN-1-locked and certified as a consistency/determinism machine ([[mm_join1_reconciliation_findings]]);
this layer is what produces the per-market *viable / fragile / dead (bracketed)* read that
Task 5 builds the v1 strategy from.

Hard rules honored throughout (per ``brain/CODEX.md`` realism discipline and the task brief):

* **Every result is bracketed** under both the optimistic (``OptimisticQueue``) and pessimistic
  (``RiskAverseQueue``) queue models. ``ProbQueue(0.5)`` is reported as the middle.
* **Confidence intervals, never point estimates**, for any headline number (block bootstrap).
* **No profitability claim.** Until Join-2 live calibration collapses the queue bracket toward a
  live-measured fill rate, every number is a conditional range, not an edge.

Module map:

* :mod:`mm_eval.markets`   — discover quotable tokens in the VPS Parquet and materialize a small
  per-token replay dir (one token per engine run — the multi-token ``OrderManager`` would cancel a
  token's quotes on the next *other-token* event, so per-market replay is required).
* :mod:`mm_eval.metrics`   — the pure numerics: signed markout curve + adverse-selection rate,
  the full scorecard (ND-PnL, PnLMAP, profit ratio, Sharpe, max-DD, fill rate, max inventory,
  quote uptime), block-bootstrap CIs, and the breakeven / viable-fragile-dead verdict.
* :mod:`mm_eval.runner`    — orchestrate engine runs over a materialized token under the queue
  bracket, then attach markout + scorecard.
* :mod:`mm_eval.stability` — CPCV-style temporal stability across the capture window (the
  meaningful-now use of the CV machinery — is the read driven by one window?).
* :mod:`mm_eval.overfitting_hook` — the WIRED-BUT-DORMANT Deflated-Sharpe / CPCV + OOS apparatus,
  labeled honestly: vacuous on a single-config 0-parameter quoter (1 trial → nothing to deflate).
* :mod:`mm_eval.ab`        — the A/B scaffolding (one arm = the symmetric quoter, today).
* :mod:`mm_eval.report`    — assemble the standardized per-universe report.
"""
from __future__ import annotations

__all__ = [
    "markets",
    "metrics",
    "runner",
    "stability",
    "overfitting_hook",
    "ab",
    "report",
]
