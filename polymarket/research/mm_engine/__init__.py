"""Strategy-agnostic Polymarket market-making engine (Phase 0 scaffold).

One event-driven code path runs the same quoting/queue/latency code in **replay** and in
**live-shadow**, so the backtest can be reconciled against live and made trustworthy. See
the build plan ([[mm_engine_phase01_buildplan]]) and the methodology
([[mm_backtesting_methodology_explainer]]).

Layout:
* :mod:`mm_engine.interfaces` — the shared contract (``MarketEvent``, ``BookState``,
  ``Order``, ``QueueModel``, ``LatencyModel``, ``Strategy``).
* :mod:`mm_engine.events` — envelope -> ``MarketEvent`` conversion shared by both feeds.
* :mod:`mm_engine.book` — ``BookTracker`` (reuses ``lib.clob_book.ClobBook``).
* :mod:`mm_engine.strategies` / :mod:`mm_engine.queue_models` /
  :mod:`mm_engine.latency_models` — the ``SymmetricQuoter`` strategy stub, the three
  queue models (``OptimisticQueue`` / ``RiskAverseQueue`` / ``ProbQueue``), and the
  ``ConstantLatency`` stub.
* :mod:`mm_engine.runner` — ``run_strategy``, the same-code-path harness.
* :mod:`mm_engine.feeds` — ``replay_feed`` and ``live_shadow_feed``.
"""
from __future__ import annotations

from mm_engine.events import GapMarker, envelope_to_events
from mm_engine.interfaces import (
    BookState,
    FillResult,
    LatencyModel,
    MarketEvent,
    Order,
    QueueModel,
    Strategy,
)
from mm_engine.engine import BACKTEST, LIVE_SHADOW, EngineResult, Settlement, run_engine
from mm_engine.fees import FEE_FREE, FeeModel, FeeSchedule
from mm_engine.fills import FillSimulator
from mm_engine.latency_models import ConstantLatency
from mm_engine.orders import OrderManager
from mm_engine.queue_models import OptimisticQueue, ProbQueue, RiskAverseQueue
from mm_engine.reconcile import (
    ReconReport,
    reconcile_against_recording,
    reconcile_results,
    run_and_reconcile,
)
from mm_engine.runner import Decision, run_strategy
from mm_engine.strategies import SymmetricQuoter
from mm_engine.telemetry import Telemetry

__all__ = [
    "BACKTEST",
    "LIVE_SHADOW",
    "FEE_FREE",
    "BookState",
    "ConstantLatency",
    "Decision",
    "EngineResult",
    "FeeModel",
    "FeeSchedule",
    "FillResult",
    "FillSimulator",
    "GapMarker",
    "LatencyModel",
    "MarketEvent",
    "OptimisticQueue",
    "Order",
    "OrderManager",
    "ProbQueue",
    "QueueModel",
    "ReconReport",
    "RiskAverseQueue",
    "Settlement",
    "Strategy",
    "SymmetricQuoter",
    "Telemetry",
    "envelope_to_events",
    "reconcile_against_recording",
    "reconcile_results",
    "run_and_reconcile",
    "run_engine",
    "run_strategy",
]
