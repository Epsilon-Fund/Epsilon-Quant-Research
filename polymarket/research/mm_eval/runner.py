"""Orchestrate engine runs over a materialized token under the optimistic↔pessimistic queue bracket.

This module *drives* :func:`mm_engine.engine.run_engine` — it never modifies the engine. For one
token it runs the ``SymmetricQuoter`` (the A/B baseline arm) under each queue model with a fresh
engine state, then attaches the markout curve, scorecard, and breakeven reads from
:mod:`mm_eval.metrics`.

**Fee handling — one run, both fee modes.** The fee model changes ``rebates_earned`` but **not**
fills (rebate is a passive credit, not a fill gate), so a single run yields both the captured
truth and the rebate sensitivity:

* ``no_rebate`` (PRIMARY) — the captured ``fee_rate_bps = 0`` reality: rebate = 0, so
  ``net_ex_rebate`` *is* the maker's PnL. This is the honest read.
* ``representative`` (SENSITIVITY, *borrowed* per ``brain/CODEX.md`` rule 2) — the canonical
  ``FEE_BY_CATEGORY`` schedule for the universe (Politics 0.04/0.25, Sports 0.03/0.25): what a
  rebate *would* add if PM turned it on. ``net_with_rebate`` and ``rebate_per_contract`` come from
  this; the breakeven is reported both ways.

Latency is held at ``ConstantLatency(0)`` so the **queue gate is isolated** (latency is
~immaterial for slow politics and is itself a Join-2 live-calibration target — see the methodology
explainer §2). Every number stays bracketed by queue model; nothing is a point estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mm_engine import (BACKTEST, ConstantLatency, FeeModel, OptimisticQueue, ProbQueue,
                       RiskAverseQueue, SymmetricQuoter, Telemetry, run_engine)
from mm_engine.fees import FeeSchedule
from mm_engine.feeds.replay_parquet import replay_parquet
from mm_engine.telemetry import JsonlSink

from mm_eval.markets import MarketSpec
from mm_eval.metrics import (BreakevenRead, MarkoutCurvePoint, Scorecard, breakeven_read,
                             build_scorecard, compute_markout, markout_curve, verdict_from_bracket)
from mm_eval.stability import StabilityReport, temporal_stability

# The queue bracket: optimistic (upper bound on fills) → middle → pessimistic (lower bound).
QUEUE_FACTORIES = {
    "Optimistic": OptimisticQueue,
    "Prob(0.5)": lambda: ProbQueue(0.5),
    "RiskAverse": RiskAverseQueue,
}
OPTIMISTIC = "Optimistic"
PESSIMISTIC = "RiskAverse"

# Representative (borrowed) category schedules for the rebate sensitivity. The CAPTURED truth is
# fee=0/rebate=0; these say "what if PM charged its category fee and paid the historical rebate".
REPRESENTATIVE_FEE = {
    "politics_negrisk": FeeSchedule(0.04, 0.25, fees_enabled=True, source="representative:Politics"),
    "esports": FeeSchedule(0.03, 0.25, fees_enabled=True, source="representative:Sports"),
}
DEFAULT_REP_FEE = FeeSchedule(0.05, 0.25, fees_enabled=True, source="representative:Other")

DEFAULT_PRIMARY_HORIZON_S = 30


def representative_fee_model(spec: MarketSpec) -> FeeModel:
    """A FeeModel that applies the universe's representative schedule to this token (deterministic)."""
    sched = REPRESENTATIVE_FEE.get(spec.universe, DEFAULT_REP_FEE)
    return FeeModel(market_schedules={spec.token_id: sched})


@dataclass
class QueueRun:
    """One (token × queue-model) engine run, with its markout curve, scorecard, and breakeven."""

    queue: str
    scorecard: Scorecard
    markout: list[MarkoutCurvePoint]
    stability: StabilityReport
    # breakeven at the primary horizon, both fee modes
    breakeven_no_rebate: BreakevenRead
    breakeven_representative: BreakevenRead


@dataclass
class MarketEval:
    """Full per-market evaluation across the queue bracket, with the bracketed verdict."""

    spec: MarketSpec
    primary_horizon_s: int
    runs: dict[str, QueueRun] = field(default_factory=dict)
    verdict_no_rebate: str = ""
    verdict_representative: str = ""

    @property
    def optimistic(self) -> QueueRun:
        return self.runs[OPTIMISTIC]

    @property
    def pessimistic(self) -> QueueRun:
        return self.runs[PESSIMISTIC]


def evaluate_market(
    spec: MarketSpec,
    token_dir: Path,
    *,
    days: float,
    gaps: list[int] | None = None,
    queues: dict | None = None,
    primary_horizon_s: int = DEFAULT_PRIMARY_HORIZON_S,
    size: float = 100.0,
    tick: float = 0.001,
    seed: int = 0,
    n_boot: int = 2000,
    n_blocks: int = 8,
    strategy_factory=SymmetricQuoter,
    extra_params: dict | None = None,
    end_date_ms: float | None = None,
) -> MarketEval:
    """Run the engine over ``token_dir`` under each queue model and build the bracketed verdict.

    ``strategy_factory`` defaults to the ``SymmetricQuoter`` (the A/B baseline arm). A second arm
    (Task 5's parameterized strategy) drops in here unchanged — the whole eval pipeline is
    strategy-agnostic.

    ``end_date_ms`` is the Task-5 τ anchor: the frozen ``BookState`` carries no
    time-to-resolution, so the runner (which knows each market's Gamma ``end_date``) injects
    it via ``params`` and the strategy derives the per-event τ from ``book.ts_exchange`` —
    the interface stays untouched.
    """
    queues = queues or QUEUE_FACTORIES
    gaps = gaps if gaps is not None else []
    params = {"half_spread": spec.half_spread, "size": size, "tick": tick, **(extra_params or {})}
    if end_date_ms is not None:
        params["end_date_ms"] = float(end_date_ms)
    rep_fee = representative_fee_model(spec)

    ev = MarketEval(spec=spec, primary_horizon_s=primary_horizon_s)
    for qname, qfactory in queues.items():
        # Lean telemetry: we never read the order log (374k+ ops/run); keeping only fills + quotes
        # (quotes carry the mid trajectory markout needs + the uptime/stale flags) cuts memory and
        # time on the ~0.5M-event tokens.
        tele = Telemetry(fills=JsonlSink(keep=True), orders=JsonlSink(keep=False),
                         quotes=JsonlSink(keep=True))
        result = run_engine(
            replay_parquet(token_dir, gaps=gaps),
            strategy=strategy_factory(),
            queue_model=qfactory(),
            latency_model=ConstantLatency(0.0),
            mode=BACKTEST,
            params=params,
            fee_model=rep_fee,  # rebate computed under the representative schedule; net_ex_rebate = captured truth
            telemetry=tele,
        )
        mr = compute_markout(result.fills, result.quotes)
        curve = markout_curve(mr, seed=seed, n_boot=n_boot)
        sc = build_scorecard(result, days=days)
        stab = temporal_stability(result.fills, result.quotes,
                                  horizon_s=primary_horizon_s, n_blocks=n_blocks)
        cp = next((c for c in curve if c.horizon_s == primary_horizon_s), curve[-1] if curve else None)
        # PRIMARY: captured fee=0 -> rebate 0. SENSITIVITY: representative rebate from the run.
        be_no = breakeven_read(cp, queue=qname, half_spread=spec.half_spread,
                               rebate_per_contract=0.0, fee_mode="no_rebate") if cp else None
        be_rep = breakeven_read(cp, queue=qname, half_spread=spec.half_spread,
                                rebate_per_contract=sc.rebate_per_contract,
                                fee_mode="representative") if cp else None
        ev.runs[qname] = QueueRun(qname, sc, curve, stab, be_no, be_rep)

    if OPTIMISTIC in ev.runs and PESSIMISTIC in ev.runs:
        ev.verdict_no_rebate = verdict_from_bracket(
            ev.runs[OPTIMISTIC].breakeven_no_rebate, ev.runs[PESSIMISTIC].breakeven_no_rebate)
        ev.verdict_representative = verdict_from_bracket(
            ev.runs[OPTIMISTIC].breakeven_representative, ev.runs[PESSIMISTIC].breakeven_representative)
    return ev
