"""Reconciliation harness — the Join-1 artifact.

Runs the SAME engine over two event sources of the same window (e.g. the replay adapter
reading captured JSONL, and the live-shadow adapter fed the identical frames) and reports
whether **fill rate / position path / equity** agree within tolerance (target **<5%**).

This is what proves the same-code-path: if the two adapters deliver the same events, the
deterministic engine must produce identical fills, positions, and equity (gap ≈ 0). A
non-zero gap localizes a divergence between the two ingestion paths. (Live-shadow has no
*real* orders, so Join-1 proves consistency + same-code-path; real fills calibrate at Join-2.)

Each leg gets its OWN strategy/queue/latency/tracker/order-manager (built from the supplied
factories) so no state leaks between legs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mm_engine.engine import BACKTEST, EngineResult, run_engine


def _gap(a: float, b: float) -> float:
    denom = max(abs(a), abs(b))
    return 0.0 if denom < 1e-12 else abs(a - b) / denom


def _net(position: dict[str, float]) -> float:
    return sum(position.values())


def _fill_rate(r: EngineResult) -> float:
    return r.fill_count / r.placed_count if r.placed_count else 0.0


@dataclass
class ReconMetric:
    name: str
    a: float
    b: float
    gap: float
    within: bool


@dataclass
class ReconReport:
    tolerance: float
    metrics: list[ReconMetric]
    equity_path_match: bool
    summary_a: dict
    summary_b: dict
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(m.within for m in self.metrics) and self.equity_path_match

    def render(self) -> str:
        lines = [
            "MM engine reconciliation report (Join 1)",
            f"tolerance: {self.tolerance:.1%}   verdict: {'PASS' if self.passed else 'FAIL'}",
            "",
            f"{'metric':<16}{'A':>16}{'B':>16}{'gap':>10}  ok",
            "-" * 60,
        ]
        for m in self.metrics:
            lines.append(
                f"{m.name:<16}{m.a:>16.6g}{m.b:>16.6g}{m.gap:>10.4f}  {'Y' if m.within else 'N'}"
            )
        lines += [
            "-" * 60,
            f"equity_path_match: {self.equity_path_match}",
            f"A L1 both-match frac: {self.summary_a.get('l1_both_match_frac')}",
            f"B L1 both-match frac: {self.summary_b.get('l1_both_match_frac')}",
        ]
        if self.notes:
            lines += ["", "notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def reconcile_results(
    a: EngineResult,
    b: EngineResult,
    *,
    tolerance: float = 0.05,
    equity_eps: float = 1e-9,
) -> ReconReport:
    pairs = [
        ("fill_rate", _fill_rate(a), _fill_rate(b)),
        ("fill_count", float(a.fill_count), float(b.fill_count)),
        ("filled_qty", a.filled_qty, b.filled_qty),
        ("net_position", _net(a.position), _net(b.position)),
        ("cash", a.cash, b.cash),
        ("equity", a.equity, b.equity),
        ("placed", float(a.placed_count), float(b.placed_count)),
        ("quotes", float(a.quote_count), float(b.quote_count)),
    ]
    metrics = [
        ReconMetric(name, av, bv, g := _gap(av, bv), g <= tolerance)
        for (name, av, bv) in pairs
    ]

    # equity path: same length and value-aligned within eps
    equity_path_match = len(a.equity_path) == len(b.equity_path) and all(
        ta == tb and abs(ea - eb) <= equity_eps
        for (ta, ea), (tb, eb) in zip(a.equity_path, b.equity_path)
    )

    notes: list[str] = []
    if not equity_path_match:
        notes.append("equity paths differ — the two event sources diverged (check adapters)")
    return ReconReport(
        tolerance=tolerance,
        metrics=metrics,
        equity_path_match=equity_path_match,
        summary_a=a.summary(),
        summary_b=b.summary(),
        notes=notes,
    )


def run_and_reconcile(
    make_feed_a,
    make_feed_b,
    *,
    strategy_factory,
    queue_factory,
    latency_factory,
    params: dict | None = None,
    mode: str = BACKTEST,
    tolerance: float = 0.05,
) -> ReconReport:
    """Run the engine on two freshly-built feeds and reconcile the results."""
    result_a = run_engine(
        make_feed_a(),
        strategy=strategy_factory(),
        queue_model=queue_factory(),
        latency_model=latency_factory(),
        mode=mode,
        params=params,
    )
    result_b = run_engine(
        make_feed_b(),
        strategy=strategy_factory(),
        queue_model=queue_factory(),
        latency_model=latency_factory(),
        mode=mode,
        params=params,
    )
    return reconcile_results(result_a, result_b, tolerance=tolerance)
