"""A/B framework — built with ONE arm now (the symmetric quoter), ready for Task 5's second arm.

The task brief: "The A/B framework's baseline arm is the symmetric quoter; there is no second
strategy to compare against until Task 5, so build the A/B scaffolding with one arm now."

So this is deliberately a one-arm harness. An :class:`Arm` wraps a strategy factory + params;
:func:`run_arm` evaluates it across a set of materialized markets under the queue bracket. When
Task 5 produces a parameterized quoter, it becomes a second ``Arm`` and :func:`compare_arms`
diffs the per-market verdicts and per-contract edges — no pipeline change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from mm_engine import SymmetricQuoter

from mm_eval.markets import MarketSpec
from mm_eval.runner import MarketEval, evaluate_market


@dataclass(frozen=True)
class Arm:
    """One strategy under test. ``strategy_factory`` returns a fresh ``Strategy`` per run."""

    name: str
    strategy_factory: Callable = SymmetricQuoter
    extra_params: dict | None = None


BASELINE_ARM = Arm(name="symmetric", strategy_factory=SymmetricQuoter, extra_params=None)


@dataclass
class ArmResult:
    arm: Arm
    evals: dict[str, MarketEval] = field(default_factory=dict)   # token_id -> MarketEval


def run_arm(
    arm: Arm,
    specs_and_dirs: list[tuple[MarketSpec, Path]],
    *,
    days: float,
    gaps: list[int] | None = None,
    **eval_kwargs,
) -> ArmResult:
    """Evaluate one arm across all (spec, materialized-dir) pairs under the queue bracket."""
    out = ArmResult(arm=arm)
    for spec, token_dir in specs_and_dirs:
        out.evals[spec.token_id] = evaluate_market(
            spec, token_dir, days=days, gaps=gaps,
            strategy_factory=arm.strategy_factory, extra_params=arm.extra_params,
            **eval_kwargs,
        )
    return out


def compare_arms(results: list[ArmResult]) -> list[dict]:
    """Per-market diff of arms (verdict + pessimistic net-edge). One arm now → a degenerate table.

    Returns one row per token with each arm's no-rebate verdict and pessimistic-queue net edge —
    the shape Task 5 reads to decide whether its parameterized arm beats the symmetric baseline.
    """
    if not results:
        return []
    tokens = list(results[0].evals)
    rows = []
    for tok in tokens:
        row = {"token_id": tok}
        for r in results:
            ev = r.evals.get(tok)
            if ev is None:
                continue
            row[f"{r.arm.name}__verdict"] = ev.verdict_no_rebate
            row[f"{r.arm.name}__pess_net_edge_cents"] = ev.pessimistic.breakeven_no_rebate.net_edge_cents.point
        rows.append(row)
    return rows
