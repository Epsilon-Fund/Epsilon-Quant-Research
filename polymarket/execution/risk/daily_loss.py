"""Halt new orders if daily PnL falls below configured threshold.

Realised PnL only. Mark-to-market is deferred (per PLAN.md open
question). Realised = sum of FillRecorded events today where
side=SELL minus side=BUY at fill price. mirror_engine computes this;
this breaker just reads state.realised_pnl_today_usd.

Comparison is `<=`: a halt threshold of 200 means halt when realised
loss reaches -200 or worse (i.e. realised_pnl_today_usd <= -200).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import CandidateOrder, RiskState, Veto

if TYPE_CHECKING:
    from polymarket.execution.config import ExecutionConfig


def check_daily_loss(
    config: ExecutionConfig, state: RiskState, order: CandidateOrder
) -> Veto | None:
    threshold = -config.daily_loss_halt_usd
    if state.realised_pnl_today_usd <= threshold:
        return Veto(
            reason="daily_loss",
            detail=(
                f"realised PnL today {state.realised_pnl_today_usd:.2f} USD "
                f"at or below halt threshold {threshold:.2f} USD "
                f"(daily_loss_halt_usd={config.daily_loss_halt_usd:.2f})"
            ),
        )
    return None
