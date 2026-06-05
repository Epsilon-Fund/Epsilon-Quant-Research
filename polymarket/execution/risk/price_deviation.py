"""Reject mirror orders if current price has moved >N% from leader's fill price.

Per PLAN.md decision 11: this is a slippage-tolerance check, not a
comparison-to-leader for fairness. The bot will submit at current
price, not leader's; this breaker just decides whether the move from
leader's price to now is too large to follow.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import CandidateOrder, RiskState, Veto

if TYPE_CHECKING:
    from polymarket.execution.config import ExecutionConfig


def check_price_deviation(
    config: ExecutionConfig, state: RiskState, order: CandidateOrder
) -> Veto | None:
    if order.leader_fill_price <= 0:
        raise ValueError(
            f"leader_fill_price must be > 0 (got {order.leader_fill_price}); "
            "this is a bug upstream of risk"
        )
    pct_diff = (
        abs(state.current_market_price - order.leader_fill_price)
        / order.leader_fill_price
        * 100.0
    )
    if pct_diff > config.price_deviation_pct:
        return Veto(
            reason="price_deviation",
            detail=(
                f"current {state.current_market_price:.4f} vs leader "
                f"{order.leader_fill_price:.4f}, pct_diff "
                f"{pct_diff:.3f}% exceeds tolerance "
                f"{config.price_deviation_pct:.3f}%"
            ),
        )
    return None
