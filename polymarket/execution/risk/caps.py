"""Per-trade, per-market, and total-deployed dollar caps."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import CandidateOrder, RiskState, Veto

if TYPE_CHECKING:
    from polymarket.execution.config import ExecutionConfig


def check_size_cap(
    config: ExecutionConfig, state: RiskState, order: CandidateOrder
) -> Veto | None:
    if order.size_usd > config.per_trade_cap_usd:
        return Veto(
            reason="size_cap",
            detail=(
                f"order size {order.size_usd:.2f} USD exceeds "
                f"per_trade_cap_usd {config.per_trade_cap_usd:.2f}"
            ),
        )
    return None


def check_market_cap(
    config: ExecutionConfig, state: RiskState, order: CandidateOrder
) -> Veto | None:
    projected = state.deployed_in_market_usd + order.size_usd
    if projected > config.per_market_cap_usd:
        return Veto(
            reason="market_cap",
            detail=(
                f"projected market deployment {projected:.2f} USD "
                f"(existing {state.deployed_in_market_usd:.2f} + new "
                f"{order.size_usd:.2f}) exceeds per_market_cap_usd "
                f"{config.per_market_cap_usd:.2f} for condition_id "
                f"{order.condition_id}"
            ),
        )
    return None


def check_deployed_cap(
    config: ExecutionConfig, state: RiskState, order: CandidateOrder
) -> Veto | None:
    projected = state.deployed_usd + order.size_usd
    if projected > config.max_capital_usd:
        return Veto(
            reason="deployed_cap",
            detail=(
                f"projected total deployment {projected:.2f} USD "
                f"(existing {state.deployed_usd:.2f} + new "
                f"{order.size_usd:.2f}) exceeds max_capital_usd "
                f"{config.max_capital_usd:.2f}"
            ),
        )
    return None
