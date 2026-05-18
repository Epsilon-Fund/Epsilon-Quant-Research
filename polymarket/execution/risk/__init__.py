"""Circuit breakers and pre-order checks.

All breakers are pure functions of (config, state, order) returning
Veto | None. mirror_engine assembles state once per candidate order
and calls run_all_checks; the first veto wins and subsequent breakers
are not evaluated.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .caps import check_deployed_cap, check_market_cap, check_size_cap
from .daily_loss import check_daily_loss
from .kill_switch import check_kill_switch
from .max_open_positions import check_max_open_positions
from .price_deviation import check_price_deviation
from .types import VETO_REASONS, CandidateOrder, RiskState, Veto

if TYPE_CHECKING:
    from polymarket.execution.config import ExecutionConfig

__all__ = [
    "CandidateOrder",
    "RiskState",
    "VETO_REASONS",
    "Veto",
    "check_daily_loss",
    "check_deployed_cap",
    "check_kill_switch",
    "check_market_cap",
    "check_max_open_positions",
    "check_price_deviation",
    "check_size_cap",
    "run_all_checks",
]


def run_all_checks(
    config: ExecutionConfig, state: RiskState, order: CandidateOrder
) -> Veto | None:
    """Run breakers in fail-fast order; return the first veto or None."""
    for check in (
        check_kill_switch,
        check_max_open_positions,
        check_size_cap,
        check_market_cap,
        check_deployed_cap,
        check_daily_loss,
        check_price_deviation,
    ):
        veto = check(config, state, order)
        if veto is not None:
            return veto
    return None
