"""Cap on the number of simultaneously-open positions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import CandidateOrder, RiskState, Veto

if TYPE_CHECKING:
    from polymarket.execution.config import ExecutionConfig


def check_max_open_positions(
    config: ExecutionConfig, state: RiskState, order: CandidateOrder
) -> Veto | None:
    if state.open_positions_count >= config.max_open_positions:
        return Veto(
            reason="max_open_positions",
            detail=(
                f"open positions {state.open_positions_count} at or above "
                f"cap {config.max_open_positions}; refusing new entry"
            ),
        )
    return None
