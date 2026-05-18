"""File-presence-based global halt.

This breaker does NOT touch the filesystem. mirror_engine checks
config.killswitch_path.exists() once per cycle and passes the boolean
via state.killswitch_present. Keeping this pure makes it trivially
testable without filesystem mocks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import CandidateOrder, RiskState, Veto

if TYPE_CHECKING:
    from polymarket.execution.config import ExecutionConfig


def check_kill_switch(
    config: ExecutionConfig, state: RiskState, order: CandidateOrder
) -> Veto | None:
    if state.killswitch_present:
        return Veto(
            reason="kill_switch",
            detail=(
                f"kill switch file present at {config.killswitch_path}; "
                "delete the file to resume new order submission"
            ),
        )
    return None
