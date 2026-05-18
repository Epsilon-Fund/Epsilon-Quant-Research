"""Shared types for risk breakers.

All breakers consume (config, state, order) and return Veto | None.
Pure functions; no I/O. State is a snapshot assembled by mirror_engine
once per candidate order, not a live read.
"""
from __future__ import annotations

from dataclasses import dataclass

VETO_REASONS: frozenset[str] = frozenset({
    "size_cap",
    "market_cap",
    "deployed_cap",
    "price_deviation",
    "daily_loss",
    "kill_switch",
    "max_open_positions",
    # Real-venue safety harness (written by mirror_engine, not by a breaker).
    "max_real_orders",
    "operator_aborted",
})


@dataclass(frozen=True, slots=True)
class Veto:
    reason: str
    detail: str

    def __post_init__(self) -> None:
        if self.reason not in VETO_REASONS:
            raise ValueError(
                f"Veto.reason must be one of {sorted(VETO_REASONS)} "
                f"(got {self.reason!r}); see VETO_REASONS"
            )


@dataclass(frozen=True, slots=True)
class CandidateOrder:
    client_order_id: str
    condition_id: str
    asset_id: str
    side: str
    size_usd: float
    leader_fill_price: float


@dataclass(frozen=True, slots=True)
class RiskState:
    current_market_price: float
    deployed_usd: float
    deployed_in_market_usd: float
    open_positions_count: int
    realised_pnl_today_usd: float
    killswitch_present: bool
