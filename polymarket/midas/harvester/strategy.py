from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .registry import TokenRegistry


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    min_reprice_ticks: int = 2  # ignore price moves smaller than this to avoid churn


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


Action = Literal["PLACE", "CANCEL", "NO_OP"]


@dataclass(frozen=True, slots=True)
class StrategySignal:
    action: Action
    event_slug: str
    token_id: str | None = None
    condition_id: str | None = None
    price_ticks: int | None = None
    tick_size: float = 0.01  # minimum price increment for this market


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def floor_to_tick(price: float, tick_size: float = 0.01) -> int:
    """Convert a [0,1] float price to integer ticks, always rounding down.

    tick_size=0.01  → 1¢ markets:  0.942 → 94  (range 0–99)
    tick_size=0.001 → 0.1¢ markets: 0.942 → 942 (range 0–999)
    """
    ticks_per_unit = round(1.0 / tick_size)  # 100 or 1000
    return min(math.floor(price * ticks_per_unit), ticks_per_unit - 1)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class TailHarvesterStrategy:
    """Pure decision engine — no I/O, no order state.

    Reads price state from the registry and returns a signal telling the
    execution layer what to do next for a given event slug.

    Entry gate (both must be true):
      1. At least one YES token's best_bid >= bid_threshold  (enforced by registry.qualified)
      2. Market is not marked closed  (registry.is_closed — set by accepting_orders poller)

    If the gate is open, target the cheapest qualified token (lowest best_bid)
    and emit PLACE.  If the gate closes, emit CANCEL when has_open_order is True
    so the execution layer pulls any resting unfilled order.

    Repricing: PLACE is only emitted when no order is open, or when the desired
    price has moved by >= min_reprice_ticks from the current order price.  Smaller
    moves return NO_OP to prevent cancel/replace churn on noisy ticks.
    """

    __slots__ = ("_registry", "_config")

    def __init__(self, registry: TokenRegistry, config: StrategyConfig) -> None:
        self._registry = registry
        self._config = config

    def evaluate(
        self,
        event_slug: str,
        *,
        has_open_order: bool = False,
        current_price_ticks: int | None = None,
        current_token_id: str | None = None,
    ) -> StrategySignal:
        """Return the desired action for *event_slug*.

        has_open_order: True when the OMS currently has a live order for this event.
        current_price_ticks: the price of that order, used to suppress trivial reprices.
        current_token_id: which token the open order is on, used to avoid token churn.

        Selection rule: always target the cheapest qualified token (lowest best_bid =
        most margin to resolution). If a better token appears after we already have an
        open order we switch to it, unless both the token AND the price are within
        min_reprice_ticks of where we already are (to prevent cancel/replace churn).
        """
        if self._registry.is_closed(event_slug):
            action: Action = "CANCEL" if has_open_order else "NO_OP"
            return StrategySignal(action=action, event_slug=event_slug)

        qualified = self._registry.qualified()
        event_tokens = [s for s in qualified if s.record.event_slug == event_slug]

        if not event_tokens:
            action = "CANCEL" if has_open_order else "NO_OP"
            return StrategySignal(action=action, event_slug=event_slug)

        # Always target the cheapest qualified token (most margin).
        target = event_tokens[0]
        ts = target.record.tick_size
        new_price_ticks = floor_to_tick(target.best_bid, ts)

        # Skip tokens at the maximum tick — no spread left to capture.
        max_ticks = round(1.0 / ts) - 1
        if new_price_ticks >= max_ticks:
            action = "CANCEL" if has_open_order else "NO_OP"
            return StrategySignal(action=action, event_slug=event_slug)

        if has_open_order:
            # current_token_id=None means unknown — treat as same token so the price
            # guard still applies (we can't meaningfully switch from an unknown token).
            token_unchanged = current_token_id is None or target.record.token_id == current_token_id
            price_within_threshold = (
                current_price_ticks is not None
                and abs(new_price_ticks - current_price_ticks) < self._config.min_reprice_ticks
            )
            # Suppress if we're already on the best token and the price hasn't moved enough.
            if token_unchanged and price_within_threshold:
                return StrategySignal(action="NO_OP", event_slug=event_slug)

        return StrategySignal(
            action="PLACE",
            event_slug=event_slug,
            token_id=target.record.token_id,
            condition_id=target.record.condition_id,
            price_ticks=new_price_ticks,
            tick_size=ts,
        )
