"""Strategy stubs for the MM engine. Phase 0 ships only the symmetric quoter.

The real quoting logic (slow-market, inventory-skewed, news-avoiding — see
[[mm_concepts_and_strategy_buildup]]) is deliberately deferred. What matters in Phase 0
is that *a* :class:`~mm_engine.interfaces.Strategy` runs unchanged through both the replay
and live-shadow adapters, proving the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass

from mm_engine.interfaces import BookState, Order


def best_bid(book: BookState) -> float | None:
    return book.bids[0][0] if book.bids else None


def best_ask(book: BookState) -> float | None:
    return book.asks[0][0] if book.asks else None


def mid(book: BookState) -> float | None:
    bid = best_bid(book)
    ask = best_ask(book)
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


# Defaults: 1c half-spread, 100-contract clips, PM's 0.001 price tick.
DEFAULT_PARAMS = {"half_spread": 0.01, "size": 100.0, "tick": 0.001}


@dataclass
class SymmetricQuoter:
    """Fixed-width, inventory-agnostic two-sided quoter (the A-S A/B baseline).

    Quotes ``mid ± half_spread`` with a constant clip size, ignoring inventory entirely.
    Returns no orders when the book is stale or has no two-sided mid — a strategy must
    never quote off a stale book. Prices are rounded to ``tick`` and clamped to the open
    interval ``(0, 1)`` so a quote is always a valid resting price.

    Example: with ``half_spread=0.01`` and a book of best bid 0.47 / best ask 0.49
    (mid 0.48), it emits BUY 0.47 and SELL 0.49 at the configured size. If the book is
    marked stale (gap, or >5s since the last depth update), it emits nothing.
    """

    name: str = "symmetric"

    def quote(self, book: BookState, inventory: float, params: dict) -> list[Order]:
        if book.stale:
            return []
        m = mid(book)
        if m is None:
            return []

        cfg = {**DEFAULT_PARAMS, **(params or {})}
        half_spread = float(cfg["half_spread"])
        size = float(cfg["size"])
        tick = float(cfg["tick"])

        bid_price = self._round_clamp(m - half_spread, tick)
        ask_price = self._round_clamp(m + half_spread, tick)
        if bid_price is None or ask_price is None or bid_price >= ask_price:
            return []

        return [
            Order(book.token_id, "BUY", bid_price, size, tag=self.name),
            Order(book.token_id, "SELL", ask_price, size, tag=self.name),
        ]

    @staticmethod
    def _round_clamp(price: float, tick: float) -> float | None:
        if tick > 0:
            price = round(round(price / tick) * tick, 10)
        lo, hi = (tick if tick > 0 else 0.0), 1.0 - (tick if tick > 0 else 0.0)
        if price <= 0.0 or price >= 1.0:
            return None
        return min(max(price, lo), hi)
