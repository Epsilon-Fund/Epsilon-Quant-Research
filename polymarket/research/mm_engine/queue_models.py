"""Queue-model STUB. Alvaro owns the real models; this is the placeholder swap point.

Phase 0 ships only :class:`OptimisticQueue` as a minimal, faithful-to-its-name stub so
the engine has *a* :class:`~mm_engine.interfaces.QueueModel` to route fills through. The
realism work — ``RiskAverseQueue`` (advance only on trades), ``ProbQueue``
(Rigtorp/probabilistic size-decrease attribution with a tunable ``f``), and the
``calibrate(live_fills)`` math — is explicitly NOT built here (build plan: "Do NOT build
the queue/latency MODELS beyond stubs"). See [[mm_backtesting_methodology_explainer]] §1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mm_engine.interfaces import BookState, MarketEvent, Order


def _to_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out


def _order_key(order: Order) -> tuple[str, str, float]:
    return (order.token_id, order.side, round(float(order.price), 6))


def _size_at_price(book: BookState, side: str, price: float) -> float:
    """Resting size at ``price`` on the book side our order would join."""
    levels = book.bids if side == "BUY" else book.asks
    for lvl_price, lvl_size in levels:
        if abs(lvl_price - price) < 1e-9:
            return lvl_size
    return 0.0


@dataclass
class OptimisticQueue:
    """Back-of-queue stub: a resting order fills only once the level trades through it.

    When an order is first seen, we assume it joins at the **back** of its price level —
    ``queue_ahead`` = the size already resting there. Each eligible trade (a taker hitting
    our side at our price) first consumes ``queue_ahead``; any overflow fills our order, up
    to its size. This is the "optimistic" extreme in the sense that once we reach the front
    we fill immediately on the next trade-through with no priority haircut — the realistic
    haircuts/attribution are Alvaro's :class:`ProbQueue` to add.

    Eligibility (PM ``last_trade.side`` = aggressor side):
    * our **BUY** (resting bid) fills from a **SELL** trade at price ``<=`` our price;
    * our **SELL** (resting ask) fills from a **BUY** trade at price ``>=`` our price.
    """

    _ahead: dict[tuple[str, str, float], float] = field(default_factory=dict)

    def on_event(self, ev: MarketEvent, book: BookState) -> None:
        # Stub: no incremental queue bookkeeping between trades. The real models track
        # size-ahead from book/price_change here.
        return None

    def fill(self, our_order: Order, book: BookState, trade: MarketEvent | None) -> float:
        if trade is None or trade.type != "last_trade":
            return 0.0

        trade_side = str(trade.payload.get("side") or "").upper()
        trade_price = _to_float(trade.payload.get("price"))
        trade_size = _to_float(trade.payload.get("size"))
        if trade_price is None or trade_size is None or trade_size <= 0:
            return 0.0

        if not self._eligible(our_order, trade_side, trade_price):
            return 0.0

        key = _order_key(our_order)
        if key not in self._ahead:
            self._ahead[key] = _size_at_price(book, our_order.side, our_order.price)

        ahead = self._ahead[key]
        consumed_ahead = min(ahead, trade_size)
        self._ahead[key] = ahead - consumed_ahead
        overflow = trade_size - consumed_ahead
        if overflow <= 0:
            return 0.0
        return min(overflow, float(our_order.size))

    def calibrate(self, live_fills) -> None:
        # Stub: no-op. Real calibration tunes the queue from our own live fills.
        return None

    @staticmethod
    def _eligible(order: Order, trade_side: str, trade_price: float) -> bool:
        if order.side == "BUY":
            return trade_side == "SELL" and trade_price <= order.price + 1e-9
        if order.side == "SELL":
            return trade_side == "BUY" and trade_price >= order.price - 1e-9
        return False
