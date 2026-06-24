"""Queue-model STUB. Alvaro owns the real models; this is the placeholder swap point.

Phase 0/1 ship only :class:`OptimisticQueue` as a minimal, faithful-to-its-name stub so
the engine has *a* :class:`~mm_engine.interfaces.QueueModel` to route fills through. The
realism work — ``RiskAverseQueue`` (advance only on trades), ``ProbQueue``
(Rigtorp/probabilistic size-decrease attribution with a tunable ``f``), and the real
``calibrate(live_fills)`` math — is explicitly NOT built here (build plan: "Do NOT build
the queue/latency MODELS beyond stubs"). See [[mm_backtesting_methodology_explainer]] §1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mm_engine.interfaces import BookState, FillResult, MarketEvent, Order


def _to_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out


def _order_key(order: Order) -> tuple[str, str, float]:
    return (order.token_id, order.side, round(float(order.price), 6))


def _size_at_price(book: BookState | None, side: str, price: float) -> float:
    """Resting size at ``price`` on the book side our order would join."""
    if book is None:
        return 0.0
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
    to its size. This is the "optimistic" extreme: once we reach the front we fill
    immediately on the next trade-through with no priority haircut — realistic haircuts /
    probabilistic attribution are Alvaro's :class:`ProbQueue` to add.

    Eligibility (PM ``last_trade.side`` = aggressor side):
    * our **BUY** (resting bid) fills from a **SELL** trade at price ``<=`` our price;
    * our **SELL** (resting ask) fills from a **BUY** trade at price ``>=`` our price.

    ``on_event`` caches the latest book per token so :meth:`get_queue_ahead` can initialize
    a freshly-placed order's queue position without being handed a book.
    """

    _ahead: dict[tuple[str, str, float], float] = field(default_factory=dict)
    _books: dict[str, BookState] = field(default_factory=dict)

    def on_event(self, ev: MarketEvent, book: BookState) -> None:
        # Cache the latest book per token so get_queue_ahead() can seed new orders.
        self._books[ev.token_id] = book

    def _ensure(self, order: Order, book: BookState | None = None) -> tuple[str, str, float]:
        key = _order_key(order)
        if key not in self._ahead:
            ref = book if book is not None else self._books.get(order.token_id)
            self._ahead[key] = _size_at_price(ref, order.side, order.price)
        return key

    def get_queue_ahead(self, our_order: Order) -> float:
        return self._ahead[self._ensure(our_order)]

    def fill(self, our_order: Order, book: BookState, trade: MarketEvent | None) -> FillResult:
        key = self._ensure(our_order, book)
        if trade is None or trade.type != "last_trade":
            return FillResult(qty=0.0, queue_ahead=self._ahead[key])

        trade_side = str(trade.payload.get("side") or "").upper()
        trade_price = _to_float(trade.payload.get("price"))
        trade_size = _to_float(trade.payload.get("size"))
        if (
            trade_price is None
            or trade_size is None
            or trade_size <= 0
            or not self._eligible(our_order, trade_side, trade_price)
        ):
            return FillResult(qty=0.0, queue_ahead=self._ahead[key])

        ahead = self._ahead[key]
        consumed_ahead = min(ahead, trade_size)
        self._ahead[key] = ahead - consumed_ahead
        overflow = trade_size - consumed_ahead
        qty = min(overflow, float(our_order.size)) if overflow > 0 else 0.0
        return FillResult(qty=qty, queue_ahead=self._ahead[key])

    def forget(self, our_order: Order) -> None:
        """Drop per-order queue state (called when an order is cancelled/filled out)."""
        self._ahead.pop(_order_key(our_order), None)

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
