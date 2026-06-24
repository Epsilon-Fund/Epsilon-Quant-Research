"""The Phase-1 engine — one event-driven machine, backtest and live-shadow on one code path.

For each item from a feed (``MarketEvent`` or ``GapMarker``):

1. update the per-token book (:class:`~mm_engine.book.BookTracker`) and notify the queue model;
2. **fills first** — on a ``last_trade`` event in BACKTEST mode, the fill simulator realizes
   fills against the resting orders placed on earlier events (a trade hits what's already
   there), updating per-token position + cash; LIVE_SHADOW mode logs orders only (no fills);
3. mark equity to current mids;
4. re-quote — the strategy proposes orders, the order manager reconciles them (place/cancel/
   replace/throttle), and a ``get_queue_ahead`` snapshot is taken per resting order.

A ``GapMarker`` marks the book stale and cancels all resting orders (a disconnect drops them).
Everything is logged raw via :class:`~mm_engine.telemetry.Telemetry`. The same function runs
replay and live-shadow — only the feed differs — which is the reconciliation premise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mm_engine.book import BookTracker
from mm_engine.events import GapMarker
from mm_engine.fills import FillSimulator
from mm_engine.interfaces import LatencyModel, MarketEvent, QueueModel, Strategy
from mm_engine.orders import OrderManager
from mm_engine.strategies import best_ask, best_bid, mid
from mm_engine.telemetry import Telemetry


BACKTEST = "backtest"
LIVE_SHADOW = "live_shadow"


@dataclass
class EngineResult:
    mode: str
    fills: list[dict]
    orders: list[dict]
    quotes: list[dict]
    position: dict[str, float]
    cash: float
    equity: float
    equity_path: list[tuple[int, float]]
    fill_count: int
    filled_qty: float
    placed_count: int
    quote_count: int
    event_count: int
    l1_crosscheck: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "events": self.event_count,
            "quotes": self.quote_count,
            "placed": self.placed_count,
            "fills": self.fill_count,
            "filled_qty": self.filled_qty,
            "net_position": sum(self.position.values()),
            "cash": self.cash,
            "equity": self.equity,
            "l1_both_match_frac": self.l1_crosscheck.get("both_match_frac"),
        }


def run_engine(
    feed,
    *,
    strategy: Strategy,
    queue_model: QueueModel,
    latency_model: LatencyModel,
    mode: str = BACKTEST,
    params: dict | None = None,
    tracker: BookTracker | None = None,
    order_manager: OrderManager | None = None,
    telemetry: Telemetry | None = None,
) -> EngineResult:
    params = params or {}
    tracker = tracker if tracker is not None else BookTracker()
    om = order_manager if order_manager is not None else OrderManager()
    fsim = FillSimulator(queue_model, latency_model)
    tele = telemetry if telemetry is not None else Telemetry.in_memory()
    route_fills = mode == BACKTEST

    position: dict[str, float] = {}
    last_mid: dict[str, float] = {}
    cash = 0.0
    equity_path: list[tuple[int, float]] = []
    fill_count = 0
    filled_qty = 0.0
    placed_count = 0
    quote_count = 0
    event_count = 0
    last_ts = 0

    def mark_equity(ts: int) -> float:
        eq = cash + sum(position.get(t, 0.0) * m for t, m in last_mid.items())
        equity_path.append((ts, eq))
        return eq

    for item in feed:
        if isinstance(item, GapMarker):
            tracker.note_gap(item)
            for op in om.cancel_all(last_ts):
                tele.orders.emit(op.as_dict())
            continue

        ev: MarketEvent = item
        event_count += 1
        last_ts = ev.ts_exchange
        book = tracker.apply(ev)
        queue_model.on_event(ev, book)

        m = mid(book)
        if m is not None:
            last_mid[ev.token_id] = m

        # 1. fills first: a real trade hits orders resting from earlier events
        if route_fills and ev.type == "last_trade":
            for rf in fsim.simulate(ev, book, om.active_orders()):
                o = rf.active.order
                if o.side == "BUY":
                    position[o.token_id] = position.get(o.token_id, 0.0) + rf.qty
                    cash -= rf.qty * o.price
                else:
                    position[o.token_id] = position.get(o.token_id, 0.0) - rf.qty
                    cash += rf.qty * o.price
                fill_count += 1
                filled_qty += rf.qty
                tele.fills.emit({
                    "ts_exchange": ev.ts_exchange,
                    "token_id": o.token_id,
                    "side": o.side,
                    "price": o.price,
                    "qty": rf.qty,
                    "queue_ahead": rf.queue_ahead,
                    "mid_at_fill": m,
                    "position_after": position.get(o.token_id, 0.0),
                    "cash_after": cash,
                    "client_id": rf.active.client_id,
                    "trade_ts": ev.ts_exchange,
                    "trade_price": float(ev.payload.get("price")),
                    "trade_size": float(ev.payload.get("size")),
                })
            om.drop_filled()

        # 2. mark equity to current mids
        mark_equity(ev.ts_exchange)

        # 3. re-quote and reconcile
        inventory = position.get(ev.token_id, 0.0)
        desired = strategy.quote(book, inventory, params)
        for op in om.reconcile(desired, ev.ts_exchange):
            if op.op == "place":
                placed_count += 1
            tele.orders.emit(op.as_dict())
        quote_count += 1

        # 4. per-quote queue snapshot for every resting order
        snap = [
            {
                "client_id": ao.client_id,
                "side": ao.order.side,
                "price": ao.order.price,
                "size": ao.order.size,
                "remaining": ao.remaining,
                "queue_ahead": queue_model.get_queue_ahead(ao.order),
            }
            for ao in om.active_orders()
        ]
        tele.quotes.emit({
            "ts_exchange": ev.ts_exchange,
            "token_id": ev.token_id,
            "event_type": ev.type,
            "stale": book.stale,
            "best_bid": best_bid(book),
            "best_ask": best_ask(book),
            "mid": m,
            "orders": snap,
        })

    final_equity = cash + sum(position.get(t, 0.0) * mm for t, mm in last_mid.items())
    return EngineResult(
        mode=mode,
        fills=tele.fills.records,
        orders=tele.orders.records,
        quotes=tele.quotes.records,
        position=position,
        cash=cash,
        equity=final_equity,
        equity_path=equity_path,
        fill_count=fill_count,
        filled_qty=filled_qty,
        placed_count=placed_count,
        quote_count=quote_count,
        event_count=event_count,
        l1_crosscheck=tracker.l1_crosscheck_summary(),
    )
