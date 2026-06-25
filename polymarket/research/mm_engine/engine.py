"""The Phase-1 engine — one event-driven machine, backtest and live-shadow on one code path.

For each item from a feed (``MarketEvent`` or ``GapMarker``):

1. update the per-token book (:class:`~mm_engine.book.BookTracker`) and notify the queue model;
2. **fills first** — on a ``last_trade`` event in BACKTEST mode, the fill simulator realizes
   fills against the resting orders placed on earlier events (a trade hits what's already
   there). Each of our fills is a **passive maker fill**: pay 0 maker fee, **credit the maker
   rebate** to cash, and update cost-basis PnL; LIVE_SHADOW mode logs orders only (no fills);
3. mark equity to current mids;
4. re-quote — the strategy proposes orders, the order manager reconciles them, a
   ``get_queue_ahead`` snapshot is taken per resting order.

PnL is decomposed (measure, don't assume): **realized** (offsetting round-trips), **unrealized**
(open inventory marked-to-mid), and — after :meth:`EngineResult.settle` — **settled** (open
inventory carried to the resolution payoff). PnL is reported three ways — **gross**,
**net-with-rebate**, **net-ex-rebate** — so the ``net_without_rebate`` discipline
([[block_k5_stress_findings]]) is always visible. Fee/rebate schedule comes from
:class:`~mm_engine.fees.FeeModel` (per-market → category → fee_free).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mm_engine.book import BookTracker
from mm_engine.events import GapMarker
from mm_engine.fees import FeeModel
from mm_engine.fills import FillSimulator
from mm_engine.interfaces import LatencyModel, MarketEvent, QueueModel, Strategy
from mm_engine.orders import OrderManager
from mm_engine.strategies import best_ask, best_bid, mid
from mm_engine.telemetry import Telemetry


BACKTEST = "backtest"
LIVE_SHADOW = "live_shadow"


@dataclass
class _Pos:
    """Signed position with average cost basis (BUY adds +qty, SELL adds −qty)."""

    qty: float = 0.0
    cost_basis: float = 0.0


def _apply_fill(pos: _Pos, dq: float, price: float) -> float:
    """Apply a signed fill ``dq`` at ``price``; return realized PnL from any offset.

    Average-cost: extending a position re-averages the basis; reducing/closing realizes
    ``(price − basis)`` on the closed portion (signed by the held direction); a flip seeds
    the new side's basis at ``price``.
    """
    if pos.qty == 0.0 or (pos.qty > 0.0) == (dq > 0.0):
        new_qty = pos.qty + dq
        if abs(new_qty) > 1e-12:
            pos.cost_basis = (pos.cost_basis * pos.qty + price * dq) / new_qty
        else:
            pos.cost_basis = 0.0
        pos.qty = new_qty
        return 0.0

    # opposite sign: closing (and possibly flipping)
    closing = min(abs(dq), abs(pos.qty))
    direction = 1.0 if pos.qty > 0.0 else -1.0
    realized = (price - pos.cost_basis) * closing * direction
    pos.qty += dq
    if abs(pos.qty) < 1e-12:
        pos.qty = 0.0
        pos.cost_basis = 0.0
    elif (pos.qty > 0.0) != (direction > 0.0):
        # flipped past zero — remainder opens a fresh position at this price
        pos.cost_basis = price
    return realized


@dataclass(frozen=True)
class Settlement:
    """PnL decomposition after settling open inventory to a resolution map."""

    realized_pnl: float
    settled_pnl: float
    unrealized_pnl: float          # remaining unmapped tokens, marked to mid
    rebates_earned: float
    taker_fees_paid: float
    gross_pnl: float               # realized + settled + remaining-unrealized
    net_ex_rebate: float
    net_with_rebate: float
    settled_tokens: tuple[str, ...]
    unsettled_tokens: tuple[str, ...]

    def summary(self) -> dict:
        return {
            "realized": self.realized_pnl,
            "settled": self.settled_pnl,
            "unrealized_remaining": self.unrealized_pnl,
            "rebates": self.rebates_earned,
            "gross": self.gross_pnl,
            "net_ex_rebate": self.net_ex_rebate,
            "net_with_rebate": self.net_with_rebate,
            "settled_tokens": list(self.settled_tokens),
            "unsettled_tokens": list(self.unsettled_tokens),
        }


@dataclass
class EngineResult:
    mode: str
    fills: list[dict]
    orders: list[dict]
    quotes: list[dict]
    position: dict[str, float]                       # token -> net qty
    open_positions: dict[str, tuple[float, float]]   # token -> (qty, cost_basis)
    last_mid: dict[str, float]
    gross_cash: float                                # trade cash flow only (no fees/rebate)
    realized_pnl: float
    unrealized_pnl: float                            # open inventory marked to mid (UNREALIZED)
    rebates_earned: float
    taker_fees_paid: float
    equity_path: list[tuple[int, float]]             # (ts, net-with-rebate equity)
    fill_count: int
    filled_qty: float
    placed_count: int
    quote_count: int
    event_count: int
    l1_crosscheck: dict = field(default_factory=dict)

    @property
    def gross_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def net_ex_rebate(self) -> float:
        return self.gross_pnl - self.taker_fees_paid

    @property
    def net_with_rebate(self) -> float:
        return self.net_ex_rebate + self.rebates_earned

    def settle(self, resolution_map: dict[str, float]) -> Settlement:
        """Settle open inventory to resolved payoffs (0.0/1.0). Unmapped tokens stay unrealized.

        Matches the prior research accounting ``position·(payoff − entry) + rebate``
        ([[od_strategy_a_v3_findings]] / k6 static-hedge).
        """
        settled_pnl = 0.0
        unrealized_remaining = 0.0
        settled_tokens: list[str] = []
        unsettled_tokens: list[str] = []
        for token, (qty, basis) in self.open_positions.items():
            if abs(qty) < 1e-12:
                continue
            if token in resolution_map:
                settled_pnl += qty * (float(resolution_map[token]) - basis)
                settled_tokens.append(token)
            else:
                # Unmapped -> stays unrealized. Mark to last mid; if the token never had a
                # two-sided mid, mark at entry (cost_basis -> 0 PnL) rather than silently
                # dropping it, so the unsettled partition and the gross_pnl sum stay consistent.
                mark = self.last_mid.get(token, basis)
                unrealized_remaining += qty * (mark - basis)
                unsettled_tokens.append(token)
        gross = self.realized_pnl + settled_pnl + unrealized_remaining
        net_ex = gross - self.taker_fees_paid
        return Settlement(
            realized_pnl=self.realized_pnl,
            settled_pnl=settled_pnl,
            unrealized_pnl=unrealized_remaining,
            rebates_earned=self.rebates_earned,
            taker_fees_paid=self.taker_fees_paid,
            gross_pnl=gross,
            net_ex_rebate=net_ex,
            net_with_rebate=net_ex + self.rebates_earned,
            settled_tokens=tuple(settled_tokens),
            unsettled_tokens=tuple(unsettled_tokens),
        )

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "events": self.event_count,
            "quotes": self.quote_count,
            "placed": self.placed_count,
            "fills": self.fill_count,
            "filled_qty": self.filled_qty,
            "net_position": sum(self.position.values()),
            "gross_cash": self.gross_cash,
            "realized": self.realized_pnl,
            "unrealized": self.unrealized_pnl,
            "rebates": self.rebates_earned,
            "taker_fees": self.taker_fees_paid,
            "gross_pnl": self.gross_pnl,
            "net_ex_rebate": self.net_ex_rebate,
            "net_with_rebate": self.net_with_rebate,
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
    fee_model: FeeModel | None = None,
    tracker: BookTracker | None = None,
    order_manager: OrderManager | None = None,
    telemetry: Telemetry | None = None,
) -> EngineResult:
    params = params or {}
    tracker = tracker if tracker is not None else BookTracker()
    om = order_manager if order_manager is not None else OrderManager()
    fsim = FillSimulator(queue_model, latency_model)
    tele = telemetry if telemetry is not None else Telemetry.in_memory()
    fees = fee_model if fee_model is not None else FeeModel()
    route_fills = mode == BACKTEST

    positions: dict[str, _Pos] = {}
    last_mid: dict[str, float] = {}
    gross_cash = 0.0
    realized_pnl = 0.0
    rebates_earned = 0.0
    # Stays 0.0 in this passive-only engine: our fills are always MAKER fills (a resting
    # quote hit by someone else's taker), so we pay no taker fee. A future crossing/taker
    # leg is the ONLY place that would accumulate here, via fees.taker_fee(token, qty, p);
    # net_ex_rebate already subtracts this term, so wiring that leg needs no other change.
    taker_fees_paid = 0.0
    equity_path: list[tuple[int, float]] = []
    fill_count = 0
    filled_qty = 0.0
    placed_count = 0
    quote_count = 0
    event_count = 0
    last_ts = 0

    def unrealized_total() -> float:
        out = 0.0
        for tok, pos in positions.items():
            m = last_mid.get(tok)
            if m is not None and abs(pos.qty) > 1e-12:
                out += pos.qty * (m - pos.cost_basis)
        return out

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
                dq = rf.qty if o.side == "BUY" else -rf.qty
                pos = positions.setdefault(o.token_id, _Pos())
                realized_delta = _apply_fill(pos, dq, o.price)
                realized_pnl += realized_delta
                gross_cash += -dq * o.price
                rebate = fees.maker_rebate(o.token_id, rf.qty, o.price)   # passive: earn rebate, no fee
                rebates_earned += rebate
                fill_count += 1
                filled_qty += rf.qty
                sched = fees.schedule_for(o.token_id)
                tele.fills.emit({
                    "ts_exchange": ev.ts_exchange,
                    "token_id": o.token_id,
                    "side": o.side,
                    "price": o.price,
                    "qty": rf.qty,
                    "queue_ahead": rf.queue_ahead,
                    "mid_at_fill": m,
                    "maker_fee": 0.0,
                    "maker_rebate": rebate,
                    "taker_fee_ref": sched.taker_fee(rf.qty, o.price),  # what a taker WOULD pay
                    "fee_rate": sched.fee_rate,
                    "rebate_rate": sched.rebate_rate,
                    "fee_source": sched.source,
                    "realized_delta": realized_delta,
                    "position_after": pos.qty,
                    "cost_basis_after": pos.cost_basis,
                    "gross_cash_after": gross_cash,
                    "client_id": rf.active.client_id,
                    "trade_ts": ev.ts_exchange,
                    "trade_price": float(ev.payload.get("price")),
                    "trade_size": float(ev.payload.get("size")),
                })
            om.drop_filled()

        # 2. mark net-with-rebate equity to current mids
        equity_path.append(
            (ev.ts_exchange, realized_pnl + unrealized_total() + rebates_earned - taker_fees_paid)
        )

        # 3. re-quote and reconcile
        inventory = positions.get(ev.token_id, _Pos()).qty
        desired = strategy.quote(book, inventory, params)
        for op in om.reconcile(desired, ev.ts_exchange):
            if op.op == "place":
                placed_count += 1
            tele.orders.emit(op.as_dict())
        quote_count += 1

        # 4. per-quote queue snapshot
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

    return EngineResult(
        mode=mode,
        fills=tele.fills.records,
        orders=tele.orders.records,
        quotes=tele.quotes.records,
        position={t: p.qty for t, p in positions.items()},
        open_positions={t: (p.qty, p.cost_basis) for t, p in positions.items() if abs(p.qty) > 1e-12},
        last_mid=dict(last_mid),
        gross_cash=gross_cash,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_total(),
        rebates_earned=rebates_earned,
        taker_fees_paid=taker_fees_paid,
        equity_path=equity_path,
        fill_count=fill_count,
        filled_qty=filled_qty,
        placed_count=placed_count,
        quote_count=quote_count,
        event_count=event_count,
        l1_crosscheck=tracker.l1_crosscheck_summary(),
    )
