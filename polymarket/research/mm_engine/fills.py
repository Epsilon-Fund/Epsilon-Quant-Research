"""Fill simulator — realize fills against the REAL trade tape, gated by latency.

On each ``last_trade`` event the engine asks the simulator which resting orders fill. Two
gates, in order:

1. **Latency gate.** A resting order is only "live" once it could have landed:
   ``placement_ts + LatencyModel.round_trip_ms(placement_ts) <= trade.ts_exchange``. A trade
   that prints before our quote could have reached the book cannot fill it. (With the
   ``ConstantLatency`` stub at 0ms, any order placed on an earlier event is live.)
2. **Queue gate.** For orders that pass, the :class:`~mm_engine.interfaces.QueueModel`
   decides the filled quantity via ``fill(order, book, trade) -> FillResult``.

Crucially, fills are realized ONLY when a real trade prints — we never simulate price and
order flow independently (the pitfall in arXiv:2409.12721); the queue model is asked about a
real trade against the real book. Partial fills decrement the order's remaining size.
"""
from __future__ import annotations

from dataclasses import dataclass

from mm_engine.interfaces import LatencyModel, MarketEvent, QueueModel
from mm_engine.orders import ActiveOrder


@dataclass(frozen=True)
class RawFill:
    active: ActiveOrder
    qty: float
    queue_ahead: float


@dataclass
class FillSimulator:
    queue_model: QueueModel
    latency_model: LatencyModel

    def simulate(
        self,
        trade: MarketEvent,
        book,
        active_orders: list[ActiveOrder],
    ) -> list[RawFill]:
        if trade.type != "last_trade":
            return []
        fills: list[RawFill] = []
        for ao in active_orders:
            if ao.order.token_id != trade.token_id or ao.remaining <= 1e-9:
                continue
            # latency gate: quote must have been live (net of round-trip) before the trade
            live_at = ao.placement_ts + self.latency_model.round_trip_ms(ao.placement_ts)
            if trade.ts_exchange < live_at:
                continue
            result = self.queue_model.fill(ao.order, book, trade)
            if result.qty <= 0:
                continue
            qty = min(result.qty, ao.remaining)
            ao.remaining -= qty
            fills.append(RawFill(active=ao, qty=qty, queue_ahead=result.queue_ahead))
        return fills
