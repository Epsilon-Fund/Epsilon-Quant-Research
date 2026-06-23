"""The Phase 0 engine driver — the same-code-path harness.

:func:`run_strategy` drives any :class:`~mm_engine.interfaces.Strategy` over any feed of
:class:`~mm_engine.interfaces.MarketEvent` (interleaved with
:class:`~mm_engine.events.GapMarker`s): it maintains the per-token book via
:class:`~mm_engine.book.BookTracker`, asks the strategy to quote on each event, and logs
the resulting :class:`Decision`. Because both the replay and live-shadow feeds yield the
same events, running this one function over either feed exercises the identical strategy
code — that equivalence is the Phase 0 acceptance proof.

This is intentionally minimal. Phase 1 grows it into the full machine (order manager,
fill-simulator slot wired to a :class:`QueueModel` × :class:`LatencyModel`, telemetry
journal, reconciliation harness); Phase 0 just needs to log quotes and prove parity.
"""
from __future__ import annotations

from dataclasses import dataclass

from mm_engine.book import BookTracker
from mm_engine.events import GapMarker
from mm_engine.interfaces import MarketEvent, Order, Strategy


@dataclass(frozen=True)
class Decision:
    """One strategy evaluation: the orders it wanted, given the book at one event."""

    seq: int
    ts_exchange: int
    token_id: str
    event_type: str
    stale: bool
    orders: tuple[Order, ...]


def run_strategy(
    feed,
    strategy: Strategy,
    params: dict | None = None,
    *,
    tracker: BookTracker | None = None,
    inventory: float = 0.0,
) -> list[Decision]:
    """Run ``strategy`` over ``feed``; return the ordered list of quote decisions.

    ``feed`` is any iterable of ``MarketEvent | GapMarker``. GapMarkers are routed to the
    book tracker (which marks affected books stale) and produce no decision. Inventory is
    held constant in Phase 0 (the stub strategy is inventory-agnostic anyway).
    """
    tracker = tracker if tracker is not None else BookTracker()
    params = params or {}
    decisions: list[Decision] = []
    seq = 0

    for item in feed:
        if isinstance(item, GapMarker):
            tracker.note_gap(item)
            continue
        ev: MarketEvent = item
        book = tracker.apply(ev)
        orders = strategy.quote(book, inventory, params)
        decisions.append(
            Decision(
                seq=seq,
                ts_exchange=ev.ts_exchange,
                token_id=ev.token_id,
                event_type=ev.type,
                stale=book.stale,
                orders=tuple(orders),
            )
        )
        seq += 1

    return decisions
