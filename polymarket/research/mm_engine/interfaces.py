"""The shared MM-engine interface — the ONLY thing both halves of the build touch.

This is the whole "contract" from the Phase 0/1 build plan
([[mm_engine_phase01_buildplan]] / `brain/handoffs/2026-06-23_mm_engine_phase01_buildplan.md`):
one event shape (:class:`MarketEvent`), one book-state shape (:class:`BookState`),
and three function shapes (:class:`QueueModel`, :class:`LatencyModel`, :class:`Strategy`).

Once these are agreed (Join 0) the two halves are independent:

* **Justin (the machine):** feed adapters, book builder, order manager, fill-simulator
  slot, telemetry, reconciliation — all written against these types, with stub
  implementations of the three Protocols.
* **Alvaro (the models):** the real :class:`QueueModel` and :class:`LatencyModel`
  realism models, which drop in wherever the stubs are wired.

Both replay and live-shadow adapters emit IDENTICAL :class:`MarketEvent` objects, so the
same strategy/queue/latency code runs in backtest and in live-shadow — that "same code
path" is what makes the backtest trustworthy (see
[[mm_backtesting_methodology_explainer]] §5).

Repo invariants honored throughout the engine: ``PYTHONPATH=. uv run`` from
``polymarket/research/``; lookahead-free (events ordered by ``ts_exchange`` before any
aggregation); deterministic/seeded replay; lowercase ``0x`` addresses; non-overlap
accounting downstream. ``token_id`` values are CLOB token IDs (decimal strings), not
addresses, so they are left verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class MarketEvent:
    """A single market-channel event, emitted IDENTICALLY by replay and live adapters.

    ``type`` uses the canonical engine vocabulary
    ``"book" | "price_change" | "last_trade" | "best_bid_ask"``. The adapters normalize
    the raw capture name ``last_trade_price`` to ``last_trade`` so both feeds agree.

    ``ts_exchange`` is the PM ``message.timestamp`` (ms epoch) and is the event-time
    ordering key — replay sorts by it, which is what keeps the backtest lookahead-free.
    ``ts_local_iso`` (``received_at``) and ``ts_monotonic_ns`` (``received_monotonic_ns``)
    are local receive clocks for gap/latency measurement only and are NOT comparable
    across the replay/live boundary (replay carries the recorded values; live stamps
    fresh ones).
    """

    type: str                      # "book" | "price_change" | "last_trade" | "best_bid_ask"
    token_id: str
    ts_exchange: int               # PM message.timestamp (ms epoch) — event-time ordering
    ts_local_iso: str              # received_at (UTC ISO)
    ts_monotonic_ns: int           # received_monotonic_ns — gaps/latency only
    payload: dict                  # raw fields (levels+sizes / price+side / etc.)


@dataclass(frozen=True)
class BookState:
    """Reconstructed top-N book for one token at one instant.

    Frozen + tuple-typed so a snapshot is an immutable value that can be shared across the
    strategy, queue model, and telemetry without any of them mutating it (agreed with
    Alvaro when the interface was frozen). ``bids`` are sorted best-first (descending
    price); ``asks`` best-first (ascending price). ``stale`` is ``True`` when the book is
    not yet anchored by a full ``book`` snapshot, when it sits beyond the staleness window,
    or when a capture gap / WS disconnect has invalidated it until the next full ``book``
    re-anchors it. Strategies must not quote off a stale book.
    """

    token_id: str
    bids: tuple[tuple[float, float], ...]  # (price, size), top-N, best-first
    asks: tuple[tuple[float, float], ...]
    ts_exchange: int
    stale: bool                            # True if beyond staleness window or across a capture gap


@dataclass(frozen=True)
class Order:
    """A quote the strategy wants resting in the book.

    Minimal by design — Phase 0 only needs to log/route quotes. ``side`` is ``"BUY"``
    (a resting bid) or ``"SELL"`` (a resting ask); ``price`` is a probability in
    ``(0, 1)``; ``size`` is in contracts. The order manager (Phase 1) adds idempotency
    keys, client IDs, and cancel/replace bookkeeping on top of this.
    """

    token_id: str
    side: str          # "BUY" | "SELL"
    price: float
    size: float
    tag: str = ""      # optional free-form label (e.g. "symmetric")


@dataclass(frozen=True)
class FillResult:
    """Outcome of a queue-model fill check for one resting order against one trade.

    ``qty`` is the filled quantity (0..order size) realized by this trade. ``queue_ahead``
    is the model's estimate of resting size still ahead of our order *after* the trade —
    raw telemetry the validation layer (and the per-quote queue snapshot) need.
    """

    qty: float
    queue_ahead: float


@runtime_checkable
class QueueModel(Protocol):          # ALVARO
    """How a resting order advances through the queue and when it fills.

    The engine calls :meth:`on_event` for every market event (so the model can track
    size-ahead / book mutations), and :meth:`fill` whenever a trade could touch our
    resting order. :meth:`get_queue_ahead` returns the current size-ahead estimate for a
    resting order (snapshotted per quote, no trade needed). :meth:`calibrate` tunes the
    model from our own live fills — the one thing public anonymous L2 can never give us
    offline (see methodology explainer §1).
    """

    def on_event(self, ev: MarketEvent, book: BookState) -> None: ...
    def fill(self, our_order, book: BookState, trade: MarketEvent | None) -> FillResult: ...
    def get_queue_ahead(self, our_order) -> float: ...
    def calibrate(self, live_fills) -> None: ...


@runtime_checkable
class LatencyModel(Protocol):        # ALVARO
    """Round-trip latency (ms) used to decide whether our quote was live for a trade."""

    def round_trip_ms(self, ts_exchange: int) -> float: ...


@runtime_checkable
class Strategy(Protocol):            # JUSTIN stubs (symmetric quoter); real one later
    """Produce the desired resting orders given the current book and inventory."""

    def quote(self, book: BookState, inventory: float, params: dict) -> list[Order]: ...
