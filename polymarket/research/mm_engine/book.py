"""Per-token book reconstruction behind the :class:`BookState` interface.

Wraps :class:`lib.clob_book.ClobBook` (the existing book builder / OFI engine) and adds
the two things the engine needs on top of raw level-keeping:

* **Staleness** — a book is stale until a full ``book`` snapshot anchors it, once it sits
  beyond the staleness window (default 5s, per the methodology explainer's gap rule), or
  while a capture gap / disconnect has invalidated it.
* **Gap handling** — :meth:`note_gap` marks every known token's book suspect until its
  next full ``book`` re-anchors it. In replay the gap comes from ``capture_gaps.jsonl``;
  in live shadow it comes from a WS disconnect. Either way the recovery is the same: a
  fresh ``book`` snapshot clears it.

Convention preserved from ``dali_clob_replay_features.py``: ``best_bid_ask`` is
telemetry-only and never mutates the executable book; only ``book`` and ``price_change``
move levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lib.clob_book import ClobBook
from mm_engine.events import GapMarker
from mm_engine.interfaces import BookState, MarketEvent


DEFAULT_STALENESS_MS = 5_000
DEFAULT_DEPTH = 10


def _as_levels(raw: object) -> list[tuple[float, float]]:
    """Parse a ``[{"price": .., "size": ..}, ...]`` array into ``(price, size)`` tuples."""
    out: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        price = item.get("price")
        size = item.get("size")
        try:
            price_f = float(price)
            size_f = float(size)
        except (TypeError, ValueError):
            continue
        out.append((price_f, size_f))
    return out


def _to_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out


@dataclass
class _TokenBook:
    book: ClobBook = field(default_factory=ClobBook)
    last_depth_ts: int | None = None   # ts_exchange of last book/price_change applied
    anchored: bool = False             # a full `book` snapshot has been seen
    gap_pending: bool = False          # a gap invalidated us; awaiting a fresh `book`
    telemetry_best_bid: float | None = None
    telemetry_best_ask: float | None = None


@dataclass
class BookTracker:
    """Maintains a :class:`ClobBook` per token and emits :class:`BookState` snapshots."""

    staleness_ms: int = DEFAULT_STALENESS_MS
    depth: int = DEFAULT_DEPTH
    _books: dict[str, _TokenBook] = field(default_factory=dict)

    def _token(self, token_id: str) -> _TokenBook:
        tb = self._books.get(token_id)
        if tb is None:
            tb = _TokenBook()
            self._books[token_id] = tb
        return tb

    def note_gap(self, marker: GapMarker | None = None) -> None:
        """Mark every known token's book suspect until its next full ``book`` snapshot."""
        for tb in self._books.values():
            tb.gap_pending = True

    def apply(self, ev: MarketEvent) -> BookState:
        """Update the relevant token's book for ``ev`` and return its current state."""
        tb = self._token(ev.token_id)

        if ev.type == "book":
            tb.book.replace(_as_levels(ev.payload.get("bids")), _as_levels(ev.payload.get("asks")))
            tb.anchored = True
            tb.gap_pending = False          # a full snapshot re-anchors us after a gap
            tb.last_depth_ts = ev.ts_exchange

        elif ev.type == "price_change":
            side = str(ev.payload.get("side") or "").upper()
            price = _to_float(ev.payload.get("price"))
            size = _to_float(ev.payload.get("size"))
            tb.telemetry_best_bid = _to_float(ev.payload.get("best_bid"))
            tb.telemetry_best_ask = _to_float(ev.payload.get("best_ask"))
            if side in {"BUY", "SELL"} and price is not None and size is not None:
                # Deltas applied after a gap are unreliable; we update levels for
                # continuity but leave gap_pending set so the state stays stale until
                # a full `book` re-anchors.
                tb.book.update_level(side, price, size)
                tb.last_depth_ts = ev.ts_exchange

        elif ev.type == "best_bid_ask":
            # Telemetry only — never mutates the executable book (replay-features rule).
            tb.telemetry_best_bid = _to_float(ev.payload.get("best_bid"))
            tb.telemetry_best_ask = _to_float(ev.payload.get("best_ask"))

        elif ev.type == "last_trade":
            # Trades do not mutate the resting book here; the fill simulator (Phase 1)
            # consumes them via the QueueModel.
            pass

        return self.snapshot(ev.token_id, ev.ts_exchange)

    def snapshot(self, token_id: str, ts_exchange: int) -> BookState:
        tb = self._token(token_id)
        bids = tuple(tb.book.levels("bid")[: self.depth])
        asks = tuple(tb.book.levels("ask")[: self.depth])
        stale = self._is_stale(tb, ts_exchange)
        return BookState(
            token_id=token_id,
            bids=bids,
            asks=asks,
            ts_exchange=ts_exchange,
            stale=stale,
        )

    def _is_stale(self, tb: _TokenBook, ts_exchange: int) -> bool:
        if not tb.anchored or tb.gap_pending or tb.last_depth_ts is None:
            return True
        if not tb.book.is_complete:
            return True
        return (ts_exchange - tb.last_depth_ts) > self.staleness_ms
