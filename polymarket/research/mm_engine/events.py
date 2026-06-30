"""Row/record -> :class:`MarketEvent` conversion — the shared core for ALL feed adapters.

Whether an event arrives as a captured JSONL envelope (replay), a live WS frame
(live-shadow), or a typed Parquet row (replay_parquet), it is turned into a
:class:`MarketEvent` by the SAME per-type builders here (:func:`book_event`,
:func:`trade_event`, :func:`price_change_event`, :func:`bba_event`). That is what keeps the
adapters from drifting — the JSONL and Parquet paths cannot disagree because they construct
the event through identical code.

**Canonical payload.** The builders normalize the payload to exactly the fields the engine
consumes (`price/size/side/bids/asks/best_bid/best_ask`) plus `asset_id`/`market`, with
numbers coerced to ``float``. This is deliberately the intersection that the typed Parquet
schema (see ``polymarket_l2_ingestion.md`` § Parquet schema) can carry — raw-only fields
(``hash``/``tick_size``/``spread``/…) are dropped so a JSONL event and the Parquet row it was
compressed from produce byte-identical :class:`MarketEvent`s.

:class:`GapMarker` is the out-of-band control item the feeds interleave to signal a capture
gap / disconnect (NOT a :class:`MarketEvent`, so the event stream stays identical across
sources while gap signalling rides alongside).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mm_engine.interfaces import MarketEvent


# Raw capture event_type -> canonical engine vocabulary. Everything else passes through
# unchanged so an unexpected type is still emitted (and visible) rather than silently dropped.
_TYPE_MAP = {
    "book": "book",
    "price_change": "price_change",
    "last_trade_price": "last_trade",
    "best_bid_ask": "best_bid_ask",
}

# Event types whose canonical name we know how to turn into MarketEvents.
KNOWN_TYPES = frozenset({"book", "price_change", "last_trade", "best_bid_ask"})


@dataclass(frozen=True)
class GapMarker:
    """Out-of-band signal that a capture gap / disconnect happened in the stream.

    ``reason`` is ``"capture_gap"`` (replay / replay_parquet, from the gaps log/table) or
    ``"disconnect"`` / ``"reconnect"`` (live shadow). The book tracker treats it as
    "the book is now suspect until the next full ``book`` snapshot re-anchors it."
    """

    reason: str
    ts_local_iso: str | None = None
    detail: dict = field(default_factory=dict)


def normalize_event_type(raw: Any) -> str:
    name = str(raw or "")
    return _TYPE_MAP.get(name, name)


def _iso_to_epoch_ms(value: Any) -> int | None:
    """Parse a ``received_at``-style ISO string to ms epoch (fallback ts only)."""
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


def _f(value: Any) -> float | None:
    """Coerce to a finite float, or None (non-finite NaN/inf -> None, matching book._to_float).

    Mapping non-finite to None keeps payloads comparable: a NaN would make a payload dict
    compare unequal to an identical one (NaN != NaN), which would spuriously break the
    JSONL↔Parquet equivalence even when the two sources are truly identical.
    """
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _norm_levels(raw: Any) -> list[dict]:
    """Normalize book levels to ``[{"price": float, "size": float}, ...]``, order-preserving.

    Accepts the JSONL form ``[{"price": "..", "size": ".."}, ...]`` AND the Parquet form
    ``[[price, size], ...]`` so both adapters get an identical canonical payload.
    """
    out: list[dict] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        if isinstance(item, dict):
            p, s = _f(item.get("price")), _f(item.get("size"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            p, s = _f(item[0]), _f(item[1])
        else:
            continue
        if p is None or s is None:
            continue
        out.append({"price": p, "size": s})
    return out


def _side(value: Any) -> str:
    return str(value or "")


# --------------------------------------------------------------------------------------
# canonical per-type builders — the ONLY place a MarketEvent is constructed
# --------------------------------------------------------------------------------------

def book_event(*, asset_id, market, bids, asks, ts_exchange, ts_local_iso, ts_monotonic_ns) -> MarketEvent:
    return MarketEvent(
        type="book",
        token_id=str(asset_id),
        ts_exchange=int(ts_exchange),
        ts_local_iso=str(ts_local_iso or ""),
        ts_monotonic_ns=int(ts_monotonic_ns or 0),
        payload={
            "asset_id": str(asset_id),
            "market": market,
            "bids": _norm_levels(bids),
            "asks": _norm_levels(asks),
        },
    )


def trade_event(*, asset_id, market, price, side, size, ts_exchange, ts_local_iso, ts_monotonic_ns) -> MarketEvent:
    return MarketEvent(
        type="last_trade",
        token_id=str(asset_id),
        ts_exchange=int(ts_exchange),
        ts_local_iso=str(ts_local_iso or ""),
        ts_monotonic_ns=int(ts_monotonic_ns or 0),
        payload={
            "asset_id": str(asset_id),
            "market": market,
            "price": _f(price),
            "side": _side(side),
            "size": _f(size),
        },
    )


def price_change_event(*, asset_id, market, price, side, size, ts_exchange, ts_local_iso, ts_monotonic_ns) -> MarketEvent:
    # DELIBERATELY omits best_bid/best_ask even though the raw JSONL change carries them: the
    # typed Parquet price_change schema has NO best_bid/best_ask columns, so including them
    # would make the JSONL payload differ from the Parquet one and break byte-equivalence.
    # They only ever fed book._TokenBook.telemetry_best_bid/ask, which nothing reads (the L1
    # cross-check uses the best_bid_ask EVENT, whose canonical payload DOES keep them). Keep
    # this omission unless the Parquet schema gains those columns.
    return MarketEvent(
        type="price_change",
        token_id=str(asset_id),
        ts_exchange=int(ts_exchange),
        ts_local_iso=str(ts_local_iso or ""),
        ts_monotonic_ns=int(ts_monotonic_ns or 0),
        payload={
            "asset_id": str(asset_id),
            "market": market,
            "price": _f(price),
            "side": _side(side),
            "size": _f(size),
        },
    )


def bba_event(*, asset_id, market, best_bid, best_ask, bid_size, ask_size, ts_exchange, ts_local_iso, ts_monotonic_ns) -> MarketEvent:
    return MarketEvent(
        type="best_bid_ask",
        token_id=str(asset_id),
        ts_exchange=int(ts_exchange),
        ts_local_iso=str(ts_local_iso or ""),
        ts_monotonic_ns=int(ts_monotonic_ns or 0),
        payload={
            "asset_id": str(asset_id),
            "market": market,
            "best_bid": _f(best_bid),
            "best_ask": _f(best_ask),
            "bid_size": _f(bid_size),
            "ask_size": _f(ask_size),
        },
    )


# --------------------------------------------------------------------------------------
# JSONL envelope -> events (parses the raw message, then routes through the builders)
# --------------------------------------------------------------------------------------

def _ts_exchange(message: dict, received_at: Any) -> int:
    """PM ``message.timestamp`` (ms epoch). Fall back to ``received_at`` if absent."""
    raw = message.get("timestamp")
    if raw not in (None, ""):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            pass
    fallback = _iso_to_epoch_ms(received_at)
    return fallback if fallback is not None else 0


def _monotonic_ns(rec: dict) -> int:
    raw = rec.get("received_monotonic_ns")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def envelope_to_events(rec: dict) -> list[MarketEvent]:
    """Convert one capture envelope record into zero or more :class:`MarketEvent` objects.

    Parses the raw PM message and hands primitives to the canonical builders. A
    ``price_change`` message fans out to one event per entry in ``price_changes``.
    Records without a usable ``asset_id`` (or an unknown event type) yield no events.
    """
    message = rec.get("message")
    if not isinstance(message, dict):
        return []

    etype = normalize_event_type(rec.get("event_type") or message.get("event_type"))
    received_at = rec.get("received_at")
    ts_iso = str(received_at or "")
    ts_ns = _monotonic_ns(rec)
    ts_exchange = _ts_exchange(message, received_at)

    if etype == "price_change":
        market = message.get("market")
        out: list[MarketEvent] = []
        for change in message.get("price_changes") or []:
            if not isinstance(change, dict) or not str(change.get("asset_id") or ""):
                continue
            out.append(
                price_change_event(
                    asset_id=change.get("asset_id"),
                    market=market,
                    price=change.get("price"),
                    side=change.get("side"),
                    size=change.get("size"),
                    ts_exchange=ts_exchange,
                    ts_local_iso=ts_iso,
                    ts_monotonic_ns=ts_ns,
                )
            )
        return out

    asset_id = str(message.get("asset_id") or "")
    if not asset_id:
        return []
    market = message.get("market")
    if etype == "book":
        return [book_event(asset_id=asset_id, market=market, bids=message.get("bids"),
                           asks=message.get("asks"), ts_exchange=ts_exchange,
                           ts_local_iso=ts_iso, ts_monotonic_ns=ts_ns)]
    if etype == "last_trade":
        return [trade_event(asset_id=asset_id, market=market, price=message.get("price"),
                            side=message.get("side"), size=message.get("size"),
                            ts_exchange=ts_exchange, ts_local_iso=ts_iso, ts_monotonic_ns=ts_ns)]
    if etype == "best_bid_ask":
        return [bba_event(asset_id=asset_id, market=market, best_bid=message.get("best_bid"),
                          best_ask=message.get("best_ask"), bid_size=message.get("bid_size"),
                          ask_size=message.get("ask_size"), ts_exchange=ts_exchange,
                          ts_local_iso=ts_iso, ts_monotonic_ns=ts_ns)]
    return []
