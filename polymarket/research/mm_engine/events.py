"""Envelope -> :class:`MarketEvent` conversion, shared by BOTH feed adapters.

This is the single function that turns a captured/streamed envelope (the JSONL record
produced by ``scripts/dali_live_clob_capture.py``'s ``envelope()``) into the canonical
:class:`MarketEvent` objects the engine consumes. Both the replay adapter (reading the
records back from disk) and the live-shadow adapter (building them fresh from WS frames
via the same ``envelope()``) call :func:`envelope_to_events`, so the two feeds emit
identical events — that shared conversion is the heart of the same-code-path guarantee.

It also defines :class:`GapMarker`, an out-of-band control item the feeds interleave into
the event stream to signal a capture gap / WS disconnect. A GapMarker is deliberately
NOT a :class:`MarketEvent`: the MarketEvent stream stays identical across replay and live,
while gap signalling (which is feed-specific) rides alongside it.
"""
from __future__ import annotations

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

    ``reason`` is ``"capture_gap"`` (replay, from ``capture_gaps.jsonl``) or
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


def _ts_exchange(message: dict, received_at: Any) -> int:
    """PM ``message.timestamp`` (ms epoch). Fall back to ``received_at`` if absent.

    All four event types carry ``timestamp`` in practice; the fallback only guards a
    malformed record so event-time ordering never crashes.
    """
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

    * ``book`` / ``best_bid_ask`` / ``last_trade_price`` -> one event (keyed on
      ``message.asset_id``); ``payload`` is the raw message dict.
    * ``price_change`` -> one event per entry in ``message.price_changes`` (each carries
      its own ``asset_id``); ``payload`` is that individual change dict, stamped with the
      message-level ``market`` for context. All share the message-level ``timestamp``.

    Records without a usable ``asset_id`` (or an unknown event type) yield no events.
    """
    message = rec.get("message")
    if not isinstance(message, dict):
        return []

    raw_type = rec.get("event_type") or message.get("event_type")
    etype = normalize_event_type(raw_type)
    received_at = rec.get("received_at")
    ts_iso = str(received_at or "")
    ts_ns = _monotonic_ns(rec)

    if etype == "price_change":
        ts_exchange = _ts_exchange(message, received_at)
        market = message.get("market")
        out: list[MarketEvent] = []
        for change in message.get("price_changes") or []:
            if not isinstance(change, dict):
                continue
            token_id = str(change.get("asset_id") or "")
            if not token_id:
                continue
            payload = dict(change)
            payload.setdefault("market", market)
            out.append(
                MarketEvent(
                    type="price_change",
                    token_id=token_id,
                    ts_exchange=ts_exchange,
                    ts_local_iso=ts_iso,
                    ts_monotonic_ns=ts_ns,
                    payload=payload,
                )
            )
        return out

    if etype in {"book", "best_bid_ask", "last_trade"}:
        token_id = str(message.get("asset_id") or "")
        if not token_id:
            return []
        return [
            MarketEvent(
                type=etype,
                token_id=token_id,
                ts_exchange=_ts_exchange(message, received_at),
                ts_local_iso=ts_iso,
                ts_monotonic_ns=ts_ns,
                payload=dict(message),
            )
        ]

    return []
