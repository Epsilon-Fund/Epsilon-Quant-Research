"""Shared message/event builders for the MM-engine tests.

Message shapes mirror real captured data (see mm_clob_capture_semantics). Everything funnels
through the real ``envelope()`` + ``envelope_to_events()`` so direct engine feeds and the
replay/live-shadow adapters all produce identical events.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from mm_engine.events import envelope_to_events
from scripts.dali_live_clob_capture import envelope

YES = "100"
NO = "200"
MARKET = "0x" + "ab" * 20


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def book_msg(token, ts, bids, asks):
    return {
        "event_type": "book",
        "asset_id": token,
        "market": MARKET,
        "timestamp": str(ts),
        "hash": "h",
        "tick_size": "0.001",
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
    }


def pc_msg(ts, changes):
    # changes: list of (token, side, price, size, best_bid, best_ask)
    return {
        "event_type": "price_change",
        "market": MARKET,
        "timestamp": str(ts),
        "price_changes": [
            {
                "asset_id": t,
                "side": sd,
                "price": str(p),
                "size": str(s),
                "best_bid": str(bb),
                "best_ask": str(ba),
                "hash": "x",
            }
            for (t, sd, p, s, bb, ba) in changes
        ],
    }


def trade_msg(token, ts, side, price, size):
    return {
        "event_type": "last_trade_price",
        "asset_id": token,
        "market": MARKET,
        "timestamp": str(ts),
        "side": side,
        "price": str(price),
        "size": str(size),
        "fee_rate_bps": "0",
        "transaction_hash": "0xdeadbeef",
    }


def bba_msg(token, ts, bb, ba):
    return {
        "event_type": "best_bid_ask",
        "asset_id": token,
        "market": MARKET,
        "timestamp": str(ts),
        "best_bid": str(bb),
        "best_ask": str(ba),
        "spread": str(round(ba - bb, 4)),
    }


def frames(messages) -> list[str]:
    """Raw WS frames (one JSON string per message) for the live-shadow FrameTransport."""
    return [json.dumps(m) for m in messages]


def events(messages):
    """MarketEvents via the real envelope() + envelope_to_events() (message order)."""
    out = []
    for m in messages:
        for rec in envelope(json.dumps(m), {}):
            out.extend(envelope_to_events(rec))
    return out


def write_replay_jsonl(path: Path, messages) -> Path:
    """Write a replay shard via the real envelope() (same path the live adapter uses)."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for msg in messages:
            for rec in envelope(json.dumps(msg), {}):
                fh.write(json.dumps(rec) + "\n")
    return path
