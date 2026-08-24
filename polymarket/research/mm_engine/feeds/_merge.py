"""Shared stream ordering + gap interleaving for the replay adapters.

Both the JSONL (`replay.py`) and Parquet (`replay_parquet.py`) adapters merge their events
into one global, lookahead-free stream and interleave capture-gap markers through THIS
function — so the two sources cannot drift in either ordering or GapMarker placement.

Ordering key (deterministic, content-based, source-independent):
``(ts_exchange, ts_monotonic_ns, token_id, type_rank)``. Event time first (the lookahead-free
key), then the local receive clock, then a stable per-token/per-type tie-break so that
multi-asset frames (e.g. a ``price_change`` carrying several tokens, which share one
``ts_exchange``+``ts_monotonic_ns``) order identically regardless of whether the events came
from JSONL records or table-by-table Parquet reads. (Order among *different* tokens never
affects engine output anyway — each updates its own book — so this only fixes strict stream
identity.)

Gap interleaving matches the original JSONL behavior exactly: a ``GapMarker`` is emitted
before the first event whose receive time is at or after each disconnect timestamp.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

from mm_engine.events import GapMarker, _iso_to_epoch_ms
from mm_engine.interfaces import MarketEvent


_TYPE_RANK = {"book": 0, "price_change": 1, "last_trade": 2, "best_bid_ask": 3}


def _sort_key(ev: MarketEvent) -> tuple:
    # Content tie-break (json.dumps) after the (time, ns, token, type) key: if two events
    # share ALL of those — e.g. the same asset updated twice in one price_change frame (same
    # ts_exchange + ts_monotonic_ns) — a plain stable sort would order them by *input* order,
    # which differs between JSONL (record order) and Parquet (table-by-table read). Ordering
    # tied events by their canonical payload instead is deterministic and source-independent,
    # so the two adapters can't diverge. (Truly identical payloads are interchangeable.)
    return (
        ev.ts_exchange,
        ev.ts_monotonic_ns,
        ev.token_id,
        _TYPE_RANK.get(ev.type, 9),
        json.dumps(ev.payload, sort_keys=True, default=str),
    )


def _recv_ms(ev: MarketEvent) -> int:
    # Gap placement uses receive time; fall back to ts_exchange if received_at is
    # missing/malformed, so a gap marker is never silently dropped/delayed (both clocks
    # are ms-epoch and close in value).
    ms = _iso_to_epoch_ms(ev.ts_local_iso)
    return ms if ms is not None else ev.ts_exchange


def order_and_interleave(
    events: Iterable[MarketEvent],
    gap_times: Iterable[int],
) -> Iterator[MarketEvent | GapMarker]:
    """Yield ``events`` in canonical order, interleaving ``capture_gap`` GapMarkers."""
    ordered = sorted(events, key=_sort_key)          # key is fully deterministic
    gaps = sorted(set(gap_times))                    # dedupe: one marker per distinct disconnect
    gi = 0
    for ev in ordered:
        recv_ms = _recv_ms(ev)
        while gi < len(gaps) and recv_ms >= gaps[gi]:
            yield GapMarker(reason="capture_gap", detail={"disconnect_ms": gaps[gi]})
            gi += 1
        yield ev
