"""Parquet replay adapter: typed L2 tables -> ordered ``MarketEvent`` stream.

Reads the typed Parquet tables produced by the VPS compression pipeline (layout +
schema per [[mm_vps_capture_setup]] / ``polymarket_l2_ingestion.md`` § Parquet schema):

    parquet/{date}/{universe}/{table}_{shard}.parquet   table ∈ {book, trades, price_change, bba}

and emits the SAME :class:`~mm_engine.interfaces.MarketEvent`s as the JSONL adapter, because
both build events through the shared canonical builders in :mod:`mm_engine.events` and order
+ interleave gaps through the shared :func:`mm_engine.feeds._merge.order_and_interleave`. The
gappy fixture converted to Parquet therefore replays byte-identically to its JSONL original
(see ``test_mm_engine_parquet.py``).

Table → event:
    book → "book" (bids/asks JSON arrays);  trades → "last_trade" (price/size/side);
    price_change → "price_change" (price/side/size);  bba → "best_bid_ask"
    (best_bid/best_ask/bid_size/ask_size).

Gaps come from a ``capture_gaps.parquet`` sidecar (column ``disconnect_ms``) written next to
the tables by the converter — the same disconnect timestamps ``load_capture_gaps`` extracts
from ``capture_gaps.jsonl``, so GapMarkers match the JSONL path exactly.

Source may be one ``{date}/{universe}`` directory or a list of them; events are yielded in
global ``ts_exchange`` order (lookahead-free).
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import duckdb

from mm_engine.events import bba_event, book_event, price_change_event, trade_event
from mm_engine.feeds._merge import order_and_interleave
from mm_engine.interfaces import MarketEvent
from mm_engine.events import GapMarker


TABLES = ("book", "trades", "price_change", "bba")
GAPS_FILE = "capture_gaps.parquet"   # column: disconnect_ms INT64

# Authoritative column order per table (polymarket_l2_ingestion.md § Parquet schema).
SCHEMA = {
    "book": ["timestamp_ms", "received_at", "received_ns", "asset_id", "market", "bids", "asks"],
    "trades": ["timestamp_ms", "received_at", "received_ns", "asset_id", "market", "price", "size", "side"],
    "price_change": ["timestamp_ms", "received_at", "received_ns", "asset_id", "market", "price", "side", "size"],
    "bba": ["timestamp_ms", "received_at", "received_ns", "asset_id", "market",
            "best_bid", "best_ask", "bid_size", "ask_size"],
}


def _resolve_dirs(source) -> list[Path]:
    if isinstance(source, (str, Path)):
        return [Path(source)]
    if isinstance(source, Iterable):
        return [Path(p) for p in source]
    raise TypeError(f"unsupported parquet replay source: {source!r}")


def _files(directory: Path, table: str) -> list[str]:
    return sorted(str(p) for p in directory.glob(f"{table}_*.parquet"))


def _read_rows(con: duckdb.DuckDBPyConnection, files: list[str], cols: list[str]) -> list[tuple]:
    select = ", ".join(cols)
    # ORDER is re-imposed globally by order_and_interleave; ORDER BY here only for tidiness.
    return con.execute(
        f"SELECT {select} FROM read_parquet(?) ORDER BY timestamp_ms, received_ns", [files]
    ).fetchall()


def _maybe_json_levels(raw: object) -> object:
    # bids/asks stored as a JSON string ("[[price,size],...]"); _norm_levels also accepts
    # the already-parsed list, so a non-string passes straight through.
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return raw


def _events_for_dir(con: duckdb.DuckDBPyConnection, directory: Path) -> list[MarketEvent]:
    events: list[MarketEvent] = []

    book_files = _files(directory, "book")
    if book_files:
        for ts, rcv_at, rcv_ns, asset_id, market, bids, asks in _read_rows(con, book_files, SCHEMA["book"]):
            events.append(book_event(
                asset_id=asset_id, market=market,
                bids=_maybe_json_levels(bids), asks=_maybe_json_levels(asks),
                ts_exchange=ts, ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns,
            ))

    trade_files = _files(directory, "trades")
    if trade_files:
        for ts, rcv_at, rcv_ns, asset_id, market, price, size, side in _read_rows(con, trade_files, SCHEMA["trades"]):
            events.append(trade_event(
                asset_id=asset_id, market=market, price=price, side=side, size=size,
                ts_exchange=ts, ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns,
            ))

    pc_files = _files(directory, "price_change")
    if pc_files:
        for ts, rcv_at, rcv_ns, asset_id, market, price, side, size in _read_rows(con, pc_files, SCHEMA["price_change"]):
            events.append(price_change_event(
                asset_id=asset_id, market=market, price=price, side=side, size=size,
                ts_exchange=ts, ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns,
            ))

    bba_files = _files(directory, "bba")
    if bba_files:
        for ts, rcv_at, rcv_ns, asset_id, market, bbid, bask, bsz, asz in _read_rows(con, bba_files, SCHEMA["bba"]):
            events.append(bba_event(
                asset_id=asset_id, market=market, best_bid=bbid, best_ask=bask,
                bid_size=bsz, ask_size=asz, ts_exchange=ts, ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns,
            ))

    return events


def load_parquet_gaps(directories: Iterable[Path]) -> list[int]:
    """Disconnect timestamps (ms epoch) from each dir's ``capture_gaps.parquet`` sidecar."""
    out: list[int] = []
    con = duckdb.connect()
    try:
        for d in directories:
            gap_file = Path(d) / GAPS_FILE
            if gap_file.exists():
                rows = con.execute(
                    "SELECT disconnect_ms FROM read_parquet(?)", [str(gap_file)]
                ).fetchall()
                out.extend(int(r[0]) for r in rows if r[0] is not None)
    finally:
        con.close()
    return sorted(out)


def replay_parquet(
    source,
    *,
    gaps: list[int] | None = None,
) -> Iterator[MarketEvent | GapMarker]:
    """Yield ``MarketEvent``s from typed Parquet in ``ts_exchange`` order, interleaving gaps.

    ``source``: one ``{date}/{universe}`` dir or a list of them. ``gaps``: explicit disconnect
    timestamps (ms epoch); if ``None``, loaded from each dir's ``capture_gaps.parquet``.
    """
    dirs = _resolve_dirs(source)
    con = duckdb.connect()
    try:
        events: list[MarketEvent] = []
        for d in dirs:
            events.extend(_events_for_dir(con, d))
    finally:
        con.close()

    if gaps is None:
        gaps = load_parquet_gaps(dirs)

    yield from order_and_interleave(events, gaps)
