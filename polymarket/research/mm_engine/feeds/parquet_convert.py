"""JSONL envelope -> typed Parquet converter (the VPS compression schema, run locally).

Replicates the EXACT schema the VPS compression pipeline produces (per
[[mm_vps_capture_setup]] / ``polymarket_l2_ingestion.md`` § Parquet schema), so a JSONL
fixture converted here is byte-identical-on-replay to the cloud Parquet. (The repo's
``infrastructure/data/l2_ingestion/compression/pipeline.py`` is only a stub — the real logic
lives on the VPS — so the schema is replicated here, not imported.)

Reuses the JSONL reader and gap extractor from :mod:`mm_engine.feeds.replay` and the field
parsers from :mod:`mm_engine.events`, so the table values match what the replay path reads.
A ``capture_gaps.parquet`` sidecar (``disconnect_ms``) is derived from ``capture_gaps.jsonl``
via the SAME ``load_capture_gaps`` the JSONL adapter uses, so GapMarkers reproduce exactly.

Output layout: ``{out_dir}/{date}/{universe}/{table}_{shard}.parquet`` (+ ``capture_gaps.parquet``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mm_engine.events import _f, _monotonic_ns, _norm_levels, _ts_exchange, normalize_event_type
from mm_engine.feeds.replay import _iter_lines, load_capture_gaps


_PA_SCHEMA = {
    "book": pa.schema([
        ("timestamp_ms", pa.int64()), ("received_at", pa.string()), ("received_ns", pa.int64()),
        ("asset_id", pa.string()), ("market", pa.string()), ("bids", pa.string()), ("asks", pa.string()),
    ]),
    "trades": pa.schema([
        ("timestamp_ms", pa.int64()), ("received_at", pa.string()), ("received_ns", pa.int64()),
        ("asset_id", pa.string()), ("market", pa.string()),
        ("price", pa.float64()), ("size", pa.float64()), ("side", pa.string()),
    ]),
    "price_change": pa.schema([
        ("timestamp_ms", pa.int64()), ("received_at", pa.string()), ("received_ns", pa.int64()),
        ("asset_id", pa.string()), ("market", pa.string()),
        ("price", pa.float64()), ("side", pa.string()), ("size", pa.float64()),
    ]),
    "bba": pa.schema([
        ("timestamp_ms", pa.int64()), ("received_at", pa.string()), ("received_ns", pa.int64()),
        ("asset_id", pa.string()), ("market", pa.string()),
        ("best_bid", pa.float64()), ("best_ask", pa.float64()),
        ("bid_size", pa.float64()), ("ask_size", pa.float64()),
    ]),
}
_GAPS_SCHEMA = pa.schema([("disconnect_ms", pa.int64())])


def _levels_json(raw: object) -> str:
    """Normalize book levels to a JSON ``[[price, size], ...]`` array (schema form)."""
    return json.dumps([[lvl["price"], lvl["size"]] for lvl in _norm_levels(raw)])


def jsonl_to_parquet(
    jsonl_path,
    out_dir,
    *,
    date: str,
    universe: str,
    shard: str,
    gaps_path=None,
) -> dict:
    """Convert one JSONL shard to typed Parquet tables under ``{out_dir}/{date}/{universe}``.

    ``gaps_path``: a ``capture_gaps.jsonl`` whose ``disconnect_or_error`` timestamps become the
    ``capture_gaps.parquet`` sidecar. Returns ``{dir, counts}``.
    """
    rows: dict[str, list[dict]] = {t: [] for t in _PA_SCHEMA}

    for rec in _iter_lines(Path(jsonl_path)):
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        etype = normalize_event_type(rec.get("event_type") or msg.get("event_type"))
        received_at = str(rec.get("received_at") or "")
        received_ns = _monotonic_ns(rec)
        ts = _ts_exchange(msg, received_at)
        market = msg.get("market")

        if etype == "book":
            asset_id = str(msg.get("asset_id") or "")
            if not asset_id:
                continue
            rows["book"].append({
                "timestamp_ms": ts, "received_at": received_at, "received_ns": received_ns,
                "asset_id": asset_id, "market": market,
                "bids": _levels_json(msg.get("bids")), "asks": _levels_json(msg.get("asks")),
            })
        elif etype == "last_trade":
            asset_id = str(msg.get("asset_id") or "")
            if not asset_id:
                continue
            rows["trades"].append({
                "timestamp_ms": ts, "received_at": received_at, "received_ns": received_ns,
                "asset_id": asset_id, "market": market,
                "price": _f(msg.get("price")), "size": _f(msg.get("size")),
                "side": str(msg.get("side") or ""),
            })
        elif etype == "price_change":
            for change in msg.get("price_changes") or []:
                if not isinstance(change, dict):
                    continue
                asset_id = str(change.get("asset_id") or "")
                if not asset_id:
                    continue
                rows["price_change"].append({
                    "timestamp_ms": ts, "received_at": received_at, "received_ns": received_ns,
                    "asset_id": asset_id, "market": market,
                    "price": _f(change.get("price")), "side": str(change.get("side") or ""),
                    "size": _f(change.get("size")),
                })
        elif etype == "best_bid_ask":
            asset_id = str(msg.get("asset_id") or "")
            if not asset_id:
                continue
            rows["bba"].append({
                "timestamp_ms": ts, "received_at": received_at, "received_ns": received_ns,
                "asset_id": asset_id, "market": market,
                "best_bid": _f(msg.get("best_bid")), "best_ask": _f(msg.get("best_ask")),
                "bid_size": _f(msg.get("bid_size")), "ask_size": _f(msg.get("ask_size")),
            })

    out = Path(out_dir) / date / universe
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for table, table_rows in rows.items():
        counts[table] = len(table_rows)
        if table_rows:
            pq.write_table(
                pa.Table.from_pylist(table_rows, schema=_PA_SCHEMA[table]),
                out / f"{table}_{shard}.parquet",
            )

    if gaps_path is not None:
        gaps_file = out / "capture_gaps.parquet"
        disconnect_ms = set(load_capture_gaps(Path(gaps_path)))
        # UNION with any gaps already written for this {date}/{universe} (multiple shards from
        # different run dirs can share one output dir) so earlier shards' gaps aren't clobbered.
        if gaps_file.exists():
            existing = pq.read_table(gaps_file).column("disconnect_ms").to_pylist()
            disconnect_ms.update(int(x) for x in existing if x is not None)
        merged = sorted(disconnect_ms)
        pq.write_table(
            pa.Table.from_pylist([{"disconnect_ms": ms} for ms in merged], schema=_GAPS_SCHEMA),
            gaps_file,
        )
        counts["gaps"] = len(merged)

    return {"dir": str(out), "counts": counts}
