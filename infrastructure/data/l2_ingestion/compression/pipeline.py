"""Compression pipeline: finds completed JSONL.gz capture shards, parses them
into typed Parquet tables (one per event type), validates row counts, and stages
the output for cloud sync.

Standalone — imports nothing from discovery/ or capture/. It reads the raw
shards on disk and writes columnar Parquet. It never deletes or edits the source
JSONL.gz (that's the sync step's job).

Input  : data/raw/{date}/{universe}_{hour}.jsonl.gz   (envelopes, one per line)
Output : data/parquet/{date}/{universe}/{book,trades,price_change,bba}.parquet

Run:
    # scan all of data/raw for completed (>1h old), not-yet-processed shards
    python compression/pipeline.py

    # process one directory regardless of age (testing)
    python compression/pipeline.py --input data/raw/2026-06-18/ --force
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent          # .../l2_ingestion/compression
PKG_ROOT = HERE.parent                          # .../l2_ingestion
DEFAULT_RAW_ROOT = PKG_ROOT / "data" / "raw"
DEFAULT_PARQUET_ROOT = PKG_ROOT / "data" / "parquet"
PROCESSED_STATE = DEFAULT_PARQUET_ROOT / "_processed.txt"

AGE_THRESHOLD_S = 3600  # only process shards older than 1 hour (completed)

logger = logging.getLogger("l2.compress")

# event_type -> output table (Parquet basename without extension)
EVENT_TABLE = {
    "book": "book",
    "last_trade_price": "trades",
    "price_change": "price_change",
    "best_bid_ask": "bba",
}

# Column order + which columns are numeric, per table. Envelope-level fields
# (received_at, received_ns, universe) are prepended to every row. Schemas follow
# the architecture spec, trimmed to fields the real payloads actually carry
# (e.g. bba has spread, not bid_size/ask_size).
ENVELOPE_COLS = ["timestamp_ms", "received_at", "received_ns", "universe", "asset_id", "market"]
TABLE_COLS: dict[str, list[str]] = {
    "book": ENVELOPE_COLS + ["bids", "asks"],
    "trades": ENVELOPE_COLS + ["price", "size", "side", "fee_rate_bps", "transaction_hash"],
    "price_change": ENVELOPE_COLS + ["price", "side", "size", "best_bid", "best_ask"],
    "bba": ENVELOPE_COLS + ["best_bid", "best_ask", "spread"],
}
FLOAT_COLS: dict[str, list[str]] = {
    "book": [],
    "trades": ["price", "size", "fee_rate_bps"],
    "price_change": ["price", "size", "best_bid", "best_ask"],
    "bba": ["best_bid", "best_ask", "spread"],
}


def fnum(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_ms(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-row extraction. Returns a list of row dicts (price_change yields many).
# ---------------------------------------------------------------------------
def rows_from_envelope(rec: dict[str, Any]) -> tuple[str, list[dict[str, Any]]] | None:
    """Map one envelope to (table_name, rows). Returns None for non-target
    event types (e.g. new_market)."""
    etype = rec.get("event_type")
    table = EVENT_TABLE.get(str(etype))
    if table is None:
        return None

    msg = rec.get("message")
    if not isinstance(msg, dict):
        return table, []

    assets = rec.get("assets") or []
    universe = str(assets[0]["universe"]) if assets and "universe" in assets[0] else "unknown"
    base = {
        "received_at": rec.get("received_at"),
        "received_ns": rec.get("received_monotonic_ns"),
        "universe": universe,
        "timestamp_ms": to_ms(msg.get("timestamp")),
        "market": str(msg.get("market") or ""),
    }

    if table == "book":
        return table, [{
            **base,
            "asset_id": str(msg.get("asset_id") or rec.get("asset_id") or ""),
            "bids": json.dumps(msg.get("bids") or []),
            "asks": json.dumps(msg.get("asks") or []),
        }]

    if table == "trades":
        return table, [{
            **base,
            "asset_id": str(msg.get("asset_id") or rec.get("asset_id") or ""),
            "price": fnum(msg.get("price")),
            "size": fnum(msg.get("size")),
            "side": str(msg.get("side") or ""),
            "fee_rate_bps": fnum(msg.get("fee_rate_bps")),
            "transaction_hash": str(msg.get("transaction_hash") or ""),
        }]

    if table == "bba":
        return table, [{
            **base,
            "asset_id": str(msg.get("asset_id") or rec.get("asset_id") or ""),
            "best_bid": fnum(msg.get("best_bid")),
            "best_ask": fnum(msg.get("best_ask")),
            "spread": fnum(msg.get("spread")),
        }]

    # price_change: FLATTEN the price_changes array — one row per entry.
    rows: list[dict[str, Any]] = []
    for change in msg.get("price_changes") or []:
        if not isinstance(change, dict):
            continue
        rows.append({
            **base,
            "asset_id": str(change.get("asset_id") or ""),
            "price": fnum(change.get("price")),
            "side": str(change.get("side") or ""),
            "size": fnum(change.get("size")),
            "best_bid": fnum(change.get("best_bid")),
            "best_ask": fnum(change.get("best_ask")),
        })
    return table, rows


# ---------------------------------------------------------------------------
# Parse one shard file. Returns rows grouped by (universe, table), plus stats.
# ---------------------------------------------------------------------------
def parse_shard(path: Path) -> tuple[dict[tuple[str, str], list[dict]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    bad_lines = 0
    skipped_non_target = 0
    pc_envelopes = 0
    table_envelopes: Counter[str] = Counter()

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                bad_lines += 1
                logger.warning("%s:%d skipping corrupt JSON: %s", path.name, lineno, exc)
                continue
            try:
                result = rows_from_envelope(rec)
            except Exception as exc:  # never let one bad record crash the run
                bad_lines += 1
                logger.warning("%s:%d skipping unparseable envelope: %s", path.name, lineno, exc)
                continue
            if result is None:
                skipped_non_target += 1
                continue
            table, rows = result
            table_envelopes[table] += 1
            if table == "price_change":
                pc_envelopes += 1
            for row in rows:
                groups[(row["universe"], table)].append(row)

    stats = {
        "bad_lines": bad_lines,
        "skipped_non_target": skipped_non_target,
        "pc_envelopes": pc_envelopes,
        "table_envelopes": dict(table_envelopes),
    }
    return groups, stats


# ---------------------------------------------------------------------------
# Build a typed DataFrame for one (table) group.
# ---------------------------------------------------------------------------
def build_frame(table: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    cols = TABLE_COLS[table]
    df = pd.DataFrame(rows, columns=cols)
    # nullable int64 for timestamps/monotonic-ns; float64 for numeric payload cols
    df["timestamp_ms"] = df["timestamp_ms"].astype("Int64")
    df["received_ns"] = df["received_ns"].astype("Int64")
    for col in FLOAT_COLS[table]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def write_parquet(out_path: Path, df_new: pd.DataFrame) -> tuple[int, int]:
    """Append-aware write: concat onto an existing day/universe table if present
    (hourly cron accumulates into the day's file). Returns (new_rows, total_rows)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new
    combined.to_parquet(out_path, engine="pyarrow", index=False)
    return len(df_new), len(combined)


# ---------------------------------------------------------------------------
# Process a set of shard files into Parquet, with validation.
# ---------------------------------------------------------------------------
def process_files(files: list[Path], parquet_root: Path) -> dict[str, Any]:
    # Accumulate all shards of this run, keyed by (date, universe, table).
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    run_stats: Counter[str] = Counter()
    pc_envelopes_total = 0
    source_bytes = 0

    for path in files:
        date = path.parent.name  # data/raw/{date}/{file}
        source_bytes += path.stat().st_size
        groups, stats = parse_shard(path)
        run_stats["bad_lines"] += stats["bad_lines"]
        run_stats["skipped_non_target"] += stats["skipped_non_target"]
        pc_envelopes_total += stats["pc_envelopes"]
        for (universe, table), rows in groups.items():
            grouped[(date, universe, table)].extend(rows)
        logger.info(
            "parsed %s -> %s (bad_lines=%d, skipped_non_target=%d)",
            path.name, stats["table_envelopes"], stats["bad_lines"], stats["skipped_non_target"],
        )

    parquet_bytes = 0
    written: list[dict[str, Any]] = []
    per_universe: dict[str, Counter[str]] = defaultdict(Counter)

    for (date, universe, table), rows in sorted(grouped.items()):
        if not rows:
            continue
        df = build_frame(table, rows)
        expected = len(rows)
        if len(df) != expected:
            raise RuntimeError(f"frame build mismatch {date}/{universe}/{table}: {len(df)} != {expected}")

        out_path = parquet_root / date / universe / f"{table}.parquet"
        new_rows, total_rows = write_parquet(out_path, df)

        # --- VALIDATION: reread, confirm columns + that our rows landed ---
        check = pd.read_parquet(out_path)
        if list(check.columns) != TABLE_COLS[table]:
            raise RuntimeError(f"column mismatch in {out_path}: {list(check.columns)}")
        if new_rows != expected:
            raise RuntimeError(f"row-count mismatch in {out_path}: wrote {new_rows}, expected {expected}")
        if total_rows < new_rows:
            raise RuntimeError(f"reread row-count too low in {out_path}: {total_rows} < {new_rows}")

        size = out_path.stat().st_size
        parquet_bytes += size
        per_universe[universe][table] += new_rows
        written.append({
            "path": str(out_path.relative_to(parquet_root.parent)),
            "table": table, "universe": universe, "rows_added": new_rows,
            "rows_total": total_rows, "bytes": size,
        })
        logger.info(
            "wrote %-13s %-16s +%d rows (file total %d, %.1f KB)",
            table, universe, new_rows, total_rows, size / 1024,
        )

    ratio = (source_bytes / parquet_bytes) if parquet_bytes else 0.0
    return {
        "files": len(files),
        "source_bytes": source_bytes,
        "parquet_bytes": parquet_bytes,
        "compression_ratio_jsonlgz_to_parquet": round(ratio, 2),
        "pc_envelopes_total": pc_envelopes_total,
        "bad_lines": run_stats["bad_lines"],
        "skipped_non_target": run_stats["skipped_non_target"],
        "written": written,
        "per_universe": {u: dict(c) for u, c in per_universe.items()},
    }


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------
def load_processed(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    return {ln.strip() for ln in state_path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def record_processed(state_path: Path, files: list[Path]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a", encoding="utf-8") as fh:
        for path in files:
            fh.write(str(path.resolve()) + "\n")


def discover_shards(raw_root: Path, processed: set[str], force: bool) -> list[Path]:
    now = time.time()
    out: list[Path] = []
    for path in sorted(raw_root.glob("*/*.jsonl.gz")):
        if not force and str(path.resolve()) in processed:
            continue
        if not force and (now - path.stat().st_mtime) < AGE_THRESHOLD_S:
            logger.info("skipping %s — younger than 1h (still being written?)", path.name)
            continue
        out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Polymarket L2 JSONL.gz -> Parquet compression pipeline")
    parser.add_argument("--input", type=Path, default=None, help="process all *.jsonl.gz in this dir")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT)
    parser.add_argument("--force", action="store_true", help="bypass the 1h age check and processed-state skip")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    processed = load_processed(PROCESSED_STATE)
    if args.input:
        files = sorted(args.input.glob("*.jsonl.gz"))
        if not args.force:
            files = [f for f in files if str(f.resolve()) not in processed]
    else:
        files = discover_shards(args.raw_root, processed, args.force)

    if not files:
        logger.info("no shards to process.")
        return 0

    logger.info("processing %d shard(s)", len(files))
    summary = process_files(files, args.parquet_root)
    record_processed(PROCESSED_STATE, files)

    logger.info("=" * 60)
    logger.info("DONE: %d files | %d bad lines | %d non-target skipped",
                summary["files"], summary["bad_lines"], summary["skipped_non_target"])
    logger.info("price_change envelopes: %d -> flattened rows in parquet (expansion from arrays)",
                summary["pc_envelopes_total"])
    logger.info("size: %.1f KB JSONL.gz -> %.1f KB parquet (%.2fx)",
                summary["source_bytes"] / 1024, summary["parquet_bytes"] / 1024,
                summary["compression_ratio_jsonlgz_to_parquet"])
    logger.info("per-universe rows: %s", summary["per_universe"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
