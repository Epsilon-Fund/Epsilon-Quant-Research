"""Bounded incremental Goldsky fetch for Polymarket OrderFilled events.

Default mode is a dry-run freshness check. Add ``--write`` to materialize one
append-only parquet shard from the current local tail to the latest indexed
Goldsky timestamp.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/sync_goldsky_incremental.py
    PYTHONPATH=. uv run python scripts/sync_goldsky_incremental.py --write
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from data_infra.duck import connect


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRADES_DIR = DATA / "trades"
INPROG_DIR = TRADES_DIR / "_inprog"
MARKETS_DIR = DATA / "markets"

GOLDSKY_URL = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
    "/subgraphs/orderbook-subgraph/0.0.1/gn"
)

ORDER_FILLED_FIELDS = (
    "id timestamp maker makerAssetId makerAmountFilled "
    "taker takerAssetId takerAmountFilled fee transactionHash"
)

ENRICHED_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("us")),
        pa.field("market_id", pa.string()),
        pa.field("condition_id", pa.string()),
        pa.field("neg_risk", pa.bool_()),
        pa.field("maker", pa.string()),
        pa.field("taker", pa.string()),
        pa.field("maker_asset_id", pa.string()),
        pa.field("taker_asset_id", pa.string()),
        pa.field("usd_amount", pa.float64()),
        pa.field("token_amount", pa.float64()),
        pa.field("price", pa.float64()),
        pa.field("maker_side", pa.string()),
        pa.field("transaction_hash", pa.string()),
    ]
)


def epoch_to_utc(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def dt_to_epoch(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def utc_stamp(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def latest_markets_parquet() -> Path:
    candidates = sorted(MARKETS_DIR.glob("markets_*.parquet"))
    if not candidates:
        raise SystemExit(f"no markets_*.parquet found in {MARKETS_DIR}")
    return candidates[-1]


def local_tail_epoch() -> int:
    con = connect()
    row = con.sql(
        f"""
        SELECT max(timestamp) AS local_tail
        FROM (
            SELECT timestamp FROM read_parquet('{TRADES_DIR / "trades_seed.parquet"}')
            UNION ALL BY NAME
            SELECT timestamp FROM read_parquet('{TRADES_DIR / "trades_delta_shard*.parquet"}')
        )
        """
    ).fetchone()
    if row is None or row[0] is None:
        raise SystemExit(f"no local trades found in {TRADES_DIR}")
    return dt_to_epoch(row[0])


def load_markets_tokens(
    markets_path: Path,
) -> dict[str, tuple[str | None, str | None, bool | None]]:
    con = connect()
    rows = con.sql(
        f"""
        SELECT
            CAST(id AS VARCHAR) AS market_id,
            condition_id,
            neg_risk,
            UNNEST(clob_token_ids) AS token_id
        FROM read_parquet('{markets_path}')
        WHERE len(clob_token_ids) > 0
        """
    ).fetchall()
    lookup: dict[str, tuple[str | None, str | None, bool | None]] = {}
    for market_id, condition_id, neg_risk, token_id in rows:
        lookup.setdefault(str(token_id), (market_id, condition_id, neg_risk))
    return lookup


def execute_graphql(
    client: httpx.Client,
    query: str,
    *,
    max_attempts: int,
    timeout_s: float,
) -> dict[str, Any]:
    backoff_s = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.post(GOLDSKY_URL, json={"query": query}, timeout=timeout_s)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == max_attempts:
                    response.raise_for_status()
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 120.0)
                continue
            response.raise_for_status()
            payload = response.json()
            if "errors" in payload:
                raise RuntimeError(f"Goldsky GraphQL errors: {payload['errors']}")
            return payload["data"]
        except (httpx.TransportError, httpx.TimeoutException):
            if attempt == max_attempts:
                raise
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, 120.0)
    raise RuntimeError("unreachable: Goldsky retries exhausted")


def latest_goldsky_tail(
    client: httpx.Client,
    *,
    max_attempts: int,
    timeout_s: float,
) -> tuple[int, str]:
    query = (
        "{ orderFilledEvents(first: 1, orderBy: timestamp, orderDirection: desc) "
        "{ id timestamp } }"
    )
    rows = execute_graphql(client, query, max_attempts=max_attempts, timeout_s=timeout_s)[
        "orderFilledEvents"
    ]
    if not rows:
        raise SystemExit("Goldsky returned no orderFilledEvents")
    return int(rows[0]["timestamp"]), rows[0]["id"]


def regular_query(after_ts: int, end_ts: int, page_size: int) -> str:
    return (
        "{ orderFilledEvents("
        f"first: {page_size}, orderBy: timestamp, orderDirection: asc, "
        f'where: {{ timestamp_gt: "{after_ts}", timestamp_lte: "{end_ts}" }}'
        f") {{ {ORDER_FILLED_FIELDS} }} }}"
    )


def exact_timestamp_query(ts: int, after_id: str, page_size: int) -> str:
    return (
        "{ orderFilledEvents("
        f"first: {page_size}, orderBy: id, orderDirection: asc, "
        f'where: {{ timestamp: "{ts}", id_gt: "{after_id}" }}'
        f") {{ {ORDER_FILLED_FIELDS} }} }}"
    )


def fetch_page(
    client: httpx.Client,
    query: str,
    *,
    max_attempts: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    data = execute_graphql(client, query, max_attempts=max_attempts, timeout_s=timeout_s)
    return data["orderFilledEvents"]


def fetch_all_at_timestamp(
    client: httpx.Client,
    ts: int,
    *,
    page_size: int,
    max_attempts: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last_id = ""
    while True:
        page = fetch_page(
            client,
            exact_timestamp_query(ts, last_id, page_size),
            max_attempts=max_attempts,
            timeout_s=timeout_s,
        )
        if not page:
            return out
        out.extend(page)
        last_id = page[-1]["id"]
        if len(page) < page_size:
            return out


def iter_goldsky_increment(
    client: httpx.Client,
    *,
    start_after_ts: int,
    end_ts: int,
    page_size: int,
    max_attempts: int,
    timeout_s: float,
    max_regular_pages: int | None = None,
):
    """Yield raw rows without splitting a timestamp across page boundaries."""
    last_ts = start_after_ts
    regular_pages = 0

    while last_ts < end_ts:
        page = fetch_page(
            client,
            regular_query(last_ts, end_ts, page_size),
            max_attempts=max_attempts,
            timeout_s=timeout_s,
        )
        if not page:
            return

        regular_pages += 1
        page_ts = [int(row["timestamp"]) for row in page]
        boundary_ts = max(page_ts)

        if len(page) < page_size:
            yield page
            return

        safe_rows = [row for row in page if int(row["timestamp"]) < boundary_ts]
        if safe_rows:
            yield safe_rows

        boundary_rows = fetch_all_at_timestamp(
            client,
            boundary_ts,
            page_size=page_size,
            max_attempts=max_attempts,
            timeout_s=timeout_s,
        )
        if boundary_rows:
            yield boundary_rows

        last_ts = boundary_ts
        if max_regular_pages is not None and regular_pages >= max_regular_pages:
            return


def enrich_rows(
    rows: list[dict[str, Any]],
    lookup: dict[str, tuple[str | None, str | None, bool | None]],
) -> tuple[dict[str, list[Any]], int]:
    cols: dict[str, list[Any]] = {name: [] for name in ENRICHED_SCHEMA.names}
    missing_market_rows = 0

    for row in rows:
        maker_asset = str(row["makerAssetId"])
        taker_asset = str(row["takerAssetId"])
        is_buy = maker_asset == "0"
        usd_base = float(row["makerAmountFilled"] if is_buy else row["takerAmountFilled"])
        token_base = float(row["takerAmountFilled"] if is_buy else row["makerAmountFilled"])
        market_token_id = taker_asset if is_buy else maker_asset
        market_id, condition_id, neg_risk = lookup.get(market_token_id, (None, None, None))
        if market_id is None:
            missing_market_rows += 1

        cols["timestamp"].append(
            datetime.fromtimestamp(int(row["timestamp"]), tz=UTC).replace(tzinfo=None)
        )
        cols["market_id"].append(market_id)
        cols["condition_id"].append(condition_id)
        cols["neg_risk"].append(neg_risk)
        cols["maker"].append((row["maker"] or "").lower())
        cols["taker"].append((row["taker"] or "").lower())
        cols["maker_asset_id"].append(maker_asset)
        cols["taker_asset_id"].append(taker_asset)
        cols["usd_amount"].append(usd_base / 1e6)
        cols["token_amount"].append(token_base / 1e6)
        cols["price"].append((usd_base / token_base) if token_base else None)
        cols["maker_side"].append("BUY" if is_buy else "SELL")
        cols["transaction_hash"].append(row["transactionHash"])

    return cols, missing_market_rows


def final_shard_path(start_after_ts: int, end_ts: int) -> Path:
    return (
        TRADES_DIR
        / f"trades_delta_shardinc_{utc_stamp(start_after_ts)}_{utc_stamp(end_ts)}.parquet"
    )


def write_increment(
    *,
    client: httpx.Client,
    start_after_ts: int,
    end_ts: int,
    lookup: dict[str, tuple[str | None, str | None, bool | None]],
    page_size: int,
    max_attempts: int,
    timeout_s: float,
    replace_inprogress: bool,
) -> dict[str, Any]:
    final_path = final_shard_path(start_after_ts, end_ts)
    inprog_path = INPROG_DIR / f"{final_path.name}.inprogress"
    if final_path.exists():
        raise SystemExit(f"refusing to overwrite existing shard: {final_path}")
    if inprog_path.exists():
        if not replace_inprogress:
            raise SystemExit(
                f"in-progress file exists: {inprog_path}; inspect it or rerun with "
                "--replace-inprogress"
            )
        inprog_path.unlink()

    INPROG_DIR.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(inprog_path, ENRICHED_SCHEMA, compression="zstd")
    rows_written = 0
    missing_market_rows = 0
    first_event_ts: int | None = None
    last_event_ts: int | None = None

    try:
        for rows in iter_goldsky_increment(
            client,
            start_after_ts=start_after_ts,
            end_ts=end_ts,
            page_size=page_size,
            max_attempts=max_attempts,
            timeout_s=timeout_s,
        ):
            if not rows:
                continue
            cols, missing = enrich_rows(rows, lookup)
            writer.write_table(pa.Table.from_pydict(cols, schema=ENRICHED_SCHEMA))
            rows_written += len(rows)
            missing_market_rows += missing
            row_ts = [int(row["timestamp"]) for row in rows]
            first_event_ts = min(row_ts) if first_event_ts is None else min(first_event_ts, *row_ts)
            last_event_ts = max(row_ts) if last_event_ts is None else max(last_event_ts, *row_ts)
            if rows_written % 500_000 < len(rows):
                print(
                    f"wrote {rows_written:,} rows through {epoch_to_utc(last_event_ts)}",
                    flush=True,
                )
    finally:
        writer.close()

    if rows_written == 0:
        inprog_path.unlink(missing_ok=True)
        return {
            "rows_written": 0,
            "missing_market_rows": 0,
            "first_event_ts": None,
            "last_event_ts": None,
            "shard_path": None,
        }

    inprog_path.replace(final_path)
    return {
        "rows_written": rows_written,
        "missing_market_rows": missing_market_rows,
        "first_event_ts": first_event_ts,
        "last_event_ts": last_event_ts,
        "shard_path": str(final_path),
    }


def split_intervals(start_after_ts: int, end_ts: int, chunk_seconds: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cur = start_after_ts
    while cur < end_ts:
        nxt = min(cur + chunk_seconds, end_ts)
        out.append((cur, nxt))
        cur = nxt
    return out


def part_path(final_path: Path, idx: int) -> Path:
    return INPROG_DIR / f"{final_path.name}.part{idx:04d}.parquet"


def write_interval_part(
    *,
    idx: int,
    start_after_ts: int,
    end_ts: int,
    part: Path,
    lookup: dict[str, tuple[str | None, str | None, bool | None]],
    page_size: int,
    max_attempts: int,
    timeout_s: float,
) -> dict[str, Any]:
    rows_written = 0
    missing_market_rows = 0
    first_event_ts: int | None = None
    last_event_ts: int | None = None
    writer = pq.ParquetWriter(part, ENRICHED_SCHEMA, compression="zstd")
    try:
        with httpx.Client() as client:
            for rows in iter_goldsky_increment(
                client,
                start_after_ts=start_after_ts,
                end_ts=end_ts,
                page_size=page_size,
                max_attempts=max_attempts,
                timeout_s=timeout_s,
            ):
                if not rows:
                    continue
                cols, missing = enrich_rows(rows, lookup)
                writer.write_table(pa.Table.from_pydict(cols, schema=ENRICHED_SCHEMA))
                rows_written += len(rows)
                missing_market_rows += missing
                row_ts = [int(row["timestamp"]) for row in rows]
                first_event_ts = (
                    min(row_ts) if first_event_ts is None else min(first_event_ts, *row_ts)
                )
                last_event_ts = (
                    max(row_ts) if last_event_ts is None else max(last_event_ts, *row_ts)
                )
    finally:
        writer.close()

    if rows_written == 0:
        part.unlink(missing_ok=True)

    print(
        f"chunk {idx:04d}: rows={rows_written:,} "
        f"{epoch_to_utc(first_event_ts)} -> {epoch_to_utc(last_event_ts)}",
        flush=True,
    )
    return {
        "idx": idx,
        "part": str(part) if rows_written else None,
        "rows_written": rows_written,
        "missing_market_rows": missing_market_rows,
        "first_event_ts": first_event_ts,
        "last_event_ts": last_event_ts,
    }


def write_increment_chunked(
    *,
    start_after_ts: int,
    end_ts: int,
    lookup: dict[str, tuple[str | None, str | None, bool | None]],
    page_size: int,
    max_attempts: int,
    timeout_s: float,
    replace_inprogress: bool,
    workers: int,
    chunk_hours: float,
) -> dict[str, Any]:
    final_path = final_shard_path(start_after_ts, end_ts)
    stale_single = INPROG_DIR / f"{final_path.name}.inprogress"
    stale_parts = sorted(INPROG_DIR.glob(f"{final_path.name}.part*.parquet"))
    if final_path.exists():
        raise SystemExit(f"refusing to overwrite existing shard: {final_path}")
    if stale_single.exists() or stale_parts:
        if not replace_inprogress:
            raise SystemExit(
                f"in-progress files exist for {final_path.name}; inspect them or rerun "
                "with --replace-inprogress"
            )
        stale_single.unlink(missing_ok=True)
        for part in stale_parts:
            part.unlink()

    INPROG_DIR.mkdir(parents=True, exist_ok=True)
    chunk_seconds = max(1, int(chunk_hours * 3600))
    intervals = split_intervals(start_after_ts, end_ts, chunk_seconds)
    print(
        f"writing {len(intervals)} chunks with {workers} workers "
        f"from {epoch_to_utc(start_after_ts)} exclusive to {epoch_to_utc(end_ts)} inclusive",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                write_interval_part,
                idx=idx,
                start_after_ts=start,
                end_ts=end,
                part=part_path(final_path, idx),
                lookup=lookup,
                page_size=page_size,
                max_attempts=max_attempts,
                timeout_s=timeout_s,
            )
            for idx, (start, end) in enumerate(intervals)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda row: row["idx"])
    parts = [Path(row["part"]) for row in results if row["part"] is not None]
    rows_written = sum(int(row["rows_written"]) for row in results)
    missing_market_rows = sum(int(row["missing_market_rows"]) for row in results)
    first_values = [int(row["first_event_ts"]) for row in results if row["first_event_ts"] is not None]
    last_values = [int(row["last_event_ts"]) for row in results if row["last_event_ts"] is not None]

    if rows_written == 0:
        return {
            "rows_written": 0,
            "missing_market_rows": 0,
            "first_event_ts": None,
            "last_event_ts": None,
            "shard_path": None,
        }

    paths_sql = ", ".join(f"'{part}'" for part in parts)
    con = connect()
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet([{paths_sql}])
        )
        TO '{final_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    for part in parts:
        part.unlink()

    return {
        "rows_written": rows_written,
        "missing_market_rows": missing_market_rows,
        "first_event_ts": min(first_values),
        "last_event_ts": max(last_values),
        "shard_path": str(final_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write a new append-only parquet shard; default is dry-run",
    )
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel chunk workers used with --write",
    )
    parser.add_argument(
        "--chunk-hours",
        type=float,
        default=2.0,
        help="time span per parallel Goldsky chunk used with --write",
    )
    parser.add_argument(
        "--max-regular-pages",
        type=int,
        default=None,
        help="dry-run smoke limit; rejected with --write",
    )
    parser.add_argument(
        "--replace-inprogress",
        action="store_true",
        help="replace this script's stale in-progress parquet temp file",
    )
    parser.add_argument(
        "--markets-path",
        type=Path,
        default=None,
        help="market metadata parquet for token enrichment; defaults to latest data/markets/markets_*.parquet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write and args.max_regular_pages is not None:
        raise SystemExit("--max-regular-pages is for dry-run smoke checks only")

    local_tail = local_tail_epoch()
    with httpx.Client() as client:
        goldsky_tail, goldsky_tail_id = latest_goldsky_tail(
            client,
            max_attempts=args.max_attempts,
            timeout_s=args.timeout_s,
        )

        summary: dict[str, Any] = {
            "local_tail_utc": epoch_to_utc(local_tail),
            "local_tail_epoch": local_tail,
            "goldsky_tail_utc": epoch_to_utc(goldsky_tail),
            "goldsky_tail_epoch": goldsky_tail,
            "goldsky_tail_id": goldsky_tail_id,
            "new_shard_written": False,
            "shard_path": None,
            "rows_written": 0,
            "missing_market_rows": 0,
        }

        if goldsky_tail <= local_tail:
            summary["status"] = "up_to_date"
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        summary["planned_start_exclusive_utc"] = epoch_to_utc(local_tail)
        summary["planned_end_inclusive_utc"] = epoch_to_utc(goldsky_tail)

        if not args.write:
            probe_rows = 0
            probe_last_ts: int | None = None
            for rows in iter_goldsky_increment(
                client,
                start_after_ts=local_tail,
                end_ts=goldsky_tail,
                page_size=args.page_size,
                max_attempts=args.max_attempts,
                timeout_s=args.timeout_s,
                max_regular_pages=args.max_regular_pages or 1,
            ):
                probe_rows += len(rows)
                if rows:
                    probe_last_ts = max(int(row["timestamp"]) for row in rows)
            summary["status"] = "dry_run"
            summary["probe_rows_seen"] = probe_rows
            summary["probe_last_ts_utc"] = epoch_to_utc(probe_last_ts)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        markets_path = args.markets_path or latest_markets_parquet()
        summary["markets_path"] = str(markets_path)
        lookup = load_markets_tokens(markets_path)
        summary["market_token_mappings"] = len(lookup)
        write_result = write_increment_chunked(
            start_after_ts=local_tail,
            end_ts=goldsky_tail,
            lookup=lookup,
            page_size=args.page_size,
            max_attempts=args.max_attempts,
            timeout_s=args.timeout_s,
            replace_inprogress=args.replace_inprogress,
            workers=args.workers,
            chunk_hours=args.chunk_hours,
        )
        summary.update(write_result)
        summary["new_shard_written"] = write_result["shard_path"] is not None
        summary["status"] = "written" if summary["new_shard_written"] else "no_new_rows"
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
