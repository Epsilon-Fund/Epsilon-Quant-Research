# Goldsky subgraph backfill for Polymarket OrderFilled events.
#
# Endpoint: https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw
#           /subgraphs/orderbook-subgraph/0.0.1/gn
#
# GraphQL query (composite (timestamp, id) cursor; "sticky" mode within a single timestamp):
#   regular: { orderFilledEvents(first: N, orderBy: timestamp, orderDirection: asc,
#              where: { timestamp_gt: "<last_ts>", timestamp_lte: "<end_ts>" }) {
#              id timestamp maker makerAssetId makerAmountFilled
#              taker takerAssetId takerAmountFilled fee transactionHash } }
#   sticky:  { orderFilledEvents(first: N, orderBy: id, orderDirection: asc,
#              where: { timestamp: "<sticky_ts>", id_gt: "<sticky_last_id>" }) { ... } }
#
# Notes:
# - "0" in makerAssetId or takerAssetId means USDC; the non-zero side is an outcome token.
# - Subgraph indexer lags real-time by ~9 days as of 2026-05-07; ranges past that return empty.

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from data_infra.duck import connect

ROOT = Path(__file__).resolve().parents[1]
TRADES_DIR = ROOT / "data" / "trades"
INPROG_DIR = TRADES_DIR / "_inprog"
MARKETS_DIR = ROOT / "data" / "markets"

GOLDSKY_URL = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
    "/subgraphs/orderbook-subgraph/0.0.1/gn"
)

PAGE_SIZE = 1000
PART_ROWS = 500_000
PROGRESS_INTERVAL = 60.0
HTTP_TIMEOUT = 60.0
MAX_BACKOFF_S = 120.0
MAX_ATTEMPTS_PER_PAGE = 12
MAX_CONCURRENT_FETCHES = 15  # Goldsky per-IP limit; >10 collapses per-worker rate

_fetch_sem: asyncio.Semaphore | None = None

SHARDS: list[tuple[str, int, int]] = [
    ("gap_a", 1759855190, 1760195911),  # 2025-10-07 16:39:50 UTC → 2025-10-11 14:38:31 UTC (seed→delta gap)
    ("gap_b", 1760195911, 1760536633),  # 2025-10-11 14:38:31 UTC → 2025-10-15 13:57:13 UTC (seed→delta gap)
    ("1",  1760536633, 1763053571),
    ("2",  1763053571, 1765570509),
    ("3",  1765570509, 1768087447),
    ("4",  1768087447, 1768176000),  # SHRUNK 2026-01-12 00:00 UTC; new sub-shards take over
    ("4b", 1768176000, 1768953600),
    ("4c", 1768953600, 1769731200),
    ("4d", 1769731200, 1770604385),
    ("5",  1770604385, 1770681600),  # SHRUNK 2026-02-10 00:00 UTC
    ("5b", 1770681600, 1771459200),
    ("5c", 1771459200, 1772236800),
    ("5d",   1772236800, 1772867920),  # SHRUNK 2026-05-08 mid-flight split
    ("5d_2", 1772867920, 1773121323),
    ("6",  1773121323, 1773187200),  # SHRUNK 2026-03-11 00:00 UTC
    ("6b", 1773187200, 1773532800),
    ("6c", 1773532800, 1773878400),
    ("6d", 1773878400, 1774224000),
    ("6e", 1774224000, 1774569600),
    ("6f", 1774569600, 1774915200),
    ("6g", 1774915200, 1775260800),
    ("6h",   1775260800, 1775515860),  # SHRUNK 2026-05-08 mid-flight split
    ("6h_2", 1775515860, 1775638261),
    ("7",  1775638261, 1775692800),  # SHRUNK 2026-04-09 00:00 UTC
    ("7b",   1775692800, 1775950726),  # SHRUNK 2026-05-08 mid-flight split
    ("7b_2", 1775950726, 1776124800),
    ("7c",   1776124800, 1776357756),  # SHRUNK 2026-05-08 mid-flight split
    ("7c_2", 1776357756, 1776556800),
    ("7d",   1776556800, 1776774745),  # SHRUNK 2026-05-08 mid-flight split
    ("7d_2", 1776774745, 1776988800),
    # 7e and 7f dropped: Goldsky subgraph index lags ~9 days; ranges past ~2026-04-28 return empty.
    # Re-add when subgraph catches up.
]

ENRICHED_SCHEMA = pa.schema([
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
])

ORDER_FILLED_FIELDS = (
    "id timestamp maker makerAssetId makerAmountFilled "
    "taker takerAssetId takerAmountFilled fee transactionHash"
)

progress: dict[str, dict] = {}


def date_str(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")


def fmt_ts(ts: int | None) -> str:
    if ts is None or ts == 0:
        return "—"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def cursor_path(shard_idx: str) -> Path:
    return INPROG_DIR / f"shard{shard_idx}_cursor.json"


def part_path(shard_idx: str, part: int) -> Path:
    return INPROG_DIR / f"shard{shard_idx}_part{part:05d}.parquet"


def shard_existing_parts(shard_idx: str) -> list[Path]:
    return sorted(INPROG_DIR.glob(f"shard{shard_idx}_part*.parquet"))


def final_shard_path(shard_idx: str, start_ts: int, end_ts: int) -> Path:
    return TRADES_DIR / f"trades_delta_shard{shard_idx}_{date_str(start_ts)}_{date_str(end_ts)}.parquet"


def load_cursor(shard_idx: str) -> dict | None:
    p = cursor_path(shard_idx)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save_cursor(shard_idx: str, state: dict) -> None:
    p = cursor_path(shard_idx)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(p)


def cleanup_bad_parts(shard_idx: str) -> int:
    removed = 0
    for p in shard_existing_parts(shard_idx):
        try:
            pq.ParquetFile(p).num_row_groups
        except Exception:
            p.unlink()
            removed += 1
    return removed


def next_available_part_idx(shard_idx: str) -> int:
    parts = shard_existing_parts(shard_idx)
    if not parts:
        return 0
    nums = [int(p.stem.split("part")[1]) for p in parts]
    return max(nums) + 1


def latest_markets_parquet() -> Path:
    cands = sorted(MARKETS_DIR.glob("markets_*.parquet"))
    if not cands:
        raise SystemExit("no markets_*.parquet")
    return cands[-1]


def load_markets_tokens(con) -> dict[str, tuple[str | None, str | None, bool | None]]:
    path = latest_markets_parquet()
    print(f"loading markets-tokens dict from {path.name}...")
    rows = con.sql(f"""
        SELECT
            CAST(id AS VARCHAR) AS market_id,
            condition_id,
            neg_risk,
            UNNEST(clob_token_ids) AS token_id
        FROM read_parquet('{path}')
        WHERE len(clob_token_ids) > 0
    """).fetchall()
    d: dict[str, tuple[str | None, str | None, bool | None]] = {}
    for market_id, condition_id, neg_risk, token_id in rows:
        if token_id not in d:
            d[token_id] = (market_id, condition_id, neg_risk)
    print(f"  loaded {len(d):,} unique token_id → market mappings")
    return d


def enrich_rows(rows: list[dict], lookup: dict) -> dict[str, list]:
    timestamps: list = []
    market_ids: list = []
    condition_ids: list = []
    neg_risks: list = []
    makers: list = []
    takers: list = []
    maker_asset_ids: list = []
    taker_asset_ids: list = []
    usd_amounts: list = []
    token_amounts: list = []
    prices: list = []
    maker_sides: list = []
    tx_hashes: list = []
    for r in rows:
        maker_asset = r["makerAssetId"]
        taker_asset = r["takerAssetId"]
        is_buy = maker_asset == "0"
        usd_base = float(r["makerAmountFilled"] if is_buy else r["takerAmountFilled"])
        token_base = float(r["takerAmountFilled"] if is_buy else r["makerAmountFilled"])
        market_token_id = taker_asset if is_buy else maker_asset
        m = lookup.get(market_token_id)
        if m is None:
            mi, ci, nr = None, None, None
        else:
            mi, ci, nr = m
        timestamps.append(
            datetime.fromtimestamp(int(r["timestamp"]), tz=timezone.utc).replace(tzinfo=None)
        )
        market_ids.append(mi)
        condition_ids.append(ci)
        neg_risks.append(nr)
        makers.append((r["maker"] or "").lower())
        takers.append((r["taker"] or "").lower())
        maker_asset_ids.append(maker_asset)
        taker_asset_ids.append(taker_asset)
        usd_amounts.append(usd_base / 1e6)
        token_amounts.append(token_base / 1e6)
        prices.append((usd_base / token_base) if token_base != 0 else None)
        maker_sides.append("BUY" if is_buy else "SELL")
        tx_hashes.append(r["transactionHash"])
    return {
        "timestamp": timestamps,
        "market_id": market_ids,
        "condition_id": condition_ids,
        "neg_risk": neg_risks,
        "maker": makers,
        "taker": takers,
        "maker_asset_id": maker_asset_ids,
        "taker_asset_id": taker_asset_ids,
        "usd_amount": usd_amounts,
        "token_amount": token_amounts,
        "price": prices,
        "maker_side": maker_sides,
        "transaction_hash": tx_hashes,
    }


def build_query(last_ts: int, sticky_ts: int | None, sticky_last_id: str, end_ts: int) -> str:
    if sticky_ts is None:
        return (
            "{ orderFilledEvents("
            f"first: {PAGE_SIZE}, orderBy: timestamp, orderDirection: asc, "
            f'where: {{ timestamp_gt: "{last_ts}", timestamp_lte: "{end_ts}" }}'
            f") {{ {ORDER_FILLED_FIELDS} }} }}"
        )
    return (
        "{ orderFilledEvents("
        f"first: {PAGE_SIZE}, orderBy: id, orderDirection: asc, "
        f'where: {{ timestamp: "{sticky_ts}", id_gt: "{sticky_last_id}" }}'
        f") {{ {ORDER_FILLED_FIELDS} }} }}"
    )


async def fetch_page_async(client: httpx.AsyncClient, query: str, shard_idx: str) -> list[dict]:
    backoff = 2.0
    for attempt in range(1, MAX_ATTEMPTS_PER_PAGE + 1):
        try:
            assert _fetch_sem is not None
            async with _fetch_sem:
                r = await client.post(GOLDSKY_URL, json={"query": query}, timeout=HTTP_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                progress[shard_idx]["errors"] += 1
                if attempt == MAX_ATTEMPTS_PER_PAGE:
                    r.raise_for_status()
                await asyncio.sleep(min(backoff, MAX_BACKOFF_S))
                backoff = min(backoff * 2, MAX_BACKOFF_S)
                continue
            r.raise_for_status()
            data = r.json()
            if "errors" in data:
                raise RuntimeError(f"GraphQL errors (shard {shard_idx}): {data['errors']}")
            return data["data"]["orderFilledEvents"]
        except httpx.TransportError:
            progress[shard_idx]["errors"] += 1
            if attempt == MAX_ATTEMPTS_PER_PAGE:
                raise
            await asyncio.sleep(min(backoff, MAX_BACKOFF_S))
            backoff = min(backoff * 2, MAX_BACKOFF_S)
    raise RuntimeError(f"unreachable: shard {shard_idx} exhausted retries")


async def run_shard(shard_idx: str, start_ts: int, end_ts: int, lookup: dict) -> None:
    progress[shard_idx] = {
        "rows": 0,
        "last_ts": start_ts,
        "errors": 0,
        "started": time.time(),
        "finished": False,
    }
    final = final_shard_path(shard_idx, start_ts, end_ts)
    if final.exists():
        progress[shard_idx]["finished"] = True
        progress[shard_idx]["last_ts"] = end_ts
        return

    removed = cleanup_bad_parts(shard_idx)
    if removed:
        print(f"  shard{shard_idx}: removed {removed} corrupt part(s)", flush=True)

    state = load_cursor(shard_idx) or {}
    last_ts = int(state.get("last_ts", start_ts))
    sticky_ts_raw = state.get("sticky_ts")
    sticky_ts: int | None = int(sticky_ts_raw) if sticky_ts_raw is not None else None
    sticky_last_id = state.get("sticky_last_id", "")
    rows_total = 0
    if shard_existing_parts(shard_idx):
        con_local = connect()
        paths = ", ".join(f"'{p}'" for p in shard_existing_parts(shard_idx))
        rows_total = con_local.sql(f"SELECT count(*) FROM read_parquet([{paths}])").fetchone()[0]
    progress[shard_idx]["rows"] = rows_total
    progress[shard_idx]["last_ts"] = sticky_ts or last_ts

    cur_part_idx = next_available_part_idx(shard_idx)
    cur_writer = pq.ParquetWriter(part_path(shard_idx, cur_part_idx), ENRICHED_SCHEMA, compression="zstd")
    rows_in_cur_part = 0

    async with httpx.AsyncClient() as client:
        try:
            while True:
                query = build_query(last_ts, sticky_ts, sticky_last_id, end_ts)
                page = await fetch_page_async(client, query, shard_idx)
                if not page:
                    if sticky_ts is not None:
                        last_ts = sticky_ts
                        sticky_ts = None
                        sticky_last_id = ""
                        save_cursor(shard_idx, {
                            "last_ts": last_ts,
                            "sticky_ts": None,
                            "sticky_last_id": "",
                        })
                        continue
                    break

                cols = enrich_rows(page, lookup)
                cur_writer.write_table(pa.Table.from_pydict(cols, schema=ENRICHED_SCHEMA))
                rows_in_cur_part += len(page)
                rows_total += len(page)
                progress[shard_idx]["rows"] = rows_total

                page_ts = [int(r["timestamp"]) for r in page]
                progress[shard_idx]["last_ts"] = sticky_ts if sticky_ts is not None else page_ts[-1]

                if sticky_ts is not None:
                    sticky_last_id = page[-1]["id"]
                    if len(page) < PAGE_SIZE:
                        last_ts = sticky_ts
                        sticky_ts = None
                        sticky_last_id = ""
                else:
                    if len(page) == PAGE_SIZE and page_ts[0] == page_ts[-1]:
                        sticky_ts = page_ts[0]
                        sticky_last_id = page[-1]["id"]
                    else:
                        last_ts = page_ts[-1]

                save_cursor(shard_idx, {
                    "last_ts": last_ts,
                    "sticky_ts": sticky_ts,
                    "sticky_last_id": sticky_last_id,
                })

                if rows_in_cur_part >= PART_ROWS:
                    cur_writer.close()
                    cur_part_idx += 1
                    cur_writer = pq.ParquetWriter(
                        part_path(shard_idx, cur_part_idx), ENRICHED_SCHEMA, compression="zstd"
                    )
                    rows_in_cur_part = 0
        finally:
            try:
                cur_writer.close()
            except Exception:
                pass

    parts = shard_existing_parts(shard_idx)
    parts = [p for p in parts if p.stat().st_size > 0]
    if parts:
        con_local = connect()
        paths = ", ".join(f"'{p}'" for p in parts)
        con_local.execute(f"""
            COPY (SELECT * FROM read_parquet([{paths}]))
            TO '{final}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        for p in parts:
            p.unlink()
    if cursor_path(shard_idx).exists():
        cursor_path(shard_idx).unlink()
    progress[shard_idx]["finished"] = True
    progress[shard_idx]["last_ts"] = end_ts


async def progress_reporter() -> None:
    try:
        while True:
            await asyncio.sleep(PROGRESS_INTERVAL)
            now = time.time()
            lines = [f"--- {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} ---"]
            for idx, start_ts, end_ts in SHARDS:
                p = progress.get(idx)
                if p is None:
                    lines.append(f"  shard{idx}: pending")
                    continue
                elapsed = max(now - p["started"], 1)
                rate = p["rows"] / elapsed
                span = end_ts - start_ts
                covered = max(0, p["last_ts"] - start_ts)
                pct = 100 * covered / span if span > 0 else 0
                if p["finished"]:
                    eta_str = "DONE"
                elif covered > 0:
                    eta_h = (elapsed * (span - covered) / covered) / 3600
                    eta_str = f"ETA={eta_h:5.1f}h"
                else:
                    eta_str = "ETA=  ?"
                lines.append(
                    f"  shard{idx}: {pct:5.1f}%  rows={p['rows']:>10,}  "
                    f"latest={fmt_ts(p['last_ts'])}  rate={rate:>4.0f}r/s  "
                    f"err={p['errors']:>2}  {eta_str}"
                )
            print("\n".join(lines), flush=True)
            if all(progress.get(i, {}).get("finished") for i, _, _ in SHARDS):
                return
    except asyncio.CancelledError:
        return


async def shard_worker(queue: "asyncio.Queue[tuple[str, int, int]]", lookup: dict) -> None:
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        shard_idx, start_ts, end_ts = item
        try:
            await run_shard(shard_idx, start_ts, end_ts, lookup)
        finally:
            queue.task_done()


async def main_async() -> None:
    global _fetch_sem
    _fetch_sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
    INPROG_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()
    lookup = load_markets_tokens(con)

    print(f"\n{'=' * 70}\nshard plan\n{'=' * 70}")
    for idx, s, e in SHARDS:
        print(f"  shard{idx}: {fmt_ts(s)} → {fmt_ts(e)}  ({(e - s) / 86400:.1f} days)")

    queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
    for shard in SHARDS:
        queue.put_nowait(shard)

    pool_size = MAX_CONCURRENT_FETCHES
    print(f"\nlaunching {pool_size} workers consuming queue of {queue.qsize()} shards...")
    workers = [asyncio.create_task(shard_worker(queue, lookup)) for _ in range(pool_size)]
    progress_task = asyncio.create_task(progress_reporter())
    try:
        await asyncio.gather(*workers)
    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    print(f"\n{'=' * 70}\nall shards complete\n{'=' * 70}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
