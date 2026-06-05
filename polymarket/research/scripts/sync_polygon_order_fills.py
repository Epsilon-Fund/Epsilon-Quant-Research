"""Backfill Polymarket OrderFilled events directly from Polygon logs.

This is the post-Goldsky freshness path. It decodes both the legacy exchange
event and the CTF Exchange V2 event into the existing enriched trade schema.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/sync_polygon_order_fills.py --write

Use ``POLYGON_RPC_URL`` for a private endpoint. Without it, the script falls
back to a public Polygon RPC and should be run with modest concurrency.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
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
TOKEN_CACHE = MARKETS_DIR / "gamma_token_lookup_cache.jsonl"

DEFAULT_RPC_URL = "https://polygon.drpc.org"

V1_ORDER_FILLED_TOPIC = (
    "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
)
V2_ORDER_FILLED_TOPIC = (
    "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
)

V1_CONTRACTS = [
    "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "0xC5d563A36AE78145C45a50134d48A1215220f80a",
]
V2_CONTRACTS = [
    "0xE111180000d2663C0091e4f400237545B87B996B",
    "0xe2222d279d744050d28e00520010520000310F59",
]
NEG_RISK_CONTRACTS = {addr.lower() for addr in (V1_CONTRACTS[1], V2_CONTRACTS[1])}

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


def epoch_to_dt(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


def dt_to_epoch(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def fmt_ts(ts: int | None) -> str | None:
    if ts is None:
        return None
    return epoch_to_dt(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def stamp(ts: int) -> str:
    return epoch_to_dt(ts).strftime("%Y%m%dT%H%M%SZ")


def rpc_call(
    client: httpx.Client,
    url: str,
    method: str,
    params: list[Any],
    *,
    timeout_s: float,
    max_attempts: int,
) -> Any:
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                timeout=timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                message = str(payload["error"])
                if attempt == max_attempts:
                    raise RuntimeError(message)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            return payload["result"]
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
            if attempt == max_attempts:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    raise RuntimeError(f"RPC retries exhausted for {method}")


class PolygonRpc:
    def __init__(self, url: str, timeout_s: float, max_attempts: int) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.client = httpx.Client()
        self._block_ts: dict[int, int] = {}

    def close(self) -> None:
        self.client.close()

    def block_number(self) -> int:
        return int(
            rpc_call(
                self.client,
                self.url,
                "eth_blockNumber",
                [],
                timeout_s=self.timeout_s,
                max_attempts=self.max_attempts,
            ),
            16,
        )

    def block_timestamp(self, block: int) -> int:
        if block not in self._block_ts:
            result = rpc_call(
                self.client,
                self.url,
                "eth_getBlockByNumber",
                [hex(block), False],
                timeout_s=self.timeout_s,
                max_attempts=self.max_attempts,
            )
            self._block_ts[block] = int(result["timestamp"], 16)
        return self._block_ts[block]

    def first_block_at_or_after(self, ts: int, hi: int) -> int:
        lo = 0
        while lo < hi:
            mid = (lo + hi) // 2
            if self.block_timestamp(mid) < ts:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def logs(self, addresses: list[str], topic: str, start_block: int, end_block: int) -> list[dict]:
        return rpc_call(
            self.client,
            self.url,
            "eth_getLogs",
            [
                {
                    "address": addresses,
                    "topics": [topic],
                    "fromBlock": hex(start_block),
                    "toBlock": hex(end_block),
                }
            ],
            timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
        )


def latest_markets_parquet() -> Path | None:
    candidates = sorted(MARKETS_DIR.glob("markets_*.parquet"))
    return candidates[-1] if candidates else None


def parse_list(v: Any) -> list[Any]:
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v:
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def load_market_lookup() -> dict[str, tuple[str | None, str | None, bool | None]]:
    lookup: dict[str, tuple[str | None, str | None, bool | None]] = {}
    markets_path = latest_markets_parquet()
    if markets_path is not None:
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
        for market_id, condition_id, neg_risk, token_id in rows:
            lookup[str(token_id)] = (market_id, condition_id, neg_risk)

    if TOKEN_CACHE.exists():
        for line in TOKEN_CACHE.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            lookup[str(row["token_id"])] = (
                row.get("market_id"),
                row.get("condition_id"),
                row.get("neg_risk"),
            )
    return lookup


def cache_market_tokens(
    market: dict[str, Any],
    lookup: dict[str, tuple[str | None, str | None, bool | None]],
) -> list[dict[str, Any]]:
    market_id = str(market.get("id")) if market.get("id") is not None else None
    condition_id = market.get("conditionId")
    neg_risk = market.get("negRisk")
    cached: list[dict[str, Any]] = []
    for token_id in parse_list(market.get("clobTokenIds")):
        token = str(token_id)
        value = (market_id, condition_id, neg_risk)
        if lookup.get(token) != value:
            lookup[token] = value
            cached.append(
                {
                    "token_id": token,
                    "market_id": market_id,
                    "condition_id": condition_id,
                    "neg_risk": neg_risk,
                }
            )
    return cached


def fetch_gamma_token_mappings(
    tokens: list[str],
    lookup: dict[str, tuple[str | None, str | None, bool | None]],
    *,
    batch_size: int,
    timeout_s: float,
    max_attempts: int = 5,
    closed_first: bool = False,
) -> None:
    missing = [token for token in dict.fromkeys(tokens) if token not in lookup]
    if not missing:
        return

    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    query_modes: list[list[tuple[str, str]]] = (
        [[("closed", "true")], []]
        if closed_first
        else [[], [("closed", "true")]]
    )
    with httpx.Client(base_url="https://gamma-api.polymarket.com", timeout=timeout_s) as client:
        with TOKEN_CACHE.open("a") as cache:
            for i in range(0, len(missing), batch_size):
                batch = missing[i : i + batch_size]
                unresolved = {token for token in batch if token not in lookup}
                for extra_params in query_modes:
                    if not unresolved:
                        break
                    query_tokens = [token for token in batch if token in unresolved]
                    params = [("clob_token_ids", token) for token in query_tokens]
                    params.extend(extra_params)
                    markets: list[dict[str, Any]] = []
                    backoff = 1.0
                    for attempt in range(1, max_attempts + 1):
                        try:
                            response = client.get("/markets", params=params)
                            response.raise_for_status()
                            markets = response.json()
                            break
                        except (
                            httpx.TimeoutException,
                            httpx.TransportError,
                            httpx.HTTPStatusError,
                        ) as exc:
                            if attempt == max_attempts:
                                detail = exc.__class__.__name__
                                if isinstance(exc, httpx.HTTPStatusError):
                                    detail = f"HTTP {exc.response.status_code}"
                                print(
                                    f"warning: Gamma token lookup failed for "
                                    f"{len(query_tokens)} tokens after "
                                    f"{max_attempts} attempts: {detail}",
                                    flush=True,
                                )
                                markets = []
                                break
                            time.sleep(backoff)
                            backoff = min(backoff * 2, 30.0)
                    for market in markets:
                        for row in cache_market_tokens(market, lookup):
                            cache.write(json.dumps(row, sort_keys=True) + "\n")
                            unresolved.discard(row["token_id"])
                for token in unresolved:
                    if token not in lookup:
                        lookup[token] = (None, None, None)


def address_from_topic(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def words(data: str) -> list[int]:
    raw = data[2:] if data.startswith("0x") else data
    return [int(raw[i : i + 64], 16) for i in range(0, len(raw), 64)]


def decode_v1(log: dict[str, Any]) -> dict[str, Any]:
    w = words(log["data"])
    return {
        "timestamp": int(log.get("blockTimestamp", "0x0"), 16),
        "maker": address_from_topic(log["topics"][2]),
        "taker": address_from_topic(log["topics"][3]),
        "maker_asset_id": str(w[0]),
        "taker_asset_id": str(w[1]),
        "maker_amount_filled": w[2],
        "taker_amount_filled": w[3],
        "transaction_hash": log["transactionHash"],
        "contract": log["address"].lower(),
    }


def decode_v2(log: dict[str, Any]) -> dict[str, Any]:
    w = words(log["data"])
    side = w[0]
    token_id = str(w[1])
    maker_amount = w[2]
    taker_amount = w[3]
    maker_asset_id = "0" if side == 0 else token_id
    taker_asset_id = token_id if side == 0 else "0"
    return {
        "timestamp": int(log.get("blockTimestamp", "0x0"), 16),
        "maker": address_from_topic(log["topics"][2]),
        "taker": address_from_topic(log["topics"][3]),
        "maker_asset_id": maker_asset_id,
        "taker_asset_id": taker_asset_id,
        "maker_amount_filled": maker_amount,
        "taker_amount_filled": taker_amount,
        "transaction_hash": log["transactionHash"],
        "contract": log["address"].lower(),
    }


def market_token_id(row: dict[str, Any]) -> str:
    return row["taker_asset_id"] if row["maker_asset_id"] == "0" else row["maker_asset_id"]


def rows_to_table(
    rows: list[dict[str, Any]],
    lookup: dict[str, tuple[str | None, str | None, bool | None]],
    *,
    start_after_ts: int,
    end_ts: int,
) -> tuple[pa.Table | None, int, int | None, int | None]:
    cols: dict[str, list[Any]] = {name: [] for name in ENRICHED_SCHEMA.names}
    null_market_rows = 0
    min_ts: int | None = None
    max_ts: int | None = None

    for row in rows:
        ts = int(row["timestamp"])
        if ts <= start_after_ts or ts > end_ts:
            continue

        maker_asset = row["maker_asset_id"]
        taker_asset = row["taker_asset_id"]
        is_buy = maker_asset == "0"
        usd_base = row["maker_amount_filled"] if is_buy else row["taker_amount_filled"]
        token_base = row["taker_amount_filled"] if is_buy else row["maker_amount_filled"]
        token_id = taker_asset if is_buy else maker_asset
        market_id, condition_id, neg_risk = lookup.get(token_id, (None, None, None))
        if neg_risk is None:
            neg_risk = row["contract"] in NEG_RISK_CONTRACTS
        if market_id is None:
            null_market_rows += 1

        min_ts = ts if min_ts is None else min(min_ts, ts)
        max_ts = ts if max_ts is None else max(max_ts, ts)
        cols["timestamp"].append(epoch_to_dt(ts).replace(tzinfo=None))
        cols["market_id"].append(market_id)
        cols["condition_id"].append(condition_id)
        cols["neg_risk"].append(bool(neg_risk) if neg_risk is not None else None)
        cols["maker"].append(row["maker"])
        cols["taker"].append(row["taker"])
        cols["maker_asset_id"].append(maker_asset)
        cols["taker_asset_id"].append(taker_asset)
        cols["usd_amount"].append(usd_base / 1e6)
        cols["token_amount"].append(token_base / 1e6)
        cols["price"].append((usd_base / token_base) if token_base else None)
        cols["maker_side"].append("BUY" if is_buy else "SELL")
        cols["transaction_hash"].append(row["transaction_hash"])

    if not cols["timestamp"]:
        return None, 0, None, None
    return pa.Table.from_pydict(cols, schema=ENRICHED_SCHEMA), null_market_rows, min_ts, max_ts


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
        raise SystemExit("no local trades found")
    return dt_to_epoch(row[0])


def fetch_log_chunk(
    rpc_url: str,
    start_block: int,
    end_block: int,
    timeout_s: float,
    max_attempts: int,
) -> list[dict[str, Any]]:
    rpc = PolygonRpc(rpc_url, timeout_s, max_attempts)
    try:
        rows = [decode_v1(log) for log in rpc.logs(V1_CONTRACTS, V1_ORDER_FILLED_TOPIC, start_block, end_block)]
        rows.extend(
            decode_v2(log)
            for log in rpc.logs(V2_CONTRACTS, V2_ORDER_FILLED_TOPIC, start_block, end_block)
        )
        return rows
    finally:
        rpc.close()


def shard_path(start_ts: int, end_ts: int) -> Path:
    return TRADES_DIR / f"trades_delta_shardpolygon_{stamp(start_ts)}_{stamp(end_ts)}.parquet"


def write_time_shard(
    *,
    rpc_url: str,
    start_after_ts: int,
    end_ts: int,
    start_block: int,
    end_block: int,
    lookup: dict[str, tuple[str | None, str | None, bool | None]],
    block_chunk: int,
    workers: int,
    timeout_s: float,
    max_attempts: int,
    gamma_batch_size: int,
    write: bool,
    replace_inprogress: bool,
) -> dict[str, Any]:
    out = shard_path(start_after_ts, end_ts)
    tmp = INPROG_DIR / f"{out.name}.inprogress"
    if out.exists():
        return {"shard_path": str(out), "status": "exists"}
    if tmp.exists():
        if not replace_inprogress:
            raise SystemExit(f"in-progress file exists: {tmp}")
        tmp.unlink()

    ranges = [
        (block, min(block + block_chunk - 1, end_block))
        for block in range(start_block, end_block + 1, block_chunk)
    ]
    summary = {
        "shard_path": str(out),
        "status": "dry_run" if not write else "written",
        "rows": 0,
        "null_market_rows": 0,
        "min_ts": None,
        "max_ts": None,
        "start_block": start_block,
        "end_block": end_block,
    }
    if not write:
        ranges = ranges[: min(3, len(ranges))]

    writer: pq.ParquetWriter | None = None
    if write:
        INPROG_DIR.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(tmp, ENRICHED_SCHEMA, compression="zstd")

    try:
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(fetch_log_chunk, rpc_url, a, b, timeout_s, max_attempts)
                for a, b in ranges
            ]
            for future in as_completed(futures):
                raw_rows = future.result()
                tokens = [market_token_id(row) for row in raw_rows]
                fetch_gamma_token_mappings(
                    tokens,
                    lookup,
                    batch_size=gamma_batch_size,
                    timeout_s=timeout_s,
                )
                table, nulls, min_ts, max_ts = rows_to_table(
                    raw_rows,
                    lookup,
                    start_after_ts=start_after_ts,
                    end_ts=end_ts,
                )
                if table is not None:
                    summary["rows"] += table.num_rows
                    summary["null_market_rows"] += nulls
                    summary["min_ts"] = (
                        min_ts
                        if summary["min_ts"] is None
                        else min(int(summary["min_ts"]), int(min_ts))
                    )
                    summary["max_ts"] = (
                        max_ts
                        if summary["max_ts"] is None
                        else max(int(summary["max_ts"]), int(max_ts))
                    )
                    if writer is not None:
                        writer.write_table(table)
                done += 1
                if done % 50 == 0 or done == len(ranges):
                    print(
                        f"{out.name}: chunks={done}/{len(ranges)} "
                        f"rows={summary['rows']:,} max_ts={fmt_ts(summary['max_ts'])}",
                        flush=True,
                    )
    finally:
        if writer is not None:
            writer.close()

    if write:
        if summary["rows"] == 0:
            tmp.unlink(missing_ok=True)
            summary["status"] = "empty"
        else:
            tmp.replace(out)
    return summary


def parse_utc(value: str) -> int:
    text = value.strip()
    if text.isdigit():
        return int(text)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return dt_to_epoch(datetime.fromisoformat(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--rpc-url", default=os.environ.get("POLYGON_RPC_URL", DEFAULT_RPC_URL))
    parser.add_argument("--from-ts", default=None, help="UTC ISO timestamp or epoch; default local max")
    parser.add_argument("--to-ts", default=None, help="UTC ISO timestamp or epoch; default latest-finality")
    parser.add_argument("--finality-blocks", type=int, default=256)
    parser.add_argument("--block-chunk", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--gamma-batch-size", type=int, default=40)
    parser.add_argument("--replace-inprogress", action="store_true")
    parser.add_argument("--max-days", type=int, default=None, help="cap run length for smoke/resume")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lookup = load_market_lookup()
    start_after_ts = parse_utc(args.from_ts) if args.from_ts else local_tail_epoch()

    rpc = PolygonRpc(args.rpc_url, args.timeout_s, args.max_attempts)
    try:
        latest_block = max(0, rpc.block_number() - args.finality_blocks)
        latest_ts = rpc.block_timestamp(latest_block)
        end_ts = parse_utc(args.to_ts) if args.to_ts else latest_ts
        end_ts = min(end_ts, latest_ts)
        start_block = rpc.first_block_at_or_after(start_after_ts + 1, latest_block)
        end_block = rpc.first_block_at_or_after(end_ts, latest_block)
    finally:
        rpc.close()

    print(
        json.dumps(
            {
                "rpc_url": args.rpc_url,
                "write": args.write,
                "from_exclusive_utc": fmt_ts(start_after_ts),
                "to_inclusive_utc": fmt_ts(end_ts),
                "start_block": start_block,
                "end_block": end_block,
                "market_token_mappings_loaded": len(lookup),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    cur = start_after_ts
    day_count = 0
    summaries: list[dict[str, Any]] = []
    while cur < end_ts:
        next_midnight = dt_to_epoch(
            (epoch_to_dt(cur) + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        )
        shard_end_ts = min(end_ts, next_midnight)
        rpc = PolygonRpc(args.rpc_url, args.timeout_s, args.max_attempts)
        try:
            shard_start_block = rpc.first_block_at_or_after(cur + 1, latest_block)
            shard_end_block = rpc.first_block_at_or_after(shard_end_ts, latest_block)
        finally:
            rpc.close()
        summaries.append(
            write_time_shard(
                rpc_url=args.rpc_url,
                start_after_ts=cur,
                end_ts=shard_end_ts,
                start_block=shard_start_block,
                end_block=shard_end_block,
                lookup=lookup,
                block_chunk=args.block_chunk,
                workers=args.workers,
                timeout_s=args.timeout_s,
                max_attempts=args.max_attempts,
                gamma_batch_size=args.gamma_batch_size,
                write=args.write,
                replace_inprogress=args.replace_inprogress,
            )
        )
        cur = shard_end_ts
        day_count += 1
        if args.max_days is not None and day_count >= args.max_days:
            break

    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
