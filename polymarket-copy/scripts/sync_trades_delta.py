import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_infra.duck import connect
from data_infra.goldsky import GoldskyClient

ROOT = Path(__file__).resolve().parents[1]
TRADES_DIR = ROOT / "data" / "trades"
MARKETS_DIR = ROOT / "data" / "markets"
RAW_TMP = ROOT / "data" / "_delta_raw_tmp.parquet"

WATCHLIST = {
    "domah":      "0x9d84ce0306f8551e02efef1680475fc0f1dc1344",
    "bossoskil1": "0xa5ea13a81d2b7e8e424b182bdc1db08e756bd96a",
}

RAW_SCHEMA = pa.schema([
    pa.field("timestamp", pa.int64()),
    pa.field("maker", pa.string()),
    pa.field("makerAssetId", pa.string()),
    pa.field("makerAmountFilled", pa.string()),
    pa.field("taker", pa.string()),
    pa.field("takerAssetId", pa.string()),
    pa.field("takerAmountFilled", pa.string()),
    pa.field("fee", pa.string()),
    pa.field("transactionHash", pa.string()),
])


def section(s: str) -> None:
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


def latest_markets_parquet() -> Path:
    candidates = sorted(MARKETS_DIR.glob("markets_*.parquet"))
    if not candidates:
        raise SystemExit(f"no markets_*.parquet in {MARKETS_DIR}")
    return candidates[-1]


def existing_max_timestamp(con) -> int:
    shards = sorted(TRADES_DIR.glob("*.parquet"))
    if not shards:
        return 0
    paths = ", ".join(f"'{p}'" for p in shards)
    (ts,) = con.sql(f"SELECT max(epoch(timestamp))::BIGINT FROM read_parquet([{paths}])").fetchone()
    return int(ts or 0)


def to_batch(rows: list[dict]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {
            "timestamp":         [int(r["timestamp"]) for r in rows],
            "maker":             [r["maker"] for r in rows],
            "makerAssetId":      [r["makerAssetId"] for r in rows],
            "makerAmountFilled": [r["makerAmountFilled"] for r in rows],
            "taker":             [r["taker"] for r in rows],
            "takerAssetId":      [r["takerAssetId"] for r in rows],
            "takerAmountFilled": [r["takerAmountFilled"] for r in rows],
            "fee":               [r["fee"] for r in rows],
            "transactionHash":   [r["transactionHash"] for r in rows],
        },
        schema=RAW_SCHEMA,
    )


def fmt_ts(ts: int | None) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fetch_to_raw(cursor_ts: int) -> tuple[int, int | None, int | None]:
    if RAW_TMP.exists():
        RAW_TMP.unlink()
    seen_ids: set[str] = set()
    n_rows = 0
    n_pages = 0
    min_ts: int | None = None
    max_ts: int | None = None
    t0 = time.time()
    gc = GoldskyClient()
    writer = pq.ParquetWriter(RAW_TMP, RAW_SCHEMA, compression="zstd")
    try:
        for page in gc.iter_order_filled_events(timestamp_gt=cursor_ts, page_size=1000):
            rows = [r for r in page if r["id"] not in seen_ids]
            for r in rows:
                seen_ids.add(r["id"])
            if not rows:
                continue
            writer.write_batch(to_batch(rows))
            n_rows += len(rows)
            n_pages += 1
            page_ts = [int(r["timestamp"]) for r in rows]
            page_min, page_max = min(page_ts), max(page_ts)
            min_ts = page_min if min_ts is None else min(min_ts, page_min)
            max_ts = page_max if max_ts is None else max(max_ts, page_max)
            if n_pages % 25 == 0:
                rate = n_rows / max(time.time() - t0, 1)
                print(f"  pages={n_pages}  rows={n_rows:,}  latest={fmt_ts(max_ts)}  rate={rate:.0f} rows/s")
    finally:
        writer.close()
    elapsed = time.time() - t0
    print(f"\nfetched {n_rows:,} rows in {elapsed:.0f}s  ({n_rows / max(elapsed, 1):.0f} rows/s)")
    return n_rows, min_ts, max_ts


def enrich_to_shard(min_ts: int, max_ts: int, markets_path: Path, con) -> Path:
    start_date = datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    end_date = datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    out_path = TRADES_DIR / f"trades_delta_{start_date}_{end_date}.parquet"
    con.execute(f"""
        COPY (
            WITH markets_tokens AS (
                SELECT
                    token_id,
                    any_value(market_id) AS market_id,
                    any_value(condition_id) AS condition_id,
                    any_value(neg_risk) AS neg_risk
                FROM (
                    SELECT
                        CAST(id AS VARCHAR) AS market_id,
                        condition_id,
                        neg_risk,
                        UNNEST(clob_token_ids) AS token_id
                    FROM read_parquet('{markets_path}')
                    WHERE len(clob_token_ids) > 0
                )
                GROUP BY token_id
            ),
            trades_src AS (
                SELECT
                    timestamp,
                    lower(maker) AS maker,
                    lower(taker) AS taker,
                    makerAssetId AS maker_asset_id,
                    transactionHash,
                    CASE WHEN makerAssetId = '0' THEN takerAssetId ELSE makerAssetId END AS market_token_id,
                    CASE WHEN makerAssetId = '0'
                         THEN CAST(makerAmountFilled AS DOUBLE)
                         ELSE CAST(takerAmountFilled AS DOUBLE) END AS usd_base,
                    CASE WHEN makerAssetId = '0'
                         THEN CAST(takerAmountFilled AS DOUBLE)
                         ELSE CAST(makerAmountFilled AS DOUBLE) END AS token_base
                FROM read_parquet('{RAW_TMP}')
            )
            SELECT
                CAST(to_timestamp(t.timestamp) AT TIME ZONE 'UTC' AS TIMESTAMP) AS timestamp,
                m.market_id,
                m.condition_id,
                m.neg_risk,
                t.maker,
                t.taker,
                t.usd_base / 1e6 AS usd_amount,
                t.token_base / 1e6 AS token_amount,
                CASE WHEN t.token_base = 0
                     THEN NULL
                     ELSE t.usd_base / t.token_base END AS price,
                CASE WHEN t.maker_asset_id = '0' THEN 'BUY' ELSE 'SELL' END AS maker_side,
                t.transactionHash AS transaction_hash
            FROM trades_src t
            LEFT JOIN markets_tokens m ON m.token_id = t.market_token_id
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    RAW_TMP.unlink()
    return out_path


def trader_stats(con, parquet_arg: str, addr: str):
    return con.sql(f"""
        SELECT
            count(*) AS trades,
            sum(usd_amount) AS total_usd,
            count(DISTINCT market_id) FILTER (WHERE market_id IS NOT NULL) AS distinct_markets,
            min(timestamp) AS earliest,
            max(timestamp) AS latest
        FROM read_parquet({parquet_arg})
        WHERE maker = '{addr}'
    """).fetchdf()


def main() -> None:
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    markets_path = latest_markets_parquet()
    con = connect()

    cursor_ts = existing_max_timestamp(con)
    print(f"existing max timestamp : {cursor_ts} ({fmt_ts(cursor_ts)})")
    print(f"markets parquet        : {markets_path}")

    section("fetching from Goldsky")
    n_rows, min_ts, max_ts = fetch_to_raw(cursor_ts)
    if n_rows == 0:
        print("\nno new rows; nothing to write.")
        if RAW_TMP.exists():
            RAW_TMP.unlink()
        return

    section("enriching delta + joining markets")
    out_path = enrich_to_shard(min_ts, max_ts, markets_path, con)

    section("delta summary")
    out_arg = f"'{out_path}'"
    summary = con.sql(f"""
        SELECT
            count(*) AS rows,
            100.0 * sum(CASE WHEN market_id IS NOT NULL THEN 1 ELSE 0 END) / count(*) AS pct_matched,
            min(timestamp) AS earliest,
            max(timestamp) AS latest
        FROM read_parquet({out_arg})
    """).fetchone()
    print(f"  shard          : {out_path.name}")
    print(f"  rows           : {summary[0]:,}")
    print(f"  % matched      : {summary[1]:.2f}%")
    print(f"  earliest       : {summary[2]}")
    print(f"  new max ts     : {summary[3]}")
    print(f"  size on disk   : {out_path.stat().st_size / (1024**3):.2f} GB")

    seed_arg = f"'{TRADES_DIR / 'trades_seed.parquet'}'"
    all_arg = f"'{TRADES_DIR}/*.parquet'"
    for name, addr in WATCHLIST.items():
        section(f"{name}  ({addr})")
        print("seed only:")
        print(trader_stats(con, seed_arg, addr).to_string(index=False))
        print("\nseed + delta:")
        print(trader_stats(con, all_arg, addr).to_string(index=False))
        print("\n5 most recent (seed+delta):")
        print(con.sql(f"""
            SELECT
                t.timestamp,
                t.market_id,
                mk.question,
                t.maker_side,
                round(t.usd_amount, 2) AS usd_amount,
                round(t.token_amount, 4) AS token_amount,
                round(t.price, 4) AS price
            FROM read_parquet({all_arg}) t
            LEFT JOIN read_parquet('{markets_path}') mk ON mk.id = t.market_id
            WHERE t.maker = '{addr}'
            ORDER BY t.timestamp DESC
            LIMIT 5
        """).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
