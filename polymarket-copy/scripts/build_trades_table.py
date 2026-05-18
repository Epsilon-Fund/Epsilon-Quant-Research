import time
from pathlib import Path

from data_infra.duck import connect

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "data" / "raw" / "orderFilled_complete.csv"
MARKETS_DIR = ROOT / "data" / "markets"
OUT_DIR = ROOT / "data" / "trades"
OUT_PATH = OUT_DIR / "trades_seed.parquet"

WATCHLIST_ADDR = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"


def latest_markets_parquet() -> Path:
    candidates = sorted(MARKETS_DIR.glob("markets_*.parquet"))
    if not candidates:
        raise SystemExit(f"no markets_*.parquet in {MARKETS_DIR}")
    return candidates[-1]


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    if not TRADES_CSV.exists():
        raise SystemExit(f"missing {TRADES_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markets_path = latest_markets_parquet()
    print(f"trades CSV : {TRADES_CSV}")
    print(f"markets    : {markets_path}")
    print(f"output     : {OUT_PATH}")

    con = connect()

    section("building trades parquet")
    t0 = time.time()
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
                    takerAssetId AS taker_asset_id,
                    makerAmountFilled,
                    takerAmountFilled,
                    transactionHash,
                    CASE WHEN makerAssetId = '0' THEN takerAssetId ELSE makerAssetId END AS market_token_id,
                    CASE WHEN makerAssetId = '0'
                         THEN CAST(makerAmountFilled AS DOUBLE)
                         ELSE CAST(takerAmountFilled AS DOUBLE) END AS usd_base,
                    CASE WHEN makerAssetId = '0'
                         THEN CAST(takerAmountFilled AS DOUBLE)
                         ELSE CAST(makerAmountFilled AS DOUBLE) END AS token_base
                FROM read_csv(
                    '{TRADES_CSV}',
                    types = {{
                        'makerAssetId': 'VARCHAR',
                        'takerAssetId': 'VARCHAR',
                        'makerAmountFilled': 'VARCHAR',
                        'takerAmountFilled': 'VARCHAR'
                    }},
                    header = TRUE
                )
            )
            SELECT
                CAST(to_timestamp(t.timestamp) AT TIME ZONE 'UTC' AS TIMESTAMP) AS timestamp,
                m.market_id,
                m.condition_id,
                m.neg_risk,
                t.maker,
                t.taker,
                t.maker_asset_id,
                t.taker_asset_id,
                t.usd_base / 1e6 AS usd_amount,
                t.token_base / 1e6 AS token_amount,
                CASE WHEN t.token_base = 0
                     THEN NULL
                     ELSE t.usd_base / t.token_base END AS price,
                CASE WHEN t.maker_asset_id = '0' THEN 'BUY' ELSE 'SELL' END AS maker_side,
                t.transactionHash AS transaction_hash
            FROM trades_src t
            LEFT JOIN markets_tokens m ON m.token_id = t.market_token_id
        ) TO '{OUT_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    elapsed = time.time() - t0
    print(f"build elapsed: {elapsed:.1f}s")

    out = f"read_parquet('{OUT_PATH}')"
    section("summary")
    summary = con.sql(f"""
        SELECT
            count(*) AS rows_out,
            100.0 * sum(CASE WHEN market_id IS NULL THEN 1 ELSE 0 END) / count(*) AS null_market_pct,
            min(timestamp) AS earliest,
            max(timestamp) AS latest,
            count(DISTINCT market_id) FILTER (WHERE market_id IS NOT NULL) AS unique_markets,
            count(DISTINCT maker) AS unique_makers,
            count(DISTINCT taker) AS unique_takers
        FROM {out}
    """).fetchone()
    rows_out = summary[0]
    file_size_gb = OUT_PATH.stat().st_size / (1024**3)
    print(f"  rows in (== rows out, LEFT JOIN preserves) : {rows_out:,}")
    print(f"  rows out                                   : {rows_out:,}")
    print(f"  null market_id pct                         : {summary[1]:.2f}%")
    print(f"  date range                                 : {summary[2]} → {summary[3]}")
    print(f"  file size on disk                          : {file_size_gb:.2f} GB")
    print(f"  unique markets matched                     : {summary[4]:,}")
    print(f"  unique makers                              : {summary[5]:,}")
    print(f"  unique takers                              : {summary[6]:,}")

    section(f"sanity: {WATCHLIST_ADDR} (maker side)")
    print(con.sql(f"""
        SELECT
            count(*) AS trades,
            sum(usd_amount) AS total_usd,
            count(DISTINCT market_id) FILTER (WHERE market_id IS NOT NULL) AS distinct_markets,
            min(timestamp) AS earliest,
            max(timestamp) AS latest
        FROM {out}
        WHERE maker = '{WATCHLIST_ADDR}'
    """).fetchdf().to_string(index=False))

    section(f"5 most recent trades for {WATCHLIST_ADDR} (with market question)")
    print(con.sql(f"""
        SELECT
            t.timestamp,
            t.market_id,
            mk.question,
            t.maker_side,
            round(t.usd_amount, 2) AS usd_amount,
            round(t.token_amount, 4) AS token_amount,
            round(t.price, 4) AS price
        FROM {out} t
        LEFT JOIN read_parquet('{markets_path}') mk ON mk.id = t.market_id
        WHERE t.maker = '{WATCHLIST_ADDR}'
        ORDER BY t.timestamp DESC
        LIMIT 5
    """).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
