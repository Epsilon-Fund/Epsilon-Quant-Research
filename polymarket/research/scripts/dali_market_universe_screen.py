"""Dali market-universe screen from historical OrderFilled data.

This deliberately avoids repeated 1B-row exploratory scans. The workflow is:

1. Build a narrow market-level aggregate once with ``--rebuild-basic``.
2. Run all candidate screens from that small aggregate plus Gamma metadata.
3. Do heavier trade-flow/order-flow research only on the shortlisted markets.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/dali_market_universe_screen.py
    PYTHONPATH=. uv run python scripts/dali_market_universe_screen.py --rebuild-basic
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
TRADES_GLOB = str(DATA / "trades" / "trades_delta_shard*.parquet")
SEED_PATH = DATA / "trades" / "trades_seed.parquet"
BASIC_STATS = ANALYSIS / "dali_market_fill_stats_basic.parquet"
FORWARD_OUT = ANALYSIS / "csv_outputs" / "dali" / "dali_forward_viable_market_screen.csv"
OPEN_OUT = ANALYSIS / "csv_outputs" / "dali" / "dali_open_recent_market_screen.csv"
CLOSED_OUT = ANALYSIS / "csv_outputs" / "dali" / "dali_closed_backtest_market_screen.csv"


def latest_markets_parquet() -> Path:
    candidates = sorted((DATA / "markets").glob("markets_*.parquet"))
    if not candidates:
        raise SystemExit(f"no markets_*.parquet found in {DATA / 'markets'}")
    return candidates[-1]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute("PRAGMA preserve_insertion_order=false")
    return con


def build_basic_stats(con: duckdb.DuckDBPyConnection) -> None:
    """One narrow full-history pass.

    Avoid approximate distincts/quantiles here; they are the expensive bits that
    caused earlier brute-force attempts to balloon. Add those later only for a
    short candidate list.
    """
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    con.execute(
        f"""
        CREATE OR REPLACE VIEW raw AS
            SELECT * FROM read_parquet('{TRADES_GLOB}')
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{SEED_PATH}')
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT
                market_id,
                any_value(condition_id) AS condition_id,
                count(*) AS n_fills,
                sum(usd_amount) AS usd_volume,
                min(timestamp) AS first_fill_ts,
                max(timestamp) AS last_fill_ts,
                count(DISTINCT CAST(timestamp AS DATE)) AS active_fill_days,
                avg(price) AS avg_fill_price,
                min(price) AS min_fill_price,
                max(price) AS max_fill_price,
                sum(CASE WHEN maker_side = 'BUY' THEN usd_amount ELSE 0 END) AS maker_buy_usd,
                sum(CASE WHEN maker_side = 'SELL' THEN usd_amount ELSE 0 END) AS maker_sell_usd,
                count(*) FILTER (WHERE price BETWEEN 0.05 AND 0.95) AS midband_fills
            FROM raw
            WHERE market_id IS NOT NULL
            GROUP BY market_id
        ) TO '{BASIC_STATS}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    print(f"rebuilt {BASIC_STATS.relative_to(ROOT)} in {time.time() - t0:.1f}s")


def screen_query(markets_path: Path, where_clause: str, limit: int) -> str:
    return f"""
    WITH s AS (SELECT * FROM read_parquet('{BASIC_STATS}')),
    m AS (
        SELECT
            CAST(id AS VARCHAR) AS market_id,
            question,
            slug,
            volume AS gamma_volume,
            liquidity,
            active,
            closed,
            TRY_CAST(end_date AS TIMESTAMP) AS end_ts,
            TRY_CAST(created_at AS TIMESTAMP) AS created_ts
        FROM read_parquet('{markets_path}')
    ),
    j AS (
        SELECT
            s.market_id,
            s.condition_id,
            m.question,
            m.slug,
            m.active,
            m.closed,
            m.end_ts,
            m.created_ts,
            m.gamma_volume,
            m.liquidity,
            s.n_fills,
            round(s.usd_volume, 0) AS usd_volume,
            s.first_fill_ts,
            s.last_fill_ts,
            s.active_fill_days,
            round(s.avg_fill_price, 4) AS avg_fill_price,
            round(100.0 * s.midband_fills / NULLIF(s.n_fills, 0), 1) AS midband_pct,
            round(
                (s.maker_buy_usd - s.maker_sell_usd)
                / NULLIF(s.maker_buy_usd + s.maker_sell_usd, 0),
                3
            ) AS maker_buy_sell_imbalance,
            CASE
                WHEN s.n_fills BETWEEN 200 AND 5000
                 AND s.usd_volume BETWEEN 10000 AND 250000
                    THEN 'thin_alive'
                WHEN s.n_fills BETWEEN 5000 AND 50000
                 AND s.usd_volume BETWEEN 250000 AND 2000000
                    THEN 'liquid_baseline'
                ELSE 'other'
            END AS bucket
        FROM s JOIN m USING (market_id)
    )
    SELECT
        market_id,
        question,
        slug,
        n_fills,
        usd_volume,
        active_fill_days,
        first_fill_ts,
        last_fill_ts,
        end_ts,
        round(gamma_volume, 0) AS gamma_volume,
        round(liquidity, 0) AS liquidity,
        avg_fill_price,
        midband_pct,
        maker_buy_sell_imbalance,
        bucket
    FROM j
    WHERE {where_clause}
    ORDER BY
        CASE WHEN bucket = 'thin_alive' THEN 0
             WHEN bucket = 'liquid_baseline' THEN 1
             ELSE 2 END,
        last_fill_ts DESC,
        active_fill_days DESC,
        n_fills DESC
    LIMIT {limit}
    """


def write_screen(con: duckdb.DuckDBPyConnection, query: str, out_path: Path) -> None:
    con.execute(f"COPY ({query}) TO '{out_path}' (HEADER, DELIMITER ',')")
    n = con.sql(f"SELECT count(*) FROM read_csv_auto('{out_path}')").fetchone()[0]
    print(f"wrote {out_path.relative_to(ROOT)} ({n} rows)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild-basic",
        action="store_true",
        help="rebuild the small market-level aggregate from raw OrderFilled parquets",
    )
    args = parser.parse_args()

    con = connect()
    markets_path = latest_markets_parquet()
    if args.rebuild_basic or not BASIC_STATS.exists():
        build_basic_stats(con)
    else:
        print(f"using cached {BASIC_STATS.relative_to(ROOT)}")

    common = """
        active_fill_days >= 2
        AND n_fills BETWEEN 200 AND 50000
        AND usd_volume BETWEEN 10000 AND 2000000
        AND midband_pct >= 30
    """
    open_recent = f"""
        active AND NOT closed
        AND last_fill_ts >= TIMESTAMP '2026-04-01'
        AND {common}
    """
    forward_viable = f"""
        active AND NOT closed
        AND (end_ts IS NULL OR end_ts >= TIMESTAMP '2026-05-23')
        AND last_fill_ts >= TIMESTAMP '2026-04-01'
        AND active_fill_days >= 5
        AND {common}
    """
    closed_backtest = f"""
        closed
        AND last_fill_ts >= TIMESTAMP '2026-01-01'
        AND {common}
    """

    write_screen(con, screen_query(markets_path, forward_viable, 40), FORWARD_OUT)
    write_screen(con, screen_query(markets_path, open_recent, 50), OPEN_OUT)
    write_screen(con, screen_query(markets_path, closed_backtest, 100), CLOSED_OUT)

    summary = con.sql(
        f"""
        SELECT
            count(*) AS markets,
            sum(n_fills) AS fills,
            round(sum(usd_volume), 0) AS usd_volume
        FROM read_parquet('{BASIC_STATS}')
        """
    ).fetchone()
    print(
        "basic stats universe: "
        f"{summary[0]:,} markets, {summary[1]:,.0f} fills, ${summary[2]:,.0f} volume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
