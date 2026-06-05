"""Trade-flow imbalance baseline for Dali.

This is intentionally a small, repeatable harness rather than a monolithic
research notebook. It uses the cached market-level universe screen, selects a
bounded set of markets, materialises only those fills, aggregates to
market-second bars, then tests whether signed historical flow predicts future
price changes.

The historical sign is a proxy: ``maker_side`` is the maker's side, not a
guaranteed aggressor side. The summary reports both the maker-side sign and
its inverse so the first pass can catch sign-convention mistakes.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/dali_tfi_baseline.py --family daily_crypto_up_down
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
TRADES_GLOB = str(DATA / "trades" / "trades_delta_shard*.parquet")
SEED_PATH = DATA / "trades" / "trades_seed.parquet"
BASIC_STATS = ANALYSIS / "dali_market_fill_stats_basic.parquet"


FAMILY_SQL = """
CASE
WHEN regexp_matches(lower(question), '(bitcoin|ethereum|solana|xrp).*up or down|up or down.*(bitcoin|ethereum|solana|xrp)')
    THEN 'daily_crypto_up_down'
WHEN regexp_matches(lower(question), '(s&p|spx|sp500|nasdaq|ndx|dow jones|russell|spy|qqq).*(up|down|green|red|performance|close|open)')
    THEN 'daily_equity_index'
WHEN regexp_matches(lower(question), '(tesla|tsla|nvidia|nvda|apple|aapl|microsoft|msft|meta|amazon|amzn|google|googl).*(up|down|above|below|close|open)')
    THEN 'daily_single_stock'
WHEN regexp_matches(lower(question), '(highest temperature|lowest temperature|temperature in|\\brain\\b|\\bsnow\\b)')
    THEN 'weather_daily'
WHEN regexp_matches(lower(question), '(openai|claude|gpt|gemini|ai model|frontiermath|deepseek|anthropic)')
    THEN 'ai_product'
WHEN regexp_matches(lower(question), '(trump|iran|israel|hezbollah|ukraine|russia|china|fisa|administration|election|governor|mayor|presidential)')
    THEN 'geopolitics_policy'
WHEN regexp_matches(lower(question), '(vs\\.| vs |spread:|o/u|over/under|playoffs|series|win on 2026|mlb|nba|nhl|fifa|world cup)')
    THEN 'sports_game_lines'
ELSE 'other' END
"""


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


def safe_name(raw: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return out or "screen"


def build_candidates(
    con: duckdb.DuckDBPyConnection,
    markets_path: Path,
    out_path: Path,
    family: str,
    since: str,
    max_markets: int,
    closed_only: bool,
    min_fills: int,
    max_fills: int,
    min_usd_volume: float,
    max_usd_volume: float,
    min_active_days: int,
) -> None:
    closed_filter = "AND closed" if closed_only else ""
    con.execute(
        f"""
        COPY (
            WITH s AS (SELECT * FROM read_parquet('{BASIC_STATS}')),
            m AS (
                SELECT
                    CAST(id AS VARCHAR) AS market_id,
                    question,
                    slug,
                    active,
                    closed,
                    TRY_CAST(end_date AS TIMESTAMP) AS end_ts,
                    volume AS gamma_volume,
                    liquidity
                FROM read_parquet('{markets_path}')
            ),
            j AS (
                SELECT
                    s.market_id,
                    s.condition_id,
                    m.question,
                    m.slug,
                    m.closed,
                    m.end_ts,
                    s.n_fills,
                    round(s.usd_volume, 0) AS usd_volume,
                    s.first_fill_ts,
                    s.last_fill_ts,
                    s.active_fill_days,
                    round(100.0 * s.midband_fills / NULLIF(s.n_fills, 0), 1) AS midband_pct,
                    round(m.gamma_volume, 0) AS gamma_volume,
                    round(m.liquidity, 0) AS liquidity,
                    {FAMILY_SQL} AS family
                FROM s JOIN m USING (market_id)
            )
            SELECT *
            FROM j
            WHERE family = '{family}'
              {closed_filter}
              AND last_fill_ts >= TIMESTAMP '{since}'
              AND active_fill_days >= {min_active_days}
              AND n_fills BETWEEN {min_fills} AND {max_fills}
              AND usd_volume BETWEEN {min_usd_volume} AND {max_usd_volume}
              AND midband_pct >= 30
            ORDER BY last_fill_ts DESC, active_fill_days DESC, n_fills DESC
            LIMIT {max_markets}
        ) TO '{out_path}' (HEADER, DELIMITER ',')
        """
    )


def materialise_fills(
    con: duckdb.DuckDBPyConnection,
    candidates_path: Path,
    out_path: Path,
) -> None:
    con.execute(
        f"""
        COPY (
            WITH cand AS (
                SELECT CAST(market_id AS VARCHAR) AS market_id
                FROM read_csv_auto('{candidates_path}')
            ),
            raw AS (
                SELECT * FROM read_parquet('{TRADES_GLOB}')
                UNION ALL BY NAME
                SELECT * FROM read_parquet('{SEED_PATH}')
            )
            SELECT raw.*
            FROM raw JOIN cand USING (market_id)
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def build_second_bars(
    con: duckdb.DuckDBPyConnection,
    fills_path: Path,
    candidates_path: Path,
    out_path: Path,
) -> None:
    con.execute(
        f"""
        COPY (
            WITH cand AS (
                SELECT
                    CAST(market_id AS VARCHAR) AS market_id,
                    end_ts
                FROM read_csv_auto('{candidates_path}')
            ),
            fills AS (
                SELECT f.*, cand.end_ts
                FROM read_parquet('{fills_path}') f
                LEFT JOIN cand USING (market_id)
            )
            SELECT
                market_id,
                any_value(end_ts) AS end_ts,
                timestamp AS second_ts,
                count(*) AS n_fills,
                count(DISTINCT transaction_hash) AS n_txs,
                sum(usd_amount) AS gross_usd,
                sum(CASE WHEN maker_side = 'BUY' THEN usd_amount ELSE 0 END) AS maker_buy_usd,
                sum(CASE WHEN maker_side = 'SELL' THEN usd_amount ELSE 0 END) AS maker_sell_usd,
                sum(CASE WHEN maker_side = 'BUY' THEN usd_amount ELSE -usd_amount END) AS signed_maker_usd,
                sum(price * usd_amount) / NULLIF(sum(usd_amount), 0) AS vwap_price,
                min(price) AS min_price,
                max(price) AS max_price
            FROM fills
            WHERE price IS NOT NULL
              AND usd_amount > 0
            GROUP BY market_id, timestamp
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def build_eval_for_horizon(
    con: duckdb.DuckDBPyConnection,
    bars_path: Path,
    eval_path: Path,
    horizon_seconds: int,
    append: bool,
) -> None:
    mode = "INSERT INTO eval_rows" if append else "CREATE OR REPLACE TABLE eval_rows AS"
    con.execute(
        f"""
        CREATE OR REPLACE VIEW bars AS
        SELECT * FROM read_parquet('{bars_path}')
        WHERE vwap_price BETWEEN 0.01 AND 0.99
        """
    )
    con.execute(
        f"""
        {mode}
        WITH base AS (
            SELECT
                *,
                second_ts + INTERVAL {horizon_seconds} SECOND AS target_ts
            FROM bars
        )
        SELECT
            {horizon_seconds} AS horizon_seconds,
            b.market_id,
            b.second_ts,
            b.target_ts,
            b.end_ts,
            f.second_ts AS future_ts,
            date_diff('second', b.target_ts, f.second_ts) AS future_gap_seconds,
            b.n_fills,
            b.n_txs,
            b.gross_usd,
            b.signed_maker_usd,
            b.vwap_price,
            f.vwap_price AS future_vwap_price,
            f.vwap_price - b.vwap_price AS future_price_change
        FROM base b
        ASOF LEFT JOIN bars f
          ON b.market_id = f.market_id
         AND b.target_ts <= f.second_ts
        """
    )
    con.execute(
        f"""
        COPY (SELECT * FROM eval_rows)
        TO '{eval_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def write_summary(
    con: duckdb.DuckDBPyConnection,
    eval_path: Path,
    out_path: Path,
    min_signal_usd: float,
    max_future_gap_seconds: int,
    exclude_last_seconds: int,
) -> None:
    con.execute(
        f"""
        COPY (
            WITH e AS (
                SELECT *
                FROM read_parquet('{eval_path}')
                WHERE future_vwap_price IS NOT NULL
                  AND future_gap_seconds <= {max_future_gap_seconds}
                  AND (end_ts IS NULL OR future_ts <= end_ts)
                  AND (
                      end_ts IS NULL
                      OR date_diff('second', second_ts, end_ts) >= {exclude_last_seconds}
                  )
                  AND abs(signed_maker_usd) >= {min_signal_usd}
                  AND signed_maker_usd != 0
            ),
            labelled AS (
                SELECT
                    horizon_seconds,
                    'maker_side' AS sign_convention,
                    sign(signed_maker_usd) AS signal_direction,
                    future_price_change,
                    abs(signed_maker_usd) AS abs_signal_usd,
                    gross_usd,
                    future_gap_seconds
                FROM e
                UNION ALL
                SELECT
                    horizon_seconds,
                    'inverse_maker_side' AS sign_convention,
                    -sign(signed_maker_usd) AS signal_direction,
                    future_price_change,
                    abs(signed_maker_usd) AS abs_signal_usd,
                    gross_usd,
                    future_gap_seconds
                FROM e
            )
            SELECT
                horizon_seconds,
                sign_convention,
                count(*) AS n_obs,
                round(avg(abs_signal_usd), 2) AS avg_abs_signal_usd,
                round(avg(gross_usd), 2) AS avg_second_gross_usd,
                round(avg(future_gap_seconds), 2) AS avg_future_gap_seconds,
                round(100.0 * avg(signal_direction * future_price_change), 4) AS edge_cents_per_share,
                round(100.0 * avg(CASE WHEN signal_direction * future_price_change > 0 THEN 1 ELSE 0 END), 2) AS hit_rate_pct,
                round(100.0 * avg(future_price_change), 4) AS unconditional_move_cents,
                round(100.0 * quantile_cont(signal_direction * future_price_change, 0.10), 4) AS p10_signed_move_cents,
                round(100.0 * quantile_cont(signal_direction * future_price_change, 0.50), 4) AS p50_signed_move_cents,
                round(100.0 * quantile_cont(signal_direction * future_price_change, 0.90), 4) AS p90_signed_move_cents
            FROM labelled
            GROUP BY horizon_seconds, sign_convention
            ORDER BY horizon_seconds, sign_convention
        ) TO '{out_path}' (HEADER, DELIMITER ',')
        """
    )


def print_count(con: duckdb.DuckDBPyConnection, label: str, path: Path, kind: str) -> None:
    if kind == "csv":
        n = con.sql(f"SELECT count(*) FROM read_csv_auto('{path}')").fetchone()[0]
    else:
        n = con.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    print(f"{label}: {n:,} rows -> {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="daily_crypto_up_down")
    parser.add_argument("--since", default="2026-01-01")
    parser.add_argument("--max-markets", type=int, default=250)
    parser.add_argument("--min-fills", type=int, default=200)
    parser.add_argument("--max-fills", type=int, default=50_000)
    parser.add_argument("--min-usd-volume", type=float, default=10_000)
    parser.add_argument("--max-usd-volume", type=float, default=2_000_000)
    parser.add_argument("--min-active-days", type=int, default=2)
    parser.add_argument("--include-open", action="store_true")
    parser.add_argument("--min-signal-usd", type=float, default=25.0)
    parser.add_argument("--max-future-gap-seconds", type=int, default=300)
    parser.add_argument(
        "--exclude-last-seconds",
        type=int,
        default=0,
        help="drop observations this close to market end; useful for avoiding resolution collapse",
    )
    parser.add_argument("--horizons", default="30,120,300")
    parser.add_argument("--prefix")
    args = parser.parse_args()

    if not BASIC_STATS.exists():
        raise SystemExit(
            f"missing {BASIC_STATS}; run scripts/dali_market_universe_screen.py first"
        )

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"dali_tfi_{safe_name(args.family)}_{args.max_markets}"
    candidates_path = ANALYSIS / f"{prefix}_candidates.csv"
    fills_path = ANALYSIS / f"{prefix}_fills.parquet"
    bars_path = ANALYSIS / f"{prefix}_seconds.parquet"
    eval_path = ANALYSIS / f"{prefix}_eval.parquet"
    summary_path = ANALYSIS / f"{prefix}_summary.csv"

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    con = connect()
    markets_path = latest_markets_parquet()

    t0 = time.time()
    build_candidates(
        con=con,
        markets_path=markets_path,
        out_path=candidates_path,
        family=args.family,
        since=args.since,
        max_markets=args.max_markets,
        closed_only=not args.include_open,
        min_fills=args.min_fills,
        max_fills=args.max_fills,
        min_usd_volume=args.min_usd_volume,
        max_usd_volume=args.max_usd_volume,
        min_active_days=args.min_active_days,
    )
    print_count(con, "candidates", candidates_path, "csv")

    materialise_fills(con, candidates_path, fills_path)
    print_count(con, "fills", fills_path, "parquet")

    build_second_bars(con, fills_path, candidates_path, bars_path)
    print_count(con, "market-second bars", bars_path, "parquet")

    for i, horizon in enumerate(horizons):
        build_eval_for_horizon(con, bars_path, eval_path, horizon, append=i > 0)
    print_count(con, "eval rows", eval_path, "parquet")

    write_summary(
        con,
        eval_path,
        summary_path,
        min_signal_usd=args.min_signal_usd,
        max_future_gap_seconds=args.max_future_gap_seconds,
        exclude_last_seconds=args.exclude_last_seconds,
    )
    print_count(con, "summary", summary_path, "csv")
    print(f"elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
