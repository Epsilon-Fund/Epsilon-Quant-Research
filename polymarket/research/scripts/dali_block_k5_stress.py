"""Block K5-STRESS: de-biased maker PnL and deployability gate.

Research-only sidecar. This is deliberately adversarial to the K5 maker edge:
it expands from the K5 top-analysis sample to every ex-ante maker-heavy wallet,
marks unresolved inventory, excludes incumbent top-3 market makers, and reports
whether any category remains buildable after selection, survivorship, capacity,
rebate, and stability checks.

Data limitation: the owned historical files contain fills, not order/cancel
events or historical order-book mids. Resolved inventory is marked to settlement
through closed_positions.parquet. Unresolved/open inventory is reconstructed from
raw fills and marked to the latest executed price for that outcome, used as a
last-mid proxy.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
NOTES = ROOT / "notes"
TRADES_GLOB = str(DATA / "trades" / "*.parquet")
CLOSED_POSITIONS = DATA / "closed_positions.parquet"
TRADERS = DATA / "traders.parquet"
MARKETS = sorted((DATA / "markets").glob("markets_*.parquet"))[-1]
K5_CSV = ANALYSIS / "csv_outputs" / "market_making" / "k5_real_maker_pnl.csv"
OUT_CSV = ANALYSIS / "csv_outputs" / "market_making" / "k5_stress.csv"
NOTE = NOTES / "block_k5_stress_findings.md"
WALLET_MARKET_CACHE = ANALYSIS / "k5_stress_wallet_market_full.parquet"
MAKERS_CACHE = ANALYSIS / "k5_stress_qualified_makers.parquet"
MARKOUT_CACHE = ANALYSIS / "k5_stress_markout_sample.parquet"
CALIBRATION_CACHE = ANALYSIS / "k5_stress_calibration_obs.parquet"

sys.path.insert(0, str(ROOT))
from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG  # noqa: E402


RNG_SEED = 20260531
BOOTSTRAP_SAMPLES = 500
MAKER_SHARE_CUTOFF = 0.70
MIN_PASSIVE_FILLS = 1_000
DATA_START = "2025-08-01"
DATA_END = "2026-05-31"
TAIL_MIN_PRICE = 0.005
TAIL_MAX_PRICE = 0.20

STRUCT_TWO_SIDED_MIN = 0.60
STRUCT_CARRY_MIN = 0.50
STRUCT_SPIKE_MAX = 0.02
STABILITY_POS_MONTH_SHARE_MIN = 2.0 / 3.0
MARKOUT_HASH_MOD = 1_000
MARKOUT_HASH_KEEP = 25
MARKOUT_MARKET_HASH_MOD = 100
MARKOUT_MARKET_HASH_KEEP = 2
MARKOUT_MAX_LAG_SEC = 300

HORIZONS = [
    ("1h", 1 * 3600, 30 * 60),
    ("4h", 4 * 3600, 2 * 3600),
    ("1d", 24 * 3600, 12 * 3600),
    ("7d", 7 * 24 * 3600, 3 * 24 * 3600),
    ("30d", 30 * 24 * 3600, 7 * 24 * 3600),
]

FEE_BY_FAMILY = {
    "Crypto": (0.07, 0.20),
    "Sports": (0.03, 0.25),
    "Finance": (0.04, 0.25),
    "Politics": (0.04, 0.25),
    "Economics": (0.05, 0.25),
    "Culture": (0.05, 0.25),
    "Weather": (0.05, 0.25),
    "Tech": (0.04, 0.25),
    "Other": (0.05, 0.25),
    "Geopolitics": (0.00, 0.00),
}


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:,.0f} bps" if abs(value) >= 100 else f"{value:,.1f} bps"


def dollars(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def cents(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}c"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def sql_list(values: set[str] | list[str] | tuple[str, ...]) -> str:
    vals = sorted({str(v).lower().replace("'", "''") for v in values if str(v)})
    return ", ".join(f"'{v}'" for v in vals) if vals else "''"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    temp_dir = ANALYSIS / ".duckdb_tmp_k5_stress"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    return con


def install_market_tables(con: duckdb.DuckDBPyConnection) -> None:
    cases_rate = " ".join(
        f"WHEN fee_family = '{fam}' THEN {rate}" for fam, (rate, _) in FEE_BY_FAMILY.items()
    )
    cases_rebate = " ".join(
        f"WHEN fee_family = '{fam}' THEN {rebate}" for fam, (_, rebate) in FEE_BY_FAMILY.items()
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE market_meta AS
        WITH base AS (
            SELECT
                CAST(id AS VARCHAR) AS market_id,
                CAST(condition_id AS VARCHAR) AS condition_id,
                lower(coalesce(slug, '')) AS slug_l,
                coalesce(slug, '') AS slug,
                lower(coalesce(question, '')) AS question_l,
                coalesce(question, '') AS question,
                coalesce(neg_risk, false) AS neg_risk,
                coalesce(closed, false) AS closed,
                TRY_CAST(end_date AS TIMESTAMP) AS end_ts,
                TRY_CAST(created_at AS TIMESTAMP) AS created_ts,
                len(clob_token_ids) AS n_tokens
            FROM read_parquet('{MARKETS}')
        ),
        categorized AS (
            SELECT
                *,
                CASE
                    WHEN regexp_matches(slug_l, '^(btc|eth|sol)-updown-4h-[0-9]+')
                        THEN 'crypto_4h'
                    WHEN (
                        slug_l LIKE '%bitcoin%' OR slug_l LIKE '%ethereum%' OR slug_l LIKE '%solana%'
                        OR slug_l LIKE '%btc%' OR slug_l LIKE '%eth%' OR slug_l LIKE '%sol-%'
                        OR question_l LIKE '%bitcoin%' OR question_l LIKE '%ethereum%' OR question_l LIKE '%solana%'
                    )
                    AND (
                        slug_l LIKE '%up-or-down%' OR slug_l LIKE '%above%' OR slug_l LIKE '%below%'
                        OR question_l LIKE '%up or down%' OR question_l LIKE '%above%' OR question_l LIKE '%below%'
                    )
                        THEN 'daily_crypto'
                    WHEN (
                        slug_l LIKE '%iran%' OR slug_l LIKE '%israel%' OR slug_l LIKE '%ukraine%'
                        OR slug_l LIKE '%russia%' OR slug_l LIKE '%gaza%' OR slug_l LIKE '%ceasefire%'
                        OR slug_l LIKE '%nuclear%' OR slug_l LIKE '%taiwan%' OR slug_l LIKE '%china%'
                        OR slug_l LIKE '%war%' OR slug_l LIKE '%hormuz%' OR question_l LIKE '%iran%'
                        OR question_l LIKE '%israel%' OR question_l LIKE '%ukraine%' OR question_l LIKE '%russia%'
                        OR question_l LIKE '%gaza%' OR question_l LIKE '%ceasefire%' OR question_l LIKE '%nuclear%'
                        OR question_l LIKE '%taiwan%' OR question_l LIKE '%china%' OR question_l LIKE '%war%'
                    )
                        THEN 'geopolitics'
                    WHEN (
                        slug_l LIKE '%nba%' OR slug_l LIKE '%nfl%' OR slug_l LIKE '%nhl%' OR slug_l LIKE '%mlb%'
                        OR slug_l LIKE '%ufc%' OR slug_l LIKE '%soccer%' OR slug_l LIKE '%champions-league%'
                        OR slug_l LIKE '%premier-league%' OR question_l LIKE '% win the game%'
                        OR question_l LIKE '%beat the%' OR question_l LIKE '%score%'
                        OR question_l LIKE '%points%' OR question_l LIKE '%goals%'
                    )
                        THEN 'sports'
                    WHEN neg_risk AND (
                        slug_l LIKE '%election%' OR slug_l LIKE '%president%' OR slug_l LIKE '%senate%'
                        OR slug_l LIKE '%congress%' OR slug_l LIKE '%trump%' OR slug_l LIKE '%biden%'
                        OR slug_l LIKE '%democrat%' OR slug_l LIKE '%republican%'
                        OR question_l LIKE '%election%' OR question_l LIKE '%president%' OR question_l LIKE '%senate%'
                        OR question_l LIKE '%congress%' OR question_l LIKE '%trump%' OR question_l LIKE '%biden%'
                    )
                        THEN 'politics_negrisk'
                    WHEN (
                        slug_l LIKE '%fed%' OR slug_l LIKE '%inflation%' OR slug_l LIKE '%recession%'
                        OR slug_l LIKE '%cpi%' OR slug_l LIKE '%gdp%' OR question_l LIKE '%inflation%'
                        OR question_l LIKE '%recession%' OR question_l LIKE '%federal reserve%'
                    )
                        THEN 'economics'
                    WHEN (
                        slug_l LIKE '%weather%' OR slug_l LIKE '%temperature%' OR slug_l LIKE '%hurricane%'
                        OR slug_l LIKE '%rain%' OR slug_l LIKE '%snow%' OR question_l LIKE '%weather%'
                        OR question_l LIKE '%temperature%' OR question_l LIKE '%hurricane%'
                    )
                        THEN 'weather'
                    WHEN (
                        slug_l LIKE '%stock%' OR slug_l LIKE '%nasdaq%' OR slug_l LIKE '%s&p%'
                        OR slug_l LIKE '%dow%' OR slug_l LIKE '%earnings%' OR slug_l LIKE '%ipo%'
                        OR question_l LIKE '%stock%' OR question_l LIKE '%earnings%'
                    )
                        THEN 'finance'
                    WHEN (
                        slug_l LIKE '%grammy%' OR slug_l LIKE '%oscar%' OR slug_l LIKE '%movie%'
                        OR slug_l LIKE '%album%' OR slug_l LIKE '%music%' OR question_l LIKE '%oscar%'
                        OR question_l LIKE '%grammy%' OR question_l LIKE '%movie%'
                    )
                        THEN 'culture'
                    WHEN (
                        slug_l LIKE '%openai%' OR slug_l LIKE '%ai%' OR slug_l LIKE '%nvidia%'
                        OR slug_l LIKE '%tesla%' OR slug_l LIKE '%spacex%' OR slug_l LIKE '%apple%'
                        OR slug_l LIKE '%iphone%' OR slug_l LIKE '%google%' OR slug_l LIKE '%meta%'
                        OR question_l LIKE '%openai%' OR question_l LIKE '%nvidia%' OR question_l LIKE '%tesla%'
                        OR question_l LIKE '%spacex%' OR question_l LIKE '%iphone%'
                    )
                        THEN 'tech'
                    ELSE 'other'
                END AS category,
                CASE
                    WHEN regexp_matches(slug_l, '^(btc|eth|sol)-updown-4h-[0-9]+') THEN 'Crypto'
                    WHEN (
                        slug_l LIKE '%bitcoin%' OR slug_l LIKE '%ethereum%' OR slug_l LIKE '%solana%'
                        OR slug_l LIKE '%btc%' OR slug_l LIKE '%eth%'
                    ) THEN 'Crypto'
                    WHEN (
                        slug_l LIKE '%iran%' OR slug_l LIKE '%israel%' OR slug_l LIKE '%ukraine%'
                        OR slug_l LIKE '%russia%' OR slug_l LIKE '%gaza%' OR slug_l LIKE '%ceasefire%'
                        OR slug_l LIKE '%nuclear%' OR slug_l LIKE '%taiwan%' OR slug_l LIKE '%china%'
                        OR slug_l LIKE '%war%' OR slug_l LIKE '%hormuz%'
                    ) THEN 'Geopolitics'
                    WHEN (
                        slug_l LIKE '%nba%' OR slug_l LIKE '%nfl%' OR slug_l LIKE '%nhl%' OR slug_l LIKE '%mlb%'
                        OR slug_l LIKE '%ufc%' OR slug_l LIKE '%soccer%' OR slug_l LIKE '%league%'
                    ) THEN 'Sports'
                    WHEN neg_risk THEN 'Politics'
                    WHEN (
                        slug_l LIKE '%stock%' OR slug_l LIKE '%nasdaq%' OR slug_l LIKE '%earnings%'
                        OR slug_l LIKE '%ipo%'
                    ) THEN 'Finance'
                    WHEN (
                        slug_l LIKE '%fed%' OR slug_l LIKE '%inflation%' OR slug_l LIKE '%recession%'
                        OR slug_l LIKE '%cpi%' OR slug_l LIKE '%gdp%'
                    ) THEN 'Economics'
                    WHEN (
                        slug_l LIKE '%weather%' OR slug_l LIKE '%temperature%' OR slug_l LIKE '%hurricane%'
                    ) THEN 'Weather'
                    WHEN (
                        slug_l LIKE '%grammy%' OR slug_l LIKE '%oscar%' OR slug_l LIKE '%movie%'
                        OR slug_l LIKE '%music%'
                    ) THEN 'Culture'
                    WHEN (
                        slug_l LIKE '%openai%' OR slug_l LIKE '%ai%' OR slug_l LIKE '%nvidia%'
                        OR slug_l LIKE '%tesla%' OR slug_l LIKE '%spacex%' OR slug_l LIKE '%apple%'
                        OR slug_l LIKE '%iphone%' OR slug_l LIKE '%google%' OR slug_l LIKE '%meta%'
                    ) THEN 'Tech'
                    ELSE 'Other'
                END AS fee_family
            FROM base
        )
        SELECT
            *,
            CASE {cases_rate} ELSE 0.05 END AS fee_rate,
            CASE {cases_rebate} ELSE 0.25 END AS maker_rebate_share
        FROM categorized
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE market_tokens AS
        SELECT
            CAST(m.id AS VARCHAR) AS market_id,
            CAST(m.condition_id AS VARCHAR) AS condition_id,
            r.i AS outcome_index,
            CAST(m.clob_token_ids[r.i] AS VARCHAR) AS outcome_token_id,
            coalesce(m.outcomes[r.i], '') AS outcome_label,
            TRY_CAST(m.outcome_prices[r.i] AS DOUBLE) AS resolution_price
        FROM read_parquet('{MARKETS}') m,
             range(1, len(m.clob_token_ids) + 1) AS r(i)
        WHERE len(m.clob_token_ids) > 0
        """
    )


def identify_full_maker_population(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE maker_counts_all AS
        SELECT
            lower(maker) AS address,
            count(*) AS maker_fills,
            sum(usd_amount) AS maker_usd
        FROM read_parquet('{TRADES_GLOB}')
        WHERE maker IS NOT NULL
          AND taker IS NOT NULL
          AND maker <> taker
          AND market_id IS NOT NULL
          AND lower(maker) NOT IN ({internals})
          AND lower(taker) NOT IN ({internals})
        GROUP BY 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE taker_counts_all AS
        SELECT
            lower(taker) AS address,
            count(*) AS taker_fills,
            sum(usd_amount) AS taker_usd
        FROM read_parquet('{TRADES_GLOB}')
        WHERE maker IS NOT NULL
          AND taker IS NOT NULL
          AND maker <> taker
          AND market_id IS NOT NULL
          AND lower(maker) NOT IN ({internals})
          AND lower(taker) NOT IN ({internals})
        GROUP BY 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE qualified_makers AS
        SELECT
            coalesce(m.address, t.address) AS address,
            coalesce(m.maker_fills, 0) AS maker_fills,
            coalesce(t.taker_fills, 0) AS taker_fills,
            coalesce(m.maker_usd, 0.0) AS maker_usd,
            coalesce(t.taker_usd, 0.0) AS taker_usd,
            coalesce(m.maker_fills, 0)::DOUBLE
                / nullif(coalesce(m.maker_fills, 0) + coalesce(t.taker_fills, 0), 0) AS maker_share
        FROM maker_counts_all m
        FULL OUTER JOIN taker_counts_all t USING (address)
        WHERE coalesce(m.maker_fills, 0) >= {MIN_PASSIVE_FILLS}
          AND coalesce(m.maker_fills, 0)::DOUBLE
              / nullif(coalesce(m.maker_fills, 0) + coalesce(t.taker_fills, 0), 0)
              >= {MAKER_SHARE_CUTOFF}
        """
    )
    return con.execute("SELECT * FROM qualified_makers ORDER BY maker_usd DESC").df()


def build_fee_behavior_and_capacity(con: duckdb.DuckDBPyConnection) -> None:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE wallet_market_fees AS
        WITH maker_fee AS (
            SELECT
                lower(rt.maker) AS address,
                CAST(rt.market_id AS VARCHAR) AS market_id,
                coalesce(mm.category, 'unknown') AS category,
                count(*) AS raw_fill_count,
                count(*) AS raw_maker_fill_count,
                0 AS raw_taker_fill_count,
                sum(rt.usd_amount) AS raw_usd_volume,
                sum(rt.usd_amount) AS raw_maker_usd,
                0.0 AS raw_taker_usd,
                sum(rt.token_amount * coalesce(mm.fee_rate, 0.05)
                    * least(greatest(rt.price, 0.001), 0.999)
                    * (1.0 - least(greatest(rt.price, 0.001), 0.999))
                    * coalesce(mm.maker_rebate_share, 0.25)) AS maker_rebate_usd,
                0.0 AS taker_fee_usd
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN qualified_makers q ON lower(rt.maker) = q.address
            LEFT JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
            GROUP BY 1, 2, 3
        ),
        taker_fee AS (
            SELECT
                lower(rt.taker) AS address,
                CAST(rt.market_id AS VARCHAR) AS market_id,
                coalesce(mm.category, 'unknown') AS category,
                count(*) AS raw_fill_count,
                0 AS raw_maker_fill_count,
                count(*) AS raw_taker_fill_count,
                sum(rt.usd_amount) AS raw_usd_volume,
                0.0 AS raw_maker_usd,
                sum(rt.usd_amount) AS raw_taker_usd,
                0.0 AS maker_rebate_usd,
                sum(rt.token_amount * coalesce(mm.fee_rate, 0.05)
                    * least(greatest(rt.price, 0.001), 0.999)
                    * (1.0 - least(greatest(rt.price, 0.001), 0.999))) AS taker_fee_usd
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN qualified_makers q ON lower(rt.taker) = q.address
            LEFT JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
            GROUP BY 1, 2, 3
        )
        SELECT
            address,
            market_id,
            any_value(category) AS category,
            sum(raw_fill_count) AS raw_fill_count,
            sum(raw_maker_fill_count) AS raw_maker_fill_count,
            sum(raw_taker_fill_count) AS raw_taker_fill_count,
            sum(raw_usd_volume) AS raw_usd_volume,
            sum(raw_maker_usd) AS raw_maker_usd,
            sum(raw_taker_usd) AS raw_taker_usd,
            sum(maker_rebate_usd) AS maker_rebate_usd,
            sum(taker_fee_usd) AS taker_fee_usd,
            sum(maker_rebate_usd) - sum(taker_fee_usd) AS fee_adjustment_usd
        FROM (
            SELECT * FROM maker_fee
            UNION ALL
            SELECT * FROM taker_fee
        )
        GROUP BY 1, 2
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE maker_behavior AS
        SELECT
            lower(rt.maker) AS address,
            CAST(rt.market_id AS VARCHAR) AS market_id,
            coalesce(mm.category, 'unknown') AS category,
            count(*) AS maker_fills,
            sum(rt.usd_amount) AS maker_usd,
            count(DISTINCT CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END)
                AS distinct_outcomes_made,
            sum(CASE WHEN rt.maker_asset_id = '0' THEN rt.usd_amount ELSE 0 END) AS maker_buy_usd,
            sum(CASE WHEN rt.maker_asset_id <> '0' THEN rt.usd_amount ELSE 0 END) AS maker_sell_usd,
            sum(
                CASE
                    WHEN coalesce(mm.category, 'unknown') = 'crypto_4h'
                     AND regexp_matches(mm.slug_l, 'updown-4h-[0-9]+')
                     AND TRY_CAST(regexp_extract(mm.slug_l, 'updown-4h-([0-9]+)', 1) AS BIGINT) IS NOT NULL
                     AND epoch(rt.timestamp) >= TRY_CAST(regexp_extract(mm.slug_l, 'updown-4h-([0-9]+)', 1) AS BIGINT) + 3.75 * 3600
                     AND rt.price BETWEEN 0.40 AND 0.60
                    THEN rt.usd_amount ELSE 0
                END
            ) AS spike_zone_usd,
            min(rt.timestamp) AS first_maker_fill_ts,
            max(rt.timestamp) AS last_maker_fill_ts
        FROM read_parquet('{TRADES_GLOB}') rt
        JOIN qualified_makers q ON lower(rt.maker) = q.address
        LEFT JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
        WHERE rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND rt.market_id IS NOT NULL
          AND lower(rt.maker) NOT IN ({internals})
          AND lower(rt.taker) NOT IN ({internals})
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE global_market_maker_rank AS
        WITH by_wallet AS (
            SELECT
                CAST(rt.market_id AS VARCHAR) AS market_id,
                lower(rt.maker) AS address,
                sum(rt.usd_amount) AS global_maker_usd,
                count(*) AS global_maker_fills
            FROM read_parquet('{TRADES_GLOB}') rt
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
              AND CAST(rt.market_id AS VARCHAR) IN (SELECT DISTINCT market_id FROM wallet_market_fees)
            GROUP BY 1, 2
        )
        SELECT
            *,
            row_number() OVER (PARTITION BY market_id ORDER BY global_maker_usd DESC) AS global_market_maker_rank
        FROM by_wallet
        """
    )


def build_marked_pnl_tables(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        """
        CREATE OR REPLACE TABLE closed_wallet_market AS
        SELECT
            lower(cp.address) AS address,
            CAST(cp.market_id AS VARCHAR) AS market_id,
            coalesce(mm.category, 'unknown') AS category,
            min(cp.first_fill_ts) AS first_fill_ts,
            max(cp.last_fill_ts) AS last_fill_ts,
            max(cp.resolution_ts) AS mark_ts,
            sum(cp.n_fills) AS position_fills,
            sum(cp.gross_token_volume) AS gross_token_volume,
            sum(cp.gross_usd_volume) AS gross_usd_volume,
            sum(cp.realised_pnl) AS base_pnl_usd,
            sum(abs(cp.final_token_position)) AS abs_final_token_position,
            count(*) AS position_rows,
            'settlement' AS mark_source
        FROM read_parquet(?) cp
        JOIN qualified_makers q ON lower(cp.address) = q.address
        LEFT JOIN market_meta mm ON CAST(cp.market_id AS VARCHAR) = mm.market_id
        GROUP BY 1, 2, 3
        """,
        [str(CLOSED_POSITIONS)],
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE open_last_price AS
        WITH ranked AS (
            SELECT
                CAST(rt.market_id AS VARCHAR) AS market_id,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                    AS outcome_token_id,
                rt.price,
                rt.timestamp,
                row_number() OVER (
                    PARTITION BY
                        CAST(rt.market_id AS VARCHAR),
                        CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                    ORDER BY rt.timestamp DESC
                ) AS rn
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            WHERE (NOT mm.closed OR mm.closed IS NULL)
              AND rt.market_id IS NOT NULL
              AND rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
        )
        SELECT market_id, outcome_token_id, price AS mark_price, timestamp AS mark_ts
        FROM ranked
        WHERE rn = 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE open_position_token AS
        WITH actions AS (
            SELECT
                lower(rt.maker) AS address,
                CAST(rt.market_id AS VARCHAR) AS market_id,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                    AS outcome_token_id,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.token_amount ELSE -rt.token_amount END
                    AS token_delta,
                CASE WHEN rt.maker_asset_id = '0' THEN -rt.usd_amount ELSE rt.usd_amount END
                    AS usd_delta,
                rt.token_amount,
                rt.usd_amount,
                rt.timestamp
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN qualified_makers q ON lower(rt.maker) = q.address
            JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            WHERE (NOT mm.closed OR mm.closed IS NULL)
              AND rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
            UNION ALL
            SELECT
                lower(rt.taker) AS address,
                CAST(rt.market_id AS VARCHAR) AS market_id,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                    AS outcome_token_id,
                CASE WHEN rt.maker_asset_id = '0' THEN -rt.token_amount ELSE rt.token_amount END
                    AS token_delta,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.usd_amount ELSE -rt.usd_amount END
                    AS usd_delta,
                rt.token_amount,
                rt.usd_amount,
                rt.timestamp
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN qualified_makers q ON lower(rt.taker) = q.address
            JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            WHERE (NOT mm.closed OR mm.closed IS NULL)
              AND rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
        ),
        agg AS (
            SELECT
                address,
                market_id,
                outcome_token_id,
                sum(token_delta) AS final_token_position,
                sum(usd_delta) AS realised_cash_flow,
                sum(token_amount) AS gross_token_volume,
                sum(usd_amount) AS gross_usd_volume,
                count(*) AS n_fills,
                min(timestamp) AS first_fill_ts,
                max(timestamp) AS last_fill_ts
            FROM actions
            GROUP BY 1, 2, 3
        )
        SELECT
            a.address,
            a.market_id,
            coalesce(mm.category, 'unknown') AS category,
            a.outcome_token_id,
            a.first_fill_ts,
            a.last_fill_ts,
            lp.mark_ts,
            a.n_fills,
            a.gross_token_volume,
            a.gross_usd_volume,
            a.realised_cash_flow,
            a.final_token_position,
            lp.mark_price,
            a.realised_cash_flow + a.final_token_position * coalesce(lp.mark_price, 0.0)
                AS marked_pnl_usd,
            abs(a.final_token_position) AS abs_final_token_position
        FROM agg a
        LEFT JOIN market_meta mm USING (market_id)
        LEFT JOIN open_last_price lp
          ON a.market_id = lp.market_id AND a.outcome_token_id = lp.outcome_token_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE open_wallet_market AS
        SELECT
            address,
            market_id,
            category,
            min(first_fill_ts) AS first_fill_ts,
            max(last_fill_ts) AS last_fill_ts,
            max(mark_ts) AS mark_ts,
            sum(n_fills) AS position_fills,
            sum(gross_token_volume) AS gross_token_volume,
            sum(gross_usd_volume) AS gross_usd_volume,
            sum(marked_pnl_usd) AS base_pnl_usd,
            sum(abs_final_token_position) AS abs_final_token_position,
            count(*) AS position_rows,
            'last_trade_proxy' AS mark_source
        FROM open_position_token
        GROUP BY 1, 2, 3
        HAVING gross_token_volume > 0
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE wallet_market_base AS
        SELECT * FROM closed_wallet_market
        UNION ALL BY NAME
        SELECT * FROM open_wallet_market
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE wallet_market_full AS
        SELECT
            b.address,
            b.market_id,
            b.category,
            min(b.first_fill_ts) AS first_fill_ts,
            max(b.last_fill_ts) AS last_fill_ts,
            max(b.mark_ts) AS mark_ts,
            sum(b.position_fills) AS position_fills,
            sum(b.gross_token_volume) AS gross_token_volume,
            sum(b.gross_usd_volume) AS gross_usd_volume,
            sum(b.base_pnl_usd) AS base_pnl_usd,
            sum(b.abs_final_token_position) AS abs_final_token_position,
            sum(b.position_rows) AS position_rows,
            string_agg(DISTINCT b.mark_source, '+') AS mark_source,
            coalesce(f.raw_fill_count, 0) AS raw_fill_count,
            coalesce(f.raw_maker_fill_count, 0) AS raw_maker_fill_count,
            coalesce(f.raw_taker_fill_count, 0) AS raw_taker_fill_count,
            coalesce(f.raw_usd_volume, 0.0) AS raw_usd_volume,
            coalesce(f.raw_maker_usd, 0.0) AS raw_maker_usd,
            coalesce(f.raw_taker_usd, 0.0) AS raw_taker_usd,
            coalesce(f.maker_rebate_usd, 0.0) AS maker_rebate_usd,
            coalesce(f.taker_fee_usd, 0.0) AS taker_fee_usd,
            coalesce(f.fee_adjustment_usd, 0.0) AS fee_adjustment_usd,
            sum(b.base_pnl_usd) + coalesce(f.fee_adjustment_usd, 0.0) AS net_pnl_usd,
            coalesce(mb.maker_fills, 0) AS maker_fills,
            coalesce(mb.maker_usd, 0.0) AS maker_usd,
            coalesce(mb.distinct_outcomes_made, 0) AS distinct_outcomes_made,
            coalesce(mb.maker_buy_usd, 0.0) AS maker_buy_usd,
            coalesce(mb.maker_sell_usd, 0.0) AS maker_sell_usd,
            coalesce(mb.spike_zone_usd, 0.0) AS spike_zone_usd,
            coalesce(gmr.global_maker_usd, 0.0) AS global_maker_usd,
            coalesce(gmr.global_maker_fills, 0) AS global_maker_fills,
            coalesce(gmr.global_market_maker_rank, 999999) AS global_market_maker_rank,
            coalesce(gmr.global_market_maker_rank, 999999) <= 3 AS is_global_top3_market_maker
        FROM wallet_market_base b
        LEFT JOIN wallet_market_fees f
          ON b.address = f.address AND b.market_id = f.market_id
        LEFT JOIN maker_behavior mb
          ON b.address = mb.address AND b.market_id = mb.market_id
        LEFT JOIN global_market_maker_rank gmr
          ON b.address = gmr.address AND b.market_id = gmr.market_id
        GROUP BY
            b.address, b.market_id, b.category,
            f.raw_fill_count, f.raw_maker_fill_count, f.raw_taker_fill_count,
            f.raw_usd_volume, f.raw_maker_usd, f.raw_taker_usd,
            f.maker_rebate_usd, f.taker_fee_usd, f.fee_adjustment_usd,
            mb.maker_fills, mb.maker_usd, mb.distinct_outcomes_made,
            mb.maker_buy_usd, mb.maker_sell_usd, mb.spike_zone_usd,
            gmr.global_maker_usd, gmr.global_maker_fills, gmr.global_market_maker_rank
        """
    )
    return con.execute("SELECT * FROM wallet_market_full").df()


def build_markouts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE maker_fill_markout_sample AS
        SELECT
            row_number() OVER () AS fill_id,
            rt.timestamp AS fill_ts,
            lower(rt.maker) AS address,
            CAST(rt.market_id AS VARCHAR) AS market_id,
            coalesce(mm.category, 'unknown') AS category,
            CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                AS outcome_token_id,
            CASE WHEN rt.maker_asset_id = '0' THEN 1.0 ELSE -1.0 END AS token_side,
            rt.price,
            rt.usd_amount,
            rt.token_amount,
            mt.resolution_price
        FROM read_parquet('{TRADES_GLOB}') rt
        JOIN qualified_makers q ON lower(rt.maker) = q.address
        LEFT JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
        LEFT JOIN market_tokens mt
          ON CAST(rt.market_id AS VARCHAR) = mt.market_id
         AND (CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END)
             = mt.outcome_token_id
        WHERE rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND rt.market_id IS NOT NULL
          AND lower(rt.maker) NOT IN ({internals})
          AND lower(rt.taker) NOT IN ({internals})
          AND (
              coalesce(mm.category, 'unknown') IN ('crypto_4h', 'culture')
              OR (
                  abs(hash(CAST(rt.market_id AS VARCHAR)))
                      % {MARKOUT_MARKET_HASH_MOD} < {MARKOUT_MARKET_HASH_KEEP}
                  AND abs(hash(coalesce(rt.transaction_hash, ''), lower(rt.maker), CAST(rt.market_id AS VARCHAR)))
                      % {MARKOUT_HASH_MOD} < {MARKOUT_HASH_KEEP}
              )
          )
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE markout_price_stream AS
        SELECT
            CAST(rt.market_id AS VARCHAR) AS market_id,
            CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                AS outcome_token_id,
            rt.timestamp,
            avg(rt.price) AS price
        FROM read_parquet('{TRADES_GLOB}') rt
        WHERE CAST(rt.market_id AS VARCHAR) IN (
            SELECT DISTINCT market_id FROM maker_fill_markout_sample
        )
          AND rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND rt.market_id IS NOT NULL
          AND lower(rt.maker) NOT IN ({internals})
          AND lower(rt.taker) NOT IN ({internals})
        GROUP BY 1, 2, 3
        """
    )
    for sec in (5, 30, 60):
        con.execute(
            f"""
            CREATE OR REPLACE TABLE markout_{sec}s AS
            SELECT
                f.fill_id,
                p.timestamp AS future_ts_{sec}s,
                p.price AS future_price_{sec}s
            FROM (
                SELECT
                    fill_id,
                    market_id,
                    outcome_token_id,
                    fill_ts + INTERVAL {sec} SECOND AS target_ts
                FROM maker_fill_markout_sample
            ) f
            ASOF LEFT JOIN markout_price_stream p
              ON f.market_id = p.market_id
             AND f.outcome_token_id = p.outcome_token_id
             AND f.target_ts <= p.timestamp
            """
        )
    con.execute(
        """
        CREATE OR REPLACE TABLE markout_joined AS
        SELECT
            f.*,
            m5.future_ts_5s,
            m5.future_price_5s,
            m30.future_ts_30s,
            m30.future_price_30s,
            m60.future_ts_60s,
            m60.future_price_60s,
            epoch(m5.future_ts_5s - (f.fill_ts + INTERVAL 5 SECOND)) AS lag_5s,
            epoch(m30.future_ts_30s - (f.fill_ts + INTERVAL 30 SECOND)) AS lag_30s,
            epoch(m60.future_ts_60s - (f.fill_ts + INTERVAL 60 SECOND)) AS lag_60s
        FROM maker_fill_markout_sample f
        LEFT JOIN markout_5s m5 USING (fill_id)
        LEFT JOIN markout_30s m30 USING (fill_id)
        LEFT JOIN markout_60s m60 USING (fill_id)
        """
    )
    return con.execute("SELECT * FROM markout_joined").df()


def build_calibration(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    horizon_values = ", ".join(
        f"('{label}', {sec}, {max_age})" for label, sec, max_age in HORIZONS
    )
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE binary_market_meta AS
        SELECT *
        FROM market_meta
        WHERE closed
          AND n_tokens = 2
          AND end_ts >= TIMESTAMP '{DATA_START}'
          AND end_ts <= TIMESTAMP '{DATA_END}'
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE binary_market_tokens AS
        SELECT mt.*
        FROM market_tokens mt
        JOIN binary_market_meta bm USING (market_id)
        WHERE mt.resolution_price IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE calibration_obs AS
        WITH candidates AS (
            SELECT
                CAST(rt.market_id AS VARCHAR) AS market_id,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                    AS outcome_token_id,
                bm.category,
                bm.fee_family,
                bm.fee_rate,
                bm.maker_rebate_share,
                mt.resolution_price,
                h.horizon_label,
                h.horizon_sec,
                bm.end_ts - h.horizon_sec * INTERVAL 1 SECOND AS target_ts,
                rt.timestamp AS price_ts,
                epoch((bm.end_ts - h.horizon_sec * INTERVAL 1 SECOND) - rt.timestamp)
                    AS price_age_sec,
                rt.price,
                row_number() OVER (
                    PARTITION BY
                        CAST(rt.market_id AS VARCHAR),
                        CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END,
                        h.horizon_label
                    ORDER BY rt.timestamp DESC
                ) AS rn
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN binary_market_meta bm ON CAST(rt.market_id AS VARCHAR) = bm.market_id
            JOIN binary_market_tokens mt
              ON CAST(rt.market_id AS VARCHAR) = mt.market_id
             AND (CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END)
                 = mt.outcome_token_id
            CROSS JOIN (VALUES {horizon_values}) AS h(horizon_label, horizon_sec, max_age_sec)
            WHERE rt.timestamp >= TIMESTAMP '{DATA_START}'
              AND rt.timestamp <= TIMESTAMP '{DATA_END}'
              AND rt.timestamp <= bm.end_ts
              AND rt.timestamp <= bm.end_ts - h.horizon_sec * INTERVAL 1 SECOND
              AND rt.timestamp >= bm.end_ts - (h.horizon_sec + h.max_age_sec) * INTERVAL 1 SECOND
              AND bm.end_ts - h.horizon_sec * INTERVAL 1 SECOND >= TIMESTAMP '{DATA_START}'
              AND rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
              AND rt.price BETWEEN 0.0 AND 1.0
              AND rt.token_amount > 0
              AND rt.usd_amount > 0
        )
        SELECT
            market_id,
            outcome_token_id,
            category,
            fee_family,
            fee_rate,
            maker_rebate_share,
            resolution_price,
            horizon_label,
            horizon_sec,
            target_ts,
            price_ts,
            price_age_sec,
            price,
            price - resolution_price AS longshot_gap,
            price - resolution_price
                + fee_rate * price * (1.0 - price) * maker_rebate_share AS maker_net_gap,
            CASE
                WHEN price < 0.01 THEN '00_01c'
                WHEN price < 0.02 THEN '01_02c'
                WHEN price < 0.05 THEN '02_05c'
                WHEN price < 0.10 THEN '05_10c'
                WHEN price < 0.20 THEN '10_20c'
                WHEN price < 0.35 THEN '20_35c'
                WHEN price < 0.65 THEN '35_65c'
                WHEN price < 0.80 THEN '65_80c'
                WHEN price < 0.90 THEN '80_90c'
                WHEN price < 0.95 THEN '90_95c'
                ELSE '95_100c'
            END AS price_bucket,
            price BETWEEN {TAIL_MIN_PRICE} AND {TAIL_MAX_PRICE} AS is_tail
        FROM candidates
        WHERE rn = 1
        """
    )
    return con.execute("SELECT * FROM calibration_obs").df()


def ratio_ci(
    df: pd.DataFrame,
    num_col: str = "net_pnl_usd",
    den_col: str = "gross_usd_volume",
    block_col: str = "market_id",
    multiplier: float = 10_000.0,
) -> tuple[float, float]:
    d = df[[block_col, num_col, den_col]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d[den_col].gt(0)]
    if d.empty:
        return math.nan, math.nan
    blocks = d.groupby(block_col, sort=False)[[num_col, den_col]].sum().reset_index()
    if len(blocks) < 2:
        return math.nan, math.nan
    vals = blocks[[num_col, den_col]].to_numpy(float)
    rng = np.random.default_rng(RNG_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(vals), len(vals))
        den = vals[idx, 1].sum()
        estimates.append(vals[idx, 0].sum() / den * multiplier if den > 0 else math.nan)
    lo, hi = np.nanquantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_pnl(piece: pd.DataFrame, section: str, scope: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    gross = float(piece["gross_usd_volume"].sum())
    net = float(piece["net_pnl_usd"].sum())
    ci_lo, ci_hi = ratio_ci(piece)
    wallet = (
        piece.groupby("address", as_index=False)
        .agg(net_pnl_usd=("net_pnl_usd", "sum"), gross_usd_volume=("gross_usd_volume", "sum"))
    )
    wallet["net_pnl_bps"] = wallet["net_pnl_usd"] / wallet["gross_usd_volume"].replace(0, np.nan) * 10_000
    row: dict[str, Any] = {
        "section": section,
        "scope": scope,
        "wallets": int(piece["address"].nunique()),
        "markets": int(piece["market_id"].nunique()),
        "gross_usd_volume": gross,
        "net_pnl_usd": net,
        "net_pnl_bps": net / gross * 10_000 if gross > 0 else math.nan,
        "ci_lo_bps": ci_lo,
        "ci_hi_bps": ci_hi,
        "wallet_mean_net_pnl_bps": float(wallet["net_pnl_bps"].mean()) if not wallet.empty else math.nan,
        "wallet_median_net_pnl_bps": float(wallet["net_pnl_bps"].median()) if not wallet.empty else math.nan,
        "wallet_q25_net_pnl_bps": float(wallet["net_pnl_bps"].quantile(0.25)) if not wallet.empty else math.nan,
        "wallet_q75_net_pnl_bps": float(wallet["net_pnl_bps"].quantile(0.75)) if not wallet.empty else math.nan,
        "wallet_median_net_pnl_usd": float(wallet["net_pnl_usd"].median()) if not wallet.empty else math.nan,
        "base_pnl_usd": float(piece["base_pnl_usd"].sum()),
        "base_pnl_bps": float(piece["base_pnl_usd"].sum() / gross * 10_000) if gross > 0 else math.nan,
        "maker_rebate_usd": float(piece["maker_rebate_usd"].sum()),
        "maker_rebate_bps": float(piece["maker_rebate_usd"].sum() / gross * 10_000) if gross > 0 else math.nan,
        "taker_fee_usd": float(piece["taker_fee_usd"].sum()),
        "taker_fee_bps": float(piece["taker_fee_usd"].sum() / gross * 10_000) if gross > 0 else math.nan,
        "net_without_rebate_usd": float(net - piece["maker_rebate_usd"].sum()),
        "net_without_rebate_bps": float((net - piece["maker_rebate_usd"].sum()) / gross * 10_000)
        if gross > 0
        else math.nan,
        "maker_fills": int(piece["maker_fills"].sum()),
        "maker_usd": float(piece["maker_usd"].sum()),
        "raw_maker_fill_count": int(piece["raw_maker_fill_count"].sum()),
        "resolution_carry_token_share": float(
            piece["abs_final_token_position"].sum() / piece["gross_token_volume"].sum()
        )
        if piece["gross_token_volume"].sum() > 0
        else math.nan,
        "two_sided_usd_share": float(
            piece.loc[piece["distinct_outcomes_made"].ge(2), "maker_usd"].sum() / piece["maker_usd"].sum()
        )
        if piece["maker_usd"].sum() > 0
        else math.nan,
        "spike_zone_usd_share": float(piece["spike_zone_usd"].sum() / piece["maker_usd"].sum())
        if piece["maker_usd"].sum() > 0
        else math.nan,
        "top3_market_maker_gross_share": float(
            piece.loc[piece["is_global_top3_market_maker"], "gross_usd_volume"].sum() / gross
        )
        if gross > 0
        else math.nan,
    }
    if extra:
        row.update(extra)
    return row


def build_wallet_category(wallet_market: pd.DataFrame) -> pd.DataFrame:
    wm = wallet_market.copy()
    grouped = wm.groupby(["address", "category"], as_index=False).agg(
        markets=("market_id", "nunique"),
        gross_usd_volume=("gross_usd_volume", "sum"),
        net_pnl_usd=("net_pnl_usd", "sum"),
        base_pnl_usd=("base_pnl_usd", "sum"),
        maker_rebate_usd=("maker_rebate_usd", "sum"),
        taker_fee_usd=("taker_fee_usd", "sum"),
        maker_fills=("maker_fills", "sum"),
        maker_usd=("maker_usd", "sum"),
        gross_token_volume=("gross_token_volume", "sum"),
        abs_final_token_position=("abs_final_token_position", "sum"),
        spike_zone_usd=("spike_zone_usd", "sum"),
    )
    two_sided = (
        wm.loc[wm["distinct_outcomes_made"].ge(2)]
        .groupby(["address", "category"])["maker_usd"]
        .sum()
        .rename("two_sided_maker_usd")
        .reset_index()
    )
    grouped = grouped.merge(two_sided, on=["address", "category"], how="left")
    grouped["two_sided_maker_usd"] = grouped["two_sided_maker_usd"].fillna(0.0)
    grouped["net_pnl_bps"] = grouped["net_pnl_usd"] / grouped["gross_usd_volume"].replace(0, np.nan) * 10_000
    grouped["two_sided_usd_share"] = grouped["two_sided_maker_usd"] / grouped["maker_usd"].replace(0, np.nan)
    grouped["carry_token_share"] = (
        grouped["abs_final_token_position"] / grouped["gross_token_volume"].replace(0, np.nan)
    )
    grouped["spike_zone_usd_share"] = grouped["spike_zone_usd"] / grouped["maker_usd"].replace(0, np.nan)
    grouped[["two_sided_usd_share", "carry_token_share", "spike_zone_usd_share"]] = grouped[
        ["two_sided_usd_share", "carry_token_share", "spike_zone_usd_share"]
    ].fillna(0.0)
    grouped["structured_playbook"] = (
        grouped["two_sided_usd_share"].ge(STRUCT_TWO_SIDED_MIN)
        & grouped["carry_token_share"].ge(STRUCT_CARRY_MIN)
        & grouped["spike_zone_usd_share"].le(STRUCT_SPIKE_MAX)
        & grouped["maker_fills"].gt(0)
    )
    return grouped


def summarize_outputs(
    makers: pd.DataFrame,
    wallet_market: pd.DataFrame,
    markouts: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "section": "maker_population",
            "scope": "ex_ante_rule",
            "wallets": len(makers),
            "maker_fills": float(makers["maker_fills"].sum()),
            "taker_fills": float(makers["taker_fills"].sum()),
            "maker_usd": float(makers["maker_usd"].sum()),
            "median_maker_share": float(makers["maker_share"].median()),
            "min_passive_fills": MIN_PASSIVE_FILLS,
            "maker_share_cutoff": MAKER_SHARE_CUTOFF,
        }
    )

    closed_piece = wallet_market[wallet_market["mark_source"].astype(str).str.contains("settlement", na=False)]
    full_piece = wallet_market
    rows.append(summarize_pnl(closed_piece, "survivorship_comparison", "closed_resolved_only"))
    rows.append(summarize_pnl(full_piece, "survivorship_comparison", "full_marked_resolved_plus_open"))
    if not closed_piece.empty and not full_piece.empty:
        closed_bps = rows[-2]["net_pnl_bps"]
        full_bps = rows[-1]["net_pnl_bps"]
        rows.append(
            {
                "section": "survivorship_gap",
                "scope": "full_minus_closed",
                "net_pnl_bps_gap": full_bps - closed_bps,
                "net_pnl_usd_gap": rows[-1]["net_pnl_usd"] - rows[-2]["net_pnl_usd"],
                "open_marked_gross_usd": rows[-1]["gross_usd_volume"] - rows[-2]["gross_usd_volume"],
            }
        )

    for category, piece in wallet_market.groupby("category", sort=False):
        rows.append(summarize_pnl(piece, "category_full_population", str(category), {"category": category}))

    wallet_category = build_wallet_category(wallet_market)
    wc_out = wallet_category.copy()
    wc_out.insert(0, "section", "wallet_category_distribution")
    wc_out.insert(1, "scope", wc_out["category"])
    rows.extend(wc_out.to_dict("records"))

    tagged = wallet_market.merge(
        wallet_category[["address", "category", "structured_playbook"]],
        on=["address", "category"],
        how="left",
    )
    for (category, structured), piece in tagged.groupby(["category", "structured_playbook"], sort=False):
        label = "structured" if bool(structured) else "unstructured"
        rows.append(
            summarize_pnl(
                piece,
                "structure_test",
                f"{category}:{label}",
                {"category": category, "structured_playbook": bool(structured)},
            )
        )
    non_top3 = tagged[~tagged["is_global_top3_market_maker"]].copy()
    for (category, structured), piece in non_top3.groupby(["category", "structured_playbook"], sort=False):
        label = "structured" if bool(structured) else "unstructured"
        rows.append(
            summarize_pnl(
                piece,
                "deployability_non_top3",
                f"{category}:{label}",
                {"category": category, "structured_playbook": bool(structured)},
            )
        )

    month = wallet_market.copy()
    month["month"] = pd.to_datetime(month["mark_ts"], errors="coerce").dt.to_period("M").astype(str)
    for (category, m), piece in month.groupby(["category", "month"], sort=False):
        rows.append(summarize_pnl(piece, "month_stability", f"{category}:{m}", {"category": category, "month": m}))

    if not markouts.empty:
        total_maker_fills = (
            wallet_market.groupby("category")["maker_fills"].sum().rename("category_total_maker_fills")
        )
        for category, piece in markouts.groupby("category", sort=False):
            denom = piece["price"].clip(lower=0.01, upper=0.99)
            total = float(total_maker_fills.get(category, math.nan))
            sample_fraction = (
                1.0
                if category in {"crypto_4h", "culture"}
                else float(len(piece) / total)
                if np.isfinite(total) and total > 0
                else math.nan
            )
            for sec in (5, 30, 60):
                lag = piece[f"lag_{sec}s"]
                valid = piece[f"future_price_{sec}s"].notna() & lag.ge(0) & lag.le(MARKOUT_MAX_LAG_SEC)
                mark = piece.loc[valid, "token_side"] * (
                    piece.loc[valid, f"future_price_{sec}s"] - piece.loc[valid, "price"]
                ) / denom.loc[valid] * 10_000
                rows.append(
                    {
                        "section": "markout_full_population_sample",
                        "scope": f"{category}:{sec}s",
                        "category": category,
                        "horizon_sec": sec,
                        "sample_fills": int(len(piece)),
                        "covered_fills": int(valid.sum()),
                        "sample_fraction_of_category_maker_fills": sample_fraction,
                        "mean_markout_bps": float(mark.mean()) if len(mark) else math.nan,
                        "median_markout_bps": float(mark.median()) if len(mark) else math.nan,
                        "mean_adverse_cost_bps": float(-mark.mean()) if len(mark) else math.nan,
                        "win_rate_markout_positive": float(mark.gt(0).mean()) if len(mark) else math.nan,
                        "mean_future_lag_sec": float(lag[valid].mean()) if valid.any() else math.nan,
                    }
                )
            valid_res = piece["resolution_price"].notna()
            res_mark = piece.loc[valid_res, "token_side"] * (
                piece.loc[valid_res, "resolution_price"] - piece.loc[valid_res, "price"]
            ) / denom.loc[valid_res] * 10_000
            rows.append(
                {
                    "section": "markout_full_population_sample",
                    "scope": f"{category}:resolution",
                    "category": category,
                    "horizon_sec": "resolution",
                    "sample_fills": int(len(piece)),
                    "covered_fills": int(valid_res.sum()),
                    "sample_fraction_of_category_maker_fills": sample_fraction,
                    "mean_markout_bps": float(res_mark.mean()) if len(res_mark) else math.nan,
                    "median_markout_bps": float(res_mark.median()) if len(res_mark) else math.nan,
                    "mean_adverse_cost_bps": float(-res_mark.mean()) if len(res_mark) else math.nan,
                    "win_rate_markout_positive": float(res_mark.gt(0).mean()) if len(res_mark) else math.nan,
                }
            )

    if not calibration.empty:
        bucket = (
            calibration.groupby(["category", "horizon_label", "price_bucket"], as_index=False)
            .agg(
                obs=("price", "size"),
                markets=("market_id", "nunique"),
                avg_price=("price", "mean"),
                realized_yes_rate=("resolution_price", "mean"),
                avg_longshot_gap=("longshot_gap", "mean"),
                avg_maker_net_gap=("maker_net_gap", "mean"),
            )
        )
        bucket.insert(0, "section", "calibration_bucket")
        bucket.insert(1, "scope", bucket["category"] + ":" + bucket["horizon_label"] + ":" + bucket["price_bucket"])
        rows.extend(bucket.to_dict("records"))
        tail = calibration[calibration["is_tail"]].copy()
        if not tail.empty:
            for category, piece in tail.groupby("category", sort=False):
                blocks = piece.groupby("market_id", as_index=False).agg(
                    price=("price", "mean"),
                    resolution_price=("resolution_price", "mean"),
                    maker_net_gap=("maker_net_gap", "mean"),
                    longshot_gap=("longshot_gap", "mean"),
                )
                rng = np.random.default_rng(RNG_SEED)
                vals = blocks[["longshot_gap", "maker_net_gap"]].to_numpy(float)
                boot_gap = []
                boot_net = []
                for _ in range(BOOTSTRAP_SAMPLES):
                    idx = rng.integers(0, len(vals), len(vals))
                    boot_gap.append(float(np.nanmean(vals[idx, 0])))
                    boot_net.append(float(np.nanmean(vals[idx, 1])))
                gap_lo, gap_hi = np.nanquantile(boot_gap, [0.025, 0.975])
                net_lo, net_hi = np.nanquantile(boot_net, [0.025, 0.975])
                rows.append(
                    {
                        "section": "calibration_tail_category",
                        "scope": category,
                        "category": category,
                        "obs": int(len(piece)),
                        "markets": int(piece["market_id"].nunique()),
                        "avg_price": float(piece["price"].mean()),
                        "realized_yes_rate": float(piece["resolution_price"].mean()),
                        "avg_longshot_gap": float(piece["longshot_gap"].mean()),
                        "gap_ci_lo": float(gap_lo),
                        "gap_ci_hi": float(gap_hi),
                        "avg_maker_net_gap": float(piece["maker_net_gap"].mean()),
                        "maker_net_gap_ci_lo": float(net_lo),
                        "maker_net_gap_ci_hi": float(net_hi),
                    }
                )

    # Category gate roll-up.
    out_so_far = pd.DataFrame(rows)
    category_rows = out_so_far[out_so_far["section"].eq("category_full_population")].copy()
    deploy = out_so_far[
        out_so_far["section"].eq("deployability_non_top3")
        & out_so_far["structured_playbook"].fillna(False).astype(bool)
    ].copy()
    months = out_so_far[out_so_far["section"].eq("month_stability")].copy()
    cal = out_so_far[out_so_far["section"].eq("calibration_tail_category")].copy()
    for _, r in category_rows.iterrows():
        category = str(r["category"])
        d = deploy[deploy["category"].eq(category)]
        drow = d.iloc[0] if not d.empty else pd.Series(dtype=object)
        m = months[months["category"].eq(category)]
        active_months = int(m["month"].nunique()) if not m.empty and "month" in m else 0
        pos_month_share = float(m["net_pnl_bps"].gt(0).mean()) if not m.empty else math.nan
        one_episode = bool(np.isfinite(pos_month_share) and pos_month_share < STABILITY_POS_MONTH_SHARE_MIN)
        calrow = cal[cal["category"].eq(category)]
        cal_net_lo = float(calrow["maker_net_gap_ci_lo"].iloc[0]) if not calrow.empty else math.nan
        full_pass = bool(float(r["ci_lo_bps"]) > 0 and float(r["net_pnl_bps"]) > 0)
        deploy_pass = bool(
            not drow.empty
            and float(drow.get("ci_lo_bps", math.nan)) > 0
            and float(drow.get("net_pnl_bps", math.nan)) > 0
        )
        structured_typical_bps = (
            float(drow.get("wallet_median_net_pnl_bps", math.nan)) if not drow.empty else math.nan
        )
        structured_typical_pass = bool(np.isfinite(structured_typical_bps) and structured_typical_bps > 0)
        rebate_pass = bool(float(r["net_without_rebate_bps"]) > 0)
        stability_pass = not one_episode
        calibration_pass = bool(np.isfinite(cal_net_lo) and cal_net_lo > 0)
        mechanical_pnl_gate_pass = full_pass and deploy_pass and rebate_pass and stability_pass
        negrisk_reliable_pass = category != "politics_negrisk"
        proceed = mechanical_pnl_gate_pass and structured_typical_pass and negrisk_reliable_pass
        rows.append(
            {
                "section": "gate_summary",
                "scope": category,
                "category": category,
                "mechanical_pnl_gate_pass": mechanical_pnl_gate_pass,
                "full_population_pass": full_pass,
                "structured_non_top3_pass": deploy_pass,
                "structured_non_top3_typical_pass": structured_typical_pass,
                "negrisk_reliable_pass": negrisk_reliable_pass,
                "not_rebate_propped_pass": rebate_pass,
                "stability_pass": stability_pass,
                "calibration_tail_pass": calibration_pass,
                "proceed_gate": proceed,
                "full_net_bps": float(r["net_pnl_bps"]),
                "full_ci_lo_bps": float(r["ci_lo_bps"]),
                "full_ci_hi_bps": float(r["ci_hi_bps"]),
                "typical_wallet_median_bps": float(r["wallet_median_net_pnl_bps"]),
                "structured_non_top3_bps": float(drow.get("net_pnl_bps", math.nan)) if not drow.empty else math.nan,
                "structured_non_top3_median_bps": structured_typical_bps,
                "structured_non_top3_ci_lo_bps": float(drow.get("ci_lo_bps", math.nan))
                if not drow.empty
                else math.nan,
                "structured_non_top3_ci_hi_bps": float(drow.get("ci_hi_bps", math.nan))
                if not drow.empty
                else math.nan,
                "net_without_rebate_bps": float(r["net_without_rebate_bps"]),
                "active_months": active_months,
                "positive_month_share": pos_month_share,
                "calibration_net_gap": float(calrow["avg_maker_net_gap"].iloc[0]) if not calrow.empty else math.nan,
                "calibration_net_gap_ci_lo": cal_net_lo,
                "calibration_net_gap_ci_hi": float(calrow["maker_net_gap_ci_hi"].iloc[0])
                if not calrow.empty
                else math.nan,
            }
        )
    rows.append(
        {
            "section": "negrisk_warning",
            "scope": "politics_negrisk",
            "category": "politics_negrisk",
            "reliable": False,
            "note": (
                "NegRisk merge/split/redemption mechanics are not in raw fill files; "
                "token-level trade PnL is reported, but linked-outcome settlement accounting "
                "remains unreliable for this gate."
            ),
        }
    )
    rows.append(
        {
            "section": "method_limit",
            "scope": "markout",
                "note": (
                f"Markouts use a deterministic full-population sample: "
                "all crypto_4h and culture maker fills, plus "
                f"market_hash % {MARKOUT_MARKET_HASH_MOD} < {MARKOUT_MARKET_HASH_KEEP} and "
                f"fill_hash % {MARKOUT_HASH_MOD} < {MARKOUT_HASH_KEEP} for other categories; "
                "no PnL/winner conditioning."
            ),
        }
    )
    rows.append(
        {
            "section": "method_limit",
            "scope": "open_inventory_mark",
            "note": "Open/unresolved inventory is marked to latest executed trade price, a last-mid proxy.",
        }
    )
    return pd.DataFrame(rows)


def write_note(output: pd.DataFrame) -> None:
    gates = output[output["section"].eq("gate_summary")].copy()
    gates = gates.sort_values(["proceed_gate", "full_net_bps"], ascending=[False, False])
    pop = output[output["section"].eq("maker_population")].head(1)
    surv = output[output["section"].eq("survivorship_comparison")].copy()
    gap = output[output["section"].eq("survivorship_gap")].head(1)
    markout = output[output["section"].eq("markout_full_population_sample")].copy()
    negrisk = output[output["section"].eq("negrisk_warning")].head(1)

    gate_rows = []
    for _, r in gates.iterrows():
        gate_rows.append(
            [
                str(r["category"]),
                "YES" if bool(r["proceed_gate"]) else "NO",
                bps(float(r["full_net_bps"])),
                f"[{bps(float(r['full_ci_lo_bps']))}, {bps(float(r['full_ci_hi_bps']))}]",
                bps(float(r["typical_wallet_median_bps"])),
                bps(float(r["structured_non_top3_bps"])),
                bps(float(r["structured_non_top3_median_bps"])),
                f"[{bps(float(r['structured_non_top3_ci_lo_bps']))}, {bps(float(r['structured_non_top3_ci_hi_bps']))}]",
                bps(float(r["net_without_rebate_bps"])),
                pct(float(r["positive_month_share"])),
                cents(float(r["calibration_net_gap"])),
            ]
        )

    surv_rows = []
    for _, r in surv.iterrows():
        surv_rows.append(
            [
                str(r["scope"]),
                str(int(r["wallets"])),
                str(int(r["markets"])),
                dollars(float(r["gross_usd_volume"])),
                dollars(float(r["net_pnl_usd"])),
                bps(float(r["net_pnl_bps"])),
                f"[{bps(float(r['ci_lo_bps']))}, {bps(float(r['ci_hi_bps']))}]",
            ]
        )

    markout_rows = []
    if not markout.empty:
        for _, r in markout[markout["scope"].astype(str).str.endswith(":60s")].sort_values("mean_markout_bps").iterrows():
            markout_rows.append(
                [
                    str(r["category"]),
                    str(int(r["covered_fills"])),
                    pct(float(r["sample_fraction_of_category_maker_fills"])),
                    bps(float(r["mean_markout_bps"])),
                    bps(float(r["mean_adverse_cost_bps"])),
                    pct(float(r["win_rate_markout_positive"])),
                ]
            )

    proceed_count = int(gates["proceed_gate"].fillna(False).sum()) if not gates.empty else 0
    mechanical_count = int(gates["mechanical_pnl_gate_pass"].fillna(False).sum()) if not gates.empty else 0
    calibration_count = int(gates["calibration_tail_pass"].fillna(False).sum()) if not gates.empty else 0
    if proceed_count:
        verdict = (
            f"**{proceed_count} category/categories pass the strict K5-STRESS gate** "
            f"after typical structured-non-top3 and NegRisk-reliability filters. "
            f"The looser mechanical PnL gate passes {mechanical_count} categories. "
            "Treat passers as frozen paper-test candidates, not production green-lights."
        )
    else:
        verdict = (
            "**No category passes the strict K5-STRESS gate.** The maker build should not proceed; "
            "return effort to copytrade unless a new, pre-registered niche is tested."
        )

    pop_text = ""
    if not pop.empty:
        p = pop.iloc[0]
        pop_text = (
            f"The ex-ante maker population is **{int(p['wallets']):,} wallets**, "
            f"defined only by corrected maker share >= {MAKER_SHARE_CUTOFF:.0%} and "
            f">= {MIN_PASSIVE_FILLS:,} passive fills. It covers "
            f"**{int(float(p['maker_fills'])):,} passive fills** and "
            f"**{dollars(float(p['maker_usd']))}** maker notional."
        )

    gap_text = ""
    if not gap.empty:
        g = gap.iloc[0]
        gap_text = (
            f"Adding open/unresolved inventory changes PnL by **{dollars(float(g['net_pnl_usd_gap']))}** "
            f"or **{bps(float(g['net_pnl_bps_gap']))}** versus resolved-only."
        )

    negrisk_text = ""
    if not negrisk.empty:
        negrisk_text = str(negrisk.iloc[0]["note"])

    text = f"""# Block K5-STRESS Findings

## Verdict

{verdict}

{pop_text}

{gap_text}

## Gate Summary

{markdown_table(['category', 'proceed', 'full-pop bps', 'CI', 'typical wallet', 'structured non-top3', 'structured median', 'non-top3 CI', 'ex rebate bps', 'positive months', 'calib net gap'], gate_rows)}

Gate rule used here: full-population CI lower bound above zero, structured non-top3 CI lower bound above zero, typical structured non-top3 wallet above zero, net still positive without rebate, not concentrated into fewer than two-thirds positive active months, and no unresolved NegRisk accounting caveat. Calibration is reported as an independent survivorship-immune cross-check, not as a substitute for wallet PnL; **{calibration_count} categories** have tail calibration CI above zero.

## Survivorship Fix

{markdown_table(['scope', 'wallets', 'markets', 'gross', 'net PnL', 'bps', 'CI'], surv_rows)}

Resolved inventory uses `closed_positions.parquet`, which already settles held/never-closed positions to the resolved $1/$0 outcome. Open and unresolved markets are reconstructed from raw fills and marked to the latest executed trade price because historical book mids are not in the owned fill files.

## Structure And Deployability

The structured playbook is defined ex ante as:

- two-sided maker USD share >= {STRUCT_TWO_SIDED_MIN:.0%}
- carry-token share >= {STRUCT_CARRY_MIN:.0%}
- crypto late near-50c spike-zone share <= {STRUCT_SPIKE_MAX:.0%}

The deployable number is the structured sub-population after excluding each market's global top-3 maker wallets. That is the closest historical proxy for what a non-incumbent can expect.

## Rebate / Cost Interpretation

The `net_without_rebate_bps` column removes maker rebates from the full-population result. Any category that only survives with rebate is policy-fragile. `data/analysis/csv_outputs/market_making/k5_stress.csv` also includes base marked PnL, maker rebate, and taker-fee columns for every aggregate.

## Adverse Selection

60s markouts below use a deterministic full-population sample of maker fills, not only winners.

{markdown_table(['category', 'covered fills', 'sample share', 'mean markout', 'adverse cost', 'positive rate'], markout_rows)}

Positive markout means maker-favorable reversion; adverse cost is `-markout`.

## NegRisk

{negrisk_text}

Politics NegRisk rows are therefore shown for diagnostics but should not be used as a green-light until merge/split/redemption events are explicitly reconstructed.

## Calibration Cross-Check

The calibration rows bucket resolved binary markets by last observed traded price at fixed times-to-resolution. This is survivorship-immune to wallet selection, but it is still a trade-price proxy rather than a historical quote book. A positive tail calibration gap corroborates a real longshot premium; it does not prove that a new maker can capture it after inventory and capacity.

## My Read

K5's closed-only result was a useful reality check, but K5-STRESS is the gating version. The only acceptable build signal is a category that remains positive for the whole maker population and for structured non-incumbents, without being rebate-only or one-episode. Categories failing that standard are not maker-build candidates.
"""
    NOTE.write_text(text, encoding="utf-8")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    con = connect()
    print("[k5-stress] installing market tables")
    install_market_tables(con)
    if MAKERS_CACHE.exists():
        print("[k5-stress] loading cached full ex-ante maker population")
        con.execute(
            f"CREATE OR REPLACE TABLE qualified_makers AS SELECT * FROM read_parquet('{MAKERS_CACHE}')"
        )
        makers = con.execute("SELECT * FROM qualified_makers ORDER BY maker_usd DESC").df()
    else:
        print("[k5-stress] identifying full ex-ante maker population")
        makers = identify_full_maker_population(con)
        con.execute(f"COPY qualified_makers TO '{MAKERS_CACHE}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"[k5-stress] qualified makers: {len(makers):,}")
    if WALLET_MARKET_CACHE.exists():
        print("[k5-stress] loading cached full marked PnL table")
        wallet_market = con.execute(f"SELECT * FROM read_parquet('{WALLET_MARKET_CACHE}')").df()
    else:
        print("[k5-stress] building fee, behavior, and global top3 tables")
        build_fee_behavior_and_capacity(con)
        print("[k5-stress] building full marked PnL tables")
        wallet_market = build_marked_pnl_tables(con)
        con.execute(
            f"COPY wallet_market_full TO '{WALLET_MARKET_CACHE}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    print(f"[k5-stress] wallet-market rows: {len(wallet_market):,}")
    if MARKOUT_CACHE.exists():
        print("[k5-stress] loading cached markout sample")
        markouts = con.execute(f"SELECT * FROM read_parquet('{MARKOUT_CACHE}')").df()
    else:
        print("[k5-stress] building full-population markout sample")
        markouts = build_markouts(con)
        con.execute(f"COPY markout_joined TO '{MARKOUT_CACHE}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"[k5-stress] markout sample rows: {len(markouts):,}")
    if CALIBRATION_CACHE.exists():
        print("[k5-stress] loading cached calibration observations")
        calibration = con.execute(f"SELECT * FROM read_parquet('{CALIBRATION_CACHE}')").df()
    else:
        print("[k5-stress] building survivorship-immune calibration")
        calibration = build_calibration(con)
        con.execute(f"COPY calibration_obs TO '{CALIBRATION_CACHE}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"[k5-stress] calibration observations: {len(calibration):,}")
    print("[k5-stress] summarizing")
    output = summarize_outputs(makers, wallet_market, markouts, calibration)
    output.to_csv(OUT_CSV, index=False)
    write_note(output)
    print(f"[k5-stress] wrote {OUT_CSV}")
    print(f"[k5-stress] wrote {NOTE}")


if __name__ == "__main__":
    main()
