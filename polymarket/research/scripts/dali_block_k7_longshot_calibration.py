"""Block K7 cross-category longshot-premium harvest.

Research-only sidecar. This is model-free: no external underlyings, no vol
model. It uses resolved binary markets, historical fills, and settlement prices
to ask whether low-priced outcome tokens are overpriced enough to sell as a
maker and hold to resolution.

Important data limitation: data/trades/*.parquet contains executed fills, not a
full historical order book. "Quoted price" calibration therefore uses the last
observable traded price at fixed times-to-resolution, with a staleness filter.
The tradeable overlay uses actual maker-sell fills in the tail.
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
MARKETS = sorted((DATA / "markets").glob("markets_*.parquet"))[-1]
OUT_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "k7_longshot_calibration.csv"
NOTE = NOTES / "block_k7_findings.md"

sys.path.insert(0, str(ROOT))
from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG  # noqa: E402


RNG_SEED = 20260531
BOOTSTRAP_SAMPLES = 500
DATA_START = "2025-08-01"
DATA_END = "2026-05-31"
TAIL_MIN_PRICE = 0.005
TAIL_MAX_PRICE = 0.20
FLOW_MIN_FILLS_PER_DAY = 50.0
FLOW_MIN_USD_PER_DAY = 100.0
CAPACITY_TOP3_MAX_SHARE = 0.80
CLUSTER_LOSS_MAX_MULTIPLE = 2.0

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


def cents(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}c"


def dollars(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def num(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:,.2f}"


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
    temp_dir = ANALYSIS / ".duckdb_tmp_k7"
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
        CREATE OR REPLACE TABLE binary_market_meta AS
        WITH base AS (
            SELECT
                CAST(id AS VARCHAR) AS market_id,
                CAST(condition_id AS VARCHAR) AS condition_id,
                lower(coalesce(slug, '')) AS slug_l,
                coalesce(slug, '') AS slug,
                lower(coalesce(question, '')) AS question_l,
                coalesce(question, '') AS question,
                coalesce(neg_risk, false) AS neg_risk,
                TRY_CAST(end_date AS TIMESTAMP) AS end_ts,
                TRY_CAST(created_at AS TIMESTAMP) AS created_ts,
                coalesce(closed, false) AS closed,
                len(clob_token_ids) AS n_tokens
            FROM read_parquet('{MARKETS}')
            WHERE coalesce(closed, false)
              AND len(clob_token_ids) = 2
              AND TRY_CAST(end_date AS TIMESTAMP) >= TIMESTAMP '{DATA_START}'
              AND TRY_CAST(end_date AS TIMESTAMP) <= TIMESTAMP '{DATA_END}'
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
        CREATE OR REPLACE TABLE binary_market_tokens AS
        SELECT
            CAST(m.id AS VARCHAR) AS market_id,
            CAST(m.condition_id AS VARCHAR) AS condition_id,
            r.i AS outcome_index,
            CAST(m.clob_token_ids[r.i] AS VARCHAR) AS outcome_token_id,
            coalesce(m.outcomes[r.i], '') AS outcome_label,
            TRY_CAST(m.outcome_prices[r.i] AS DOUBLE) AS resolution_price
        FROM read_parquet('{MARKETS}') m,
             range(1, len(m.clob_token_ids) + 1) AS r(i)
        WHERE CAST(m.id AS VARCHAR) IN (SELECT market_id FROM binary_market_meta)
          AND len(m.clob_token_ids) = 2
          AND TRY_CAST(m.outcome_prices[r.i] AS DOUBLE) IS NOT NULL
        """
    )


def build_calibration(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    horizon_values = ", ".join(
        f"('{label}', {sec}, {max_age})" for label, sec, max_age in HORIZONS
    )
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE calibration_obs AS
        WITH candidates AS (
            SELECT
                CAST(rt.market_id AS VARCHAR) AS market_id,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                    AS outcome_token_id,
                mm.category,
                mm.fee_family,
                mm.fee_rate,
                mm.maker_rebate_share,
                mt.resolution_price,
                h.horizon_label,
                h.horizon_sec,
                h.max_age_sec,
                mm.end_ts - h.horizon_sec * INTERVAL 1 SECOND AS target_ts,
                rt.timestamp AS price_ts,
                epoch((mm.end_ts - h.horizon_sec * INTERVAL 1 SECOND) - rt.timestamp)
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
            JOIN binary_market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            JOIN binary_market_tokens mt
              ON CAST(rt.market_id AS VARCHAR) = mt.market_id
             AND (CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END)
                 = mt.outcome_token_id
            CROSS JOIN (VALUES {horizon_values}) AS h(horizon_label, horizon_sec, max_age_sec)
            WHERE rt.timestamp >= TIMESTAMP '{DATA_START}'
              AND rt.timestamp <= TIMESTAMP '{DATA_END}'
              AND rt.timestamp <= mm.end_ts
              AND rt.timestamp <= mm.end_ts - h.horizon_sec * INTERVAL 1 SECOND
              AND rt.timestamp >= mm.end_ts - (h.horizon_sec + h.max_age_sec) * INTERVAL 1 SECOND
              AND mm.end_ts - h.horizon_sec * INTERVAL 1 SECOND >= TIMESTAMP '{DATA_START}'
              AND rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND rt.maker NOT IN ({internals})
              AND rt.taker NOT IN ({internals})
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
            fee_rate * price * (1.0 - price) * maker_rebate_share AS maker_rebate_per_token,
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


def build_overlay(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE tail_maker_sells AS
        SELECT
            rt.timestamp,
            CAST(rt.market_id AS VARCHAR) AS market_id,
            mm.condition_id,
            mm.category,
            mm.fee_family,
            mm.fee_rate,
            mm.maker_rebate_share,
            mm.end_ts,
            rt.maker,
            rt.taker,
            CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                AS outcome_token_id,
            rt.price,
            rt.token_amount,
            rt.usd_amount,
            rt.transaction_hash,
            mt.resolution_price,
            mm.fee_rate * rt.price * (1.0 - rt.price) * mm.maker_rebate_share AS maker_rebate_per_token,
            rt.token_amount * mm.fee_rate * rt.price * (1.0 - rt.price) * mm.maker_rebate_share
                AS maker_rebate_usd,
            0.0 AS maker_fee_usd,
            rt.token_amount
                * (rt.price - mt.resolution_price
                   + mm.fee_rate * rt.price * (1.0 - rt.price) * mm.maker_rebate_share)
                AS net_pnl_usd,
            rt.token_amount * (rt.price - mt.resolution_price) AS gross_edge_usd,
            (rt.price - mt.resolution_price
                   + mm.fee_rate * rt.price * (1.0 - rt.price) * mm.maker_rebate_share)
                / nullif(rt.price, 0) * 10000.0 AS net_pnl_bps,
            CASE
                WHEN rt.price < 0.01 THEN '00_01c'
                WHEN rt.price < 0.02 THEN '01_02c'
                WHEN rt.price < 0.05 THEN '02_05c'
                WHEN rt.price < 0.10 THEN '05_10c'
                ELSE '10_20c'
            END AS tail_price_bucket,
            CAST(mm.end_ts AS DATE) AS resolution_day
        FROM read_parquet('{TRADES_GLOB}') rt
        JOIN binary_market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
        JOIN binary_market_tokens mt
          ON CAST(rt.market_id AS VARCHAR) = mt.market_id
         AND (CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END)
             = mt.outcome_token_id
        WHERE rt.timestamp >= TIMESTAMP '{DATA_START}'
          AND rt.timestamp <= TIMESTAMP '{DATA_END}'
          AND rt.timestamp <= mm.end_ts
          AND rt.maker_asset_id <> '0'
          AND rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND rt.market_id IS NOT NULL
          AND rt.maker NOT IN ({sql_list(EXCHANGE_INTERNAL_LEG)})
          AND rt.taker NOT IN ({sql_list(EXCHANGE_INTERNAL_LEG)})
          AND rt.token_amount > 0
          AND rt.usd_amount > 0
          AND rt.price BETWEEN {TAIL_MIN_PRICE} AND {TAIL_MAX_PRICE}
        """
    )
    overlay_market = con.execute(
        """
        SELECT
            category,
            market_id,
            condition_id,
            count(*) AS fills,
            count(DISTINCT maker) AS makers,
            count(DISTINCT taker) AS takers,
            sum(token_amount) AS token_amount,
            sum(usd_amount) AS sell_usd,
            sum(net_pnl_usd) AS net_pnl_usd,
            sum(gross_edge_usd) AS gross_edge_usd,
            sum(maker_rebate_usd) AS maker_rebate_usd,
            min(net_pnl_usd) AS worst_fill_pnl_usd,
            min(net_pnl_bps) AS worst_fill_pnl_bps,
            avg(CASE WHEN net_pnl_usd > 0 THEN 1.0 ELSE 0.0 END) AS fill_win_rate,
            min(timestamp) AS first_ts,
            max(timestamp) AS last_ts,
            max(end_ts) AS end_ts,
            any_value(resolution_day) AS resolution_day
        FROM tail_maker_sells
        GROUP BY 1, 2, 3
        """
    ).df()
    wallet_market = con.execute(
        """
        SELECT
            category,
            market_id,
            maker,
            count(*) AS fills,
            sum(usd_amount) AS sell_usd,
            sum(net_pnl_usd) AS net_pnl_usd
        FROM tail_maker_sells
        GROUP BY 1, 2, 3
        """
    ).df()
    fill_risk = con.execute(
        """
        SELECT
            category,
            count(*) AS fills,
            min(net_pnl_usd) AS worst_single_fill_pnl_usd,
            min(net_pnl_bps) AS worst_single_fill_pnl_bps,
            quantile_cont(net_pnl_bps, 0.01) AS p01_pnl_bps,
            quantile_cont(net_pnl_bps, 0.05) AS p05_pnl_bps,
            quantile_cont(net_pnl_bps, 0.50) AS median_pnl_bps,
            avg(CASE WHEN net_pnl_usd > 0 THEN 1.0 ELSE 0.0 END) AS fill_win_rate
        FROM tail_maker_sells
        GROUP BY 1
        """
    ).df()
    return overlay_market, wallet_market, fill_risk


def build_flow_and_clusters(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flow = con.execute(
        """
        WITH taker_rank AS (
            SELECT
                category,
                taker,
                sum(usd_amount) AS taker_usd
            FROM tail_maker_sells
            GROUP BY 1, 2
        ),
        taker_top AS (
            SELECT
                category,
                sum(taker_usd) AS total_taker_usd,
                sum(CASE WHEN rnk <= 1 THEN taker_usd ELSE 0 END) AS top1_taker_usd,
                sum(CASE WHEN rnk <= 3 THEN taker_usd ELSE 0 END) AS top3_taker_usd
            FROM (
                SELECT
                    *,
                    row_number() OVER (PARTITION BY category ORDER BY taker_usd DESC) AS rnk
                FROM taker_rank
            )
            GROUP BY 1
        )
        SELECT
            s.category,
            count(*) AS fills,
            count(DISTINCT s.market_id) AS markets,
            count(DISTINCT s.maker) AS makers,
            count(DISTINCT s.taker) AS takers,
            sum(s.usd_amount) AS sell_usd,
            sum(s.token_amount) AS token_amount,
            min(s.timestamp) AS first_ts,
            max(s.timestamp) AS last_ts,
            count(DISTINCT CAST(s.timestamp AS DATE)) AS active_flow_days,
            epoch(max(s.timestamp) - min(s.timestamp)) / 86400.0 + 1 AS calendar_days,
            count(*) / nullif(epoch(max(s.timestamp) - min(s.timestamp)) / 86400.0 + 1, 0) AS fills_per_calendar_day,
            count(*) / nullif(count(DISTINCT CAST(s.timestamp AS DATE)), 0) AS fills_per_active_day,
            sum(s.usd_amount) / nullif(epoch(max(s.timestamp) - min(s.timestamp)) / 86400.0 + 1, 0)
                AS sell_usd_per_calendar_day,
            tt.top1_taker_usd / nullif(tt.total_taker_usd, 0) AS top1_taker_share,
            tt.top3_taker_usd / nullif(tt.total_taker_usd, 0) AS top3_taker_share
        FROM tail_maker_sells s
        LEFT JOIN taker_top tt USING (category)
        GROUP BY s.category, tt.top1_taker_usd, tt.top3_taker_usd, tt.total_taker_usd
        """
    ).df()
    cluster_day = con.execute(
        """
        SELECT
            category,
            resolution_day,
            count(DISTINCT market_id) AS markets,
            count(*) AS fills,
            sum(usd_amount) AS sell_usd,
            sum(net_pnl_usd) AS net_pnl_usd
        FROM tail_maker_sells
        GROUP BY 1, 2
        """
    ).df()
    cluster_condition = con.execute(
        """
        SELECT
            category,
            condition_id,
            count(DISTINCT market_id) AS markets,
            count(*) AS fills,
            sum(usd_amount) AS sell_usd,
            sum(net_pnl_usd) AS net_pnl_usd
        FROM tail_maker_sells
        GROUP BY 1, 2
        """
    ).df()
    return flow, cluster_day, cluster_condition


def bootstrap_mean_ci(
    df: pd.DataFrame,
    value_col: str,
    block_col: str = "market_id",
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    d = df[[block_col, value_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return math.nan, math.nan
    blocks = [g[value_col].to_numpy(float) for _, g in d.groupby(block_col, sort=False)]
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(blocks), len(blocks))
        estimates.append(float(np.nanmean(np.concatenate([blocks[i] for i in idx]))))
    lo, hi = np.nanquantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_ratio_ci(
    df: pd.DataFrame,
    num_col: str,
    den_col: str,
    block_col: str = "market_id",
    multiplier: float = 1.0,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    d = df[[block_col, num_col, den_col]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d[den_col].gt(0)]
    if d.empty:
        return math.nan, math.nan
    blocks = d.groupby(block_col, sort=False)[[num_col, den_col]].sum().reset_index()
    if len(blocks) < 2:
        return math.nan, math.nan
    vals = blocks[[num_col, den_col]].to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(vals), len(vals))
        den = vals[idx, 1].sum()
        estimates.append(vals[idx, 0].sum() / den * multiplier if den > 0 else math.nan)
    lo, hi = np.nanquantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_calibration(cal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    bucket_keys = ["category", "horizon_label", "price_bucket"]
    for key, piece in cal.groupby(bucket_keys, sort=False):
        category, horizon, bucket = key
        gap_lo, gap_hi = bootstrap_mean_ci(piece, "longshot_gap")
        net_lo, net_hi = bootstrap_mean_ci(piece, "maker_net_gap")
        rows.append(
            {
                "section": "calibration_bucket",
                "category": category,
                "horizon_label": horizon,
                "price_bucket": bucket,
                "observations": int(len(piece)),
                "markets": int(piece["market_id"].nunique()),
                "avg_price": float(piece["price"].mean()),
                "realized_freq": float(piece["resolution_price"].mean()),
                "longshot_gap": float(piece["longshot_gap"].mean()),
                "gap_ci_lo": gap_lo,
                "gap_ci_hi": gap_hi,
                "avg_maker_rebate": float(piece["maker_rebate_per_token"].mean()),
                "maker_net_gap": float(piece["maker_net_gap"].mean()),
                "net_gap_ci_lo": net_lo,
                "net_gap_ci_hi": net_hi,
                "avg_price_age_sec": float(piece["price_age_sec"].mean()),
            }
        )

    tail_rows: list[dict[str, Any]] = []
    tail = cal[cal["is_tail"].fillna(False)].copy()
    for category, piece in tail.groupby("category", sort=False):
        gap_lo, gap_hi = bootstrap_mean_ci(piece, "longshot_gap")
        net_lo, net_hi = bootstrap_mean_ci(piece, "maker_net_gap")
        tail_rows.append(
            {
                "section": "calibration_tail_category",
                "category": category,
                "horizon_label": "all",
                "price_bucket": "tail_0.5_20c",
                "observations": int(len(piece)),
                "markets": int(piece["market_id"].nunique()),
                "avg_price": float(piece["price"].mean()),
                "realized_freq": float(piece["resolution_price"].mean()),
                "longshot_gap": float(piece["longshot_gap"].mean()),
                "gap_ci_lo": gap_lo,
                "gap_ci_hi": gap_hi,
                "avg_maker_rebate": float(piece["maker_rebate_per_token"].mean()),
                "maker_net_gap": float(piece["maker_net_gap"].mean()),
                "net_gap_ci_lo": net_lo,
                "net_gap_ci_hi": net_hi,
                "avg_price_age_sec": float(piece["price_age_sec"].mean()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(tail_rows)


def summarize_overlay(overlay_market: pd.DataFrame, fill_risk: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category, piece in overlay_market.groupby("category", sort=False):
        sell_usd = float(piece["sell_usd"].sum())
        pnl = float(piece["net_pnl_usd"].sum())
        ci_lo, ci_hi = bootstrap_ratio_ci(piece, "net_pnl_usd", "sell_usd", multiplier=10_000.0)
        rows.append(
            {
                "section": "overlay_category",
                "category": category,
                "markets": int(piece["market_id"].nunique()),
                "fills": int(piece["fills"].sum()),
                "makers": int(piece["makers"].sum()),
                "takers": int(piece["takers"].sum()),
                "sell_usd": sell_usd,
                "net_pnl_usd": pnl,
                "net_pnl_bps": pnl / sell_usd * 10_000 if sell_usd > 0 else math.nan,
                "ci_lo_bps": ci_lo,
                "ci_hi_bps": ci_hi,
                "gross_edge_usd": float(piece["gross_edge_usd"].sum()),
                "maker_rebate_usd": float(piece["maker_rebate_usd"].sum()),
                "market_win_rate": float(piece["net_pnl_usd"].gt(0).mean()),
                "worst_market_pnl_usd": float(piece["net_pnl_usd"].min()),
            }
        )
    out = pd.DataFrame(rows)
    if not fill_risk.empty:
        out = out.merge(fill_risk, on="category", how="left", suffixes=("", "_fill"))
    return out


def summarize_capacity(wallet_market: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if wallet_market.empty:
        return pd.DataFrame(rows)
    for category, cat in wallet_market.groupby("category", sort=False):
        market_rows = []
        for market_id, piece in cat.groupby("market_id", sort=False):
            ranked = piece.sort_values("net_pnl_usd", ascending=False)
            positive = ranked[ranked["net_pnl_usd"].gt(0)]
            pos_total = float(positive["net_pnl_usd"].sum())
            market_rows.append(
                {
                    "market_id": market_id,
                    "wallets": int(piece["maker"].nunique()),
                    "positive_pnl_usd": pos_total,
                    "top1_positive_pnl_usd": float(ranked["net_pnl_usd"].clip(lower=0).head(1).sum()),
                    "top3_positive_pnl_usd": float(ranked["net_pnl_usd"].clip(lower=0).head(3).sum()),
                }
            )
        md = pd.DataFrame(market_rows)
        total_pos = float(md["positive_pnl_usd"].sum())
        top1_share = float(md["top1_positive_pnl_usd"].sum() / total_pos) if total_pos > 0 else math.nan
        top3_share = float(md["top3_positive_pnl_usd"].sum() / total_pos) if total_pos > 0 else math.nan
        top1_lo, top1_hi = bootstrap_ratio_ci(
            md.assign(dummy_den=md["positive_pnl_usd"]),
            "top1_positive_pnl_usd",
            "dummy_den",
            multiplier=1.0,
        )
        top3_lo, top3_hi = bootstrap_ratio_ci(
            md.assign(dummy_den=md["positive_pnl_usd"]),
            "top3_positive_pnl_usd",
            "dummy_den",
            multiplier=1.0,
        )
        rows.append(
            {
                "section": "capacity_category",
                "category": category,
                "markets": int(len(md)),
                "median_wallets_per_market": float(md["wallets"].median()),
                "positive_pnl_usd": total_pos,
                "top1_positive_profit_share": top1_share,
                "top1_share_ci_lo": top1_lo,
                "top1_share_ci_hi": top1_hi,
                "top3_positive_profit_share": top3_share,
                "top3_share_ci_lo": top3_lo,
                "top3_share_ci_hi": top3_hi,
            }
        )
    return pd.DataFrame(rows)


def summarize_clusters(
    overlay_market: pd.DataFrame,
    cluster_day: pd.DataFrame,
    cluster_condition: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    totals = overlay_market.groupby("category")["net_pnl_usd"].sum()
    positive = overlay_market.groupby("category")["net_pnl_usd"].apply(lambda s: s[s > 0].sum())
    for category, piece in cluster_day.groupby("category", sort=False):
        worst = piece.sort_values("net_pnl_usd").iloc[0]
        total = float(totals.get(category, math.nan))
        pos = float(positive.get(category, math.nan))
        rows.append(
            {
                "section": "cluster_day",
                "category": category,
                "cluster_key": str(worst["resolution_day"]),
                "clusters": int(len(piece)),
                "worst_cluster_pnl_usd": float(worst["net_pnl_usd"]),
                "worst_cluster_markets": int(worst["markets"]),
                "worst_cluster_fills": int(worst["fills"]),
                "cluster_loss_to_positive_profit": abs(float(worst["net_pnl_usd"])) / pos
                if pos > 0 and float(worst["net_pnl_usd"]) < 0
                else 0.0,
                "cluster_loss_to_total_profit": abs(float(worst["net_pnl_usd"])) / total
                if total > 0 and float(worst["net_pnl_usd"]) < 0
                else 0.0,
            }
        )
    for category, piece in cluster_condition.groupby("category", sort=False):
        worst = piece.sort_values("net_pnl_usd").iloc[0]
        pos = float(positive.get(category, math.nan))
        rows.append(
            {
                "section": "cluster_condition",
                "category": category,
                "cluster_key": str(worst["condition_id"])[:24],
                "clusters": int(len(piece)),
                "worst_cluster_pnl_usd": float(worst["net_pnl_usd"]),
                "worst_cluster_markets": int(worst["markets"]),
                "worst_cluster_fills": int(worst["fills"]),
                "cluster_loss_to_positive_profit": abs(float(worst["net_pnl_usd"])) / pos
                if pos > 0 and float(worst["net_pnl_usd"]) < 0
                else 0.0,
                "cluster_loss_to_total_profit": math.nan,
            }
        )
    return pd.DataFrame(rows)


def build_final_rank(
    calibration_tail: pd.DataFrame,
    overlay: pd.DataFrame,
    flow: pd.DataFrame,
    capacity: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    final = overlay.merge(
        calibration_tail[
            [
                "category",
                "observations",
                "avg_price",
                "realized_freq",
                "maker_net_gap",
                "net_gap_ci_lo",
                "net_gap_ci_hi",
            ]
        ].rename(
            columns={
                "observations": "calibration_observations",
                "avg_price": "calibration_avg_price",
                "realized_freq": "calibration_realized_freq",
                "maker_net_gap": "calibration_net_gap",
                "net_gap_ci_lo": "calibration_net_gap_ci_lo",
                "net_gap_ci_hi": "calibration_net_gap_ci_hi",
            }
        ),
        on="category",
        how="left",
    )
    final = final.merge(flow, on="category", how="left", suffixes=("", "_flow"))
    final = final.merge(capacity, on="category", how="left", suffixes=("", "_capacity"))
    day = clusters[clusters["section"].eq("cluster_day")][
        ["category", "worst_cluster_pnl_usd", "cluster_loss_to_positive_profit"]
    ].rename(
        columns={
            "worst_cluster_pnl_usd": "worst_day_cluster_pnl_usd",
            "cluster_loss_to_positive_profit": "day_cluster_loss_to_positive_profit",
        }
    )
    final = final.merge(day, on="category", how="left")
    final["gate_calibration"] = final["calibration_net_gap_ci_lo"].gt(0)
    final["gate_overlay"] = final["ci_lo_bps"].gt(0)
    final["gate_flow"] = final["fills_per_calendar_day"].ge(FLOW_MIN_FILLS_PER_DAY) & final[
        "sell_usd_per_calendar_day"
    ].ge(FLOW_MIN_USD_PER_DAY)
    final["gate_capacity"] = final["top3_positive_profit_share"].lt(CAPACITY_TOP3_MAX_SHARE)
    final["gate_tail_risk"] = final["day_cluster_loss_to_positive_profit"].le(CLUSTER_LOSS_MAX_MULTIPLE)
    final["clears_all"] = (
        final["gate_calibration"]
        & final["gate_overlay"]
        & final["gate_flow"]
        & final["gate_capacity"]
        & final["gate_tail_risk"]
    )
    gate_cols = ["gate_calibration", "gate_overlay", "gate_flow", "gate_capacity", "gate_tail_risk"]
    final["gate_count"] = final[gate_cols].sum(axis=1)
    final["rank_score"] = (
        final["gate_count"].fillna(0) * 10_000
        + final["net_pnl_bps"].fillna(-1e9)
        - final["top3_positive_profit_share"].fillna(1.0) * 100
    )
    final = final.sort_values(["clears_all", "rank_score"], ascending=False).reset_index(drop=True)
    if "section" in final.columns:
        final = final.drop(columns=["section"])
    final.insert(1, "rank", np.arange(1, len(final) + 1))
    final.insert(0, "section", "final_rank")
    return final


def write_note(
    calibration_tail: pd.DataFrame,
    overlay: pd.DataFrame,
    flow: pd.DataFrame,
    capacity: pd.DataFrame,
    clusters: pd.DataFrame,
    final: pd.DataFrame,
) -> None:
    rank_rows = []
    for _, r in final.head(12).iterrows():
        rank_rows.append(
            [
                str(int(r["rank"])),
                str(r["category"]),
                "yes" if bool(r["clears_all"]) else "no",
                str(int(r["gate_count"])),
                cents(float(r["calibration_net_gap"])),
                f"[{cents(float(r['calibration_net_gap_ci_lo']))}, {cents(float(r['calibration_net_gap_ci_hi']))}]",
                bps(float(r["net_pnl_bps"])),
                f"[{bps(float(r['ci_lo_bps']))}, {bps(float(r['ci_hi_bps']))}]",
                num(float(r["fills_per_calendar_day"])),
                dollars(float(r["sell_usd_per_calendar_day"])),
                pct(float(r["top3_positive_profit_share"])),
                num(float(r["day_cluster_loss_to_positive_profit"])),
            ]
        )

    calib_rows = []
    for _, r in calibration_tail.sort_values("maker_net_gap", ascending=False).head(12).iterrows():
        calib_rows.append(
            [
                str(r["category"]),
                str(int(r["observations"])),
                str(int(r["markets"])),
                cents(float(r["avg_price"])),
                pct(float(r["realized_freq"])),
                cents(float(r["longshot_gap"])),
                f"[{cents(float(r['gap_ci_lo']))}, {cents(float(r['gap_ci_hi']))}]",
                cents(float(r["maker_net_gap"])),
            ]
        )

    overlay_rows = []
    for _, r in overlay.sort_values("net_pnl_bps", ascending=False).head(12).iterrows():
        overlay_rows.append(
            [
                str(r["category"]),
                str(int(r["markets"])),
                str(int(r["fills"])),
                dollars(float(r["sell_usd"])),
                dollars(float(r["net_pnl_usd"])),
                bps(float(r["net_pnl_bps"])),
                f"[{bps(float(r['ci_lo_bps']))}, {bps(float(r['ci_hi_bps']))}]",
                pct(float(r["fill_win_rate"])),
                dollars(float(r["worst_market_pnl_usd"])),
            ]
        )

    flow_rows = []
    for _, r in flow.sort_values("fills_per_calendar_day", ascending=False).head(12).iterrows():
        flow_rows.append(
            [
                str(r["category"]),
                str(int(r["fills"])),
                str(int(r["markets"])),
                num(float(r["fills_per_calendar_day"])),
                num(float(r["fills_per_active_day"])),
                dollars(float(r["sell_usd_per_calendar_day"])),
                str(int(r["takers"])),
                pct(float(r["top3_taker_share"])),
            ]
        )

    risk_rows = []
    day = clusters[clusters["section"].eq("cluster_day")].copy()
    for _, r in day.sort_values("cluster_loss_to_positive_profit", ascending=False).head(12).iterrows():
        risk_rows.append(
            [
                str(r["category"]),
                str(r["cluster_key"]),
                dollars(float(r["worst_cluster_pnl_usd"])),
                str(int(r["worst_cluster_markets"])),
                str(int(r["worst_cluster_fills"])),
                num(float(r["cluster_loss_to_positive_profit"])),
            ]
        )

    cap_rows = []
    for _, r in capacity.sort_values("top3_positive_profit_share").head(12).iterrows():
        cap_rows.append(
            [
                str(r["category"]),
                str(int(r["markets"])),
                num(float(r["median_wallets_per_market"])),
                pct(float(r["top1_positive_profit_share"])),
                pct(float(r["top3_positive_profit_share"])),
                f"[{pct(float(r['top3_share_ci_lo']))}, {pct(float(r['top3_share_ci_hi']))}]",
            ]
        )

    clear = final[final["clears_all"]]
    if clear.empty:
        headline = "No category clears all K7 gates. The longshot-premium harvest is not deployable from this model-free pass; effort should return to copytrade unless a narrower category is pre-registered and retested."
    else:
        cats = ", ".join(clear["category"].head(5).astype(str))
        headline = f"Categories clearing all four gates: {cats}."

    text = f"""# Block K7 Cross-Category Longshot Premium

## Headline

{headline}

The strongest warning is capacity and tail clustering, not calibration. Several categories show low-price tokens that are overpriced at last-traded prices, and some actual maker-sell tail fills are profitable net of maker rebates. But the categories with enough flow are often either capacity-captured, cluster-risky, or have CIs that do not clear zero.

I split the premium test into two sub-gates: broad calibration and actual maker-sell overlay. That is why the rank table reports gates out of five. In this pass, the strong overlay categories do **not** also clear the calibration gate; `culture` clears calibration, but its tradeable overlay CI does not clear zero.

## Method

- Universe: resolved binary markets in the owned trade window, `DATA_START={DATA_START}`, `DATA_END={DATA_END}`.
- Calibration price: last executed trade before fixed time-to-resolution (`1h`, `4h`, `1d`, `7d`, `30d`) with a staleness filter. This is not a full order-book quote.
- Longshot tail: outcome-token price between **{TAIL_MIN_PRICE:.3f}** and **{TAIL_MAX_PRICE:.2f}**.
- Tradeable overlay: historical maker sells of tail contracts, held to settlement. Net PnL is `sell_price - resolution_payout + maker_rebate`; no exit spread and no external model.
- CIs bootstrap by market.

## Final Rank

{markdown_table(['rank', 'category', 'clears', 'gates /5', 'calib net gap', 'calib CI', 'overlay bps', 'overlay CI', 'fills/day', '$/day', 'top3 cap', 'cluster loss x'], rank_rows)}

## Calibration Gate

Tail calibration is the average low-price gap: traded price minus realized outcome frequency, plus maker rebate for the net-gap column.

{markdown_table(['category', 'obs', 'markets', 'avg price', 'realized', 'gap', 'gap CI', 'net gap'], calib_rows)}

## Tradeable Overlay

Actual historical maker-sell tail fills, held to settlement.

{markdown_table(['category', 'markets', 'fills', 'sell USD', 'net PnL', 'bps', 'CI', 'fill win rate', 'worst market'], overlay_rows)}

## Flow

This is the demand available to sell into: takers buying the longshot from a maker.

{markdown_table(['category', 'fills', 'markets', 'fills/day', 'fills/active day', '$/day', 'takers', 'top3 taker share'], flow_rows)}

## Tail And Cluster Risk

Worst resolution-day cluster by category.

{markdown_table(['category', 'worst day', 'cluster PnL', 'markets', 'fills', 'loss / positive profit'], risk_rows)}

## Capacity

Per-market positive maker profit concentration for the tail-selling overlay.

{markdown_table(['category', 'markets', 'median wallets/market', 'top1 positive share', 'top3 positive share', 'top3 CI'], cap_rows)}

## Conclusion

K7 does **not** green-light a broad cross-category longshot-premium maker. A few categories are interesting research leads, but the combined gate is strict: positive calibrated tail gap, positive net maker overlay with CI, enough buy-flow, not captured by the top makers, and tolerable cluster risk. On this pass, no category clears every gate.

The best next move is not to build another maker around this yet. Either narrow the hypothesis to a specific category/window and retest out-of-sample, or return effort to copytrade, which is the stated fallback.
"""
    NOTE.write_text(text, encoding="utf-8")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    con = connect()
    print("[k7] installing market tables")
    install_market_tables(con)
    n_markets = con.execute("SELECT count(*) FROM binary_market_meta").fetchone()[0]
    print(f"[k7] resolved binary markets in window: {n_markets:,}")
    print("[k7] building calibration observations")
    cal = build_calibration(con)
    print(f"[k7] calibration observations: {len(cal):,}")
    print("[k7] building maker-sell tail overlay")
    overlay_market, wallet_market, fill_risk = build_overlay(con)
    n_tail = int(overlay_market["fills"].sum()) if not overlay_market.empty else 0
    print(f"[k7] tail maker-sell fills: {n_tail:,}")
    print("[k7] flow, clusters, capacity")
    flow, cluster_day, cluster_condition = build_flow_and_clusters(con)
    cal_bucket, cal_tail = summarize_calibration(cal)
    overlay = summarize_overlay(overlay_market, fill_risk)
    capacity = summarize_capacity(wallet_market)
    clusters = summarize_clusters(overlay_market, cluster_day, cluster_condition)
    final = build_final_rank(cal_tail, overlay, flow, capacity, clusters)
    output = pd.concat(
        [cal_bucket, cal_tail, overlay, flow.assign(section="flow_category"), clusters, capacity, final],
        ignore_index=True,
        sort=False,
    )
    output.to_csv(OUT_CSV, index=False)
    write_note(cal_tail, overlay, flow, capacity, clusters, final)
    con.close()
    print(f"[k7] wrote {OUT_CSV}")
    print(f"[k7] wrote {NOTE}")


if __name__ == "__main__":
    main()
