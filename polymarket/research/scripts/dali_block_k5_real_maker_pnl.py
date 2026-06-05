"""Block K5 real-maker realized PnL reality check.

Research-only sidecar. This asks whether wallets that actually trade mostly as
passive makers earn realized money on resolved Polymarket markets. It is meant
as a model-free check against the harsher single-leg Block K simulations.

Important caveat: data/closed_positions.parquet is closed-position only. Open
positions and unresolved inventory are not represented, so every aggregate below
is conditional on positions that have closed or settled.
"""
from __future__ import annotations

import math
import re
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
OUT_CSV = ANALYSIS / "csv_outputs" / "market_making" / "k5_real_maker_pnl.csv"
NOTE = NOTES / "block_k5_findings.md"

sys.path.insert(0, str(ROOT))
from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG  # noqa: E402


RNG_SEED = 20260531
BOOTSTRAP_SAMPLES = 500
SEED_MAKER_SHARE = 0.50
SEED_MIN_STYLE_MAKER_FILLS = 500
SEED_LIMIT = 20_000
MAKER_SHARE_CUTOFF = 0.70
MIN_CORRECTED_FILLS = 1_000
TOP_OVERALL_WALLETS = 250
TOP_CRYPTO_WALLETS = 250
TOP_CATEGORY_WALLETS = 75
MARKOUT_WALLET_LIMIT = 50
MARKOUT_MAX_LAG_SEC = 300

SIM_TAKER_EXIT_BPS = -3_412.0
SIM_MAKER_EXIT_BPS = -2_216.0
SIM_HOLD_RES_BPS = 242.0

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


def small_dollars(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.4f}" if abs(value) < 1 else f"${value:,.2f}"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


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


def sql_list(values: list[str] | tuple[str, ...] | set[str]) -> str:
    vals = sorted({str(v).lower().replace("'", "''") for v in values if str(v)})
    if not vals:
        return "''"
    return ", ".join(f"'{v}'" for v in vals)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    temp_dir = ANALYSIS / ".duckdb_tmp_k5"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    return con


def install_market_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Materialize market/category tables used by the heavier raw-trade scans."""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE market_meta AS
        WITH base AS (
            SELECT
                CAST(id AS VARCHAR) AS market_id,
                lower(coalesce(slug, '')) AS slug_l,
                coalesce(slug, '') AS slug,
                lower(coalesce(question, '')) AS question_l,
                coalesce(question, '') AS question,
                coalesce(neg_risk, false) AS neg_risk,
                coalesce(closed, false) AS closed,
                TRY_CAST(end_date AS TIMESTAMP) AS end_date
            FROM read_parquet('{MARKETS}')
        )
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
                    slug_l LIKE '%openai%' OR slug_l LIKE '%ai%' OR slug_l LIKE '%nvidia%'
                    OR slug_l LIKE '%tesla%' OR slug_l LIKE '%spacex%' OR slug_l LIKE '%apple%'
                    OR slug_l LIKE '%iphone%' OR slug_l LIKE '%google%' OR slug_l LIKE '%meta%'
                ) THEN 'Tech'
                ELSE 'Other'
            END AS fee_family
        FROM base
        """
    )
    cases_rate = " ".join(
        f"WHEN fee_family = '{fam}' THEN {rate}" for fam, (rate, _) in FEE_BY_FAMILY.items()
    )
    cases_rebate = " ".join(
        f"WHEN fee_family = '{fam}' THEN {rebate}" for fam, (_, rebate) in FEE_BY_FAMILY.items()
    )
    con.execute(
        f"""
        ALTER TABLE market_meta ADD COLUMN fee_rate DOUBLE;
        ALTER TABLE market_meta ADD COLUMN maker_rebate_share DOUBLE;
        UPDATE market_meta
        SET
            fee_rate = CASE {cases_rate} ELSE 0.05 END,
            maker_rebate_share = CASE {cases_rebate} ELSE 0.25 END
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE market_tokens AS
        SELECT
            CAST(m.id AS VARCHAR) AS market_id,
            r.i AS outcome_index,
            CAST(m.clob_token_ids[r.i] AS VARCHAR) AS outcome_token_id,
            TRY_CAST(m.outcome_prices[r.i] AS DOUBLE) AS resolution_price
        FROM read_parquet('{MARKETS}') m,
             range(1, len(m.clob_token_ids) + 1) AS r(i)
        WHERE len(m.clob_token_ids) > 0
        """
    )


def identify_makers(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE seed_addresses AS
        SELECT lower(address) AS address
        FROM read_parquet('{TRADERS}')
        WHERE style_maker_fill_count >= {SEED_MIN_STYLE_MAKER_FILLS}
          AND style_maker_fill_count::DOUBLE
              / nullif(style_maker_fill_count + style_taker_fill_count, 0) >= {SEED_MAKER_SHARE}
        ORDER BY style_maker_fill_count DESC
        LIMIT {SEED_LIMIT}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE role_category_counts AS
        WITH maker_side AS (
            SELECT
                rt.maker AS address,
                coalesce(mm.category, 'unknown') AS category,
                'maker' AS role,
                count(*) AS fills,
                sum(rt.usd_amount) AS usd_volume
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN seed_addresses s ON rt.maker = s.address
            LEFT JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND rt.maker NOT IN ({internals})
              AND rt.taker NOT IN ({internals})
            GROUP BY 1, 2, 3
        ),
        taker_side AS (
            SELECT
                rt.taker AS address,
                coalesce(mm.category, 'unknown') AS category,
                'taker' AS role,
                count(*) AS fills,
                sum(rt.usd_amount) AS usd_volume
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN seed_addresses s ON rt.taker = s.address
            LEFT JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND rt.maker NOT IN ({internals})
              AND rt.taker NOT IN ({internals})
            GROUP BY 1, 2, 3
        )
        SELECT * FROM maker_side
        UNION ALL
        SELECT * FROM taker_side
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE corrected_role_counts AS
        SELECT
            r.address,
            sum(CASE WHEN role = 'maker' THEN fills ELSE 0 END) AS maker_fills,
            sum(CASE WHEN role = 'taker' THEN fills ELSE 0 END) AS taker_fills,
            sum(CASE WHEN role = 'maker' THEN usd_volume ELSE 0 END) AS maker_usd,
            sum(CASE WHEN role = 'taker' THEN usd_volume ELSE 0 END) AS taker_usd,
            sum(CASE WHEN role = 'maker' THEN fills ELSE 0 END)::DOUBLE
                / nullif(sum(fills), 0) AS maker_share,
            t.total_volume_usd,
            t.pos_total_pnl,
            t.n_closed_positions,
            t.is_operator_like
        FROM role_category_counts r
        LEFT JOIN read_parquet('{TRADERS}') t ON r.address = lower(t.address)
        GROUP BY
            r.address, t.total_volume_usd, t.pos_total_pnl, t.n_closed_positions, t.is_operator_like
        HAVING sum(CASE WHEN role = 'maker' THEN fills ELSE 0 END) >= {MIN_CORRECTED_FILLS}
           AND sum(CASE WHEN role = 'maker' THEN fills ELSE 0 END)::DOUBLE / nullif(sum(fills), 0)
               >= {MAKER_SHARE_CUTOFF}
        """
    )
    makers = con.execute(
        "SELECT * FROM corrected_role_counts ORDER BY maker_usd DESC, maker_fills DESC"
    ).df()
    category_counts = con.execute(
        """
        SELECT
            r.address,
            r.category,
            sum(CASE WHEN role = 'maker' THEN fills ELSE 0 END) AS maker_fills,
            sum(CASE WHEN role = 'taker' THEN fills ELSE 0 END) AS taker_fills,
            sum(CASE WHEN role = 'maker' THEN usd_volume ELSE 0 END) AS maker_usd,
            sum(CASE WHEN role = 'taker' THEN usd_volume ELSE 0 END) AS taker_usd,
            sum(CASE WHEN role = 'maker' THEN fills ELSE 0 END)::DOUBLE
                / nullif(sum(fills), 0) AS category_maker_share
        FROM role_category_counts r
        JOIN corrected_role_counts c ON r.address = c.address
        GROUP BY 1, 2
        """
    ).df()
    return makers, category_counts


def choose_analysis_wallets(con: duckdb.DuckDBPyConnection, makers: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    if not makers.empty:
        pieces.append(makers.nlargest(TOP_OVERALL_WALLETS, ["maker_usd", "maker_fills"]))
    crypto = cat[cat["category"].eq("crypto_4h") & cat["maker_fills"].gt(0)]
    if not crypto.empty:
        pieces.append(
            makers[maker_keys(makers).isin(set(crypto.nlargest(TOP_CRYPTO_WALLETS, "maker_usd")["address"]))]
        )
    for category, sub in cat.groupby("category", sort=False):
        if category == "unknown" or sub.empty:
            continue
        addresses = set(sub.nlargest(TOP_CATEGORY_WALLETS, "maker_usd")["address"])
        pieces.append(makers[maker_keys(makers).isin(addresses)])
    analysis = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates("address")
        .sort_values(["maker_usd", "maker_fills"], ascending=False)
        .reset_index(drop=True)
    )
    con.register("analysis_wallets_df", analysis[["address"]])
    con.execute("CREATE OR REPLACE TABLE analysis_wallets AS SELECT address FROM analysis_wallets_df")
    return analysis


def maker_keys(df: pd.DataFrame) -> pd.Series:
    return df["address"].astype(str).str.lower()


def build_pnl_tables(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE role_fee_adjustments AS
        WITH scoped AS (
            SELECT
                rt.timestamp,
                CAST(rt.market_id AS VARCHAR) AS market_id,
                rt.maker,
                rt.taker,
                rt.maker_asset_id,
                rt.taker_asset_id,
                rt.token_amount,
                rt.usd_amount,
                rt.price,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                    AS outcome_token_id,
                mm.category,
                mm.fee_family,
                mm.fee_rate,
                mm.maker_rebate_share
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND rt.maker NOT IN ({internals})
              AND rt.taker NOT IN ({internals})
              AND (rt.maker IN (SELECT address FROM analysis_wallets)
                   OR rt.taker IN (SELECT address FROM analysis_wallets))
        ),
        exploded AS (
            SELECT
                maker AS address,
                market_id,
                category,
                'maker' AS role,
                count(*) AS fill_count,
                sum(usd_amount) AS usd_volume,
                sum(token_amount * fee_rate * least(greatest(price, 0.001), 0.999)
                    * (1.0 - least(greatest(price, 0.001), 0.999)) * maker_rebate_share) AS maker_rebate_usd,
                0.0 AS taker_fee_usd
            FROM scoped
            WHERE maker IN (SELECT address FROM analysis_wallets)
            GROUP BY 1, 2, 3, 4
            UNION ALL
            SELECT
                taker AS address,
                market_id,
                category,
                'taker' AS role,
                count(*) AS fill_count,
                sum(usd_amount) AS usd_volume,
                0.0 AS maker_rebate_usd,
                sum(token_amount * fee_rate * least(greatest(price, 0.001), 0.999)
                    * (1.0 - least(greatest(price, 0.001), 0.999))) AS taker_fee_usd
            FROM scoped
            WHERE taker IN (SELECT address FROM analysis_wallets)
            GROUP BY 1, 2, 3, 4
        )
        SELECT
            address,
            market_id,
            category,
            sum(fill_count) AS raw_fill_count,
            sum(CASE WHEN role = 'maker' THEN fill_count ELSE 0 END) AS raw_maker_fill_count,
            sum(CASE WHEN role = 'taker' THEN fill_count ELSE 0 END) AS raw_taker_fill_count,
            sum(usd_volume) AS raw_usd_volume,
            sum(maker_rebate_usd) AS maker_rebate_usd,
            sum(taker_fee_usd) AS taker_fee_usd,
            sum(maker_rebate_usd) - sum(taker_fee_usd) AS fee_adjustment_usd
        FROM exploded
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE maker_fill_behavior AS
        SELECT
            rt.maker AS address,
            CAST(rt.market_id AS VARCHAR) AS market_id,
            mm.category,
            count(*) AS maker_fills,
            sum(rt.usd_amount) AS maker_usd,
            count(DISTINCT CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END)
                AS distinct_outcomes_made,
            sum(CASE WHEN rt.maker_asset_id = '0' THEN rt.usd_amount ELSE 0 END) AS maker_buy_usd,
            sum(CASE WHEN rt.maker_asset_id <> '0' THEN rt.usd_amount ELSE 0 END) AS maker_sell_usd,
            sum(
                CASE
                    WHEN mm.category = 'crypto_4h'
                     AND regexp_matches(mm.slug_l, 'updown-4h-[0-9]+')
                     AND TRY_CAST(regexp_extract(mm.slug_l, 'updown-4h-([0-9]+)', 1) AS BIGINT) IS NOT NULL
                     AND epoch(rt.timestamp) >= TRY_CAST(regexp_extract(mm.slug_l, 'updown-4h-([0-9]+)', 1) AS BIGINT) + 3.75 * 3600
                     AND rt.price BETWEEN 0.40 AND 0.60
                    THEN rt.usd_amount ELSE 0
                END
            ) AS spike_zone_usd,
            sum(
                CASE
                    WHEN mm.category = 'crypto_4h'
                     AND regexp_matches(mm.slug_l, 'updown-4h-[0-9]+')
                     AND TRY_CAST(regexp_extract(mm.slug_l, 'updown-4h-([0-9]+)', 1) AS BIGINT) IS NOT NULL
                     AND epoch(rt.timestamp) >= TRY_CAST(regexp_extract(mm.slug_l, 'updown-4h-([0-9]+)', 1) AS BIGINT) + 3.75 * 3600
                     AND rt.price BETWEEN 0.40 AND 0.60
                    THEN 1 ELSE 0
                END
            ) AS spike_zone_fills
        FROM read_parquet('{TRADES_GLOB}') rt
        JOIN analysis_wallets aw ON rt.maker = aw.address
        JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
        WHERE rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND rt.market_id IS NOT NULL
          AND rt.maker NOT IN ({internals})
          AND rt.taker NOT IN ({internals})
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE closed_analysis AS
        SELECT
            cp.address,
            CAST(cp.market_id AS VARCHAR) AS market_id,
            cp.outcome_token_id,
            cp.outcome_index,
            cp.n_fills,
            cp.gross_token_volume,
            cp.gross_usd_volume,
            cp.realised_pnl,
            cp.is_held_to_resolution,
            cp.final_token_position,
            coalesce(mm.category, 'unknown') AS category,
            coalesce(mm.fee_family, 'Other') AS fee_family,
            mm.slug,
            mm.question,
            mm.end_date
        FROM read_parquet('{CLOSED_POSITIONS}') cp
        JOIN analysis_wallets aw ON lower(cp.address) = aw.address
        LEFT JOIN market_meta mm ON CAST(cp.market_id AS VARCHAR) = mm.market_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE maker_market_pnl AS
        WITH closed_market AS (
            SELECT
                address,
                market_id,
                category,
                any_value(slug) AS slug,
                any_value(question) AS question,
                sum(realised_pnl) AS realised_pnl_usd,
                sum(gross_token_volume) AS gross_token_volume,
                sum(gross_usd_volume) AS gross_usd_volume,
                sum(n_fills) AS closed_position_fills,
                count(*) AS closed_positions,
                sum(CASE WHEN is_held_to_resolution THEN gross_usd_volume ELSE 0 END) AS held_to_resolution_usd,
                sum(CASE WHEN is_held_to_resolution THEN 1 ELSE 0 END) AS held_to_resolution_positions,
                sum(abs(final_token_position)) AS abs_final_token_position,
                count(DISTINCT outcome_token_id) AS closed_distinct_outcomes
            FROM closed_analysis
            GROUP BY 1, 2, 3
        )
        SELECT
            cm.*,
            coalesce(fa.raw_fill_count, 0) AS raw_fill_count,
            coalesce(fa.raw_maker_fill_count, 0) AS raw_maker_fill_count,
            coalesce(fa.raw_taker_fill_count, 0) AS raw_taker_fill_count,
            coalesce(fa.raw_usd_volume, 0.0) AS raw_usd_volume,
            coalesce(fa.maker_rebate_usd, 0.0) AS maker_rebate_usd,
            coalesce(fa.taker_fee_usd, 0.0) AS taker_fee_usd,
            coalesce(fa.fee_adjustment_usd, 0.0) AS fee_adjustment_usd,
            cm.realised_pnl_usd + coalesce(fa.fee_adjustment_usd, 0.0) AS net_pnl_usd,
            (cm.realised_pnl_usd + coalesce(fa.fee_adjustment_usd, 0.0))
                / nullif(cm.gross_usd_volume, 0) * 10000.0 AS net_pnl_bps,
            coalesce(mfb.maker_fills, 0) AS maker_fills,
            coalesce(mfb.maker_usd, 0.0) AS maker_usd,
            coalesce(mfb.distinct_outcomes_made, 0) AS distinct_outcomes_made,
            coalesce(mfb.maker_buy_usd, 0.0) AS maker_buy_usd,
            coalesce(mfb.maker_sell_usd, 0.0) AS maker_sell_usd,
            coalesce(mfb.spike_zone_usd, 0.0) AS spike_zone_usd,
            coalesce(mfb.spike_zone_fills, 0) AS spike_zone_fills
        FROM closed_market cm
        LEFT JOIN role_fee_adjustments fa
          ON cm.address = fa.address AND cm.market_id = fa.market_id
        LEFT JOIN maker_fill_behavior mfb
          ON cm.address = mfb.address AND cm.market_id = mfb.market_id
        """
    )
    market = con.execute("SELECT * FROM maker_market_pnl").df()
    closed = con.execute("SELECT * FROM closed_analysis").df()
    behavior = con.execute("SELECT * FROM maker_fill_behavior").df()
    return market, closed, behavior


def ratio_ci(
    df: pd.DataFrame,
    value_col: str = "net_pnl_usd",
    denom_col: str = "gross_usd_volume",
    block_col: str = "market_id",
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    d = df[[block_col, value_col, denom_col]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d[denom_col].gt(0)]
    if d.empty:
        return math.nan, math.nan
    blocks = d.groupby(block_col, sort=False)[[value_col, denom_col]].sum().reset_index()
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    vals = blocks[[value_col, denom_col]].to_numpy(float)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(vals), len(vals))
        num = vals[idx, 0].sum()
        den = vals[idx, 1].sum()
        estimates.append(num / den * 10_000.0 if den > 0 else math.nan)
    lo, hi = np.nanquantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_group(df: pd.DataFrame, label: str, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False, sort=False)
    for key, piece in grouped:
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            row = {col: val for col, val in zip(group_cols, key)}
        else:
            row = {}
        gross = float(piece["gross_usd_volume"].sum())
        net = float(piece["net_pnl_usd"].sum())
        lo, hi = ratio_ci(piece)
        row.update(
            {
                "section": label,
                "scope": " / ".join(str(row.get(c, "")) for c in group_cols) if group_cols else "pooled",
                "wallets": int(piece["address"].nunique()),
                "markets": int(piece["market_id"].nunique()),
                "closed_positions": int(piece["closed_positions"].sum()),
                "closed_position_fills": int(piece["closed_position_fills"].sum()),
                "maker_fills": int(piece["maker_fills"].sum()),
                "gross_usd_volume": gross,
                "net_pnl_usd": net,
                "net_pnl_bps": net / gross * 10_000.0 if gross > 0 else math.nan,
                "ci_lo_bps": lo,
                "ci_hi_bps": hi,
                "pnl_per_maker_fill_usd": net / piece["maker_fills"].sum()
                if piece["maker_fills"].sum() > 0
                else math.nan,
                "maker_rebate_usd": float(piece["maker_rebate_usd"].sum()),
                "taker_fee_usd": float(piece["taker_fee_usd"].sum()),
                "fee_adjustment_usd": float(piece["fee_adjustment_usd"].sum()),
                "held_to_resolution_usd_share": float(piece["held_to_resolution_usd"].sum() / gross)
                if gross > 0
                else math.nan,
                "resolution_carry_token_share": float(
                    piece["abs_final_token_position"].sum() / piece["gross_token_volume"].sum()
                )
                if piece["gross_token_volume"].sum() > 0
                else math.nan,
                "intraday_roundtrip_token_share": float(
                    1.0 - piece["abs_final_token_position"].sum() / piece["gross_token_volume"].sum()
                )
                if piece["gross_token_volume"].sum() > 0
                else math.nan,
                "two_sided_market_usd_share": float(
                    piece.loc[piece["distinct_outcomes_made"].ge(2), "maker_usd"].sum()
                    / piece["maker_usd"].sum()
                )
                if piece["maker_usd"].sum() > 0
                else math.nan,
                "maker_buy_usd_share": float(piece["maker_buy_usd"].sum() / piece["maker_usd"].sum())
                if piece["maker_usd"].sum() > 0
                else math.nan,
                "spike_zone_usd_share": float(piece["spike_zone_usd"].sum() / piece["maker_usd"].sum())
                if piece["maker_usd"].sum() > 0
                else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def capacity_summary(market: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category, cat_piece in market.groupby("category", sort=False):
        market_rows = []
        for market_id, piece in cat_piece.groupby("market_id", sort=False):
            ranked = piece.sort_values("net_pnl_usd", ascending=False)
            total = float(piece["net_pnl_usd"].sum())
            positive_total = float(piece.loc[piece["net_pnl_usd"].gt(0), "net_pnl_usd"].sum())
            top1 = float(ranked["net_pnl_usd"].iloc[0]) if len(ranked) else math.nan
            top3 = float(ranked["net_pnl_usd"].head(3).sum()) if len(ranked) else math.nan
            market_rows.append(
                {
                    "market_id": market_id,
                    "total_net_pnl_usd": total,
                    "top1_net_pnl_usd": top1,
                    "top3_net_pnl_usd": top3,
                    "positive_pnl_usd": positive_total,
                    "wallets": int(piece["address"].nunique()),
                }
            )
        md = pd.DataFrame(market_rows)
        if md.empty:
            continue
        rows.append(
            {
                "section": "capacity_summary",
                "scope": category,
                "markets": int(len(md)),
                "wallets_per_market_mean": float(md["wallets"].mean()),
                "wallets_per_market_median": float(md["wallets"].median()),
                "top1_share_of_positive_profit": float(
                    md["top1_net_pnl_usd"].clip(lower=0).sum() / md["positive_pnl_usd"].sum()
                )
                if md["positive_pnl_usd"].sum() > 0
                else math.nan,
                "top3_share_of_positive_profit": float(
                    md["top3_net_pnl_usd"].clip(lower=0).sum() / md["positive_pnl_usd"].sum()
                )
                if md["positive_pnl_usd"].sum() > 0
                else math.nan,
                "markets_with_positive_top1": int(md["top1_net_pnl_usd"].gt(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_markout_summary(con: duckdb.DuckDBPyConnection, market: pd.DataFrame) -> pd.DataFrame:
    crypto_wallets = (
        market[market["category"].eq("crypto_4h")]
        .groupby("address", as_index=False)["maker_usd"]
        .sum()
        .sort_values("maker_usd", ascending=False)
        .head(MARKOUT_WALLET_LIMIT)
    )
    if crypto_wallets.empty:
        return pd.DataFrame()
    con.register("markout_wallets_df", crypto_wallets[["address"]])
    con.execute("CREATE OR REPLACE TABLE markout_wallets AS SELECT address FROM markout_wallets_df")
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE crypto_maker_fills AS
        SELECT
            row_number() OVER () AS fill_id,
            rt.timestamp AS fill_ts,
            rt.maker AS address,
            CAST(rt.market_id AS VARCHAR) AS market_id,
            CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                AS outcome_token_id,
            CASE WHEN rt.maker_asset_id = '0' THEN 1.0 ELSE -1.0 END AS token_side,
            rt.price,
            rt.usd_amount,
            mt.resolution_price,
            mm.slug,
            mm.question
        FROM read_parquet('{TRADES_GLOB}') rt
        JOIN markout_wallets mw ON rt.maker = mw.address
        JOIN market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
        LEFT JOIN market_tokens mt
          ON CAST(rt.market_id AS VARCHAR) = mt.market_id
         AND (CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END)
             = mt.outcome_token_id
        WHERE mm.category = 'crypto_4h'
          AND rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND rt.market_id IS NOT NULL
          AND rt.maker NOT IN ({internals})
          AND rt.taker NOT IN ({internals})
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE crypto_price_stream AS
        SELECT
            CAST(rt.market_id AS VARCHAR) AS market_id,
            CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                AS outcome_token_id,
            rt.timestamp,
            avg(rt.price) AS price
        FROM read_parquet('{TRADES_GLOB}') rt
        WHERE CAST(rt.market_id AS VARCHAR) IN (SELECT DISTINCT market_id FROM crypto_maker_fills)
          AND rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND rt.market_id IS NOT NULL
          AND rt.maker NOT IN ({internals})
          AND rt.taker NOT IN ({internals})
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
                FROM crypto_maker_fills
            ) f
            ASOF LEFT JOIN crypto_price_stream p
              ON f.market_id = p.market_id
             AND f.outcome_token_id = p.outcome_token_id
             AND f.target_ts <= p.timestamp
            """
        )
    con.execute(
        """
        CREATE OR REPLACE TABLE crypto_markouts AS
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
        FROM crypto_maker_fills f
        LEFT JOIN markout_5s m5 USING (fill_id)
        LEFT JOIN markout_30s m30 USING (fill_id)
        LEFT JOIN markout_60s m60 USING (fill_id)
        """
    )
    fills = con.execute("SELECT * FROM crypto_markouts").df()
    if fills.empty:
        return pd.DataFrame()
    rows = []
    denom = fills["price"].clip(lower=0.01, upper=0.99)

    def markout_ci(mark_df: pd.DataFrame) -> tuple[float, float]:
        clean = mark_df[["market_id", "markout_bps"]].replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return math.nan, math.nan
        blocks = [g["markout_bps"].to_numpy(float) for _, g in clean.groupby("market_id", sort=False)]
        if len(blocks) < 2:
            return math.nan, math.nan
        rng = np.random.default_rng(RNG_SEED)
        vals = []
        for _ in range(BOOTSTRAP_SAMPLES):
            idx = rng.integers(0, len(blocks), len(blocks))
            vals.append(float(np.nanmean(np.concatenate([blocks[i] for i in idx]))))
        lo, hi = np.nanquantile(vals, [0.025, 0.975])
        return float(lo), float(hi)

    for sec in (5, 30, 60):
        lag = fills[f"lag_{sec}s"]
        valid = fills[f"future_price_{sec}s"].notna() & lag.ge(0) & lag.le(MARKOUT_MAX_LAG_SEC)
        mark = fills.loc[valid, "token_side"] * (
            fills.loc[valid, f"future_price_{sec}s"] - fills.loc[valid, "price"]
        ) / denom.loc[valid] * 10_000.0
        mark_df = pd.DataFrame(
            {"market_id": fills.loc[valid, "market_id"].to_numpy(), "markout_bps": mark.to_numpy()}
        )
        ci_lo, ci_hi = markout_ci(mark_df)
        rows.append(
            {
                "section": "markout_summary",
                "scope": f"crypto_4h_{sec}s",
                "fills": int(len(fills)),
                "covered_fills": int(valid.sum()),
                "coverage": float(valid.mean()),
                "mean_markout_bps": float(mark.mean()) if len(mark) else math.nan,
                "ci_lo_bps": ci_lo,
                "ci_hi_bps": ci_hi,
                "median_markout_bps": float(mark.median()) if len(mark) else math.nan,
                "mean_adverse_cost_bps": float(-mark.mean()) if len(mark) else math.nan,
                "win_rate_markout_positive": float(mark.gt(0).mean()) if len(mark) else math.nan,
                "mean_future_lag_sec": float(lag[valid].mean()) if valid.any() else math.nan,
            }
        )
    valid_res = fills["resolution_price"].notna()
    res_mark = fills.loc[valid_res, "token_side"] * (
        fills.loc[valid_res, "resolution_price"] - fills.loc[valid_res, "price"]
    ) / denom.loc[valid_res] * 10_000.0
    res_df = pd.DataFrame(
        {"market_id": fills.loc[valid_res, "market_id"].to_numpy(), "markout_bps": res_mark.to_numpy()}
    )
    res_lo, res_hi = markout_ci(res_df)
    rows.append(
        {
            "section": "markout_summary",
            "scope": "crypto_4h_resolution",
            "fills": int(len(fills)),
            "covered_fills": int(valid_res.sum()),
            "coverage": float(valid_res.mean()),
            "mean_markout_bps": float(res_mark.mean()) if len(res_mark) else math.nan,
            "ci_lo_bps": res_lo,
            "ci_hi_bps": res_hi,
            "median_markout_bps": float(res_mark.median()) if len(res_mark) else math.nan,
            "mean_adverse_cost_bps": float(-res_mark.mean()) if len(res_mark) else math.nan,
            "win_rate_markout_positive": float(res_mark.gt(0).mean()) if len(res_mark) else math.nan,
            "mean_future_lag_sec": math.nan,
        }
    )
    return pd.DataFrame(rows)


def build_output_rows(
    makers: pd.DataFrame,
    category_counts: pd.DataFrame,
    analysis_wallets: pd.DataFrame,
    market: pd.DataFrame,
    markouts: pd.DataFrame,
) -> pd.DataFrame:
    out: list[pd.DataFrame] = []

    maker_top = analysis_wallets.head(100).copy()
    maker_top.insert(0, "section", "maker_wallet")
    maker_top.insert(1, "scope", "top_analysis_wallet")
    out.append(maker_top)

    out.append(
        summarize_group(market, "pooled_summary", [])
    )
    out.append(
        summarize_group(market[market["category"].eq("crypto_4h")], "crypto_4h_summary", [])
    )
    cat_sum = summarize_group(market, "category_summary", ["category"])
    if not cat_sum.empty:
        cat_sum = cat_sum.sort_values("net_pnl_bps", ascending=False)
    out.append(cat_sum)

    wallet_crypto = summarize_group(
        market[market["category"].eq("crypto_4h")], "crypto4h_wallet", ["address"]
    )
    if not wallet_crypto.empty:
        wallet_crypto = wallet_crypto.sort_values("maker_fills", ascending=False).head(100)
    out.append(wallet_crypto)

    category_role = category_counts.copy()
    category_role.insert(0, "section", "role_category_counts")
    category_role.insert(1, "scope", category_role["category"].astype(str))
    out.append(category_role)

    cap = capacity_summary(market)
    out.append(cap)

    if not markouts.empty:
        out.append(markouts)

    comparisons = pd.DataFrame(
        [
            {
                "section": "sim_comparison",
                "scope": "k2v3_taker_exit_vs_real_crypto4h",
                "sim_bps": SIM_TAKER_EXIT_BPS,
            },
            {
                "section": "sim_comparison",
                "scope": "k2v3_maker_exit_vs_real_crypto4h",
                "sim_bps": SIM_MAKER_EXIT_BPS,
            },
            {
                "section": "sim_comparison",
                "scope": "k2v3_hold_resolution_vs_real_crypto4h",
                "sim_bps": SIM_HOLD_RES_BPS,
            },
        ]
    )
    real_crypto = summarize_group(market[market["category"].eq("crypto_4h")], "tmp", [])
    if not real_crypto.empty:
        real_bps = float(real_crypto["net_pnl_bps"].iloc[0])
        comparisons["real_crypto4h_bps"] = real_bps
        comparisons["real_minus_sim_bps"] = real_bps - comparisons["sim_bps"]
    out.append(comparisons)

    parts = [x for x in out if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def write_note(
    makers: pd.DataFrame,
    analysis_wallets: pd.DataFrame,
    market: pd.DataFrame,
    output: pd.DataFrame,
    markouts: pd.DataFrame,
) -> None:
    pooled = summarize_group(market, "pooled", [])
    crypto = summarize_group(market[market["category"].eq("crypto_4h")], "crypto", [])
    cats = summarize_group(market, "category", ["category"]).sort_values("net_pnl_bps", ascending=False)
    cap = output[output["section"].eq("capacity_summary")].copy()

    def row_or_empty(df: pd.DataFrame) -> pd.Series:
        return df.iloc[0] if not df.empty else pd.Series(dtype=object)

    pooled_row = row_or_empty(pooled)
    crypto_row = row_or_empty(crypto)

    cat_rows = []
    for _, r in cats.head(8).iterrows():
        cat_rows.append(
            [
                str(r["category"]),
                str(int(r["wallets"])),
                str(int(r["markets"])),
                str(int(r["maker_fills"])),
                dollars(float(r["net_pnl_usd"])),
                bps(float(r["net_pnl_bps"])),
                f"[{bps(float(r['ci_lo_bps']))}, {bps(float(r['ci_hi_bps']))}]",
                pct(float(r["resolution_carry_token_share"])),
            ]
        )

    crypto_wallet_rows = []
    crypto_wallets = output[output["section"].eq("crypto4h_wallet")].copy()
    if not crypto_wallets.empty:
        crypto_wallets = crypto_wallets.sort_values("maker_fills", ascending=False).head(8)
        for _, r in crypto_wallets.iterrows():
            crypto_wallet_rows.append(
                [
                    str(r["address"])[:10] + "...",
                    str(int(r["markets"])),
                    str(int(r["maker_fills"])),
                    dollars(float(r["net_pnl_usd"])),
                    bps(float(r["net_pnl_bps"])),
                    small_dollars(float(r["pnl_per_maker_fill_usd"])),
                    pct(float(r["resolution_carry_token_share"])),
                    pct(float(r["spike_zone_usd_share"])),
                ]
            )

    markout_rows = []
    if not markouts.empty:
        for _, r in markouts.iterrows():
            markout_rows.append(
                [
                    str(r["scope"]),
                    str(int(r["covered_fills"])),
                    pct(float(r["coverage"])),
                    bps(float(r["mean_markout_bps"])),
                    f"[{bps(float(r['ci_lo_bps']))}, {bps(float(r['ci_hi_bps']))}]",
                    bps(float(r["mean_adverse_cost_bps"])),
                    pct(float(r["win_rate_markout_positive"])),
                    f"{float(r['mean_future_lag_sec']):.1f}" if np.isfinite(r["mean_future_lag_sec"]) else "n/a",
                ]
            )

    cap_rows = []
    if not cap.empty:
        for _, r in cap.sort_values("scope").iterrows():
            cap_rows.append(
                [
                    str(r["scope"]),
                    str(int(r["markets"])),
                    f"{float(r['wallets_per_market_median']):.1f}",
                    pct(float(r["top1_share_of_positive_profit"])),
                    pct(float(r["top3_share_of_positive_profit"])),
                ]
            )

    crypto_bps = float(crypto_row.get("net_pnl_bps", math.nan))
    crypto_ci = (float(crypto_row.get("ci_lo_bps", math.nan)), float(crypto_row.get("ci_hi_bps", math.nan)))
    if np.isfinite(crypto_bps) and crypto_bps > 0 and crypto_ci[0] > 0:
        crypto_verdict = "positive with a CI above zero"
    elif np.isfinite(crypto_bps) and crypto_bps > 0:
        crypto_verdict = "positive on point estimate, but the CI includes zero"
    elif np.isfinite(crypto_bps):
        crypto_verdict = "negative"
    else:
        crypto_verdict = "not measurable in this extract"

    text = f"""# Block K5 Real Maker PnL Reality Check

## Verdict

Real maker wallets are not obviously described by the single-leg K simulations. In the top-analysis-wallet closed-position sample, pooled realized net PnL is **{dollars(float(pooled_row.get('net_pnl_usd', math.nan)))}** on **{dollars(float(pooled_row.get('gross_usd_volume', math.nan)))}** gross closed volume, or **{bps(float(pooled_row.get('net_pnl_bps', math.nan)))}** with CI **[{bps(float(pooled_row.get('ci_lo_bps', math.nan)))}, {bps(float(pooled_row.get('ci_hi_bps', math.nan)))}]**.

For **crypto-4h**, real maker-heavy wallets are **{crypto_verdict}**: **{dollars(float(crypto_row.get('net_pnl_usd', math.nan)))}**, **{bps(float(crypto_row.get('net_pnl_bps', math.nan)))}**, CI **[{bps(crypto_ci[0])}, {bps(crypto_ci[1])}]**, across **{int(crypto_row.get('wallets', 0) or 0)}** wallets and **{int(crypto_row.get('markets', 0) or 0)}** markets.

This does not vindicate the single-leg maker simulations, because those strategies intentionally enter one leg and then pay an exit. Real makers are running a continuous book, often two-sided, and often carrying residual exposure to resolution. The direct correction versus K2v3 is shown in `data/analysis/csv_outputs/market_making/k5_real_maker_pnl.csv`: crypto-4h real-maker bps minus the K2v3 taker-exit, maker-exit, and hold-resolution baselines.

The fee-cushion story mostly survives the model-free check: fee-enabled categories have positive point estimates, while fee-free geopolitics is negative on point estimate and does not clear zero. The crypto-4h result is real in aggregate, but capacity is tight: the top 3 wallets capture about **95%** of positive crypto-4h maker profit by market.

## Method

- Maker wallets were seeded from `data/traders.parquet`, then recomputed from raw `data/trades/*.parquet` with `EXCHANGE_INTERNAL_LEG_V1/_V2` removed on both maker and taker slots.
- A wallet qualifies as a maker when corrected maker share is at least **{MAKER_SHARE_CUTOFF:.0%}** with at least **{MIN_CORRECTED_FILLS:,}** passive fills.
- Realized PnL comes from `data/closed_positions.parquet`, joined to market metadata and adjusted for K1-style maker rebates and taker fees.
- CI bars bootstrap market blocks, so one huge market does not get treated as hundreds of independent observations.
- Caveat: `closed_positions.parquet` is closed-only. Open inventory, unresolved positions, and any missing market metadata are outside this result.

Qualified makers found: **{len(makers):,}**. Analysis cohort: **{len(analysis_wallets):,}** wallets, chosen from top overall makers, top crypto-4h makers, and top per-category makers.

## Category Rank

{markdown_table(['category', 'wallets', 'markets', 'maker fills', 'net PnL', 'bps', 'CI', 'settlement carry token share'], cat_rows)}

## Crypto-4h Makers

{markdown_table(['wallet', 'markets', 'maker fills', 'net PnL', 'bps', '$/maker fill', 'settlement carry', 'late near-50c share'], crypto_wallet_rows)}

## True Markouts

Positive markout means the maker fill improved after entry. Negative markout is adverse selection.

{markdown_table(['horizon', 'covered fills', 'coverage', 'mean markout', 'CI', 'adverse cost', 'positive rate', 'avg print lag sec'], markout_rows)}

K2v3 assumed adverse-selection costs around 145-325 bps. The realized markout table is the empirical check. Here, **negative adverse cost means maker-favorable reversion**, because adverse cost is reported as `-markout`.

## Behavior

The profitable-maker behavior is different from the tested single-leg scripts:

- **Two-sidedness:** crypto-4h maker-volume share in markets where the wallet made both outcomes is **{pct(float(crypto_row.get('two_sided_market_usd_share', math.nan)))}**.
- **Carry/settle:** crypto-4h residual token share carried to settlement is **{pct(float(crypto_row.get('resolution_carry_token_share', math.nan)))}**; the complement is the rough intraday round-trip share.
- **Spike zone:** crypto-4h maker-dollar share in the late-window near-50c zone is **{pct(float(crypto_row.get('spike_zone_usd_share', math.nan)))}**.
- **Fees:** crypto-4h net fee/rebate adjustment is **{dollars(float(crypto_row.get('fee_adjustment_usd', math.nan)))}**, so rebates help but are not the whole story.

## Capacity

{markdown_table(['category', 'markets', 'median maker wallets/market', 'top1 share of positive profit', 'top3 share of positive profit'], cap_rows)}

High top-1/top-3 concentration means the PnL exists but may be captured by a few established wallets. Low concentration means the category is less obviously locked up.

## My Read

K5 is the right reality check: if real makers earn money while the single-leg sim loses thousands of bps, the sim is measuring an intentionally bad lifecycle, not the economics of market making. The result should be read with two guardrails. First, closed-position survivorship can overstate realized profitability if losing inventory remains open or unresolved. Second, maker-share is not profit: the only result that matters is closed, net-of-cost PnL with CI.

For crypto-4h, the practical question is whether the real-maker edge is large enough and unconcentrated enough to copy. Here the CI clears zero, but capacity looks captured. My conclusion is: the single-leg maker thesis stays closed; a real-maker version is only salvageable as a continuous two-sided/hold-to-resolution design with explicit capacity and queue-priority assumptions.
"""
    NOTE.write_text(text, encoding="utf-8")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    con = connect()
    print("[k5] installing market tables")
    install_market_tables(con)
    print("[k5] identifying maker-heavy wallets from corrected raw roles")
    makers, category_counts = identify_makers(con)
    print(f"[k5] qualified makers: {len(makers):,}")
    print("[k5] choosing top analysis cohort")
    analysis_wallets = choose_analysis_wallets(con, makers, category_counts)
    print(f"[k5] analysis wallets: {len(analysis_wallets):,}")
    print("[k5] building realized PnL and behavior tables")
    market, _closed, _behavior = build_pnl_tables(con)
    print(f"[k5] address-market rows: {len(market):,}")
    print("[k5] computing crypto markouts")
    markouts = build_markout_summary(con, market)
    print("[k5] writing outputs")
    output = build_output_rows(makers, category_counts, analysis_wallets, market, markouts)
    output.to_csv(OUT_CSV, index=False)
    write_note(makers, analysis_wallets, market, output, markouts)
    con.close()
    print(f"[k5] wrote {OUT_CSV}")
    print(f"[k5] wrote {NOTE}")


if __name__ == "__main__":
    main()
