"""MM deployable-cell targeting from K5-STRESS.

Research-only sidecar. This turns category-level K5-STRESS passes into a
ranked paper-trade target list. It is intentionally MM-namespaced, not K9:
the task belongs to market-making execution/lifecycle, not the OD valuation
layer.

Inputs are the K5-STRESS full-population caches:
  - k5_stress_wallet_market_full.parquet
  - k5_stress.csv

The screen keeps the K5-STRESS rules:
  - full ex-ante maker population
  - all inventory marked
  - structured playbook defined ex ante
  - exclude each market's global top-3 makers
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
REPO = ROOT.parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
CSV_DIR = ANALYSIS / "csv_outputs" / "market_making"
NOTES = ROOT / "notes" / "market_making"
TRADES_GLOB = str(DATA / "trades" / "*.parquet")
MARKETS = sorted((DATA / "markets").glob("markets_*.parquet"))[-1]
K5_STRESS_CSV = CSV_DIR / "k5_stress.csv"
WALLET_MARKET = ANALYSIS / "k5_stress_wallet_market_full.parquet"
OUT_CSV = CSV_DIR / "mm_deployable_cells.csv"
NOTE = NOTES / "mm_deployable_cells_findings.md"

sys.path.insert(0, str(ROOT))
from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG  # noqa: E402


RNG_SEED = 20260601
BOOTSTRAP_SAMPLES = 400
PASS_CATEGORIES = ("crypto_4h", "culture", "other", "sports")
STRUCT_TWO_SIDED_MIN = 0.60
STRUCT_CARRY_MIN = 0.50
STRUCT_SPIKE_MAX = 0.02
MONTH_POSITIVE_MIN = 2.0 / 3.0
OPEN_TOP3_MAX = 0.60
OPEN_MIN_STRUCTURED_MAKERS = 10
OPEN_MIN_NON_TOP3_FLOW_USD = 25_000.0
SPECIFIC_EDGE_MIN_BPS = 0.0


def sql_list(values: set[str] | list[str] | tuple[str, ...]) -> str:
    vals = sorted({str(v).lower().replace("'", "''") for v in values if str(v)})
    return ", ".join(f"'{v}'" for v in vals) if vals else "''"


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


def multiple(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f}x"


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


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    temp_dir = ANALYSIS / ".duckdb_tmp_mm_deployable_cells"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    return con


def install_market_meta(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE market_meta_cells AS
        WITH base AS (
            SELECT
                CAST(id AS VARCHAR) AS market_id,
                lower(coalesce(slug, '')) AS slug_l,
                coalesce(slug, '') AS slug,
                lower(coalesce(question, '')) AS question_l,
                coalesce(question, '') AS question,
                TRY_CAST(end_date AS TIMESTAMP) AS end_ts,
                TRY_CAST(regexp_extract(lower(coalesce(slug, '')), 'updown-4h-([0-9]+)', 1) AS BIGINT)
                    AS crypto_window_open_epoch
            FROM read_parquet('{MARKETS}')
        ),
        tagged AS (
            SELECT
                *,
                CASE
                    WHEN slug_l LIKE 'btc-%' OR slug_l LIKE '%bitcoin%' OR question_l LIKE '%bitcoin%' THEN 'btc'
                    WHEN slug_l LIKE 'eth-%' OR slug_l LIKE '%ethereum%' OR question_l LIKE '%ethereum%' THEN 'eth'
                    WHEN slug_l LIKE 'sol-%' OR slug_l LIKE '%solana%' OR question_l LIKE '%solana%' THEN 'sol'
                    ELSE 'crypto_other'
                END AS crypto_asset,
                CASE
                    WHEN crypto_window_open_epoch IS NULL THEN 'unknown_slot'
                    WHEN TRY_CAST(strftime(to_timestamp(crypto_window_open_epoch), '%H') AS INTEGER) < 6 THEN '00_06utc'
                    WHEN TRY_CAST(strftime(to_timestamp(crypto_window_open_epoch), '%H') AS INTEGER) < 12 THEN '06_12utc'
                    WHEN TRY_CAST(strftime(to_timestamp(crypto_window_open_epoch), '%H') AS INTEGER) < 18 THEN '12_18utc'
                    ELSE '18_24utc'
                END AS crypto_slot,
                CASE
                    WHEN slug_l LIKE '%nfl%' OR question_l LIKE '%nfl%' THEN 'nfl'
                    WHEN slug_l LIKE '%nba%' OR question_l LIKE '%nba%' THEN 'nba'
                    WHEN slug_l LIKE '%mlb%' OR question_l LIKE '%mlb%' THEN 'mlb'
                    WHEN slug_l LIKE '%nhl%' OR question_l LIKE '%nhl%' THEN 'nhl'
                    WHEN slug_l LIKE '%ufc%' OR question_l LIKE '%ufc%' THEN 'ufc'
                    WHEN slug_l LIKE '%wnba%' OR question_l LIKE '%wnba%' THEN 'wnba'
                    WHEN slug_l LIKE '%champions-league%' OR slug_l LIKE '%premier-league%'
                      OR slug_l LIKE '%soccer%' OR question_l LIKE '%soccer%' THEN 'soccer'
                    WHEN slug_l LIKE '%tennis%' OR question_l LIKE '%tennis%' THEN 'tennis'
                    WHEN slug_l LIKE '%golf%' OR question_l LIKE '%golf%' THEN 'golf'
                    WHEN slug_l LIKE '%ncaa%' OR question_l LIKE '%college%' THEN 'college'
                    ELSE 'sports_other'
                END AS sports_league,
                CASE
                    WHEN slug_l LIKE '%player%' OR question_l LIKE '%rebounds%' OR question_l LIKE '%assists%'
                      OR question_l LIKE '%yards%' OR question_l LIKE '%touchdown%' OR question_l LIKE '%goals%'
                      OR question_l LIKE '%points%' THEN 'player_prop'
                    WHEN slug_l LIKE '%spread%' OR question_l LIKE '%spread%' OR question_l LIKE '%handicap%' THEN 'spread'
                    WHEN slug_l LIKE '%total%' OR question_l LIKE '% total%' OR question_l LIKE '%over %'
                      OR question_l LIKE '%under %' THEN 'total'
                    WHEN slug_l LIKE '%champion%' OR question_l LIKE '%champion%' OR question_l LIKE '%win the 20%'
                      OR question_l LIKE '%win the league%' OR question_l LIKE '%win the tournament%' THEN 'outright'
                    WHEN question_l LIKE '% win%' OR question_l LIKE '%beat %' OR slug_l LIKE '%-vs-%' THEN 'moneyline'
                    ELSE 'market_other'
                END AS sports_market_type,
                CASE
                    WHEN slug_l LIKE '%oscar%' OR question_l LIKE '%oscar%' THEN 'oscars'
                    WHEN slug_l LIKE '%grammy%' OR question_l LIKE '%grammy%' THEN 'grammys'
                    WHEN slug_l LIKE '%movie%' OR question_l LIKE '%movie%' OR question_l LIKE '%box office%' THEN 'film'
                    WHEN slug_l LIKE '%album%' OR slug_l LIKE '%music%' OR question_l LIKE '%album%' OR question_l LIKE '%music%' THEN 'music'
                    WHEN slug_l LIKE '%tv%' OR question_l LIKE '%tv%' OR question_l LIKE '%show%' THEN 'tv'
                    WHEN slug_l LIKE '%celebrity%' OR question_l LIKE '%celebrity%' THEN 'celebrity'
                    ELSE 'culture_misc'
                END AS culture_theme,
                CASE
                    WHEN slug_l LIKE '%election%' OR question_l LIKE '%election%' OR slug_l LIKE '%trump%'
                      OR question_l LIKE '%trump%' OR slug_l LIKE '%president%' OR question_l LIKE '%president%' THEN 'politics_regular'
                    WHEN slug_l LIKE '%fed%' OR question_l LIKE '%inflation%' OR slug_l LIKE '%cpi%'
                      OR question_l LIKE '%recession%' THEN 'macro_econ_misc'
                    WHEN slug_l LIKE '%ai%' OR question_l LIKE '%openai%' OR question_l LIKE '%nvidia%' THEN 'tech_ai_misc'
                    WHEN slug_l LIKE '%weather%' OR question_l LIKE '%weather%' OR question_l LIKE '%temperature%' THEN 'weather_misc'
                    WHEN slug_l LIKE '%crypto%' OR question_l LIKE '%bitcoin%' OR question_l LIKE '%ethereum%' THEN 'crypto_misc'
                    WHEN slug_l LIKE '%earnings%' OR question_l LIKE '%stock%' OR slug_l LIKE '%ipo%' THEN 'finance_misc'
                    ELSE 'misc_other'
                END AS other_theme
            FROM base
        )
        SELECT * FROM tagged
        """
    )
    pass_sql = sql_list(PASS_CATEGORIES)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE wm_enriched AS
        SELECT
            wm.*,
            mm.slug,
            mm.question,
            CASE
                WHEN wm.category = 'crypto_4h' THEN 'crypto_4h:' || mm.crypto_asset || ':' || mm.crypto_slot
                WHEN wm.category = 'sports' THEN 'sports:' || mm.sports_league || ':' || mm.sports_market_type
                WHEN wm.category = 'culture' THEN 'culture:' || mm.culture_theme
                WHEN wm.category = 'other' THEN 'other:' || mm.other_theme
                ELSE wm.category || ':misc'
            END AS subcell,
            CASE
                WHEN wm.category = 'crypto_4h' THEN mm.crypto_asset || ':' || coalesce(strftime(mm.end_ts, '%Y-%m-%d'), 'unknown_day')
                WHEN wm.category = 'sports' THEN mm.sports_league || ':' || regexp_replace(mm.slug_l, '-(yes|no)$', '')
                WHEN wm.category = 'culture' THEN mm.culture_theme || ':' || coalesce(strftime(mm.end_ts, '%Y-%m'), 'unknown_month')
                WHEN wm.category = 'other' THEN mm.other_theme || ':' || coalesce(strftime(mm.end_ts, '%Y-%m'), 'unknown_month')
                ELSE wm.category || ':unknown'
            END AS cluster_tag
        FROM read_parquet('{WALLET_MARKET}') wm
        LEFT JOIN market_meta_cells mm USING (market_id)
        WHERE wm.category IN ({pass_sql})
        """
    )


def build_cell_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE wallet_cell_non_top3 AS
        WITH grouped AS (
            SELECT
                address,
                category,
                subcell,
                count(DISTINCT market_id) AS markets,
                sum(gross_usd_volume) AS gross_usd_volume,
                sum(net_pnl_usd) AS net_pnl_usd,
                sum(base_pnl_usd) AS base_pnl_usd,
                sum(maker_rebate_usd) AS maker_rebate_usd,
                sum(taker_fee_usd) AS taker_fee_usd,
                sum(net_pnl_usd - maker_rebate_usd) AS net_without_rebate_usd,
                sum(gross_token_volume) AS gross_token_volume,
                sum(abs_final_token_position) AS abs_final_token_position,
                sum(maker_fills) AS maker_fills,
                sum(maker_usd) AS maker_usd,
                sum(maker_sell_usd) AS maker_sell_usd,
                sum(maker_buy_usd) AS maker_buy_usd,
                sum(CASE WHEN distinct_outcomes_made >= 2 THEN maker_usd ELSE 0 END) AS two_sided_maker_usd,
                sum(spike_zone_usd) AS spike_zone_usd
            FROM wm_enriched
            WHERE NOT is_global_top3_market_maker
            GROUP BY 1, 2, 3
        )
        SELECT
            *,
            net_pnl_usd / nullif(gross_usd_volume, 0) * 10000.0 AS net_pnl_bps,
            net_without_rebate_usd / nullif(gross_usd_volume, 0) * 10000.0 AS net_without_rebate_bps,
            two_sided_maker_usd / nullif(maker_usd, 0) AS two_sided_usd_share,
            abs_final_token_position / nullif(gross_token_volume, 0) AS carry_token_share,
            spike_zone_usd / nullif(maker_usd, 0) AS spike_zone_usd_share,
            maker_sell_usd / nullif(maker_usd, 0) AS maker_sell_usd_share,
            (
                two_sided_maker_usd / nullif(maker_usd, 0) >= {STRUCT_TWO_SIDED_MIN}
                AND abs_final_token_position / nullif(gross_token_volume, 0) >= {STRUCT_CARRY_MIN}
                AND coalesce(spike_zone_usd / nullif(maker_usd, 0), 0.0) <= {STRUCT_SPIKE_MAX}
                AND maker_fills > 0
            ) AS structured_playbook
        FROM grouped
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE structured_non_top3_market AS
        SELECT e.*
        FROM wm_enriched e
        JOIN wallet_cell_non_top3 wc
          ON e.address = wc.address
         AND e.category = wc.category
         AND e.subcell = wc.subcell
        WHERE wc.structured_playbook
          AND NOT e.is_global_top3_market_maker
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE cell_market_blocks AS
        SELECT
            category,
            subcell,
            market_id,
            any_value(cluster_tag) AS cluster_tag,
            sum(gross_usd_volume) AS gross_usd_volume,
            sum(net_pnl_usd) AS net_pnl_usd,
            sum(base_pnl_usd) AS base_pnl_usd,
            sum(maker_rebate_usd) AS maker_rebate_usd,
            sum(taker_fee_usd) AS taker_fee_usd,
            sum(net_pnl_usd - maker_rebate_usd) AS net_without_rebate_usd,
            sum(maker_fills) AS maker_fills,
            sum(maker_usd) AS maker_usd,
            sum(maker_sell_usd) AS maker_sell_usd,
            sum(global_maker_usd) AS global_maker_usd
        FROM structured_non_top3_market
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE cell_month_blocks AS
        SELECT
            category,
            subcell,
            strftime(mark_ts, '%Y-%m') AS month,
            sum(gross_usd_volume) AS gross_usd_volume,
            sum(net_pnl_usd) AS net_pnl_usd
        FROM structured_non_top3_market
        WHERE mark_ts IS NOT NULL
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE cell_capacity AS
        SELECT
            category,
            subcell,
            count(DISTINCT market_id) AS capacity_markets,
            count(DISTINCT address) AS observed_qualified_makers,
            count(DISTINCT CASE WHEN is_global_top3_market_maker THEN address END) AS observed_top3_makers,
            sum(global_maker_usd) AS observed_maker_usd,
            sum(CASE WHEN global_market_maker_rank <= 3 THEN global_maker_usd ELSE 0 END) AS top3_maker_usd,
            sum(CASE WHEN global_market_maker_rank > 3 THEN global_maker_usd ELSE 0 END) AS non_top3_maker_usd,
            sum(global_maker_fills) AS observed_maker_fills,
            sum(CASE WHEN global_market_maker_rank <= 3 THEN global_maker_fills ELSE 0 END) AS top3_maker_fills,
            sum(CASE WHEN global_market_maker_rank > 3 THEN global_maker_fills ELSE 0 END) AS non_top3_maker_fills
        FROM wm_enriched
        GROUP BY 1, 2
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE cell_cluster AS
        WITH c AS (
            SELECT
                category,
                subcell,
                cluster_tag,
                sum(net_pnl_usd) AS cluster_net_pnl_usd,
                sum(gross_usd_volume) AS cluster_gross_usd_volume
            FROM structured_non_top3_market
            GROUP BY 1, 2, 3
        )
        SELECT
            category,
            subcell,
            count(*) AS clusters,
            arg_max(cluster_tag, cluster_gross_usd_volume) AS dominant_cluster_tag,
            min(cluster_net_pnl_usd) AS worst_cluster_pnl_usd,
            sum(CASE WHEN cluster_net_pnl_usd > 0 THEN cluster_net_pnl_usd ELSE 0 END) AS positive_cluster_pnl_usd,
            abs(min(cluster_net_pnl_usd)) / nullif(sum(CASE WHEN cluster_net_pnl_usd > 0 THEN cluster_net_pnl_usd ELSE 0 END), 0)
                AS worst_cluster_loss_to_positive_profit
        FROM c
        GROUP BY 1, 2
        """
    )


def build_raw_flow(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE raw_flow_subcell AS
        WITH scoped AS (
            SELECT
                mm.subcell,
                mm.category,
                rt.maker,
                rt.taker,
                rt.maker_asset_id,
                rt.price,
                rt.usd_amount,
                rt.token_amount,
                rt.timestamp
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN (
                SELECT DISTINCT market_id, category, subcell
                FROM wm_enriched
            ) mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
              AND rt.usd_amount > 0
        ),
        taker_rank AS (
            SELECT
                category,
                subcell,
                taker,
                sum(usd_amount) AS taker_usd
            FROM scoped
            GROUP BY 1, 2, 3
        ),
        taker_top AS (
            SELECT
                category,
                subcell,
                sum(taker_usd) AS taker_usd,
                sum(CASE WHEN rnk <= 3 THEN taker_usd ELSE 0 END) AS top3_taker_usd
            FROM (
                SELECT
                    *,
                    row_number() OVER (PARTITION BY category, subcell ORDER BY taker_usd DESC) AS rnk
                FROM taker_rank
            )
            GROUP BY 1, 2
        )
        SELECT
            'all' AS period,
            NULL::INTEGER AS flow_year,
            s.category,
            s.subcell,
            count(*) AS raw_fills,
            count(DISTINCT s.maker) AS raw_makers,
            count(DISTINCT s.taker) AS raw_takers,
            min(s.timestamp) AS flow_start_ts,
            max(s.timestamp) AS flow_end_ts,
            count(DISTINCT CAST(s.timestamp AS DATE)) AS active_flow_days,
            sum(s.usd_amount) AS raw_usd,
            sum(CASE WHEN s.maker_asset_id <> '0' THEN s.usd_amount ELSE 0 END) AS maker_sell_usd,
            sum(CASE WHEN s.maker_asset_id = '0' THEN s.usd_amount ELSE 0 END) AS maker_buy_usd,
            sum(CASE WHEN s.maker_asset_id <> '0' AND s.price BETWEEN 0.005 AND 0.20 THEN s.usd_amount ELSE 0 END)
                AS tail_buy_demand_usd,
            tt.top3_taker_usd / nullif(tt.taker_usd, 0) AS top3_taker_share
        FROM scoped s
        LEFT JOIN taker_top tt USING (category, subcell)
        GROUP BY s.category, s.subcell, tt.top3_taker_usd, tt.taker_usd
        UNION ALL
        SELECT
            strftime(s.timestamp, '%Y') AS period,
            TRY_CAST(strftime(s.timestamp, '%Y') AS INTEGER) AS flow_year,
            s.category,
            s.subcell,
            count(*) AS raw_fills,
            count(DISTINCT s.maker) AS raw_makers,
            count(DISTINCT s.taker) AS raw_takers,
            min(s.timestamp) AS flow_start_ts,
            max(s.timestamp) AS flow_end_ts,
            count(DISTINCT CAST(s.timestamp AS DATE)) AS active_flow_days,
            sum(s.usd_amount) AS raw_usd,
            sum(CASE WHEN s.maker_asset_id <> '0' THEN s.usd_amount ELSE 0 END) AS maker_sell_usd,
            sum(CASE WHEN s.maker_asset_id = '0' THEN s.usd_amount ELSE 0 END) AS maker_buy_usd,
            sum(CASE WHEN s.maker_asset_id <> '0' AND s.price BETWEEN 0.005 AND 0.20 THEN s.usd_amount ELSE 0 END)
                AS tail_buy_demand_usd,
            NULL::DOUBLE AS top3_taker_share
        FROM scoped s
        GROUP BY s.category, s.subcell, strftime(s.timestamp, '%Y')
        """
    )
    return con.execute("SELECT * FROM raw_flow_subcell").df()


def ratio_ci(blocks: pd.DataFrame) -> tuple[float, float]:
    d = blocks[["market_id", "net_pnl_usd", "gross_usd_volume"]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d["gross_usd_volume"].gt(0)]
    if len(d["market_id"].unique()) < 2:
        return math.nan, math.nan
    b = d.groupby("market_id", sort=False)[["net_pnl_usd", "gross_usd_volume"]].sum().reset_index()
    vals = b[["net_pnl_usd", "gross_usd_volume"]].to_numpy(float)
    rng = np.random.default_rng(RNG_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(vals), len(vals))
        den = vals[idx, 1].sum()
        estimates.append(vals[idx, 0].sum() / den * 10_000 if den > 0 else math.nan)
    lo, hi = np.nanquantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def summarize(con: duckdb.DuckDBPyConnection, raw_flow: pd.DataFrame) -> pd.DataFrame:
    market = con.execute("SELECT * FROM cell_market_blocks").df()
    wallets = con.execute("SELECT * FROM wallet_cell_non_top3 WHERE structured_playbook").df()
    months = con.execute("SELECT * FROM cell_month_blocks").df()
    capacity = con.execute("SELECT * FROM cell_capacity").df()
    cluster = con.execute("SELECT * FROM cell_cluster").df()
    raw_all = raw_flow[raw_flow["period"].eq("all")].copy()
    raw_year = raw_flow[raw_flow["period"].ne("all")].copy()
    parent = pd.read_csv(K5_STRESS_CSV, low_memory=False)
    parent = parent[
        parent["section"].eq("deployability_non_top3")
        & parent["structured_playbook"].fillna(False).astype(bool)
    ][["category", "net_pnl_bps", "wallet_median_net_pnl_bps"]].rename(
        columns={
            "net_pnl_bps": "parent_structured_net_bps",
            "wallet_median_net_pnl_bps": "parent_structured_median_bps",
        }
    )

    rows: list[dict[str, Any]] = []
    for (category, subcell), piece in market.groupby(["category", "subcell"], sort=False):
        gross = float(piece["gross_usd_volume"].sum())
        net = float(piece["net_pnl_usd"].sum())
        base = float(piece["base_pnl_usd"].sum())
        rebate = float(piece["maker_rebate_usd"].sum())
        taker_fee = float(piece["taker_fee_usd"].sum())
        no_rebate = float(piece["net_without_rebate_usd"].sum())
        ci_lo, ci_hi = ratio_ci(piece)
        w = wallets[(wallets["category"].eq(category)) & (wallets["subcell"].eq(subcell))].copy()
        m = months[(months["category"].eq(category)) & (months["subcell"].eq(subcell))].copy()
        m["month_bps"] = m["net_pnl_usd"] / m["gross_usd_volume"].replace(0, np.nan) * 10_000
        cap = capacity[(capacity["category"].eq(category)) & (capacity["subcell"].eq(subcell))]
        caprow = cap.iloc[0] if not cap.empty else pd.Series(dtype=object)
        cl = cluster[(cluster["category"].eq(category)) & (cluster["subcell"].eq(subcell))]
        clrow = cl.iloc[0] if not cl.empty else pd.Series(dtype=object)
        flow = raw_all[(raw_all["category"].eq(category)) & (raw_all["subcell"].eq(subcell))]
        flowrow = flow.iloc[0] if not flow.empty else pd.Series(dtype=object)
        flow_year = raw_year[(raw_year["category"].eq(category)) & (raw_year["subcell"].eq(subcell))]
        flow_2026 = flow_year[flow_year["flow_year"].eq(2026)]
        flow_2025 = flow_year[flow_year["flow_year"].eq(2025)]
        flowrow_2026 = flow_2026.iloc[0] if not flow_2026.empty else pd.Series(dtype=object)
        flowrow_2025 = flow_2025.iloc[0] if not flow_2025.empty else pd.Series(dtype=object)

        top3_share = float(caprow.get("top3_maker_usd", math.nan)) / float(
            caprow.get("observed_maker_usd", math.nan)
        ) if float(caprow.get("observed_maker_usd", 0) or 0) > 0 else math.nan
        active_structured = int(w["address"].nunique())
        non_top3_flow = float(caprow.get("non_top3_maker_usd", math.nan))
        if np.isfinite(top3_share) and top3_share <= OPEN_TOP3_MAX and active_structured >= OPEN_MIN_STRUCTURED_MAKERS and non_top3_flow >= OPEN_MIN_NON_TOP3_FLOW_USD:
            capacity_status = "open"
        elif np.isfinite(top3_share) and top3_share <= 0.80 and active_structured >= 5:
            capacity_status = "constrained"
        else:
            capacity_status = "captured"

        if capacity_status == "open":
            entrant_capture_rate = min(0.05, max(0.0025, (1.0 - top3_share) / max(active_structured + 1, 1) * 0.5))
        else:
            entrant_capture_rate = 0.0
        headroom_flow = non_top3_flow * entrant_capture_rate if np.isfinite(non_top3_flow) else math.nan

        mean_bps = net / gross * 10_000 if gross > 0 else math.nan
        median_bps = float(w["net_pnl_bps"].median()) if not w.empty else math.nan
        q25_bps = float(w["net_pnl_bps"].quantile(0.25)) if not w.empty else math.nan
        q75_bps = float(w["net_pnl_bps"].quantile(0.75)) if not w.empty else math.nan
        no_rebate_bps = no_rebate / gross * 10_000 if gross > 0 else math.nan
        months_positive = float(m["month_bps"].gt(0).mean()) if not m.empty else math.nan
        active_months = int(m["month"].nunique()) if not m.empty else 0

        parent_row = parent[parent["category"].eq(category)]
        parent_mean = float(parent_row["parent_structured_net_bps"].iloc[0]) if not parent_row.empty else math.nan
        parent_median = float(parent_row["parent_structured_median_bps"].iloc[0]) if not parent_row.empty else math.nan
        above_parent_bps = mean_bps - parent_mean if np.isfinite(mean_bps) and np.isfinite(parent_mean) else math.nan
        subcell_specific_edge = bool(
            np.isfinite(above_parent_bps)
            and above_parent_bps > SPECIFIC_EDGE_MIN_BPS
            and np.isfinite(median_bps)
            and median_bps > 0
        )

        tail_buy_share = float(flowrow.get("tail_buy_demand_usd", math.nan)) / float(
            flowrow.get("raw_usd", math.nan)
        ) if float(flowrow.get("raw_usd", 0) or 0) > 0 else math.nan
        maker_sell_share = float(flowrow.get("maker_sell_usd", math.nan)) / float(
            flowrow.get("raw_usd", math.nan)
        ) if float(flowrow.get("raw_usd", 0) or 0) > 0 else math.nan
        if mean_bps > 0 and median_bps > 0 and np.isfinite(tail_buy_share) and tail_buy_share >= 0.05 and months_positive >= MONTH_POSITIVE_MIN:
            flow_confidence = "high"
        elif mean_bps > 0 and median_bps > 0 and months_positive >= 0.50:
            flow_confidence = "medium"
        else:
            flow_confidence = "low"

        base_ev = headroom_flow * median_bps / 10_000 if np.isfinite(headroom_flow) and np.isfinite(median_bps) else math.nan
        optimistic_ev = headroom_flow * mean_bps / 10_000 if np.isfinite(headroom_flow) and np.isfinite(mean_bps) else math.nan
        active_flow_days = int(flowrow.get("active_flow_days", 0) or 0)
        headroom_per_active_day = headroom_flow / active_flow_days if active_flow_days > 0 and np.isfinite(headroom_flow) else math.nan
        base_ev_per_active_day = base_ev / active_flow_days if active_flow_days > 0 and np.isfinite(base_ev) else math.nan
        optimistic_ev_per_active_day = (
            optimistic_ev / active_flow_days if active_flow_days > 0 and np.isfinite(optimistic_ev) else math.nan
        )
        raw_usd = float(flowrow.get("raw_usd", math.nan)) if not flow.empty else math.nan
        raw_usd_per_active_day = raw_usd / active_flow_days if active_flow_days > 0 and np.isfinite(raw_usd) else math.nan
        raw_usd_2026 = float(flowrow_2026.get("raw_usd", math.nan)) if not flow_2026.empty else math.nan
        active_flow_days_2026 = int(flowrow_2026.get("active_flow_days", 0) or 0)
        raw_usd_per_active_day_2026 = (
            raw_usd_2026 / active_flow_days_2026
            if active_flow_days_2026 > 0 and np.isfinite(raw_usd_2026)
            else math.nan
        )
        raw_usd_2025 = float(flowrow_2025.get("raw_usd", math.nan)) if not flow_2025.empty else math.nan
        active_flow_days_2025 = int(flowrow_2025.get("active_flow_days", 0) or 0)
        raw_usd_per_active_day_2025 = (
            raw_usd_2025 / active_flow_days_2025
            if active_flow_days_2025 > 0 and np.isfinite(raw_usd_2025)
            else math.nan
        )
        flow_growth_2026_vs_all = (
            raw_usd_per_active_day_2026 / raw_usd_per_active_day
            if np.isfinite(raw_usd_per_active_day_2026)
            and np.isfinite(raw_usd_per_active_day)
            and raw_usd_per_active_day > 0
            else math.nan
        )
        flow_growth_2026_vs_2025 = (
            raw_usd_per_active_day_2026 / raw_usd_per_active_day_2025
            if np.isfinite(raw_usd_per_active_day_2026)
            and np.isfinite(raw_usd_per_active_day_2025)
            and raw_usd_per_active_day_2025 > 0
            else math.nan
        )
        headroom_raw_share = headroom_flow / raw_usd if np.isfinite(headroom_flow) and np.isfinite(raw_usd) and raw_usd > 0 else math.nan
        headroom_2026_flow_usd = raw_usd_2026 * headroom_raw_share if np.isfinite(raw_usd_2026) and np.isfinite(headroom_raw_share) else math.nan
        headroom_2026_per_active_day = (
            headroom_2026_flow_usd / active_flow_days_2026
            if active_flow_days_2026 > 0 and np.isfinite(headroom_2026_flow_usd)
            else math.nan
        )
        base_ev_2026_flow_usd = (
            headroom_2026_flow_usd * median_bps / 10_000
            if np.isfinite(headroom_2026_flow_usd) and np.isfinite(median_bps)
            else math.nan
        )
        base_ev_2026_per_active_day = (
            headroom_2026_per_active_day * median_bps / 10_000
            if np.isfinite(headroom_2026_per_active_day) and np.isfinite(median_bps)
            else math.nan
        )
        optimistic_ev_2026_per_active_day = (
            headroom_2026_per_active_day * mean_bps / 10_000
            if np.isfinite(headroom_2026_per_active_day) and np.isfinite(mean_bps)
            else math.nan
        )
        qualified = bool(
            np.isfinite(ci_lo)
            and ci_lo > 0
            and np.isfinite(median_bps)
            and median_bps > 0
            and capacity_status == "open"
            and np.isfinite(months_positive)
            and months_positive >= MONTH_POSITIVE_MIN
            and np.isfinite(no_rebate_bps)
            and no_rebate_bps > 0
            and subcell_specific_edge
        )

        rows.append(
            {
                "section": "deployable_cell",
                "category": category,
                "subcell": subcell,
                "qualified_for_paper": qualified,
                "rank_metric_base_ev_usd": base_ev if qualified else 0.0,
                "rank_metric_base_ev_2026_per_active_day_usd": (
                    base_ev_2026_per_active_day if qualified and np.isfinite(base_ev_2026_per_active_day) else 0.0
                ),
                "structured_non_top3_wallets": active_structured,
                "markets": int(piece["market_id"].nunique()),
                "gross_usd_volume": gross,
                "net_pnl_usd": net,
                "structured_non_top3_net_bps": mean_bps,
                "ci_lo_bps": ci_lo,
                "ci_hi_bps": ci_hi,
                "median_structured_wallet_bps": median_bps,
                "q25_structured_wallet_bps": q25_bps,
                "q75_structured_wallet_bps": q75_bps,
                "net_without_rebate_bps": no_rebate_bps,
                "base_pnl_usd": base,
                "maker_rebate_usd": rebate,
                "taker_fee_usd": taker_fee,
                "parent_structured_net_bps": parent_mean,
                "parent_structured_median_bps": parent_median,
                "above_parent_bps": above_parent_bps,
                "subcell_specific_edge": subcell_specific_edge,
                "active_months": active_months,
                "positive_month_share": months_positive,
                "capacity_status": capacity_status,
                "observed_top3_maker_usd_share": top3_share,
                "observed_non_top3_maker_usd": non_top3_flow,
                "entrant_capture_rate": entrant_capture_rate,
                "headroom_flow_usd": headroom_flow,
                "base_case_ev_usd": base_ev,
                "optimistic_ev_usd": optimistic_ev,
                "raw_fills": int(flowrow.get("raw_fills", 0) or 0),
                "raw_usd": raw_usd,
                "raw_makers": int(flowrow.get("raw_makers", 0) or 0),
                "raw_takers": int(flowrow.get("raw_takers", 0) or 0),
                "flow_start_ts": str(flowrow.get("flow_start_ts", "")) if not flow.empty else "",
                "flow_end_ts": str(flowrow.get("flow_end_ts", "")) if not flow.empty else "",
                "active_flow_days": active_flow_days,
                "raw_usd_per_active_day": raw_usd_per_active_day,
                "headroom_per_active_day_usd": headroom_per_active_day,
                "base_ev_per_active_day_usd": base_ev_per_active_day,
                "optimistic_ev_per_active_day_usd": optimistic_ev_per_active_day,
                "raw_usd_2026": raw_usd_2026,
                "active_flow_days_2026": active_flow_days_2026,
                "raw_usd_per_active_day_2026": raw_usd_per_active_day_2026,
                "raw_usd_2025": raw_usd_2025,
                "active_flow_days_2025": active_flow_days_2025,
                "raw_usd_per_active_day_2025": raw_usd_per_active_day_2025,
                "flow_growth_2026_vs_all": flow_growth_2026_vs_all,
                "flow_growth_2026_vs_2025": flow_growth_2026_vs_2025,
                "headroom_raw_share": headroom_raw_share,
                "headroom_2026_flow_usd": headroom_2026_flow_usd,
                "headroom_2026_per_active_day_usd": headroom_2026_per_active_day,
                "base_ev_2026_flow_usd": base_ev_2026_flow_usd,
                "base_ev_2026_per_active_day_usd": base_ev_2026_per_active_day,
                "optimistic_ev_2026_per_active_day_usd": optimistic_ev_2026_per_active_day,
                "tail_buy_demand_usd_share": tail_buy_share,
                "maker_sell_usd_share": maker_sell_share,
                "top3_taker_share": float(flowrow.get("top3_taker_share", math.nan)) if not flow.empty else math.nan,
                "uninformed_flow_confidence": flow_confidence,
                "cluster_tag": str(clrow.get("dominant_cluster_tag", "")) if not cl.empty else "",
                "clusters": int(clrow.get("clusters", 0) or 0),
                "worst_cluster_pnl_usd": float(clrow.get("worst_cluster_pnl_usd", math.nan)) if not cl.empty else math.nan,
                "worst_cluster_loss_to_positive_profit": float(
                    clrow.get("worst_cluster_loss_to_positive_profit", math.nan)
                )
                if not cl.empty
                else math.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["qualified_for_paper", "rank_metric_base_ev_2026_per_active_day_usd", "structured_non_top3_net_bps"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def write_note(out: pd.DataFrame) -> None:
    qualified = out[out["qualified_for_paper"]].copy()
    top = out.head(15).copy()

    rows = []
    for _, r in top.iterrows():
        rows.append(
            [
                str(int(r["rank"])),
                str(r["subcell"]),
                "YES" if bool(r["qualified_for_paper"]) else "NO",
                bps(float(r["structured_non_top3_net_bps"])),
                f"[{bps(float(r['ci_lo_bps']))}, {bps(float(r['ci_hi_bps']))}]",
                bps(float(r["median_structured_wallet_bps"])),
                str(r["capacity_status"]),
                dollars(float(r["raw_usd_per_active_day_2026"])),
                dollars(float(r["headroom_2026_per_active_day_usd"])),
                dollars(float(r["base_ev_2026_per_active_day_usd"])),
                multiple(float(r["flow_growth_2026_vs_all"])),
                str(r["uninformed_flow_confidence"]),
                pct(float(r["positive_month_share"])),
            ]
        )

    qual_rows = []
    for _, r in qualified.iterrows():
        qual_rows.append(
            [
                str(int(r["rank"])),
                str(r["subcell"]),
                bps(float(r["median_structured_wallet_bps"])),
                dollars(float(r["raw_usd_per_active_day_2026"])),
                dollars(float(r["headroom_2026_per_active_day_usd"])),
                dollars(float(r["base_ev_2026_per_active_day_usd"])),
                dollars(float(r["base_ev_2026_per_active_day_usd"]) * 30.0),
                multiple(float(r["flow_growth_2026_vs_all"])),
                pct(float(r["observed_top3_maker_usd_share"])),
                str(int(r["structured_non_top3_wallets"])),
                str(r["cluster_tag"])[:80],
            ]
        )

    verdict = (
        f"**{len(qualified)} sub-cells qualify for MM paper-trading.**"
        if len(qualified)
        else "**No sub-cell qualifies for MM paper-trading.**"
    )
    if len(qualified):
        base_ev = float(qualified["base_case_ev_usd"].sum())
        opt_ev = float(qualified["optimistic_ev_usd"].sum())
        headroom = float(qualified["headroom_flow_usd"].sum())
        base_ev_2026_day = float(qualified["base_ev_2026_per_active_day_usd"].sum())
        headroom_2026_day = float(qualified["headroom_2026_per_active_day_usd"].sum())
        verdict += (
            f" Historical median-based open-headroom EV is **{dollars(base_ev)}** "
            f"on **{dollars(headroom)}** capturable flow, but the deployable run-rate should use 2026 flow: "
            f"**{dollars(base_ev_2026_day)}/active day** on **{dollars(headroom_2026_day)}/active day** estimated headroom "
            f"(~**{dollars(base_ev_2026_day * 30.0)}** per 30 active days). Mean-based historical optimistic EV is **{dollars(opt_ev)}**."
        )

    text = f"""# MM Deployable Cells Findings

> Hub: [[strat_market_making]] · [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Headline

{verdict}

This is the MM-track replacement for the draft "K9" prompt. It does **not** introduce OD valuation, Binance
fair value, vol, or directional skew. It only asks where the K5-STRESS real-maker playbook can be paper-tested:
two-sided passive making, carry-to-resolution, spike-zone avoidance, and no incumbent top-3 dependence.

Important window note: this screen does **not** use the A1/A0 order-book capture panel (roughly 48h in the
maker-sim work). It uses the K5-STRESS historical raw fills plus closed/open position reconstruction. Local raw
fills currently span 2022-11-21 to 2026-05-26. Because flow is much larger now than in the early history, the
paper-trade run-rate uses **2026 raw flow/day** as the baseline. Historical `base EV` is kept in the CSV for
auditability, but table ranking is by `base_ev_2026_per_active_day_usd`.

## Design

Input: `data/analysis/csv_outputs/market_making/k5_stress.csv` plus the cached full marked wallet-market table
`data/analysis/k5_stress_wallet_market_full.parquet`.

The unit of observation in the output CSV is one deployable MM sub-cell. A sub-cell is a category split such as
`sports:nba:moneyline`, `culture:oscars`, or `crypto_4h:btc:12_18utc`.

For every sub-cell I recomputed the K5-STRESS structured non-top3 result:

- structured playbook: two-sided USD share >= {STRUCT_TWO_SIDED_MIN:.0%}, carry-token share >= {STRUCT_CARRY_MIN:.0%}, spike-zone share <= {STRUCT_SPIKE_MAX:.0%}
- non-incumbent: exclude each market's global top-3 maker wallets
- paper gate: CI lower > 0, median structured wallet > 0, capacity open, positive in >= 2/3 active months, net without rebate > 0, and sub-cell edge above the parent average

## How To Read The Main Table

- `net bps`: aggregate structured non-top3 PnL per gross dollar in the cell.
- `CI`: market-block bootstrap confidence interval.
- `median`: the base-case EV rate; this avoids letting a few right-tail wallets define expected returns.
- `capacity`: open/constrained/captured based on observed top-3 maker share, active structured makers, and non-top3 flow.
- `2026 flow/day`: raw sub-cell fill notional per active 2026 day.
- `2026 headroom/day`: 2026 flow/day times the historical capturable-headroom share.
- `2026 base EV/day`: 2026 headroom/day times the median structured maker bps.
- `growth`: 2026 raw flow/day divided by the full-history raw flow/day.

{markdown_table(['rank', 'sub-cell', 'paper?', 'net bps', 'CI', 'median', 'capacity', '2026 flow/day', '2026 headroom/day', '2026 base EV/day', 'growth', 'flow conf', '+months'], rows)}

## Paper-Qualified Cells

{markdown_table(['rank', 'sub-cell', 'median bps', '2026 flow/day', '2026 headroom/day', '2026 base EV/day', '30d base EV', 'growth', 'top3 share', 'structured wallets', 'cluster tag'], qual_rows)}

## Interpretation

The median-based EV is intentionally much smaller than the category-level K5-STRESS dollar totals. That is the
point: entering the market changes capacity, and the historical non-top3 aggregate is an upper bound, not our
expected fill share. The practical sizing number is the **2026 base EV/day** column, not the older historical
total.

The cells that survive are the ones where the edge is not merely "the parent category was good." They have
positive market-block CI, positive typical structured wallet, enough non-incumbent flow, and month stability.
Cells with high mean bps but captured capacity are not paper targets.

## Guardrails

- This remains historical research, not a live bot statement.
- Capacity uses realized maker-flow proxies because historical order/cancel queues are not in the owned fill layer.
- The flow-confidence label is diagnostic, not a causal proof of retail uninformed flow.
- Any paper test must log live quotes, cancels, queue position, and missed fills; otherwise we cannot tell whether
failure is capacity, speed, or bad quoting.

## Output

CSV: `data/analysis/csv_outputs/market_making/mm_deployable_cells.csv`
"""
    NOTE.write_text(text, encoding="utf-8")


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    con = connect()
    print("[mm-cells] install market metadata", flush=True)
    install_market_meta(con)
    print("[mm-cells] build structured/non-top3 cell tables", flush=True)
    build_cell_tables(con)
    print("[mm-cells] build raw flow/counterparty reads", flush=True)
    raw_flow = build_raw_flow(con)
    print(f"[mm-cells] raw flow rows: {len(raw_flow):,}", flush=True)
    print("[mm-cells] summarize/rank", flush=True)
    out = summarize(con, raw_flow)
    out.to_csv(OUT_CSV, index=False)
    write_note(out)
    print(f"[mm-cells] wrote {OUT_CSV}", flush=True)
    print(f"[mm-cells] wrote {NOTE}", flush=True)


if __name__ == "__main__":
    main()
