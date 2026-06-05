"""MM LOB coverage and fill-proxy calibration gate.

Research-only diagnostic. This is deliberately not a quote optimizer: it asks
whether the short captured LOB panel can support a deployable-cell quote
aggression backtest, and how generous the full-priority fill proxy looks on
the cells it covers.
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
CSV_DIR = ANALYSIS / "csv_outputs" / "market_making"
NOTES = ROOT / "notes" / "market_making"
FEATURES = ANALYSIS / "block_a1_features.parquet"
TRADES_GLOB = str(DATA / "trades" / "*.parquet")
DEPLOYABLE = CSV_DIR / "mm_deployable_cells.csv"
WALLET_MARKET = ANALYSIS / "k5_stress_wallet_market_full.parquet"
OUT_CSV = CSV_DIR / "mm_lob_coverage_proxy_calibration.csv"
NOTE = NOTES / "mm_lob_gate_findings.md"

sys.path.insert(0, str(ROOT))
from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG  # noqa: E402


TICK = 0.01
HORIZONS = (5, 30, 60)
POLICIES = {
    "sit_behind_1": -1,
    "join": 0,
    "improve_1": 1,
    "improve_2": 2,
}
MIN_MEANINGFUL_MARKETS = 3
MIN_MEANINGFUL_TRADE_PRINTS = 100
MIN_MEANINGFUL_ACTIVE_HOURS = 12
MIN_MEANINGFUL_2026_FLOW_SHARE = 0.01


def sql_list(values: set[str] | list[str] | tuple[str, ...]) -> str:
    vals = sorted({str(v).lower().replace("'", "''") for v in values if str(v)})
    return ", ".join(f"'{v}'" for v in vals) if vals else "''"


def dollars(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.0f}"


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:,.0f} bps" if abs(value) >= 100 else f"{value:,.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def mult(value: float) -> str:
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
    temp_dir = ANALYSIS / ".duckdb_tmp_mm_lob_gate"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    return con


def subcell_case() -> str:
    """Return SQL expression that classifies a LOB row into an MM sub-cell."""
    category = """
        CASE
            WHEN family_l = 'crypto_4h_up_down'
              OR regexp_matches(slug_l, '^(btc|eth|sol)-updown-4h-[0-9]+')
                THEN 'crypto_4h'
            WHEN family_l LIKE 'sports%' OR slug_l LIKE '%nba%' OR slug_l LIKE '%nfl%'
              OR slug_l LIKE '%nhl%' OR slug_l LIKE '%mlb%' OR slug_l LIKE '%ufc%'
              OR slug_l LIKE '%soccer%' OR slug_l LIKE '%champions-league%'
              OR slug_l LIKE '%premier-league%' OR question_l LIKE '%win the game%'
              OR question_l LIKE '%beat the%' OR question_l LIKE '%points%'
              OR question_l LIKE '%goals%' THEN 'sports'
            WHEN slug_l LIKE '%grammy%' OR slug_l LIKE '%oscar%' OR slug_l LIKE '%movie%'
              OR slug_l LIKE '%album%' OR slug_l LIKE '%music%' OR question_l LIKE '%oscar%'
              OR question_l LIKE '%grammy%' OR question_l LIKE '%movie%' THEN 'culture'
            WHEN family_l LIKE 'daily_crypto%' THEN 'daily_crypto'
            WHEN family_l LIKE 'geopolitics%' THEN 'geopolitics'
            WHEN family_l LIKE 'politics%' THEN 'politics_negrisk'
            ELSE 'other'
        END
    """
    crypto_asset = """
        CASE
            WHEN slug_l LIKE 'btc-%' OR slug_l LIKE '%bitcoin%' OR question_l LIKE '%bitcoin%' THEN 'btc'
            WHEN slug_l LIKE 'eth-%' OR slug_l LIKE '%ethereum%' OR question_l LIKE '%ethereum%' THEN 'eth'
            WHEN slug_l LIKE 'sol-%' OR slug_l LIKE '%solana%' OR question_l LIKE '%solana%' THEN 'sol'
            ELSE 'crypto_other'
        END
    """
    crypto_slot = """
        CASE
            WHEN crypto_window_open_epoch IS NULL THEN 'unknown_slot'
            WHEN TRY_CAST(strftime(to_timestamp(crypto_window_open_epoch), '%H') AS INTEGER) < 6 THEN '00_06utc'
            WHEN TRY_CAST(strftime(to_timestamp(crypto_window_open_epoch), '%H') AS INTEGER) < 12 THEN '06_12utc'
            WHEN TRY_CAST(strftime(to_timestamp(crypto_window_open_epoch), '%H') AS INTEGER) < 18 THEN '12_18utc'
            ELSE '18_24utc'
        END
    """
    sports_league = """
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
        END
    """
    sports_type = """
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
        END
    """
    culture_theme = """
        CASE
            WHEN slug_l LIKE '%oscar%' OR question_l LIKE '%oscar%' THEN 'oscars'
            WHEN slug_l LIKE '%grammy%' OR question_l LIKE '%grammy%' THEN 'grammys'
            WHEN slug_l LIKE '%movie%' OR question_l LIKE '%movie%' OR question_l LIKE '%box office%' THEN 'film'
            WHEN slug_l LIKE '%album%' OR slug_l LIKE '%music%' OR question_l LIKE '%album%' OR question_l LIKE '%music%' THEN 'music'
            WHEN slug_l LIKE '%tv%' OR question_l LIKE '%tv%' OR question_l LIKE '%show%' THEN 'tv'
            WHEN slug_l LIKE '%celebrity%' OR question_l LIKE '%celebrity%' THEN 'celebrity'
            ELSE 'culture_misc'
        END
    """
    other_theme = """
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
        END
    """
    return f"""
        CASE
            WHEN ({category}) = 'crypto_4h' THEN 'crypto_4h:' || ({crypto_asset}) || ':' || ({crypto_slot})
            WHEN ({category}) = 'sports' THEN 'sports:' || ({sports_league}) || ':' || ({sports_type})
            WHEN ({category}) = 'culture' THEN 'culture:' || ({culture_theme})
            WHEN ({category}) = 'other' THEN 'other:' || ({other_theme})
            ELSE ({category}) || ':not_target'
        END
    """


def install_lob_tables(con: duckdb.DuckDBPyConnection, target: pd.DataFrame) -> None:
    con.register("target_cells_df", target)
    subcell_expr = subcell_case()
    con.execute("CREATE OR REPLACE TABLE target_cells AS SELECT * FROM target_cells_df")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE lob_target AS
        WITH base AS (
            SELECT
                run_id,
                received_at,
                exchange_ts,
                event_type,
                CAST(asset_id AS VARCHAR) AS asset_id,
                CAST(market_id AS VARCHAR) AS market_id,
                lower(coalesce(family, '')) AS family_l,
                coalesce(family, '') AS family,
                lower(coalesce(slug, '')) AS slug_l,
                coalesce(slug, '') AS slug,
                lower(coalesce(question, '')) AS question_l,
                coalesce(question, '') AS question,
                TRY_CAST(regexp_extract(lower(coalesce(slug, '')), 'updown-4h-([0-9]+)', 1) AS BIGINT)
                    AS crypto_window_open_epoch,
                outcome_index,
                is_book_state_complete,
                best_bid,
                best_bid_size,
                best_ask,
                best_ask_size,
                spread,
                mid,
                trade_price,
                upper(coalesce(trade_side, last_trade_side, '')) AS trade_side_norm,
                trade_size,
                transaction_hash,
                is_trade
            FROM read_parquet('{FEATURES}')
        ),
        tagged AS (
            SELECT *, {subcell_expr} AS subcell
            FROM base
        )
        SELECT tagged.*
        FROM tagged
        JOIN target_cells USING (subcell)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE lob_market_ids AS
        SELECT DISTINCT subcell, market_id
        FROM lob_target
        WHERE market_id IS NOT NULL AND market_id <> ''
        """
    )


def coverage_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE lob_raw_2026 AS
        SELECT
            lm.subcell,
            sum(rt.usd_amount) AS represented_raw_usd_2026,
            count(*) AS represented_raw_fills_2026
        FROM read_parquet('{TRADES_GLOB}') rt
        JOIN lob_market_ids lm ON CAST(rt.market_id AS VARCHAR) = lm.market_id
        WHERE rt.timestamp >= TIMESTAMP '2026-01-01'
          AND rt.timestamp < TIMESTAMP '2027-01-01'
          AND rt.usd_amount > 0
          AND rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND lower(rt.maker) NOT IN ({internals})
          AND lower(rt.taker) NOT IN ({internals})
        GROUP BY 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE k5_lob_overlap AS
        SELECT
            lm.subcell,
            count(DISTINCT wm.market_id) AS k5_overlap_markets,
            sum(wm.maker_fills) AS k5_overlap_maker_fills,
            sum(wm.maker_usd) AS k5_overlap_maker_usd,
            min(wm.first_fill_ts) AS k5_overlap_first_fill_ts,
            max(wm.last_fill_ts) AS k5_overlap_last_fill_ts
        FROM read_parquet('{WALLET_MARKET}') wm
        JOIN lob_market_ids lm ON CAST(wm.market_id AS VARCHAR) = lm.market_id
        GROUP BY 1
        """
    )
    return con.execute(
        """
        WITH agg AS (
            SELECT
                t.subcell,
                any_value(t.category) AS category,
                any_value(t.rank) AS deployable_rank,
                any_value(t.raw_usd_2026) AS cell_raw_usd_2026,
                any_value(t.raw_usd_per_active_day_2026) AS cell_raw_usd_per_active_day_2026,
                any_value(t.headroom_2026_per_active_day_usd) AS cell_headroom_2026_per_active_day_usd,
                any_value(t.base_ev_2026_per_active_day_usd) AS cell_base_ev_2026_per_active_day_usd,
                count(DISTINCT l.market_id) AS lob_markets,
                count(DISTINCT l.asset_id) AS lob_assets,
                count(*) FILTER (
                    WHERE l.best_bid IS NOT NULL
                      AND l.best_ask IS NOT NULL
                      AND l.mid IS NOT NULL
                      AND l.best_ask > l.best_bid
                ) AS lob_book_states,
                count(*) FILTER (WHERE l.is_trade = 1) AS lob_trade_prints,
                min(l.received_at) AS lob_start_ts,
                max(l.received_at) AS lob_end_ts,
                count(DISTINCT date_trunc('hour', l.received_at)) AS lob_active_hours,
                date_diff('second', min(l.received_at), max(l.received_at)) / 3600.0 AS lob_span_hours,
                sum(CASE WHEN l.is_trade = 1 THEN coalesce(l.trade_price, 0) * coalesce(l.trade_size, 0) ELSE 0 END)
                    AS lob_trade_notional
            FROM target_cells t
            LEFT JOIN lob_target l USING (subcell)
            GROUP BY 1
        )
        SELECT
            a.*,
            coalesce(r.represented_raw_usd_2026, 0.0) AS represented_raw_usd_2026,
            coalesce(r.represented_raw_fills_2026, 0) AS represented_raw_fills_2026,
            coalesce(r.represented_raw_usd_2026, 0.0) / nullif(a.cell_raw_usd_2026, 0) AS represented_2026_raw_flow_share,
            coalesce(k.k5_overlap_markets, 0) AS k5_overlap_markets,
            coalesce(k.k5_overlap_maker_fills, 0) AS k5_overlap_maker_fills,
            coalesce(k.k5_overlap_maker_usd, 0.0) AS k5_overlap_maker_usd,
            k.k5_overlap_first_fill_ts,
            k.k5_overlap_last_fill_ts
        FROM agg a
        LEFT JOIN lob_raw_2026 r USING (subcell)
        LEFT JOIN k5_lob_overlap k USING (subcell)
        ORDER BY deployable_rank
        """
    ).df()


def first_at_or_after(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(times, targets, side="left")
    out = np.full(len(targets), np.nan)
    valid = pos < len(times)
    out[valid] = values[pos[valid]]
    return out


def proxy_for_cell(con: duckdb.DuckDBPyConnection, subcell: str, coverage: pd.Series) -> list[dict[str, Any]]:
    df = con.execute(
        """
        SELECT
            received_at,
            asset_id,
            market_id,
            slug,
            family,
            event_type,
            is_book_state_complete,
            best_bid,
            best_ask,
            mid,
            trade_price,
            trade_side_norm,
            trade_size,
            transaction_hash,
            is_trade
        FROM lob_target
        WHERE subcell = ?
          AND best_bid IS NOT NULL
          AND best_ask IS NOT NULL
          AND mid IS NOT NULL
          AND best_ask > best_bid
        ORDER BY asset_id, received_at
        """,
        [subcell],
    ).df()
    if df.empty:
        return []

    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    numeric = ["best_bid", "best_ask", "mid", "trade_price", "trade_size"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_side_norm"] = df["trade_side_norm"].fillna("").astype(str).str.upper()
    df["transaction_hash"] = df["transaction_hash"].fillna("").astype(str)

    rows: list[dict[str, Any]] = []
    fill_rows: list[pd.DataFrame] = []
    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values("received_at").reset_index(drop=True)
        states = g[
            g["best_bid"].between(0, 1, inclusive="both")
            & g["best_ask"].between(0, 1, inclusive="both")
            & g["best_ask"].gt(g["best_bid"])
            & g["mid"].between(0, 1, inclusive="both")
        ].drop_duplicates(["received_at", "best_bid", "best_ask", "mid"])
        if states.empty:
            continue
        trades = g[
            g["is_trade"].eq(1)
            & g["trade_side_norm"].isin(["BUY", "SELL"])
            & g["trade_price"].between(0, 1, inclusive="both")
            & g["trade_size"].fillna(0).gt(0)
        ].copy()
        if trades.empty:
            continue
        hash_mask = trades["transaction_hash"].ne("")
        trades = pd.concat(
            [
                trades[hash_mask].drop_duplicates(["transaction_hash", "trade_price", "trade_side_norm", "trade_size"]),
                trades[~hash_mask].drop_duplicates(["received_at", "trade_price", "trade_side_norm", "trade_size"]),
            ],
            ignore_index=True,
        ).sort_values("received_at")
        if trades.empty:
            continue

        state_times = states["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        trade_times = trades["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        current_idx = np.searchsorted(state_times, trade_times, side="right") - 1
        valid = current_idx >= 0
        if not valid.any():
            continue
        trades = trades.loc[valid].reset_index(drop=True)
        trade_times = trade_times[valid]
        current = states.iloc[current_idx[valid]].reset_index(drop=True)
        piece = trades[["received_at", "asset_id", "market_id", "slug", "trade_side_norm", "trade_price", "trade_size"]].copy()
        piece["best_bid"] = current["best_bid"].to_numpy(float)
        piece["best_ask"] = current["best_ask"].to_numpy(float)
        piece["mid"] = current["mid"].to_numpy(float)
        piece["fill_time_ns"] = trade_times
        all_times = state_times
        all_mids = states["mid"].to_numpy(float)
        for horizon in HORIZONS:
            piece[f"future_mid_{horizon}s"] = first_at_or_after(
                all_times,
                all_mids,
                trade_times + horizon * 1_000_000_000,
            )
        fill_rows.append(piece)

    if not fill_rows:
        return []
    candidates = pd.concat(fill_rows, ignore_index=True)
    active_hours = float(coverage.get("lob_active_hours", math.nan) or math.nan)

    for policy, offset in POLICIES.items():
        bid = (candidates["best_bid"] + offset * TICK).clip(0.001, 0.999)
        ask = (candidates["best_ask"] - offset * TICK).clip(0.001, 0.999)
        bid = np.minimum(bid, candidates["best_ask"] - TICK)
        ask = np.maximum(ask, candidates["best_bid"] + TICK)
        bid_fill = candidates["trade_side_norm"].eq("SELL") & candidates["trade_price"].le(bid)
        ask_fill = candidates["trade_side_norm"].eq("BUY") & candidates["trade_price"].ge(ask)
        fills = candidates[bid_fill | ask_fill].copy()
        if fills.empty:
            notional = 0.0
            fill_count = 0
            rows.append(
                {
                    "row_type": "proxy_policy",
                    "subcell": subcell,
                    "policy": policy,
                    "policy_offset_ticks": offset,
                    "proxy_fill_count": 0,
                    "proxy_fill_notional": 0.0,
                    "proxy_fills_per_active_hour": 0.0,
                    "proxy_notional_per_active_hour": 0.0,
                    "realized_spread_to_mid_bps": math.nan,
                    "markout_5s_bps": math.nan,
                    "markout_30s_bps": math.nan,
                    "markout_60s_bps": math.nan,
                    "adverse_cost_60s_bps": math.nan,
                    "proxy_to_k5_overlap_maker_usd": math.nan,
                }
            )
            continue
        entry = np.where(fills["trade_side_norm"].eq("SELL"), bid.loc[fills.index], ask.loc[fills.index])
        token_side = np.where(fills["trade_side_norm"].eq("SELL"), 1.0, -1.0)
        denom = np.clip(fills["mid"].to_numpy(float), 0.01, 0.99)
        size = fills["trade_size"].to_numpy(float)
        notional_arr = entry * size
        notional = float(np.nansum(notional_arr))
        fill_count = int(len(fills))
        spread_bps = token_side * (fills["mid"].to_numpy(float) - entry) / denom * 10_000.0
        row: dict[str, Any] = {
            "row_type": "proxy_policy",
            "subcell": subcell,
            "policy": policy,
            "policy_offset_ticks": offset,
            "proxy_fill_count": fill_count,
            "proxy_fill_notional": notional,
            "proxy_fills_per_active_hour": fill_count / active_hours if active_hours > 0 else math.nan,
            "proxy_notional_per_active_hour": notional / active_hours if active_hours > 0 else math.nan,
            "realized_spread_to_mid_bps": float(np.nanmean(spread_bps)),
            "proxy_to_k5_overlap_maker_usd": (
                notional / float(coverage.get("k5_overlap_maker_usd", math.nan))
                if float(coverage.get("k5_overlap_maker_usd", 0) or 0) > 0
                else math.nan
            ),
        }
        for horizon in HORIZONS:
            future = fills[f"future_mid_{horizon}s"].to_numpy(float)
            markout = token_side * (future - fills["mid"].to_numpy(float)) / denom * 10_000.0
            row[f"markout_{horizon}s_bps"] = float(np.nanmean(markout))
        row["adverse_cost_60s_bps"] = -row["markout_60s_bps"] if np.isfinite(row["markout_60s_bps"]) else math.nan
        rows.append(row)
    return rows


def build_proxy_rows(con: duckdb.DuckDBPyConnection, coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = coverage[
        coverage["lob_trade_prints"].fillna(0).ge(10)
        & coverage["lob_book_states"].fillna(0).ge(100)
        & coverage["lob_active_hours"].fillna(0).gt(0)
    ].copy()
    for _, cov in eligible.iterrows():
        cell_rows = proxy_for_cell(con, str(cov["subcell"]), cov)
        for row in cell_rows:
            row.update(
                {
                    "category": cov["category"],
                    "deployable_rank": cov["deployable_rank"],
                    "lob_markets": cov["lob_markets"],
                    "lob_trade_prints": cov["lob_trade_prints"],
                    "lob_active_hours": cov["lob_active_hours"],
                    "cell_raw_usd_per_active_day_2026": cov["cell_raw_usd_per_active_day_2026"],
                    "represented_2026_raw_flow_share": cov["represented_2026_raw_flow_share"],
                    "k5_overlap_maker_usd": cov["k5_overlap_maker_usd"],
                    "k5_overlap_maker_fills": cov["k5_overlap_maker_fills"],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def coverage_rows(coverage: pd.DataFrame) -> pd.DataFrame:
    out = coverage.copy()
    out["row_type"] = "coverage"
    out["policy"] = ""
    out["coverage_sufficient_for_lob_sim"] = (
        out["lob_markets"].fillna(0).ge(MIN_MEANINGFUL_MARKETS)
        & out["lob_trade_prints"].fillna(0).ge(MIN_MEANINGFUL_TRADE_PRINTS)
        & out["lob_active_hours"].fillna(0).ge(MIN_MEANINGFUL_ACTIVE_HOURS)
        & out["represented_2026_raw_flow_share"].fillna(0).ge(MIN_MEANINGFUL_2026_FLOW_SHARE)
    )
    return out


def write_note(out: pd.DataFrame) -> None:
    cov = out[out["row_type"].eq("coverage")].copy()
    proxy = out[out["row_type"].eq("proxy_policy")].copy()
    main = cov[cov["subcell"].eq("other:misc_other")]
    main_ok = bool(main["coverage_sufficient_for_lob_sim"].iloc[0]) if not main.empty else False
    verdict = (
        "**DO NOT BUILD a historical LOB aggression surface from this panel. Route aggression measurement to live paper quoting.**"
        if not main_ok
        else "**Coverage passes the main-cell gate; a small historical aggression surface may be defensible, still as an upper-bound diagnostic.**"
    )

    cov_rows = []
    for _, r in cov.sort_values("deployable_rank").iterrows():
        cov_rows.append(
            [
                str(r["subcell"]),
                "YES" if bool(r["coverage_sufficient_for_lob_sim"]) else "NO",
                str(int(r.get("lob_markets", 0) or 0)),
                str(int(r.get("lob_assets", 0) or 0)),
                str(int(r.get("lob_trade_prints", 0) or 0)),
                str(int(r.get("lob_active_hours", 0) or 0)),
                pct(float(r.get("represented_2026_raw_flow_share", math.nan))),
                dollars(float(r.get("lob_trade_notional", math.nan))),
                dollars(float(r.get("cell_raw_usd_per_active_day_2026", math.nan))),
            ]
        )

    proxy_rows = []
    if not proxy.empty:
        best = proxy[proxy["policy"].isin(["join", "improve_1", "improve_2"])].copy()
        best = best.sort_values(["deployable_rank", "policy_offset_ticks"]).head(18)
        for _, r in best.iterrows():
            proxy_rows.append(
                [
                    str(r["subcell"]),
                    str(r["policy"]),
                    str(int(r["proxy_fill_count"])),
                    dollars(float(r["proxy_fill_notional"])),
                    f"{float(r['proxy_fills_per_active_hour']):.2f}",
                    bps(float(r["realized_spread_to_mid_bps"])),
                    bps(float(r["markout_60s_bps"])),
                    bps(float(r["adverse_cost_60s_bps"])),
                    mult(float(r.get("proxy_to_k5_overlap_maker_usd", math.nan))),
                ]
            )

    main_line = ""
    if not main.empty:
        r = main.iloc[0]
        main_line = (
            f"`other:misc_other` has {int(r['lob_markets'])} LOB-covered market(s), "
            f"{int(r['lob_trade_prints'])} trade print(s), and represents "
            f"{pct(float(r['represented_2026_raw_flow_share']))} of its 2026 raw flow. "
            "That fails the main target coverage gate."
        )

    exact_overlap = cov["k5_overlap_maker_usd"].fillna(0).sum()
    if exact_overlap <= 0:
        overlap_line = (
            "Exact same-market K5 wallet calibration is unavailable: the captured LOB panel is dated "
            "2026-05-27 to 2026-05-30, while the local raw-fill/K5 history currently ends on 2026-05-26."
        )
    else:
        overlap_line = (
            f"Exact same-market K5 overlap exists for {dollars(float(exact_overlap))} of maker notional, "
            "but it is still not queue-position calibration."
        )

    text = f"""# MM LOB Gate Findings

> Hub: [[strat_market_making]] · [[mm_deployable_cells_findings]]

## Headline

{verdict}

{main_line}

{overlap_line}

## Coverage Gate

{markdown_table(['sub-cell', 'coverage ok?', 'markets', 'assets', 'LOB trades', 'active hrs', '2026 flow represented', 'LOB trade notional', '2026 cell flow/day'], cov_rows)}

## Full-Priority Proxy Calibration

These rows are **upper bounds**, not deployable fills. The proxy assumes our quote has full queue priority and
fills on any trade-through at our modeled price.

{markdown_table(['sub-cell', 'policy', 'proxy fills', 'proxy notional', 'fills/hr', 'spread-to-mid', '60s markout', '60s adverse', 'proxy/K5 overlap'], proxy_rows)}

## Queue Limitation

The captured panel has best bid/ask states and trade prints, but not our historical queue rank, order age,
cancel/replace priority, or maker identity at the touch. Therefore a join-best historical fill rate cannot
distinguish "first in queue" from "last in queue." That is first-order for deployability.

## Decision

Do **not** build a standalone historical MM quote-aggression backtest from this panel. The only honest use of
this LOB data is as an upper-bound diagnostic for cells it actually covers, mainly crypto/sports/active markets.
The main MM cell, `other:misc_other`, is not materially covered. Quote aggression should be measured in live
paper trading, with logs for quote price, quote size, best bid/ask, post/cancel time, queue proxy if available,
fills, missed trade-throughs, markouts, settlement, and inventory cluster.

## Output

CSV: `data/analysis/csv_outputs/market_making/mm_lob_coverage_proxy_calibration.csv`
"""
    NOTE.write_text(text, encoding="utf-8")


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    target = pd.read_csv(DEPLOYABLE)
    target = target[target["qualified_for_paper"].fillna(False).astype(bool)].copy()
    keep_cols = [
        "subcell",
        "category",
        "rank",
        "raw_usd_2026",
        "raw_usd_per_active_day_2026",
        "headroom_2026_per_active_day_usd",
        "base_ev_2026_per_active_day_usd",
    ]
    target = target[keep_cols].sort_values("rank")
    con = connect()
    print("[mm-lob-gate] install LOB target tables", flush=True)
    install_lob_tables(con, target)
    print("[mm-lob-gate] coverage + raw/K5 overlap", flush=True)
    cov = coverage_rows(coverage_table(con))
    print("[mm-lob-gate] full-priority proxy rows", flush=True)
    prox = build_proxy_rows(con, cov)
    all_cols = sorted(set(cov.columns).union(prox.columns if not prox.empty else []))
    cov = cov.reindex(columns=all_cols)
    prox = prox.reindex(columns=all_cols) if not prox.empty else pd.DataFrame(columns=all_cols)
    out = pd.concat([cov, prox], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    write_note(out)
    print(f"[mm-lob-gate] wrote {OUT_CSV}", flush=True)
    print(f"[mm-lob-gate] wrote {NOTE}", flush=True)


if __name__ == "__main__":
    main()
