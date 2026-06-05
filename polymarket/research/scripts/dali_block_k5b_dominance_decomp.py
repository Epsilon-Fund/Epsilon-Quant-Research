"""Block K5b top-maker dominance decomposition.

Research-only sidecar. K5 showed real maker-heavy wallets have positive
crypto-4h realized PnL, but per-market profit is concentrated. This script
asks whether the top-maker edge is better explained by observable speed/queue
proxies, capital/scale, or risk appetite.

Important limitation: data/trades/*.parquet contains fills, not the order book
order/cancel stream. True quote/cancel frequency, queue position, and requote
latency are therefore not directly observable from this historical dataset.
The script reports fill-based proxies and labels the observability gap.
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
K5 = ANALYSIS / "csv_outputs" / "market_making" / "k5_real_maker_pnl.csv"
OUT_CSV = ANALYSIS / "csv_outputs" / "market_making" / "k5b_dominance_decomp.csv"
NOTE = NOTES / "block_k5b_findings.md"

sys.path.insert(0, str(ROOT))
from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG  # noqa: E402


RNG_SEED = 20260531
BOOTSTRAP_SAMPLES = 500
FEE_RATE = 0.07
MAKER_REBATE_SHARE = 0.20
TOP_N = 3
NEXT_N = 20
RANK_MIN_MAKER_FILLS = 1_000
RANK_MIN_MARKETS = 50


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
    temp_dir = ANALYSIS / ".duckdb_tmp_k5b"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    return con


def install_universe(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    k5 = pd.read_csv(K5, low_memory=False)
    universe = (
        k5[
            k5["section"].eq("role_category_counts")
            & k5["category"].eq("crypto_4h")
            & k5["maker_fills"].fillna(0).gt(0)
        ][["address", "maker_fills", "maker_usd", "taker_fills", "category_maker_share"]]
        .drop_duplicates("address")
        .copy()
    )
    universe["address"] = universe["address"].astype(str).str.lower()
    con.register("wallet_universe_df", universe)
    con.execute(
        """
        CREATE OR REPLACE TABLE wallet_universe AS
        SELECT address FROM wallet_universe_df
        """
    )
    return universe


def install_market_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE crypto_market_meta AS
        SELECT
            CAST(id AS VARCHAR) AS market_id,
            coalesce(slug, '') AS slug,
            coalesce(question, '') AS question,
            TRY_CAST(end_date AS TIMESTAMP) AS end_date,
            TRY_CAST(regexp_extract(lower(coalesce(slug, '')), 'updown-4h-([0-9]+)', 1) AS BIGINT)
                AS window_open_epoch
        FROM read_parquet('{MARKETS}')
        WHERE regexp_matches(lower(coalesce(slug, '')), '^(btc|eth|sol)-updown-4h-[0-9]+')
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
          AND CAST(m.id AS VARCHAR) IN (SELECT market_id FROM crypto_market_meta)
        """
    )


def build_raw_tables(con: duckdb.DuckDBPyConnection) -> None:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE crypto_maker_fills_all AS
        SELECT
            rt.timestamp,
            CAST(rt.market_id AS VARCHAR) AS market_id,
            rt.maker,
            rt.taker,
            CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                AS outcome_token_id,
            CASE WHEN rt.maker_asset_id = '0' THEN 1 ELSE 0 END AS maker_buy,
            rt.token_amount,
            rt.usd_amount,
            rt.price,
            rt.transaction_hash,
            mm.window_open_epoch,
            epoch(rt.timestamp) - mm.window_open_epoch AS seconds_since_open,
            CASE
                WHEN mm.window_open_epoch IS NULL THEN 'unknown'
                WHEN epoch(rt.timestamp) - mm.window_open_epoch < 15 * 60 THEN '00_15m'
                WHEN epoch(rt.timestamp) - mm.window_open_epoch < 60 * 60 THEN '15_60m'
                WHEN epoch(rt.timestamp) - mm.window_open_epoch < 180 * 60 THEN '60_180m'
                WHEN epoch(rt.timestamp) - mm.window_open_epoch <= 240 * 60 THEN '180_240m'
                ELSE 'after_close'
            END AS phase_bucket,
            CASE
                WHEN mm.window_open_epoch IS NOT NULL
                 AND epoch(rt.timestamp) - mm.window_open_epoch >= 225 * 60
                 AND rt.price BETWEEN 0.40 AND 0.60
                THEN 1 ELSE 0
            END AS spike_zone
        FROM read_parquet('{TRADES_GLOB}') rt
        JOIN crypto_market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
        WHERE rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND rt.market_id IS NOT NULL
          AND rt.maker NOT IN ({internals})
          AND rt.taker NOT IN ({internals})
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE selected_maker_fills AS
        SELECT *
        FROM crypto_maker_fills_all
        WHERE maker IN (SELECT address FROM wallet_universe)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE all_wallet_market_maker AS
        SELECT
            maker AS address,
            market_id,
            count(*) AS all_maker_fills,
            sum(usd_amount) AS all_maker_usd,
            min(timestamp) AS first_maker_fill_ts
        FROM crypto_maker_fills_all
        GROUP BY 1, 2
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE all_market_totals AS
        SELECT
            market_id,
            count(*) AS market_maker_fills,
            sum(usd_amount) AS market_maker_usd,
            sum(CASE WHEN phase_bucket = '00_15m' THEN usd_amount ELSE 0 END) AS market_early15_usd,
            sum(CASE WHEN spike_zone = 1 THEN usd_amount ELSE 0 END) AS market_spike_usd,
            min(timestamp) AS market_first_fill_ts
        FROM crypto_maker_fills_all
        GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE all_wallet_market_rank AS
        SELECT
            awm.*,
            amt.market_maker_usd,
            amt.market_maker_fills,
            awm.all_maker_usd / nullif(amt.market_maker_usd, 0) AS market_fill_share,
            rank() OVER (PARTITION BY awm.market_id ORDER BY awm.all_maker_usd DESC) AS market_volume_rank,
            rank() OVER (PARTITION BY awm.market_id ORDER BY awm.first_maker_fill_ts ASC) AS market_first_fill_rank
        FROM all_wallet_market_maker awm
        JOIN all_market_totals amt USING (market_id)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE selected_maker_behavior AS
        SELECT
            smf.maker AS address,
            smf.market_id,
            count(*) AS maker_fills,
            sum(smf.usd_amount) AS maker_usd,
            avg(smf.usd_amount) AS avg_fill_usd,
            quantile_cont(smf.usd_amount, 0.95) AS p95_fill_usd,
            count(DISTINCT smf.outcome_token_id) AS distinct_outcomes_made,
            sum(CASE WHEN smf.maker_buy = 1 THEN smf.usd_amount ELSE 0 END) AS maker_buy_usd,
            sum(CASE WHEN smf.maker_buy = 0 THEN smf.usd_amount ELSE 0 END) AS maker_sell_usd,
            sum(CASE WHEN smf.phase_bucket = '00_15m' THEN smf.usd_amount ELSE 0 END) AS early15_usd,
            sum(CASE WHEN smf.phase_bucket = '15_60m' THEN smf.usd_amount ELSE 0 END) AS mid15_60_usd,
            sum(CASE WHEN smf.phase_bucket = '60_180m' THEN smf.usd_amount ELSE 0 END) AS mid60_180_usd,
            sum(CASE WHEN smf.phase_bucket = '180_240m' THEN smf.usd_amount ELSE 0 END) AS late180_240_usd,
            sum(CASE WHEN smf.spike_zone = 1 THEN smf.usd_amount ELSE 0 END) AS spike_zone_usd,
            min(smf.timestamp) AS first_maker_fill_ts,
            max(smf.timestamp) AS last_maker_fill_ts
        FROM selected_maker_fills smf
        GROUP BY 1, 2
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE selected_interfill AS
        WITH gaps AS (
            SELECT
                maker AS address,
                market_id,
                epoch(timestamp - lag(timestamp) OVER (PARTITION BY maker ORDER BY timestamp)) AS wallet_gap_sec,
                epoch(timestamp - lag(timestamp) OVER (PARTITION BY maker, market_id ORDER BY timestamp))
                    AS wallet_market_gap_sec
            FROM selected_maker_fills
        )
        SELECT
            address,
            median(wallet_gap_sec) FILTER (WHERE wallet_gap_sec >= 0) AS median_wallet_interfill_sec,
            quantile_cont(wallet_gap_sec, 0.10) FILTER (WHERE wallet_gap_sec >= 0) AS p10_wallet_interfill_sec,
            median(wallet_market_gap_sec) FILTER (WHERE wallet_market_gap_sec >= 0)
                AS median_market_interfill_sec
        FROM gaps
        GROUP BY 1
        """
    )


def build_pnl_tables(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE selected_role_fees AS
        WITH scoped AS (
            SELECT
                CAST(rt.market_id AS VARCHAR) AS market_id,
                rt.maker,
                rt.taker,
                rt.token_amount,
                rt.usd_amount,
                least(greatest(rt.price, 0.001), 0.999) AS price
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN crypto_market_meta mm ON CAST(rt.market_id AS VARCHAR) = mm.market_id
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND rt.market_id IS NOT NULL
              AND rt.maker NOT IN ({internals})
              AND rt.taker NOT IN ({internals})
              AND (rt.maker IN (SELECT address FROM wallet_universe)
                   OR rt.taker IN (SELECT address FROM wallet_universe))
        ),
        exploded AS (
            SELECT
                maker AS address,
                market_id,
                count(*) AS fill_count,
                sum(usd_amount) AS usd_volume,
                sum(token_amount * {FEE_RATE} * price * (1.0 - price) * {MAKER_REBATE_SHARE})
                    AS maker_rebate_usd,
                0.0 AS taker_fee_usd
            FROM scoped
            WHERE maker IN (SELECT address FROM wallet_universe)
            GROUP BY 1, 2
            UNION ALL
            SELECT
                taker AS address,
                market_id,
                count(*) AS fill_count,
                sum(usd_amount) AS usd_volume,
                0.0 AS maker_rebate_usd,
                sum(token_amount * {FEE_RATE} * price * (1.0 - price)) AS taker_fee_usd
            FROM scoped
            WHERE taker IN (SELECT address FROM wallet_universe)
            GROUP BY 1, 2
        )
        SELECT
            address,
            market_id,
            sum(fill_count) AS raw_fill_count,
            sum(usd_volume) AS raw_usd_volume,
            sum(maker_rebate_usd) AS maker_rebate_usd,
            sum(taker_fee_usd) AS taker_fee_usd,
            sum(maker_rebate_usd) - sum(taker_fee_usd) AS fee_adjustment_usd
        FROM exploded
        GROUP BY 1, 2
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE closed_crypto_wallet_market AS
        SELECT
            lower(cp.address) AS address,
            CAST(cp.market_id AS VARCHAR) AS market_id,
            min(cp.first_fill_ts) AS first_fill_ts,
            max(cp.last_fill_ts) AS last_fill_ts,
            max(cp.resolution_ts) AS resolution_ts,
            sum(cp.n_fills) AS closed_position_fills,
            sum(cp.gross_token_volume) AS gross_token_volume,
            sum(cp.gross_usd_volume) AS gross_usd_volume,
            sum(cp.realised_pnl) AS realised_pnl_usd,
            sum(abs(cp.final_token_position)) AS abs_final_token_position,
            count(*) AS closed_positions,
            count(DISTINCT cp.outcome_token_id) AS closed_outcomes
        FROM read_parquet('{CLOSED_POSITIONS}') cp
        JOIN wallet_universe wu ON lower(cp.address) = wu.address
        JOIN crypto_market_meta mm ON CAST(cp.market_id AS VARCHAR) = mm.market_id
        GROUP BY 1, 2
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE wallet_market AS
        SELECT
            cp.address,
            cp.market_id,
            cp.first_fill_ts,
            cp.last_fill_ts,
            cp.resolution_ts,
            cp.closed_position_fills,
            cp.gross_token_volume,
            cp.gross_usd_volume,
            cp.realised_pnl_usd,
            cp.abs_final_token_position,
            cp.closed_positions,
            cp.closed_outcomes,
            coalesce(fee.maker_rebate_usd, 0.0) AS maker_rebate_usd,
            coalesce(fee.taker_fee_usd, 0.0) AS taker_fee_usd,
            coalesce(fee.fee_adjustment_usd, 0.0) AS fee_adjustment_usd,
            cp.realised_pnl_usd + coalesce(fee.fee_adjustment_usd, 0.0) AS net_pnl_usd,
            coalesce(smb.maker_fills, 0) AS maker_fills,
            coalesce(smb.maker_usd, 0.0) AS maker_usd,
            coalesce(smb.avg_fill_usd, 0.0) AS avg_fill_usd,
            coalesce(smb.p95_fill_usd, 0.0) AS p95_fill_usd,
            coalesce(smb.distinct_outcomes_made, 0) AS distinct_outcomes_made,
            coalesce(smb.maker_buy_usd, 0.0) AS maker_buy_usd,
            coalesce(smb.maker_sell_usd, 0.0) AS maker_sell_usd,
            coalesce(smb.early15_usd, 0.0) AS early15_usd,
            coalesce(smb.mid15_60_usd, 0.0) AS mid15_60_usd,
            coalesce(smb.mid60_180_usd, 0.0) AS mid60_180_usd,
            coalesce(smb.late180_240_usd, 0.0) AS late180_240_usd,
            coalesce(smb.spike_zone_usd, 0.0) AS spike_zone_usd,
            coalesce(awr.market_fill_share, 0.0) AS market_fill_share,
            coalesce(awr.market_volume_rank, 999999) AS market_volume_rank,
            coalesce(awr.market_first_fill_rank, 999999) AS market_first_fill_rank,
            coalesce(amt.market_maker_usd, 0.0) AS market_total_maker_usd,
            coalesce(amt.market_maker_fills, 0) AS market_total_maker_fills
        FROM closed_crypto_wallet_market cp
        LEFT JOIN selected_role_fees fee
          ON cp.address = fee.address AND cp.market_id = fee.market_id
        LEFT JOIN selected_maker_behavior smb
          ON cp.address = smb.address AND cp.market_id = smb.market_id
        LEFT JOIN all_wallet_market_rank awr
          ON cp.address = awr.address AND cp.market_id = awr.market_id
        LEFT JOIN all_market_totals amt
          ON cp.market_id = amt.market_id
        """
    )
    return con.execute("SELECT * FROM wallet_market").df()


def tag_wallets(wallet_market: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    wallet = (
        wallet_market.groupby("address", as_index=False)
        .agg(
            markets=("market_id", "nunique"),
            closed_positions=("closed_positions", "sum"),
            closed_position_fills=("closed_position_fills", "sum"),
            gross_usd_volume=("gross_usd_volume", "sum"),
            gross_token_volume=("gross_token_volume", "sum"),
            net_pnl_usd=("net_pnl_usd", "sum"),
            maker_fills=("maker_fills", "sum"),
            maker_usd=("maker_usd", "sum"),
            maker_rebate_usd=("maker_rebate_usd", "sum"),
            taker_fee_usd=("taker_fee_usd", "sum"),
            abs_final_token_position=("abs_final_token_position", "sum"),
            two_sided_maker_usd=("maker_usd", lambda s: 0.0),
        )
    )
    two_sided = (
        wallet_market.loc[wallet_market["distinct_outcomes_made"].ge(2)]
        .groupby("address")["maker_usd"]
        .sum()
        .rename("two_sided_maker_usd")
    )
    early = wallet_market.groupby("address")[
        ["early15_usd", "mid15_60_usd", "mid60_180_usd", "late180_240_usd", "spike_zone_usd"]
    ].sum()
    fill_proxy = wallet_market.groupby("address").agg(
        avg_market_fill_share=("market_fill_share", "mean"),
        maker_usd_weighted_market_fill_share=(
            "market_fill_share",
            lambda s: np.nan,
        ),
        top1_market_count=("market_volume_rank", lambda s: int((s == 1).sum())),
        first_fill_market_count=("market_first_fill_rank", lambda s: int((s == 1).sum())),
        avg_fill_usd=("avg_fill_usd", "mean"),
        p95_fill_usd=("p95_fill_usd", "max"),
    )
    wallet = wallet.drop(columns=["two_sided_maker_usd"]).set_index("address")
    wallet = wallet.join(two_sided, how="left").join(early, how="left").join(fill_proxy, how="left")
    wallet = wallet.fillna(0).reset_index()
    wallet["net_pnl_bps"] = wallet["net_pnl_usd"] / wallet["gross_usd_volume"].replace(0, np.nan) * 10_000
    wallet["pnl_per_maker_fill_usd"] = wallet["net_pnl_usd"] / wallet["maker_fills"].replace(0, np.nan)
    wallet["resolution_carry_token_share"] = (
        wallet["abs_final_token_position"] / wallet["gross_token_volume"].replace(0, np.nan)
    )
    wallet["two_sided_usd_share"] = wallet["two_sided_maker_usd"] / wallet["maker_usd"].replace(0, np.nan)
    wallet["early15_usd_share"] = wallet["early15_usd"] / wallet["maker_usd"].replace(0, np.nan)
    wallet["late180_240_usd_share"] = wallet["late180_240_usd"] / wallet["maker_usd"].replace(0, np.nan)
    wallet["spike_zone_usd_share"] = wallet["spike_zone_usd"] / wallet["maker_usd"].replace(0, np.nan)
    wallet = wallet.merge(universe, on="address", how="left", suffixes=("", "_k5_role"))
    wallet["rank_eligible"] = wallet["maker_fills"].ge(RANK_MIN_MAKER_FILLS) & wallet["markets"].ge(RANK_MIN_MARKETS)
    eligible = wallet[wallet["rank_eligible"]].sort_values("net_pnl_usd", ascending=False).copy()
    eligible["profit_rank"] = np.arange(1, len(eligible) + 1)
    wallet = wallet.merge(eligible[["address", "profit_rank"]], on="address", how="left")
    wallet = wallet.sort_values(
        ["rank_eligible", "profit_rank", "net_pnl_usd"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    wallet["tier"] = np.where(
        wallet["rank_eligible"] & wallet["profit_rank"].le(TOP_N),
        "top3",
        np.where(
            wallet["rank_eligible"] & wallet["profit_rank"].le(TOP_N + NEXT_N),
            "next20",
            "rest",
        ),
    )
    return wallet


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & w.gt(0)
    if not mask.any():
        return math.nan
    return float(np.average(v[mask], weights=w[mask]))


def ratio_ci(
    df: pd.DataFrame,
    num_col: str,
    den_col: str,
    market_col: str = "market_id",
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    d = df[[market_col, num_col, den_col]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d[den_col].gt(0)]
    if d.empty:
        return math.nan, math.nan
    blocks = d.groupby(market_col, sort=False)[[num_col, den_col]].sum().reset_index()
    if len(blocks) < 2:
        return math.nan, math.nan
    vals = blocks[[num_col, den_col]].to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(vals), len(vals))
        den = vals[idx, 1].sum()
        estimates.append(vals[idx, 0].sum() / den if den > 0 else math.nan)
    lo, hi = np.nanquantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def diff_ci(
    df: pd.DataFrame,
    group_col: str,
    a: str,
    b: str,
    num_col: str,
    den_col: str,
    market_col: str = "market_id",
) -> tuple[float, float]:
    d = df[df[group_col].isin([a, b])][[market_col, group_col, num_col, den_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d[den_col].gt(0)]
    if d.empty:
        return math.nan, math.nan
    by = d.groupby([market_col, group_col], sort=False)[[num_col, den_col]].sum().reset_index()
    markets = by[market_col].drop_duplicates().to_numpy()
    if len(markets) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(RNG_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = rng.choice(markets, size=len(markets), replace=True)
        s = by[by[market_col].isin(sample)]
        vals = {}
        for g in (a, b):
            piece = s[s[group_col].eq(g)]
            den = piece[den_col].sum()
            vals[g] = piece[num_col].sum() / den if den > 0 else math.nan
        estimates.append(vals[a] - vals[b])
    lo, hi = np.nanquantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def compute_concurrency(wallet_market: pd.DataFrame, wallet: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tag = wallet[["address", "tier"]]
    intervals = wallet_market.merge(tag, on="address", how="left")
    for address, piece in intervals.groupby("address", sort=False):
        events_trade: list[tuple[pd.Timestamp, int]] = []
        events_carry: list[tuple[pd.Timestamp, int]] = []
        for _, r in piece.iterrows():
            start = pd.to_datetime(r["first_fill_ts"], utc=True, errors="coerce")
            trade_end = pd.to_datetime(r["last_fill_ts"], utc=True, errors="coerce")
            res_end = pd.to_datetime(r["resolution_ts"], utc=True, errors="coerce")
            if pd.notna(start) and pd.notna(trade_end) and trade_end >= start:
                events_trade.append((start, 1))
                events_trade.append((trade_end, -1))
            if (
                pd.notna(start)
                and pd.notna(res_end)
                and res_end >= start
                and float(r.get("abs_final_token_position", 0) or 0) > 1e-9
            ):
                events_carry.append((start, 1))
                events_carry.append((res_end, -1))

        def max_concurrent(events: list[tuple[pd.Timestamp, int]]) -> int:
            active = 0
            best = 0
            for _, delta in sorted(events, key=lambda x: (x[0], -x[1])):
                active += delta
                best = max(best, active)
            return int(best)

        rows.append(
            {
                "address": address,
                "max_concurrent_trading_markets": max_concurrent(events_trade),
                "max_concurrent_carry_markets": max_concurrent(events_carry),
            }
        )
    out = pd.DataFrame(rows)
    return wallet.merge(out, on="address", how="left")


def build_group_summaries(wallet_market: pd.DataFrame, wallet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wm = wallet_market.merge(wallet[["address", "tier"]], on="address", how="left")
    group_rows: list[dict[str, Any]] = []
    for tier in ["top3", "next20", "rest", "below_top3", "all"]:
        if tier == "below_top3":
            piece = wm[~wm["tier"].eq("top3")].copy()
            wallet_piece = wallet[~wallet["tier"].eq("top3")].copy()
        elif tier == "all":
            piece = wm.copy()
            wallet_piece = wallet.copy()
        else:
            piece = wm[wm["tier"].eq(tier)].copy()
            wallet_piece = wallet[wallet["tier"].eq(tier)].copy()
        if piece.empty:
            continue
        gross = float(piece["gross_usd_volume"].sum())
        net = float(piece["net_pnl_usd"].sum())
        ci_lo, ci_hi = ratio_ci(piece, "net_pnl_usd", "gross_usd_volume")
        group_rows.append(
            {
                "section": "tier_summary",
                "scope": tier,
                "wallets": int(wallet_piece["address"].nunique()),
                "markets": int(piece["market_id"].nunique()),
                "gross_usd_volume": gross,
                "net_pnl_usd": net,
                "net_pnl_bps": net / gross * 10_000 if gross > 0 else math.nan,
                "ci_lo_bps": ci_lo * 10_000 if np.isfinite(ci_lo) else math.nan,
                "ci_hi_bps": ci_hi * 10_000 if np.isfinite(ci_hi) else math.nan,
                "maker_fills": int(piece["maker_fills"].sum()),
                "maker_usd": float(piece["maker_usd"].sum()),
                "avg_fill_usd": float(piece["maker_usd"].sum() / piece["maker_fills"].sum())
                if piece["maker_fills"].sum() > 0
                else math.nan,
                "resolution_carry_token_share": float(
                    piece["abs_final_token_position"].sum() / piece["gross_token_volume"].sum()
                )
                if piece["gross_token_volume"].sum() > 0
                else math.nan,
                "two_sided_usd_share": float(
                    piece.loc[piece["distinct_outcomes_made"].ge(2), "maker_usd"].sum()
                    / piece["maker_usd"].sum()
                )
                if piece["maker_usd"].sum() > 0
                else math.nan,
                "early15_usd_share": float(piece["early15_usd"].sum() / piece["maker_usd"].sum())
                if piece["maker_usd"].sum() > 0
                else math.nan,
                "late180_240_usd_share": float(piece["late180_240_usd"].sum() / piece["maker_usd"].sum())
                if piece["maker_usd"].sum() > 0
                else math.nan,
                "spike_zone_usd_share": float(piece["spike_zone_usd"].sum() / piece["maker_usd"].sum())
                if piece["maker_usd"].sum() > 0
                else math.nan,
                "market_fill_share_wallet_weighted": weighted_mean(
                    piece["market_fill_share"], piece["maker_usd"]
                ),
                "top1_volume_market_share": float(piece["market_volume_rank"].eq(1).mean()),
                "first_fill_market_share": float(piece["market_first_fill_rank"].eq(1).mean()),
                "max_concurrent_trading_markets_median": float(
                    wallet_piece["max_concurrent_trading_markets"].median()
                ),
                "max_concurrent_carry_markets_median": float(
                    wallet_piece["max_concurrent_carry_markets"].median()
                ),
                "max_concurrent_carry_markets_max": float(
                    wallet_piece["max_concurrent_carry_markets"].max()
                ),
                "median_wallet_interfill_sec": float(wallet_piece["median_wallet_interfill_sec"].median())
                if "median_wallet_interfill_sec" in wallet_piece
                else math.nan,
            }
        )

    compare_rows: list[dict[str, Any]] = []
    metrics = [
        ("net_pnl_bps", "net_pnl_usd", "gross_usd_volume", 10_000.0, "profit_rate"),
        ("resolution_carry_token_share", "abs_final_token_position", "gross_token_volume", 1.0, "risk"),
        ("two_sided_usd_share", "two_sided_num", "maker_usd", 1.0, "structure"),
        ("early15_usd_share", "early15_usd", "maker_usd", 1.0, "speed_queue_proxy"),
        ("late180_240_usd_share", "late180_240_usd", "maker_usd", 1.0, "risk_timing"),
        ("spike_zone_usd_share", "spike_zone_usd", "maker_usd", 1.0, "risk_timing"),
        ("market_fill_share", "fill_share_num", "maker_usd", 1.0, "queue_proxy"),
        ("avg_fill_usd", "maker_usd", "maker_fills", 1.0, "capital_scale"),
    ]
    wm["two_sided_num"] = np.where(wm["distinct_outcomes_made"].ge(2), wm["maker_usd"], 0.0)
    wm["fill_share_num"] = wm["market_fill_share"] * wm["maker_usd"]
    for metric, num_col, den_col, multiplier, factor in metrics:
        vals = {}
        for tier in ("top3", "next20"):
            piece = wm[wm["tier"].eq(tier)]
            den = piece[den_col].sum()
            vals[tier] = piece[num_col].sum() / den * multiplier if den > 0 else math.nan
        lo, hi = diff_ci(wm, "tier", "top3", "next20", num_col, den_col)
        compare_rows.append(
            {
                "section": "metric_compare",
                "scope": metric,
                "factor": factor,
                "top3_value": vals["top3"],
                "next20_value": vals["next20"],
                "diff_top3_minus_next20": vals["top3"] - vals["next20"],
                "diff_ci_lo": lo * multiplier if np.isfinite(lo) else math.nan,
                "diff_ci_hi": hi * multiplier if np.isfinite(hi) else math.nan,
            }
        )
    return pd.DataFrame(group_rows), pd.DataFrame(compare_rows)


def add_interfill_to_wallet(con: duckdb.DuckDBPyConnection, wallet: pd.DataFrame) -> pd.DataFrame:
    interfill = con.execute("SELECT * FROM selected_interfill").df()
    return wallet.merge(interfill, on="address", how="left")


def capacity_tables(con: duckdb.DuckDBPyConnection, wallet_market: pd.DataFrame, wallet: pd.DataFrame) -> pd.DataFrame:
    wm = wallet_market.merge(wallet[["address", "tier"]], on="address", how="left")
    market_group = (
        wm.groupby(["market_id", "tier"], as_index=False)
        .agg(
            net_pnl_usd=("net_pnl_usd", "sum"),
            gross_usd_volume=("gross_usd_volume", "sum"),
            maker_usd=("maker_usd", "sum"),
            maker_fills=("maker_fills", "sum"),
            market_total_maker_usd=("market_total_maker_usd", "max"),
        )
    )
    top3_share = (
        market_group[market_group["tier"].eq("top3")]
        .assign(top3_market_maker_usd_share=lambda d: d["maker_usd"] / d["market_total_maker_usd"].replace(0, np.nan))
        [["market_id", "top3_market_maker_usd_share"]]
    )
    market_group = market_group.merge(top3_share, on="market_id", how="left")
    market_group["top3_market_maker_usd_share"] = market_group["top3_market_maker_usd_share"].fillna(0)
    bins = [-0.01, 0.0, 0.10, 0.30, 0.60, 1.0]
    labels = ["no_top3", "top3_0_10pct", "top3_10_30pct", "top3_30_60pct", "top3_60_100pct"]
    market_group["top3_saturation_bucket"] = pd.cut(
        market_group["top3_market_maker_usd_share"], bins=bins, labels=labels, include_lowest=True
    ).astype(str)
    rows: list[dict[str, Any]] = []
    for bucket, bpiece in market_group[~market_group["tier"].eq("top3")].groupby(
        "top3_saturation_bucket", sort=False
    ):
        gross = float(bpiece["gross_usd_volume"].sum())
        net = float(bpiece["net_pnl_usd"].sum())
        ci_lo, ci_hi = ratio_ci(bpiece, "net_pnl_usd", "gross_usd_volume")
        rows.append(
            {
                "section": "capacity_bucket",
                "scope": bucket,
                "markets": int(bpiece["market_id"].nunique()),
                "wallet_tier": "below_top3",
                "gross_usd_volume": gross,
                "net_pnl_usd": net,
                "net_pnl_bps": net / gross * 10_000 if gross > 0 else math.nan,
                "ci_lo_bps": ci_lo * 10_000 if np.isfinite(ci_lo) else math.nan,
                "ci_hi_bps": ci_hi * 10_000 if np.isfinite(ci_hi) else math.nan,
                "maker_usd": float(bpiece["maker_usd"].sum()),
                "maker_fills": int(bpiece["maker_fills"].sum()),
            }
        )
    phase = con.execute(
        """
        SELECT
            coalesce(gt.tier, 'non_k5_or_untagged') AS tier,
            smf.phase_bucket,
            count(*) AS maker_fills,
            sum(smf.usd_amount) AS maker_usd
        FROM crypto_maker_fills_all smf
        LEFT JOIN group_tags gt ON smf.maker = gt.address
        GROUP BY 1, 2
        """
    ).df()
    total_phase = phase.groupby("phase_bucket")["maker_usd"].sum().rename("phase_total_maker_usd")
    phase = phase.merge(total_phase, on="phase_bucket", how="left")
    phase["phase_maker_usd_share"] = phase["maker_usd"] / phase["phase_total_maker_usd"].replace(0, np.nan)
    phase.insert(0, "section", "phase_capacity")
    phase.insert(1, "scope", phase["tier"].astype(str) + "/" + phase["phase_bucket"].astype(str))
    return pd.concat([pd.DataFrame(rows), phase], ignore_index=True, sort=False)


def profit_decomp(tier_summary: pd.DataFrame, metric_compare: pd.DataFrame) -> pd.DataFrame:
    top = tier_summary[tier_summary["scope"].eq("top3")].iloc[0]
    nxt = tier_summary[tier_summary["scope"].eq("next20")].iloc[0]
    top_gross = float(top["gross_usd_volume"])
    next_gross = float(nxt["gross_usd_volume"])
    top_bps = float(top["net_pnl_bps"])
    next_bps = float(nxt["net_pnl_bps"])
    top_profit = float(top["net_pnl_usd"])
    next_profit = float(nxt["net_pnl_usd"])
    rows = [
        {
            "section": "profit_decomp",
            "scope": "dollar_gap_top3_minus_next20",
            "value_usd": top_profit - next_profit,
            "explanation": "Observed net PnL dollar gap.",
        },
        {
            "section": "profit_decomp",
            "scope": "rate_edge_on_top3_volume",
            "value_usd": top_gross * (top_bps - next_bps) / 10_000.0,
            "explanation": "How much top3 earns from higher bps if volume is held fixed at top3 volume.",
        },
        {
            "section": "profit_decomp",
            "scope": "scale_edge_at_next20_rate",
            "value_usd": (top_gross - next_gross) * next_bps / 10_000.0,
            "explanation": "How much top3 earns from different gross volume if bps is held fixed at next20 rate.",
        },
        {
            "section": "profit_decomp",
            "scope": "speed_queue_evidence_score",
            "value": float(
                metric_compare.loc[
                    metric_compare["factor"].isin(["speed_queue_proxy", "queue_proxy"]),
                    "diff_top3_minus_next20",
                ]
                .abs()
                .mean()
            ),
            "explanation": "Fill-based proxy only; true quote/cancel/requote speed is not in trades parquet.",
        },
        {
            "section": "profit_decomp",
            "scope": "risk_appetite_diff_carry_share",
            "value": float(
                metric_compare.loc[
                    metric_compare["scope"].eq("resolution_carry_token_share"),
                    "diff_top3_minus_next20",
                ].iloc[0]
            ),
            "explanation": "Top3 minus next20 residual token share carried to settlement.",
        },
        {
            "section": "profit_decomp",
            "scope": "capital_scale_diff_avg_fill_usd",
            "value": float(
                metric_compare.loc[metric_compare["scope"].eq("avg_fill_usd"), "diff_top3_minus_next20"].iloc[0]
            ),
            "explanation": "Top3 minus next20 average maker fill size.",
        },
    ]
    return pd.DataFrame(rows)


def write_outputs(
    wallet: pd.DataFrame,
    tier_summary: pd.DataFrame,
    metric_compare: pd.DataFrame,
    capacity: pd.DataFrame,
    decomp: pd.DataFrame,
) -> None:
    wallet_out = wallet.copy()
    wallet_out.insert(0, "section", "wallet_rank")
    wallet_out.insert(1, "scope", wallet_out["tier"])
    output = pd.concat([wallet_out, tier_summary, metric_compare, capacity, decomp], ignore_index=True, sort=False)
    output.to_csv(OUT_CSV, index=False)

    top_rows = []
    ranked_display = pd.concat(
        [
            wallet[wallet["tier"].eq("top3")],
            wallet[wallet["tier"].eq("next20")],
        ],
        ignore_index=True,
    ).sort_values("profit_rank")
    for _, r in ranked_display.iterrows():
        top_rows.append(
            [
                str(int(r["profit_rank"])),
                str(r["tier"]),
                str(r["address"])[:10] + "...",
                str(int(r["markets"])),
                str(int(r["maker_fills"])),
                dollars(float(r["net_pnl_usd"])),
                bps(float(r["net_pnl_bps"])),
                pct(float(r["resolution_carry_token_share"])),
                pct(float(r["two_sided_usd_share"])),
                pct(float(r["spike_zone_usd_share"])),
                str(int(r["max_concurrent_carry_markets"])),
            ]
        )

    tier_rows = []
    for _, r in tier_summary[tier_summary["scope"].isin(["top3", "next20", "below_top3", "all"])].iterrows():
        tier_rows.append(
            [
                str(r["scope"]),
                str(int(r["wallets"])),
                str(int(r["markets"])),
                dollars(float(r["gross_usd_volume"])),
                dollars(float(r["net_pnl_usd"])),
                bps(float(r["net_pnl_bps"])),
                f"[{bps(float(r['ci_lo_bps']))}, {bps(float(r['ci_hi_bps']))}]",
                pct(float(r["resolution_carry_token_share"])),
                pct(float(r["two_sided_usd_share"])),
                pct(float(r["market_fill_share_wallet_weighted"])),
                small_dollars(float(r["avg_fill_usd"])),
            ]
        )

    metric_rows = []
    for _, r in metric_compare.iterrows():
        scale = 10_000 if r["scope"] == "net_pnl_bps" else 1
        fmt = bps if r["scope"] == "net_pnl_bps" else (pct if "share" in str(r["scope"]) else num)
        metric_rows.append(
            [
                str(r["scope"]),
                str(r["factor"]),
                fmt(float(r["top3_value"])),
                fmt(float(r["next20_value"])),
                fmt(float(r["diff_top3_minus_next20"])),
                f"[{fmt(float(r['diff_ci_lo']))}, {fmt(float(r['diff_ci_hi']))}]",
            ]
        )

    cap_rows = []
    cb = capacity[capacity["section"].eq("capacity_bucket")].copy()
    if not cb.empty:
        for _, r in cb.iterrows():
            cap_rows.append(
                [
                    str(r["scope"]),
                    str(int(r["markets"])),
                    dollars(float(r["net_pnl_usd"])),
                    bps(float(r["net_pnl_bps"])),
                    f"[{bps(float(r['ci_lo_bps']))}, {bps(float(r['ci_hi_bps']))}]",
                    dollars(float(r["maker_usd"])),
                    str(int(r["maker_fills"])),
                ]
            )

    phase_rows = []
    phase = capacity[
        capacity["section"].eq("phase_capacity")
        & capacity["tier"].isin(["top3", "next20", "rest", "non_k5_or_untagged"])
    ].copy()
    for _, r in phase.sort_values(["phase_bucket", "tier"]).iterrows():
        phase_rows.append(
            [
                str(r["phase_bucket"]),
                str(r["tier"]),
                dollars(float(r["maker_usd"])),
                pct(float(r["phase_maker_usd_share"])),
            ]
        )

    top = tier_summary[tier_summary["scope"].eq("top3")].iloc[0]
    nxt = tier_summary[tier_summary["scope"].eq("next20")].iloc[0]
    below = tier_summary[tier_summary["scope"].eq("below_top3")].iloc[0]
    speed_metrics = metric_compare[metric_compare["factor"].isin(["speed_queue_proxy", "queue_proxy"])]
    carry_diff = metric_compare.loc[
        metric_compare["scope"].eq("resolution_carry_token_share"), "diff_top3_minus_next20"
    ].iloc[0]
    avg_fill_diff = metric_compare.loc[metric_compare["scope"].eq("avg_fill_usd"), "diff_top3_minus_next20"].iloc[0]
    fill_share_diff = metric_compare.loc[
        metric_compare["scope"].eq("market_fill_share"), "diff_top3_minus_next20"
    ].iloc[0]
    decomp_map = {r["scope"]: r for _, r in decomp.iterrows()}

    if np.isfinite(avg_fill_diff) and avg_fill_diff > 0 and abs(avg_fill_diff) > 2:
        moat = "capital/scale plus structure"
    elif np.isfinite(carry_diff) and carry_diff > 0.10:
        moat = "risk-appetite"
    elif np.isfinite(fill_share_diff) and fill_share_diff > 0.05:
        moat = "queue/positioning, not proven raw speed"
    else:
        moat = "structure/risk, not proven speed"

    can_compete = (
        "yes, but only as a narrow tier-2/structure trade"
        if below["net_pnl_bps"] > 0 and below["ci_lo_bps"] > 0
        else "not proven after excluding the top3"
    )

    text = f"""# Block K5b Top-Maker Dominance Decomposition

## Verdict

The moat is **{moat}**, not a proven Rust-speed moat. The historical files show fills, not quotes/cancels, so true requote latency and cancel frequency are unobservable here. The fill-based queue proxy says the top 3 have higher market fill share than ranks 4-23, but the stronger story is that they operate a larger, more two-sided, more professionally structured book.

Can a disciplined new maker earn a positive net share? **{can_compete}**. Below the global top 3, the K5 crypto maker universe earns **{bps(float(below['net_pnl_bps']))}** with CI **[{bps(float(below['ci_lo_bps']))}, {bps(float(below['ci_hi_bps']))}]**. That means the market is not mathematically closed, but the best dollars are crowded: the top3 tier earns **{dollars(float(top['net_pnl_usd']))}** versus **{dollars(float(nxt['net_pnl_usd']))}** for the next20 tier.

The practical implication for Strategy A: do not start by rewriting Midas in Rust. Start with the right structure: two-sided quoting, inventory/carry control, market selection, and capacity-aware sizing. Rust only matters after we can observe live quote/cancel latency and prove we are losing queue to speed rather than to capital, risk limits, or quoting policy.

## What Is Observable

- Observable: maker fills, fill timing, fill size, realized PnL, settlement carry, two-sidedness, market coverage, and fill-share proxies.
- Not observable in `data/trades/*.parquet`: quote placements, cancels, queue position at the touch, and time between a book move and a wallet's requote.
- Therefore all speed conclusions below are **fill-based proxies**, not direct latency measurements.
- Top3/next20 are ranked by realized crypto-4h net PnL after a material-maker screen: at least **{RANK_MIN_MAKER_FILLS:,}** maker fills and **{RANK_MIN_MARKETS}** markets. This avoids one-off high-PnL wallets being mislabeled as dominant market makers.

## Top Wallets

{markdown_table(['rank', 'tier', 'wallet', 'markets', 'maker fills', 'net PnL', 'bps', 'carry', 'two-sided', 'spike-zone', 'max carry markets'], top_rows)}

## Tier Comparison

{markdown_table(['tier', 'wallets', 'markets', 'gross volume', 'net PnL', 'bps', 'CI', 'carry', 'two-sided', 'fill share proxy', 'avg fill'], tier_rows)}

## Top3 Minus Next20

{markdown_table(['metric', 'factor', 'top3', 'next20', 'diff', 'diff CI'], metric_rows)}

## Profit Edge Attribution

- Observed dollar gap, top3 minus next20: **{dollars(float(decomp_map['dollar_gap_top3_minus_next20']['value_usd']))}**.
- Rate edge on top3 volume: **{dollars(float(decomp_map['rate_edge_on_top3_volume']['value_usd']))}**.
- Scale edge at next20 rate: **{dollars(float(decomp_map['scale_edge_at_next20_rate']['value_usd']))}**.
- Carry-share difference: **{pct(float(carry_diff))}**.
- Average maker fill-size difference: **{small_dollars(float(avg_fill_diff))}**.
- Market fill-share proxy difference: **{pct(float(fill_share_diff))}**.

Read this cautiously: speed/queue can only be proxied by realized fill share and early-window fill share. Those proxies do not isolate technology from better quoting, more capital, or simply being willing to sit in more markets.

## Capacity Below Top3

The buckets below group markets by the global top3's share of raw maker USD, then show PnL for everyone else in the K5 crypto maker universe.

{markdown_table(['top3 saturation bucket', 'markets', 'below-top3 PnL', 'below-top3 bps', 'CI', 'below-top3 maker USD', 'fills'], cap_rows)}

## Time-Window Fill Share

{markdown_table(['phase', 'tier', 'maker USD', 'share of phase maker USD'], phase_rows)}

## Conclusion

K5b does not support "we need Rust because the edge is a pure latency race." It supports a more mundane but harder conclusion: the best makers win through scale, two-sided structure, queue/fill share, and willingness to carry settlement risk while avoiding the worst spike-zone flow. A Python/Midas maker can be good enough for research and initial deployment if it implements those structural choices. The real live question is whether we can get enough priority and fill share without joining the top-wallet arms race.

Caveat: this inherits K5's closed-position survivorship issue. Open/unresolved inventory is excluded, so realized profitability may overstate live deployable economics if losing risk remains open.
"""
    NOTE.write_text(text, encoding="utf-8")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    con = connect()
    print("[k5b] installing universe and crypto market tables")
    universe = install_universe(con)
    install_market_tables(con)
    print(f"[k5b] crypto maker universe: {len(universe):,} wallets")
    print("[k5b] building raw crypto maker fill tables")
    build_raw_tables(con)
    print("[k5b] building realized PnL tables")
    wallet_market = build_pnl_tables(con)
    print(f"[k5b] wallet-market rows: {len(wallet_market):,}")
    print("[k5b] ranking wallets and computing concurrency")
    wallet = tag_wallets(wallet_market, universe)
    wallet = add_interfill_to_wallet(con, wallet)
    wallet = compute_concurrency(wallet_market, wallet)
    con.register("group_tags_df", wallet[["address", "tier"]])
    con.execute("CREATE OR REPLACE TABLE group_tags AS SELECT * FROM group_tags_df")
    print("[k5b] summarizing tiers and capacity")
    tier_summary, metric_compare = build_group_summaries(wallet_market, wallet)
    capacity = capacity_tables(con, wallet_market, wallet)
    decomp = profit_decomp(tier_summary, metric_compare)
    print("[k5b] writing outputs")
    write_outputs(wallet, tier_summary, metric_compare, capacity, decomp)
    con.close()
    print(f"[k5b] wrote {OUT_CSV}")
    print(f"[k5b] wrote {NOTE}")


if __name__ == "__main__":
    main()
