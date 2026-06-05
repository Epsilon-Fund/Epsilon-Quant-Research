"""Weather-market tail-risk characterisation.

For each weather market and each outcome token, in the W-hour pre-resolution window:
  - did the price cross barrier p (for p in BARRIERS)?
  - if it did, did the token resolve to 0 (crash)?
  - if it crashed, how long from first cross to first fill below 5c (or resolution_ts)?

Outputs:
  data/analysis/weather_tail_analysis.parquet         (24h primary, per (slug_family, p))
  data/analysis/weather_tail_analysis_48h.parquet     (48h sensitivity, same schema)
  data/analysis/weather_tail_per_instance.parquet     (per (market, token, p) long-form; 24h)
  data/analysis/weather_universe.parquet              (market-level catalogue)

CLI:
  python weather_tail_analysis.py            # runs 24h then 48h
  python weather_tail_analysis.py --skip-48h
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
RESEARCH_ROOT = THIS.parents[1]            # polymarket/research

DATA_OUT = RESEARCH_ROOT / "data" / "analysis"
DATA_OUT.mkdir(parents=True, exist_ok=True)

MARKETS_PARQUET = RESEARCH_ROOT / "data" / "markets" / "markets_2026-05-06.parquet"
TRADES_GLOB     = str(RESEARCH_ROOT / "data" / "trades" / "*.parquet")
CLOSED_POS      = RESEARCH_ROOT / "data" / "closed_positions.parquet"

BARRIERS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

# Universe filter: ONLY highest/lowest temperature markets for a specific city.
# Anchored on the slug start so we exclude:
#  - global-temperature-increase-* (different dynamics, monthly resolution)
#  - "tyra-hurricane-black" pickleball markets and other accidental matches
#  - rain/snow/hurricane/tornado/earthquake markets (sparse, different dynamics)
WEATHER_SLUG_RX = r'^(will-the-)?(highest|lowest)-temperature-in-'

# Last 12 months — today is 2026-05-14, so we keep markets resolving from 2025-05-14 on.
UNIVERSE_END_DATE_FLOOR = '2025-05-14'
UNIVERSE_END_DATE_CEIL  = '2026-05-14'

MONTHS = r'(january|february|march|april|may|june|july|august|september|october|november|december)'


def slug_family(slug: str) -> str:
    """Strip variable parts (date, temperature band) to reveal the recurring family name."""
    if not slug:
        return "_empty_"
    s = slug.lower()
    s = re.sub(r'^will-(the-)?', '', s)
    s = re.sub(r'^be-(the-)?', '', s)
    # temperature points and ranges (with celsius/fahrenheit + the 'pt' decimal convention)
    s = re.sub(r'-?\d+pt\d+c\b', '', s)
    s = re.sub(r'-between-\d+(pt\d+)?(-(and-)?\d+(pt\d+)?)?(f|c)?\b', '', s)
    s = re.sub(r'-(more-than|less-than|over|under|at-or-above|or-higher|or-lower)-?\d+(pt\d+)?(f|c)?', '', s)
    s = re.sub(r'-\d+-\d+(f|c)\b', '', s)
    s = re.sub(r'-\d+(f|c)\b', '', s)
    # iso dates and year suffixes
    s = re.sub(r'-\d{4}-\d{2}-\d{2}\b', '', s)
    s = re.sub(r'-\d{4}\b', '', s)
    # month-day pairs
    s = re.sub(rf'-on-?{MONTHS}(-\d+)?', '', s)
    s = re.sub(rf'-in-?{MONTHS}(-\d+)?', '', s)
    s = re.sub(rf'-{MONTHS}(-\d+)?', '', s)
    s = re.sub(r'-(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b', '', s)
    # trailing connectors
    s = re.sub(r'-(on|in|by|between|and|or|will|be)$', '', s)
    s = re.sub(r'-+', '-', s)
    s = s.strip('-')
    return s or "_empty_"


def build_universe(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT
            CAST(id AS VARCHAR)         AS market_id,
            slug,
            question,
            end_date::TIMESTAMP         AS end_ts,
            neg_risk,
            outcomes,
            outcome_prices,
            clob_token_ids,
            volume,
            closed
        FROM read_parquet('{MARKETS_PARQUET}')
        WHERE regexp_matches(slug, '{WEATHER_SLUG_RX}')
          AND len(clob_token_ids) >= 2
          AND closed
          AND end_date IS NOT NULL
          AND end_date >= '{UNIVERSE_END_DATE_FLOOR}'
          AND end_date <= '{UNIVERSE_END_DATE_CEIL}'
    """).df()
    df["slug_family"] = df["slug"].map(slug_family)
    return df


def compute_crossings(con: duckdb.DuckDBPyConnection, window_hours: int) -> pd.DataFrame:
    """Per (market_id, outcome_token_id): first-cross timestamps for each barrier + first-below-5c.

    Also captures the maker/taker/maker_side of the fill that constitutes the
    first-cross at each barrier — used downstream by the next-fill slippage
    model to enforce 'different trader' on the ASOF match.
    """
    barrier_cols = []
    for p in BARRIERS:
        pct = int(p * 100)
        # arg_min(arg, key) returns arg at the row with smallest key; NULL keys ignored,
        # so CASE WHEN price >= p THEN ts END restricts the population correctly.
        barrier_cols.append(
            f"MIN(CASE WHEN price >= {p} THEN ts END) AS first_cross_{pct:03d}"
        )
        barrier_cols.append(
            f"arg_min(maker,      CASE WHEN price >= {p} THEN ts END) AS first_cross_maker_{pct:03d}"
        )
        barrier_cols.append(
            f"arg_min(taker,      CASE WHEN price >= {p} THEN ts END) AS first_cross_taker_{pct:03d}"
        )
        barrier_cols.append(
            f"arg_min(maker_side, CASE WHEN price >= {p} THEN ts END) AS first_cross_maker_side_{pct:03d}"
        )
    barrier_exprs = ",\n        ".join(barrier_cols)
    q = f"""
    WITH wm AS (
        SELECT market_id, end_ts FROM weather_markets
    ),
    fills AS (
        SELECT
            t.market_id,
            CASE WHEN t.maker_asset_id = '0' OR t.maker_asset_id IS NULL
                 THEN t.taker_asset_id ELSE t.maker_asset_id END AS outcome_token_id,
            t.timestamp AS ts,
            t.price,
            t.maker,
            t.taker,
            t.maker_side
        FROM read_parquet('{TRADES_GLOB}') t
        JOIN wm w ON t.market_id = w.market_id
        WHERE t.timestamp >= w.end_ts - INTERVAL {window_hours} HOUR
          AND t.timestamp <  w.end_ts
    )
    SELECT
        market_id, outcome_token_id,
        COUNT(*)            AS n_fills_in_window,
        MIN(ts)             AS first_fill_ts,
        MAX(ts)             AS last_fill_ts,
        MIN(price)          AS min_price,
        MAX(price)          AS max_price,
        {barrier_exprs},
        MIN(CASE WHEN price < 0.05 THEN ts END) AS first_below_5c
    FROM fills
    GROUP BY market_id, outcome_token_id
    """
    return con.execute(q).df()


def attach_resolution(con: duckdb.DuckDBPyConnection, cross_df: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """For each (market, outcome_token_id), attach resolution (0/1) and resolution_ts."""
    # Resolution from market.outcome_prices[ list_position(clob_token_ids, outcome_token_id) ]
    con.register("u", universe[["market_id", "clob_token_ids", "outcome_prices"]])
    con.register("c", cross_df)
    res_df = con.execute("""
        SELECT
            c.market_id, c.outcome_token_id,
            list_position(u.clob_token_ids, c.outcome_token_id) AS token_idx,
            CASE
                WHEN list_position(u.clob_token_ids, c.outcome_token_id) > 0
                THEN CAST(u.outcome_prices[list_position(u.clob_token_ids, c.outcome_token_id)] AS DOUBLE)
                ELSE NULL
            END AS resolution
        FROM c JOIN u USING (market_id)
    """).df()
    con.unregister("u"); con.unregister("c")

    # resolution_ts per (market, outcome_token_id) from closed_positions (any address yields same value).
    res_ts = con.execute(f"""
        SELECT
            CAST(market_id AS VARCHAR) AS market_id,
            outcome_token_id,
            MAX(resolution_ts) AS resolution_ts
        FROM read_parquet('{CLOSED_POS}')
        GROUP BY 1,2
    """).df()
    return cross_df.merge(res_df, on=["market_id","outcome_token_id"], how="left") \
                   .merge(res_ts, on=["market_id","outcome_token_id"], how="left")


def build_per_instance(cross_with_res: pd.DataFrame, universe: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    """Long-form: one row per (market_id, outcome_token_id, barrier) with cross/crash flags + time-to-5c."""
    df = cross_with_res.merge(
        universe[["market_id", "slug", "slug_family", "end_ts"]],
        on="market_id", how="left",
    )
    out_rows = []
    for p in BARRIERS:
        pct = int(p * 100)
        ts_col    = f"first_cross_{pct:03d}"
        maker_col = f"first_cross_maker_{pct:03d}"
        taker_col = f"first_cross_taker_{pct:03d}"
        side_col  = f"first_cross_maker_side_{pct:03d}"
        sub = df[["market_id", "outcome_token_id", "slug", "slug_family", "end_ts",
                  "resolution", "resolution_ts", "first_below_5c", "n_fills_in_window",
                  "min_price", "max_price",
                  ts_col, maker_col, taker_col, side_col]].copy()
        sub.rename(columns={
            ts_col:    "first_cross_ts",
            maker_col: "first_cross_maker",
            taker_col: "first_cross_taker",
            side_col:  "first_cross_maker_side",
        }, inplace=True)
        sub["barrier_price"] = p
        # A "crossing" is a GENUINE level-break: token transitioned through p in the window
        # (min_price < p AND first_cross_ts is not null). Tokens that entered the window
        # already above p with no sub-p fills are NOT counted — they're not level-break events.
        sub["crossed"] = sub["first_cross_ts"].notna() & (sub["min_price"] < p)
        sub["resolved_to_zero"]    = (sub["resolution"] == 0).fillna(False)
        sub["crossed_and_crashed"] = sub["crossed"] & sub["resolved_to_zero"]
        out_rows.append(sub)
    per_inst = pd.concat(out_rows, ignore_index=True)

    # time-to-5c for the crashed subset
    crashed = per_inst["crossed_and_crashed"]
    # fill_below_5c only valid if it occurs at/after first_cross_ts
    fb5 = per_inst["first_below_5c"]
    fc  = per_inst["first_cross_ts"]
    valid_fill = crashed & fb5.notna() & (fb5 >= fc)
    per_inst["t5c_source"] = np.where(valid_fill, "fill_below_5c", "")
    # fallback: resolution_ts if available, else end_ts
    fb_ts = per_inst["resolution_ts"].fillna(per_inst["end_ts"])
    use_fallback = crashed & ~valid_fill
    per_inst.loc[use_fallback, "t5c_source"] = np.where(
        per_inst.loc[use_fallback, "resolution_ts"].notna(),
        "resolution_fallback", "end_ts_fallback",
    )
    chosen_ts = pd.to_datetime(np.where(valid_fill, fb5, fb_ts))
    delta = chosen_ts - pd.to_datetime(fc)
    per_inst["time_to_5c_hours"] = pd.Series(delta).dt.total_seconds().values / 3600.0
    per_inst.loc[~crashed, "time_to_5c_hours"] = np.nan
    per_inst.loc[~crashed, "t5c_source"] = ""
    per_inst["window_hours"] = window_hours
    return per_inst


def aggregate(per_inst: pd.DataFrame, universe: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    """Per (slug_family, barrier) + pooled-across-all-weather aggregations."""
    # n_total_instances per (slug_family, barrier) = distinct (market, outcome_token_id) in the weather universe
    # outcome_token_id list comes from universe.clob_token_ids (one row per token per market).
    inst_universe_rows = []
    for _, r in universe.iterrows():
        toks = r["clob_token_ids"]
        if toks is None or (hasattr(toks, "__len__") and len(toks) == 0):
            continue
        for tok in toks:
            inst_universe_rows.append({"market_id": r["market_id"], "outcome_token_id": str(tok),
                                       "slug_family": r["slug_family"]})
    inst_universe = pd.DataFrame(inst_universe_rows)
    per_family_total = inst_universe.groupby("slug_family").size().rename("n_total_instances")

    def _t5c_stats(sub_chopped: pd.DataFrame) -> dict:
        s = sub_chopped["time_to_5c_hours"]
        srcs = sub_chopped["t5c_source"]
        return {
            "time_to_5c_median_hr": s.median() if len(s) else np.nan,
            "time_to_5c_p10_hr":    s.quantile(0.10) if len(s) else np.nan,
            "time_to_5c_p25_hr":    s.quantile(0.25) if len(s) else np.nan,
            "time_to_5c_p75_hr":    s.quantile(0.75) if len(s) else np.nan,
            "time_to_5c_p90_hr":    s.quantile(0.90) if len(s) else np.nan,
            "pct_using_fallback":   (srcs != "fill_below_5c").mean() if len(srcs) else np.nan,
        }

    rows = []
    for p in BARRIERS:
        sub = per_inst[per_inst["barrier_price"] == p]

        per_family = (
            sub.groupby("slug_family")
               .agg(n_crossed=("crossed", "sum"),
                    n_crossed_and_crashed=("crossed_and_crashed", "sum"))
               .reset_index()
        )
        per_family["barrier_price"] = p
        per_family = per_family.merge(per_family_total.reset_index(), on="slug_family", how="right").fillna(0)

        t5 = (
            sub[sub["crossed_and_crashed"]]
               .groupby("slug_family")
               .apply(lambda g: pd.Series(_t5c_stats(g)), include_groups=False)
               .reset_index()
        )
        rows.append(per_family.merge(t5, on="slug_family", how="left"))

        chopped = sub[sub["crossed_and_crashed"]]
        pooled_row = {
            "slug_family": "POOLED_ALL_WEATHER",
            "barrier_price": p,
            "n_total_instances": int(per_family_total.sum()),
            "n_crossed": int(sub["crossed"].sum()),
            "n_crossed_and_crashed": int(sub["crossed_and_crashed"].sum()),
            **_t5c_stats(chopped),
        }
        rows.append(pd.DataFrame([pooled_row]))

    agg = pd.concat(rows, ignore_index=True)
    agg["chop_rate"]       = np.where(agg["n_crossed"] > 0,
                                      agg["n_crossed_and_crashed"] / agg["n_crossed"], np.nan)
    agg["edge_per_signal"] = (1 - agg["chop_rate"]) - agg["barrier_price"]
    agg["window_hours"]    = window_hours

    cols = ["slug_family", "barrier_price", "window_hours",
            "n_total_instances",
            "n_crossed", "n_crossed_and_crashed",
            "chop_rate", "edge_per_signal",
            "time_to_5c_median_hr", "time_to_5c_p10_hr", "time_to_5c_p25_hr",
            "time_to_5c_p75_hr", "time_to_5c_p90_hr", "pct_using_fallback"]
    return agg[cols]


def sanity_checks(universe: pd.DataFrame, per_inst: pd.DataFrame, agg: pd.DataFrame, window_hours: int) -> None:
    print(f"\n========== SANITY CHECKS (window={window_hours}h) ==========")
    # neg_risk
    nr = universe["neg_risk"].fillna(False).astype(bool)
    print(f"  weather universe: {len(universe)} markets; neg_risk=TRUE share = {nr.mean():.3f}")

    # crash rate at 0.5 vs 0.9 (pooled)
    pooled = agg[agg["slug_family"] == "POOLED_ALL_WEATHER"].set_index("barrier_price")
    cr_50 = pooled.loc[0.50, "chop_rate"]
    cr_70 = pooled.loc[0.70, "chop_rate"]
    cr_80 = pooled.loc[0.80, "chop_rate"]
    cr_90 = pooled.loc[0.90, "chop_rate"]
    print(f"  pooled chop  @0.50 = {cr_50:.3f}; @0.70 = {cr_70:.3f}; @0.80 = {cr_80:.3f}; @0.90 = {cr_90:.3f}")
    print(f"  pooled edge  @0.70 = ${(1-cr_70)-0.70:+.4f}; @0.80 = ${(1-cr_80)-0.80:+.4f} / $1 of payout")

    # total fills sanity
    n_fills = per_inst["n_fills_in_window"].fillna(0).sum() / len(BARRIERS)  # repeated per barrier
    n_markets_w_data = per_inst[(per_inst["n_fills_in_window"]>0)]["market_id"].nunique()
    print(f"  fills processed: {int(n_fills):,} across {n_markets_w_data:,} markets")
    if len(universe):
        print(f"  avg fills/market w/ trading activity = {n_fills/max(n_markets_w_data,1):.1f}")

    # 5 random markets at p=0.85
    sub = per_inst[(per_inst["barrier_price"] == 0.85) & per_inst["crossed"]].copy()
    if len(sub):
        sample = sub.sample(min(5, len(sub)), random_state=42)
        print(f"\n  Sample 5 crossed-at-0.85 markets:")
        for _, r in sample.iterrows():
            tag = "CRASH" if r["crossed_and_crashed"] else "ok"
            print(f"    [{tag:5}] {r['slug'][:70]}  cross@{r['first_cross_ts']} res={r['resolution']}")


def run_window(con: duckdb.DuckDBPyConnection, universe: pd.DataFrame, window_hours: int, write_per_instance: bool) -> pd.DataFrame:
    print(f"\n===== Computing crossings (window={window_hours}h) =====")
    con.register("weather_markets", universe[["market_id","end_ts"]])
    try:
        cross = compute_crossings(con, window_hours)
    finally:
        con.unregister("weather_markets")
    print(f"  crossings rows (per market×token): {len(cross):,}")

    cross_with_res = attach_resolution(con, cross, universe)
    per_inst = build_per_instance(cross_with_res, universe, window_hours)
    agg = aggregate(per_inst, universe, window_hours)

    sanity_checks(universe, per_inst, agg, window_hours)

    suffix = "" if window_hours == 24 else f"_{window_hours}h"
    out_agg = DATA_OUT / f"weather_tail_analysis{suffix}.parquet"
    agg.to_parquet(out_agg, index=False)
    print(f"\n  wrote {out_agg} ({len(agg)} rows)")
    if write_per_instance:
        out_inst = DATA_OUT / "weather_tail_per_instance.parquet"
        per_inst.to_parquet(out_inst, index=False)
        print(f"  wrote {out_inst} ({len(per_inst):,} rows)")
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-48h", action="store_true", help="Only run the 24h primary analysis")
    args = ap.parse_args()

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    universe = build_universe(con)
    print(f"weather universe: {len(universe)} markets")
    print(f"  neg_risk=TRUE: {int(universe['neg_risk'].fillna(False).sum())}  "
          f"NA/False: {int((~universe['neg_risk'].fillna(False)).sum())}")
    print(f"  date range: {universe['end_ts'].min()} → {universe['end_ts'].max()}")
    fam_counts = universe.groupby("slug_family").size().sort_values(ascending=False)
    print(f"  slug_family distinct: {len(fam_counts)}")
    print(f"\n  Top 20 families:")
    for fam, n in fam_counts.head(20).items():
        print(f"    {n:6d}  {fam}")
    print(f"\n  Family-count distribution: "
          f"mean={fam_counts.mean():.1f}, median={fam_counts.median():.0f}, "
          f"max={fam_counts.max()}, n_families_w_1={int((fam_counts==1).sum())}")

    out_universe = DATA_OUT / "weather_universe.parquet"
    universe[["market_id", "slug", "slug_family", "end_ts", "neg_risk", "volume"]].to_parquet(out_universe, index=False)
    print(f"\nwrote {out_universe}")

    run_window(con, universe, window_hours=24, write_per_instance=True)
    if not args.skip_48h:
        run_window(con, universe, window_hours=48, write_per_instance=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
