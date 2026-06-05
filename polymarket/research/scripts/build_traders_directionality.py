"""Build data/directionality_classification/traders_directionality.parquet.

Per-trader fill-side metrics that discriminate genuine cross-outcome arb
from two-sided directional betting. Replaces phantom_position_score as the
arb-vs-directional filter (phantom retained as a column for backward
compatibility but no longer the discriminator).

WHY a new metric:
  phantom_position_score = sum(|pos_pnl|) / |sum(pos_pnl)| per market,
  vol-weighted. It blows up for:
    (a) genuine NegRisk merge/split arb     ← we want this
    (b) two-sided directional traders who swap sides mid-market
        without closing first leg          ← phantom flags this as arb
    (c) merge/split provenance artifacts   ← rare in our pipeline (~5%)
  The Domah diagnostic (notes/api_reconciliation_v1.md) showed 8.45 phantom
  but ~0% balanced/offsetting markets — phantom conflates (a) with (b).

WHAT IS COMPUTED (per trader, lifetime, on resolved positions only):

  fill_concentration_p50    : vol-weighted median of per-market
                              max_outcome_notional / total_market_notional.
                              1.0 = all volume on one outcome (directional).
                              1/N = perfectly balanced across N outcomes (arb).
  fill_concentration_p10    : vol-weighted 10th percentile of same.
                              Captures whether trader has ANY markets where
                              they meaningfully split between outcomes.
  net_to_gross_exposure     : vol-weighted median of per-market
                              |Σ signed final_token_position| / Σ |final_token_position|.
                              0 = balanced offsetting; 1 = pure long one side.
  pct_markets_balanced_and_offsetting : share of markets where
                              max_outcome_fill_share < FC_BALANCED AND
                              net_to_gross < NTG_OFFSETTING. Genuine-arb fraction.
                              Both count-share and volume-weighted share are emitted;
                              primary_style is computed off the volume-weighted share
                              (matches how phantom_position_score itself is weighted).
  pct_markets_two_sided_directional   : share of markets where
                              max_outcome_fill_share > FC_SKEWED AND
                              net_to_gross > NTG_DIRECTIONAL AND
                              n_outcomes_traded >= 2. The Domah-style
                              hedge-or-reposition-without-closing signature.
  primary_style             : categorical — pure_directional / two_sided_directional /
                              arb_like / mixed / insufficient_data.
                              Rule: arb_like wins first (rarest, most operationally
                              distinct); then two_sided; then pure_directional gated
                              on a high fc_p50 floor; else mixed.

THRESHOLDS (these placeholders survived the population diagnostic — see
data/directionality_classification/metric_distributions.md for justification):

  FC_BALANCED   = 0.60  (fills not concentrated on one outcome)
  NTG_OFFSETTING = 0.20 (positions roughly offset across outcomes)
  FC_SKEWED     = 0.80  (>=80% of fills on one outcome — clearly directional)
  NTG_DIRECTIONAL = 0.60 (net exposure dominates gross — clearly directional)
  ARB_LIKE_THRESH   = 0.30  (>=30% vol-share of arb-shape markets ⇒ arb_like).
                            Calibrated against the population distribution
                            (see data/directionality_classification/
                            metric_distributions.md): at 0.30, 6.6% of qualifying
                            traders are arb_like — a defensible "arbitrageur" rate
                            for prediction markets. At the earlier placeholder
                            of 0.10, 19% qualified, sweeping in many casual
                            cross-side traders.
  TWO_SIDED_THRESH  = 0.20  (>=20% vol-share of two-sided markets — ~41% of pop).
                            Two-sided behaviour is common (anyone who switches
                            sides once produces some); we want the rule to flag
                            traders for whom it's a substantive share of their book.
  PURE_DIR_FC_FLOOR = 0.85  (vol-wtd median market >=85% one-sided ⇒ pure directional).
                            Below the population p50 of 0.97 so it's permissive
                            enough to catch any genuinely one-sided bettor.

INPUT FILTERS:
  - n_closed_positions > 50   (cost control; metric noisy below this)
  - Operators excluded via data_infra.operator_denylist

OUTPUT: data/directionality_classification/traders_directionality.parquet
Keyed on `address`.

IMPLEMENTATION NOTE — why DuckDB-only aggregation:
  closed_positions.parquet is 29 GB / 270 M rows / 2.6 M addresses. Materialising
  the per-(address, market) frame in pandas OOMs. We do the entire reduction
  in SQL, with weighted quantiles via the cumulative-weight idiom, and only
  bring per-trader rows into Python at the end.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from data_infra.operator_denylist import OPERATOR_ADDRESSES

ROOT = Path(__file__).resolve().parents[1]
CLOSED_POS = ROOT / "data" / "closed_positions.parquet"
OUT_DIR = ROOT / "data" / "directionality_classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = OUT_DIR / "traders_directionality.parquet"

# Thresholds.
FC_BALANCED       = 0.60
NTG_OFFSETTING    = 0.20
FC_SKEWED         = 0.80
NTG_DIRECTIONAL   = 0.60
ARB_LIKE_THRESH   = 0.30
TWO_SIDED_THRESH  = 0.20
PURE_DIR_FC_FLOOR = 0.85
MIN_CLOSED_POSITIONS = 50


def build_per_market_table(con: duckdb.DuckDBPyConnection) -> None:
    """Materialise per-(address, market) metrics as DuckDB table `per_market`.

    Filters operators and traders with <= MIN_CLOSED_POSITIONS upstream.
    """
    op_sql = ", ".join(f"'{a}'" for a in OPERATOR_ADDRESSES)
    con.execute(f"""
    CREATE OR REPLACE TABLE qualifying_addrs AS
    SELECT address
    FROM read_parquet('{CLOSED_POS}')
    WHERE address NOT IN ({op_sql})
    GROUP BY address
    HAVING count(*) > {MIN_CLOSED_POSITIONS}
    """)
    n = con.sql("SELECT count(*) FROM qualifying_addrs").fetchone()[0]
    print(f"  qualifying addresses (>{MIN_CLOSED_POSITIONS} closed positions, non-operator): {n:,}")

    con.execute(f"""
    CREATE OR REPLACE TABLE per_market AS
    WITH agg AS (
        SELECT
            cp.address,
            cp.market_id,
            any_value(cp.neg_risk) AS neg_risk,
            count(*) AS n_outcomes_traded,
            sum(cp.gross_usd_volume) AS market_volume,
            max(cp.gross_usd_volume) AS top_outcome_volume,
            sum(abs(cp.final_token_position)) AS gross_signed_tokens,
            abs(sum(cp.final_token_position)) AS net_signed_tokens
        FROM read_parquet('{CLOSED_POS}') cp
        JOIN qualifying_addrs qa USING (address)
        GROUP BY cp.address, cp.market_id
    )
    SELECT
        address,
        market_id,
        neg_risk,
        n_outcomes_traded,
        market_volume,
        CASE WHEN market_volume > 0
             THEN top_outcome_volume / market_volume ELSE NULL END
          AS fill_concentration,
        CASE WHEN gross_signed_tokens > 0
             THEN net_signed_tokens / gross_signed_tokens ELSE NULL END
          AS net_to_gross,
        -- discrete per-market classification flags
        CASE WHEN market_volume > 0
              AND gross_signed_tokens > 0
              AND (top_outcome_volume / market_volume) < {FC_BALANCED}
              AND (net_signed_tokens / gross_signed_tokens) < {NTG_OFFSETTING}
             THEN 1 ELSE 0 END AS is_balanced_offsetting,
        CASE WHEN market_volume > 0
              AND gross_signed_tokens > 0
              AND n_outcomes_traded >= 2
              AND (top_outcome_volume / market_volume) > {FC_SKEWED}
              AND (net_signed_tokens / gross_signed_tokens) > {NTG_DIRECTIONAL}
             THEN 1 ELSE 0 END AS is_two_sided_directional
    FROM agg
    """)
    n_mkt = con.sql("SELECT count(*) FROM per_market").fetchone()[0]
    print(f"  per-(address, market) rows: {n_mkt:,}")


def aggregate_per_trader(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Reduce per_market → one row per address using SQL only."""
    # Volume-weighted quantiles via cumulative-weight idiom: order rows within
    # each address by metric value, take the first row whose cumulative
    # volume share crosses the target quantile.
    con.execute("""
    CREATE OR REPLACE TABLE wq AS
    WITH base AS (
        SELECT address, fill_concentration, net_to_gross, market_volume
        FROM per_market
        WHERE fill_concentration IS NOT NULL
    ),
    by_fc AS (
        SELECT address, fill_concentration, market_volume,
               SUM(market_volume) OVER (PARTITION BY address ORDER BY fill_concentration
                                         ROWS UNBOUNDED PRECEDING) AS cum_w,
               SUM(market_volume) OVER (PARTITION BY address) AS total_w
        FROM base
    ),
    fc_p10 AS (
        SELECT address, MIN(fill_concentration) AS fill_concentration_p10
        FROM by_fc WHERE cum_w >= 0.10 * total_w GROUP BY address
    ),
    fc_p50 AS (
        SELECT address, MIN(fill_concentration) AS fill_concentration_p50
        FROM by_fc WHERE cum_w >= 0.50 * total_w GROUP BY address
    ),
    by_ntg AS (
        SELECT address, net_to_gross, market_volume,
               SUM(market_volume) OVER (PARTITION BY address ORDER BY net_to_gross
                                         ROWS UNBOUNDED PRECEDING) AS cum_w,
               SUM(market_volume) OVER (PARTITION BY address) AS total_w
        FROM base WHERE net_to_gross IS NOT NULL
    ),
    ntg_p50 AS (
        SELECT address, MIN(net_to_gross) AS net_to_gross_exposure
        FROM by_ntg WHERE cum_w >= 0.50 * total_w GROUP BY address
    )
    SELECT a.address, fc_p10.fill_concentration_p10,
           fc_p50.fill_concentration_p50,
           ntg_p50.net_to_gross_exposure
    FROM (SELECT DISTINCT address FROM base) a
    LEFT JOIN fc_p10 USING (address)
    LEFT JOIN fc_p50 USING (address)
    LEFT JOIN ntg_p50 USING (address)
    """)

    # Count- and volume-weighted shares of arb-like / two-sided markets.
    con.execute("""
    CREATE OR REPLACE TABLE shares AS
    SELECT
        address,
        count(*) AS n_markets_evaluated,
        sum(market_volume) AS total_market_volume_usd,
        SUM(is_balanced_offsetting)::DOUBLE / count(*) AS pct_markets_balanced_and_offsetting,
        SUM(is_two_sided_directional)::DOUBLE / count(*) AS pct_markets_two_sided_directional,
        CASE WHEN sum(market_volume) > 0
             THEN SUM(is_balanced_offsetting * market_volume) / sum(market_volume)
             ELSE NULL END AS pct_markets_balanced_and_offsetting_vw,
        CASE WHEN sum(market_volume) > 0
             THEN SUM(is_two_sided_directional * market_volume) / sum(market_volume)
             ELSE NULL END AS pct_markets_two_sided_directional_vw
    FROM per_market
    GROUP BY address
    """)

    df = con.execute("""
    SELECT s.address,
           s.n_markets_evaluated,
           s.total_market_volume_usd,
           wq.fill_concentration_p10,
           wq.fill_concentration_p50,
           wq.net_to_gross_exposure,
           s.pct_markets_balanced_and_offsetting,
           s.pct_markets_two_sided_directional,
           s.pct_markets_balanced_and_offsetting_vw,
           s.pct_markets_two_sided_directional_vw
    FROM shares s
    LEFT JOIN wq USING (address)
    """).fetchdf()
    return df


def classify_style_vec(df: pd.DataFrame) -> pd.Series:
    """Apply primary_style decision rule row-wise. Vectorised."""
    n  = df["n_markets_evaluated"]
    pa = df["pct_markets_balanced_and_offsetting_vw"]
    p2 = df["pct_markets_two_sided_directional_vw"]
    fc = df["fill_concentration_p50"]

    style = pd.Series("mixed", index=df.index, dtype=object)
    style[n < 5] = "insufficient_data"
    mask = (n >= 5)
    is_arb = mask & (pa >= ARB_LIKE_THRESH)
    style[is_arb] = "arb_like"
    is_2s = mask & ~is_arb & (p2 >= TWO_SIDED_THRESH)
    style[is_2s] = "two_sided_directional"
    is_pure = mask & ~is_arb & ~is_2s & (fc >= PURE_DIR_FC_FLOOR)
    style[is_pure] = "pure_directional"
    return style


def main() -> None:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_dir_class'")
    con.execute("SET preserve_insertion_order=false")

    print(f"reading {CLOSED_POS} ...")
    t0 = time.time()
    build_per_market_table(con)
    print(f"  per_market built in {time.time()-t0:.1f}s")

    print("aggregating per-trader metrics ...")
    t0 = time.time()
    df = aggregate_per_trader(con)
    print(f"  {len(df):,} traders in {time.time()-t0:.1f}s")

    df["primary_style"] = classify_style_vec(df)

    # Stamp thresholds as constant columns for audit.
    for k, val in {
        "_threshold_fc_balanced": FC_BALANCED,
        "_threshold_ntg_offsetting": NTG_OFFSETTING,
        "_threshold_fc_skewed": FC_SKEWED,
        "_threshold_ntg_directional": NTG_DIRECTIONAL,
        "_threshold_arb_like_share": ARB_LIKE_THRESH,
        "_threshold_two_sided_share": TWO_SIDED_THRESH,
        "_threshold_pure_dir_fc_floor": PURE_DIR_FC_FLOOR,
    }.items():
        df[k] = val

    df.to_parquet(OUT_PARQUET, index=False, compression="zstd")
    print(f"\nwrote {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size/1e6:.2f} MB)")

    print("\n=== style distribution ===")
    print(df["primary_style"].value_counts(dropna=False).to_string())

    print("\n=== quick distributions ===")
    cols = ["fill_concentration_p10", "fill_concentration_p50",
            "net_to_gross_exposure",
            "pct_markets_balanced_and_offsetting",
            "pct_markets_two_sided_directional",
            "pct_markets_balanced_and_offsetting_vw",
            "pct_markets_two_sided_directional_vw"]
    print(df[cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(3).to_string())


if __name__ == "__main__":
    main()
