"""Phase 4 — materialise 6 stratified cohort pools as parquet files.

Each pool is a SQL filter on traders_filtered (operator-excluded view),
augmented with per-trader pos_std_pnl / mkt_std_pnl computed on the fly
from closed_positions (these guard columns aren't in traders.parquet).

Outputs:
    data/cohorts/{pool}.parquet  — one parquet per pool, all traders.parquet
                                   columns + pos_std_pnl + mkt_std_pnl,
                                   ordered by mkt_total_pnl DESC.

Caveats baked into the cohort definitions:
  - Sharpe-based cohorts have THREE guards: n_closed_positions > 200,
    active_days > 90, mkt_std_pnl > 1.0. Without these, naive
    sqrt(N/years) annualisation produces values up to 1.66e15.
  - Profit-factor cohorts cap at 100 (the column is already capped at
    100 in the build).
  - Kelly cohorts cap at 0.5 (>0.5 implies degenerate sample).
"""
import time
from pathlib import Path

import duckdb

from data_infra.views import load_views

ROOT = Path(__file__).resolve().parents[1]
TRADERS = ROOT / "data" / "traders.parquet"
CLOSED_POS = ROOT / "data" / "closed_positions.parquet"
COHORT_DIR = ROOT / "data" / "cohorts"
DIRECTIONALITY = ROOT / "data" / "directionality_classification" / "traders_directionality.parquet"

# Threshold for the Pool C arb gate. See
# data/directionality_classification/metric_distributions.md and
# scripts/build_traders_directionality.py for rationale. We use 0.10
# (looser than the population-wide arb_like cut of 0.30) because the other
# Pool C gates (negrisk_volume_share > 0.7, mkt_profit_factor > 1.3,
# n_closed_positions > 200) already preselect aggressively, and because the
# population diagnostic shows that NegRisk-heavy traders are *structurally*
# not arb-shaped (zero of 113 current Pool C members exceed 0.20 arb_vw).
# 0.10 keeps a handful of the most arb-leaning current members; the operator
# can tighten if the cohort proves directional-contaminated downstream.
POOL_C_ARB_THRESHOLD = 0.10


POOLS: dict[str, str] = {
    "high_sharpe_directional": """
        SELECT * FROM traders_aug
        WHERE mkt_sharpe > 1.5
          AND mkt_sharpe < 10
          AND mkt_std_pnl > 1.0
          AND n_closed_positions > 200
          AND active_days > 90
          AND negrisk_volume_share < 0.5
          AND NOT is_operator_like
        ORDER BY mkt_total_pnl DESC NULLS LAST
    """,
    "high_profit_factor_with_size": """
        SELECT * FROM traders_aug
        WHERE mkt_profit_factor > 2.0
          AND mkt_profit_factor < 100
          AND mkt_total_pnl > 50000
          AND n_closed_positions > 100
          AND active_days > 90
          AND NOT is_operator_like
        ORDER BY mkt_total_pnl DESC NULLS LAST
    """,
    "negrisk_specialists": f"""
        -- Pool C — arb-shaped NegRisk specialists.
        -- pct_markets_balanced_and_offsetting_vw replaces the implicit
        -- phantom-score arb signal: it measures the share of a trader's
        -- volume on markets with balanced fills AND offsetting positions
        -- across outcomes — the genuine arb fingerprint. phantom_position_score
        -- is retained as a column in traders_aug for backward compat but no
        -- longer used to gate this pool.
        SELECT * FROM traders_aug
        WHERE negrisk_volume_share > 0.7
          AND mkt_total_pnl > 50000
          AND n_closed_positions > 200
          AND active_days > 90
          AND mkt_profit_factor > 1.3
          AND pct_markets_balanced_and_offsetting_vw > {POOL_C_ARB_THRESHOLD}
          AND NOT is_operator_like
        ORDER BY mkt_total_pnl DESC NULLS LAST
    """,
    "sports_directional_fast": """
        SELECT * FROM traders_aug
        WHERE negrisk_volume_share < 0.3
          AND mkt_sharpe > 1.0
          AND mkt_sharpe < 10
          AND mkt_std_pnl > 1.0
          AND style_avg_holding_hours < 48
          AND n_closed_positions > 200
          AND active_days > 90
          AND NOT is_operator_like
        ORDER BY mkt_total_pnl DESC NULLS LAST
    """,
    "patient_accumulators": """
        SELECT * FROM traders_aug
        WHERE style_role_balance > 0.7
          AND style_avg_holding_hours > 168
          AND mkt_total_pnl > 100000
          AND n_closed_positions > 100
          AND active_days > 180
          AND NOT is_operator_like
        ORDER BY mkt_total_pnl DESC NULLS LAST
    """,
    "high_kelly_edge": """
        SELECT * FROM traders_aug
        WHERE mkt_kelly_fraction > 0.05
          AND mkt_kelly_fraction < 0.5
          AND n_closed_positions > 200
          AND active_days > 90
          AND mkt_dollar_win_rate > 0.55
          AND NOT is_operator_like
        ORDER BY mkt_total_pnl DESC NULLS LAST
    """,
}


def build(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, sql in POOLS.items():
        out_path = COHORT_DIR / f"{name}.parquet"
        t0 = time.time()
        con.execute(f"CREATE OR REPLACE TABLE _cohort_tmp AS {sql}")
        con.execute(f"COPY _cohort_tmp TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        n = con.sql("SELECT count(*) FROM _cohort_tmp").fetchone()[0]
        counts[name] = n
        print(f"  {name:35s} {n:>6,}   ({time.time() - t0:.1f}s)  → {out_path.name}")
    con.execute("DROP TABLE IF EXISTS _cohort_tmp")
    return counts


def summary(con: duckdb.DuckDBPyConnection) -> None:
    print("\n=== per-pool summary ===")
    rows = []
    for name in POOLS:
        path = COHORT_DIR / f"{name}.parquet"
        df = con.sql(f"""
            SELECT
                count(*) AS n,
                round(median(n_closed_positions), 0) AS med_n_pos,
                round(median(active_days), 0) AS med_days,
                round(median(mkt_sharpe), 2) AS med_mkt_sharpe,
                round(median(mkt_profit_factor), 2) AS med_pf,
                round(median(mkt_total_pnl), 0) AS med_pnl,
                round(median(phantom_position_score), 2) AS med_phantom
            FROM read_parquet('{path}')
        """).fetchdf().iloc[0]
        rows.append({"pool": name, **df.to_dict()})

    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== union overlap (addresses qualifying for k+ pools) ===")
    paths = ", ".join(f"'{COHORT_DIR / (p + '.parquet')}'" for p in POOLS)
    con.execute(f"""
        CREATE OR REPLACE TABLE _all_cohorts AS
        SELECT * FROM read_parquet([{paths}], filename = TRUE)
    """)
    print(con.sql("""
        WITH per_addr AS (
            SELECT address, count(DISTINCT filename) AS n_pools
            FROM _all_cohorts GROUP BY address
        )
        SELECT
            sum(CASE WHEN n_pools >= 1 THEN 1 ELSE 0 END) AS qualified_in_1plus,
            sum(CASE WHEN n_pools >= 2 THEN 1 ELSE 0 END) AS in_2plus,
            sum(CASE WHEN n_pools >= 3 THEN 1 ELSE 0 END) AS in_3plus,
            sum(CASE WHEN n_pools >= 4 THEN 1 ELSE 0 END) AS in_4plus,
            sum(CASE WHEN n_pools >= 5 THEN 1 ELSE 0 END) AS in_5plus,
            sum(CASE WHEN n_pools = 6 THEN 1 ELSE 0 END) AS in_all_6
        FROM per_addr
    """).fetchdf().to_string(index=False))


def sanity(con: duckdb.DuckDBPyConnection) -> None:
    """Confirm guards held: no qualifying row in any pool has artifact-shaped values."""
    print("\n=== guard sanity (must be 0 across all rows) ===")
    paths = ", ".join(f"'{COHORT_DIR / (p + '.parquet')}'" for p in POOLS)
    con.execute(f"""
        CREATE OR REPLACE VIEW _all_pool_rows AS
        SELECT * FROM read_parquet([{paths}], filename = TRUE)
    """)
    df = con.sql("""
        SELECT
            sum(CASE WHEN mkt_sharpe > 100 THEN 1 ELSE 0 END) AS sharpe_above_100,
            sum(CASE WHEN mkt_kelly_fraction > 1 THEN 1 ELSE 0 END) AS kelly_above_1,
            sum(CASE WHEN pos_std_pnl < 0.01 THEN 1 ELSE 0 END) AS pos_std_below_1cent,
            sum(CASE WHEN mkt_std_pnl < 0.01 THEN 1 ELSE 0 END) AS mkt_std_below_1cent
        FROM _all_pool_rows
    """).fetchdf()
    print(df.to_string(index=False))
    failed = df.iloc[0].to_dict()
    bad = {k: v for k, v in failed.items() if v and v > 0}
    if bad:
        print(f"\n!! GUARD FAILURE: {bad}")
        print("Surfacing offending rows:")
        print(con.sql("""
            SELECT filename, address, n_closed_positions, mkt_sharpe,
                   mkt_kelly_fraction, pos_std_pnl, mkt_std_pnl
            FROM _all_pool_rows
            WHERE mkt_sharpe > 100 OR mkt_kelly_fraction > 1
               OR pos_std_pnl < 0.01 OR mkt_std_pnl < 0.01
            LIMIT 20
        """).fetchdf().to_string(index=False))
        raise SystemExit("guards not working — halt")
    print("  PASS — guards held")


def main() -> None:
    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")

    print("loading views.sql...")
    t0 = time.time()
    load_views(con)
    print(f"  loaded in {time.time() - t0:.2f}s")

    print("\ncomputing pos_std_pnl / mkt_std_pnl from closed_positions...")
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TABLE m_std_aux AS
        WITH pos_lvl AS (
            SELECT address, stddev_pop(realised_pnl) AS pos_std_pnl
            FROM read_parquet('{CLOSED_POS}')
            GROUP BY address
        ),
        mkt_per_market AS (
            SELECT address, market_id, sum(realised_pnl) AS market_pnl
            FROM read_parquet('{CLOSED_POS}')
            GROUP BY address, market_id
        ),
        mkt_lvl AS (
            SELECT address, stddev_pop(market_pnl) AS mkt_std_pnl
            FROM mkt_per_market GROUP BY address
        )
        SELECT pl.address, pl.pos_std_pnl, ml.mkt_std_pnl
        FROM pos_lvl pl
        LEFT JOIN mkt_lvl ml USING (address)
    """)
    n = con.sql("SELECT count(*) FROM m_std_aux").fetchone()[0]
    print(f"  {n:,} rows  ({time.time() - t0:.1f}s)")

    # Augmented view of traders_filtered with pos_std / mkt_std joined.
    # Also joins the directionality classification sidecar so Pool C can gate
    # on pct_markets_balanced_and_offsetting_vw. The sidecar is built by
    # scripts/build_traders_directionality.py; if it's missing the pool that
    # uses it will fail loudly (we don't fall back to phantom_position_score).
    if not DIRECTIONALITY.exists():
        raise SystemExit(
            f"missing directionality sidecar: {DIRECTIONALITY}\n"
            f"run scripts/build_traders_directionality.py first."
        )
    con.execute(f"""
        CREATE OR REPLACE VIEW traders_aug AS
        SELECT t.*,
               s.pos_std_pnl, s.mkt_std_pnl,
               d.fill_concentration_p10,
               d.fill_concentration_p50,
               d.net_to_gross_exposure,
               d.pct_markets_balanced_and_offsetting,
               d.pct_markets_balanced_and_offsetting_vw,
               d.pct_markets_two_sided_directional,
               d.pct_markets_two_sided_directional_vw,
               d.primary_style
        FROM traders_filtered t
        LEFT JOIN m_std_aux s USING (address)
        LEFT JOIN read_parquet('{DIRECTIONALITY}') d USING (address)
    """)

    print("\nbuilding cohorts...")
    build(con)
    summary(con)
    sanity(con)


if __name__ == "__main__":
    main()
