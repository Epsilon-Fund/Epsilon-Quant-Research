"""Phase 4 diagnostics — cross-pool overlap, top-20 union, multi-pool members,
domah profile, and inspection candidates.

Run after scripts/build_cohorts.py has produced data/cohorts/*.parquet.
"""
from pathlib import Path

import duckdb
import numpy as np

from data_infra.trader_profile import POOL_NAMES, profile_trader

ROOT = Path(__file__).resolve().parents[1]
COHORT_DIR = ROOT / "data" / "cohorts"
TRADERS = ROOT / "data" / "traders.parquet"

DOMAH = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(f"CREATE VIEW traders AS SELECT * FROM read_parquet('{TRADERS}')")
    for pool in POOL_NAMES:
        path = COHORT_DIR / f"{pool}.parquet"
        con.execute(f"CREATE OR REPLACE VIEW pool_{pool} AS SELECT * FROM read_parquet('{path}')")
    paths = ", ".join(f"'{COHORT_DIR / (p + '.parquet')}'" for p in POOL_NAMES)
    con.execute(f"CREATE OR REPLACE TABLE _all AS SELECT * FROM read_parquet([{paths}], filename = TRUE)")

    print("\n=========================================================")
    print("Phase 4 diagnostics")
    print("=========================================================")

    # 1. Per-pool n_qualifying summary
    print("\n=== per-pool n_qualifying ===")
    for pool in POOL_NAMES:
        n = con.sql(f"SELECT count(*) FROM pool_{pool}").fetchone()[0]
        print(f"  {pool:35s} {n:>6,}")

    # 2. Cross-pool overlap matrix
    print("\n=== cross-pool overlap matrix (count of addresses qualifying for both) ===")
    short = {
        "high_sharpe_directional": "A_sharpe",
        "high_profit_factor_with_size": "B_pf",
        "negrisk_specialists": "C_negr",
        "sports_directional_fast": "D_sport",
        "patient_accumulators": "E_patient",
        "high_kelly_edge": "F_kelly",
    }
    mat = np.zeros((len(POOL_NAMES), len(POOL_NAMES)), dtype=int)
    for i, p1 in enumerate(POOL_NAMES):
        for j, p2 in enumerate(POOL_NAMES):
            n = con.sql(
                f"SELECT count(*) FROM (SELECT address FROM pool_{p1} INTERSECT SELECT address FROM pool_{p2})"
            ).fetchone()[0]
            mat[i, j] = n
    header = "             " + " ".join(f"{short[p]:>10s}" for p in POOL_NAMES)
    print(header)
    for i, p in enumerate(POOL_NAMES):
        row = " ".join(f"{mat[i,j]:>10,}" for j in range(len(POOL_NAMES)))
        print(f"{short[p]:>12s} {row}")

    # 3. Top-20 union table
    print("\n=== top 20 across all pools (by mkt_total_pnl) ===")
    print("(mkt_sharpe is DIAGNOSTIC; trust requires n_pos>200 AND active_days>90)")
    df = con.sql("""
        WITH per_addr AS (
            SELECT address, count(DISTINCT filename) AS n_pools
            FROM _all GROUP BY address
        )
        SELECT
            substr(t.address, 1, 14) || '...' AS addr,
            p.n_pools AS pools,
            round(t.mkt_total_pnl, 0) AS mkt_pnl,
            round(t.mkt_profit_factor, 2) AS pf,
            round(t.mkt_dollar_win_rate, 3) AS dwr,
            round(t.mkt_sharpe, 2) AS mkt_sharpe_diag,
            t.n_closed_positions AS n_pos,
            t.active_days AS days,
            round(t.phantom_position_score, 2) AS phantom,
            round(t.negrisk_volume_share, 2) AS nr,
            round(t.style_role_balance, 2) AS rolebal,
            round(t.est_bankroll_usd_30d_max_approx, 0) AS bnkr_peak
        FROM traders t JOIN per_addr p USING (address)
        ORDER BY t.mkt_total_pnl DESC NULLS LAST
        LIMIT 20
    """).fetchdf()
    print(df.to_string(index=False))

    # 4. Multi-pool members (3+ pools)
    print("\n=== multi-pool members (3+ pools) ===")
    df = con.sql("""
        WITH per_addr AS (
            SELECT address, count(DISTINCT filename) AS n_pools,
                   list(DISTINCT regexp_extract(filename, '([^/]+)\\.parquet$', 1)) AS pools_list
            FROM _all GROUP BY address
        )
        SELECT
            substr(t.address, 1, 14) || '...' AS addr,
            p.n_pools,
            round(t.mkt_total_pnl, 0) AS mkt_pnl,
            round(t.mkt_profit_factor, 2) AS pf,
            round(t.mkt_dollar_win_rate, 3) AS dwr,
            t.n_closed_positions AS n_pos,
            t.active_days AS days,
            round(t.phantom_position_score, 2) AS phantom,
            round(t.negrisk_volume_share, 2) AS nr,
            round(t.style_role_balance, 2) AS rolebal,
            p.pools_list AS pools
        FROM traders t JOIN per_addr p USING (address)
        WHERE p.n_pools >= 3
        ORDER BY t.mkt_total_pnl DESC NULLS LAST
    """).fetchdf()
    print(f"({len(df)} addresses)")
    print(df.head(40).to_string(index=False))

    # 5. domah profile (dict mode, headline only)
    print("\n=== domah profile (sanity check) ===")
    p = profile_trader(DOMAH, con=con)
    print("header:", p["header"])
    print("\nheadline_metrics:")
    for k, v in p["headline_metrics"].items():
        print(f"  {k}: {v}")
    print("\npools qualified:", p["header"]["pools_qualified"])
    print("\ncohort_positioning:")
    for pool, vals in p["cohort_positioning"].items():
        print(f"  {pool}: pnl_pct={vals['pnl_percentile']:.0f}, pf_pct={vals['pf_percentile']:.0f}")
    print(f"\nstyle_role_balance: {p['style']['style_role_balance_1eqMaker_0eqTaker']:.2f}  (1.0 = pure maker)")
    print(f"style_avg_holding_hours: {p['style']['style_avg_holding_hours']:.1f}")
    print(f"capital footprint: ${p['capital_footprint']['est_bankroll_usd_30d_max_approx']:,.0f}")
    print(f"  → {p['capital_footprint']['label']}")

    # 6. Confirm guards held
    print("\n=== guard sanity (must be 0) ===")
    print(con.sql("""
        SELECT
            sum(CASE WHEN mkt_sharpe > 100 THEN 1 ELSE 0 END) AS sharpe_above_100,
            sum(CASE WHEN mkt_kelly_fraction > 1 THEN 1 ELSE 0 END) AS kelly_above_1
        FROM _all
    """).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
