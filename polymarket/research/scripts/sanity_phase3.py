"""Pre-Phase-4 sanity queries against traders.parquet.

Three checks (stdout only):
  a. mkt_total_pnl vs pos_total_pnl drift — should be ~0 by construction
  b. Sharpe distribution — flags annualisation artifacts (p99 >> 5)
  c. drawdown vs total PnL physical sanity
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
TRADERS = ROOT / "data" / "traders.parquet"


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(f"CREATE VIEW traders AS SELECT * FROM read_parquet('{TRADERS}')")
    con.execute("CREATE VIEW traders_filtered AS SELECT * FROM traders WHERE NOT is_operator_like")

    print("\n=== (a) mkt_total_pnl vs pos_total_pnl drift ===")
    print("(should be ~0 — they aggregate the same realised_pnl differently)")
    print(con.sql("""
        SELECT
            count(*) FILTER (WHERE abs(mkt_total_pnl - pos_total_pnl) > 1.0
                              AND n_closed_positions > 10) AS n_drift_above_1usd,
            count(*) FILTER (WHERE abs(mkt_total_pnl - pos_total_pnl) > 0.01
                              AND n_closed_positions > 10) AS n_drift_above_1cent,
            round(max(abs(mkt_total_pnl - pos_total_pnl)), 4) AS max_abs_drift,
            count(*) AS total_rows
        FROM traders
    """).fetchdf().to_string(index=False))

    print("\n  --- top 5 largest drifts (if any) ---")
    df = con.sql("""
        SELECT substr(address, 1, 14) || '...' AS addr,
               n_closed_positions,
               round(pos_total_pnl, 4) AS pos_pnl,
               round(mkt_total_pnl, 4) AS mkt_pnl,
               round(mkt_total_pnl - pos_total_pnl, 4) AS drift,
               is_operator_like
        FROM traders
        WHERE n_closed_positions > 10
        ORDER BY abs(mkt_total_pnl - pos_total_pnl) DESC NULLS LAST
        LIMIT 5
    """).fetchdf()
    print(df.to_string(index=False))

    print("\n=== (b) Sharpe distribution (traders_filtered, n_closed_positions > 50) ===")
    print("(p99 should be < 5 for honest data; > 10 ⇒ annualisation artifacts)")
    print(con.sql("""
        SELECT
            count(*) AS n_traders,
            round(quantile_cont(pos_sharpe, 0.50), 4) AS pos_sharpe_p50,
            round(quantile_cont(pos_sharpe, 0.90), 4) AS pos_sharpe_p90,
            round(quantile_cont(pos_sharpe, 0.95), 4) AS pos_sharpe_p95,
            round(quantile_cont(pos_sharpe, 0.99), 4) AS pos_sharpe_p99,
            round(max(pos_sharpe), 4) AS pos_sharpe_max,
            round(quantile_cont(mkt_sharpe, 0.50), 4) AS mkt_sharpe_p50,
            round(quantile_cont(mkt_sharpe, 0.95), 4) AS mkt_sharpe_p95,
            round(quantile_cont(mkt_sharpe, 0.99), 4) AS mkt_sharpe_p99,
            round(max(mkt_sharpe), 4) AS mkt_sharpe_max
        FROM traders_filtered
        WHERE n_closed_positions > 50
          AND pos_sharpe IS NOT NULL AND mkt_sharpe IS NOT NULL
    """).fetchdf().to_string(index=False))

    print("\n  --- top 10 mkt_sharpe (filtered, n_pos > 50) — check for outliers ---")
    print(con.sql("""
        SELECT substr(address, 1, 14) || '...' AS addr,
               n_closed_positions AS n_pos,
               round(mkt_sharpe, 2) AS mkt_sharpe,
               round(mkt_total_pnl, 0) AS mkt_pnl,
               round(mkt_win_rate, 3) AS win,
               active_days,
               round(phantom_position_score, 2) AS phantom
        FROM traders_filtered
        WHERE n_closed_positions > 50 AND mkt_sharpe IS NOT NULL
        ORDER BY mkt_sharpe DESC NULLS LAST
        LIMIT 10
    """).fetchdf().to_string(index=False))

    print("\n=== (c) drawdown sanity ===")
    print("(eyeball: drawdown bigger than abs(total_pnl) is physically possible — "
          "trader had a deep mid-period drawdown then recovered. Look for absurdity.)")
    print("\n  --- count where pos_max_drawdown_usd > abs(pos_total_pnl) AND pos_total_pnl > 0 ---")
    print(con.sql("""
        SELECT
            count(*) FILTER (WHERE pos_max_drawdown_usd > abs(pos_total_pnl)
                              AND pos_total_pnl > 0
                              AND n_closed_positions > 10) AS n_recovered_traders,
            count(*) AS total
        FROM traders
    """).fetchdf().to_string(index=False))

    print("\n  --- top 20 by drawdown:total_pnl ratio (winners only, n_pos>50) ---")
    print(con.sql("""
        SELECT substr(address, 1, 14) || '...' AS addr,
               n_closed_positions AS n_pos,
               round(pos_total_pnl, 0) AS pos_pnl,
               round(pos_max_drawdown_usd, 0) AS pos_dd,
               round(pos_max_drawdown_usd / pos_total_pnl, 2) AS dd_to_pnl,
               round(mkt_max_drawdown_usd, 0) AS mkt_dd,
               is_operator_like
        FROM traders
        WHERE pos_total_pnl > 1000 AND n_closed_positions > 50
          AND pos_max_drawdown_usd IS NOT NULL
        ORDER BY (pos_max_drawdown_usd / pos_total_pnl) DESC NULLS LAST
        LIMIT 20
    """).fetchdf().to_string(index=False))

    print("\n=== (d) bankroll column added — quick distribution check ===")
    print(con.sql("""
        SELECT
            count(*) AS total_rows,
            count(*) FILTER (WHERE est_bankroll_usd_30d_max_approx IS NOT NULL) AS non_null,
            round(quantile_cont(est_bankroll_usd_30d_max_approx, 0.50), 0) AS p50,
            round(quantile_cont(est_bankroll_usd_30d_max_approx, 0.90), 0) AS p90,
            round(quantile_cont(est_bankroll_usd_30d_max_approx, 0.99), 0) AS p99,
            round(max(est_bankroll_usd_30d_max_approx), 0) AS max_val
        FROM traders
    """).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
