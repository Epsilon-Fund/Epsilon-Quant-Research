"""Sanity checks for bankroll_timeseries.parquet + updated traders.parquet.

Re-runs the (d) future-leakage check (the original used `check` as a CTE
alias, which is a DuckDB reserved keyword) and the (e) 5-candidate
comparison table with the verified-full addresses.
"""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
TRADERS = ROOT / "data" / "traders.parquet"
BANKROLL = ROOT / "data" / "bankroll_timeseries.parquet"
CLOSED_POS = ROOT / "data" / "closed_positions.parquet"

CANDIDATES = [
    ("top_mkt_pnl_taker_heavy",  "0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029"),
    ("top_overall",              "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee"),
    ("high_pf_small_sample",     "0x17db3fcd93ba12d38382a0cade24b200185c5f6d"),
    ("large_sample_stable",      "0xee00ba338c59557141789b127927a55f5cc5cea1"),
    ("four_pool",                "0x629bc4a1e53e1d475beb7ea3d388791e96dd995a"),
]


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    print("========================== SANITY CHECKS ==========================")

    # (a) domah comparison
    print("\n=== (a) domah: new rolling_bankroll vs old lifetime_peak ===")
    print(con.sql(f"""
        SELECT
            round(rolling_bankroll_usd_30d, 0)              AS new_rolling,
            round(est_bankroll_lifetime_peak_deprecated, 0) AS old_peak,
            round(rolling_bankroll_usd_30d
                  - est_bankroll_lifetime_peak_deprecated, 0) AS diff,
            round(100.0 * rolling_bankroll_usd_30d
                  / nullif(est_bankroll_lifetime_peak_deprecated, 0), 2) AS pct_of_old
        FROM read_parquet('{TRADERS}')
        WHERE address = '0x9d84ce0306f8551e02efef1680475fc0f1dc1344'
    """).fetchdf().to_string(index=False))

    # (b) coverage
    print("\n=== (b) coverage (population-wide) ===")
    print(con.sql(f"""
        SELECT
            count(*) AS total_traders,
            count(*) FILTER (WHERE rolling_bankroll_usd_30d IS NOT NULL) AS with_new,
            count(*) FILTER (WHERE est_bankroll_lifetime_peak_deprecated IS NOT NULL) AS with_old,
            round(100.0 * count(*) FILTER (WHERE rolling_bankroll_usd_30d IS NOT NULL) / count(*), 2) AS pct_new,
            count(*) FILTER (WHERE est_bankroll_lifetime_peak_deprecated IS NOT NULL
                              AND rolling_bankroll_usd_30d IS NULL) AS old_but_not_new
        FROM read_parquet('{TRADERS}')
    """).fetchdf().to_string(index=False))

    # (c) percentile distribution
    print("\n=== (c) rolling_bankroll percentile distribution ===")
    print(con.sql(f"""
        SELECT
            round(quantile_cont(rolling_bankroll_usd_30d, 0.50), 0) AS p50,
            round(quantile_cont(rolling_bankroll_usd_30d, 0.75), 0) AS p75,
            round(quantile_cont(rolling_bankroll_usd_30d, 0.90), 0) AS p90,
            round(quantile_cont(rolling_bankroll_usd_30d, 0.95), 0) AS p95,
            round(quantile_cont(rolling_bankroll_usd_30d, 0.99), 0) AS p99,
            round(max(rolling_bankroll_usd_30d), 0) AS max_val
        FROM read_parquet('{TRADERS}')
        WHERE rolling_bankroll_usd_30d IS NOT NULL
    """).fetchdf().to_string(index=False))

    # (d) FUTURE-LEAKAGE CHECK (fixed: renamed `check` → `chk`)
    print("\n=== (d) no future leakage (sample of 20 (address, date) rows) ===")
    print("For each, sum total_bought_usd over positions with first_fill_ts <= date.")
    print("bankroll_30d_prior(t) <= sum_bought_by_t (since the rolling 30-day max can")
    print("only reach values that were once concurrent deployed, which is bounded by")
    print("cumulative-bought-by-date). Failures would indicate temporal leakage.")
    print()
    print(con.sql(f"""
        WITH samp AS (
            SELECT * FROM read_parquet('{BANKROLL}') USING SAMPLE 20 ROWS
        ),
        chk AS (
            SELECT s.address, s.date, s.bankroll_30d_prior,
                   coalesce(sum(cp.total_bought_usd), 0) AS cum_bought_by_date
            FROM samp s
            LEFT JOIN read_parquet('{CLOSED_POS}') cp
              ON cp.address = s.address
             AND date_trunc('day', cp.first_fill_ts)::DATE <= s.date
             AND cp.total_bought_usd > 0
            GROUP BY s.address, s.date, s.bankroll_30d_prior
        )
        SELECT substr(address, 1, 14) || '...' AS addr,
               date,
               round(bankroll_30d_prior, 0) AS bankroll,
               round(cum_bought_by_date, 0) AS cum_bought,
               CASE WHEN bankroll_30d_prior <= cum_bought_by_date + 0.01
                    THEN 'OK' ELSE 'LEAK!' END AS verdict
        FROM chk
        ORDER BY date
    """).fetchdf().to_string(index=False))

    # (e) 5 candidates table
    print("\n=== (e) Phase-4 top candidates — old lifetime_peak vs new rolling_30d ===")
    addrs = ", ".join(f"'{a}'" for _, a in CANDIDATES)
    print(con.sql(f"""
        SELECT
            substr(t.address, 1, 14) || '...'                AS addr,
            t.n_closed_positions                              AS n_pos,
            t.active_days                                      AS days,
            round(t.mkt_total_pnl, 0)                          AS mkt_pnl,
            round(t.est_bankroll_lifetime_peak_deprecated, 0)  AS old_lifetime_peak,
            round(t.rolling_bankroll_usd_30d, 0)               AS new_rolling_30d,
            round(t.rolling_bankroll_usd_30d
                  - t.est_bankroll_lifetime_peak_deprecated, 0) AS diff,
            round(100.0 * t.rolling_bankroll_usd_30d
                  / nullif(t.est_bankroll_lifetime_peak_deprecated, 0), 2) AS pct_of_old
        FROM read_parquet('{TRADERS}') t
        WHERE t.address IN ({addrs})
        ORDER BY t.mkt_total_pnl DESC NULLS LAST
    """).fetchdf().to_string(index=False))

    # (f) Bankroll point-in-time inspection — domah at 5 sample dates
    print("\n=== (f) domah bankroll_30d_prior at 5 sample dates (point-in-time) ===")
    print("Cross-check: at each date, manually verify against deployed capital")
    print("from his positions that were open during [date-30, date].")
    print()
    print(con.sql(f"""
        WITH samp AS (
            SELECT date, bankroll_30d_prior
            FROM read_parquet('{BANKROLL}')
            WHERE address = '0x9d84ce0306f8551e02efef1680475fc0f1dc1344'
              AND date IN (
                  DATE '2023-12-01', DATE '2024-06-01', DATE '2024-11-01',
                  DATE '2025-06-01', DATE '2026-04-01'
              )
            ORDER BY date
        ),
        manual AS (
            SELECT s.date, round(s.bankroll_30d_prior, 0) AS bankroll,
                round((
                    SELECT max(deployed) FROM (
                        SELECT sum(cp.total_bought_usd) AS deployed
                        FROM read_parquet('{CLOSED_POS}') cp
                        WHERE cp.address = '0x9d84ce0306f8551e02efef1680475fc0f1dc1344'
                          AND cp.total_bought_usd > 0
                          AND date_trunc('day', cp.first_fill_ts)::DATE <= s.date
                          AND date_trunc('day', cp.resolution_ts)::DATE >= s.date - INTERVAL '30 days'
                    )
                ), 0) AS upper_bound_deployed_sum
            FROM samp s
        )
        SELECT * FROM manual ORDER BY date
    """).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
