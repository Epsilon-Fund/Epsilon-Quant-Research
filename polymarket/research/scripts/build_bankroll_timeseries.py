"""Build data/bankroll_timeseries.parquet — backtest-safe per-day bankroll.

For any (address, T):
    bankroll_30d_prior_at_T = max over [T-30 days, T] of
        sum(total_bought_usd) for positions open at that point in time.

Pipeline:
  1. spans            — (address, start_date, end_date, amount) from closed_positions
  2. events           — +amount on start, -amount on end+1, deduped per (addr, day)
  3. deployed_events  — cumsum of events per address → deployed at each event_date
  4. daily_deployed   — ASOF LEFT JOIN per-trader date series to deployed_events
                        (forward-fill to every day in trader's active range)
  5. bankroll_30d     — window MAX(deployed) RANGE INTERVAL '30 days' PRECEDING

Persistent work DB at data/_bankroll_build.duckdb supports resume across
runs (each stage is CREATE TABLE IF NOT EXISTS).

Output:
  data/bankroll_timeseries.parquet — (address, date, bankroll_30d_prior),
    sorted by (date, address), only days where bankroll > 0
  data/traders.parquet (rewritten) — rolling_bankroll_usd_30d added;
    old est_bankroll_usd_30d_max_approx renamed to
    est_bankroll_lifetime_peak_deprecated.
"""
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
TRADERS_PARQUET = ROOT / "data" / "traders.parquet"
CLOSED_POS = ROOT / "data" / "closed_positions.parquet"
BANKROLL_PARQUET = ROOT / "data" / "bankroll_timeseries.parquet"
DB_PATH = ROOT / "data" / "_bankroll_build.duckdb"


def step(label: str) -> float:
    print(f"\n[{label}]", flush=True)
    return time.time()


def done(t0: float) -> None:
    print(f"  ({time.time() - t0:.1f}s)", flush=True)


def _has_table(con, name: str) -> bool:
    return bool(con.sql(
        f"SELECT 1 FROM information_schema.tables WHERE table_name = '{name}'"
    ).fetchone())


def main() -> None:
    print(f"output:        {BANKROLL_PARQUET}")
    print(f"updated:       {TRADERS_PARQUET}")
    print(f"work db:       {DB_PATH}")

    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")

    # ---------- 1. Spans ----------
    if _has_table(con, "spans"):
        print("\n[1/5 spans] (cached)")
    else:
        t0 = step("1/5 building deployment spans from closed_positions")
        con.execute(f"""
            CREATE TABLE spans AS
            SELECT address,
                   date_trunc('day', first_fill_ts)::DATE AS start_date,
                   date_trunc('day', resolution_ts)::DATE AS end_date,
                   total_bought_usd AS amount
            FROM read_parquet('{CLOSED_POS}')
            WHERE total_bought_usd > 0
              AND resolution_ts IS NOT NULL
              AND resolution_ts >= first_fill_ts
        """)
        n = con.sql("SELECT count(*), count(DISTINCT address) FROM spans").fetchone()
        print(f"  {n[0]:,} spans across {n[1]:,} addresses")
        done(t0)

    # ---------- 2. Per-day events ----------
    if _has_table(con, "events"):
        print("\n[2/5 events] (cached)")
    else:
        t0 = step("2/5 building per-day events (+amount on start, -amount on end+1)")
        con.execute("""
            CREATE TABLE events AS
            WITH e AS (
                SELECT address, start_date AS event_date, amount AS delta FROM spans
                UNION ALL
                SELECT address, (end_date + INTERVAL '1 day')::DATE, -amount FROM spans
            )
            SELECT address, event_date, sum(delta) AS net_delta
            FROM e
            GROUP BY address, event_date
        """)
        n = con.sql("SELECT count(*) FROM events").fetchone()[0]
        print(f"  {n:,} deduped (address, event_date) events")
        done(t0)

    # ---------- 3. Deployed at events (cumsum) ----------
    if _has_table(con, "deployed_events"):
        print("\n[3/5 deployed_events] (cached)")
    else:
        t0 = step("3/5 cumsum events → deployed at each event_date")
        con.execute("""
            CREATE TABLE deployed_events AS
            SELECT address, event_date,
                   SUM(net_delta) OVER (
                       PARTITION BY address ORDER BY event_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS deployed
            FROM events
        """)
        done(t0)

    # ---------- 4. Daily deployed via ASOF JOIN ----------
    if _has_table(con, "daily_deployed"):
        print("\n[4/5 daily_deployed] (cached)")
    else:
        t0 = step("4/5 dense per-day series + ASOF forward-fill")
        con.execute("""
            CREATE TABLE trader_ranges AS
            SELECT address,
                   min(event_date) AS first_day,
                   max(event_date) AS last_day
            FROM events
            GROUP BY address
        """)
        con.execute("""
            CREATE TABLE daily_deployed AS
            WITH day_series AS (
                SELECT t.address, d.day::DATE AS date
                FROM trader_ranges t,
                     generate_series(t.first_day, t.last_day, INTERVAL '1 day') AS d(day)
            )
            SELECT ds.address, ds.date, COALESCE(de.deployed, 0) AS deployed
            FROM day_series ds
            ASOF LEFT JOIN deployed_events de
              ON ds.address = de.address
             AND ds.date >= de.event_date
        """)
        n = con.sql("SELECT count(*) FROM daily_deployed").fetchone()[0]
        print(f"  {n:,} (address, day) rows materialised")
        done(t0)

    # ---------- 5. Rolling 30-day max + write parquet ----------
    t0 = step("5/5 rolling 30-day max + write bankroll_timeseries.parquet")
    con.execute(f"""
        COPY (
            SELECT address, date, bankroll_30d_prior
            FROM (
                SELECT address, date,
                       MAX(deployed) OVER (
                           PARTITION BY address ORDER BY date
                           RANGE BETWEEN INTERVAL '30 days' PRECEDING
                                     AND CURRENT ROW
                       ) AS bankroll_30d_prior
                FROM daily_deployed
            ) sub
            WHERE bankroll_30d_prior > 0
            ORDER BY date, address
        )
        TO '{BANKROLL_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    size_gb = BANKROLL_PARQUET.stat().st_size / 1e9
    n = con.sql(
        f"SELECT count(*), count(DISTINCT address), min(date), max(date) "
        f"FROM read_parquet('{BANKROLL_PARQUET}')"
    ).fetchone()
    print(f"  written: {BANKROLL_PARQUET.name} — {n[0]:,} rows, "
          f"{n[1]:,} addresses, {n[2]} → {n[3]}, {size_gb:.2f} GB")
    done(t0)

    # ---------- 6. Update traders.parquet ----------
    t0 = step("6/6 updating traders.parquet — rename old col, add rolling_bankroll_usd_30d")
    con.execute(f"""
        CREATE OR REPLACE TABLE per_trader_bankroll AS
        SELECT address, MAX(bankroll_30d_prior) AS rolling_bankroll_usd_30d
        FROM read_parquet('{BANKROLL_PARQUET}')
        GROUP BY address
    """)
    new_traders = ROOT / "data" / "traders_new.parquet"
    con.execute(f"""
        COPY (
            SELECT
                t.* EXCLUDE (est_bankroll_usd_30d_max_approx),
                t.est_bankroll_usd_30d_max_approx
                    AS est_bankroll_lifetime_peak_deprecated,
                ptb.rolling_bankroll_usd_30d
            FROM read_parquet('{TRADERS_PARQUET}') t
            LEFT JOIN per_trader_bankroll ptb USING (address)
        )
        TO '{new_traders}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    # Backup + atomic swap
    backup = ROOT / "data" / "traders_pre_bankroll.parquet"
    if backup.exists():
        backup.unlink()
    TRADERS_PARQUET.rename(backup)
    new_traders.rename(TRADERS_PARQUET)
    new_size = TRADERS_PARQUET.stat().st_size / 1e6
    print(f"  backup: {backup.name}")
    print(f"  new traders.parquet: {new_size:.0f} MB")
    done(t0)

    sanity_checks(con)


def sanity_checks(con) -> None:
    print("\n\n========================== SANITY CHECKS ==========================")
    BANKROLL_PARQUET_S = str(BANKROLL_PARQUET)
    TRADERS_PARQUET_S = str(TRADERS_PARQUET)

    # 1. domah comparison vs old
    print("\n=== (a) domah: max(bankroll_30d_prior) vs old est_bankroll ===")
    print(con.sql(f"""
        SELECT
            round(rolling_bankroll_usd_30d, 0) AS new_rolling_bankroll,
            round(est_bankroll_lifetime_peak_deprecated, 0) AS old_lifetime_peak,
            round(rolling_bankroll_usd_30d - est_bankroll_lifetime_peak_deprecated, 0) AS diff,
            round(100.0 * rolling_bankroll_usd_30d
                  / nullif(est_bankroll_lifetime_peak_deprecated, 0), 2) AS pct_of_old
        FROM read_parquet('{TRADERS_PARQUET_S}')
        WHERE address = '0x9d84ce0306f8551e02efef1680475fc0f1dc1344'
    """).fetchdf().to_string(index=False))

    # 2. Coverage: % of traders with non-null rolling bankroll
    print("\n=== (b) coverage: non-null rolling_bankroll vs population ===")
    print(con.sql(f"""
        SELECT count(*) AS total_traders,
               count(*) FILTER (WHERE rolling_bankroll_usd_30d IS NOT NULL) AS with_rolling,
               count(*) FILTER (WHERE est_bankroll_lifetime_peak_deprecated IS NOT NULL) AS with_old,
               round(100.0 * count(*) FILTER (WHERE rolling_bankroll_usd_30d IS NOT NULL) / count(*), 2) AS pct_with_rolling
        FROM read_parquet('{TRADERS_PARQUET_S}')
    """).fetchdf().to_string(index=False))

    # 3. Distribution percentiles
    print("\n=== (c) bankroll percentiles (non-null only) ===")
    print(con.sql(f"""
        SELECT
            round(quantile_cont(rolling_bankroll_usd_30d, 0.50), 0) AS p50,
            round(quantile_cont(rolling_bankroll_usd_30d, 0.90), 0) AS p90,
            round(quantile_cont(rolling_bankroll_usd_30d, 0.99), 0) AS p99,
            round(max(rolling_bankroll_usd_30d), 0) AS max
        FROM read_parquet('{TRADERS_PARQUET_S}')
        WHERE rolling_bankroll_usd_30d IS NOT NULL
    """).fetchdf().to_string(index=False))

    # 4. No-future-leakage: bankroll_30d_prior(t) should never exceed
    #    sum of bought_usd for positions strictly opened by date t
    print("\n=== (d) no future leakage (sample check) ===")
    print("(picks 10 random (address, date) rows; for each, sums total_bought_usd")
    print(" for positions opened ≤ date; bankroll_30d_prior must not exceed that.)")
    print(con.sql(f"""
        WITH sample AS (
            SELECT * FROM read_parquet('{BANKROLL_PARQUET_S}') USING SAMPLE 10 ROWS
        ),
        check AS (
            SELECT s.address, s.date, s.bankroll_30d_prior,
                   coalesce(sum(cp.total_bought_usd), 0) AS sum_bought_to_date
            FROM sample s
            LEFT JOIN read_parquet('{CLOSED_POS}') cp
              ON cp.address = s.address
             AND date_trunc('day', cp.first_fill_ts)::DATE <= s.date
             AND cp.total_bought_usd > 0
            GROUP BY s.address, s.date, s.bankroll_30d_prior
        )
        SELECT substr(address, 1, 14) || '...' AS addr,
               date,
               round(bankroll_30d_prior, 0) AS bankroll,
               round(sum_bought_to_date, 0) AS cum_bought,
               CASE WHEN bankroll_30d_prior <= sum_bought_to_date + 0.01
                    THEN 'OK' ELSE 'LEAK!' END AS check
        FROM check
        ORDER BY date
    """).fetchdf().to_string(index=False))

    # 5. Comparison table for the 5 Phase-4 candidates
    print("\n=== (e) Phase-4 top candidates — old vs new bankroll ===")
    candidates = [
        ("0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029", "top_mkt_pnl_taker_heavy"),
        ("0x6a72f61820b285c11c0db3acb0ae65ac79b5a9b1", "top_overall (best guess prefix)"),
        ("0x17db3fcd93bab68e3c20bcbdba00bca2d6c0e0e0", "high_pf_small_sample (best guess)"),
        ("0xee00ba338c596497a4e95ce0eda22c54d5c8a91d", "large_sample_stable (best guess)"),
        ("0x629bc4a1e53e87a45f5cee84bca78a8870bf94d3", "four_pool (best guess)"),
    ]
    # Look up exact addresses (the prefixes above are guesses; query for actual full addresses)
    candidate_prefixes = [
        "0xd38b71f3", "0x6a72f618", "0x17db3fcd", "0xee00ba33", "0x629bc4a1",
    ]
    addrs_filter = " OR ".join(f"t.address LIKE '{p}%'" for p in candidate_prefixes)
    print(con.sql(f"""
        SELECT
            substr(t.address, 1, 14) || '...' AS addr,
            t.n_closed_positions AS n_pos,
            round(t.mkt_total_pnl, 0) AS mkt_pnl,
            round(t.est_bankroll_lifetime_peak_deprecated, 0) AS old_lifetime_peak,
            round(t.rolling_bankroll_usd_30d, 0) AS new_rolling_bankroll,
            round(100.0 * t.rolling_bankroll_usd_30d
                  / nullif(t.est_bankroll_lifetime_peak_deprecated, 0), 2) AS pct
        FROM read_parquet('{TRADERS_PARQUET_S}') t
        WHERE {addrs_filter}
        ORDER BY t.mkt_total_pnl DESC NULLS LAST
    """).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
