"""Build data/traders.parquet — Phase 3 / per-trader metrics + style profile.

One row per address that touched at least one closed market. Metrics:
  - Activity (fills, markets, volume, time range)
  - Position-level PnL stats (pos_*)  — primary, but inflated for NegRisk arb
  - Market-level PnL stats (mkt_*)    — NegRisk-robust alternative
  - phantom_position_score            — exposes split/merge arb behaviour
  - Style profile (style_*)           — maker/taker, fill sizes, holding, sub-sec
  - is_operator_like                  — deny-list ∪ heuristic flag

Run from polymarket/research/ as:
    PYTHONPATH=. uv run python scripts/build_traders_table.py

Design notes:
  - Aggregates produce ~2.6M rows (one per address). Hash-agg fits in memory.
  - Drawdown uses window functions; sort cost dominates. ~270M rows × ~60 B
    pruned columns = 16 GB sorted intermediate, fits with modest spill.
  - Sub-second clustering is approximated via 1-second bucketing (no window
    function): per-(address, sec) count > 1 ⇒ all those fills counted as
    bursts. Close enough for operator detection; cheap to compute.
  - n_distinct_counterparties uses approx_count_distinct (HyperLogLog) —
    exact COUNT(DISTINCT) over 1B fills × 2 sides is too heavy.
  - CROSS JOIN with VALUES(('maker'),('taker')) is used everywhere instead
    of UNION ALL on the same source, to avoid CTE auto-materialisation
    (the trap that OOM'd Phase 2 with 200+ GB temp).
"""
import time
from pathlib import Path

from data_infra.duck import connect
from data_infra.views import load_views
from data_infra.operator_denylist import OPERATOR_ADDRESSES

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "traders.parquet"
CLOSED_POSITIONS = ROOT / "data" / "closed_positions.parquet"


def step(label: str):
    return _Step(label)


class _Step:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        print(f"\n[{self.label}]", flush=True)
        self.t0 = time.time()
        return self

    def __exit__(self, *_):
        print(f"  ({time.time() - self.t0:.1f}s)", flush=True)


DB_PATH = ROOT / "data" / "_traders_build.duckdb"


def _has_table(con, name: str) -> bool:
    return bool(con.sql(
        f"SELECT 1 FROM information_schema.tables WHERE table_name = '{name}'"
    ).fetchone())


def maybe_run(con, label: str, table_name: str, sql: str) -> None:
    """Run `sql` (which CREATEs `table_name`) only if the table doesn't exist.

    The persistent DuckDB file at DB_PATH carries completed stages across
    runs. Re-running after a crash skips the stages that already finished.
    """
    if _has_table(con, table_name):
        print(f"\n[{label}] (cached — skipped)", flush=True)
        return
    with step(label):
        con.execute(sql)


def main() -> None:
    print(f"output: {OUT_PATH}")
    print(f"work db: {DB_PATH}")
    import duckdb
    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")

    print("loading views.sql...")
    t0 = time.time()
    load_views(con)
    print(f"  loaded in {time.time() - t0:.2f}s")

    # Register closed_positions as a view (read once, used many times).
    con.execute(
        f"CREATE OR REPLACE VIEW closed_positions "
        f"AS SELECT * FROM read_parquet('{CLOSED_POSITIONS}')"
    )

    # ---------- 1. Activity + position-level PnL ----------
    with step("1/8 activity + position-level PnL (m_pos)"):
        con.execute("""
            CREATE TABLE IF NOT EXISTS m_pos AS
            WITH base AS (
                SELECT
                    address,
                    count(*) AS n_closed_positions,
                    count(DISTINCT market_id) AS n_distinct_markets,
                    sum(n_fills) AS n_fills_total,
                    sum(total_bought_usd + total_sold_usd) AS total_volume_usd,
                    min(first_fill_ts) AS first_activity_ts,
                    max(last_fill_ts) AS last_activity_ts,
                    sum(realised_pnl) AS pos_total_pnl,
                    sum(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END) AS pos_winners,
                    sum(CASE WHEN realised_pnl < 0 THEN 1 ELSE 0 END) AS pos_losers,
                    sum(CASE WHEN realised_pnl > 0 THEN realised_pnl ELSE 0 END)
                        AS pos_sum_win,
                    sum(CASE WHEN realised_pnl < 0 THEN -realised_pnl ELSE 0 END)
                        AS pos_sum_loss_abs,
                    avg(realised_pnl) AS pos_mean_pnl,
                    stddev_pop(realised_pnl) AS pos_std_pnl,
                    sqrt(avg(CASE WHEN realised_pnl < 0
                                  THEN realised_pnl * realised_pnl ELSE 0 END))
                        AS pos_downside_std,
                    avg(CASE WHEN holding_duration_hours >= 0
                             THEN holding_duration_hours ELSE NULL END)
                        AS style_avg_holding_hours,
                    median(CASE WHEN holding_duration_hours >= 0
                                THEN holding_duration_hours ELSE NULL END)
                        AS style_median_holding_hours
                FROM closed_positions
                GROUP BY address
            )
            SELECT
                address,
                n_closed_positions,
                n_distinct_markets,
                n_closed_positions AS n_distinct_outcomes,  -- 1 row per outcome
                n_fills_total,
                total_volume_usd,
                first_activity_ts,
                last_activity_ts,
                DATEDIFF('day', first_activity_ts, last_activity_ts) AS active_days,
                pos_total_pnl,
                pos_winners,
                pos_losers,
                CASE WHEN pos_winners + pos_losers > 0
                     THEN pos_winners::DOUBLE / (pos_winners + pos_losers)
                     ELSE NULL END AS pos_win_rate,
                CASE WHEN pos_sum_win + pos_sum_loss_abs > 0
                     THEN pos_sum_win / (pos_sum_win + pos_sum_loss_abs)
                     ELSE NULL END AS pos_dollar_win_rate,
                CASE WHEN pos_winners > 0
                     THEN pos_sum_win / pos_winners ELSE NULL END AS pos_avg_win_usd,
                CASE WHEN pos_losers > 0
                     THEN pos_sum_loss_abs / pos_losers ELSE NULL END AS pos_avg_loss_usd,
                CASE WHEN pos_sum_loss_abs > 0
                     THEN LEAST(pos_sum_win / pos_sum_loss_abs, 100.0)
                     ELSE NULL END AS pos_profit_factor,
                CASE WHEN pos_std_pnl > 0
                     THEN pos_mean_pnl / pos_std_pnl
                          * sqrt(n_closed_positions::DOUBLE
                                / (GREATEST(DATEDIFF('day', first_activity_ts,
                                                      last_activity_ts), 30) / 365.25))
                     ELSE NULL END AS pos_sharpe,
                CASE WHEN pos_downside_std > 0
                     THEN pos_mean_pnl / pos_downside_std
                          * sqrt(n_closed_positions::DOUBLE
                                / (GREATEST(DATEDIFF('day', first_activity_ts,
                                                      last_activity_ts), 30) / 365.25))
                     ELSE NULL END AS pos_sortino,
                -- Kelly: f = p - (1-p) * (avg_loss / avg_win)
                CASE WHEN pos_winners > 0 AND pos_losers > 0 AND pos_sum_win > 0
                     THEN (pos_winners::DOUBLE / (pos_winners + pos_losers))
                          - (pos_losers::DOUBLE / (pos_winners + pos_losers))
                            * ((pos_sum_loss_abs / pos_losers)
                               / (pos_sum_win / pos_winners))
                     ELSE NULL END AS pos_kelly_fraction,
                style_avg_holding_hours,
                style_median_holding_hours
            FROM base
        """)
        n = con.sql("SELECT count(*) FROM m_pos").fetchone()[0]
        print(f"  {n:,} addresses")

    # ---------- 2. Market-level PnL + NegRisk volume share ----------
    with step("2/8 market-level PnL (m_mkt)"):
        con.execute("""
            CREATE TABLE IF NOT EXISTS m_mkt AS
            WITH per_market AS (
                SELECT
                    address, market_id,
                    sum(realised_pnl) AS market_pnl,
                    any_value(resolution_ts) AS market_resolution_ts,
                    any_value(neg_risk) AS neg_risk,
                    sum(gross_usd_volume) AS market_volume
                FROM closed_positions
                GROUP BY address, market_id
            ),
            agg AS (
                SELECT
                    address,
                    count(*) AS mkt_n_markets_traded,
                    sum(market_pnl) AS mkt_total_pnl,
                    sum(CASE WHEN market_pnl > 0 THEN 1 ELSE 0 END) AS mkt_winners,
                    sum(CASE WHEN market_pnl < 0 THEN 1 ELSE 0 END) AS mkt_losers,
                    sum(CASE WHEN market_pnl > 0 THEN market_pnl ELSE 0 END)
                        AS mkt_sum_win,
                    sum(CASE WHEN market_pnl < 0 THEN -market_pnl ELSE 0 END)
                        AS mkt_sum_loss_abs,
                    avg(market_pnl) AS mkt_mean_pnl,
                    stddev_pop(market_pnl) AS mkt_std_pnl,
                    sqrt(avg(CASE WHEN market_pnl < 0
                                  THEN market_pnl * market_pnl ELSE 0 END))
                        AS mkt_downside_std,
                    min(market_resolution_ts) AS first_res_ts,
                    max(market_resolution_ts) AS last_res_ts,
                    sum(market_volume) AS total_mkt_volume,
                    sum(CASE WHEN neg_risk THEN market_volume ELSE 0 END)
                        AS negrisk_volume
                FROM per_market
                GROUP BY address
            )
            SELECT
                address,
                mkt_n_markets_traded,
                mkt_total_pnl,
                mkt_winners,
                mkt_losers,
                CASE WHEN mkt_winners + mkt_losers > 0
                     THEN mkt_winners::DOUBLE / (mkt_winners + mkt_losers)
                     ELSE NULL END AS mkt_win_rate,
                CASE WHEN mkt_sum_win + mkt_sum_loss_abs > 0
                     THEN mkt_sum_win / (mkt_sum_win + mkt_sum_loss_abs)
                     ELSE NULL END AS mkt_dollar_win_rate,
                CASE WHEN mkt_winners > 0
                     THEN mkt_sum_win / mkt_winners ELSE NULL END AS mkt_avg_win_usd,
                CASE WHEN mkt_losers > 0
                     THEN mkt_sum_loss_abs / mkt_losers ELSE NULL END AS mkt_avg_loss_usd,
                CASE WHEN mkt_sum_loss_abs > 0
                     THEN LEAST(mkt_sum_win / mkt_sum_loss_abs, 100.0)
                     ELSE NULL END AS mkt_profit_factor,
                CASE WHEN mkt_std_pnl > 0
                     THEN mkt_mean_pnl / mkt_std_pnl
                          * sqrt(mkt_n_markets_traded::DOUBLE
                                / (GREATEST(DATEDIFF('day', first_res_ts,
                                                      last_res_ts), 30) / 365.25))
                     ELSE NULL END AS mkt_sharpe,
                CASE WHEN mkt_downside_std > 0
                     THEN mkt_mean_pnl / mkt_downside_std
                          * sqrt(mkt_n_markets_traded::DOUBLE
                                / (GREATEST(DATEDIFF('day', first_res_ts,
                                                      last_res_ts), 30) / 365.25))
                     ELSE NULL END AS mkt_sortino,
                CASE WHEN mkt_winners > 0 AND mkt_losers > 0 AND mkt_sum_win > 0
                     THEN (mkt_winners::DOUBLE / (mkt_winners + mkt_losers))
                          - (mkt_losers::DOUBLE / (mkt_winners + mkt_losers))
                            * ((mkt_sum_loss_abs / mkt_losers)
                               / (mkt_sum_win / mkt_winners))
                     ELSE NULL END AS mkt_kelly_fraction,
                CASE WHEN total_mkt_volume > 0
                     THEN negrisk_volume / total_mkt_volume
                     ELSE NULL END AS negrisk_volume_share
            FROM agg
        """)

    # ---------- 3. phantom_position_score ----------
    with step("3/8 phantom_position_score (m_phantom)"):
        # Per (address, market): ratio = sum(|pnl|) / |sum(pnl)|
        # Normal directional traders ~1.0; NegRisk arb >> 1.0.
        # Aggregated per address as volume-weighted average over markets.
        con.execute("""
            CREATE TABLE IF NOT EXISTS m_phantom AS
            WITH per_market AS (
                SELECT
                    address, market_id,
                    sum(realised_pnl) AS market_pnl,
                    sum(abs(realised_pnl)) AS market_abs_pnl,
                    sum(gross_usd_volume) AS market_volume
                FROM closed_positions
                GROUP BY address, market_id
            ),
            scored AS (
                SELECT *,
                    CASE WHEN abs(market_pnl) > 0.01
                         THEN LEAST(market_abs_pnl / abs(market_pnl), 1000.0)
                         ELSE NULL END AS phantom_ratio
                FROM per_market
            )
            SELECT
                address,
                CASE WHEN sum(CASE WHEN phantom_ratio IS NOT NULL THEN market_volume ELSE 0 END) > 0
                     THEN sum(CASE WHEN phantom_ratio IS NOT NULL THEN phantom_ratio * market_volume ELSE 0 END)
                          / sum(CASE WHEN phantom_ratio IS NOT NULL THEN market_volume ELSE 0 END)
                     ELSE NULL END AS phantom_position_score
            FROM scored
            GROUP BY address
        """)

    # ---------- 4. Position-level drawdown (window) ----------
    with step("4/8 position-level drawdown (m_dd_pos)"):
        con.execute("""
            CREATE TABLE IF NOT EXISTS m_dd_pos AS
            WITH ordered AS (
                SELECT address, resolution_ts, realised_pnl,
                    SUM(realised_pnl) OVER (
                        PARTITION BY address ORDER BY resolution_ts
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS cum_pnl
                FROM closed_positions
            ),
            with_peak AS (
                SELECT address, resolution_ts, cum_pnl,
                    MAX(cum_pnl) OVER (
                        PARTITION BY address ORDER BY resolution_ts
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS running_peak
                FROM ordered
            )
            SELECT address, MAX(running_peak - cum_pnl) AS pos_max_drawdown_usd
            FROM with_peak
            GROUP BY address
        """)

    # ---------- 5. Market-level drawdown (window) ----------
    with step("5/8 market-level drawdown (m_dd_mkt)"):
        con.execute("""
            CREATE TABLE IF NOT EXISTS m_dd_mkt AS
            WITH per_market AS (
                SELECT address, market_id,
                    sum(realised_pnl) AS market_pnl,
                    any_value(resolution_ts) AS market_resolution_ts
                FROM closed_positions
                GROUP BY address, market_id
            ),
            ordered AS (
                SELECT address, market_resolution_ts, market_pnl,
                    SUM(market_pnl) OVER (
                        PARTITION BY address ORDER BY market_resolution_ts
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS cum_pnl
                FROM per_market
            ),
            with_peak AS (
                SELECT address, cum_pnl,
                    MAX(cum_pnl) OVER (
                        PARTITION BY address ORDER BY market_resolution_ts
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS running_peak
                FROM ordered
            )
            SELECT address, MAX(running_peak - cum_pnl) AS mkt_max_drawdown_usd
            FROM with_peak
            GROUP BY address
        """)

    # ---------- 6a. Style summary (per-address fill stats) ----------
    # Same CROSS JOIN sides pattern that works for m_subsec/m_cps. Removed
    # approx_quantile and count(*) FILTER — the prior version silently
    # exited at the start of step 6a (no traceback, no spill). Best guess:
    # approx_quantile's per-group t-digest state interacted poorly with
    # the 2.6 M-group hash table. style_median_fill_size_usd ships as NULL
    # for v1.
    with step("6a/9 style — per-address summary (m_style_summary)"):
        con.execute("""
            CREATE TABLE IF NOT EXISTS m_style_summary AS
            SELECT
                CASE s.role WHEN 'maker' THEN jf.maker ELSE jf.taker END AS address,
                sum(CASE WHEN s.role = 'maker' THEN 1 ELSE 0 END) AS style_maker_fill_count,
                sum(CASE WHEN s.role = 'taker' THEN 1 ELSE 0 END) AS style_taker_fill_count,
                avg(jf.usd_amount) AS style_avg_fill_size_usd,
                max(jf.usd_amount) AS style_max_fill_size_usd,
                CAST(NULL AS DOUBLE) AS style_median_fill_size_usd
            FROM joined_fills jf
            CROSS JOIN (VALUES ('maker'), ('taker')) AS s(role)
            GROUP BY 1
        """)

    # ---------- 6b. Style buy/sell symmetry (per-market then per-address) ----------
    # Independent of 6a; scans joined_fills again. Cheaper than re-using a CTE
    # because DuckDB would materialise the 2 B-row CTE.
    with step("6b/9 style — buy/sell symmetry (m_style_symmetry)"):
        con.execute("""
            CREATE TABLE IF NOT EXISTS m_style_symmetry AS
            WITH per_market AS (
                SELECT
                    CASE s.role WHEN 'maker' THEN jf.maker ELSE jf.taker END AS address,
                    jf.market_id,
                    sum(CASE WHEN (s.role = 'maker' AND jf.maker_asset_id = '0')
                              OR (s.role = 'taker' AND jf.maker_asset_id <> '0')
                             THEN jf.usd_amount ELSE 0 END) AS buy_usd,
                    sum(CASE WHEN (s.role = 'maker' AND jf.maker_asset_id <> '0')
                              OR (s.role = 'taker' AND jf.maker_asset_id = '0')
                             THEN jf.usd_amount ELSE 0 END) AS sell_usd
                FROM joined_fills jf
                CROSS JOIN (VALUES ('maker'), ('taker')) AS s(role)
                GROUP BY 1, 2
            )
            SELECT address,
                avg(CASE WHEN buy_usd + sell_usd > 0
                         THEN abs(buy_usd - sell_usd) / (buy_usd + sell_usd)
                         ELSE NULL END) AS style_buy_sell_symmetry
            FROM per_market
            GROUP BY address
        """)

    # ---------- 7. Sub-second clustering (1-second-bucket approximation) ----------
    with step("7/8 sub-second clustering (m_subsec)"):
        con.execute("""
            CREATE TABLE IF NOT EXISTS m_subsec AS
            WITH per_addr_sec AS (
                SELECT
                    CASE s.role WHEN 'maker' THEN jf.maker ELSE jf.taker END
                        AS address,
                    date_trunc('second', jf.timestamp) AS sec,
                    count(*) AS n_in_sec
                FROM joined_fills jf
                CROSS JOIN (VALUES ('maker'), ('taker')) AS s(role)
                GROUP BY 1, 2
            )
            SELECT address,
                100.0 * sum(CASE WHEN n_in_sec > 1 THEN n_in_sec ELSE 0 END)::DOUBLE
                       / nullif(sum(n_in_sec), 0) AS style_pct_sub_second
            FROM per_addr_sec
            GROUP BY address
        """)

    # ---------- 8. Distinct counterparties (HyperLogLog approx) ----------
    with step("8/8 distinct counterparties (m_cps, approx)"):
        con.execute("""
            CREATE TABLE IF NOT EXISTS m_cps AS
            SELECT
                CASE s.role WHEN 'maker' THEN jf.maker ELSE jf.taker END AS address,
                approx_count_distinct(
                    CASE s.role WHEN 'maker' THEN jf.taker ELSE jf.maker END
                ) AS n_distinct_counterparties
            FROM joined_fills jf
            CROSS JOIN (VALUES ('maker'), ('taker')) AS s(role)
            GROUP BY 1
        """)

    # ---------- 8.5 Approximate bankroll (peak concurrent deployed capital) ----------
    # Per address, build event series of capital-deployed deltas:
    #   +total_bought_usd at first_fill_ts (position opens)
    #   -total_bought_usd at resolution_ts (position closes)
    # Running cumsum gives concurrent_deployed at each event.
    # Per address, MAX(concurrent_deployed) ≈ peak bankroll.
    #
    # The "30d rolling max" framing in the spec collapses to MAX(concurrent_deployed)
    # for a single summary number per trader (rolling-max of any window containing
    # the global peak equals the global peak). We document the approximation as
    # "peak deployed at entry value, not mark-to-market". For a smoother estimate
    # — e.g. average over 30-day windows — Phase 4 can refine.
    with step("8.5/9 approximate bankroll (m_bankroll)"):
        con.execute("""
            CREATE TABLE IF NOT EXISTS m_bankroll AS
            WITH events AS (
                SELECT address, first_fill_ts AS ts, total_bought_usd AS delta
                FROM closed_positions
                WHERE total_bought_usd IS NOT NULL AND total_bought_usd > 0
                UNION ALL
                SELECT address, resolution_ts, -total_bought_usd
                FROM closed_positions
                WHERE total_bought_usd IS NOT NULL AND total_bought_usd > 0
                  AND resolution_ts IS NOT NULL
                  AND resolution_ts >= first_fill_ts  -- skip placeholder pre-fill end_dates
            ),
            with_cum AS (
                SELECT address,
                    SUM(delta) OVER (
                        PARTITION BY address ORDER BY ts
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS deployed
                FROM events
            )
            SELECT address, MAX(deployed) AS est_bankroll_usd_30d_max_approx
            FROM with_cum
            GROUP BY address
        """)

    # ---------- 9. Final join + is_operator_like + write parquet ----------
    operators_sql = ", ".join(f"'{a}'" for a in OPERATOR_ADDRESSES)
    with step("9/9 final join + write parquet"):
        con.execute(f"""
            CREATE OR REPLACE TABLE traders AS
            SELECT
                mp.*,
                mm.* EXCLUDE (address),
                mph.phantom_position_score,
                ms.* EXCLUDE (address),
                CASE WHEN ms.style_taker_fill_count > 0
                     THEN LEAST(ms.style_maker_fill_count::DOUBLE
                                / ms.style_taker_fill_count, 1000.0)
                     ELSE 1000.0 END AS style_maker_taker_ratio,
                CASE WHEN ms.style_maker_fill_count + ms.style_taker_fill_count > 0
                     THEN ms.style_maker_fill_count::DOUBLE
                          / (ms.style_maker_fill_count + ms.style_taker_fill_count)
                     ELSE NULL END AS style_role_balance,
                msym.style_buy_sell_symmetry,
                mssec.style_pct_sub_second,
                coalesce(mc.n_distinct_counterparties, 0) AS n_distinct_counterparties,
                mdp.pos_max_drawdown_usd,
                mdm.mkt_max_drawdown_usd,
                (
                    mp.address IN ({operators_sql})
                    OR (
                        ms.style_taker_fill_count > 0
                        AND ms.style_maker_fill_count::DOUBLE / ms.style_taker_fill_count > 50.0
                    )
                    OR (
                        ms.style_maker_fill_count > 0 AND ms.style_taker_fill_count > 0
                        AND ms.style_maker_fill_count::DOUBLE / ms.style_taker_fill_count < 0.02
                    )
                    OR coalesce(mc.n_distinct_counterparties, 0) > 500000
                    OR (
                        coalesce(mssec.style_pct_sub_second, 0.0) > 95.0
                        AND mp.n_fills_total > 1000000
                    )
                ) AS is_operator_like,
                mb.est_bankroll_usd_30d_max_approx
            FROM m_pos mp
            LEFT JOIN m_mkt mm USING (address)
            LEFT JOIN m_phantom mph USING (address)
            LEFT JOIN m_style_summary ms USING (address)
            LEFT JOIN m_style_symmetry msym USING (address)
            LEFT JOIN m_subsec mssec USING (address)
            LEFT JOIN m_cps mc USING (address)
            LEFT JOIN m_dd_pos mdp USING (address)
            LEFT JOIN m_dd_mkt mdm USING (address)
            LEFT JOIN m_bankroll mb USING (address)
        """)
        con.execute(
            f"COPY traders TO '{OUT_PATH}' "
            f"(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        n = con.sql("SELECT count(*) FROM traders").fetchone()[0]
        size_gb = OUT_PATH.stat().st_size / 1e9
        print(f"  written: {OUT_PATH.name} — {n:,} rows, {size_gb:.2f} GB")

    sanity_checks(con)


def sanity_checks(con) -> None:
    DOMAH = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"
    print("\n\n========================== SANITY CHECKS ==========================")

    # (a) Domah row inspection
    print("\n=== (a) domah row ===")
    df = con.sql(f"SELECT * FROM traders WHERE address = '{DOMAH}'").fetchdf()
    if len(df) == 0:
        print("  WARN: domah not present in traders table")
    else:
        for col in df.columns:
            v = df[col].iloc[0]
            print(f"  {col:35s} {v}")

    # (b) Top 20 by mkt_total_pnl, filtered out operators
    print("\n=== (b) top 20 by mkt_total_pnl (excluding is_operator_like) ===")
    print(con.sql("""
        SELECT
            substr(address, 1, 14) || '...' AS addr,
            n_closed_positions AS n_pos,
            round(mkt_total_pnl, 0) AS mkt_pnl,
            round(mkt_win_rate, 3) AS win,
            round(mkt_profit_factor, 2) AS pf,
            round(mkt_sharpe, 2) AS sharpe,
            round(phantom_position_score, 2) AS phantom,
            round(negrisk_volume_share, 2) AS nr_share,
            round(style_role_balance, 2) AS role_bal
        FROM traders
        WHERE NOT is_operator_like AND n_closed_positions >= 10
        ORDER BY mkt_total_pnl DESC NULLS LAST
        LIMIT 20
    """).fetchdf().to_string(index=False))

    # (c) n_closed_positions percentiles
    print("\n=== (c) n_closed_positions percentile distribution ===")
    print(con.sql("""
        SELECT
            count(*) AS total_traders,
            quantile_cont(n_closed_positions, 0.10) AS p10,
            quantile_cont(n_closed_positions, 0.25) AS p25,
            quantile_cont(n_closed_positions, 0.50) AS p50,
            quantile_cont(n_closed_positions, 0.75) AS p75,
            quantile_cont(n_closed_positions, 0.90) AS p90,
            quantile_cont(n_closed_positions, 0.95) AS p95,
            quantile_cont(n_closed_positions, 0.99) AS p99,
            max(n_closed_positions) AS max_n
        FROM traders
    """).fetchdf().to_string(index=False))

    # (d) NegRisk-heavy cohort
    print("\n=== (d) NegRisk cohort (negrisk_volume_share > 0.5, n_closed_positions > 100) ===")
    print(con.sql("""
        SELECT
            substr(address, 1, 14) || '...' AS addr,
            n_closed_positions AS n_pos,
            round(negrisk_volume_share, 2) AS nr_share,
            round(mkt_total_pnl, 0) AS mkt_pnl,
            round(phantom_position_score, 2) AS phantom,
            round(mkt_win_rate, 3) AS win,
            round(pos_total_pnl, 0) AS pos_pnl
        FROM traders
        WHERE NOT is_operator_like
          AND negrisk_volume_share > 0.5
          AND n_closed_positions > 100
        ORDER BY mkt_total_pnl DESC NULLS LAST
        LIMIT 10
    """).fetchdf().to_string(index=False))

    # (e) Operator deny-list validation
    print("\n=== (e) operator deny-list — computed metrics ===")
    print(con.sql("""
        SELECT
            substr(address, 1, 14) || '...' AS addr,
            n_closed_positions AS n_pos,
            n_fills_total AS n_fills,
            round(style_maker_taker_ratio, 2) AS mt_ratio,
            n_distinct_counterparties AS n_cps,
            round(style_pct_sub_second, 1) AS pct_subsec,
            round(pos_total_pnl, 0) AS pos_pnl,
            round(pos_sharpe, 2) AS pos_sharpe,
            is_operator_like AS flagged
        FROM traders
        WHERE is_operator_like
        ORDER BY n_fills DESC
        LIMIT 30
    """).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
