"""Build data/closed_positions.parquet — Phase 2 / Layer B.

For each (address, market_id, outcome_index) on resolved markets:
  - Aggregate trader actions (after same-bucket collapse)
  - Synthesise redemption at resolution price
  - Compute realised_pnl

Run from polymarket/research/ as:
    PYTHONPATH=. uv run python scripts/build_closed_positions.py

Design note — predicate-pushdown and the UNION-ALL inside trader_actions:
  We do NOT JOIN closed-markets onto trader_actions, because DuckDB doesn't
  push WHERE filters cleanly through the UNION ALL inside trader_actions
  (smoke-test runs proved this — a SEMI JOIN with IN-subquery materialised
  the full 2 B-row exploded table to 178 GB temp).
  Instead, we filter at joined_fills (1 B rows, no union), THEN explode in
  a CTE here. Same logical result, smaller intermediates.
"""
import time
from pathlib import Path

from data_infra.duck import connect
from data_infra.views import load_views

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "closed_positions.parquet"


BUILD_SQL = """
COPY (
    WITH actions AS (
        -- Single pass over joined_fills x sides(maker, taker): each fill yields
        -- exactly 2 rows. The CROSS JOIN with VALUES sidesteps the
        -- UNION-ALL-with-CTE pattern that caused DuckDB to materialise
        -- a 200+ GB intermediate (CTE referenced twice → automatic
        -- materialisation). Filter to closed markets via mt.closed = TRUE.
        SELECT
            jf.timestamp,
            CASE s.role WHEN 'maker' THEN jf.maker ELSE jf.taker END AS address,
            jf.market_id, jf.condition_id, jf.neg_risk,
            jf.outcome_token_id, jf.outcome_index,
            CASE
                WHEN s.role = 'maker' AND jf.maker_asset_id = '0' THEN  jf.token_amount
                WHEN s.role = 'maker'                            THEN -jf.token_amount
                WHEN s.role = 'taker' AND jf.maker_asset_id = '0' THEN -jf.token_amount
                ELSE                                                   jf.token_amount
            END AS token_delta,
            CASE
                WHEN s.role = 'maker' AND jf.maker_asset_id = '0' THEN -jf.usd_amount
                WHEN s.role = 'maker'                            THEN  jf.usd_amount
                WHEN s.role = 'taker' AND jf.maker_asset_id = '0' THEN  jf.usd_amount
                ELSE                                                  -jf.usd_amount
            END AS usd_delta,
            jf.token_amount, jf.usd_amount, jf.transaction_hash
        FROM joined_fills jf
        JOIN markets_tokens mt
          ON mt.market_id = jf.market_id AND mt.outcome_index = jf.outcome_index
        CROSS JOIN (VALUES ('maker'), ('taker')) AS s(role)
        WHERE mt.closed = TRUE
    ),
    -- NOTE: same-bucket collapse and peak_position_size are deferred for v1.
    -- The collapse step's hash table (keyed on tx_hash + ts + address etc.)
    -- has ~1.4 B unique groups → 300 GB OOM. We skip it: aggregating
    -- directly into (address, market, outcome) gives identical numbers for
    -- final_token_position / realised_cash_flow / etc. The only diff is
    -- peak_fill_abs_token (per-fill max) is a tighter lower bound than the
    -- per-bucket-collapsed max would be. Phase 3 can revisit if needed.
    agg AS (
        SELECT
            address, market_id, condition_id, neg_risk,
            outcome_token_id, outcome_index,
            SUM(token_delta) AS final_token_position,
            SUM(usd_delta) AS realised_cash_flow,
            SUM(token_amount) AS gross_token_volume,
            SUM(usd_amount) AS gross_usd_volume,
            COUNT(*) AS n_fills,
            MIN(timestamp) AS first_fill_ts,
            MAX(timestamp) AS last_fill_ts,
            SUM(CASE WHEN token_delta > 0 THEN usd_amount ELSE 0 END) AS total_bought_usd,
            SUM(CASE WHEN token_delta < 0 THEN usd_amount ELSE 0 END) AS total_sold_usd,
            MAX(abs(token_delta)) AS peak_fill_abs_token  -- v1 proxy
        FROM actions
        GROUP BY address, market_id, condition_id, neg_risk, outcome_token_id, outcome_index
    )
    SELECT
        agg.address,
        agg.market_id,
        agg.condition_id,
        agg.neg_risk,
        agg.outcome_token_id,
        agg.outcome_index,
        agg.n_fills,
        agg.first_fill_ts,
        agg.last_fill_ts,
        mt.end_date AS resolution_ts,
        CASE WHEN mt.end_date IS NOT NULL
             THEN epoch(mt.end_date - agg.first_fill_ts) / 3600.0
             ELSE NULL END AS holding_duration_hours,
        agg.gross_token_volume,
        agg.gross_usd_volume,
        agg.total_bought_usd,
        agg.total_sold_usd,
        agg.final_token_position,
        agg.realised_cash_flow,
        CAST(mt.outcome_prices[agg.outcome_index] AS DOUBLE) AS resolution_price,
        agg.final_token_position * CAST(mt.outcome_prices[agg.outcome_index] AS DOUBLE)
            AS redemption_value,
        agg.realised_cash_flow + agg.final_token_position
            * CAST(mt.outcome_prices[agg.outcome_index] AS DOUBLE) AS realised_pnl,
        agg.peak_fill_abs_token,
        abs(agg.final_token_position) > 1e-6 AS is_held_to_resolution
    FROM agg
    JOIN markets_tokens mt
      ON mt.market_id = agg.market_id AND mt.outcome_index = agg.outcome_index
)
TO '{OUT}' (FORMAT PARQUET, COMPRESSION ZSTD)
"""


def fmt_summary(con) -> None:
    print("\n=== summary ===")
    s = con.sql("""
        SELECT
            count(*) AS rows,
            count(DISTINCT address) AS n_addresses,
            count(DISTINCT market_id) AS n_markets,
            sum(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END) AS winners,
            sum(CASE WHEN realised_pnl < 0 THEN 1 ELSE 0 END) AS losers,
            sum(CASE WHEN abs(realised_pnl) < 1e-6 THEN 1 ELSE 0 END) AS flat,
            round(min(realised_pnl), 2) AS min_pnl,
            round(max(realised_pnl), 2) AS max_pnl,
            round(sum(realised_pnl), 0) AS total_pnl,
            round(median(realised_pnl), 4) AS median_pnl,
            round(avg(realised_pnl), 4) AS mean_pnl,
            sum(CASE WHEN is_held_to_resolution THEN 1 ELSE 0 END) AS held_to_res,
            sum(CASE WHEN NOT is_held_to_resolution THEN 1 ELSE 0 END) AS exited_via_trades
        FROM closed_positions
    """).fetchone()
    keys = [
        "rows", "n_addresses", "n_markets", "winners", "losers", "flat",
        "min_pnl", "max_pnl", "total_pnl", "median_pnl", "mean_pnl",
        "held_to_res", "exited_via_trades",
    ]
    for k, v in zip(keys, s):
        print(f"  {k:25s} {v}")

    print("\n=== markets with NULL or pre-Polymarket end_date (flag) ===")
    flag = con.sql("""
        SELECT
            sum(CASE WHEN resolution_ts IS NULL THEN 1 ELSE 0 END) AS null_resolution_ts,
            sum(CASE WHEN resolution_ts < TIMESTAMP '2020-01-01' THEN 1 ELSE 0 END) AS pre_2020_resolution_ts,
            sum(CASE WHEN holding_duration_hours < 0 THEN 1 ELSE 0 END) AS negative_duration
        FROM closed_positions
    """).fetchone()
    print(f"  null resolution_ts        : {flag[0]}")
    print(f"  resolution_ts < 2020      : {flag[1]} (placeholder/test markets — Polymarket launched 2020-06)")
    print(f"  negative holding duration : {flag[2]} (resolved BEFORE first fill — same as above class)")


def sanity_checks(con) -> None:
    DOMAH = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"

    print("\n\n=========================== SANITY CHECKS ===========================")

    print(f"\n=== (a) domah closed-position PnL distribution ===")
    print(con.sql(f"""
        SELECT
            count(*) AS n_positions,
            round(sum(realised_pnl), 2) AS total_pnl,
            sum(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END) AS winners,
            sum(CASE WHEN realised_pnl < 0 THEN 1 ELSE 0 END) AS losers,
            round(max(realised_pnl), 2) AS biggest_win,
            round(min(realised_pnl), 2) AS biggest_loss,
            round(median(realised_pnl), 2) AS median_pnl
        FROM closed_positions WHERE address = '{DOMAH}'
    """).fetchdf().to_string(index=False))

    print(f"\n  domah top 5 winning positions:")
    print(con.sql(f"""
        SELECT market_id, outcome_index,
               round(realised_pnl, 0) AS pnl,
               round(total_bought_usd, 0) AS bought,
               round(final_token_position, 0) AS final_pos,
               round(resolution_price, 4) AS res_px
        FROM closed_positions WHERE address = '{DOMAH}'
        ORDER BY realised_pnl DESC LIMIT 5
    """).fetchdf().to_string(index=False))

    print(f"\n  domah top 5 losing positions:")
    print(con.sql(f"""
        SELECT market_id, outcome_index,
               round(realised_pnl, 0) AS pnl,
               round(total_bought_usd, 0) AS bought,
               round(final_token_position, 0) AS final_pos,
               round(resolution_price, 4) AS res_px
        FROM closed_positions WHERE address = '{DOMAH}'
        ORDER BY realised_pnl ASC LIMIT 5
    """).fetchdf().to_string(index=False))

    print("\n=== (b) PnL distribution: NegRisk vs regular ===")
    print(con.sql("""
        SELECT neg_risk,
               count(*) AS n,
               round(sum(realised_pnl), 0) AS total_pnl,
               round(median(realised_pnl), 4) AS median,
               round(avg(realised_pnl), 4) AS mean,
               round(min(realised_pnl), 2) AS min_pnl,
               round(max(realised_pnl), 2) AS max_pnl,
               round(stddev_pop(realised_pnl), 4) AS stddev
        FROM closed_positions
        GROUP BY neg_risk
    """).fetchdf().to_string(index=False))

    print("\n=== (c) 5 random closed positions — drill-down ===")
    sample = con.sql("""
        SELECT address, market_id, outcome_index, n_fills,
               round(realised_pnl, 2) AS pnl,
               round(final_token_position, 2) AS final_pos,
               round(realised_cash_flow, 2) AS cash_flow,
               round(redemption_value, 2) AS redemption,
               round(resolution_price, 4) AS res_px,
               is_held_to_resolution AS held
        FROM closed_positions
        WHERE n_fills BETWEEN 2 AND 20  -- pick small enough to show all
        USING SAMPLE 5 ROWS
    """).fetchdf()
    print(sample.to_string(index=False))

    for _, r in sample.iterrows():
        a, mid, oi = r["address"], r["market_id"], r["outcome_index"]
        print(f"\n--- drill: address={a[:10]}… market={mid} outcome_index={oi} ---")
        df = con.sql(f"""
            SELECT timestamp,
                   round(token_delta, 2) AS tok_d,
                   round(usd_delta, 2) AS usd_d,
                   round(token_amount, 2) AS tok_amt,
                   round(price, 4) AS price,
                   substr(transaction_hash, 1, 12) AS tx
            FROM trader_actions
            WHERE address = '{a}' AND market_id = '{mid}' AND outcome_index = {oi}
            ORDER BY timestamp
        """).fetchdf()
        print(df.to_string(index=False))
        print(f"  Σ token_delta = {df['tok_d'].sum():.2f}  (residual)")
        print(f"  Σ usd_delta   = {df['usd_d'].sum():.2f}  (cash flow)")
        print(f"  + redemption  = {r['redemption']:.2f}  (residual × {r['res_px']})")
        print(f"  realised_pnl  = {r['pnl']:.2f}")

    print("\n=== (d) self-consistency: SUM(realised_pnl) over all positions ===")
    print(con.sql("""
        SELECT round(sum(realised_pnl), 0) AS total_realised,
               round(sum(CASE WHEN neg_risk THEN realised_pnl ELSE 0 END), 0) AS negrisk_pnl,
               round(sum(CASE WHEN NOT neg_risk THEN realised_pnl ELSE 0 END), 0) AS regular_pnl,
               count(*) AS total_positions
        FROM closed_positions
    """).fetchdf().to_string(index=False))

    print("\n  (Expected ~0 if zero-sum across closed markets, modulo:")
    print("   - operator/MM addresses extracting fees/spread")
    print("   - open positions excluded (filter is closed=TRUE)")
    print("   - orphan markets excluded (~0.4 % of fills)")
    print("   - NegRisk arb (split/merge) not modelled — phantom PnL possible)")


def main() -> None:
    print(f"output: {OUT_PATH}")
    con = connect()
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")

    print("loading views.sql...")
    t0 = time.time()
    load_views(con)
    print(f"  loaded in {time.time() - t0:.2f}s")

    print("\nbuilding closed_positions parquet (this scans 1 B fills)...")
    t0 = time.time()
    con.execute(BUILD_SQL.format(OUT=OUT_PATH))
    print(f"  build elapsed: {time.time() - t0:.1f}s")
    print(f"  written: {OUT_PATH.stat().st_size / 1e9:.2f} GB")

    # Re-attach the output as a view for summary + sanity queries.
    con.execute(f"CREATE OR REPLACE VIEW closed_positions AS SELECT * FROM read_parquet('{OUT_PATH}')")

    fmt_summary(con)
    sanity_checks(con)


if __name__ == "__main__":
    main()
