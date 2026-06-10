"""Ascot 2025 Polymarket fill spike analysis.

Builds a compact first-pass dataset for Royal Ascot horse-racing markets:

- one row per candidate market
- all normalized fills for those markets
- winner-market threshold timestamps
- per-second bars around the winner's first 99% print

Timestamps are UTC. Ascot local time in June 2025 was BST (UTC+1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "analysis" / "csv_outputs" / "ascot_2025"
MARKETS_PATH = ROOT / "data" / "markets" / "markets_2026-05-06.parquet"
TRADES_SEED_PATH = ROOT / "data" / "trades" / "trades_seed.parquet"

ASCOT_RACE_SLUGS = (
    "queen-anne-stakes",
    "prince-of-wales-stakes",
    "ascot-gold-cup",
    "queen-elizabeth-ii-jubilee-stakes",
)


def sql_list(values: tuple[str, ...]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-before-seconds", type=int, default=600)
    parser.add_argument("--window-after-seconds", type=int, default=1800)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    race_slugs = sql_list(ASCOT_RACE_SLUGS)
    con = duckdb.connect()

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ascot_markets AS
        SELECT
            id::VARCHAR AS market_id,
            slug,
            question,
            try_cast(end_date AS TIMESTAMP) AS gamma_end_date_utc,
            volume AS gamma_volume,
            outcomes,
            outcome_prices,
            clob_token_ids,
            regexp_extract(slug, '^will-(.*)-win-the-2025-(.*)$', 2) AS race_slug,
            regexp_extract(slug, '^will-(.*)-win-the-2025-(.*)$', 1) AS runner_slug,
            clob_token_ids[1] AS yes_token_id,
            clob_token_ids[2] AS no_token_id,
            outcome_prices[1]::DOUBLE AS final_yes_price,
            outcome_prices[2]::DOUBLE AS final_no_price,
            outcome_prices[1]::DOUBLE >= 0.99 AS is_winner
        FROM read_parquet('{MARKETS_PATH}')
        WHERE regexp_extract(slug, '^will-(.*)-win-the-2025-(.*)$', 2) IN ({race_slugs});
        """
    )

    outcome_token_expr = (
        "CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END"
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ascot_fills AS
        SELECT
            a.race_slug,
            a.runner_slug,
            a.market_id,
            a.question,
            a.gamma_end_date_utc,
            a.is_winner,
            a.final_yes_price,
            t.timestamp AS fill_ts_utc,
            t.usd_amount,
            t.token_amount,
            t.price AS traded_token_price,
            t.maker_side,
            {outcome_token_expr} AS outcome_token_id,
            CASE
                WHEN {outcome_token_expr} = a.yes_token_id THEN 'YES'
                WHEN {outcome_token_expr} = a.no_token_id THEN 'NO'
                ELSE 'UNKNOWN'
            END AS traded_outcome,
            CASE
                WHEN {outcome_token_expr} = a.yes_token_id THEN t.price
                WHEN {outcome_token_expr} = a.no_token_id THEN 1 - t.price
                ELSE NULL
            END AS implied_yes_price,
            CASE
                WHEN a.is_winner AND {outcome_token_expr} = a.yes_token_id THEN t.price
                WHEN a.is_winner AND {outcome_token_expr} = a.no_token_id THEN 1 - t.price
                WHEN NOT a.is_winner AND {outcome_token_expr} = a.no_token_id THEN t.price
                WHEN NOT a.is_winner AND {outcome_token_expr} = a.yes_token_id THEN 1 - t.price
                ELSE NULL
            END AS implied_resolved_side_price,
            t.maker,
            t.taker,
            t.transaction_hash
        FROM ascot_markets a
        JOIN read_parquet('{TRADES_SEED_PATH}') t
          ON t.market_id = a.market_id;
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT
                race_slug,
                runner_slug,
                market_id,
                question,
                gamma_end_date_utc,
                gamma_volume,
                final_yes_price,
                final_no_price,
                is_winner,
                outcomes,
                outcome_prices,
                clob_token_ids
            FROM ascot_markets
            ORDER BY gamma_end_date_utc, race_slug, gamma_volume DESC NULLS LAST
        )
        TO '{args.output_dir / "ascot_2025_market_universe.csv"}'
        (HEADER, DELIMITER ',');
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM ascot_fills
            ORDER BY gamma_end_date_utc, race_slug, fill_ts_utc, market_id
        )
        TO '{args.output_dir / "ascot_2025_all_normalized_fills.csv"}'
        (HEADER, DELIMITER ',');
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE winner_crossings AS
        SELECT
            race_slug,
            runner_slug AS winner,
            market_id,
            gamma_end_date_utc,
            count(*) AS n_fills_total,
            round(sum(usd_amount), 2) AS usd_volume_total,
            min(fill_ts_utc) AS first_fill_ts_utc,
            max(fill_ts_utc) AS last_fill_ts_utc,
            min(fill_ts_utc) FILTER (WHERE implied_yes_price >= 0.90) AS first_90_ts_utc,
            min(fill_ts_utc) FILTER (WHERE implied_yes_price >= 0.95) AS first_95_ts_utc,
            min(fill_ts_utc) FILTER (WHERE implied_yes_price >= 0.98) AS first_98_ts_utc,
            min(fill_ts_utc) FILTER (WHERE implied_yes_price >= 0.99) AS first_99_ts_utc,
            max(implied_yes_price) AS max_yes_price
        FROM ascot_fills
        WHERE is_winner
        GROUP BY ALL;
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT
                c.*,
                prev.fill_ts_utc AS prev_fill_before_99_ts_utc,
                prev.implied_yes_price AS prev_yes_price,
                first99.implied_yes_price AS first_99_yes_price,
                date_diff('second', prev.fill_ts_utc, c.first_99_ts_utc)
                    AS seconds_from_prev_fill_to_99,
                (
                    SELECT count()
                    FROM ascot_fills f
                    WHERE f.market_id = c.market_id
                      AND f.fill_ts_utc >= c.first_99_ts_utc
                      AND f.fill_ts_utc < c.first_99_ts_utc + INTERVAL 5 SECOND
                ) AS fills_0_5s,
                (
                    SELECT round(sum(usd_amount), 2)
                    FROM ascot_fills f
                    WHERE f.market_id = c.market_id
                      AND f.fill_ts_utc >= c.first_99_ts_utc
                      AND f.fill_ts_utc < c.first_99_ts_utc + INTERVAL 5 SECOND
                ) AS usd_0_5s,
                (
                    SELECT count()
                    FROM ascot_fills f
                    WHERE f.market_id = c.market_id
                      AND f.fill_ts_utc >= c.first_99_ts_utc
                      AND f.fill_ts_utc < c.first_99_ts_utc + INTERVAL 30 SECOND
                ) AS fills_0_30s,
                (
                    SELECT round(sum(usd_amount), 2)
                    FROM ascot_fills f
                    WHERE f.market_id = c.market_id
                      AND f.fill_ts_utc >= c.first_99_ts_utc
                      AND f.fill_ts_utc < c.first_99_ts_utc + INTERVAL 30 SECOND
                ) AS usd_0_30s,
                (
                    SELECT count()
                    FROM ascot_fills f
                    WHERE f.market_id = c.market_id
                      AND f.fill_ts_utc >= c.first_99_ts_utc
                      AND f.fill_ts_utc < c.first_99_ts_utc + INTERVAL 60 SECOND
                ) AS fills_0_60s,
                (
                    SELECT round(sum(usd_amount), 2)
                    FROM ascot_fills f
                    WHERE f.market_id = c.market_id
                      AND f.fill_ts_utc >= c.first_99_ts_utc
                      AND f.fill_ts_utc < c.first_99_ts_utc + INTERVAL 60 SECOND
                ) AS usd_0_60s
            FROM winner_crossings c
            LEFT JOIN LATERAL (
                SELECT fill_ts_utc, implied_yes_price
                FROM ascot_fills f
                WHERE f.market_id = c.market_id
                  AND f.fill_ts_utc < c.first_99_ts_utc
                ORDER BY f.fill_ts_utc DESC
                LIMIT 1
            ) prev ON TRUE
            LEFT JOIN LATERAL (
                SELECT implied_yes_price
                FROM ascot_fills f
                WHERE f.market_id = c.market_id
                  AND f.fill_ts_utc = c.first_99_ts_utc
                ORDER BY f.implied_yes_price DESC
                LIMIT 1
            ) first99 ON TRUE
            ORDER BY gamma_end_date_utc, race_slug
        )
        TO '{args.output_dir / "ascot_2025_winner_spike_summary.csv"}'
        (HEADER, DELIMITER ',');
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT
                f.race_slug,
                f.runner_slug AS winner,
                f.market_id,
                c.first_99_ts_utc,
                date_diff('second', c.first_99_ts_utc, f.fill_ts_utc) AS rel_sec,
                f.fill_ts_utc,
                count(*) AS fills,
                round(sum(f.usd_amount), 2) AS usd,
                min(f.implied_yes_price) AS min_yes_price,
                max(f.implied_yes_price) AS max_yes_price,
                round(avg(f.implied_yes_price), 4) AS avg_yes_price
            FROM ascot_fills f
            JOIN winner_crossings c USING (market_id)
            WHERE f.is_winner
              AND f.fill_ts_utc >= c.first_99_ts_utc - INTERVAL {args.window_before_seconds} SECOND
              AND f.fill_ts_utc < c.first_99_ts_utc + INTERVAL {args.window_after_seconds} SECOND
            GROUP BY ALL
            ORDER BY c.first_99_ts_utc, rel_sec
        )
        TO '{args.output_dir / "ascot_2025_winner_second_bars_around_99.csv"}'
        (HEADER, DELIMITER ',');
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT
                f.race_slug,
                c.winner,
                f.gamma_end_date_utc,
                c.first_99_ts_utc,
                date_diff('second', c.first_99_ts_utc, f.fill_ts_utc) AS rel_sec,
                f.fill_ts_utc,
                count(*) AS fills,
                count(DISTINCT f.market_id) AS markets_traded,
                count(DISTINCT f.runner_slug) AS runners_traded,
                round(sum(f.usd_amount), 2) AS usd,
                min(f.implied_resolved_side_price) AS min_resolved_side_price,
                max(f.implied_resolved_side_price) AS max_resolved_side_price,
                round(avg(f.implied_resolved_side_price), 4) AS avg_resolved_side_price
            FROM ascot_fills f
            JOIN winner_crossings c USING (race_slug)
            WHERE f.fill_ts_utc >= c.first_99_ts_utc - INTERVAL {args.window_before_seconds} SECOND
              AND f.fill_ts_utc < c.first_99_ts_utc + INTERVAL {args.window_after_seconds} SECOND
            GROUP BY ALL
            ORDER BY c.first_99_ts_utc, rel_sec
        )
        TO '{args.output_dir / "ascot_2025_race_second_bars_around_winner_99.csv"}'
        (HEADER, DELIMITER ',');
        """
    )

    summary = con.execute(
        """
        SELECT
            race_slug,
            count(DISTINCT market_id) AS markets,
            count(*) AS fills,
            round(sum(usd_amount), 2) AS fill_usd,
            min(fill_ts_utc) AS first_fill_ts_utc,
            max(fill_ts_utc) AS last_fill_ts_utc
        FROM ascot_fills
        GROUP BY race_slug
        ORDER BY min(gamma_end_date_utc), race_slug;
        """
    ).fetchdf()
    print(summary.to_string(index=False))
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
