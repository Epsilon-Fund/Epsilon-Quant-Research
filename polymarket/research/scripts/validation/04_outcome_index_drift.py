"""Check 04: outcome-index drift.

For 50 random closed markets, verify that the clob_token_ids ordering
in the current Gamma snapshot is consistent with how the market actually
resolved. We do this by:

  - For each (market_id, outcome_token_id), get the LAST price observed
    in trades (or median of last 50 trades).
  - The token whose last price is closest to 1.0 → "implied resolved
    outcome" by trade prices.
  - Compare against outcome_prices in the markets parquet (the index
    where outcome_prices == 1.0 is the actual resolved outcome).

If the trade-price-implied resolution matches the markets-parquet
resolution, ordering is consistent for that market. Mismatches flag
markets for manual inspection (could be ordering drift, redemption-only
markets, or stale gamma data).
"""
from _common import connect


def main() -> None:
    con = connect()

    print("\n=== sampling 50 random closed markets that have trades ===")
    con.execute(
        """
        CREATE TEMP TABLE sample_markets AS
        SELECT m.id::VARCHAR AS market_id,
               m.condition_id,
               m.clob_token_ids,
               m.outcomes,
               m.outcome_prices
        FROM m
        WHERE m.closed = TRUE
          AND len(m.clob_token_ids) >= 2
          AND m.id::VARCHAR IN (
              SELECT DISTINCT market_id FROM t WHERE market_id IS NOT NULL
          )
        USING SAMPLE 50 ROWS
        """
    )
    print(f"sampled: {con.sql('SELECT count(*) FROM sample_markets').fetchone()[0]}")

    print("\n=== last trade price per (market_id, outcome_index) ===")
    df = con.sql(
        """
        WITH unnested AS (
            SELECT market_id, clob_token_ids, outcomes, outcome_prices,
                   list_position(clob_token_ids, tok) AS outcome_idx,
                   tok AS token_id
            FROM sample_markets, UNNEST(clob_token_ids) AS tu(tok)
        ),
        last_prices AS (
            SELECT u.market_id, u.outcome_idx, u.token_id,
                   any_value(u.outcomes) AS outcomes,
                   any_value(u.outcome_prices) AS outcome_prices,
                   median(t.price) FILTER (WHERE t.price IS NOT NULL) AS median_last_price
            FROM unnested u
            LEFT JOIN LATERAL (
                SELECT price FROM t
                WHERE t.market_id = u.market_id
                  AND CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                           ELSE t.maker_asset_id END = u.token_id
                ORDER BY timestamp DESC
                LIMIT 50
            ) t ON TRUE
            GROUP BY u.market_id, u.outcome_idx, u.token_id
        )
        SELECT * FROM last_prices ORDER BY market_id, outcome_idx
        """
    ).fetchdf()

    # Per market: which outcome_index has highest median_last_price?
    # Compare against which outcome_prices index == 1.0
    print("\n=== reconciliation per market ===")
    rows = []
    for mid, group in df.groupby("market_id"):
        group = group.sort_values("outcome_idx")
        median_prices = group["median_last_price"].tolist()
        outcomes = group["outcomes"].iloc[0] if len(group) else None
        outcome_prices_str = group["outcome_prices"].iloc[0] if len(group) else None

        # parse outcome_prices (stringified array of decimals)
        try:
            import ast
            op = ast.literal_eval(outcome_prices_str) if isinstance(outcome_prices_str, str) else outcome_prices_str
            op_floats = [float(x) for x in op]
        except Exception:
            op_floats = None

        # Index of "winner" by markets parquet (closest to 1.0)
        winner_by_markets = (max(range(len(op_floats)), key=lambda i: op_floats[i]) + 1) if op_floats else None
        # Index of "winner" by trade prices
        if median_prices and any(p is not None for p in median_prices):
            winner_by_trades = max(
                range(len(median_prices)),
                key=lambda i: (median_prices[i] if median_prices[i] is not None else -1)
            ) + 1
        else:
            winner_by_trades = None

        match = (winner_by_markets == winner_by_trades)
        outcomes_str = str(list(outcomes))[:40] if outcomes is not None else None
        rows.append({
            "market_id": mid,
            "outcomes": outcomes_str,
            "outcome_prices_top": op_floats[winner_by_markets - 1] if op_floats and winner_by_markets else None,
            "winner_by_markets": winner_by_markets,
            "winner_by_trades": winner_by_trades,
            "match": match,
            "median_prices": [round(p, 3) if p is not None else None for p in median_prices],
        })

    import pandas as pd
    summary = pd.DataFrame(rows)
    print(f"\nTotal markets sampled: {len(summary)}")
    print(f"Matches: {summary['match'].sum()} / {len(summary)}")
    print(f"Mismatches: {(~summary['match']).sum()}")

    print("\n=== mismatches (manual review needed) ===")
    mm = summary[~summary["match"]]
    if len(mm):
        print(mm.to_string(index=False))
    else:
        print("(none)")

    print("\n=== first 15 markets reconciled ===")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
