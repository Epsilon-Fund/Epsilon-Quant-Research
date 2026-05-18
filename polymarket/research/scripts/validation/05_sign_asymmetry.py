"""Check 05: sign asymmetry per (market_id, outcome_token_id).

Compare total token_amount on BUY-by-maker side vs SELL-by-maker side
for each outcome token. Mismatches reflect net direction of position-
taking (one-sided net flow); large outliers flag potential corruption
or unusual market activity.

In a fill where maker_side='BUY':  maker received tokens, taker delivered.
In a fill where maker_side='SELL': maker delivered tokens, taker received.

Sum of buy-side token_amount across all fills of token X = total tokens
that flowed toward MAKERS. Sum of sell-side = total tokens that flowed
AWAY from makers. Naively, if all positions opened eventually close, these
should balance. Imbalance = open positions or one-sided net direction.
"""
from _common import connect


def main() -> None:
    con = connect()

    print("\n=== overall buy/sell volume by maker_side ===")
    print(con.sql(
        """
        SELECT maker_side,
               count(*) AS n,
               round(sum(token_amount), 2) AS total_token_amount,
               round(sum(usd_amount), 2) AS total_usd
        FROM t
        GROUP BY maker_side
        """
    ).fetchdf().to_string(index=False))

    print("\n=== per-outcome-token imbalance (top 20 by absolute imbalance, USD) ===")
    print(con.sql(
        """
        WITH sides AS (
            SELECT
                CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END
                    AS outcome_token_id,
                market_id,
                maker_side,
                token_amount,
                usd_amount
            FROM t
            WHERE market_id IS NOT NULL
        ),
        agg AS (
            SELECT outcome_token_id, market_id,
                   sum(CASE WHEN maker_side = 'BUY' THEN token_amount ELSE 0 END)
                       AS buy_tokens,
                   sum(CASE WHEN maker_side = 'SELL' THEN token_amount ELSE 0 END)
                       AS sell_tokens,
                   sum(CASE WHEN maker_side = 'BUY' THEN usd_amount ELSE 0 END)
                       AS buy_usd,
                   sum(CASE WHEN maker_side = 'SELL' THEN usd_amount ELSE 0 END)
                       AS sell_usd,
                   count(*) AS n_fills
            FROM sides
            GROUP BY outcome_token_id, market_id
        )
        SELECT substr(outcome_token_id, 1, 24) AS token_prefix,
               market_id,
               n_fills,
               round(buy_tokens, 0) AS buy_tok,
               round(sell_tokens, 0) AS sell_tok,
               round(buy_tokens - sell_tokens, 0) AS net_tok,
               round(buy_usd, 0) AS buy_usd,
               round(sell_usd, 0) AS sell_usd,
               round(buy_usd - sell_usd, 0) AS net_usd
        FROM agg
        WHERE n_fills > 100
        ORDER BY abs(buy_usd - sell_usd) DESC
        LIMIT 20
        """
    ).fetchdf().to_string(index=False))

    print("\n=== distribution of buy/sell token-amount imbalance ===")
    print(con.sql(
        """
        WITH sides AS (
            SELECT
                CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END
                    AS outcome_token_id,
                market_id,
                maker_side,
                token_amount
            FROM t
            WHERE market_id IS NOT NULL
        ),
        agg AS (
            SELECT outcome_token_id, market_id,
                   sum(CASE WHEN maker_side = 'BUY' THEN token_amount ELSE 0 END)
                       AS buy_tokens,
                   sum(CASE WHEN maker_side = 'SELL' THEN token_amount ELSE 0 END)
                       AS sell_tokens,
                   count(*) AS n
            FROM sides
            GROUP BY outcome_token_id, market_id
        ),
        ratios AS (
            SELECT outcome_token_id, n, buy_tokens, sell_tokens,
                   CASE WHEN sell_tokens > 0 THEN buy_tokens / sell_tokens ELSE NULL END AS ratio
            FROM agg
            WHERE n > 100 AND sell_tokens > 0
        )
        SELECT
          count(*) AS n_pairs_examined,
          round(median(ratio), 3) AS median_buy_to_sell_ratio,
          round(quantile_cont(ratio, 0.10), 3) AS p10,
          round(quantile_cont(ratio, 0.25), 3) AS p25,
          round(quantile_cont(ratio, 0.75), 3) AS p75,
          round(quantile_cont(ratio, 0.90), 3) AS p90,
          round(min(ratio), 3) AS min_ratio,
          round(max(ratio), 3) AS max_ratio
        FROM ratios
        """
    ).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
