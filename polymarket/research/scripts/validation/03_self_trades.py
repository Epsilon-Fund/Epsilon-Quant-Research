"""Check 03: self-trades (maker == taker).

Total count, distribution by address, sample 10 rows.
"""
from _common import connect


def main() -> None:
    con = connect()

    print("\n=== overall self-trade totals ===")
    print(con.sql(
        """
        SELECT count(*) AS n_self_trades,
               round(sum(usd_amount), 2) AS total_usd,
               count(DISTINCT maker) AS distinct_addresses,
               min(timestamp) AS earliest,
               max(timestamp) AS latest
        FROM t WHERE maker = taker
        """
    ).fetchdf().to_string(index=False))

    print("\n=== top 20 addresses by self-trade count ===")
    print(con.sql(
        """
        SELECT maker AS addr,
               count(*) AS n,
               round(sum(usd_amount), 2) AS usd_volume,
               count(DISTINCT market_id) AS distinct_markets
        FROM t WHERE maker = taker
        GROUP BY maker
        ORDER BY n DESC
        LIMIT 20
        """
    ).fetchdf().to_string(index=False))

    print("\n=== 10 sample self-trade rows ===")
    print(con.sql(
        """
        SELECT timestamp, maker, market_id, maker_side,
               round(usd_amount, 2) AS usd_amount,
               round(token_amount, 4) AS token_amount,
               round(price, 4) AS price,
               substr(transaction_hash, 1, 12) AS tx_prefix
        FROM t WHERE maker = taker
        ORDER BY timestamp DESC
        LIMIT 10
        """
    ).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
