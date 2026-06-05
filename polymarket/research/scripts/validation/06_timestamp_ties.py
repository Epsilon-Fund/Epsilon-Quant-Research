"""Check 06: timestamp ties within transactions.

Window functions in Phase 1 will partition by (address, market_id,
outcome_index) ordered by timestamp. Multiple fills sharing the same
timestamp + transaction_hash for the same (address, market, outcome)
yield non-deterministic order.

Quantify how often this actually happens. If <0.1% of rows are
involved, we ignore the tiebreak issue. If meaningful, we discuss.
"""
from _common import connect


def main() -> None:
    con = connect()

    print("\n=== fills per (transaction_hash, timestamp) bucket ===")
    print(con.sql(
        """
        WITH g AS (
            SELECT transaction_hash, timestamp, count(*) AS n
            FROM t
            WHERE transaction_hash IS NOT NULL
            GROUP BY transaction_hash, timestamp
        )
        SELECT n AS fills_in_bucket,
               count(*) AS n_buckets,
               sum(n) AS total_fills_involved
        FROM g
        GROUP BY n
        ORDER BY n DESC
        LIMIT 25
        """
    ).fetchdf().to_string(index=False))

    print("\n=== summary: fraction of fills in multi-fill buckets ===")
    print(con.sql(
        """
        WITH g AS (
            SELECT transaction_hash, timestamp, count(*) AS n
            FROM t
            WHERE transaction_hash IS NOT NULL
            GROUP BY transaction_hash, timestamp
        )
        SELECT
            (SELECT count(*) FROM t) AS total_fills,
            sum(CASE WHEN n > 1 THEN n ELSE 0 END) AS fills_in_multi,
            round(100.0 * sum(CASE WHEN n > 1 THEN n ELSE 0 END) /
                  (SELECT count(*) FROM t), 4) AS pct_multi,
            sum(CASE WHEN n = 1 THEN 1 ELSE 0 END) AS singleton_buckets,
            sum(CASE WHEN n > 1 THEN 1 ELSE 0 END) AS multi_buckets
        FROM g
        """
    ).fetchdf().to_string(index=False))

    print("\n=== same-trader-same-market within tied bucket (the actual problem) ===")
    print(con.sql(
        """
        WITH dupe AS (
            SELECT transaction_hash, timestamp, market_id, maker AS addr,
                   CASE WHEN maker_asset_id = '0' THEN taker_asset_id
                        ELSE maker_asset_id END AS token_id,
                   count(*) AS n
            FROM t
            WHERE transaction_hash IS NOT NULL AND market_id IS NOT NULL
            GROUP BY transaction_hash, timestamp, market_id, maker,
                     CASE WHEN maker_asset_id = '0' THEN taker_asset_id
                          ELSE maker_asset_id END
            HAVING count(*) > 1
        )
        SELECT count(*) AS n_problematic_buckets,
               sum(n) AS fills_affected,
               round(100.0 * sum(n) / (SELECT count(*) FROM t), 6) AS pct_of_total
        FROM dupe
        """
    ).fetchdf().to_string(index=False))

    print("\n=== same trader as TAKER tied within tx (the symmetric case) ===")
    print(con.sql(
        """
        WITH dupe AS (
            SELECT transaction_hash, timestamp, market_id, taker AS addr,
                   CASE WHEN maker_asset_id = '0' THEN taker_asset_id
                        ELSE maker_asset_id END AS token_id,
                   count(*) AS n
            FROM t
            WHERE transaction_hash IS NOT NULL AND market_id IS NOT NULL
            GROUP BY transaction_hash, timestamp, market_id, taker,
                     CASE WHEN maker_asset_id = '0' THEN taker_asset_id
                          ELSE maker_asset_id END
            HAVING count(*) > 1
        )
        SELECT count(*) AS n_problematic_buckets,
               sum(n) AS fills_affected,
               round(100.0 * sum(n) / (SELECT count(*) FROM t), 6) AS pct_of_total
        FROM dupe
        """
    ).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
