"""Check 01: orphan rate over time (monthly).

Question: is the rate of NULL market_id stable, or rising toward the
present (which would indicate Gamma snapshot is aging)?
"""
from _common import connect


def main() -> None:
    con = connect()
    print("\n=== monthly orphan rate ===")
    df = con.sql(
        """
        SELECT date_trunc('month', timestamp) AS month,
               count(*) AS n_trades,
               sum(CASE WHEN market_id IS NULL THEN 1 ELSE 0 END) AS n_orphan,
               round(100.0 * sum(CASE WHEN market_id IS NULL THEN 1 ELSE 0 END) / count(*), 4)
                   AS pct_orphan
        FROM t
        GROUP BY month
        ORDER BY month
        """
    ).fetchdf()
    print(df.to_string(index=False))

    print("\n=== latest 6 months only ===")
    print(df.tail(6).to_string(index=False))

    print("\n=== overall orphan rate ===")
    print(con.sql(
        """
        SELECT count(*) AS n,
               sum(CASE WHEN market_id IS NULL THEN 1 ELSE 0 END) AS n_orphan,
               round(100.0 * sum(CASE WHEN market_id IS NULL THEN 1 ELSE 0 END) / count(*), 4)
                   AS pct_orphan
        FROM t
        """
    ).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
