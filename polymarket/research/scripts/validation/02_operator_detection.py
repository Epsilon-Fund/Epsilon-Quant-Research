"""Check 02: operator address detection.

Top 50 addresses by union (maker + taker) fill count, with:
  - maker_count, taker_count, maker:taker ratio
  - distinct markets touched
  - distinct counterparties (other side of fills)
  - sub-second clustering: fraction of fills with another fill from same
    address within 1-second of timestamp

Heuristic: operators have ratio ~1.0, very high counterparty count, and
high sub-second clustering.
"""
from _common import connect


def main() -> None:
    con = connect()

    print("\n=== identifying top 50 addresses by union fill count ===")
    con.execute(
        """
        CREATE TEMP TABLE top50 AS
        WITH unioned AS (
            SELECT maker AS addr, count(*) AS n FROM t WHERE maker IS NOT NULL GROUP BY maker
            UNION ALL
            SELECT taker AS addr, count(*) AS n FROM t WHERE taker IS NOT NULL GROUP BY taker
        ),
        agg AS (
            SELECT addr, sum(n) AS total_fills FROM unioned GROUP BY addr
        )
        SELECT addr, total_fills FROM agg
        ORDER BY total_fills DESC
        LIMIT 50
        """
    )

    print("\n=== per-address: maker vs taker counts ===")
    df = con.sql(
        """
        WITH mk AS (
            SELECT maker AS addr, count(*) AS m_count, sum(usd_amount) AS m_usd
            FROM t WHERE maker IN (SELECT addr FROM top50) GROUP BY maker
        ),
        tk AS (
            SELECT taker AS addr, count(*) AS t_count, sum(usd_amount) AS t_usd
            FROM t WHERE taker IN (SELECT addr FROM top50) GROUP BY taker
        ),
        mkts AS (
            SELECT addr, count(DISTINCT market_id) AS distinct_markets
            FROM (
                SELECT maker AS addr, market_id FROM t WHERE maker IN (SELECT addr FROM top50)
                UNION ALL
                SELECT taker AS addr, market_id FROM t WHERE taker IN (SELECT addr FROM top50)
            ) GROUP BY addr
        ),
        cps AS (
            SELECT me, count(DISTINCT them) AS distinct_counterparties
            FROM (
                SELECT maker AS me, taker AS them FROM t WHERE maker IN (SELECT addr FROM top50)
                UNION ALL
                SELECT taker AS me, maker AS them FROM t WHERE taker IN (SELECT addr FROM top50)
            ) GROUP BY me
        )
        SELECT t50.addr,
               t50.total_fills,
               coalesce(mk.m_count, 0) AS maker_count,
               coalesce(tk.t_count, 0) AS taker_count,
               round(coalesce(mk.m_count, 0)::DOUBLE / nullif(coalesce(tk.t_count, 0), 0), 3)
                   AS maker_taker_ratio,
               mkts.distinct_markets,
               cps.distinct_counterparties,
               round(coalesce(mk.m_usd, 0) + coalesce(tk.t_usd, 0), 0) AS total_usd
        FROM top50 t50
        LEFT JOIN mk ON mk.addr = t50.addr
        LEFT JOIN tk ON tk.addr = t50.addr
        LEFT JOIN mkts ON mkts.addr = t50.addr
        LEFT JOIN cps ON cps.me = t50.addr
        ORDER BY t50.total_fills DESC
        """
    ).fetchdf()
    print(df.to_string(index=False))

    print("\n=== sub-second clustering for top 50 ===")
    print("(fraction of fills with another fill from same address within 1s)")
    cluster_df = con.sql(
        """
        WITH all_acts AS (
            SELECT maker AS addr, timestamp FROM t WHERE maker IN (SELECT addr FROM top50)
            UNION ALL
            SELECT taker AS addr, timestamp FROM t WHERE taker IN (SELECT addr FROM top50)
        ),
        with_neighbors AS (
            SELECT addr,
                   timestamp,
                   lag(timestamp)  OVER (PARTITION BY addr ORDER BY timestamp) AS prev_ts,
                   lead(timestamp) OVER (PARTITION BY addr ORDER BY timestamp) AS next_ts
            FROM all_acts
        )
        SELECT addr,
               count(*) AS n,
               round(100.0 * sum(CASE
                   WHEN (prev_ts IS NOT NULL AND timestamp - prev_ts < INTERVAL 1 SECOND)
                     OR (next_ts IS NOT NULL AND next_ts - timestamp < INTERVAL 1 SECOND)
                   THEN 1 ELSE 0 END) / count(*), 2) AS pct_sub_second
        FROM with_neighbors
        GROUP BY addr
        ORDER BY n DESC
        """
    ).fetchdf()
    print(cluster_df.to_string(index=False))

    print("\n=== merged candidate operator scorecard ===")
    merged = df.merge(cluster_df[["addr", "pct_sub_second"]], on="addr", how="left")
    # operator score: ratio close to 1.0 AND high counterparties AND high clustering
    def score(row) -> float:
        r = row["maker_taker_ratio"]
        if r is None:
            return 0.0
        ratio_score = max(0.0, 1.0 - abs(r - 1.0))  # closer to 1 → higher
        cp = row["distinct_counterparties"] or 0
        cp_score = min(1.0, cp / 100_000.0)
        clust = row["pct_sub_second"] or 0.0
        clust_score = clust / 100.0
        return round(ratio_score * 0.4 + cp_score * 0.4 + clust_score * 0.2, 3)

    merged["operator_score"] = merged.apply(score, axis=1)
    merged = merged.sort_values("operator_score", ascending=False)
    print(merged[["addr", "total_fills", "maker_taker_ratio", "distinct_markets",
                  "distinct_counterparties", "pct_sub_second", "operator_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
