from pathlib import Path

from data_infra.duck import connect

ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = ROOT / "data" / "raw" / "orderFilled_complete.csv"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    if not TRADES_CSV.exists():
        raise SystemExit(f"missing trades CSV: {TRADES_CSV}")

    con = connect()
    trades = f"read_csv_auto('{TRADES_CSV}')"

    section("schema")
    print(con.sql(f"DESCRIBE SELECT * FROM {trades} LIMIT 0").fetchdf().to_string(index=False))

    section("row count")
    (n,) = con.sql(f"SELECT count(*) FROM {trades}").fetchone()
    print(f"{n:,}")

    section("timestamp range")
    print(con.sql(f"""
        SELECT
            min(timestamp) AS earliest,
            max(timestamp) AS latest,
            to_timestamp(min(timestamp)::BIGINT) AS earliest_utc,
            to_timestamp(max(timestamp)::BIGINT) AS latest_utc
        FROM {trades}
    """).fetchdf().to_string(index=False))

    section("unique addresses")
    print(con.sql(f"""
        SELECT
            count(DISTINCT maker) AS unique_makers,
            count(DISTINCT taker) AS unique_takers
        FROM {trades}
    """).fetchdf().to_string(index=False))

    usdc_volume_expr = """
        CASE
            WHEN makerAssetId = '0' THEN makerAmountFilled / 1e6
            WHEN takerAssetId = '0' THEN takerAmountFilled / 1e6
            ELSE 0
        END
    """

    section("top 20 makers by trade count")
    print(con.sql(f"""
        SELECT maker, count(*) AS trades
        FROM {trades}
        GROUP BY maker
        ORDER BY trades DESC
        LIMIT 20
    """).fetchdf().to_string(index=False))

    section("top 20 makers by USDC volume")
    print(con.sql(f"""
        SELECT
            maker,
            sum({usdc_volume_expr}) AS usdc_volume,
            count(*) AS trades
        FROM {trades}
        GROUP BY maker
        ORDER BY usdc_volume DESC
        LIMIT 20
    """).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
