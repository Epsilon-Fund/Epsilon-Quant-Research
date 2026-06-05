"""Check 07: sample trader sanity (domah).

Compute domah's stats from local parquet, then cross-check vs
Polymarket's public profile API. Should match within a few percent.
"""
import json
import urllib.request
import urllib.error

from _common import connect

DOMAH = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"


def fetch_polymarket_profile(addr: str) -> dict | None:
    """Try a few public endpoints and return whichever works."""
    candidates = [
        f"https://data-api.polymarket.com/profile?address={addr}",
        f"https://lb-api.polymarket.com/profile/address/{addr}",
        f"https://gamma-api.polymarket.com/profile?address={addr}",
        f"https://data-api.polymarket.com/profile?wallet_address={addr}",
    ]
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "epsilon-validation/0.1"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8")
                    print(f"  endpoint: {url}")
                    return json.loads(body)
        except urllib.error.HTTPError as e:
            print(f"  {url} → HTTP {e.code}")
        except Exception as e:
            print(f"  {url} → {e!r}")
    return None


def main() -> None:
    con = connect()

    print("\n=== domah (0x9d84…) — local parquet stats ===")
    print(con.sql(
        f"""
        SELECT
            'maker' AS role,
            count(*) AS fills,
            round(sum(usd_amount), 2) AS usd_volume,
            count(DISTINCT market_id) FILTER (WHERE market_id IS NOT NULL) AS distinct_markets,
            min(timestamp) AS earliest,
            max(timestamp) AS latest
        FROM t WHERE maker = '{DOMAH}'
        UNION ALL
        SELECT 'taker', count(*), round(sum(usd_amount), 2),
               count(DISTINCT market_id) FILTER (WHERE market_id IS NOT NULL),
               min(timestamp), max(timestamp)
        FROM t WHERE taker = '{DOMAH}'
        """
    ).fetchdf().to_string(index=False))

    print("\n=== domah — union (maker ∪ taker) ===")
    print(con.sql(
        f"""
        SELECT
            (SELECT count(*) FROM t WHERE maker = '{DOMAH}' OR taker = '{DOMAH}') AS total_fills,
            (SELECT round(sum(usd_amount), 2) FROM t WHERE maker = '{DOMAH}' OR taker = '{DOMAH}') AS total_usd,
            (SELECT count(DISTINCT market_id) FROM t
             WHERE (maker = '{DOMAH}' OR taker = '{DOMAH}') AND market_id IS NOT NULL) AS distinct_markets
        """
    ).fetchdf().to_string(index=False))

    print("\n=== domah — fetching Polymarket public profile ===")
    profile = fetch_polymarket_profile(DOMAH)
    if profile:
        print("\n--- profile keys ---")
        if isinstance(profile, list) and profile:
            print(list(profile[0].keys()))
            print(json.dumps(profile[0], indent=2)[:2000])
        elif isinstance(profile, dict):
            print(list(profile.keys()))
            print(json.dumps(profile, indent=2)[:2000])
    else:
        print("(no public profile endpoint resolved)")


if __name__ == "__main__":
    main()
