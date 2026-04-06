"""
Standalone script to discover and enrich weather-focused whales.
Run this in a separate terminal alongside the candles collection.

Usage:
    python run_whale_discovery.py              # discover + enrich
    python run_whale_discovery.py --discover   # only scan leaderboard
    python run_whale_discovery.py --enrich     # only fetch lifetime + PnL for known whales
"""

import argparse
import sys
import time

import config
from db.database import get_connection, init_schema
from collectors.weather_whales import discover_weather_whales, enrich_weather_whales


def main():
    parser = argparse.ArgumentParser(description="Weather whale discovery")
    parser.add_argument("--discover", action="store_true", help="Only scan leaderboard for whales")
    parser.add_argument("--enrich", action="store_true", help="Only enrich already-discovered whales")
    args = parser.parse_args()

    run_all = not (args.discover or args.enrich)

    if not config.FALCON_API_KEY:
        print("ERROR: FALCON_API_KEY is not set.")
        sys.exit(1)

    start_time = time.time()
    conn = get_connection()
    init_schema(conn)

    try:
        if run_all or args.discover:
            whales = discover_weather_whales(conn, max_non_weather_trades=100)

        if run_all or args.enrich:
            enrich_weather_whales(conn)

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        # Summary
        whale_count = conn.execute("SELECT COUNT(*) FROM wallet_profiles WHERE weather_pnl > 0").fetchone()[0]
        lifetime_count = conn.execute("SELECT COUNT(*) FROM wallet_lifetime").fetchone()[0]
        pnl_count = conn.execute("SELECT COUNT(*) FROM wallet_pnl_series").fetchone()[0]

        print(f"\n=== Weather Whale Discovery Complete ===")
        print(f"Weather whales:    {whale_count}")
        print(f"Lifetime profiles: {lifetime_count}")
        print(f"PnL data points:   {pnl_count:,}")
        print(f"Time elapsed:      {minutes}m {seconds}s")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved — safe to re-run.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
