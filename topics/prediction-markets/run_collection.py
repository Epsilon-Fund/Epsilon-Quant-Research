"""
Main entry point for the Polymarket research data pipeline.
Supports running the full pipeline or individual collection steps via CLI flags.
"""

import argparse
import sys
import time

from loguru import logger

import config
from db.database import get_connection, init_schema, reset_database
from collectors.markets import collect_markets
from collectors.candlesticks import collect_candlesticks
from collectors.trades import collect_trades
from collectors.weather_whales import discover_weather_whales, enrich_weather_whales


def print_summary(conn):
    """Print a final summary of all data in the database."""
    events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    candles_1d = conn.execute("SELECT COUNT(*) FROM candles_1d").fetchone()[0]
    candles_1h = conn.execute("SELECT COUNT(*) FROM candles_1h").fetchone()[0]
    trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    leaderboard = conn.execute("SELECT COUNT(*) FROM falcon_leaderboard").fetchone()[0]
    whales = conn.execute("SELECT COUNT(*) FROM wallet_profiles WHERE weather_pnl > 0").fetchone()[0]
    lifetime = conn.execute("SELECT COUNT(*) FROM wallet_lifetime").fetchone()[0]
    pnl_series = conn.execute("SELECT COUNT(*) FROM wallet_pnl_series").fetchone()[0]

    print("\n=== Collection Complete ===")
    print(f"Events:            {events:,}")
    print(f"Markets:           {markets:,}")
    print(f"Candles (daily):   {candles_1d:,}")
    print(f"Candles (hourly):  {candles_1h:,}")
    print(f"Trades:            {trades:,}")
    print(f"Leaderboard:       {leaderboard:,}")
    print(f"Weather whales:    {whales:,}")
    print(f"Wallet lifetime:   {lifetime:,}")
    print(f"Wallet PnL series: {pnl_series:,}")


def main():
    parser = argparse.ArgumentParser(description="Polymarket research data pipeline")
    parser.add_argument("--markets", action="store_true", help="Collect markets + events only")
    parser.add_argument("--candles", action="store_true", help="Collect candlesticks only")
    parser.add_argument("--trades", action="store_true", help="Collect trades only")
    parser.add_argument("--whales", action="store_true", help="Discover weather whales + enrich with lifetime + PnL")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables")
    args = parser.parse_args()

    # If no flags, run full pipeline
    run_all = not (args.markets or args.candles or args.trades or args.whales or args.reset)

    # Check API key
    if not config.FALCON_API_KEY:
        print("ERROR: FALCON_API_KEY is not set.")
        print("Create a .env file in the project root with your key:")
        print("  FALCON_API_KEY=your_key_here")
        print("See .env.example for the template.")
        sys.exit(1)

    start_time = time.time()

    conn = get_connection()
    init_schema(conn)

    if args.reset:
        confirm = input("This will DELETE all data. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            conn.close()
            return
        reset_database(conn)
        print("Database reset complete.")
        conn.close()
        return

    try:
        # Stage 1: Markets
        if run_all or args.markets:
            collect_markets(conn)

        # Stage 2: Candlesticks
        if run_all or args.candles:
            collect_candlesticks(conn)

        # Stage 3: Trades
        if run_all or args.trades:
            collect_trades(conn)

        # Stage 4: Weather whale discovery + enrichment
        if run_all or args.whales:
            discover_weather_whales(conn)
            enrich_weather_whales(conn)

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print_summary(conn)
        print(f"Time elapsed:      {minutes}m {seconds}s")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved — safe to re-run.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
