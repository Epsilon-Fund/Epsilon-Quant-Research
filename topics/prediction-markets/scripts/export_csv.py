"""
Export any database table to CSV.
Usage: python scripts/export_csv.py <table_name>
Saves to data/exports/{table_name}_{YYYY-MM-DD}.csv
"""

import os
import sys
from datetime import date

# Add project root to path so imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import csv
import config


VALID_TABLES = {
    "events", "markets", "trades", "wallet_profiles", "collection_log",
    "candles_1d", "candles_1h", "falcon_leaderboard", "wallet_lifetime",
    "wallet_pnl_series",
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python scripts/export_csv.py <table_name>")
        print(f"Valid tables: {', '.join(sorted(VALID_TABLES))}")
        sys.exit(1)

    table_name = sys.argv[1].strip()

    if table_name not in VALID_TABLES:
        print(f"Unknown table: '{table_name}'")
        print(f"Valid tables: {', '.join(sorted(VALID_TABLES))}")
        sys.exit(1)

    if not os.path.exists(config.DB_PATH):
        print("Database not found. Run 'python run_collection.py' first.")
        sys.exit(1)

    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()

    if not rows:
        print(f"Table '{table_name}' is empty.")
        conn.close()
        return

    # Create exports directory
    export_dir = os.path.join(os.path.dirname(config.DB_PATH), "exports")
    os.makedirs(export_dir, exist_ok=True)

    filename = f"{table_name}_{date.today().isoformat()}.csv"
    filepath = os.path.join(export_dir, filename)

    columns = rows[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(tuple(row))

    print(f"Exported {len(rows):,} rows to {filepath}")
    conn.close()


if __name__ == "__main__":
    main()
