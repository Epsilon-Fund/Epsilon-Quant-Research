"""
Database connection helper and schema initializer.
Provides a single function to get a connection and ensures the schema is applied on first use.
"""

import os
import sqlite3

from loguru import logger

import config


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the research database, creating it if needed."""
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)

    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    logger.info("Connected to database: {}", config.DB_PATH)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Apply schema.sql to create all tables and indexes if they don't exist."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn.executescript(schema_sql)
    conn.commit()
    logger.info("Database schema initialized")


def reset_database(conn: sqlite3.Connection) -> None:
    """Drop all tables and recreate them. Use with caution."""
    tables = [
        "collection_log", "wallet_pnl_series", "wallet_lifetime",
        "falcon_leaderboard", "candles_1h", "candles_1d",
        "trades", "wallet_profiles", "markets", "events",
    ]
    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    logger.warning("All tables dropped")
    init_schema(conn)
    logger.info("Database reset complete")
