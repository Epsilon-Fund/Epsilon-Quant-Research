"""Shared DuckDB connection + view setup for validation scripts.

Run from polymarket-copy/ as:
    PYTHONPATH=. uv run python scripts/validation/NN_*.py
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
TRADES_GLOB = str(ROOT / "data" / "trades" / "trades_delta_shard*.parquet")
SEED_PATH = str(ROOT / "data" / "trades" / "trades_seed.parquet")


def latest_markets_parquet() -> str:
    cands = sorted((ROOT / "data" / "markets").glob("markets_*.parquet"))
    if not cands:
        raise SystemExit("no markets_*.parquet found")
    return str(cands[-1])


def connect(threads: int = 8) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(
        f"""
        CREATE VIEW t AS
            SELECT * FROM read_parquet('{TRADES_GLOB}')
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{SEED_PATH}')
        """
    )
    con.execute(f"CREATE VIEW m AS SELECT * FROM read_parquet('{latest_markets_parquet()}')")
    return con
