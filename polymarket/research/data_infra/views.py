"""Loader for sql/views.sql.

Every script that uses trader_actions / trader_actions_orphan / markets_tokens
calls load_views(con) once after creating the DuckDB connection.

The traders_raw / traders_filtered views (Phase 3) require data/traders.parquet
on disk. If it's missing, those two views are stripped from the SQL before
execution so the rest still loads.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "sql" / "views.sql"
TRADES_GLOB = str(ROOT / "data" / "trades" / "trades_delta_shard*.parquet")
SEED_PATH = str(ROOT / "data" / "trades" / "trades_seed.parquet")
TRADERS_PATH = ROOT / "data" / "traders.parquet"

_TRADERS_BLOCK_START = "-- ============ TRADERS (Phase 3) ============"
_TRADERS_BLOCK_END = "-- ============ ORPHAN TRADER ACTIONS ============"


def latest_markets_path() -> str:
    cands = sorted((ROOT / "data" / "markets").glob("markets_*.parquet"))
    if not cands:
        raise SystemExit(f"no markets_*.parquet in {ROOT / 'data' / 'markets'}")
    return str(cands[-1])


def load_views(con) -> None:
    sql = SQL_PATH.read_text().format(
        TRADES_GLOB=TRADES_GLOB,
        SEED_PATH=SEED_PATH,
        MARKETS_PATH=latest_markets_path(),
        TRADERS_PATH=str(TRADERS_PATH),
    )
    if not TRADERS_PATH.exists():
        # Strip the traders views block; everything else loads fine.
        pre, _, rest = sql.partition(_TRADERS_BLOCK_START)
        _, _, post = rest.partition(_TRADERS_BLOCK_END)
        sql = pre + _TRADERS_BLOCK_END + post
    con.execute(sql)
