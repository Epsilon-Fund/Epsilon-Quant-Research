"""Market discovery + per-token replay materialization over the VPS L2 Parquet.

Why per-token: the engine's :class:`~mm_engine.orders.OrderManager` keeps at most one quote
per ``(token, side)`` and reconciles the strategy's *desired* quotes against the active set on
**every** event. The ``SymmetricQuoter`` only quotes the token of the current event, so on the
next *other-token* event a multi-token replay would see no desired quote for the previous token
and **cancel** it (Check C in [[mm_join1_reconciliation_findings]] established this). To measure a
single market honestly — where the resting order must persist across that token's own stream — we
replay **one token at a time** over a small filtered Parquet dir.

Materialization is two-stage and cached so reruns are cheap:

1. **compact** — one scan per ``(universe, table)`` writes the selected tokens' rows only
   (``WHERE asset_id IN (...)``) to a compact Parquet. This is the only pass over the full
   65.7M-row ``price_change`` table.
2. **per-token dirs** — each token's rows are sliced from the compact Parquet into
   ``<cache>/<universe>/<token>/<table>_x.parquet`` (the flat ``{table}_*.parquet`` layout
   :func:`mm_engine.feeds.replay_parquet.replay_parquet` globs).

Lookahead-free, append-only-respecting: we only ever read the captured shards and write
derived slices; nothing mutates a source shard. ``asset_id`` (CLOB token id) is a decimal
string and is kept verbatim (it is not an address).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import duckdb

# The authoritative column projection per table (live VPS schema — extra `universe`,
# `fee_rate_bps`/`transaction_hash` on trades, `spread` (not sizes) on bba, `best_bid`/`best_ask`
# on price_change). replay_parquet is schema-drift tolerant, but we project explicitly so the
# per-token slices are stable and small.
TABLE_COLS = {
    "book": "timestamp_ms, received_at, received_ns, universe, asset_id, market, bids, asks",
    "trades": ("timestamp_ms, received_at, received_ns, universe, asset_id, market, "
               "price, size, side, fee_rate_bps, transaction_hash"),
    "price_change": ("timestamp_ms, received_at, received_ns, universe, asset_id, market, "
                     "price, side, size, best_bid, best_ask"),
    "bba": ("timestamp_ms, received_at, received_ns, universe, asset_id, market, "
            "best_bid, best_ask, spread"),
}
TABLES = tuple(TABLE_COLS)


@dataclass(frozen=True)
class MarketSpec:
    """One quotable token selected for evaluation (the unit of observation in the scorecard)."""

    universe: str
    token_id: str
    market: str            # condition id (0x...) — a market may contribute several tokens
    n_trades: int
    volume: float
    avg_price: float
    median_spread: float   # median touch spread (best_ask - best_bid), from the bba stream
    median_mid: float
    bba_checkpoints: int

    @property
    def half_spread(self) -> float:
        """Half the median touch spread — the quoter rests AT the touch (queue position binds)."""
        return max(self.median_spread / 2.0, 0.001)


def _table_files(l2_root: Path, universe: str, table: str) -> list[str]:
    return [str(p) for d in sorted(l2_root.glob(f"*/{universe}")) for p in d.glob(f"{table}_*.parquet")]


def capture_span(l2_root: Path, universe: str, con: duckdb.DuckDBPyConnection | None = None) -> dict:
    """Min/max ``timestamp_ms`` and span (hours/days) of the capture for one universe.

    Grounds the realism statement (~39.5 h ≈ 1.6 days, NOT ~10 days) — :mod:`brain/CODEX.md`
    rule 1: state the sample size and why a split is or isn't warranted.
    """
    own = con is None
    con = con or duckdb.connect()
    try:
        files = _table_files(l2_root, universe, "price_change")
        lo, hi = con.execute(
            "SELECT min(timestamp_ms), max(timestamp_ms) FROM read_parquet(?)", [files]
        ).fetchone()
        hours = (hi - lo) / 1000 / 3600
        return {"universe": universe, "ts_min": int(lo), "ts_max": int(hi),
                "hours": hours, "days": hours / 24.0}
    finally:
        if own:
            con.close()


def select_markets(
    l2_root: Path,
    universe: str,
    *,
    top_k: int = 12,
    min_trades: int = 150,
    price_lo: float = 0.05,
    price_hi: float = 0.95,
    min_spread: float = 0.0,
    con: duckdb.DuckDBPyConnection | None = None,
) -> list[MarketSpec]:
    """Pick the ``top_k`` most-traded *quotable* tokens in a universe.

    Quotable = average traded price in ``[price_lo, price_hi]`` (a genuine two-sided book, not a
    0/1 resolver) and at least ``min_trades`` prints (so fills and markout are estimable). Ordered
    by trade count. The median touch spread (from the ``bba`` stream) sets the quoter's half-spread.
    """
    own = con is None
    con = con or duckdb.connect()
    try:
        tfiles = _table_files(l2_root, universe, "trades")
        bfiles = _table_files(l2_root, universe, "bba")
        rows = con.execute(
            """
            WITH t AS (
                SELECT asset_id, any_value(market) AS market, count(*) AS n_trades,
                       sum(size) AS volume, avg(price) AS avg_price
                FROM read_parquet(?) GROUP BY asset_id
                HAVING count(*) >= ? AND avg(price) BETWEEN ? AND ?
            ),
            b AS (
                SELECT asset_id,
                       median(best_ask - best_bid) AS median_spread,
                       median((best_bid + best_ask) / 2.0) AS median_mid,
                       count(*) AS bba_checkpoints
                FROM read_parquet(?)
                WHERE best_bid IS NOT NULL AND best_ask IS NOT NULL AND best_ask > best_bid
                GROUP BY asset_id
            )
            SELECT t.asset_id, t.market, t.n_trades, t.volume, t.avg_price,
                   COALESCE(b.median_spread, 0.0), COALESCE(b.median_mid, t.avg_price),
                   COALESCE(b.bba_checkpoints, 0)
            FROM t LEFT JOIN b USING (asset_id)
            WHERE COALESCE(b.median_spread, 0.0) >= ?
            ORDER BY t.n_trades DESC
            LIMIT ?
            """,
            [tfiles, min_trades, price_lo, price_hi, bfiles, min_spread, top_k],
        ).fetchall()
        return [
            MarketSpec(universe, str(r[0]), str(r[1]), int(r[2]), float(r[3]), float(r[4]),
                       float(r[5]), float(r[6]), int(r[7]))
            for r in rows
        ]
    finally:
        if own:
            con.close()


def _compact_path(cache: Path, universe: str, table: str) -> Path:
    return cache / f"_compact_{universe}_{table}.parquet"


def build_compact(
    l2_root: Path,
    universe: str,
    specs: list[MarketSpec],
    cache: Path,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """One scan per table: write the selected tokens' rows only into a compact Parquet (cached)."""
    own = con is None
    con = con or duckdb.connect()
    cache.mkdir(parents=True, exist_ok=True)
    token_list = [s.token_id for s in specs]
    placeholders = ", ".join(["?"] * len(token_list))
    out: dict[str, Path] = {}
    # Coverage sidecar: the compact is keyed by (universe, table) but holds only a token SUBSET,
    # so reuse it only when it already covers the requested tokens (else a larger top_k would
    # silently read a stale compact missing the new tokens).
    cov_path = cache / f"_compact_{universe}_tokens.json"
    covered = set()
    if cov_path.exists():
        try:
            covered = set(json.loads(cov_path.read_text()))
        except (json.JSONDecodeError, OSError):
            covered = set()
    need_rebuild = force or not set(token_list).issubset(covered)
    try:
        for table, cols in TABLE_COLS.items():
            cp = _compact_path(cache, universe, table)
            out[table] = cp
            if cp.exists() and not need_rebuild:
                continue
            files = _table_files(l2_root, universe, table)
            target = str(cp).replace("'", "''")
            con.execute(
                f"COPY (SELECT {cols} FROM read_parquet(?) "
                f"WHERE asset_id IN ({placeholders}) ORDER BY timestamp_ms, received_ns) "
                f"TO '{target}' (FORMAT parquet)",
                [files, *token_list],
            )
        if need_rebuild:
            # the rebuilt compact contains exactly token_list (the COPY overwrote it)
            cov_path.write_text(json.dumps(sorted(set(token_list))))
        return out
    finally:
        if own:
            con.close()


def materialize_token(
    spec: MarketSpec,
    cache: Path,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    force: bool = False,
) -> Path:
    """Slice one token's rows from the compact Parquet into a flat per-token replay dir.

    Returns the dir holding ``{table}_x.parquet`` for each table — the exact layout
    :func:`mm_engine.feeds.replay_parquet.replay_parquet` expects.
    """
    own = con is None
    con = con or duckdb.connect()
    out = cache / spec.universe / spec.token_id
    if out.exists() and not force:
        if all((out / f"{t}_x.parquet").exists() for t in TABLES):
            if own:
                con.close()
            return out
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        for table, cols in TABLE_COLS.items():
            cp = _compact_path(cache, spec.universe, table)
            target = str(out / f"{table}_x.parquet").replace("'", "''")
            con.execute(
                f"COPY (SELECT {cols} FROM read_parquet(?) WHERE asset_id = ? "
                f"ORDER BY timestamp_ms, received_ns) TO '{target}' (FORMAT parquet)",
                [str(cp), spec.token_id],
            )
        return out
    finally:
        if own:
            con.close()
