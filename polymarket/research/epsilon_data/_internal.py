"""Private helpers — everything underscore-prefixed, not part of the public API.

Key invariants enforced here so callers never have to remember them:
  * all id columns are strings (77-digit token ids overflow every integer type)
  * a UTC `ts` column is derived from timestamp_ms so nobody sorts on the raw ms/received_ns
    tiebreak by hand
  * tapes are read ONE token at a time (l1 is ~101M rows; never whole-table)
"""
from __future__ import annotations
import functools
from pathlib import Path

import duckdb
import pandas as pd

from .config import data_root

_ID_COLS = ("asset_id", "condition_id", "event_id", "complement_asset_id")


def _p(*parts: str) -> str:
    return str(data_root().joinpath(*parts))


def _q(path: str) -> str:
    return path.replace("\\", "/")


@functools.lru_cache(maxsize=4)
def _tokens_cached(root_key: str) -> pd.DataFrame:
    df = pd.read_parquet(_p("tokens.parquet"))
    for c in _ID_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


def tokens() -> pd.DataFrame:
    """The full tokens table (30,772 rows), ids as strings. Returns a copy."""
    return _tokens_cached(str(data_root())).copy()


@functools.lru_cache(maxsize=4)
def _exclusions_cached(root_key: str) -> frozenset:
    path = Path(_p("exclusions.csv"))
    ids: set[str] = set()
    if path.exists():
        try:
            ex = pd.read_csv(path, comment="#")
            if "asset_id" in ex.columns:
                ids = {str(x) for x in ex["asset_id"].dropna()}
        except Exception:
            pass
    return frozenset(ids)


def excluded_ids() -> frozenset:
    """asset_ids listed in exclusions.csv (operator-edited, applied at load). Empty today."""
    return _exclusions_cached(str(data_root()))


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET preserve_insertion_order=false;")
    return c


def resolve_ref(ref) -> str:
    """asset_id | path | market_slug -> asset_id (string). market_slug (2 tokens) returns the
    outcome_index-0 side; use load_pair(condition_id) for both sides."""
    t = tokens()
    ref = str(ref)
    if (t["asset_id"] == ref).any():
        return ref
    m = t[t["path"] == ref]
    if len(m) == 1:
        return m["asset_id"].iloc[0]
    if len(m) > 1:
        raise ValueError(f"path not unique: {ref!r}")
    m = t[t["market_slug"] == ref].sort_values("outcome_index")
    if len(m) >= 1:
        return m["asset_id"].iloc[0]
    raise KeyError(f"cannot resolve ref {ref!r} (not an asset_id, path, or market_slug)")


def token_row(asset_id: str) -> pd.Series:
    t = tokens()
    r = t[t["asset_id"] == str(asset_id)]
    if r.empty:
        raise KeyError(asset_id)
    return r.iloc[0]


def to_ms(x) -> int | None:
    """datetime-ish or epoch-ms -> epoch-ms int (UTC). None passes through."""
    if x is None:
        return None
    if isinstance(x, bool):
        raise TypeError("bool is not a time")
    if isinstance(x, int):
        return x
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value // 10**6)


def add_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Prepend a UTC datetime `ts` derived from timestamp_ms (raw ms/received_ns kept)."""
    if "timestamp_ms" in df.columns:
        df.insert(0, "ts", pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True))
    return df


def read_token_tape(kind: str, asset_id: str, universe: str, start=None, end=None) -> pd.DataFrame:
    """Read ONE token's l1 or trades tape. Reads only the token's universe partitions and lets
    parquet footer stats (files are sorted by asset_id) skip the rest — never a whole-table scan."""
    assert kind in ("l1", "trades")
    glob = _q(_p(kind, f"universe={universe}", "*", "*.parquet"))
    conds = [f"CAST(asset_id AS VARCHAR) = '{asset_id}'"]
    s, e = to_ms(start), to_ms(end)
    if s is not None:
        conds.append(f"timestamp_ms >= {s}")
    if e is not None:
        conds.append(f"timestamp_ms < {e}")
    where = " AND ".join(conds)
    c = con()
    try:
        df = c.execute(
            f"SELECT * FROM read_parquet('{glob}') WHERE {where} ORDER BY timestamp_ms, received_ns"
        ).df()
    finally:
        c.close()
    if "asset_id" in df.columns:
        df["asset_id"] = df["asset_id"].astype("string")
    return add_ts(df)


def align_mids(series_by_key: dict[str, pd.DataFrame], value_col: str = "mid", freq: str = "1s") -> pd.DataFrame:
    """Align several tapes onto one time index: floor `ts` to `freq`, take the last value per
    bucket, outer-join, forward-fill. Used by load_pair / load_event."""
    cols = {}
    for key, d in series_by_key.items():
        if d.empty:
            continue
        s = d[["ts", value_col]].copy()
        s["ts"] = s["ts"].dt.floor(freq)
        s = s.groupby("ts")[value_col].last()
        cols[key] = s
    if not cols:
        return pd.DataFrame()
    wide = pd.concat(cols, axis=1).sort_index().ffill()
    return wide
