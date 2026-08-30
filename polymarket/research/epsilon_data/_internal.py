"""Private helpers — everything underscore-prefixed, not part of the public API.

Root-agnostic: `EPSILON_DATA_ROOT` may be a LOCAL directory or an `s3://…` bucket path, and the
same code reads both — every read goes through DuckDB (which reads local parquet and, with an R2
secret, `s3://` parquet). This is what lets the two data paths (read-direct-from-R2 vs sync-local)
differ by only the env var, with no code change.

Invariants enforced here so callers never have to remember them:
  * all id columns are strings (77-digit token ids overflow every integer type)
  * a UTC `ts` column is derived from timestamp_ms so nobody sorts on the raw ms/received_ns tiebreak
  * tapes are read ONE token at a time (l1 is ~101M rows; never whole-table)
"""
from __future__ import annotations
import configparser
import functools
import os

import duckdb
import pandas as pd

from .config import data_root

_ID_COLS = ("asset_id", "condition_id", "event_id", "complement_asset_id")


def _root() -> str:
    return str(data_root()).rstrip("/\\")


def is_s3(root: str | None = None) -> bool:
    return (root or _root()).startswith("s3://")


def _uri(*parts: str) -> str:
    """Join under the data root, forward-slashed (DuckDB accepts '/' on Windows and for s3)."""
    return "/".join([_root().replace("\\", "/"), *parts])


def _r2_creds():
    """(key_id, secret, endpoint_host) from env (EPSILON_R2_KEY_ID / _SECRET / _ENDPOINT) or,
    as a fallback, the local rclone.conf [r2] section. None if unavailable."""
    kid, sec, ep = (os.environ.get("EPSILON_R2_KEY_ID"), os.environ.get("EPSILON_R2_SECRET"),
                    os.environ.get("EPSILON_R2_ENDPOINT"))
    if kid and sec and ep:
        return kid, sec, ep.replace("https://", "").replace("http://", "")
    for p in (os.path.expandvars(r"%APPDATA%\rclone\rclone.conf"),
              os.path.expanduser("~/.config/rclone/rclone.conf")):
        if os.path.exists(p):
            c = configparser.ConfigParser()
            try:
                c.read(p)
            except configparser.Error:
                continue
            if c.has_section("r2"):
                r = c["r2"]
                return (r.get("access_key_id"), r.get("secret_access_key"),
                        (r.get("endpoint", "") or "").replace("https://", "").replace("http://", ""))
    return None


def con() -> duckdb.DuckDBPyConnection:
    """A DuckDB connection ready to read the data root. For an s3 root it loads httpfs and installs
    the R2 secret; raises a clear error if credentials are missing (check_setup surfaces it)."""
    c = duckdb.connect()
    c.execute("SET preserve_insertion_order=false;")
    if is_s3():
        creds = _r2_creds()
        if not creds:
            c.close()
            raise RuntimeError(
                "EPSILON_DATA_ROOT is an s3:// path but no R2 credentials found. Set EPSILON_R2_KEY_ID, "
                "EPSILON_R2_SECRET, EPSILON_R2_ENDPOINT (or configure an rclone [r2] remote).")
        kid, sec, ep = creds
        c.execute("INSTALL httpfs; LOAD httpfs; SET http_retries=8; SET http_timeout=120000;")
        c.execute(f"CREATE SECRET r2 (TYPE s3, PROVIDER config, KEY_ID '{kid}', SECRET '{sec}', "
                  f"ENDPOINT '{ep}', REGION 'auto', URL_STYLE 'path', USE_SSL true);")
    return c


@functools.lru_cache(maxsize=4)
def _tokens_cached(root_key: str) -> pd.DataFrame:
    c = con()
    try:
        df = c.execute(f"SELECT * FROM read_parquet('{_uri('tokens.parquet')}')").df()
    finally:
        c.close()
    for col in _ID_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def tokens() -> pd.DataFrame:
    """The full tokens table (30,772 rows), ids as strings. Returns a copy."""
    return _tokens_cached(_root()).copy()


def _parse_exclusions_text(text: str) -> set:
    """asset_ids from an exclusions.csv body: skip blank and '#'-comment lines, treat the first
    remaining line as the header, take the 'asset_id' column. Tolerant of 3- or 5-column schemas."""
    rows = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        return set()
    header = [h.strip() for h in rows[0].split(",")]
    try:
        ai = header.index("asset_id")
    except ValueError:
        ai = 0
    out = set()
    for ln in rows[1:]:
        parts = ln.split(",")
        if ai < len(parts):
            v = parts[ai].strip()
            if v:
                out.add(v)
    return out


@functools.lru_cache(maxsize=4)
def _exclusions_cached(root_key: str) -> frozenset:
    # exclusions.csv is a LOCAL operator instrument; read the local file directly. For an s3 root
    # (read-direct-from-R2), fall back to DuckDB; empty if absent.
    if is_s3():
        c = con()
        try:
            df = c.execute(f"SELECT * FROM read_csv('{_uri('exclusions.csv')}', header=true, "
                           "ignore_errors=true, all_varchar=true)").df()
            return frozenset(str(x) for x in df["asset_id"].dropna()) if "asset_id" in df.columns else frozenset()
        except Exception:
            return frozenset()
        finally:
            c.close()
    p = os.path.join(_root(), "exclusions.csv")
    if not os.path.exists(p):
        return frozenset()
    try:
        return frozenset(_parse_exclusions_text(open(p, encoding="utf-8").read()))
    except Exception:
        return frozenset()


def excluded_ids() -> frozenset:
    """asset_ids listed in exclusions.csv (operator-edited, applied at load). Empty today."""
    return _exclusions_cached(_root())


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
    """Read ONE token's l1 or trades tape (local or s3, via DuckDB). Reads only the token's
    universe partitions and lets parquet footer stats (files sorted by asset_id) skip the rest —
    never a whole-table scan."""
    assert kind in ("l1", "trades")
    glob = _uri(kind, f"universe={universe}", "*", "*.parquet")
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


def align_mids(series_by_key: dict, value_col: str = "mid", freq: str = "1s") -> pd.DataFrame:
    """Align several tapes onto one time index: floor `ts` to `freq`, take the last value per
    bucket, outer-join, forward-fill. Used by load_pair / load_event."""
    cols = {}
    for key, d in series_by_key.items():
        if d is None or d.empty:
            continue
        s = d[["ts", value_col]].copy()
        s["ts"] = s["ts"].dt.floor(freq)
        cols[key] = s.groupby("ts")[value_col].last()
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index().ffill()
