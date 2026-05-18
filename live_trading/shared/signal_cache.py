"""
Two-tier signal cache for the live dashboards.

Tier 1 (process memory): a module-level dict keyed by
``(data_dir, cache_key, signal_date)``.  Survives every Streamlit
rerun within the same Python process — including across browser
sessions, so a second user opening the dashboard never re-pays the
``run_dashboard`` cost as long as the first user already warmed it.

Tier 2 (disk JSON): ``<data_dir>/signals_cache.json`` written after a
successful compute.  Survives Streamlit / VPS restarts.  On boot, the
first ``load()`` call hydrates Tier 1 from disk if the keys match.

``cache_key`` is the existing md5 of (active symbols + live_params)
already computed in each ``streamlit_app.py``.  ``signal_date`` is the
'YYYY-MM-DD' string of the most recently completed bar.  Together they
guarantee a miss whenever the dashboard needs to recompute (new bar,
new optimisation, changed symbol set).

Public API:
    load(data_dir, cache_key, signal_date)   -> tuple | None
    store(data_dir, cache_key, signal_date, coin_rows, signal_date_obj, generated_at)
    clear(data_dir=None)                     -> None
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date
from typing import Any, Optional


_LOCK = threading.Lock()
_MEM: dict[tuple, tuple] = {}

_FILE_NAME = "signals_cache.json"
# Bump if the cached coin_rows schema changes in a way that downstream
# code can't tolerate (e.g. removed fields).  Mismatched versions on
# disk are treated as a miss.
_SCHEMA_VERSION = 1


def _disk_path(data_dir: str) -> str:
    return os.path.join(data_dir, _FILE_NAME)


def _json_default(o: Any):
    if isinstance(o, set):
        return list(o)
    if isinstance(o, (date,)):
        return o.isoformat()
    # Last resort — coerce anything else to its repr-y string form
    return str(o)


def load(data_dir: str, cache_key: str, signal_date: str):
    """Return ``(coin_rows, signal_date_obj, generated_at)`` if the
    cache holds an entry matching ``(data_dir, cache_key, signal_date)``,
    otherwise ``None``.

    Checks process memory first; on miss, tries the on-disk JSON and
    populates memory on hit.
    """
    mem_key = (os.path.abspath(data_dir), cache_key, signal_date)

    with _LOCK:
        cached = _MEM.get(mem_key)
    if cached is not None:
        return cached

    path = _disk_path(data_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception as e:
        print(f"signal_cache: disk read skipped ({path}): {e}")
        return None

    if (payload.get("schema_version") != _SCHEMA_VERSION
            or payload.get("cache_key")    != cache_key
            or payload.get("signal_date")  != signal_date):
        return None

    try:
        sd_obj = date.fromisoformat(payload["signal_date"])
    except Exception:
        sd_obj = payload["signal_date"]

    result = (payload["coin_rows"], sd_obj, payload["generated_at"])
    with _LOCK:
        _MEM[mem_key] = result
    return result


def store(
    data_dir: str,
    cache_key: str,
    signal_date: str,
    coin_rows: list,
    signal_date_obj,
    generated_at: str,
) -> None:
    """Write the cache to process memory and disk.

    ``signal_date`` is the YYYY-MM-DD string used as the cache key.
    ``signal_date_obj`` is the original ``date`` object returned by
    ``run_dashboard`` — kept around so callers don't have to re-parse.
    Disk failures are logged but never raise; the in-memory cache is
    still populated.
    """
    mem_key = (os.path.abspath(data_dir), cache_key, signal_date)
    with _LOCK:
        _MEM[mem_key] = (coin_rows, signal_date_obj, generated_at)

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "cache_key":      cache_key,
        "signal_date":    signal_date,
        "generated_at":   generated_at,
        "coin_rows":      coin_rows,
    }
    try:
        with open(_disk_path(data_dir), "w") as f:
            json.dump(payload, f, default=_json_default)
    except Exception as e:
        print(f"signal_cache: disk write skipped: {e}")


def clear(data_dir: Optional[str] = None) -> None:
    """Drop the cached entries for ``data_dir`` (or all dirs if None).

    Removes them from process memory AND from disk so the next call to
    ``load()`` is a guaranteed miss.  Use this from the dashboard's
    "Refresh signals" button.
    """
    with _LOCK:
        if data_dir is None:
            _MEM.clear()
            dirs_to_unlink: set[str] = set()
            # Best-effort: we don't know which dirs have files; do nothing
            # else here, callers passing None usually only want memory cleared.
        else:
            abs_dir = os.path.abspath(data_dir)
            stale = [k for k in _MEM if k[0] == abs_dir]
            for k in stale:
                _MEM.pop(k, None)
            dirs_to_unlink = {abs_dir}

    for d in dirs_to_unlink:
        path = os.path.join(d, _FILE_NAME)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"signal_cache: disk clear skipped ({path}): {e}")
