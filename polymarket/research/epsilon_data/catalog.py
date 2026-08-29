"""The tree: find markets. catalog / events / search / resolve, plus research helpers
(coverage, reconciliation, activity_by_time) that the dashboard needs and that are generally
useful — all built only on the published tables, never on private parquet paths from a panel."""
from __future__ import annotations
import functools

import pandas as pd

from . import _internal as _i


def catalog(universe=None, event=None, resolved=None, min_trades=None, min_days=None,
            apply_exclusions=True) -> pd.DataFrame:
    """One row per token: identity, observation and check columns. How anyone finds a market.

    Filters: universe; event (matches event_slug, event_id or event_title); resolved
    (True=closed, False=open); min_trades; min_days. An `excluded` bool column is always
    present. apply_exclusions=True (default) drops excluded tokens; pass False to keep them
    visible-and-marked (what the dashboard does — nothing is ever hidden silently)."""
    t = _i.tokens()
    t["excluded"] = t["asset_id"].isin(_i.excluded_ids())
    if universe is not None:
        t = t[t["universe"] == universe]
    if event is not None:
        ev = str(event)
        t = t[(t["event_slug"] == ev) | (t["event_id"].astype("string") == ev) | (t["event_title"] == event)]
    if resolved is not None:
        t = t[t["closed"] == True] if resolved else t[t["closed"] != True]  # noqa: E712
    if min_trades is not None:
        t = t[t["n_trades"] >= min_trades]
    if min_days is not None:
        t = t[t["n_days"] >= min_days]
    if apply_exclusions:
        t = t[~t["excluded"]]
    return t.reset_index(drop=True)


def events(universe=None, apply_exclusions=True) -> pd.DataFrame:
    """Event rollup: n_markets, n_tokens, date span, total trades, neg_risk — one row per event."""
    t = catalog(universe=universe, apply_exclusions=apply_exclusions)
    g = (t.groupby(["event_id", "event_slug", "event_title", "universe", "neg_risk"], dropna=False)
           .agg(n_markets=("condition_id", "nunique"), n_tokens=("asset_id", "nunique"),
                first_seen=("first_seen", "min"), last_seen=("last_seen", "max"),
                total_trades=("n_trades", "sum"))
           .reset_index())
    g["first_ts"] = pd.to_datetime(g["first_seen"], unit="ms", utc=True)
    g["last_ts"] = pd.to_datetime(g["last_seen"], unit="ms", utc=True)
    return g.sort_values("total_trades", ascending=False).reset_index(drop=True)


def search(text, limit=50) -> pd.DataFrame:
    """Free-text (case-insensitive substring) over event_title, question, market_slug and
    event_slug. How a human finds 'the Fed market' among 15,386. Returns matching token rows."""
    t = _i.tokens()
    s = str(text).lower()
    mask = pd.Series(False, index=t.index)
    for c in ("event_title", "question", "market_slug", "event_slug"):
        mask = mask | t[c].astype(str).str.lower().str.contains(s, na=False, regex=False)
    out = t[mask].copy()
    out["excluded"] = out["asset_id"].isin(_i.excluded_ids())
    return out.head(limit).reset_index(drop=True)


def resolve(ref) -> str:
    """asset_id | path | market_slug -> asset_id, so nobody types a 77-digit number."""
    return _i.resolve_ref(ref)


# ---- research helpers (documented extensions used by the dashboard) ----

@functools.lru_cache(maxsize=4)
def _coverage_cached(root_key: str) -> pd.DataFrame:
    from pathlib import Path
    man = Path(_i._p("pc_file_manifest.txt"))
    rows = []
    if man.exists():
        for ln in man.read_text().splitlines():
            parts = ln.strip().split("/")
            if len(parts) != 3:
                continue
            date, uni, fname = parts
            for table in ("price_change", "book"):
                if fname.startswith(f"{table}_{uni}_"):
                    hh = fname.rsplit("_", 1)[1].split(".")[0]
                    if hh.isdigit():
                        rows.append((uni, date, table, hh))
    return pd.DataFrame(rows, columns=["universe", "date", "table", "hour"])


def coverage(universe=None) -> pd.DataFrame:
    """Per (universe, date), the coverage picture the calendar needs to tell three states apart:
      - `pc_hours`      : hours with price_change (an ACTIVE hour)
      - `book_hours`    : hours with a book snapshot (present even in a QUIET hour)
      - `missing_hours` : hours with NO book at all (a TRUE gap in capture)
      - `quiet_hours`   : book present but no price_change (a quiet market, NOT a gap)
    Built from the archive file manifest. The one known multi-hour outage is
    2026-06-22 15:00 -> 06-23 08:00 (both universes); also a genuine esports-only gap at
    2026-07-24 h12 and 2026-08-21 h11-12. 2026-06-19 h00-11 is capture start, not a gap."""
    cov = _coverage_cached(str(_i.data_root()))
    if universe is not None:
        cov = cov[cov["universe"] == universe]
    allh = {f"{h:02d}" for h in range(24)}
    out = []
    for (uni, date), g in cov.groupby(["universe", "date"]):
        pc = set(g[g["table"] == "price_change"]["hour"])
        bk = set(g[g["table"] == "book"]["hour"])
        out.append({
            "universe": uni, "date": date,
            "pc_hours": sorted(pc), "n_pc": len(pc),
            "book_hours": sorted(bk), "n_book": len(bk),
            "quiet_hours": sorted(bk - pc),                 # book, no trading
            "missing_hours": sorted(allh - bk),             # no book -> true gap
            "n_missing": len(allh - bk),
        })
    return pd.DataFrame(out).sort_values(["universe", "date"]).reset_index(drop=True)


def reconciliation() -> pd.DataFrame:
    """The identities that must hold, computed live from the published tables:
    sum(catalog.n_trades) vs the trades row count; sum(catalog.n_l1_events) vs the l1 row count.
    If either stops matching, something downstream has drifted."""
    t = _i.tokens()
    c = _i.con()
    try:
        l1_rows = c.execute(f"SELECT COUNT(*) FROM read_parquet('{_i._q(_i._p('l1','*','*','*.parquet'))}')").fetchone()[0]
        tr_rows = c.execute(f"SELECT COUNT(*) FROM read_parquet('{_i._q(_i._p('trades','*','*','*.parquet'))}')").fetchone()[0]
    finally:
        c.close()
    return pd.DataFrame([
        {"identity": "sum(n_l1_events) == l1 rows", "catalog_sum": int(t["n_l1_events"].sum()),
         "table_rows": int(l1_rows), "match": int(t["n_l1_events"].sum()) == int(l1_rows)},
        {"identity": "sum(n_trades) == trades rows", "catalog_sum": int(t["n_trades"].sum()),
         "table_rows": int(tr_rows), "match": int(t["n_trades"].sum()) == int(tr_rows)},
    ])


@functools.lru_cache(maxsize=8)
def _activity_cached(root_key: str, universe: str | None) -> pd.DataFrame:
    where = f"WHERE CAST(asset_id AS VARCHAR) IN (SELECT CAST(asset_id AS VARCHAR) FROM read_parquet('{_i._q(_i._p('tokens.parquet'))}') WHERE universe='{universe}')" if universe else ""
    glob = _i._q(_i._p("trades", "*", "*", "*.parquet"))
    c = _i.con()
    try:
        df = c.execute(f"""
            SELECT hour(make_timestamp(timestamp_ms*1000)) AS hour_of_day,
                   dayofweek(make_timestamp(timestamp_ms*1000)) AS weekday,
                   COUNT(*) AS n_trades, SUM(price*size) AS volume
            FROM read_parquet('{glob}') {where}
            GROUP BY 1,2 ORDER BY 1,2""").df()
    finally:
        c.close()
    return df


def activity_by_time(universe=None) -> pd.DataFrame:
    """Trade counts and volume by UTC hour-of-day and weekday (0=Sunday). Answers 'when is there
    flow to capture?'. Scans the 7.2M-row trades table (cached). Esports and politics differ sharply."""
    return _activity_cached(str(_i.data_root()), universe)
