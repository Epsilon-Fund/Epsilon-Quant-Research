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
    """Read the precomputed coverage.parquet (root-agnostic). Columns per (universe,date):
    pc_hours, book_hours (list<str>), n_pc, n_book, quiet_hours, missing_hours, n_missing."""
    c = _i.con()
    try:
        return c.execute(f"SELECT * FROM read_parquet('{_i._uri('coverage.parquet')}')").df()
    finally:
        c.close()


def coverage(universe=None) -> pd.DataFrame:
    """Per (universe, date), the coverage picture the calendar needs to tell three states apart:
      - `pc_hours`      : hours with price_change (an ACTIVE hour)
      - `book_hours`    : hours with a book snapshot (present even in a QUIET hour)
      - `missing_hours` : hours with NO book at all (a TRUE gap in capture)
      - `quiet_hours`   : book present but no price_change (a quiet market, NOT a gap)
    The one known multi-hour outage is 2026-06-22 14:03:30Z -> 06-23 08:29:10Z (18h26m, both
    universes; `coverage()` marks hour 14 present because it has data up to 14:03);
    2026-06-19 h00-11 is capture start (not a gap); esports 2026-07-24 h12 / 08-21 h11-12 are quiet."""
    cov = _coverage_cached(_i._root())
    if universe is not None:
        cov = cov[cov["universe"] == universe]
    return cov.sort_values(["universe", "date"]).reset_index(drop=True)


def reconciliation() -> pd.DataFrame:
    """The identities that must hold, computed live from the published tables:
    sum(catalog.n_trades) vs the trades row count; sum(catalog.n_l1_events) vs the l1 row count.
    If either stops matching, something downstream has drifted."""
    t = _i.tokens()
    c = _i.con()
    try:
        l1_rows = c.execute(f"SELECT COUNT(*) FROM read_parquet('{_i._uri('l1','*','*','*.parquet')}')").fetchone()[0]
        tr_rows = c.execute(f"SELECT COUNT(*) FROM read_parquet('{_i._uri('trades','*','*','*.parquet')}')").fetchone()[0]
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
    where = (f"WHERE CAST(asset_id AS VARCHAR) IN (SELECT CAST(asset_id AS VARCHAR) FROM "
             f"read_parquet('{_i._uri('tokens.parquet')}') WHERE universe='{universe}')" if universe else "")
    glob = _i._uri("trades", "*", "*", "*.parquet")
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
    return _activity_cached(_i._root(), universe)


def negrisk_sum(event_slug):
    """The NegRisk YES-sum for an event, on a common 1-second index — a sum of each candidate's
    LAST KNOWN mid, **not** an instantaneous sum.

    ⚠️ **The summed quotes are stale, often by many hours.** `load_event()` → `align_mids()` floors
    to 1-second buckets, outer-joins the candidates and forward-fills with **no limit**. Measured on
    `elon-musk-of-tweets-july-21-july-28` (25 YES legs, 135,931 buckets): a mean of 1.31 of 25 legs
    actually update in a given second; the median age of a summed quote is **20.2 hours**, p90 135
    hours, max 10.1 days, and 67.7% of summed values are over an hour old. The median `yes_sum` is
    4.528 as returned here versus 0.270 computed truly instantaneously (audit 2026-09,
    01_digest.md §8.1).

    So this is a **stale composite**, in the same family of error as the sum-of-per-candidate-medians
    that this docstring used to condemn — just on a finer clock. For events where fewer than two legs
    quote per second, the instantaneous NegRisk sum is **not measurable from this capture**. Use this
    for shape and for spotting the right tail, not as a live no-arbitrage figure.

    Returns a DataFrame indexed by `ts` with:
      `yes_sum` : sum of the YES-side mids **carried forward** to that instant
      `n_live`  : how many candidates have EVER quoted by that instant (it is `notna().sum()` AFTER
                  the ffill, so it does NOT count candidates quoting then — on
                  `presidential-election-winner-2028` it averages 44.15 against 1.01 legs actually
                  updating per second)
    `.attrs['n_captured']` = candidates we hold; `.attrs['note']` records the direction of the two
    biases. A sum < 1 can be explained by candidates missing from v1; a sum > 1 is at least as
    likely to be the ffill carrying dead candidates as it is to be a finding."""
    from .tape import load_event
    from . import _internal as _i2  # local alias for clarity
    wide = load_event(event_slug)
    meta = wide.attrs.get("tokens", {})
    yes_cols = [a for a in wide.columns if meta.get(a, {}).get("outcome") == "YES"]
    if not yes_cols:  # non-politics or unlabeled: fall back to outcome_index 0 per market
        yes_cols = [a for a in wide.columns if meta.get(a, {}).get("outcome_index") == 0]
    sub = wide[yes_cols]
    out = pd.DataFrame({"yes_sum": sub.sum(axis=1, min_count=1), "n_live": sub.notna().sum(axis=1)})
    out.attrs["n_captured"] = len(yes_cols)
    out.attrs["note"] = ("YES-sum of LAST-KNOWN mids on a 1s index (ffilled, no limit) — not "
                         "instantaneous; median summed quote is ~20h stale. Missing candidates pull "
                         "the sum DOWN; the ffill keeping dead candidates alive pulls it UP, and that "
                         "is the larger effect. n_live counts ever-quoted legs, not legs quoting now.")
    return out
