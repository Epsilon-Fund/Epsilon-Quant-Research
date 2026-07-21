"""Book-measured execution costs — spread/depth READ from the captured book.

Standing rule ([[pm_dali_workflow_revision_decision]] Tier 2 item 3): with real
captured L2, quoted spread and top-of-book depth at a timestamp are **measured** from
the reconstructed book, staleness-gated — never estimated. The estimated spread
surface (SPREAD-1/1b/2, ``lib.spread_surface``) is retired as a costing method
wherever the captured book covers the fill; it survives only as a **labelled
fallback** for uncovered timestamps (pre-capture history, gaps, stale windows).

Data substrate: the ``l1_states_*.parquet`` shards written by
``scripts/mm_real_l2_gate_scan.py`` — one row per change of a token's
``(best_bid, bid_size, best_ask, ask_size, stale)`` tuple, replayed through the
engine's ``BookTracker`` behind the required capture-quality gate
(``mm_eval.capture_gate``). The ``stale`` flag is the tracker's own ≤5s / gap /
no-anchor verdict at that event; a stale state is never served as a measurement.

Usage::

    idx = BookCostIndex.load([Path("data/analysis/real_l2/2026-07-01/politics_negrisk")])
    q = idx.quote(asset_id, ts_ms)          # BookQuote(source="measured_book") or None
    q = idx.quote_or_surface(asset_id, ts_ms, surface=s, price=.., ttr_hours=..,
                             trade_rate=.., category="politics")   # labelled fallback

Every returned quote carries ``source`` — ``"measured_book"`` or
``"surface_fallback"`` — so no downstream number can silently mix the two.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_STALENESS_MS = 5_000   # same ≤5s rule as BookTracker / the dali retests


@dataclass(frozen=True)
class BookQuote:
    """One costed book observation at (asset_id, ts_ms)."""

    asset_id: str
    ts_ms: int
    source: str                    # "measured_book" | "surface_fallback"
    best_bid: float | None
    best_ask: float | None
    bid_size: float | None
    ask_size: float | None
    age_ms: int | None             # ts_ms - state timestamp (measured only)

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def half_spread_cents(self) -> float | None:
        s = self.spread
        return None if s is None else s / 2.0 * 100.0

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def touch_depth(self) -> float | None:
        if self.bid_size is None or self.ask_size is None:
            return None
        return self.bid_size + self.ask_size


class BookCostIndex:
    """As-of lookup over the gated L1-state shards for a set of assets.

    Loads per-asset sorted arrays once (DuckDB-filtered, so a market-subset load
    stays small even when the shard archive holds hundreds of millions of rows),
    then answers point queries: the last non-stale L1 state at or before ``ts_ms``,
    served only while it is younger than ``max_staleness_ms``.
    """

    def __init__(self, per_asset: dict[str, dict[str, np.ndarray]],
                 max_staleness_ms: int = MAX_STALENESS_MS) -> None:
        self._per_asset = per_asset
        self.max_staleness_ms = int(max_staleness_ms)

    @classmethod
    def load(
        cls,
        state_dirs: list[Path] | Path,
        *,
        asset_ids: list[str] | None = None,
        max_staleness_ms: int = MAX_STALENESS_MS,
    ) -> "BookCostIndex":
        """Load ``l1_states_*.parquet`` under each dir (optionally asset-filtered)."""
        import duckdb

        dirs = [state_dirs] if isinstance(state_dirs, Path) else list(state_dirs)
        globs = [str(Path(d) / "l1_states_*.parquet") for d in dirs]
        con = duckdb.connect()
        try:
            where = "WHERE NOT stale"
            params: list[object] = [globs]
            if asset_ids is not None:
                where += " AND asset_id IN (SELECT unnest($assets))"
                params = [globs, asset_ids]
            rel = con.execute(
                "SELECT asset_id, timestamp_ms, best_bid, bid_size, best_ask, ask_size "
                f"FROM read_parquet($globs, union_by_name=true) {where} "
                "ORDER BY asset_id, timestamp_ms",
                {"globs": globs} if asset_ids is None else {"globs": globs, "assets": asset_ids},
            )
            df = rel.df()
        finally:
            con.close()
        per_asset: dict[str, dict[str, np.ndarray]] = {}
        for aid, g in df.groupby("asset_id", sort=False):
            per_asset[str(aid)] = {
                "ts": g["timestamp_ms"].to_numpy(np.int64),
                "bid": g["best_bid"].to_numpy(float),
                "ask": g["best_ask"].to_numpy(float),
                "bid_sz": g["bid_size"].to_numpy(float),
                "ask_sz": g["ask_size"].to_numpy(float),
            }
        return cls(per_asset, max_staleness_ms=max_staleness_ms)

    def quote(self, asset_id: str, ts_ms: int) -> BookQuote | None:
        """Last fresh measured L1 at or before ``ts_ms``; None when uncovered.

        Uncovered = no state yet, or the newest state at/<= ``ts_ms`` is older than
        ``max_staleness_ms`` (the book may have moved arbitrarily since; serving it
        would be an estimate, which is exactly what this module exists to prevent).
        """
        state = self._per_asset.get(str(asset_id))
        if state is None or len(state["ts"]) == 0:
            return None
        idx = int(np.searchsorted(state["ts"], int(ts_ms), side="right") - 1)
        if idx < 0:
            return None
        age = int(ts_ms) - int(state["ts"][idx])
        if age > self.max_staleness_ms:
            return None
        bid = state["bid"][idx]
        ask = state["ask"][idx]
        return BookQuote(
            asset_id=str(asset_id),
            ts_ms=int(ts_ms),
            source="measured_book",
            best_bid=None if np.isnan(bid) else float(bid),
            best_ask=None if np.isnan(ask) else float(ask),
            bid_size=None if np.isnan(state["bid_sz"][idx]) else float(state["bid_sz"][idx]),
            ask_size=None if np.isnan(state["ask_sz"][idx]) else float(state["ask_sz"][idx]),
            age_ms=age,
        )

    def quote_or_surface(
        self,
        asset_id: str,
        ts_ms: int,
        *,
        surface,
        price: float,
        ttr_hours: float | None,
        trade_rate: float,
        category: str,
    ) -> BookQuote:
        """Measured quote, else the retired spread surface as a LABELLED fallback.

        The fallback quote carries ``source="surface_fallback"`` and only a
        symmetric half-spread around ``price`` (no depth — the surface never knew
        depth). Downstream tables must surface the source split.
        """
        measured = self.quote(asset_id, ts_ms)
        if measured is not None:
            return measured
        pred = surface.predict(price, ttr_hours, trade_rate, category)
        half = pred.half_spread_cents / 100.0
        return BookQuote(
            asset_id=str(asset_id),
            ts_ms=int(ts_ms),
            source="surface_fallback",
            best_bid=max(price - half, 0.0),
            best_ask=min(price + half, 1.0),
            bid_size=None,
            ask_size=None,
            age_ms=None,
        )
