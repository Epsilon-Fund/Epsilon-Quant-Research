"""Tests for lib/book_costs.py — measured-book costs with labelled surface fallback."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lib.book_costs import MAX_STALENESS_MS, BookCostIndex, BookQuote


class _StubSurface:
    """Minimal SpreadSurface.predict stand-in (2c half-spread, any state)."""

    def predict(self, price, ttr_hours, trade_rate, category):
        from lib.spread_surface import SpreadPrediction

        return SpreadPrediction(half_spread_cents=2.0, source_level="stub",
                                cell_n_fills=999, raw_median_cents=2.0)


@pytest.fixture()
def state_dir(tmp_path: Path) -> Path:
    d = tmp_path / "2026-07-01" / "politics_negrisk"
    d.mkdir(parents=True)
    pd.DataFrame([
        # ts, asset, market, bid, bid_sz, ask, ask_sz, stale
        {"timestamp_ms": 1_000, "asset_id": "A", "market": "m", "best_bid": 0.40,
         "bid_size": 10.0, "best_ask": 0.42, "ask_size": 8.0, "stale": False},
        {"timestamp_ms": 2_000, "asset_id": "A", "market": "m", "best_bid": 0.41,
         "bid_size": 5.0, "best_ask": 0.43, "ask_size": 6.0, "stale": False},
        {"timestamp_ms": 3_000, "asset_id": "A", "market": "m", "best_bid": 0.41,
         "bid_size": 5.0, "best_ask": 0.43, "ask_size": 6.0, "stale": True},   # stale row
        {"timestamp_ms": 1_500, "asset_id": "B", "market": "m2", "best_bid": 0.10,
         "bid_size": 100.0, "best_ask": 0.11, "ask_size": 50.0, "stale": False},
    ]).to_parquet(d / "l1_states_000.parquet", index=False)
    return d


def test_asof_measured_quote(state_dir: Path) -> None:
    idx = BookCostIndex.load(state_dir)
    q = idx.quote("A", 2_500)
    assert q is not None and q.source == "measured_book"
    assert q.best_bid == 0.41 and q.best_ask == 0.43
    assert q.spread == pytest.approx(0.02)
    assert q.half_spread_cents == pytest.approx(1.0)
    assert q.mid == pytest.approx(0.42)
    assert q.touch_depth == pytest.approx(11.0)
    assert q.age_ms == 500


def test_staleness_gate_returns_none(state_dir: Path) -> None:
    idx = BookCostIndex.load(state_dir)
    assert idx.quote("A", 2_000 + MAX_STALENESS_MS + 1) is None   # too old
    assert idx.quote("A", 500) is None                             # before coverage
    assert idx.quote("UNKNOWN", 2_000) is None                     # unknown asset


def test_stale_rows_never_served(state_dir: Path) -> None:
    # the ts=3000 stale row must not become the served state at ts=3500
    idx = BookCostIndex.load(state_dir)
    q = idx.quote("A", 3_500)
    assert q is not None and q.age_ms == 1_500   # served from the fresh ts=2000 row


def test_asset_filtered_load(state_dir: Path) -> None:
    idx = BookCostIndex.load(state_dir, asset_ids=["B"])
    assert idx.quote("A", 2_500) is None
    qb = idx.quote("B", 1_600)
    assert qb is not None and qb.best_bid == 0.10


def test_surface_fallback_is_labelled(state_dir: Path) -> None:
    idx = BookCostIndex.load(state_dir)
    q = idx.quote_or_surface("A", 60_000, surface=_StubSurface(), price=0.40,
                             ttr_hours=24.0, trade_rate=5.0, category="politics")
    assert isinstance(q, BookQuote) and q.source == "surface_fallback"
    assert q.best_bid == pytest.approx(0.38) and q.best_ask == pytest.approx(0.42)
    assert q.bid_size is None and q.age_ms is None
    # measured wins when covered
    q2 = idx.quote_or_surface("A", 2_500, surface=_StubSurface(), price=0.40,
                              ttr_hours=24.0, trade_rate=5.0, category="politics")
    assert q2.source == "measured_book"
