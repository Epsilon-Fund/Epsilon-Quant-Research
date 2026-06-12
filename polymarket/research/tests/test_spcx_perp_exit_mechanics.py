"""Unit tests for the Block S6 perp-exit-mechanics pure helpers
(scripts/spcx_perp_exit_mechanics.py): time alignment, gap construction,
sustained-threshold detection, and funding sign conventions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts.spcx_perp_exit_mechanics import (
    at_or_before,
    build_gap_series,
    first_sustained_inside,
    short_funding_pnl,
)


def t(minute: int) -> datetime:
    return datetime(2026, 5, 14, 17, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)


def test_at_or_before_picks_latest_not_future():
    series = [(t(0), 100.0), (t(15), 110.0), (t(30), 120.0)]
    assert at_or_before(series, t(20)) == (t(15), 110.0)
    assert at_or_before(series, t(-1)) == (None, None)


def test_at_or_before_rejects_stale():
    series = [(t(0), 100.0)]
    assert at_or_before(series, t(60), max_stale=timedelta(minutes=30)) == (None, None)
    assert at_or_before(series, t(20), max_stale=timedelta(minutes=30)) == (t(0), 100.0)


def test_build_gap_series_aligns_and_excludes_pre_cross():
    sessions = [("s1", t(-120), t(60))]
    perp = [(t(-30), 200.0), (t(0), 210.0), (t(15), 220.0)]
    spot = [(t(0), 208.0), (t(14), 219.0)]
    gaps = build_gap_series(perp, spot, sessions, spot_start=t(0))
    # the t(-30) perp mark precedes spot_start -> excluded; t(0) pairs exactly; t(15)
    # pairs with the 1-minute-stale spot print
    assert [g["t"] for g in gaps] == [t(0), t(15)]
    assert gaps[0]["gap"] == pytest.approx(2.0)
    assert gaps[1]["gap"] == pytest.approx(1.0)


def test_first_sustained_inside_requires_consecutive_marks():
    gaps = [{"t": t(i * 15), "gap": g} for i, g in enumerate([5.0, 1.5, 3.0, 1.0, 0.5, 0.2])]
    # single dip at index1 does not count; the run starts at index 3
    assert first_sustained_inside(gaps, 2.0, n_consec=2) == t(45)
    assert first_sustained_inside(gaps, 0.1) is None


def test_short_funding_sign_convention():
    # positive rate = longs pay shorts = short EARNS
    funding = [{"t": t(0), "rate": 1e-4}, {"t": t(60), "rate": -2e-4}, {"t": t(120), "rate": 1e-4}]
    # window [t0, t120) excludes the last row
    assert short_funding_pnl(funding, t(0), t(120), price_ref=100.0) == pytest.approx(-0.01)
    # full window nets to zero
    assert short_funding_pnl(funding, t(0), t(121), price_ref=100.0) == pytest.approx(0.0)
