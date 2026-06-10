"""Pre-registered acceptance tests for the Cerebras lifecycle mapping in
scripts/spcx_cerebras_case_study.py. Pure functions only -- no network.

Encodes the task's acceptance criteria:
  - Lookahead-free: every phase-table perp/spot value is the candle AT-OR-BEFORE its event.
  - Timezones: ET = UTC-4 (EDT) in May 2026.
  - No spot pre-listing: spot is absent for all timestamps before the listing-open time.
  - Staleness guard: a data-coverage gap returns None, never a silently-stale price.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.spcx_cerebras_case_study import (
    ET_OFFSET,
    build_phase_table,
    cbrs_events,
    et_of,
    nearest_at_or_before,
)

UTC = timezone.utc


def test_et_is_utc_minus_4_in_may_2026() -> None:
    assert ET_OFFSET == timedelta(hours=-4)
    probe = datetime(2026, 5, 14, 18, 0, tzinfo=UTC)
    assert et_of(probe).hour == 14            # 18:00 UTC -> 14:00 EDT
    assert (probe - et_of(probe)) == timedelta(hours=4)


def test_nearest_at_or_before_is_lookahead_free() -> None:
    times = [datetime(2026, 5, 14, h, tzinfo=UTC) for h in (10, 11, 12)]
    vals = [100.0, 110.0, 120.0]
    # target strictly between 11:00 and 12:00 -> must pick 11:00 (never the future 12:00)
    t, v = nearest_at_or_before(times, vals, datetime(2026, 5, 14, 11, 30, tzinfo=UTC))
    assert t == times[1] and v == 110.0
    assert t <= datetime(2026, 5, 14, 11, 30, tzinfo=UTC)
    # exact match picks that candle
    t, v = nearest_at_or_before(times, vals, times[2])
    assert v == 120.0
    # before the first candle -> None (perp/spot didn't exist yet)
    t, v = nearest_at_or_before(times, vals, datetime(2026, 5, 14, 9, 0, tzinfo=UTC))
    assert t is None and v is None


def test_staleness_guard_flags_gaps() -> None:
    times = [datetime(2026, 5, 14, 10, tzinfo=UTC)]
    vals = [100.0]
    # target 90 min later with a 60-min max_gap -> stale -> None (do not reuse the old price)
    t, v = nearest_at_or_before(times, vals, datetime(2026, 5, 14, 11, 30, tzinfo=UTC),
                                max_gap=timedelta(minutes=60))
    assert t is None and v is None
    # within the gap -> returned
    t, v = nearest_at_or_before(times, vals, datetime(2026, 5, 14, 10, 30, tzinfo=UTC),
                                max_gap=timedelta(minutes=60))
    assert v == 100.0


def test_no_spot_before_listing_and_lookahead_free_table() -> None:
    listing = datetime(2026, 5, 14, 16, 59, tzinfo=UTC)
    events = [
        {"label": "pre (pricing night)", "utc": datetime(2026, 5, 13, 23, 45, tzinfo=UTC),
         "et": et_of(datetime(2026, 5, 13, 23, 45, tzinfo=UTC)), "time_confirmed": True, "note": ""},
        {"label": "post (listing open)", "utc": listing,
         "et": et_of(listing), "time_confirmed": True, "note": ""},
    ]
    perp_t = [datetime(2026, 5, 13, 20, tzinfo=UTC), datetime(2026, 5, 13, 23, 30, tzinfo=UTC),
              datetime(2026, 5, 14, 16, 45, tzinfo=UTC), datetime(2026, 5, 14, 17, 0, tzinfo=UTC)]
    perp_c = [280.0, 289.0, 376.0, 360.0]
    spot_t = [datetime(2026, 5, 14, 16, 55, tzinfo=UTC), datetime(2026, 5, 14, 17, 0, tzinfo=UTC)]
    spot_c = [385.0, 360.0]

    rows, spike = build_phase_table(events, perp_t, perp_c, spot_t, list(spot_c),
                                    offer=185.0, listing_open_utc=listing)
    pre, post = rows[0], rows[1]
    # NO spot before listing
    assert pre["spot"] is None
    # spot exists at/after listing, picked at-or-before the event (the 16:55 bar, not the 17:00)
    assert post["spot"] == 385.0
    # perp is lookahead-free at both events
    assert pre["perp"] == 289.0           # at-or-before pricing night
    assert post["perp"] == 376.0          # 16:45 bar, not the future 17:00 bar
    # basis math
    assert abs(pre["perp_offer_basis_%"] - (289.0 / 185.0 - 1) * 100) < 1e-9
    assert abs(post["perp_spot_basis_$"] - (376.0 - 385.0)) < 1e-9


def test_cbrs_events_provenance_and_tz_consistency() -> None:
    """Every event has a source, and confirmed-time events carry a real URL (not 'pending');
    every event's ET equals its UTC minus 4h."""
    for e in cbrs_events():
        assert e["source"], f"event missing source: {e['label']}"
        if e["time_confirmed"]:
            assert e["source"].startswith("http"), f"confirmed event needs a URL: {e['label']}"
        if e["utc"] is not None:
            assert (e["utc"] - e["et"]) == timedelta(hours=4)
