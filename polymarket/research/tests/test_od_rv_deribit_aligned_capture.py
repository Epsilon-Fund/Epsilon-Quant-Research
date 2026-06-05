from __future__ import annotations

from datetime import UTC, datetime

from scripts.od_rv_deribit_aligned_capture import choose_strikes, pm_specs_for_expiry


def test_pm_specs_for_0800_expiry_include_aligned_4h_and_hourly_slugs() -> None:
    expiry = datetime(2026, 6, 3, 8, 0, tzinfo=UTC)
    specs = pm_specs_for_expiry("BTC", expiry, ["4h", "hourly"])

    by_family = {spec.family: spec for spec in specs}
    assert by_family["4h"].slug == "btc-updown-4h-1780459200"
    assert by_family["4h"].window_end == expiry
    assert by_family["hourly"].slug == "bitcoin-up-or-down-june-3-2026-3am-et"
    assert by_family["hourly"].window_end == expiry


def test_choose_strikes_gets_requested_neighbors() -> None:
    strikes = [68000, 69000, 70000, 71000, 72000, 73000, 74000]

    assert choose_strikes(strikes, 71250, each_side=2) == [70000, 71000, 72000, 73000]
    assert choose_strikes(strikes, 68000, each_side=2) == [68000, 69000, 70000, 71000]
