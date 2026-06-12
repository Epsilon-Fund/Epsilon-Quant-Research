"""Unit tests for the Block S2 tape-study pure functions (scripts/spcx_ipo_unwind_tape_study.py).

Covers: cross detection, anchored VWAP, fade/VWAP-loss timing, volume buckets, the three
unwind policies (incl. a lookahead-free truncation check: every scheduled execution price must
be computable from the tape up to that minute), and the --meltup-dist round-trip into
spcx_convergence_calc's parser.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from scripts.spcx_ipo_unwind_tape_study import (
    OBSERVE_MIN,
    TRANCHE_SCHEDULE,
    anchored_vwap,
    bootstrap_mean_ci,
    fade_onset_minutes,
    first_print_index,
    meltup_dist_string,
    policy_sell_at,
    policy_tranche,
    policy_twap_from_cross,
    rolling_vwap_peak,
    twap,
    volume_buckets_30m,
    vwap_loss_minutes,
)
from scripts.spcx_convergence_calc import _parse_weighted_map, expected_meltup_move


def bar(minute: int, price: float, vol: float = 100.0) -> dict:
    t0 = datetime(2026, 5, 14, 17, 0, tzinfo=timezone.utc)
    return {"t": t0 + timedelta(minutes=minute), "o": price, "h": price, "l": price,
            "c": price, "v": vol}


def tape(prices: list[float], vols: list[float] | None = None) -> list[dict]:
    vols = vols or [100.0] * len(prices)
    return [bar(i, p, v) for i, (p, v) in enumerate(zip(prices, vols))]


# ---------------------------------------------------------------- cross detection
def test_first_print_skips_offer_placeholder_bars():
    bars = [bar(0, 185.0, 0), bar(1, 185.0, 0), bar(2, 350.0, 5e6), bar(3, 360.0, 1e6)]
    assert first_print_index(bars, offer=185.0) == 2


def test_first_print_requires_volume_and_price():
    # price off the placeholder but zero volume must NOT count as the cross
    bars = [bar(0, 250.0, 0), bar(1, 250.0, 1000.0)]
    assert first_print_index(bars, offer=185.0) == 1


def test_first_print_raises_when_absent():
    with pytest.raises(ValueError):
        first_print_index([bar(0, 185.0, 0)], offer=185.0)


# ---------------------------------------------------------------- anchored VWAP
def test_anchored_vwap_constant_price_is_flat():
    av = anchored_vwap(tape([100.0] * 10))
    assert np.allclose(av, 100.0)


def test_anchored_vwap_weights_by_volume():
    # one heavy bar at 100, one light at 200 -> VWAP pulled toward 100
    av = anchored_vwap(tape([100.0, 200.0], vols=[900.0, 100.0]))
    assert av[1] == pytest.approx(110.0)


def test_anchored_vwap_is_lookahead_free():
    full = tape([100, 110, 120, 90, 80])
    av_full = anchored_vwap(full)
    for i in range(1, len(full) + 1):
        av_trunc = anchored_vwap(full[:i])
        assert av_trunc[i - 1] == pytest.approx(av_full[i - 1])


# ---------------------------------------------------------------- fade / VWAP-loss timing
def test_fade_onset_uses_running_high_only():
    # high 100 at t0, drop to 94 (=6% below running high) at minute 3
    bars = tape([100, 99, 98, 94])
    assert fade_onset_minutes(bars, drop_frac=0.05) == 3


def test_fade_onset_none_when_no_fade():
    assert fade_onset_minutes(tape([100, 101, 102, 103])) is None


def test_vwap_loss_requires_persistence():
    # dips below anchored VWAP once then reclaims -> no loss with persist=3
    prices = [100, 110, 105, 112, 111, 111]
    assert vwap_loss_minutes(tape(prices), persist=3) is None
    # sustained breakdown -> first minute of the persistent run
    prices2 = [100, 110, 90, 85, 80, 75]
    m = vwap_loss_minutes(tape(prices2), persist=3)
    assert m == 2


# ---------------------------------------------------------------- volume buckets
def test_volume_buckets_sum_to_one_and_bucket_correctly():
    bars = tape([100.0] * 65, vols=[1.0] * 65)  # 65 minutes -> buckets 0,30,60
    buckets = volume_buckets_30m(bars)
    assert [k for k, _ in buckets] == [0, 30, 60]
    assert sum(v for _, v in buckets) == pytest.approx(1.0)
    assert dict(buckets)[0] == pytest.approx(30 / 65)
    assert dict(buckets)[60] == pytest.approx(5 / 65)


# ---------------------------------------------------------------- policies
def test_twap_window_and_overflow():
    bars = tape([10, 20, 30, 40])
    assert twap(bars, 1, 3) == pytest.approx(25.0)
    # window entirely beyond the tape -> executes at the last print
    assert twap(bars, 100, None) == pytest.approx(40.0)


def test_tranche_weights_sum_to_one():
    assert sum(w for _, _, w in TRANCHE_SCHEDULE) == pytest.approx(1.0)


def test_policies_on_monotone_fade_rank_by_mean_execution_minute():
    # On a strictly fading tape, earlier average execution = strictly better. Mean execution
    # minutes: C = 15, A ~ 111 (0.4*37.5 + 0.4*120 + 0.2*240), B ~ 149.5 -> C > A > B.
    prices = [300 - i for i in range(300)]
    bars = tape(prices)
    assert policy_sell_at(bars) > policy_tranche(bars) > policy_twap_from_cross(bars)


def test_policies_on_monotone_rally_rank_reversed():
    # mirror image: latest average execution wins -> B > A > C
    prices = [100 + i for i in range(300)]
    bars = tape(prices)
    assert policy_twap_from_cross(bars) > policy_tranche(bars) > policy_sell_at(bars)


def test_policy_executions_are_lookahead_free():
    """Every scheduled execution must be reproducible from the tape truncated at its own
    minute -- i.e. no policy price depends on bars after the execution window."""
    rng = np.random.default_rng(0)
    prices = list(200 + np.cumsum(rng.normal(0, 1, 240)))
    bars = tape(prices)
    # C: price at +OBSERVE_MIN equals the same computed on a tape cut right after it
    assert policy_sell_at(bars) == policy_sell_at(bars[: OBSERVE_MIN + 1])
    # A: each tranche's TWAP equals the TWAP on the tape truncated at the window end
    for s, e, _ in TRANCHE_SCHEDULE:
        cut = bars[: e] if e is not None else bars
        assert twap(bars, s, e) == pytest.approx(twap(cut, s, e))


# ---------------------------------------------------------------- meltup dist round-trip
def test_meltup_dist_string_parses_into_s1_calculator():
    moves = [0.184, 0.300, 0.466, 0.532, 0.700, 1.081]
    s = meltup_dist_string(moves)
    parsed = _parse_weighted_map(s)
    assert [m for m, _ in parsed] == pytest.approx(moves, abs=1e-3)
    assert all(w == 1.0 for _, w in parsed)
    # equal weights -> expected move is the plain mean
    assert expected_meltup_move(parsed) == pytest.approx(np.mean(moves), abs=1e-3)


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_ci_contains_mean_and_is_ordered():
    mean, lo, hi = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert lo <= mean <= hi
    assert mean == pytest.approx(3.5)
