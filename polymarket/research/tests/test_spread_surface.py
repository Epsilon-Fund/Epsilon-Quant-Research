"""Unit tests for lib/spread_surface.py + pure helpers of the SPREAD-1 build.

Covers the parts a wrong sign or wrong bucket edge would silently corrupt:
aggressor direction, bucket edges (inclusive/exclusive boundaries), the
fallback chain and tick floor of SpreadSurface.predict, and the as-of helper's
strict (lookahead-free) mode.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.spread_surface import (  # noqa: E402
    ACTIVITY_LABELS,
    FALLBACK_LEVELS,
    TICK_FLOOR_CENTS,
    SpreadSurface,
    activity_bucket,
    aggressor_dir,
    price_bucket,
    ttr_bucket,
)
from scripts.spread_surface_build import (  # noqa: E402
    _asof_value,
    hybrid_prediction,
    sign_test_share,
    tradetime_quoted_half_spreads,
)


# ----------------------------------------------------------------------------
# aggressor sign
# ----------------------------------------------------------------------------
def test_aggressor_dir_maker_sell_means_taker_buy():
    assert aggressor_dir("SELL") == 1
    assert aggressor_dir("BUY") == -1


def test_aggressor_dir_unknown_is_zero():
    assert aggressor_dir(None) == 0
    assert aggressor_dir("") == 0
    assert aggressor_dir("weird") == 0


def test_aggressor_dir_case_insensitive():
    assert aggressor_dir("sell") == 1
    assert aggressor_dir(" buy ") == -1


# ----------------------------------------------------------------------------
# buckets
# ----------------------------------------------------------------------------
def test_price_bucket_edges():
    assert price_bucket(0.01) == "p_lt_05"
    assert price_bucket(0.05) == "p_05_15"   # left-closed at the edge
    assert price_bucket(0.149) == "p_05_15"
    assert price_bucket(0.50) == "p_35_65"
    assert price_bucket(0.95) == "p_gt_95"
    assert price_bucket(0.999) == "p_gt_95"
    assert price_bucket(1.0) == "p_gt_95"


def test_ttr_bucket_edges_and_unknown():
    assert ttr_bucket(1.0) == "ttr_lt_6h"
    assert ttr_bucket(6.0) == "ttr_6_24h"
    assert ttr_bucket(100.0) == "ttr_1_7d"
    assert ttr_bucket(700.0) == "ttr_7_30d"
    assert ttr_bucket(10_000.0) == "ttr_gt_30d"
    assert ttr_bucket(None) == "ttr_unknown"
    assert ttr_bucket(float("nan")) == "ttr_unknown"


def test_ttr_bucket_negative_maps_to_shortest():
    # fills can land microseconds after end_date on stale markets
    assert ttr_bucket(-3.0) == "ttr_lt_6h"


def test_activity_bucket_quartiles():
    brk = (2.0, 10.0, 50.0)
    assert activity_bucket(0, brk) == "act_q1"
    assert activity_bucket(2, brk) == "act_q1"     # inclusive upper edge
    assert activity_bucket(5, brk) == "act_q2"
    assert activity_bucket(50, brk) == "act_q3"
    assert activity_bucket(51, brk) == "act_q4"


# ----------------------------------------------------------------------------
# SpreadSurface predict — fallback chain + tick floor
# ----------------------------------------------------------------------------
def _toy_surface() -> SpreadSurface:
    rows = [
        # level 0 full cell: sports / p_35_65 / ttr_1_7d / act_q2 — well sampled
        dict(level=0, category="sports", price_bucket="p_35_65", ttr_bucket="ttr_1_7d",
             activity_bucket="act_q2", n_fills=500, n_markets=10, median_cents=1.8,
             p25_cents=1.0, p75_cents=3.0, frac_negative=0.2),
        # level 0 thin cell (below min_cell_fills) — must be skipped
        dict(level=0, category="sports", price_bucket="p_35_65", ttr_bucket="ttr_lt_6h",
             activity_bucket="act_q2", n_fills=5, n_markets=2, median_cents=9.0,
             p25_cents=5.0, p75_cents=12.0, frac_negative=0.1),
        # level 1 pooled over ttr
        dict(level=1, category="sports", price_bucket="p_35_65", ttr_bucket="",
             activity_bucket="act_q2", n_fills=900, n_markets=15, median_cents=2.2,
             p25_cents=1.2, p75_cents=3.5, frac_negative=0.22),
        # level 2 category x price
        dict(level=2, category="sports", price_bucket="p_35_65", ttr_bucket="",
             activity_bucket="", n_fills=2000, n_markets=30, median_cents=2.5,
             p25_cents=1.5, p75_cents=4.0, frac_negative=0.25),
        # level 2 cell whose median is BELOW the tick floor
        dict(level=2, category="crypto_4h", price_bucket="p_35_65", ttr_bucket="",
             activity_bucket="", n_fills=3000, n_markets=40, median_cents=0.2,
             p25_cents=0.05, p75_cents=0.6, frac_negative=0.4),
        # level 3 price only
        dict(level=3, category="", price_bucket="p_35_65", ttr_bucket="",
             activity_bucket="", n_fills=9000, n_markets=100, median_cents=2.0,
             p25_cents=1.0, p75_cents=3.6, frac_negative=0.3),
    ]
    table = pd.DataFrame(rows)
    breaks = {"sports": (2.0, 10.0, 50.0), "crypto_4h": (5.0, 20.0, 80.0),
              "other": (1.0, 5.0, 20.0)}
    return SpreadSurface(table, breaks, min_cell_fills=50)


def test_predict_hits_full_cell():
    s = _toy_surface()
    # trade_rate 5 -> act_q2 under sports breaks; ttr 100h -> ttr_1_7d
    p = s.predict(price=0.5, ttr_hours=100.0, trade_rate=5, category="sports")
    assert p.half_spread_cents == pytest.approx(1.8)
    assert p.source_level.startswith("level0")
    assert p.cell_n_fills == 500


def test_predict_skips_thin_cell_falls_back():
    s = _toy_surface()
    # ttr 1h -> ttr_lt_6h cell exists but n_fills=5 < 50 -> level 1 pooled
    p = s.predict(price=0.5, ttr_hours=1.0, trade_rate=5, category="sports")
    assert p.half_spread_cents == pytest.approx(2.2)
    assert p.source_level.startswith("level1")


def test_predict_tick_floor_applied():
    s = _toy_surface()
    p = s.predict(price=0.5, ttr_hours=1.0, trade_rate=100, category="crypto_4h")
    assert p.half_spread_cents == TICK_FLOOR_CENTS
    assert p.raw_median_cents == pytest.approx(0.2)


def test_predict_unknown_category_falls_to_other_then_price_level():
    s = _toy_surface()
    p = s.predict(price=0.5, ttr_hours=None, trade_rate=0, category="weather")
    # no 'weather'/'other' rows at levels 0-2 -> level 3 price-only
    assert p.half_spread_cents == pytest.approx(2.0)
    assert p.source_level.startswith("level3")


def test_predict_no_match_returns_floor_only():
    s = _toy_surface()
    p = s.predict(price=0.99, ttr_hours=None, trade_rate=0, category="other")
    assert p.half_spread_cents == TICK_FLOOR_CENTS
    assert p.source_level == "tick_floor_only"
    assert p.raw_median_cents is None


def test_fallback_levels_shape():
    # predict() builds keys positionally from these — guard the contract
    assert FALLBACK_LEVELS[0] == ("category", "price_bucket", "ttr_bucket", "activity_bucket")
    assert FALLBACK_LEVELS[-1] == ("price_bucket",)
    assert len(ACTIVITY_LABELS) == 4


def test_surface_roundtrip_csv(tmp_path):
    s = _toy_surface()
    table_rows = []
    for key, row in s._lookup.items():
        table_rows.append(row)
    pd.DataFrame(table_rows).to_csv(tmp_path / "surf.csv", index=False)
    pd.DataFrame([
        {"category": c, "act_q25": b[0], "act_q50": b[1], "act_q75": b[2]}
        for c, b in s.activity_breaks.items()
    ]).to_csv(tmp_path / "meta.csv", index=False)
    s2 = SpreadSurface.load(tmp_path / "surf.csv", tmp_path / "meta.csv")
    p1 = s.predict(0.5, 100.0, 5, "sports")
    p2 = s2.predict(0.5, 100.0, 5, "sports")
    assert p1.half_spread_cents == p2.half_spread_cents
    assert p1.source_level == p2.source_level


# ----------------------------------------------------------------------------
# as-of helper — lookahead control
# ----------------------------------------------------------------------------
def test_asof_strict_excludes_same_timestamp():
    ts = np.array([10.0, 20.0, 30.0])
    vs = np.array([1.0, 2.0, 3.0])
    assert _asof_value(ts, vs, 20.0, strict=False) == 2.0
    assert _asof_value(ts, vs, 20.0, strict=True) == 1.0   # same-ts bar excluded
    assert _asof_value(ts, vs, 5.0) is None                 # nothing yet
    assert _asof_value(ts, vs, 99.0) == 3.0


def test_half_spread_sign_worked_example():
    """The worked example from the block design: taker-buy at 0.62 with mid
    0.60 -> +2c half-spread; taker-sell at 0.58 with mid 0.60 -> +2c too;
    taker-buy BELOW mid -> negative (favourable) estimate, kept not filtered."""
    mid = 0.60
    assert aggressor_dir("SELL") * (0.62 - mid) * 100 == pytest.approx(2.0)   # buy at ask
    assert aggressor_dir("BUY") * (0.58 - mid) * 100 == pytest.approx(2.0)    # sell at bid
    assert aggressor_dir("SELL") * (0.59 - mid) * 100 == pytest.approx(-1.0)  # negative kept


# ----------------------------------------------------------------------------
# SPREAD-1b helpers — trade-time truth, sign test, hybrid arm
# ----------------------------------------------------------------------------
def test_tradetime_truth_strictly_before():
    l1_ts = np.array([10.0, 20.0, 30.0])
    half = np.array([1.0, 2.0, 3.0])
    # a print exactly at a quote's timestamp must use the PREVIOUS quote
    v, age = tradetime_quoted_half_spreads(l1_ts, half, np.array([20.0, 25.0, 40.0]))
    assert v.tolist() == [1.0, 2.0, 3.0]
    assert age.tolist() == [10.0, 5.0, 10.0]


def test_tradetime_truth_drops_prints_before_first_quote():
    l1_ts = np.array([10.0])
    half = np.array([1.5])
    v, age = tradetime_quoted_half_spreads(l1_ts, half, np.array([5.0, 10.0, 12.0]))
    # 5.0 has no prior quote; 10.0 is not strictly after; only 12.0 survives
    assert v.tolist() == [1.5]
    assert age.tolist() == [2.0]


def test_sign_test_share_excludes_ties():
    a = np.array([0.5, 1.0, 2.0, 3.0])
    b = np.array([1.0, 1.0, 1.0, 4.0])
    share, wins, nontied = sign_test_share(a, b)
    assert (wins, nontied) == (2, 3)        # the exact tie at index 1 is excluded
    assert share == pytest.approx(2 / 3)


def test_sign_test_share_all_tied_is_nan():
    share, wins, nontied = sign_test_share(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert (wins, nontied) == (0, 0)
    assert math.isnan(share)


def test_hybrid_prediction_swaps_only_flagged_cells():
    repl = {("daily_crypto", "p_65_85"): 0.6,
            ("daily_crypto", "p_gt_95"): 0.1,
            ("sports", "p_35_65"): float("nan")}
    # flagged (frac_negative > 0.4) with a replacement level -> swapped
    assert hybrid_prediction(2.0, 0.55, "daily_crypto", "p_65_85", repl) == (0.6, True)
    # below the contamination threshold -> surface kept
    assert hybrid_prediction(2.0, 0.10, "daily_crypto", "p_65_85", repl) == (2.0, False)
    # flagged but replacement missing or NaN -> surface kept
    assert hybrid_prediction(2.0, 0.90, "geopolitics", "p_35_65", repl) == (2.0, False)
    assert hybrid_prediction(2.0, 0.90, "sports", "p_35_65", repl) == (2.0, False)
    # unknown frac_negative (floor-only prediction) -> surface kept
    assert hybrid_prediction(0.5, None, "daily_crypto", "p_65_85", repl) == (0.5, False)
    # the tick floor applies to the replacement level
    assert hybrid_prediction(2.0, 0.50, "daily_crypto", "p_gt_95", repl) == (
        TICK_FLOOR_CENTS, True)


def test_predict_exposes_cell_frac_negative():
    s = _toy_surface()
    p = s.predict(price=0.5, ttr_hours=100.0, trade_rate=5, category="sports")
    assert p.cell_frac_negative == pytest.approx(0.2)
    p2 = s.predict(price=0.99, ttr_hours=None, trade_rate=0, category="other")
    assert p2.cell_frac_negative is None  # tick_floor_only has no source cell
