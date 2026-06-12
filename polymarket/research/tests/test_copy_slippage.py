"""Unit tests for lib/copy_slippage.py (Block SPREAD-2).

Covers the pieces a wrong sign / wrong gate would silently corrupt:
  * the category gate (politics_negrisk excluded; validated set kept),
  * the maker full-spread doubling + floor/cap,
  * the SPREAD-1b bounce swap on contaminated cells,
  * the drift-vs-spread split and its drift flag,
  * lookahead-free MTM (no future quote ever marks a past grid point),
  * the K5 category taxonomy (neg_risk required for politics_negrisk).
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

from lib.copy_slippage import (  # noqa: E402
    SURFACE_FALLBACK_CAP_C,
    SURFACE_FALLBACK_FLOOR_C,
    SURFACE_VALIDATED_CATEGORIES,
    apply_fallback_price,
    asof_mid_before,
    decompose_next_fill_slippage,
    k5_category,
    leader_vs_mid_cents,
    mtm_equity_curve,
    mtm_equity_curve_fast,
    reprice_fallback_rows,
    surface_fallback_cents,
)
from lib.spread_surface import SpreadSurface  # noqa: E402


# ----------------------------------------------------------------------------
# toy surface (mirrors tests/test_spread_surface.py shape, adds a contaminated cell)
# ----------------------------------------------------------------------------
def _toy_surface() -> SpreadSurface:
    rows = [
        dict(level=2, category="crypto_4h", price_bucket="p_15_35", ttr_bucket="",
             activity_bucket="", n_fills=3000, n_markets=40, median_cents=2.5,
             p25_cents=1.0, p75_cents=4.0, frac_negative=0.10),
        # contaminated crypto cell (frac_negative > 0.4) -> bounce swap path
        dict(level=2, category="daily_crypto", price_bucket="p_65_85", ttr_bucket="",
             activity_bucket="", n_fills=3000, n_markets=40, median_cents=0.5,
             p25_cents=0.0, p75_cents=1.0, frac_negative=0.55),
        dict(level=2, category="other", price_bucket="p_35_65", ttr_bucket="",
             activity_bucket="", n_fills=2000, n_markets=30, median_cents=1.0,
             p25_cents=0.5, p75_cents=2.0, frac_negative=0.20),
        dict(level=3, category="", price_bucket="p_35_65", ttr_bucket="",
             activity_bucket="", n_fills=9000, n_markets=100, median_cents=1.0,
             p25_cents=0.5, p75_cents=2.0, frac_negative=0.3),
    ]
    breaks = {"crypto_4h": (5, 20, 80), "daily_crypto": (5, 20, 80),
              "other": (1, 5, 20)}
    return SpreadSurface(pd.DataFrame(rows), breaks, min_cell_fills=50)


# ----------------------------------------------------------------------------
# category gate + maker doubling + floor/cap
# ----------------------------------------------------------------------------
def test_politics_negrisk_excluded_keeps_flat3c():
    s = _toy_surface()
    q = surface_fallback_cents(s, "politics_negrisk", 0.5, 48.0, 0, leader_is_maker=True)
    assert q.source == "flat3c"
    assert q.cents == pytest.approx(3.0)


def test_validated_set_is_exactly_the_six():
    assert SURFACE_VALIDATED_CATEGORIES == frozenset(
        {"crypto_4h", "daily_crypto", "geopolitics", "sports", "tech", "other"})
    assert "politics_negrisk" not in SURFACE_VALIDATED_CATEGORIES


def test_maker_full_spread_taker_half():
    s = _toy_surface()
    taker = surface_fallback_cents(s, "crypto_4h", 0.25, 2.0, 0, leader_is_maker=False)
    maker = surface_fallback_cents(s, "crypto_4h", 0.25, 2.0, 0, leader_is_maker=True)
    assert taker.cents == pytest.approx(2.5)         # half-spread
    assert maker.cents == pytest.approx(5.0)         # full spread (2x)
    assert taker.source == "surface_fallback"


def test_floor_and_cap_applied():
    s = _toy_surface()
    # 'other' p_35_65 half = 1.0c, taker -> 1.0c (above floor)
    # force the cap: a maker on a wide cell would exceed 8c only if half>4; use a
    # synthetic high-half cell via the price-only level is 1.0 -> instead test floor.
    # floor: a tiny half still floors to 0.5c
    s2 = SpreadSurface(pd.DataFrame([dict(level=3, category="", price_bucket="p_35_65",
        ttr_bucket="", activity_bucket="", n_fills=9000, n_markets=100,
        median_cents=0.05, p25_cents=0.0, p75_cents=0.1, frac_negative=0.0)]),
        {"other": (1, 5, 20)}, min_cell_fills=50)
    q = surface_fallback_cents(s2, "other", 0.5, None, 0, leader_is_maker=False)
    assert q.cents == pytest.approx(SURFACE_FALLBACK_FLOOR_C)   # floored


def test_cap_applied_on_wide_maker():
    rows = [dict(level=2, category="other", price_bucket="p_05_15", ttr_bucket="",
                 activity_bucket="", n_fills=2000, n_markets=30, median_cents=6.0,
                 p25_cents=4.0, p75_cents=9.0, frac_negative=0.1)]
    s = SpreadSurface(pd.DataFrame(rows), {"other": (1, 5, 20)}, min_cell_fills=50)
    q = surface_fallback_cents(s, "other", 0.10, None, 0, leader_is_maker=True)
    assert q.cents == pytest.approx(SURFACE_FALLBACK_CAP_C)     # 2*6=12 capped to 8


def test_contaminated_cell_uses_bounce():
    s = _toy_surface()
    bounce = {("daily_crypto", "p_65_85"): 0.8}
    # daily_crypto p_65_85 cell has frac_negative 0.55 > 0.4 -> bounce 0.8c used
    q = surface_fallback_cents(s, "daily_crypto", 0.75, 1.0, 0, leader_is_maker=False,
                               bounce_lookup=bounce)
    assert q.used_bounce is True
    assert q.cents == pytest.approx(0.8)
    # without a bounce lookup, the surface median (0.5c) is used, no swap
    q2 = surface_fallback_cents(s, "daily_crypto", 0.75, 1.0, 0, leader_is_maker=False)
    assert q2.used_bounce is False


def test_apply_fallback_price_sign():
    assert apply_fallback_price(0.60, 3.0, "BUY") == pytest.approx(0.63)
    assert apply_fallback_price(0.60, 3.0, "SELL") == pytest.approx(0.57)


# ----------------------------------------------------------------------------
# leader_vs_mid + drift/spread split
# ----------------------------------------------------------------------------
def test_leader_vs_mid_sign():
    assert leader_vs_mid_cents(0.62, 0.60, "BUY") == pytest.approx(2.0)   # paid up
    assert leader_vs_mid_cents(0.58, 0.60, "SELL") == pytest.approx(2.0)  # sold down
    assert leader_vs_mid_cents(0.62, None, "BUY") is None


def test_decompose_drift_flag():
    # copy 3c above mid, predicted half 2c -> drift 1c, flagged
    sp = decompose_next_fill_slippage(0.63, 0.60, 2.0, "BUY")
    assert sp.spread_c == pytest.approx(2.0)
    assert sp.drift_c == pytest.approx(1.0)
    assert sp.is_drift is True
    # copy only 1c above mid, half 2c -> negative drift, not flagged
    sp2 = decompose_next_fill_slippage(0.61, 0.60, 2.0, "BUY")
    assert sp2.drift_c == pytest.approx(-1.0)
    assert sp2.is_drift is False
    # sell side
    sp3 = decompose_next_fill_slippage(0.57, 0.60, 1.0, "SELL")
    assert sp3.copy_vs_mid_c == pytest.approx(3.0)
    assert sp3.drift_c == pytest.approx(2.0)
    assert decompose_next_fill_slippage(0.63, float("nan"), 2.0, "BUY") is None


# ----------------------------------------------------------------------------
# K5 category — neg_risk required for politics_negrisk
# ----------------------------------------------------------------------------
def test_k5_category_taxonomy():
    df = pd.DataFrame({
        "slug": ["btc-updown-4h-1700000000", "trump-wins-2024", "trump-wins-2024",
                 "iran-israel-ceasefire", "openai-launches-gpt6", "nba-finals-2026",
                 "some-random-market"],
        "question": ["", "will trump win", "will trump win", "", "", "", ""],
        "neg_risk": [False, True, False, False, False, False, False],
    })
    cats = k5_category(df, "slug", "question", "neg_risk").tolist()
    assert cats[0] == "crypto_4h"
    assert cats[1] == "politics_negrisk"   # neg_risk True
    assert cats[2] != "politics_negrisk"   # SAME slug but neg_risk False -> not negrisk
    assert cats[3] == "geopolitics"
    assert cats[4] == "tech"
    assert cats[5] == "sports"
    assert cats[6] == "other"


# ----------------------------------------------------------------------------
# as-of mid + lookahead-free MTM
# ----------------------------------------------------------------------------
def test_asof_mid_strictly_before_and_age_cap():
    ts = np.array([10.0, 20.0, 30.0])
    p = np.array([0.5, 0.6, 0.7])
    m, age = asof_mid_before(ts, p, 25.0)
    assert (m, age) == (0.6, 5.0)
    # exactly at a bar -> strictly-before excludes it
    m2, _ = asof_mid_before(ts, p, 20.0)
    assert m2 == 0.5
    # nothing before
    assert asof_mid_before(ts, p, 5.0) == (None, None)
    # age cap
    assert asof_mid_before(ts, p, 100.0, max_age_s=5.0) == (None, None)


def test_mtm_lookahead_free_and_resolution():
    # one BUY position: entry day1 @0.50, resolves day3 @1.0, qty 100 tokens
    positions = pd.DataFrame([{
        "trade_timestamp": "2026-01-01T00:00:00Z",
        "resolution_date": "2026-01-03T00:00:00Z",
        "outcome_token_id": "tokA", "leader_direction": "buy",
        "copy_price": 0.50, "copy_token_amount": 100.0, "position_resolution": 1.0,
    }])
    # mid jumps to 0.90 only AFTER day2 noon; a day-1/day-2 grid point must NOT see it
    e1 = pd.Timestamp("2026-01-01", tz="UTC").timestamp()
    e2 = pd.Timestamp("2026-01-02", tz="UTC").timestamp()
    mids = {"tokA": [(e1, 0.50), (e2 + 43200, 0.90)]}  # 0.90 at day2 noon
    res = mtm_equity_curve(positions, mids)
    eq = res.equity.set_index(res.equity["date"].dt.strftime("%Y-%m-%d"))["mtm_equity"]
    # day1: marked at 0.50 -> cash -50 + 100*0.50 = 0
    assert eq["2026-01-01"] == pytest.approx(0.0, abs=1e-6)
    # day2 00:00: still 0.50 (the 0.90 quote is at noon, AFTER this grid point) -> 0
    assert eq["2026-01-02"] == pytest.approx(0.0, abs=1e-6)
    # day3: resolved at 1.0 -> cash -50 + 100*1.0 = +50
    assert eq["2026-01-03"] == pytest.approx(50.0, abs=1e-6)


def test_mtm_empty_ledger():
    res = mtm_equity_curve(pd.DataFrame(), {})
    assert res.equity.empty
    assert res.sharpe_daily_ann == 0.0
    assert res.max_drawdown_usd == 0.0


# ----------------------------------------------------------------------------
# reprice_fallback_rows — shared core used by the 3 evaluators + the driver
# ----------------------------------------------------------------------------
def test_reprice_fallback_rows_only_touches_fallback_and_gates_category():
    s = _toy_surface()
    df = pd.DataFrame({
        # row0: next_fill (untouched); row1: fallback crypto_4h maker (validated, full spread);
        # row2: fallback politics_negrisk (excluded -> flat3c); row3: fallback other taker
        "k5_category": ["crypto_4h", "crypto_4h", "politics_negrisk", "other"],
        "price": [0.25, 0.25, 0.50, 0.50],
        "ttr_h": [2.0, 2.0, 48.0, 48.0],
        "direction": ["BUY", "BUY", "SELL", "BUY"],
        "is_maker": [True, True, True, False],
        "leader_price": [0.60, 0.60, 0.40, 0.50],
        "is_fb": [False, True, True, True],
    })
    out = reprice_fallback_rows(
        df, s, None, price_col="price", ttr_h_col="ttr_h", dir_col="direction",
        maker_col=df["is_maker"], leader_price_col="leader_price",
        is_fallback=df["is_fb"], category_col="k5_category")
    # row0 next_fill: untouched
    assert out.loc[0, "sf_source"] == "next_fill"
    assert pd.isna(out.loc[0, "sf_copy_price"])
    # row1 crypto_4h maker fallback: full spread (2*2.5=5c) BUY -> 0.60+0.05
    assert out.loc[1, "sf_source"] == "surface_fallback"
    assert out.loc[1, "sf_fallback_cents"] == pytest.approx(5.0)
    assert out.loc[1, "sf_copy_price"] == pytest.approx(0.65)
    # row2 politics_negrisk fallback: excluded -> flat3c, SELL -> 0.40-0.03
    assert out.loc[2, "sf_source"] == "flat3c"
    assert out.loc[2, "sf_copy_price"] == pytest.approx(0.37)
    # flat3c comparison column is always populated on fallback rows
    assert out.loc[3, "flat3c_copy_price"] == pytest.approx(0.53)  # other BUY +3c


def test_mtm_fast_matches_slow_reference():
    """The vectorized curve must reproduce the row-loop reference on a mixed
    ledger (two tokens, buy+sell, staggered entries/resolutions, quotes that
    move mid-holding, quotes preceding all fills so the leading-edge fallback
    paths coincide)."""
    e = lambda s: pd.Timestamp(s, tz="UTC").timestamp()  # noqa: E731
    positions = pd.DataFrame([
        {"trade_timestamp": "2026-01-01T06:00:00Z", "resolution_date": "2026-01-05T00:00:00Z",
         "outcome_token_id": "A", "leader_direction": "buy",
         "copy_price": 0.40, "copy_token_amount": 100.0, "position_resolution": 1.0},
        {"trade_timestamp": "2026-01-02T12:00:00Z", "resolution_date": "2026-01-04T00:00:00Z",
         "outcome_token_id": "B", "leader_direction": "sell",
         "copy_price": 0.70, "copy_token_amount": 50.0, "position_resolution": 0.0},
        {"trade_timestamp": "2026-01-03T01:00:00Z", "resolution_date": "2026-01-05T00:00:00Z",
         "outcome_token_id": "A", "leader_direction": "sell",
         "copy_price": 0.55, "copy_token_amount": 40.0, "position_resolution": 1.0},
    ])
    mids = {"A": [(e("2026-01-01T00:00"), 0.40), (e("2026-01-02T03:00"), 0.50),
                  (e("2026-01-03T18:00"), 0.60)],
            "B": [(e("2026-01-01T00:00"), 0.70), (e("2026-01-03T09:00"), 0.55)]}
    slow = mtm_equity_curve(positions, mids)

    dirn = positions["leader_direction"].str.upper().to_numpy()
    qty = positions["copy_token_amount"].to_numpy() * np.where(dirn == "BUY", 1.0, -1.0)
    fills = pd.DataFrame({
        "entry_ts": positions["trade_timestamp"], "res_ts": positions["resolution_date"],
        "token": positions["outcome_token_id"], "qty": qty,
        "cash": -qty * positions["copy_price"].to_numpy(),
        "res_price": positions["position_resolution"],
    })
    fast = mtm_equity_curve_fast(fills, mids)
    assert len(fast.equity) == len(slow.equity)
    np.testing.assert_allclose(fast.equity["mtm_equity"], slow.equity["mtm_equity"], atol=1e-9)
    np.testing.assert_allclose(fast.equity["realized"], slow.equity["realized"], atol=1e-9)
    assert fast.sharpe_daily_ann == pytest.approx(slow.sharpe_daily_ann)
    assert fast.max_drawdown_usd == pytest.approx(slow.max_drawdown_usd)


def test_mtm_fast_unresolved_position_stays_open_marked():
    # one BUY, never resolves (res_ts NaT) -> marked to ffilled mid through grid end
    e = lambda s: pd.Timestamp(s, tz="UTC").timestamp()  # noqa: E731
    fills = pd.DataFrame({
        "entry_ts": ["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"],
        "res_ts": [pd.NaT, pd.Timestamp("2026-01-04T00:00:00Z", tz="UTC")],
        "token": ["A", "A"], "qty": [100.0, 50.0],
        "cash": [-50.0, -30.0],   # buys at 0.50 and 0.60
        "res_price": [float("nan"), 1.0],
    })
    mids = {"A": [(e("2026-01-01T00:00"), 0.50), (e("2026-01-02T06:00"), 0.80)]}
    res = mtm_equity_curve_fast(fills, mids)
    eq = res.equity.set_index(res.equity["date"].dt.strftime("%Y-%m-%d"))
    # day1: open fill1 at 0.50 mark -> -50 + 100*0.5 = 0
    assert eq.loc["2026-01-01", "mtm_equity"] == pytest.approx(0.0)
    # day4: fill2 realized (-30 + 50*1.0 = +20); fill1 still open at 0.80 -> -50+80=30
    assert eq.loc["2026-01-04", "mtm_equity"] == pytest.approx(50.0)
    assert eq.loc["2026-01-04", "realized"] == pytest.approx(20.0)
    assert eq.loc["2026-01-04", "n_open"] == 1


def test_reprice_fallback_rows_handles_numpy_masks():
    s = _toy_surface()
    df = pd.DataFrame({
        "k5_category": ["other"], "price": [0.5], "ttr_h": [10.0],
        "direction": ["BUY"], "leader_price": [0.5],
    })
    out = reprice_fallback_rows(
        df, s, None, price_col="price", ttr_h_col="ttr_h", dir_col="direction",
        maker_col=np.array([False]), leader_price_col="leader_price",
        is_fallback=np.array([True]), category_col="k5_category")
    assert out.loc[0, "sf_source"] == "surface_fallback"
