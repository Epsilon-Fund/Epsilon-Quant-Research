"""Pre-registered acceptance tests for the SPCX IPO convergence calculator.

These encode the four acceptance criteria from the task spec:
  (i)   when perp_base == IPO_base the naive and units-matched gaps are equal; when bases
        differ they diverge by exactly the denominator ratio.
  (ii)  locked P&L is invariant to the assumed settlement price across a sweep (direction-
        neutrality).
  (iii) liquidation flips SURVIVE->LIQUIDATED at the algebraically correct adverse move.
  (iv)  Cerebras regression fixture: short @277 vs long @185, settle @311 -> locked +92, and
        the +39% path flags LIQUIDATED at >=2x; short @340 -> +155 and SURVIVE at low leverage.
"""
from __future__ import annotations

import math

from scripts.spcx_convergence_calc import (
    PER_SHARE,
    VALUATION_PER_1E9,
    capital_deployed,
    denominator_ratio,
    fdv_neutral_contract_count,
    hedge_ratio_from_short,
    implied_fdv_usd,
    liq_adverse_frac,
    liq_buffer_summary,
    liq_price_short,
    locked_pnl_per_ipo_share,
    locked_pnl_total,
    maintenance_margin_frac,
    max_survivable_leverage,
    naive_per_share_gap,
    per_ipo_share_equiv,
    realized_pnl_per_share_at_settlement,
    residual_pnl_total,
    residual_slope_per_dollar,
    return_on_capital,
    short_contracts_from_hedge,
    simple_annualized,
    survives_move,
    total_pnl_at_close,
    units_matched_gap,
)

IPO_BASE = 13_075_865_175
OFFER = 135.0


# ----------------------------------------------------------------------------------------
# (i) units normalization: naive == matched at R=1; diverge by exactly R otherwise
# ----------------------------------------------------------------------------------------
def test_naive_equals_matched_when_bases_equal() -> None:
    mark = 159.6
    naive = naive_per_share_gap(mark, OFFER)
    matched = units_matched_gap(mark, base=IPO_BASE, units=PER_SHARE, ipo_base=IPO_BASE, offer=OFFER)
    assert denominator_ratio(IPO_BASE, IPO_BASE) == 1.0
    assert math.isclose(naive, matched, rel_tol=1e-12)


def test_naive_and_matched_diverge_by_exactly_R() -> None:
    mark = 159.6
    perp_base = 11_870_000_000
    R = denominator_ratio(IPO_BASE, perp_base)
    assert R > 1.0  # smaller perp base -> R>1

    matched = units_matched_gap(mark, base=perp_base, units=PER_SHARE, ipo_base=IPO_BASE, offer=OFFER)
    # per-IPO-share-equiv of the mark must equal mark / R exactly
    pse = per_ipo_share_equiv(mark, perp_base, PER_SHARE, IPO_BASE)
    assert math.isclose(pse, mark / R, rel_tol=1e-12)
    # the matched gap is built from mark/R, the naive from mark; their difference is mark*(1-1/R)
    naive = naive_per_share_gap(mark, OFFER)
    assert math.isclose(naive - matched, mark * (1.0 - 1.0 / R), rel_tol=1e-12)
    # and a naive 1:1 short over-hedges by exactly (R-1)
    neutral = fdv_neutral_contract_count(1000.0, IPO_BASE, perp_base, PER_SHARE, mark)
    assert math.isclose(1000.0 / neutral, R, rel_tol=1e-12)


def test_valuation_units_fdv_and_per_share() -> None:
    # vntl: price = valuation / 1e9, so mark 2025.1 => $2.0251T FDV, per-IPO-share = mark/13.076
    mark = 2025.1
    assert math.isclose(implied_fdv_usd(mark, IPO_BASE, VALUATION_PER_1E9), mark * 1e9, rel_tol=1e-12)
    pse = per_ipo_share_equiv(mark, IPO_BASE, VALUATION_PER_1E9, IPO_BASE)
    assert math.isclose(pse, mark * 1e9 / IPO_BASE, rel_tol=1e-12)
    # FDV-neutral contract count for valuation units
    neutral = fdv_neutral_contract_count(1000.0, IPO_BASE, IPO_BASE, VALUATION_PER_1E9, mark)
    assert math.isclose(neutral, 1000.0 * 1e9 / IPO_BASE, rel_tol=1e-12)


# ----------------------------------------------------------------------------------------
# (ii) locked P&L invariant to assumed settlement price (direction-neutrality)
# ----------------------------------------------------------------------------------------
def test_locked_pnl_invariant_to_settlement_price() -> None:
    perp_entry_ipo_units = 159.6
    locked = locked_pnl_per_ipo_share(perp_entry_ipo_units, OFFER)
    vals = [
        realized_pnl_per_share_at_settlement(perp_entry_ipo_units, OFFER, settle)
        for settle in [80.0, 110.0, 135.0, 167.0, 200.0, 260.0, 320.0]
    ]
    assert max(vals) - min(vals) < 1e-9          # constant across the whole sweep
    assert all(math.isclose(v, locked, rel_tol=1e-12) for v in vals)
    assert math.isclose(locked, 159.6 - 135.0, rel_tol=1e-12)


# ----------------------------------------------------------------------------------------
# (iii) liquidation flips at the algebraically correct adverse move
# ----------------------------------------------------------------------------------------
def test_liquidation_algebra_and_flip() -> None:
    mmr = 0.0  # bankruptcy model: liq adverse frac == 1/L exactly
    for L in [1.0, 2.0, 4.0, 5.0]:
        assert math.isclose(liq_adverse_frac(L, mmr), 1.0 / L, rel_tol=1e-12)

    mmr = 0.1
    L = 2.0
    f = liq_adverse_frac(L, mmr)
    assert math.isclose(f, (1.0 + 1.0 / L) / (1.0 + mmr) - 1.0, rel_tol=1e-12)
    # SURVIVE just below the boundary, LIQUIDATED just above
    assert survives_move(f - 1e-6, L, mmr) is True
    assert survives_move(f + 1e-6, L, mmr) is False
    # liq price is entry scaled by (1+f)
    assert math.isclose(liq_price_short(100.0, L, mmr), 100.0 * (1.0 + f), rel_tol=1e-12)


def test_max_survivable_leverage_inverts_liq() -> None:
    mmr = 0.1
    s = 0.39
    Lmax = max_survivable_leverage(s, mmr)
    # at exactly Lmax, the liq move equals the scenario (boundary)
    assert math.isclose(liq_adverse_frac(Lmax, mmr), s, rel_tol=1e-9)
    # slightly less leverage survives, slightly more does not
    assert survives_move(s, Lmax * 0.999, mmr) is True
    assert survives_move(s, Lmax * 1.001, mmr) is False


# ----------------------------------------------------------------------------------------
# (iv) Cerebras regression fixture
# ----------------------------------------------------------------------------------------
def test_cerebras_regression() -> None:
    long_entry = 185.0
    cbrs_max_lev = 5.0
    mmr = maintenance_margin_frac(cbrs_max_lev)  # = 0.1
    assert math.isclose(mmr, 0.1, rel_tol=1e-12)

    # short @277 vs long @185, settle @311 -> locked +92, invariant to settle
    locked = locked_pnl_per_ipo_share(277.0, long_entry)
    assert math.isclose(locked, 92.0, rel_tol=1e-12)
    assert math.isclose(
        realized_pnl_per_share_at_settlement(277.0, long_entry, 311.0), 92.0, rel_tol=1e-12)
    # invariant to a different settle
    assert math.isclose(
        realized_pnl_per_share_at_settlement(277.0, long_entry, 250.0), 92.0, rel_tol=1e-12)

    # +39% path (the $385 high from $277) LIQUIDATES the short at >=2x
    assert survives_move(0.39, leverage=2.0, mmr=mmr) is False
    # ... and at low leverage (1x) it survives
    assert survives_move(0.39, leverage=1.0, mmr=mmr) is True

    # short @340 -> locked +155, survives the +13% path (385 high from 340 is +13.2%) at low lev
    assert math.isclose(locked_pnl_per_ipo_share(340.0, long_entry), 155.0, rel_tol=1e-12)
    assert survives_move(0.13, leverage=1.0, mmr=mmr) is True
    assert survives_move(0.39, leverage=1.0, mmr=mmr) is True


# ----------------------------------------------------------------------------------------
# (v) partial-hedge / over-hedge decomposition (the "not cover 1:1" generalization)
# ----------------------------------------------------------------------------------------
def test_hedge_ratio_decomposition() -> None:
    N, basis, offer = 1000.0, 24.92, 135.0
    # locked: 0 at h=0, full at h=1, scales linearly
    assert locked_pnl_total(0.0, N, basis) == 0.0
    assert math.isclose(locked_pnl_total(1.0, N, basis), N * basis, rel_tol=1e-12)
    assert math.isclose(locked_pnl_total(0.5, N, basis), 0.5 * N * basis, rel_tol=1e-12)

    # residual sign: under-hedge (h<1) is net LONG (slope>0), over-hedge (h>1) is net SHORT (slope<0)
    assert residual_slope_per_dollar(0.5, N) > 0
    assert residual_slope_per_dollar(1.0, N) == 0.0
    assert residual_slope_per_dollar(1.5, N) < 0
    assert math.isclose(residual_slope_per_dollar(0.5, N), 0.5 * N, rel_tol=1e-12)

    # total reduces correctly: h=0 -> pure long; h=1 -> locked, P_close cancels
    for px in [100.0, 167.0, 222.0]:
        assert math.isclose(total_pnl_at_close(0.0, N, basis, px, offer), N * (px - offer), rel_tol=1e-12)
        assert math.isclose(total_pnl_at_close(1.0, N, basis, px, offer), N * basis, rel_tol=1e-12)

    # over-hedge LOSES on a melt-up: h=1.5, close well above offer -> total can go negative
    hi = total_pnl_at_close(1.5, N, basis, 222.0, offer)
    assert hi < total_pnl_at_close(1.5, N, basis, 100.0, offer)  # net short: worse as price rises

    # hedge ratio from a chosen short count, and its inverse
    neutral = 76.48
    assert math.isclose(hedge_ratio_from_short(short_contracts_from_hedge(0.7, neutral), neutral), 0.7, rel_tol=1e-12)


def test_capital_and_return_on_capital() -> None:
    long_notional = 135_000.0
    short_margin = 159_920.0  # unlevered: margin == notional
    cap = capital_deployed(long_notional, short_margin)
    assert math.isclose(cap, 294_920.0, rel_tol=1e-12)
    roc = return_on_capital(24_884.0, cap)
    assert math.isclose(roc, 24_884.0 / 294_920.0, rel_tol=1e-12)
    # simple (linear) annualization, NOT compounded; 72h -> x (8760/72)
    ann = simple_annualized(roc, 72.0)
    assert math.isclose(ann, roc * (8760.0 / 72.0), rel_tol=1e-12)
    # sanity: compounding would be absurdly larger -> we must NOT be doing it
    compounded = (1.0 + roc) ** (8760.0 / 72.0) - 1.0
    assert compounded > 100 * ann  # compounded one-shot is nonsense; confirm we report the small one


def test_liq_buffer_bands() -> None:
    # liq price fixed at 240; SAFE far below, WARN within alert%, BREACH at/through liq
    liq = 240.0
    assert liq_buffer_summary(160.0, liq, alert_pct=0.10)["band"] == "SAFE"   # +50% buffer
    assert liq_buffer_summary(228.0, liq, alert_pct=0.10)["band"] == "WARN"   # +5.3% buffer
    assert liq_buffer_summary(240.0, liq, alert_pct=0.10)["band"] == "BREACH" # at liq (0%)
    assert liq_buffer_summary(245.0, liq, alert_pct=0.10)["band"] == "BREACH" # past liq (<0)
    # boundary: buffer just under alert% -> WARN; just over -> SAFE (avoid the exact FP knife-edge)
    assert liq_buffer_summary(liq / 1.095, liq, alert_pct=0.10)["band"] == "WARN"  # ~9.5% buffer
    assert liq_buffer_summary(liq / 1.105, liq, alert_pct=0.10)["band"] == "SAFE"  # ~10.5% buffer


def test_no_lookahead_realized_only_in_sweep() -> None:
    """The decision-facing locked basis must not depend on the realized settlement price;
    realized_pnl_*_at_settlement is the ONLY function that takes a settle price, and it is
    only used for the invariance sweep."""
    import inspect

    from scripts import spcx_convergence_calc as mod

    src = inspect.getsource(mod.ContractEval.__post_init__)
    # the decision/verdict path must never reference a realized settlement price
    assert "realized_pnl_per_share_at_settlement" not in src
    assert "settle_close" not in src
