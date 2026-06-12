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


# ========================================================================================
# Block S1 — hedge grid + basis-decay fit + pre-hedge timing rule
# ========================================================================================
from datetime import datetime, timedelta, timezone  # noqa: E402

from scripts.spcx_convergence_calc import (  # noqa: E402
    DecayFit,
    NodeDecision,
    breakeven_net_basis,
    build_decision_table,
    build_hedge_grid,
    expected_meltup_move,
    fill_price_axis,
    fit_premium_decay,
    margin_cap_shares,
    naked_loss_per_share,
    net_basis_per_share,
    prehedge_ev_per_share,
    prehedge_threshold,
    shares_requested,
    should_hedge,
)

UTC = timezone.utc


# ----------------------------------------------------------------------------------------
# S1(a): grid axes + sizing math + a hand-computed cell
# ----------------------------------------------------------------------------------------
def test_fill_price_axis_has_step5_and_both_corners() -> None:
    axis = fill_price_axis()
    assert axis == [135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 162.0]
    assert axis[0] == 135.0 and axis[-1] == 162.0  # corners always present


def test_shares_requested_and_margin_cap() -> None:
    # EUR 10k at EURUSD 1.15 = $11,500; at $135 -> 85.185 shares requested
    assert math.isclose(shares_requested(10_000.0, 1.15, 135.0), 11_500.0 / 135.0, rel_tol=1e-12)
    # margin cap: $2,300 at 1.5x over (1 contract/share x $160) -> 21.5625 shares
    assert math.isclose(margin_cap_shares(2_300.0, 1.5, 160.0, 1.0), 21.5625, rel_tol=1e-12)
    # zero margin -> zero capacity, everywhere (trivial corner of the rule)
    assert margin_cap_shares(0.0, 1.5, 160.0, 1.0) == 0.0
    # valuation-style contract: more $ of contracts per hedged share scales the cap down
    assert margin_cap_shares(2_300.0, 1.0, 160.0, 2.0) == 0.5 * margin_cap_shares(
        2_300.0, 1.0, 160.0, 1.0)


def test_hedge_grid_hand_computed_cell_and_cap_binding() -> None:
    # mark 160 (per-share, R=1 -> per-IPO-share-equiv 160), EURUSD 1.15, margin EUR2000
    cells = build_hedge_grid(
        per_ipo_share_mark=160.0, mark=160.0, contracts_per_share=1.0,
        funding_hourly=0.0, hours_to_settle=48.0, fee_bps=10.0, fee_sides=1.0,
        subscription_eur=10_000.0, eurusd=1.15, margin_eur=2_000.0,
        fill_prices=[135.0, 162.0], fill_fracs=[0.25, 1.0], leverages=[1.5])
    by = {(c.fill_price, c.fill_frac): c for c in cells}

    c = by[(135.0, 0.25)]
    # requested = 11500/135 = 85.185; filled 25% = 21.296; cap = 2300*1.5/160 = 21.5625
    assert math.isclose(c.shares_requested, 85.18518518518519, rel_tol=1e-12)
    assert math.isclose(c.cap_shares, 21.5625, rel_tol=1e-12)
    assert math.isclose(c.hedged_shares, 21.296296296296298, rel_tol=1e-12)  # fill < cap
    assert math.isclose(c.residual_shares, 0.0, abs_tol=1e-12)
    # locked: 21.296 * (160-135) = 532.41 gross; fee = notional 3407.4 * 10bps = 3.41
    assert math.isclose(c.basis_per_share, 25.0, rel_tol=1e-12)
    assert math.isclose(c.short_notional, c.hedged_shares * 160.0, rel_tol=1e-12)
    assert math.isclose(c.locked_net,
                        c.hedged_shares * 25.0 - c.short_notional * 10.0 / 1e4, rel_tol=1e-12)
    # ROC on hedge-sleeve capital = hedged*135 + margin (notional/1.5)
    cap_dep = c.hedged_shares * 135.0 + c.short_notional / 1.5
    assert math.isclose(c.capital, cap_dep, rel_tol=1e-12)
    assert math.isclose(c.roc, c.locked_net / cap_dep, rel_tol=1e-12)

    # 100% fill: the margin ceiling binds; residual = filled - cap
    c100 = by[(135.0, 1.0)]
    assert math.isclose(c100.hedged_shares, 21.5625, rel_tol=1e-12)
    assert math.isclose(c100.residual_shares, c100.shares_filled - 21.5625, rel_tol=1e-12)

    # fill price ABOVE the mark -> negative basis -> negative locked net
    c162 = by[(162.0, 0.25)]
    assert c162.basis_per_share < 0 and c162.locked_net < 0


# ----------------------------------------------------------------------------------------
# S1(b): decay fit recovers a synthetic decay; the fit cannot look past its cutoff
# ----------------------------------------------------------------------------------------
def _synthetic_candles(b_per_day: float, prem0: float = 45.0, offer: float = 135.0,
                       n_hours: int = 480, seed: int = 7):
    import random

    rng = random.Random(seed)
    t0 = datetime(2026, 5, 17, 22, 0, tzinfo=UTC)
    times, closes = [], []
    for i in range(n_hours):
        t = t0 + timedelta(hours=i)
        prem = prem0 * math.exp(b_per_day * (i / 24.0)) * math.exp(rng.gauss(0.0, 0.01))
        times.append(t)
        closes.append(offer + prem)
    return times, closes


def test_decay_fit_recovers_synthetic_slope() -> None:
    b_true = -0.02  # -2%/day level-premium decay
    times, closes = _synthetic_candles(b_true)
    cutoff = times[-1] + timedelta(hours=2)
    fit = fit_premium_decay(times, closes, 135.0, cutoff_utc=cutoff, boot_n=200)
    assert fit.n_used == len(times)
    assert math.isclose(fit.b, b_true, abs_tol=0.004)
    # bootstrap CI brackets the true slope
    bs = sorted(b for _, b in fit.boot_ab)
    assert bs[int(0.025 * len(bs))] <= b_true <= bs[int(0.975 * len(bs))]
    # projection: one day past the window end, premium decays by ~exp(b)
    t_next = fit.window_end_utc + timedelta(days=1)
    assert math.isclose(fit.premium_at(t_next) / fit.premium_at(fit.window_end_utc),
                        math.exp(fit.b), rel_tol=1e-9)
    # half-life of a -2%/day decay is ~34.7 days
    assert math.isclose(fit.half_life_days, math.log(2) / 0.02, rel_tol=0.25)


def test_decay_fit_is_lookahead_free() -> None:
    """Candles closing after the cutoff must not change the fit at all."""
    times, closes = _synthetic_candles(-0.02)
    cutoff = times[300] + timedelta(hours=1)  # candle 300's close is exactly at cutoff
    fit_a = fit_premium_decay(times[:301], closes[:301], 135.0, cutoff_utc=cutoff, boot_n=50)
    # append a violent future melt-up AFTER the cutoff; the fit must be identical
    fit_b = fit_premium_decay(times, [cl * 2.0 if t > cutoff else cl
                                      for t, cl in zip(times, closes)],
                              135.0, cutoff_utc=cutoff, boot_n=50)
    assert fit_a.n_used == fit_b.n_used
    assert math.isclose(fit_a.a, fit_b.a, rel_tol=1e-12)
    assert math.isclose(fit_a.b, fit_b.b, rel_tol=1e-12)


def test_decay_fit_drops_nonpositive_premium() -> None:
    times, closes = _synthetic_candles(-0.02, n_hours=200)
    closes[10] = 135.0   # zero premium -> log undefined -> dropped + counted
    closes[11] = 130.0   # negative premium -> dropped + counted
    fit = fit_premium_decay(times, closes, 135.0,
                            cutoff_utc=times[-1] + timedelta(hours=2), boot_n=0)
    assert fit.n_dropped_nonpos == 2
    assert fit.n_used == 198


# ----------------------------------------------------------------------------------------
# S1(c): EV / threshold algebra + the trivial corners of the pre-registered rule
# ----------------------------------------------------------------------------------------
def test_meltup_expectation_and_naked_loss() -> None:
    dist = [(0.13, 1.0), (0.26, 1.0), (0.39, 1.0)]
    assert math.isclose(expected_meltup_move(dist), 0.26, rel_tol=1e-12)
    # unnormalized weights are normalized
    assert math.isclose(expected_meltup_move([(0.13, 2.0), (0.39, 2.0)]), 0.26, rel_tol=1e-12)
    # naked loss = mark*E[move] + exit fee at the melted-up price
    nl = naked_loss_per_share(160.0, 0.26, exit_fee_bps=4.5)
    assert math.isclose(nl, 160.0 * 0.26 + 160.0 * 1.26 * 4.5e-4, rel_tol=1e-12)


def test_ev_and_breakeven_algebra() -> None:
    p, nl = 0.8, 42.0
    z = breakeven_net_basis(p, nl)
    assert math.isclose(z, 0.25 * 42.0, rel_tol=1e-12)  # (1-p)/p = 0.25
    # EV at exactly Z is zero; above Z positive; below Z negative (EV>0 <=> basis>Z)
    assert math.isclose(prehedge_ev_per_share(p, z, nl), 0.0, abs_tol=1e-12)
    assert prehedge_ev_per_share(p, z + 0.01, nl) > 0
    assert prehedge_ev_per_share(p, z - 0.01, nl) < 0
    # p=1 (allocation known) -> Z=0; p=0 -> Z=inf (never pre-hedge blind)
    assert breakeven_net_basis(1.0, nl) == 0.0
    assert math.isinf(breakeven_net_basis(0.0, nl))
    # the rule is strict: at zero basis it must NOT hedge even when Z=0
    assert should_hedge(0.0, 0.0) is False
    assert should_hedge(0.01, 0.0) is True


def test_prehedge_threshold_includes_wait_comparator() -> None:
    """Z* must price the option to WAIT: pre-hedging beats waiting only when the basis
    locked now exceeds today's projected allocation-node basis by the naked-risk premium."""
    p, nl = 0.8, 42.0
    # positive projected allocation basis raises the bar by exactly that amount
    assert math.isclose(prehedge_threshold(p, nl, 20.0),
                        breakeven_net_basis(p, nl) + 20.0, rel_tol=1e-12)
    # a NEGATIVE projected allocation basis adds nothing (you'd simply not hedge then)
    assert math.isclose(prehedge_threshold(p, nl, -5.0),
                        breakeven_net_basis(p, nl), rel_tol=1e-12)
    # EV-vs-wait is zero exactly at Z*: p*(B - A) - (1-p)*NL == 0 at B = Z*
    z_star = prehedge_threshold(p, nl, 20.0)
    assert math.isclose(prehedge_ev_per_share(p, z_star - 20.0, nl), 0.0, abs_tol=1e-12)


def test_net_basis_per_share_components() -> None:
    # mark 160 vs fill 135, fee 4.5bps on mark, funding +1e-6/hr over 50h on mark
    nb = net_basis_per_share(160.0, 135.0, funding_hourly=1e-6,
                             hours_node_to_settle=50.0, entry_fee_bps=4.5)
    assert math.isclose(nb, 25.0 - 160.0 * 4.5e-4 + 1e-6 * 160.0 * 50.0, rel_tol=1e-12)
    # zero gross basis -> net is NEGATIVE (fees) -> can never clear a Z >= 0 threshold
    nb0 = net_basis_per_share(135.0, 135.0, 0.0, 0.0, entry_fee_bps=4.5)
    assert nb0 < 0


def _decision_rows(margin_eur: float, live_mark: float,
                   fit: "DecayFit | None" = None) -> list[NodeDecision]:
    nodes = [("NOW", datetime(2026, 6, 10, 17, 0, tzinfo=UTC)),
             ("D1", datetime(2026, 6, 11, 20, 0, tzinfo=UTC)),
             ("D2", datetime(2026, 6, 12, 6, 0, tzinfo=UTC))]
    return build_decision_table(
        fill_prices=[135.0, 162.0], nodes=nodes, now_name="NOW", live_mark=live_mark,
        fit=fit, p_fill_map={0.10: 0.8, 0.25: 0.5}, prehedge_fill_frac=0.10,
        meltup_dist=[(0.13, 1.0), (0.26, 1.0), (0.39, 1.0)],
        subscription_eur=10_000.0, eurusd=1.15, margin_eur=margin_eur, leverage=1.5,
        contracts_per_share=1.0, funding_hourly=0.0,
        settle_utc=datetime(2026, 6, 12, 20, 0, tzinfo=UTC))


def test_decision_rule_trivial_corner_zero_margin_never_hedges() -> None:
    rows = _decision_rows(margin_eur=0.0, live_mark=164.0)
    assert rows, "decision table should not be empty"
    assert all(r.prehedge_shares == 0.0 for r in rows)
    assert all(r.armed is False for r in rows)


def test_decision_rule_trivial_corner_zero_basis_never_hedges() -> None:
    # live mark == offer == every node's mark (flat fit absent -> live mark everywhere):
    # gross basis 0 at fill$135, negative at 162 -> no node may arm
    rows = _decision_rows(margin_eur=2_000.0, live_mark=135.0)
    assert all(r.armed is False for r in rows)
    # and net basis is fee-negative at the $135 row, strictly negative at $162
    for r in rows:
        assert r.net_basis < 0 if r.fill_price == 135.0 else r.net_basis < -20


def test_decision_rule_wait_dominates_without_decay_information() -> None:
    """With no decay fit (every node projected at the live mark) and zero funding, the
    projected allocation basis equals today's basis -- so pre-hedging can NEVER beat
    waiting (same basis, plus a naked tail). The early nodes must not arm."""
    rows = _decision_rows(margin_eur=2_000.0, live_mark=164.0)
    by = {(r.fill_price, r.node): r for r in rows}
    for price in (135.0, 162.0):
        assert by[(price, "NOW")].armed is False
        assert by[(price, "D1")].armed is False
    # the rich-basis $135 row still arms the Friday gate (p=1, Z=0, basis > 0)
    d2 = by[(135.0, "D2")]
    assert d2.p_fill_ge == 1.0 and d2.z_threshold == 0.0
    assert math.isclose(d2.prehedge_shares, margin_cap_shares(2_300.0, 1.5, 164.0, 1.0),
                        rel_tol=1e-12)
    assert d2.armed is True
    # X at the early nodes = pessimistic 10% of requested, under the margin ceiling
    now_135 = by[(135.0, "NOW")]
    assert math.isclose(now_135.prehedge_shares,
                        0.10 * shares_requested(10_000.0, 1.15, 135.0), rel_tol=1e-12)
    # armed <=> EV-vs-wait > 0 <=> B > Z*, and ev_total = X * ev_vs_wait
    for r in rows:
        assert r.armed == (r.prehedge_shares > 0 and r.net_basis > r.z_threshold)
        assert math.isclose(r.ev_total, r.prehedge_shares * r.ev_vs_wait_per_share,
                            rel_tol=1e-12)
    # at fill$162 the basis is small but positive (mark 164) -> the Friday gate still
    # arms; the NEGATIVE-basis no-hedge corner is covered by the zero-basis test above
    d2_162 = by[(162.0, "D2")]
    assert d2_162.net_basis > 0 and d2_162.armed is True


def test_decision_rule_arms_early_when_decay_and_fill_odds_justify_it() -> None:
    """With a real decay fit (basis bleeding ~2%/day) pre-hedging arms only when the
    fill probability is high enough to overcome the naked tail + the wait option."""
    times, closes = _synthetic_candles(-0.02, prem0=29.0, n_hours=400)
    fit = fit_premium_decay(times, closes, 135.0,
                            cutoff_utc=times[-1] + timedelta(hours=2), boot_n=0)
    nodes = [("NOW", datetime(2026, 6, 10, 17, 0, tzinfo=UTC)),
             ("D1", datetime(2026, 6, 11, 20, 0, tzinfo=UTC)),
             ("D2", datetime(2026, 6, 12, 6, 0, tzinfo=UTC))]

    def rows_at(p10: float, meltup: list[tuple[float, float]]):
        return build_decision_table(
            fill_prices=[135.0], nodes=nodes, now_name="NOW", live_mark=164.0,
            fit=fit, p_fill_map={0.10: p10}, prehedge_fill_frac=0.10,
            meltup_dist=meltup, subscription_eur=10_000.0, eurusd=1.15,
            margin_eur=2_000.0, leverage=1.5, contracts_per_share=1.0,
            funding_hourly=0.0, settle_utc=datetime(2026, 6, 12, 20, 0, tzinfo=UTC))

    base_meltup = [(0.13, 1.0), (0.26, 1.0), (0.39, 1.0)]
    # at p=0.80 with the full Cerebras melt-up tail, waiting dominates -> NOW not armed
    by = {r.node: r for r in rows_at(0.80, base_meltup)}
    assert by["NOW"].armed is False
    # near-certain fill + tiny melt-up tail -> locking today's richer basis beats the
    # projected (decayed) Friday basis -> NOW arms
    by = {r.node: r for r in rows_at(0.999, [(0.01, 1.0)])}
    assert by["NOW"].armed is True
    assert by["NOW"].net_basis > by["NOW"].z_threshold
    # and the threshold decomposes as Z* = Z_naked + max(0, projected alloc basis)
    r = by["NOW"]
    assert math.isclose(r.z_threshold, r.z_naked + max(0.0, r.proj_alloc_basis),
                        rel_tol=1e-12)


def test_decision_table_uses_rate_only_projection_for_future_nodes() -> None:
    """Future nodes project the LIVE premium forward at the fitted decay RATE only.
    Projecting from the fitted trend line instead would smuggle a mean-reversion-to-
    trend bet into the risk rule (the live mark can sit below the trend, and the
    'projection' would then say the basis will GROW by waiting)."""
    times, closes = _synthetic_candles(-0.02, prem0=29.0, n_hours=400)
    fit = fit_premium_decay(times, closes, 135.0,
                            cutoff_utc=times[-1] + timedelta(hours=2), boot_n=50)
    live_mark = 164.0  # live premium $29
    rows = _decision_rows(margin_eur=2_000.0, live_mark=live_mark, fit=fit)
    by = {(r.fill_price, r.node): r for r in rows}
    # NOW uses the live mark exactly
    now = by[(135.0, "NOW")]
    assert now.mark_used == live_mark
    # D1/D2 = live premium x exp(b * dt), anchored at the NOW node's time
    d1, d2 = by[(135.0, "D1")], by[(135.0, "D2")]
    for d in (d1, d2):
        dt_days = (d.t_utc - now.t_utc).total_seconds() / 86400.0
        expect = 135.0 + (live_mark - 135.0) * math.exp(fit.b * dt_days)
        assert math.isclose(d.mark_used, expect, rel_tol=1e-12)
    # with b<0 the projected mark strictly DECAYS from the live level across nodes --
    # the anti-reversion property the rate-only choice guarantees
    assert live_mark > d1.mark_used > d2.mark_used
    # bootstrap CI brackets the point projection
    assert d1.mark_lo <= d1.mark_used <= d1.mark_hi


def test_s1_decision_path_has_no_lookahead() -> None:
    """Like the existing no-lookahead test: nothing in the S1 decision path may touch a
    realized settlement price, and the decision-table builder cannot reach raw candles
    (it only receives the causal DecayFit), so no future bar can leak into a decision."""
    import inspect

    from scripts import spcx_convergence_calc as mod

    for fn in (mod.build_decision_table, mod.prehedge_ev_per_share,
               mod.breakeven_net_basis, mod.prehedge_threshold, mod.net_basis_per_share,
               mod.build_hedge_grid, mod.should_hedge):
        src = inspect.getsource(fn)
        assert "realized_pnl_per_share_at_settlement" not in src
        assert "settle_close" not in src
        assert "fetch_hl_candles" not in src
    # and the fitter itself enforces its cutoff (tested functionally above)
    assert "cutoff_utc" in inspect.getsource(mod.fit_premium_decay)
