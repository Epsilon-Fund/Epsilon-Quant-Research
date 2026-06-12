"""SPCX IPO convergence basis + liquidation-survival calculator.

PLAIN ENGLISH
-------------
The trade this models: buy SpaceX IPO Class A shares at the FIXED $135 offer (long leg),
short a Hyperliquid SpaceX pre-IPO perp (short leg), and hold BOTH legs to the perp's
first-trading-day settlement to capture the convergence basis. If held to settlement and
units match, the P&L is locked the moment both legs are on -- it does not depend on which
way the stock moves. The only thing that can break that promise is (a) a units / share-base
mismatch that makes the "spread" a mirage, or (b) the short getting LIQUIDATED on a pre-
settlement melt-up before convergence happens (the Cerebras ruin mode).

This is the OFFLINE gate. It turns Friday-morning inputs into a go/no-go. It does NOT decide
the live-only unknowns (will you actually get the TR allocation, real book depth at size,
whether richness persists, oracle/transition behavior at listing). Per the COWORK "terminus =
live" rule, a green verdict justifies a minimal instrumented live test, not a trading system.

WHAT IT ENFORCES (the core of the tool)
---------------------------------------
Every price is restated to BOTH (a) total implied FDV in $ and (b) a per-IPO-share-equivalent
on the 13,075,865,175 IPO share base BEFORE any spread or sizing math. The naive per-share
gap (perp_mark - 135) is WRONG whenever the perp's share base != the IPO base. The tool shows
the denominator ratio R = IPO_base / perp_base and the over-/under-sizing error it induces,
and it REFUSES to print a naive per-share "spread" for a valuation-unit contract (vntl).

TWO CONTRACTS, TWO SETTLEMENT REFERENCES (encode both)
------------------------------------------------------
- vntl:SPACEX (Ventuals): quoted in valuation-units (price = valuation / $1e9). Cash-settles
  at 4:00pm ET first trading day to (basic shares outstanding x first-day CLOSE), via
  haltTrading. Ventuals docs state basic shares outstanding -- we default that to the same
  13,075,865,175 IPO base, so it converges cleanly to the IPO long leg.
- xyz:SPCX (trade[XYZ] / HIP-3): quoted "per expected Class A share". trade[XYZ] publishes NO
  share base for SPCX (the docs explicitly disclaim share-count / FDV). Per the IPOP docs it
  may CONVERT IN PLACE to a listed-equity perp rather than cash-settle, so its convergence
  reference is the post-listing trading price, not necessarily the IPO close. We default its
  base to the IPO base (R=1, naive == units-matched) but flag the divergence and let you pass
  --xyz-base to test the split-adjusted ~11.87B hypothesis.

PARTIAL HEDGE / UNLEVERED (the realistic generalization)
---------------------------------------------------------
The short need not be levered, nor sized 1:1 to the long. Sizing is by a FDV-anchored hedge
ratio h = short_contracts / FDV-neutral_contracts (h=1 fully hedged/locked, h<1 net-long
residual, h>1 net-short residual). Total P&L = LOCKED [h.N.basis, settle-invariant] + RESIDUAL
[(1-h).N.(close-135), a directional bet on the uncovered fraction]. Leverage and liquidation are
separate from h: at L=1 (unlevered) the liquidation buffer is wide (~+82% xyz) but FINITE -- a
perp short's loss is unbounded, only margin >> notional (L->0) is truly un-liquidatable.

RUN
---
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py                 # live, h=1, UNLEVERED
    PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --hedge-ratio 0.5  # half-hedged net-long
    PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --leverage 2       # levered (riskier)
    PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --offline --chart out.png --json
    # localhost live monitor (terminal dashboard), 30s refresh, with a declared live short:
    PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --watch 30 \
        --live-entry 159.9 --live-short-notional 50000 --live-margin 33000 --parquet-log

BLOCK S1 (hedge grid + pre-hedge timing rule; see spcx_listing_day_gameplan.md section 7)
-----------------------------------------------------------------------------------------
    # (a) fill-price x fill-fraction x margin hedge grid (live EURUSD, flagged):
    PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --grid
    # (b) basis-decay fit on HL hourly candles (level premium vs $135), bootstrap CI:
    PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --decay-fit
    # (c) the one-page pre-registered rule table ("hedge X at node Y iff net basis >= Z"):
    PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --decision
Tests: PYTHONPATH=. uv run pytest tests/test_spcx_convergence_calc.py -q
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------------------
# Constants / anchors (sourced; see spacex_ipo_market_map_handoff.md + venue docs)
# --------------------------------------------------------------------------------------
IPO_OFFER_DEFAULT = 135.0
IPO_BASE_DEFAULT = 13_075_865_175  # S-1/A no-option Class A + Class B (store the convention!)
XYZ_SPLIT_HYPOTHESIS_BASE = 11_870_000_000  # ~11.87B, the "may-be-split-adjusted" base to stress

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "analysis" / "spcx_convergence"

PER_SHARE = "per_share"
VALUATION_PER_1E9 = "valuation_per_1e9"

CASH_TO_IPO_CLOSE = "cash_to_ipo_close"   # vntl: cash-settles to basic-shares x first-day close
CONVERT_IN_PLACE = "convert_in_place"     # xyz: may convert to listed-equity perp (no clean cash settle)


# --------------------------------------------------------------------------------------
# Pure units / basis math  (stdlib only, so tests need no numpy/scipy)
# --------------------------------------------------------------------------------------
def implied_fdv_usd(price: float, base: float, units: str) -> float:
    """Total implied fully-diluted valuation in $ for a quote, given its units convention.

    per_share        : FDV = price * share_base
    valuation_per_1e9: FDV = price * 1e9   (Ventuals: price IS valuation / 1e9)
    """
    if units == PER_SHARE:
        return price * base
    if units == VALUATION_PER_1E9:
        return price * 1e9
    raise ValueError(f"unknown units convention: {units!r}")


def per_ipo_share_equiv(price: float, base: float, units: str, ipo_base: float) -> float:
    """Restate a quote into $ per IPO-share-equivalent on the IPO base. This is the ONLY
    representation in which it is legitimate to subtract the $135 offer."""
    return implied_fdv_usd(price, base, units) / ipo_base


def denominator_ratio(ipo_base: float, effective_base: float) -> float:
    """R = IPO_base / perp_(or settlement_)base. R=1 means naive per-share gap is correct.
    R>1 means the perp base is SMALLER than the IPO base, so the naive (mark-135) gap
    over-states the real per-IPO-share basis and a 1:1 short over-hedges by exactly R."""
    return ipo_base / effective_base


def naive_per_share_gap(mark: float, offer: float) -> float:
    """mark - offer. ONLY meaningful for a per-share contract on the IPO base (R=1)."""
    return mark - offer


def units_matched_gap(price: float, base: float, units: str, ipo_base: float, offer: float) -> float:
    """The real, units-matched per-IPO-share basis = per-IPO-share-equiv(entry) - offer."""
    return per_ipo_share_equiv(price, base, units, ipo_base) - offer


def fdv_neutral_contract_count(shares_long: float, ipo_base: float, perp_base: float, units: str,
                               perp_price: float) -> float:
    """How many perp contracts make the short FDV-neutral against `shares_long` IPO shares.

    per_share        : the long has dFDV/dPrice_ipo = ipo_base; one perp contract has
                       dFDV/dPrice_perp = perp_base. Matching dollar-delta wrt FDV gives
                       M = shares_long * perp_base / ipo_base = shares_long / R.
    valuation_per_1e9: one contract moves $1e9 of FDV per +1.0 in price, so
                       M = shares_long * (ipo_share_price_sensitivity) -> shares_long * 1e9/ipo_base...
                       expressed as contracts: M = shares_long / (ipo_base / 1e9) ... we instead
                       report the FDV-neutral *notional*, since contract-count is unit-specific.
    """
    if units == PER_SHARE:
        return shares_long * perp_base / ipo_base
    # valuation units: 1 contract moves $1e9 of FDV per +1.0 in price; the long moves
    # $1 of IPO-share price per ipo_base of FDV. FDV-neutral => M/1e9 == shares_long/ipo_base.
    return shares_long * 1e9 / ipo_base


def oversize_error_pct(naive_contracts: float, neutral_contracts: float) -> float:
    """% by which a naive 1:1 (or naive-count) short over- or under-hedges vs FDV-neutral."""
    if neutral_contracts == 0:
        return float("nan")
    return 100.0 * (naive_contracts - neutral_contracts) / neutral_contracts


# --------------------------------------------------------------------------------------
# Locked P&L (direction-independence)  &  realized-settlement invariance sweep
# --------------------------------------------------------------------------------------
def locked_pnl_per_ipo_share(perp_entry_ipo_units: float, offer: float) -> float:
    """The locked basis: (short entry restated to IPO-share units) - long entry.
    Direction-INDEPENDENT iff held to settlement AND units match AND both legs converge
    to the same settlement FDV reference."""
    return perp_entry_ipo_units - offer


def realized_pnl_per_share_at_settlement(perp_entry_ipo_units: float, offer: float,
                                         settle_close_ipo_price: float) -> float:
    """Net per-IPO-share P&L of the FDV-matched long+short if the first-day close lands at
    `settle_close_ipo_price`. Used ONLY for the invariance sweep -- never as a decision input
    (no look-ahead: the realized close is not known at decision time).

    long  per share : settle_close - offer
    short per share : perp_entry_ipo_units - settle_close   (FDV-matched, settles to same close)
    total           : perp_entry_ipo_units - offer          (settle_close cancels)
    """
    long_leg = settle_close_ipo_price - offer
    short_leg = perp_entry_ipo_units - settle_close_ipo_price
    return long_leg + short_leg


# --------------------------------------------------------------------------------------
# Liquidation / path survival
# --------------------------------------------------------------------------------------
def maintenance_margin_frac(max_leverage: float) -> float:
    """Hyperliquid maintenance-margin fraction = 1 / (2 * asset_max_leverage)."""
    return 1.0 / (2.0 * max_leverage)


def liq_adverse_frac(leverage: float, mmr: float) -> float:
    """Fractional ADVERSE move (price UP, for a short) that triggers liquidation.

    Isolated short: equity = M + S*(E - P); maintenance = mmr * S * P.
    Liquidation when equity == maintenance:
        P_liq = E * (1 + 1/L) / (1 + mmr)
    so the adverse fraction is (1 + 1/L)/(1 + mmr) - 1.   (unit-free; works in any base)
    """
    return (1.0 + 1.0 / leverage) / (1.0 + mmr) - 1.0


def liq_price_short(entry: float, leverage: float, mmr: float) -> float:
    return entry * (1.0 + liq_adverse_frac(leverage, mmr))


def survives_move(scenario_frac: float, leverage: float, mmr: float) -> bool:
    """True if a +scenario_frac adverse spike does NOT reach the liquidation price."""
    return scenario_frac < liq_adverse_frac(leverage, mmr)


def max_survivable_leverage(worst_scenario_frac: float, mmr: float) -> float:
    """Highest leverage whose liquidation move strictly exceeds the worst melt-up scenario.

    Solve (1 + 1/L)/(1 + mmr) - 1 = s  ->  L = 1 / [ (1 + s)(1 + mmr) - 1 ].
    """
    denom = (1.0 + worst_scenario_frac) * (1.0 + mmr) - 1.0
    if denom <= 0:
        return float("inf")
    return 1.0 / denom


# --------------------------------------------------------------------------------------
# Hedge ratio + partial-coverage P&L decomposition  (generalizes the 1:1-cover assumption)
# --------------------------------------------------------------------------------------
# Hedge ratio h is defined in FDV-DELTA space, anchored to the FDV-neutral short count:
#   h = short_contracts / fdv_neutral_contracts.
# It is unit-agnostic by construction (a ratio of FDV sensitivities), so h=1 means truly
# direction-neutral on BOTH the per-share (xyz) and valuation-unit (vntl) contracts. Defining
# h off matched dollar-notional instead would silently under-hedge vntl by the R/units gap.
#   h = 0  -> no short (pure long).   h = 1 -> FDV-neutral (locked).   h > 1 -> net SHORT.
def hedge_ratio_from_short(short_contracts: float, neutral_contracts: float) -> float:
    if neutral_contracts == 0:
        return float("nan")
    return short_contracts / neutral_contracts


def short_contracts_from_hedge(h: float, neutral_contracts: float) -> float:
    return h * neutral_contracts


def locked_pnl_total(h: float, shares_long: float, matched_gap: float) -> float:
    """Direction-INDEPENDENT component = h * N_long * (units-matched basis). The covered
    fraction's settlement price cancels against the long; only the entry basis remains."""
    return h * shares_long * matched_gap


def residual_pnl_total(h: float, shares_long: float, close_ipo_price: float, offer: float) -> float:
    """Uncovered DIRECTIONAL component = (1 - h) * N_long * (close - offer). This is NOT locked;
    it is a naked long (h<1) or naked short (h>1) on the settlement close, sized by (1-h)."""
    return (1.0 - h) * shares_long * (close_ipo_price - offer)


def total_pnl_at_close(h: float, shares_long: float, matched_gap: float,
                       close_ipo_price: float, offer: float) -> float:
    """locked + residual. Reduces to pure long at h=0 and to the fixed locked basis at h=1."""
    return (locked_pnl_total(h, shares_long, matched_gap)
            + residual_pnl_total(h, shares_long, close_ipo_price, offer))


def residual_slope_per_dollar(h: float, shares_long: float) -> float:
    """d(total P&L)/d(close), in $ per $1 of settlement close. >0 net long, <0 net short,
    0 fully neutral. This is what the forcing question must report when h != 1."""
    return (1.0 - h) * shares_long


def capital_deployed(long_notional: float, short_margin: float) -> float:
    """The IPO long is fully cash-funded at the fixed offer (no leverage); only the short
    carries leverage, so total capital = long notional + short isolated margin."""
    return long_notional + short_margin


def return_on_capital(net_locked: float, capital: float) -> float:
    """Locked-only return on capital. The residual is a directional bet and is deliberately
    NOT counted as guaranteed return."""
    if capital == 0:
        return float("nan")
    return net_locked / capital


def simple_annualized(roc: float, hours: float) -> float:
    """Simple (linear) annualization for a ONE-SHOT convergence. Compounding a ~3-day,
    non-repeatable event produces nonsense, so we never compound."""
    if hours == 0:
        return float("nan")
    return roc * (8760.0 / hours)


def liq_buffer_summary(current_mark: float, liq_px: float, alert_pct: float = 0.10) -> dict:
    """Live liquidation-buffer for a declared short. `buffer_frac` is the fractional adverse
    move from the CURRENT mark to the (entry-fixed) liquidation price. Bands:
        BREACH  : current mark at/through the liq price (buffer <= 0)
        WARN    : within alert_pct of liquidation
        SAFE    : otherwise
    """
    if current_mark <= 0:
        return {"current_mark": current_mark, "liq_px": liq_px, "buffer_frac": float("nan"),
                "band": "BREACH"}
    buffer_frac = (liq_px - current_mark) / current_mark
    if buffer_frac <= 0:
        band = "BREACH"
    elif buffer_frac <= alert_pct:
        band = "WARN"
    else:
        band = "SAFE"
    return {"current_mark": current_mark, "liq_px": liq_px, "buffer_frac": buffer_frac, "band": band}


# --------------------------------------------------------------------------------------
# Contract + leg dataclasses
# --------------------------------------------------------------------------------------
@dataclass
class Contract:
    name: str
    mark: float
    oracle: float
    units: str                 # PER_SHARE | VALUATION_PER_1E9
    base: float                # per-share base, OR (valuation) the basic-shares-outstanding settlement base
    max_leverage: float
    oi_cap_usd: float
    oi_units: float            # current open interest in contract units (from API)
    funding_hourly: float      # API funding rate (per hour); >0 => longs pay shorts => short receives
    premium: float             # API premium (mark vs oracle), informational
    settlement: str            # CASH_TO_IPO_CLOSE | CONVERT_IN_PLACE
    margin_asset: str = "USDC"
    base_is_assumed: bool = False  # True when we defaulted the base (xyz) rather than reading it from API

    def current_oi_notional_usd(self) -> float:
        # per_share OI is in shares; valuation OI is in valuation-units whose $ notional == price.
        return self.oi_units * self.mark


@dataclass
class Legs:
    shares_long: float        # IPO shares filled (long), each at the fixed offer
    short_notional: float     # USD notional of the perp short
    posted_margin: float      # USD isolated margin posted
    hl_fee_bps: float = 4.5   # HL taker fee, one side, bps of notional (cash-settle => no exit fee)
    fee_both_sides: bool = False
    fx_cost_bps: float = 0.0  # EUR/USD round-trip / hedge cost on the TR long leg, bps of long notional
    stable_basis_bps: float = 0.0  # stablecoin / USDH off-peg basis, bps of margin
    hours_to_settle: float = 72.0  # funding accrual horizon

    @property
    def effective_leverage(self) -> float:
        if self.posted_margin <= 0:
            return float("inf")
        return self.short_notional / self.posted_margin


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
@dataclass
class ContractEval:
    contract: Contract
    offer: float
    ipo_base: float
    legs: Legs
    scenarios: list[float]
    close_grid: list[float] = field(default_factory=lambda: [100.0, 135.0, 167.0, 222.0])

    # filled in __post_init__
    fdv_usd: float = 0.0
    per_ipo_share: float = 0.0
    R: float = 1.0
    naive_gap: Optional[float] = None
    matched_gap: float = 0.0
    oversize_pct: float = 0.0
    neutral_contracts: float = 0.0
    naive_contracts: float = 0.0
    mmr: float = 0.0
    short_size: float = 0.0
    liq_frac: float = 0.0
    liq_px: float = 0.0
    scenario_results: list[tuple[float, float, bool]] = field(default_factory=list)
    # hedge ratio + partial-coverage decomposition
    hedge_ratio: float = 1.0
    locked_total: float = 0.0           # h * shares * matched_gap (direction-independent)
    residual_slope: float = 0.0         # (1-h) * shares; d(PnL)/d(close)
    residual_by_close: list[tuple[float, float, float]] = field(default_factory=list)  # (close, residual, total)
    is_direction_neutral: bool = True
    # P&L
    gross_locked_total: float = 0.0     # locked piece, gross of costs
    fee_cost: float = 0.0
    funding_income: float = 0.0
    fx_cost: float = 0.0
    stable_cost: float = 0.0
    net_total: float = 0.0              # net LOCKED only (residual excluded by design)
    net_per_share: float = 0.0
    # capital / return
    long_notional: float = 0.0
    capital: float = 0.0
    roc: float = 0.0
    roc_annualized: float = 0.0
    # capacity
    cap_frac_of_oi_cap: float = 0.0
    cap_frac_of_book: float = 0.0
    # verdict
    verdict: str = ""
    verdict_reasons: list[str] = field(default_factory=list)
    max_survive_leverage: float = 0.0
    max_survive_notional: float = 0.0

    def __post_init__(self) -> None:
        c, legs = self.contract, self.legs
        self.fdv_usd = implied_fdv_usd(c.mark, c.base, c.units)
        self.per_ipo_share = per_ipo_share_equiv(c.mark, c.base, c.units, self.ipo_base)
        self.R = denominator_ratio(self.ipo_base, c.base)

        # naive gap only legitimate for a per-share contract; refuse it for valuation units
        if c.units == PER_SHARE:
            self.naive_gap = naive_per_share_gap(c.mark, self.offer)
        else:
            self.naive_gap = None  # REFUSED: cannot mix valuation-units with a $135/share offer
        self.matched_gap = self.per_ipo_share - self.offer

        # sizing
        self.neutral_contracts = fdv_neutral_contract_count(
            legs.shares_long, self.ipo_base, c.base, c.units, c.mark)
        # a "naive" short matches contract-count to share-count for per-share, or matches
        # notional 1:1 (shares*offer) for valuation
        if c.units == PER_SHARE:
            self.naive_contracts = legs.shares_long
        else:
            self.naive_contracts = legs.shares_long * self.offer / c.mark
        self.oversize_pct = oversize_error_pct(self.naive_contracts, self.neutral_contracts)

        # liquidation
        self.mmr = maintenance_margin_frac(c.max_leverage)
        self.short_size = legs.short_notional / c.mark if c.mark else 0.0
        L = legs.effective_leverage
        self.liq_frac = liq_adverse_frac(L, self.mmr)
        self.liq_px = liq_price_short(c.mark, L, self.mmr)
        self.scenario_results = [
            (s, c.mark * (1.0 + s), survives_move(s, L, self.mmr)) for s in self.scenarios
        ]

        # hedge ratio = actual short contracts / FDV-neutral contracts (anchored in FDV space,
        # NOT dollar-notional, so h=1 is truly direction-neutral on both unit conventions)
        self.hedge_ratio = hedge_ratio_from_short(self.short_size, self.neutral_contracts)
        h = self.hedge_ratio
        self.is_direction_neutral = abs(h - 1.0) < 1e-9

        # P&L decomposition: LOCKED (h fraction, settle-price-independent) + RESIDUAL (uncovered
        # (1-h) directional bet on the close). Costs apply to the locked book only.
        self.locked_total = locked_pnl_total(h, legs.shares_long, self.matched_gap)
        self.gross_locked_total = self.locked_total
        self.residual_slope = residual_slope_per_dollar(h, legs.shares_long)
        self.residual_by_close = [
            (px, residual_pnl_total(h, legs.shares_long, px, self.offer),
             total_pnl_at_close(h, legs.shares_long, self.matched_gap, px, self.offer))
            for px in self.close_grid
        ]

        notional = legs.short_notional
        sides = 2.0 if legs.fee_both_sides else 1.0
        self.fee_cost = notional * (legs.hl_fee_bps / 1e4) * sides
        # short receives funding when funding_hourly>0 (longs pay shorts); sign preserved
        self.funding_income = c.funding_hourly * notional * legs.hours_to_settle
        self.long_notional = legs.shares_long * self.offer
        self.fx_cost = self.long_notional * (legs.fx_cost_bps / 1e4)
        self.stable_cost = legs.posted_margin * (legs.stable_basis_bps / 1e4)
        # net is the LOCKED piece net of costs; the residual is directional and excluded here
        self.net_total = (self.gross_locked_total - self.fee_cost + self.funding_income
                          - self.fx_cost - self.stable_cost)
        self.net_per_share = self.net_total / legs.shares_long if legs.shares_long else 0.0

        # capital / return on capital (locked-only return; residual is a bet, not return)
        self.capital = capital_deployed(self.long_notional, legs.posted_margin)
        self.roc = return_on_capital(self.net_total, self.capital)
        self.roc_annualized = simple_annualized(self.roc, legs.hours_to_settle)

        # capacity
        self.cap_frac_of_oi_cap = notional / c.oi_cap_usd if c.oi_cap_usd else float("nan")
        book = c.current_oi_notional_usd()
        self.cap_frac_of_book = notional / book if book else float("nan")

        # verdict (hedge-aware): gate the LOCKED piece; flag residual directional exposure
        worst = max(self.scenarios) if self.scenarios else 0.0
        self.max_survive_leverage = max_survivable_leverage(worst, self.mmr)
        self.max_survive_notional = legs.posted_margin * self.max_survive_leverage
        any_liquidated = any(not ok for _, _, ok in self.scenario_results)
        reasons: list[str] = []
        if self.net_total <= 0:
            reasons.append(
                f"locked net basis <= 0 (net ${self.net_total:,.0f} on the hedged fraction)")
        if any_liquidated:
            liq_list = ", ".join(
                f"+{s*100:.0f}%" for s, _, ok in self.scenario_results if not ok)
            reasons.append(
                f"melt-up liquidates the short at L={L:.2f}x (scenarios {liq_list}; "
                f"liq move = +{self.liq_frac*100:.1f}%)")
        if reasons:
            self.verdict = "NO-TRADE"
            self.verdict_reasons = reasons
        else:
            self.verdict = "TRADE-ABLE (offline gate green)"
            self.verdict_reasons = [
                f"survives all default melt-ups (liq move +{self.liq_frac*100:.1f}% > worst +{worst*100:.0f}%)",
                f"max short notional that survives +{worst*100:.0f}% at this margin = "
                f"${self.max_survive_notional:,.0f} (L<= {self.max_survive_leverage:.2f}x)",
            ]
        if not self.is_direction_neutral:
            kind = "net LONG" if h < 1.0 else "net SHORT"
            self.verdict_reasons.append(
                f"NOT direction-neutral: h={h:.3f} leaves a {kind} residual of "
                f"${self.residual_slope:,.0f} P&L per $1 of settlement close "
                f"(this fraction is a directional bet, not locked basis)")


# --------------------------------------------------------------------------------------
# Live fetch + snapshot cache
# --------------------------------------------------------------------------------------
def fetch_hl_snapshot(timeout: float = 15.0) -> dict:
    """Read-only pull of xyz:SPCX and vntl:SPACEX context from the Hyperliquid info endpoint.
    Never hardcodes marks; returns a timestamped snapshot dict."""
    import httpx  # local import so --offline / tests don't require network

    out: dict = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "assets": {}}
    with httpx.Client(timeout=timeout) as client:
        for dex, want in (("xyz", "xyz:SPCX"), ("vntl", "vntl:SPACEX")):
            r = client.post(HL_INFO_URL, json={"type": "metaAndAssetCtxs", "dex": dex})
            r.raise_for_status()
            meta, ctxs = r.json()
            for i, a in enumerate(meta["universe"]):
                if a["name"] == want:
                    ctx = ctxs[i]
                    out["assets"][want] = {
                        "name": want,
                        "mark": float(ctx["markPx"]),
                        "oracle": float(ctx["oraclePx"]),
                        "mid": float(ctx.get("midPx", ctx["markPx"])),
                        "open_interest_units": float(ctx["openInterest"]),
                        "day_ntl_vlm": float(ctx["dayNtlVlm"]),
                        "premium": float(ctx.get("premium", 0.0)),
                        "funding_hourly": float(ctx.get("funding", 0.0)),
                        "max_leverage": float(a["maxLeverage"]),
                        "margin_mode": a.get("marginMode", "strictIsolated"),
                    }
    return out


def save_snapshot(snap: dict, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = snap["fetched_at_utc"].replace(":", "").replace("-", "").replace(".", "")[:15]
    p = out_dir / f"hl_snapshot_{ts}.json"
    p.write_text(json.dumps(snap, indent=2))
    (out_dir / "latest.json").write_text(json.dumps(snap, indent=2))
    return p


def load_latest_snapshot(out_dir: Path = DEFAULT_OUT_DIR) -> Optional[dict]:
    p = out_dir / "latest.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


# --------------------------------------------------------------------------------------
# Build contracts from a snapshot
# --------------------------------------------------------------------------------------
def contracts_from_snapshot(snap: dict, ipo_base: float, xyz_base: float) -> dict[str, Contract]:
    a = snap["assets"]
    out: dict[str, Contract] = {}
    if "xyz:SPCX" in a:
        x = a["xyz:SPCX"]
        out["xyz"] = Contract(
            name="xyz:SPCX", mark=x["mark"], oracle=x["oracle"], units=PER_SHARE,
            base=xyz_base, max_leverage=x["max_leverage"],
            oi_cap_usd=150e6,  # trade[XYZ] IPOP spec: SPCX open-interest cap $150m
            oi_units=x["open_interest_units"], funding_hourly=x["funding_hourly"],
            premium=x["premium"], settlement=CONVERT_IN_PLACE, margin_asset="USDC",
            base_is_assumed=True,  # trade[XYZ] publishes NO share base for SPCX
        )
    if "vntl:SPACEX" in a:
        v = a["vntl:SPACEX"]
        out["vntl"] = Contract(
            name="vntl:SPACEX", mark=v["mark"], oracle=v["oracle"], units=VALUATION_PER_1E9,
            base=ipo_base,  # Ventuals settles to basic-shares-outstanding x close; default = IPO base
            max_leverage=v["max_leverage"],
            oi_cap_usd=10e6,  # Ventuals OI cap (handoff)
            oi_units=v["open_interest_units"], funding_hourly=v["funding_hourly"],
            premium=v["premium"], settlement=CASH_TO_IPO_CLOSE, margin_asset="USDC",
            base_is_assumed=False,
        )
    return out


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def _fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def render_text(ev: ContractEval, snap_age: str) -> str:
    c, legs = ev.contract, ev.legs
    L = legs.effective_leverage
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f" {c.name}  [{c.settlement}]   (snapshot: {snap_age})")
    lines.append("=" * 78)
    # 1. units-matched basis
    lines.append("UNITS NORMALIZATION")
    lines.append(f"  mark                      = {c.mark:,.4f}  ({c.units})")
    lines.append(f"  oracle                    = {c.oracle:,.4f}")
    lines.append(f"  implied FDV               = {_fmt_usd(ev.fdv_usd)}  ({ev.fdv_usd/1e12:.3f}T)")
    lines.append(f"  per-IPO-share-equiv       = ${ev.per_ipo_share:,.2f}  (FDV / {ev.ipo_base:,.0f})")
    if c.base_is_assumed:
        lines.append(f"  share base                = {c.base:,.0f}  [ASSUMED -- venue publishes none]")
    else:
        lines.append(f"  settlement share base     = {c.base:,.0f}")
    lines.append(f"  denominator ratio R       = {ev.R:.4f}  (IPO_base / contract_base)")
    if c.units == PER_SHARE:
        lines.append(f"  naive per-share gap       = ${ev.naive_gap:,.2f}  (mark - {ev.offer:.0f})"
                     + ("  [== units-matched, R=1]" if abs(ev.R - 1) < 1e-9
                        else "  [WRONG: mixes share-count conventions]"))
    else:
        lines.append(f"  naive per-share gap       = REFUSED (valuation-unit quote; "
                     f"{c.mark:,.0f} is not a $/share number)")
    lines.append(f"  units-matched basis       = ${ev.matched_gap:,.2f} / IPO-share")
    lines.append(f"  FDV-neutral short count   = {ev.neutral_contracts:,.2f} contracts "
                 f"(naive {ev.naive_contracts:,.2f}; over/under-size {ev.oversize_pct:+.2f}%)")

    # 2. hedge ratio + P&L decomposition (locked vs residual directional)
    lines.append("")
    hr = ev.hedge_ratio
    lines.append(f"HEDGE & P&L DECOMPOSITION  (hedge ratio h = {hr:.3f}, FDV-anchored)")
    lines.append(f"  long {legs.shares_long:,.0f} IPO shares @ ${ev.offer:.0f} "
                 f"({_fmt_usd(ev.long_notional)}); short {ev.short_size:,.2f} contracts "
                 f"= {_fmt_usd(legs.short_notional)} notional")
    lines.append(f"  FDV-neutral short would be {ev.neutral_contracts:,.2f} contracts -> "
                 f"h = short/neutral = {hr:.3f} "
                 + ("[FULLY HEDGED, direction-neutral]" if ev.is_direction_neutral
                    else (f"[UNDER-hedged: net LONG residual]" if hr < 1.0
                          else "[OVER-hedged: net SHORT residual]")))
    lines.append("")
    lines.append("  LOCKED (direction-independent) component:")
    lines.append(f"    gross locked basis      = {_fmt_usd(ev.gross_locked_total)}  "
                 f"(h {hr:.3f} x ${ev.matched_gap:,.2f}/share x {legs.shares_long:,.0f})")
    lines.append(f"     - HL fees               = {_fmt_usd(-ev.fee_cost)}")
    lines.append(f"     + funding (short {'recv' if ev.funding_income>=0 else 'pays'}) = "
                 f"{_fmt_usd(ev.funding_income)}  "
                 f"({c.funding_hourly*100:.6f}%/hr x {legs.hours_to_settle:.0f}h)")
    lines.append(f"     - EUR/USD on TR leg     = {_fmt_usd(-ev.fx_cost)}  ({legs.fx_cost_bps:.1f} bps)")
    lines.append(f"     - stablecoin/USDH basis = {_fmt_usd(-ev.stable_cost)}  ({legs.stable_basis_bps:.1f} bps)")
    lines.append(f"    NET LOCKED              = {_fmt_usd(ev.net_total)}  "
                 f"(${ev.net_per_share:,.2f}/share)")
    if not ev.is_direction_neutral:
        kind = "net LONG" if hr < 1.0 else "net SHORT"
        lines.append("")
        lines.append(f"  RESIDUAL (uncovered DIRECTIONAL bet -- NOT locked): {kind}, "
                     f"slope ${ev.residual_slope:,.0f} P&L per $1 of close")
        lines.append("    settlement close ($/sh) ->  residual P&L   total P&L (locked+residual)")
        for px, resid, tot in ev.residual_by_close:
            lines.append(f"      ${px:>7,.0f}                {_fmt_usd(resid):>12}   {_fmt_usd(tot):>14}")
        lines.append("    (this fraction has the SAME risk as a naked long/short of the stock; "
                     "it is a directional view, not arbitrage.)")

    # 2b. capital / return on capital
    lines.append("")
    lines.append("CAPITAL & RETURN  (locked-only; residual excluded as it is a bet)")
    lines.append(f"  capital deployed          = {_fmt_usd(ev.capital)}  "
                 f"(long {_fmt_usd(ev.long_notional)} + short margin {_fmt_usd(legs.posted_margin)})")
    lines.append(f"  return on capital (locked) = {ev.roc*100:.2f}%  over {legs.hours_to_settle:.0f}h"
                 f"  -> ~{ev.roc_annualized*100:,.0f}%/yr simple (one-shot, NOT repeatable)")

    # 3. forcing question (hedge-aware: report the residual slope, not an unconditional YES)
    lines.append("")
    fq = "YES" if ev.net_total > 0 else "NO"
    lines.append("FORCING QUESTION (COWORK): if the stock didn't move between entry and "
                 "settlement, would the position still make money?")
    if ev.is_direction_neutral:
        lines.append(f"  -> {fq}: net locked {_fmt_usd(ev.net_total)} ({ev.net_per_share:+,.2f}/share), "
                     f"settle-price-invariant (h=1, units match).")
    else:
        lines.append(f"  -> {fq} on the LOCKED piece ({_fmt_usd(ev.net_total)}), BUT the position is "
                     f"NOT settle-invariant: total P&L slopes ${ev.residual_slope:,.0f} per $1 of close "
                     f"(h={hr:.3f}). The answer is conditional on the close, not unconditional.")

    # 4. path / liquidation survival
    lines.append("")
    unlev = " [UNLEVERED]" if L <= 1.0 + 1e-9 else ""
    lines.append(f"PATH / LIQUIDATION SURVIVAL  (isolated short, L={L:.2f}x{unlev}, "
                 f"mmr={ev.mmr:.3f} from max-lev {c.max_leverage:.0f}x)")
    lines.append(f"  liquidation move          = +{ev.liq_frac*100:.1f}%  "
                 f"(liq price {c.mark*(1+ev.liq_frac):,.2f})")
    for s, px, ok in ev.scenario_results:
        tag = "SURVIVE" if ok else "LIQUIDATED"
        lines.append(f"  +{s*100:>4.0f}% melt-up -> {px:>10,.2f}   {tag}")
    if L <= 1.0 + 1e-9:
        lines.append(f"  NOTE: unlevered/over-collateralized gives a WIDE but FINITE buffer "
                     f"(+{ev.liq_frac*100:.0f}%). A perp short's loss is unbounded -- low leverage")
        lines.append("        only widens the buffer; only margin >> notional (L->0) is truly "
                     "un-liquidatable. You still tie up the full notional as margin.")
    lines.append("  (primary ruin mode: liquidation-before-convergence leaves you long-only "
                 "into the spike.)")

    # 5. capacity
    lines.append("")
    lines.append("CAPACITY vs OPEN-INTEREST CAP")
    lines.append(f"  desired short notional    = {_fmt_usd(legs.short_notional)}")
    lines.append(f"  OI cap                    = {_fmt_usd(c.oi_cap_usd)}  "
                 f"-> you would be {ev.cap_frac_of_oi_cap*100:.2f}% of the cap")
    lines.append(f"  current book OI notional  = {_fmt_usd(c.current_oi_notional_usd())}  "
                 f"-> you would be {ev.cap_frac_of_book*100:.2f}% of the live book")

    # 6. settlement divergence flag
    lines.append("")
    if c.settlement == CONVERT_IN_PLACE:
        lines.append("SETTLEMENT FLAG: xyz:SPCX may CONVERT IN PLACE to a listed-equity perp rather "
                     "than cash-settle to the IPO close. Convergence reference is the post-listing "
                     "trading price; transition/oracle behavior at listing is a LIVE-ONLY unknown.")
    else:
        lines.append("SETTLEMENT FLAG: vntl:SPACEX cash-settles at 4:00pm ET first day to "
                     "(basic shares outstanding x first-day CLOSE) via haltTrading -- converges to "
                     "the IPO long leg by construction IF basic-shares == IPO base.")

    # 7. verdict
    lines.append("")
    lines.append(f"VERDICT: {ev.verdict}")
    for r in ev.verdict_reasons:
        lines.append(f"  - {r}")
    lines.append("")
    return "\n".join(lines)


def eval_to_dict(ev: ContractEval) -> dict:
    c = ev.contract
    return {
        "contract": c.name, "settlement": c.settlement, "units": c.units,
        "mark": c.mark, "oracle": c.oracle, "base": c.base, "base_is_assumed": c.base_is_assumed,
        "implied_fdv_usd": ev.fdv_usd, "per_ipo_share_equiv": ev.per_ipo_share,
        "denominator_ratio_R": ev.R, "naive_per_share_gap": ev.naive_gap,
        "units_matched_basis_per_share": ev.matched_gap,
        "fdv_neutral_contracts": ev.neutral_contracts, "naive_contracts": ev.naive_contracts,
        "oversize_pct": ev.oversize_pct,
        "effective_leverage": ev.legs.effective_leverage, "mmr": ev.mmr,
        "liq_adverse_frac": ev.liq_frac, "liq_price": ev.liq_px,
        "scenarios": [{"move": s, "price": px, "survive": ok} for s, px, ok in ev.scenario_results],
        "hedge_ratio": ev.hedge_ratio, "is_direction_neutral": ev.is_direction_neutral,
        "residual_slope_per_dollar_close": ev.residual_slope,
        "residual_by_close": [{"close": px, "residual": r, "total": t}
                              for px, r, t in ev.residual_by_close],
        "gross_locked_total": ev.gross_locked_total, "fee_cost": ev.fee_cost,
        "funding_income": ev.funding_income, "fx_cost": ev.fx_cost, "stable_cost": ev.stable_cost,
        "net_locked_total": ev.net_total, "net_locked_per_share": ev.net_per_share,
        "capital_deployed": ev.capital, "return_on_capital": ev.roc,
        "roc_annualized_simple": ev.roc_annualized,
        "cap_frac_of_oi_cap": ev.cap_frac_of_oi_cap, "cap_frac_of_book": ev.cap_frac_of_book,
        "max_survive_leverage": ev.max_survive_leverage,
        "max_survive_notional": ev.max_survive_notional,
        "verdict": ev.verdict, "verdict_reasons": ev.verdict_reasons,
    }


def flatten_eval_row(ev: ContractEval, fetched_at_utc: str) -> dict:
    """One flat, Parquet-friendly row per contract per tick (scalars only). Shared schema
    for the --watch tick log and any downstream notebook."""
    return {
        "fetched_at_utc": fetched_at_utc,
        "contract": ev.contract.name,
        "mark": ev.contract.mark,
        "oracle": ev.contract.oracle,
        "implied_fdv_t": ev.fdv_usd / 1e12,
        "per_ipo_share_equiv": ev.per_ipo_share,
        "units_matched_basis": ev.matched_gap,
        "hedge_ratio": ev.hedge_ratio,
        "effective_leverage": ev.legs.effective_leverage,
        "liq_adverse_frac": ev.liq_frac,
        "liq_price": ev.liq_px,
        "funding_hourly": ev.contract.funding_hourly,
        "net_locked_total": ev.net_total,
        "return_on_capital": ev.roc,
        "verdict": ev.verdict,
    }


def make_chart(evals: list[ContractEval], scenarios: list[float], out_path: Path) -> None:
    """SURVIVE/LIQUIDATED heat-strip across leverage x melt-up for each contract."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levs = np.linspace(1.0, 5.0, 81)
    fig, axes = plt.subplots(1, len(evals), figsize=(6.4 * len(evals), 4.6), squeeze=False)
    for ax, ev in zip(axes[0], evals):
        mmr = ev.mmr
        liq = (1.0 + 1.0 / levs) / (1.0 + mmr) - 1.0  # liq adverse frac per leverage
        ax.plot(levs, liq * 100, color="black", lw=2, label="liquidation move")
        for s in scenarios:
            ax.axhline(s * 100, ls="--", lw=1, alpha=0.7,
                       color=("tab:red" if s == max(scenarios) else "tab:gray"))
            ax.text(5.02, s * 100, f"+{s*100:.0f}%", va="center", fontsize=8)
        # shade survive region (liq above the worst scenario)
        ax.fill_between(levs, liq * 100, max(scenarios) * 100,
                        where=(liq > max(scenarios)), color="tab:green", alpha=0.15)
        # mark chosen leverage
        L = ev.legs.effective_leverage
        if 1.0 <= L <= 5.0:
            ax.axvline(L, color="tab:blue", lw=1.5)
            ax.text(L, 5, f"chosen L={L:.1f}x", rotation=90, va="bottom", fontsize=8,
                    color="tab:blue")
        ax.set_title(f"{ev.contract.name}\nliq move vs leverage (mmr={mmr:.3f})")
        ax.set_xlabel("isolated leverage (x)")
        ax.set_ylabel("adverse move to liquidation (%)")
        ax.set_ylim(0, 110)
        ax.grid(alpha=0.3)
    fig.suptitle("SPCX short: liquidation move vs leverage. Green = survives the worst "
                 "default melt-up; dashed = Cerebras +13/+26/+39% analogs.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_hedge_chart(ev: ContractEval, hedge_ratios: list[float], out_path: Path) -> None:
    """Total P&L vs settlement close for a range of hedge ratios -- shows the locked floor
    (h=1, flat) vs the directional tilt of partial (h<1) / over (h>1) hedges."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, basis, offer = ev.legs.shares_long, ev.matched_gap, ev.offer
    closes = np.linspace(80.0, 280.0, 400)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for h in hedge_ratios:
        tot = [total_pnl_at_close(h, N, basis, px, offer) for px in closes]
        label = f"h={h:g}" + (" (locked)" if abs(h - 1) < 1e-9 else
                               " (net long)" if h < 1 else " (net short)")
        ax.plot(closes, np.array(tot) / 1e3, lw=2, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(offer, color="tab:gray", ls=":", lw=1)
    ax.text(offer, ax.get_ylim()[0], f" offer ${offer:.0f}", fontsize=8, va="bottom", color="tab:gray")
    ax.set_title(f"{ev.contract.name}: total P&L vs settlement close, by hedge ratio h\n"
                 f"(N={N:,.0f} long, units-matched basis ${basis:,.2f}/sh; h=1 is the flat locked line)")
    ax.set_xlabel("first-day settlement close ($/IPO-share)")
    ax.set_ylabel("total P&L ($000s)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ======================================================================================
# Block S1 (a) — hedge grid: fill-price x fill-fraction x margin-constraint
# ======================================================================================
# The Friday-morning question is not "is the basis positive at $135" but "given the FINAL
# price (135..162), MY fill fraction, and my <EUR2k Hyperliquid margin, how many shares can
# I actually lock, for how much, and what is left naked?" This grid answers every cell.
# Pure math; the CLI layer supplies live mark / funding / EURUSD.

S1_FILL_PRICE_LO = 135.0   # expected offer (S-1/A)
S1_FILL_PRICE_HI = 162.0   # EU prospectus maximum offering price
S1_MELTUP_ANALOGS = (0.13, 0.26, 0.39)  # Cerebras open/high analogs (calc findings)


def fill_price_axis(lo: float = S1_FILL_PRICE_LO, hi: float = S1_FILL_PRICE_HI,
                    step: float = 5.0) -> list[float]:
    """$135 -> $162 in $5 steps, with BOTH corners always present (135 and the 162 EU cap)."""
    out: list[float] = []
    p = lo
    while p < hi - 1e-9:
        out.append(round(p, 2))
        p += step
    if not out or abs(out[0] - lo) > 1e-9:
        out.insert(0, lo)
    out.append(hi)
    return out


def shares_requested(subscription_eur: float, eurusd: float, fill_price: float) -> float:
    """Shares a EUR subscription requests at a given final price (TR pro-rata base).
    EUR -> USD at the live rate, then / price. Fractional shares kept (TR supports them)."""
    if fill_price <= 0:
        return 0.0
    return subscription_eur * eurusd / fill_price


def margin_cap_shares(margin_usd: float, leverage: float, mark: float,
                      contracts_per_share: float) -> float:
    """Max IPO shares hedgeable with `margin_usd` of isolated margin at `leverage`.
    One hedged share needs `contracts_per_share` perp contracts (1.0 for xyz at R=1),
    i.e. notional = contracts_per_share * mark, margin = notional / leverage."""
    if margin_usd <= 0 or mark <= 0 or contracts_per_share <= 0 or leverage <= 0:
        return 0.0
    return margin_usd * leverage / (contracts_per_share * mark)


@dataclass
class GridCell:
    """One (fill price, fill fraction, leverage) cell of the S1 hedge grid."""
    fill_price: float
    fill_frac: float
    leverage: float
    shares_requested: float
    shares_filled: float
    cap_shares: float          # margin-supported ceiling at this leverage
    hedged_shares: float       # min(filled, cap)
    residual_shares: float     # filled - hedged  (UNHEDGED, Frame-B sleeve)
    basis_per_share: float     # per-IPO-share-equiv(mark) - fill_price
    short_notional: float
    margin_used: float
    fee_cost: float
    funding_income: float
    locked_net: float          # hedged*basis - fees + funding  (NET locked $)
    capital: float             # hedged long notional + margin_used (hedge sleeve only)
    roc: float                 # locked_net / capital


def build_hedge_grid(per_ipo_share_mark: float, mark: float, contracts_per_share: float,
                     funding_hourly: float, hours_to_settle: float,
                     fee_bps: float, fee_sides: float,
                     subscription_eur: float, eurusd: float, margin_eur: float,
                     fill_prices: list[float], fill_fracs: list[float],
                     leverages: list[float]) -> list[GridCell]:
    """The full S1(a) grid. No realized-future inputs: everything here is a function of the
    CURRENT mark and declared constraints (decision-safe)."""
    margin_usd = margin_eur * eurusd
    cells: list[GridCell] = []
    for lev in leverages:
        cap = margin_cap_shares(margin_usd, lev, mark, contracts_per_share)
        for price in fill_prices:
            req = shares_requested(subscription_eur, eurusd, price)
            basis = per_ipo_share_mark - price
            for frac in fill_fracs:
                filled = frac * req
                hedged = min(filled, cap)
                residual = filled - hedged
                notional = hedged * contracts_per_share * mark
                margin_used = notional / lev if lev > 0 else 0.0
                fee = notional * (fee_bps / 1e4) * fee_sides
                funding = funding_hourly * notional * hours_to_settle
                locked_net = hedged * basis - fee + funding
                cap_dep = hedged * price + margin_used
                cells.append(GridCell(
                    fill_price=price, fill_frac=frac, leverage=lev,
                    shares_requested=req, shares_filled=filled, cap_shares=cap,
                    hedged_shares=hedged, residual_shares=residual,
                    basis_per_share=basis, short_notional=notional,
                    margin_used=margin_used, fee_cost=fee, funding_income=funding,
                    locked_net=locked_net, capital=cap_dep,
                    roc=(locked_net / cap_dep) if cap_dep > 0 else float("nan"),
                ))
    return cells


def render_grid_text(cells: list[GridCell], contract_name: str, mark: float,
                     eurusd: float, eurusd_note: str, margin_eur: float,
                     subscription_eur: float, hours: float) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f" S1(a) HEDGE GRID  {contract_name} @ mark {mark:,.2f} | "
                 f"EUR{subscription_eur:,.0f} subscription | margin EUR{margin_eur:,.0f} | "
                 f"EURUSD {eurusd:.4f} ({eurusd_note})")
    lines.append(f" cell = hedged shares | NET locked $ (fees+funding over {hours:.0f}h) | "
                 "ROC on hedge-sleeve capital | naked residual shares")
    lines.append("=" * 100)
    levs = sorted({c.leverage for c in cells})
    fracs = sorted({c.fill_frac for c in cells})
    prices = sorted({c.fill_price for c in cells})
    by_key = {(c.leverage, c.fill_price, c.fill_frac): c for c in cells}
    for lev in levs:
        cap = next(c.cap_shares for c in cells if c.leverage == lev)
        lines.append(f"\n  leverage {lev:.1f}x  (margin-supported ceiling = {cap:,.1f} sh)")
        head = "  fill$  " + "".join(f"| fill {f*100:>3.0f}%{'':<17}" for f in fracs)
        lines.append(head)
        lines.append("  " + "-" * (len(head) - 2))
        for price in prices:
            row = f"  {price:>5.0f}  "
            for f in fracs:
                c = by_key[(lev, price, f)]
                row += (f"| {c.hedged_shares:>5.1f}sh {c.locked_net:>+7,.0f}$ "
                        f"{c.roc*100:>4.1f}% r{c.residual_shares:>4.1f} ")
            lines.append(row)
    lines.append("")
    lines.append("  read: hedged = min(fill, margin ceiling); residual r = naked Frame-B shares;")
    lines.append("  NET locked $ = hedged x (mark - fill$) - taker fees + funding carry; ROC on")
    lines.append("  (hedged x fill$ + margin used). Basis goes NEGATIVE when fill$ > mark.")
    return "\n".join(lines)


# ======================================================================================
# Block S1 (b) — basis-decay fit on HL hourly candles (level premium vs $135)
# ======================================================================================
# The perp's premium to the $135 anchor has been bleeding (~+60% mid-May -> ~+20% now).
# To rank "hedge now vs hedge Thursday night vs hedge Friday morning" we need E[basis at
# node], i.e. a causal projection of the premium. Model (stated assumption): exponential
# decay of the LEVEL premium, ln(mark - 135) = a + b*t. Fit by OLS on hourly closes,
# uncertainty by DAILY-BLOCK bootstrap (hourly residuals are autocorrelated; resampling
# whole days is the honest cheap CI). Lookahead guard: only candles whose CLOSE time is
# <= cutoff enter the fit -- enforced inside the fitter and unit-tested.

def fetch_hl_candles(coin: str, interval: str, start_utc: datetime, end_utc: datetime,
                     timeout: float = 30.0) -> list[dict]:
    """Read-only HL candleSnapshot pull (same endpoint family as the snapshot fetch)."""
    import httpx  # local import so --offline / tests don't require network

    req = {"type": "candleSnapshot",
           "req": {"coin": coin, "interval": interval,
                   "startTime": int(start_utc.timestamp() * 1000),
                   "endTime": int(end_utc.timestamp() * 1000)}}
    with httpx.Client(timeout=timeout) as client:
        r = client.post(HL_INFO_URL, json=req)
        r.raise_for_status()
        return r.json()


def save_candles(coin: str, interval: str, candles: list[dict],
                 out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(),
               "coin": coin, "interval": interval, "candles": candles}
    safe = coin.replace(":", "_")
    ts = payload["fetched_at_utc"].replace(":", "").replace("-", "").replace(".", "")[:15]
    p = out_dir / f"candles_{safe}_{interval}_{ts}.json"
    p.write_text(json.dumps(payload))
    (out_dir / f"candles_{safe}_{interval}_latest.json").write_text(json.dumps(payload))
    return p


def load_latest_candles(coin: str, interval: str,
                        out_dir: Path = DEFAULT_OUT_DIR) -> Optional[dict]:
    p = out_dir / f"candles_{coin.replace(':', '_')}_{interval}_latest.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _ols(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    """Stdlib OLS of y on x. Returns (a, b, se_a, se_b) for y = a + b*x."""
    n = len(x)
    if n < 3:
        raise ValueError(f"need >=3 points for OLS, got {n}")
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx <= 0:
        raise ValueError("degenerate x (no time spread)")
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b = sxy / sxx
    a = my - b * mx
    resid = [yi - (a + b * xi) for xi, yi in zip(x, y)]
    s2 = sum(e * e for e in resid) / (n - 2)
    se_b = math.sqrt(s2 / sxx)
    se_a = math.sqrt(s2 * (1.0 / n + mx * mx / sxx))
    return a, b, se_a, se_b


@dataclass
class DecayFit:
    """Causal exponential-decay fit of the level premium (mark - offer)."""
    offer: float
    t0_utc: datetime               # time origin (first candle close used)
    a: float                       # ln(premium) intercept at t0
    b: float                       # ln(premium) slope per DAY (b<0 = decay)
    se_a: float                    # OLS standard errors (NAIVE: hourly autocorrelation
    se_b: float                    #  understates them -- use the bootstrap for CIs)
    n_used: int
    n_dropped_nonpos: int          # candles dropped because premium <= 0 (log undefined)
    window_start_utc: datetime
    window_end_utc: datetime
    boot_ab: list[tuple[float, float]] = field(default_factory=list)  # daily-block bootstrap

    @property
    def half_life_days(self) -> float:
        return (-math.log(2.0) / self.b) if self.b < 0 else float("inf")

    def premium_at(self, t_utc: datetime) -> float:
        """Fitted TREND level at t (diagnostic only -- see mark_proj for decisions)."""
        dt_days = (t_utc - self.t0_utc).total_seconds() / 86400.0
        return math.exp(self.a + self.b * dt_days)

    # Decision projections are RATE-ONLY from the live premium: the level of a traded
    # price is taken as given, and only the fitted decay RATE is applied forward.
    # Projecting from the fitted trend line instead would smuggle in a mean-reversion-
    # to-trend alpha claim (e.g. "today's dip will bounce back"), which a risk rule
    # must not assume. The trend-vs-live endpoint gap is reported as a misfit diagnostic.
    def premium_proj(self, live_premium: float, t_from: datetime, t_utc: datetime) -> float:
        dt_days = (t_utc - t_from).total_seconds() / 86400.0
        return live_premium * math.exp(self.b * dt_days)

    def mark_proj(self, live_premium: float, t_from: datetime, t_utc: datetime) -> float:
        return self.offer + self.premium_proj(live_premium, t_from, t_utc)

    def mark_proj_ci(self, live_premium: float, t_from: datetime, t_utc: datetime,
                     lo_q: float = 0.025, hi_q: float = 0.975) -> tuple[float, float]:
        """Bootstrap percentile CI of the rate-only projection. Reflects DECAY-RATE
        uncertainty only; intraday path noise around the level is extra (flagged)."""
        if not self.boot_ab:
            m = self.mark_proj(live_premium, t_from, t_utc)
            return m, m
        dt_days = (t_utc - t_from).total_seconds() / 86400.0
        draws = sorted(self.offer + live_premium * math.exp(b * dt_days)
                       for _, b in self.boot_ab)
        n = len(draws)
        lo = draws[max(0, min(n - 1, int(lo_q * n)))]
        hi = draws[max(0, min(n - 1, int(hi_q * n)))]
        return lo, hi


def fit_premium_decay(candle_times_utc: list[datetime], closes: list[float], offer: float,
                      cutoff_utc: datetime, interval_seconds: float = 3600.0,
                      boot_n: int = 500, seed: int = 42) -> DecayFit:
    """Fit ln(close - offer) = a + b*days on candles whose CLOSE time is <= cutoff_utc.

    LOOKAHEAD GUARD: candles closing after `cutoff_utc` are excluded HERE, inside the
    fitter, so no caller can accidentally leak future bars into a decision. (The last
    in-progress hourly bar is excluded automatically by the same rule.)
    Uncertainty: daily-block bootstrap (resample whole UTC dates with replacement,
    refit) -- honest under hourly autocorrelation, deterministic via `seed`.
    """
    import random

    pts: list[tuple[datetime, float]] = []
    n_dropped = 0
    for t_open, close in zip(candle_times_utc, closes):
        t_close = t_open + _td(seconds=interval_seconds)
        if t_close > cutoff_utc:
            continue  # close not known at cutoff -> excluded (no lookahead)
        prem = close - offer
        if prem <= 0:
            n_dropped += 1
            continue  # log-decay model undefined at <=0 premium; drop + count
        pts.append((t_close, prem))
    if len(pts) < 3:
        raise ValueError(f"only {len(pts)} usable candles before cutoff -- cannot fit decay")
    pts.sort(key=lambda p: p[0])
    t0 = pts[0][0]
    xs = [(t - t0).total_seconds() / 86400.0 for t, _ in pts]
    ys = [math.log(p) for _, p in pts]
    a, b, se_a, se_b = _ols(xs, ys)

    # daily-block bootstrap on UTC dates
    by_date: dict[str, list[int]] = {}
    for i, (t, _) in enumerate(pts):
        by_date.setdefault(t.strftime("%Y-%m-%d"), []).append(i)
    dates = sorted(by_date)
    boot_ab: list[tuple[float, float]] = []
    if len(dates) >= 3 and boot_n > 0:
        rng = random.Random(seed)
        for _ in range(boot_n):
            idx: list[int] = []
            for _ in dates:
                idx.extend(by_date[rng.choice(dates)])
            try:
                ab = _ols([xs[i] for i in idx], [ys[i] for i in idx])
                boot_ab.append((ab[0], ab[1]))
            except ValueError:
                continue  # degenerate resample (e.g. single date drawn) -- skip
    return DecayFit(offer=offer, t0_utc=t0, a=a, b=b, se_a=se_a, se_b=se_b,
                    n_used=len(pts), n_dropped_nonpos=n_dropped,
                    window_start_utc=pts[0][0], window_end_utc=pts[-1][0],
                    boot_ab=boot_ab)


def _td(seconds: float):
    from datetime import timedelta
    return timedelta(seconds=seconds)


def load_watch_log_marks(watch_dir: Path, contract_name: str) -> list[tuple[datetime, float]]:
    """Supplementary cross-check ONLY (per the gameplan): marks from any --watch parquet
    shards. Never mixed into the candle fit -- compared against it."""
    if not watch_dir.exists():
        return []
    import pyarrow.parquet as pq  # local import; only needed if shards exist

    out: list[tuple[datetime, float]] = []
    for shard in sorted(watch_dir.glob("*.parquet")):
        tbl = pq.read_table(shard, columns=["fetched_at_utc", "contract", "mark"])
        for ts, name, mark in zip(tbl.column("fetched_at_utc").to_pylist(),
                                  tbl.column("contract").to_pylist(),
                                  tbl.column("mark").to_pylist()):
            if name != contract_name:
                continue
            try:
                out.append((datetime.fromisoformat(ts), float(mark)))
            except (ValueError, TypeError):
                continue
    return out


def crosscheck_watchlog_vs_candles(ticks: list[tuple[datetime, float]],
                                   candle_times_utc: list[datetime],
                                   closes: list[float]) -> Optional[dict]:
    """Max |watch-log mark - same-hour candle close|. None if no overlapping ticks."""
    if not ticks:
        return None
    by_hour = {t.replace(minute=0, second=0, microsecond=0): c
               for t, c in zip(candle_times_utc, closes)}
    diffs = []
    for t, mark in ticks:
        key = t.replace(minute=0, second=0, microsecond=0)
        if key in by_hour:
            diffs.append(abs(mark - by_hour[key]))
    if not diffs:
        return None
    return {"n_overlap": len(diffs), "max_abs_diff": max(diffs),
            "mean_abs_diff": sum(diffs) / len(diffs)}


def render_decay_text(fit: DecayFit, nodes: list[tuple[str, datetime]],
                      live_mark: float, now_utc: datetime, crosscheck: Optional[dict],
                      requested_since: datetime) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(" S1(b) BASIS-DECAY FIT  (level premium vs $%.0f, HL hourly closes)" % fit.offer)
    lines.append("=" * 78)
    if fit.window_start_utc - requested_since > _td(seconds=86400):
        lines.append(f"  NOTE: requested history since {requested_since.date()}, venue returned "
                     f"since {fit.window_start_utc.date()} (HL purges old hourly bars).")
    lines.append(f"  window                    = {fit.window_start_utc:%Y-%m-%d %H:%M} -> "
                 f"{fit.window_end_utc:%Y-%m-%d %H:%M} UTC  ({fit.n_used} hourly closes; "
                 f"{fit.n_dropped_nonpos} dropped at premium<=0)")
    lines.append(f"  model                     = ln(mark - {fit.offer:.0f}) = a + b*days  "
                 "[ASSUMPTION: exponential level-premium decay]")
    lines.append(f"  decay rate b              = {fit.b:+.4f}/day  (naive OLS se {fit.se_b:.4f}; "
                 f"CIs below use the daily-block bootstrap, n={len(fit.boot_ab)})")
    hl = fit.half_life_days
    lines.append(f"  premium half-life         = {'inf (premium not decaying)' if math.isinf(hl) else f'{hl:,.1f} days'}")
    live_prem = live_mark - fit.offer
    trend_end = fit.premium_at(fit.window_end_utc)
    lines.append(f"  trend-vs-live misfit      = fitted trend ${trend_end:,.2f} vs live premium "
                 f"${live_prem:,.2f} ({(live_prem/trend_end - 1)*100:+.1f}%) -- diagnostic only")
    lines.append("")
    lines.append("  projected mark (RATE-ONLY from the live premium; bootstrap 95% CI on the rate;")
    lines.append("  intraday path noise around the level is NOT in this band):")
    for name, t in nodes:
        m = fit.mark_proj(live_prem, now_utc, t)
        lo, hi = fit.mark_proj_ci(live_prem, now_utc, t)
        lines.append(f"    {name:<26} {t:%Y-%m-%d %H:%M} UTC -> "
                     f"{m:,.2f}  [{lo:,.2f}, {hi:,.2f}]")
    if crosscheck is None:
        lines.append("\n  watch-log cross-check: no overlapping --watch parquet ticks found "
                     "(shards absent or disjoint) -- fit rests on candles alone.")
    else:
        lines.append(f"\n  watch-log cross-check: {crosscheck['n_overlap']} overlapping ticks, "
                     f"max |mark - candle close| = ${crosscheck['max_abs_diff']:.2f} "
                     f"(mean ${crosscheck['mean_abs_diff']:.2f}) -- supplementary only, "
                     "not mixed into the fit.")
    return "\n".join(lines)


def render_recent_sensitivity(full_fit: DecayFit, recent_fit: Optional[DecayFit],
                              live_mark: float, now_utc: datetime,
                              alloc_node: tuple[str, datetime],
                              recent_days: float) -> str:
    """Regime-shift sensitivity: the same rate-only projection under a fit restricted to
    the last `recent_days`. If the two rates differ materially, the MODEL-FORM uncertainty
    (which regime persists) dominates the bootstrap band and is flagged as such."""
    if recent_fit is None:
        return ("  recent-window sensitivity: not enough recent candles to fit -- "
                "rely on the full-window rate only.")
    name, t = alloc_node
    live_prem = live_mark - full_fit.offer
    m_full = full_fit.mark_proj(live_prem, now_utc, t)
    m_recent = recent_fit.mark_proj(live_prem, now_utc, t)
    lines = [
        f"  recent-window sensitivity (last {recent_days:.0f}d, {recent_fit.n_used} closes): "
        f"b = {recent_fit.b:+.4f}/day vs full-window {full_fit.b:+.4f}/day",
        f"    -> {name} projected mark {m_recent:,.2f} (recent rate) vs {m_full:,.2f} "
        f"(full rate); the decision table uses the FULL-window rate.",
    ]
    if abs(recent_fit.b - full_fit.b) > 2.0 * max(full_fit.se_b, 1e-9):
        lines.append("    NOTE: the two rates differ by >2 naive-se -- a regime shift is "
                     "visible, so model-form uncertainty EXCEEDS the bootstrap band; read "
                     "both projections as the honest range.")
    return "\n".join(lines)


def make_decay_chart(fit: DecayFit, candle_times_utc: list[datetime], closes: list[float],
                     nodes: list[tuple[str, datetime]], live_mark: float,
                     now_utc: datetime, out_path: Path) -> None:
    """Premium level + fitted trend (history) + rate-only projection from the live
    premium (decision path) with its bootstrap band, decision nodes marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    prem = [c - fit.offer for c in closes]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(candle_times_utc, prem, lw=0.8, alpha=0.6, color="tab:blue",
            label="hourly premium (mark - $%.0f)" % fit.offer)
    hist_t = []
    t = fit.window_start_utc
    while t <= fit.window_end_utc:
        hist_t.append(t)
        t += _td(seconds=6 * 3600)
    ax.plot(hist_t, [fit.premium_at(t) for t in hist_t], color="black", lw=1.5, ls=":",
            label=f"fitted trend (diagnostic): b={fit.b:+.4f}/day (half-life "
                  f"{'inf' if math.isinf(fit.half_life_days) else f'{fit.half_life_days:.0f}d'})")
    # decision projection: rate-only from the live premium
    live_prem = live_mark - fit.offer
    last_node_t = max(t for _, t in nodes)
    proj_t = []
    t = now_utc
    while t <= last_node_t:
        proj_t.append(t)
        t += _td(seconds=3 * 3600)
    ax.plot(proj_t, [fit.premium_proj(live_prem, now_utc, t) for t in proj_t],
            color="tab:red", lw=2, label="projection: live premium x fitted decay rate")
    band = [fit.mark_proj_ci(live_prem, now_utc, t) for t in proj_t]
    ax.fill_between(proj_t, [lo - fit.offer for lo, _ in band],
                    [hi - fit.offer for _, hi in band], color="tab:red", alpha=0.15,
                    label="bootstrap 95% CI (decay rate only)")
    for name, t in nodes:
        ax.axvline(t, ls="--", lw=1, color="tab:red", alpha=0.8)
        ax.text(t, ax.get_ylim()[1] * 0.95, " " + name, rotation=90, va="top", fontsize=8,
                color="tab:red")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("xyz:SPCX level premium over the $135 expected offer -- causal decay fit\n"
                 "(hourly HL closes; dashed verticals = pre-hedge decision nodes)")
    ax.set_ylabel("premium ($/share over offer)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ======================================================================================
# Block S1 (c) — pre-hedge timing EV + the pre-registered rule
# ======================================================================================
# Pre-hedging (shorting BEFORE the allocation e-mail) trades a richer basis today against
# the risk of being naked-short on a melt-up if the fill disappoints. Task formula, applied
# per pre-hedged share:
#     EV = P(fill >= prehedge) * E[net basis at node]  -  P(shortfall) * E[naked-short loss]
# with E[naked-short loss] from the Cerebras melt-up analogs. Setting EV = 0 gives the
# pre-registered threshold  Z = (1-p)/p * naked_loss : "hedge X shares at node Y iff net
# basis >= Z". Everything probabilistic here is an ASSUMPTION and is printed as one.
# NO realized-future price enters any of these functions (enforced by a source-inspection
# unit test, like the existing no-lookahead test).

def expected_meltup_move(dist: list[tuple[float, float]]) -> float:
    """E[adverse move] under the melt-up distribution [(move_frac, weight), ...].
    Weights are normalized here so callers can pass unnormalized odds."""
    wsum = sum(w for _, w in dist)
    if wsum <= 0:
        return 0.0
    return sum(m * w for m, w in dist) / wsum


def naked_loss_per_share(mark_at_node: float, e_meltup_move: float,
                         exit_fee_bps: float = 4.5) -> float:
    """Expected $ loss per NAKED pre-hedged share if the fill falls short: forced to buy
    back the uncovered short into the melt-up at mark*(1+move), plus the exit taker fee.
    Conservative by design: the whole pre-hedge is treated as naked on shortfall."""
    move_loss = mark_at_node * e_meltup_move
    exit_fee = mark_at_node * (1.0 + e_meltup_move) * (exit_fee_bps / 1e4)
    return move_loss + exit_fee


def prehedge_ev_per_share(p_fill_ge: float, net_basis_at_node: float,
                          naked_loss: float) -> float:
    """The task formula, per pre-hedged share."""
    return p_fill_ge * net_basis_at_node - (1.0 - p_fill_ge) * naked_loss


def breakeven_net_basis(p_fill_ge: float, naked_loss: float) -> float:
    """Z_naked such that the task-formula EV (pre-hedge vs NEVER hedging) is zero:
    Z = (1-p)/p * naked_loss.  p=1 -> 0 (allocation known); p->0 -> inf (never blind)."""
    if p_fill_ge <= 0:
        return float("inf")
    return (1.0 - p_fill_ge) / p_fill_ge * naked_loss


def prehedge_threshold(p_fill_ge: float, naked_loss: float,
                       projected_alloc_basis: float) -> float:
    """The FULL pre-registered threshold Z*: pre-hedging at an early node must beat not
    just 'never hedge' but 'WAIT and hedge at allocation' (where there is no naked risk
    and the option to skip if the basis is gone, i.e. wait pays p * max(B_alloc, 0)):

        EV(pre-hedge) > EV(wait)
        p*B - (1-p)*NL > p*max(B_alloc, 0)
        B > (1-p)/p * NL + max(B_alloc, 0)  =  Z*

    `projected_alloc_basis` is TODAY'S decay-fit projection of the allocation-node net
    basis -- frozen at pre-registration time so Z* is a constant, not a moving target."""
    return breakeven_net_basis(p_fill_ge, naked_loss) + max(0.0, projected_alloc_basis)


def should_hedge(net_basis: float, z_threshold: float) -> bool:
    """The pre-registered rule, evaluated on the LIVE net basis at the node. Strict
    inequality so the zero-basis corner can never trigger a hedge (even at Z=0)."""
    return net_basis > z_threshold


def net_basis_per_share(mark_at_node: float, fill_price: float, funding_hourly: float,
                        hours_node_to_settle: float, entry_fee_bps: float = 4.5) -> float:
    """Per-share net locked basis if hedged at `mark_at_node` against a `fill_price` long:
    gross (mark - fill$) minus entry taker fee plus funding carry to settlement (funding
    held constant at the current hourly rate -- stated assumption)."""
    gross = mark_at_node - fill_price
    fee = mark_at_node * (entry_fee_bps / 1e4)
    funding = funding_hourly * mark_at_node * max(0.0, hours_node_to_settle)
    return gross - fee + funding


@dataclass
class NodeDecision:
    """One (fill price, node) cell of the pre-registered decision table."""
    node: str
    t_utc: datetime
    fill_price: float
    p_fill_ge: float           # ASSUMPTION (or 1.0 at the allocation node, fill known)
    prehedge_shares: float     # X = min(pessimistic-fill shares, margin ceiling)
    mark_used: float           # live mark (NOW) or rate-only decay projection (D1/D2)
    mark_lo: float
    mark_hi: float
    net_basis: float           # net basis per share at this node's mark
    net_basis_lo: float
    net_basis_hi: float
    naked_loss: float          # NL per share (no risk-relevance at the allocation node)
    proj_alloc_basis: float    # today's projected allocation-node net basis (wait comparator)
    z_naked: float             # (1-p)/p * NL  (pre-hedge vs NEVER hedging)
    z_threshold: float         # Z* = z_naked + max(0, proj_alloc_basis)  (vs WAITING)
    ev_per_share: float        # task-literal EV vs never hedging:  p*B - (1-p)*NL
    ev_vs_wait_per_share: float  # p*(B - max(0, proj_alloc_basis)) - (1-p)*NL
    ev_total: float            # X * ev_vs_wait_per_share (what ranks the nodes)
    armed: bool                # X > 0 and net_basis > Z*  (the pre-registered trigger)


def build_decision_table(fill_prices: list[float], nodes: list[tuple[str, datetime]],
                         now_name: str, live_mark: float, fit: Optional[DecayFit],
                         p_fill_map: dict[float, float], prehedge_fill_frac: float,
                         meltup_dist: list[tuple[float, float]],
                         subscription_eur: float, eurusd: float, margin_eur: float,
                         leverage: float, contracts_per_share: float,
                         funding_hourly: float, settle_utc: datetime,
                         fee_bps: float = 4.5) -> list[NodeDecision]:
    """The S1(c) decision table. Inputs are the CURRENT mark, the causal decay fit, and
    declared assumptions -- no candle list and no settlement price can reach this code,
    so no decision can look ahead. The LAST node is treated as allocation-known (p=1)."""
    e_move = expected_meltup_move(meltup_dist)
    margin_usd = margin_eur * eurusd
    t_now = next(t for n, t in nodes if n == now_name)

    def mark_at_node(name: str, t: datetime) -> tuple[float, float, float]:
        if name == now_name or fit is None:
            return live_mark, live_mark, live_mark
        # rate-only projection from the LIVE premium (no trend-level reversion)
        live_prem = live_mark - fit.offer
        return (fit.mark_proj(live_prem, t_now, t), *fit.mark_proj_ci(live_prem, t_now, t))

    rows: list[NodeDecision] = []
    alloc_name, alloc_t = nodes[-1]
    for price in fill_prices:
        req = shares_requested(subscription_eur, eurusd, price)
        cap = margin_cap_shares(margin_usd, leverage, live_mark, contracts_per_share)
        # today's projection of the allocation-node net basis = the wait comparator
        alloc_mark, _, _ = mark_at_node(alloc_name, alloc_t)
        alloc_hours = (settle_utc - alloc_t).total_seconds() / 3600.0
        alloc_basis = net_basis_per_share(alloc_mark, price, funding_hourly, alloc_hours,
                                          fee_bps)
        for i, (name, t) in enumerate(nodes):
            is_alloc_node = i == len(nodes) - 1
            mark, lo, hi = mark_at_node(name, t)
            hours = (settle_utc - t).total_seconds() / 3600.0
            nb = net_basis_per_share(mark, price, funding_hourly, hours, fee_bps)
            nb_lo = net_basis_per_share(lo, price, funding_hourly, hours, fee_bps)
            nb_hi = net_basis_per_share(hi, price, funding_hourly, hours, fee_bps)
            nl = naked_loss_per_share(mark, e_move, exit_fee_bps=fee_bps)
            if is_alloc_node:
                p = 1.0   # allocation known: no fill uncertainty, no wait comparator left
                x = cap   # ceiling; realized X = min(fill, cap), printed as the formula
                z_naked = 0.0
                z_star = 0.0
            else:
                p = p_fill_map.get(round(prehedge_fill_frac, 4))
                if p is None:
                    raise ValueError(f"no P(fill >= {prehedge_fill_frac:.0%}) assumption "
                                     "provided (--p-fill)")
                x = min(prehedge_fill_frac * req, cap)
                z_naked = breakeven_net_basis(p, nl)
                z_star = prehedge_threshold(p, nl, alloc_basis)
            ev_never = prehedge_ev_per_share(p, nb, nl)
            ev_wait = (nb if is_alloc_node
                       else prehedge_ev_per_share(p, nb - max(0.0, alloc_basis), nl))
            rows.append(NodeDecision(
                node=name, t_utc=t, fill_price=price, p_fill_ge=p, prehedge_shares=x,
                mark_used=mark, mark_lo=lo, mark_hi=hi,
                net_basis=nb, net_basis_lo=nb_lo, net_basis_hi=nb_hi,
                naked_loss=nl, proj_alloc_basis=alloc_basis, z_naked=z_naked,
                z_threshold=z_star, ev_per_share=ev_never,
                ev_vs_wait_per_share=ev_wait, ev_total=x * ev_wait,
                armed=(x > 0 and should_hedge(nb, z_star)),
            ))
    return rows


def render_decision_text(rows: list[NodeDecision], nodes: list[tuple[str, datetime]],
                         prehedge_fill_frac: float, p_fill_map: dict[float, float],
                         meltup_dist: list[tuple[float, float]], margin_eur: float,
                         eurusd: float, eurusd_note: str, leverage: float,
                         live_mark: float, contract_name: str,
                         fit: Optional[DecayFit]) -> str:
    e_move = expected_meltup_move(meltup_dist)
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f" S1(c) PRE-REGISTERED PRE-HEDGE RULE TABLE  ({contract_name}, "
                 f"pre-hedge sized to the {prehedge_fill_frac*100:.0f}%-fill row, "
                 f"leverage {leverage:.1f}x, margin EUR{margin_eur:,.0f})")
    lines.append("=" * 100)
    lines.append("  RULE per row: 'hedge X shares at node Y iff LIVE net basis >= Z'. Z at the")
    lines.append("  early nodes prices BOTH risks of pre-hedging: the naked-shortfall tail AND the")
    lines.append("  option to simply wait and hedge risk-free at allocation (Z = (1-p)/p x naked-")
    lines.append("  loss + today's projected allocation-node basis). Final node = allocation known")
    lines.append("  (p=1, Z=0): hedge min(fill, margin ceiling) iff net basis > 0 -- the Friday gate.")
    lines.append("  Projections only set Z ex ante; the rule binds on the basis OBSERVED at the node.")
    lines.append("")
    node_names = [n for n, _ in nodes]
    head = "  fill$ "
    for n in node_names:
        head += f"| {n:^30} "
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    by_pf: dict[float, list[NodeDecision]] = {}
    for r in rows:
        by_pf.setdefault(r.fill_price, []).append(r)
    for price in sorted(by_pf):
        row = f"  {price:>4.0f}  "
        for r in sorted(by_pf[price], key=lambda r: node_names.index(r.node)):
            flag = "HEDGE" if r.armed else "wait "
            row += (f"| X{r.prehedge_shares:>5.1f} Z{r.z_threshold:>5.2f} "
                    f"B{r.net_basis:>+6.2f} {flag} ")
        lines.append(row)
    lines.append("")
    lines.append("  column glossary: X = pre-hedge shares (min of pessimistic-fill shares and the")
    lines.append("  margin ceiling; at the final node, the ceiling -- realized X = min(fill, ceiling));")
    lines.append("  Z = the pre-registered trigger $/sh (see RULE above; frozen at run time);")
    lines.append("  B = net basis $/sh at the node (live at NOW, rate-only decay projection later;")
    lines.append("  entry fee + funding-to-settle included); HEDGE = B > Z at today's projections --")
    lines.append("  still confirm on the LIVE basis at the node before acting.")
    lines.append("")
    lines.append("  ASSUMPTION LEDGER (everything below is declared, not measured):")
    pf = ", ".join(f"P(fill>={f*100:.0f}%)={p:.2f}" for f, p in sorted(p_fill_map.items()))
    lines.append(f"    - fill probabilities: {pf}  [pure assumption; TR pro-rata is unmodelable"
                 " offline]")
    md = ", ".join(f"+{m*100:.0f}%:{w:g}" for m, w in meltup_dist)
    lines.append(f"    - melt-up distribution (Cerebras analogs, weights): {md} -> "
                 f"E[move] = +{e_move*100:.1f}%")
    lines.append("    - shortfall is total: on a miss the WHOLE pre-hedge is treated as naked and")
    lines.append("      bought back at the melt-up price (conservative); melt-up magnitudes are")
    lines.append("      listing-transition moves applied un-scaled to every node's naked window")
    lines.append("    - funding held constant at the current hourly rate; decay model is")
    lines.append("      exponential in the level premium (see S1(b) ASSUMPTION line)")
    lines.append(f"    - EURUSD {eurusd:.4f} ({eurusd_note}); live mark {live_mark:,.2f}"
                 + ("" if fit is None else
                    f"; decay b={fit.b:+.4f}/day over {fit.n_used} hourly closes"))
    lines.append("    - LIVE-ONLY unknowns this table cannot resolve: the actual fill, the actual")
    lines.append("      Friday basis, book depth at size, funding path, oracle/convert behavior.")
    return "\n".join(lines)


def render_ev_matrix_text(live_mark: float, fit: Optional[DecayFit],
                          nodes: list[tuple[str, datetime]], now_name: str,
                          fill_price: float, pess_fills: list[float],
                          p_fill_map: dict[float, float],
                          meltup_dist: list[tuple[float, float]],
                          subscription_eur: float, eurusd: float, margin_eur: float,
                          leverage: float, contracts_per_share: float,
                          funding_hourly: float, settle_utc: datetime,
                          fee_bps: float = 4.5) -> str:
    """EV (total $) for hedge-at-node x pessimistic-fill, at ONE fill price. The final
    node is allocation-known: p=1, no naked risk, EV = shares x net basis there."""
    e_move = expected_meltup_move(meltup_dist)
    margin_usd = margin_eur * eurusd
    req = shares_requested(subscription_eur, eurusd, fill_price)
    cap = margin_cap_shares(margin_usd, leverage, live_mark, contracts_per_share)
    t_now = next(t for n, t in nodes if n == now_name)
    lines: list[str] = []
    lines.append(f"  EV matrix at fill price ${fill_price:.0f} (re-run with --offer <final> "
                 "after the 424B prints):")
    head = "    pess. fill "
    for n, _ in nodes:
        head += f"| {n:^24} "
    lines.append(head)
    lines.append("    " + "-" * (len(head) - 4))
    for f in pess_fills:
        row = f"    {f*100:>6.0f}%    "
        for i, (name, t) in enumerate(nodes):
            is_alloc = i == len(nodes) - 1
            mark = (live_mark if (name == now_name or fit is None)
                    else fit.mark_proj(live_mark - fit.offer, t_now, t))
            hours = (settle_utc - t).total_seconds() / 3600.0
            nb = net_basis_per_share(mark, fill_price, funding_hourly, hours, fee_bps)
            x = min(f * req, cap)
            if is_alloc:
                ev = x * nb  # fill known: no naked term; this IS the locked EV at that size
                cell = f"X{x:>5.1f} EV${ev:>+8,.0f} p=1 "
            else:
                p = p_fill_map.get(round(f, 4))
                if f <= 0:
                    cell = f"X  0.0 EV$      +0 --   "
                elif p is None:
                    cell = "  (no P(fill) assumption)"
                else:
                    nl = naked_loss_per_share(mark, e_move, exit_fee_bps=fee_bps)
                    ev = x * prehedge_ev_per_share(p, nb, nl)
                    cell = f"X{x:>5.1f} EV${ev:>+8,.0f} p={p:.2f}"
            row += f"| {cell} "
        lines.append(row)
    lines.append("    read: EV in TOTAL $ on the pre-hedged tranche. Early nodes carry the naked-")
    lines.append("    shortfall penalty; the final (allocation) node has none but a decayed basis.")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# S1 CSV outputs (result tables -> csv_outputs/market_maps per the repo CSV convention)
# --------------------------------------------------------------------------------------
def write_s1_csvs(cells: list[GridCell], rows: list[NodeDecision], out_dir: Path) -> list[Path]:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    p1 = out_dir / "spcx_s1_hedge_grid.csv"
    with p1.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fill_price", "fill_frac", "leverage", "shares_requested", "shares_filled",
                    "cap_shares", "hedged_shares", "residual_shares", "basis_per_share",
                    "short_notional", "margin_used", "fee_cost", "funding_income",
                    "locked_net", "capital", "roc"])
        for c in cells:
            w.writerow([c.fill_price, c.fill_frac, c.leverage,
                        round(c.shares_requested, 3), round(c.shares_filled, 3),
                        round(c.cap_shares, 3), round(c.hedged_shares, 3),
                        round(c.residual_shares, 3), round(c.basis_per_share, 4),
                        round(c.short_notional, 2), round(c.margin_used, 2),
                        round(c.fee_cost, 2), round(c.funding_income, 2),
                        round(c.locked_net, 2), round(c.capital, 2), round(c.roc, 6)])
    paths.append(p1)
    p2 = out_dir / "spcx_s1_decision_table.csv"
    with p2.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fill_price", "node", "t_utc", "p_fill_ge", "prehedge_shares",
                    "mark_used", "mark_lo", "mark_hi", "net_basis", "net_basis_lo",
                    "net_basis_hi", "naked_loss", "proj_alloc_basis", "z_naked",
                    "z_threshold", "ev_vs_never_per_share", "ev_vs_wait_per_share",
                    "ev_total", "armed"])
        for r in rows:
            w.writerow([r.fill_price, r.node, r.t_utc.isoformat(), r.p_fill_ge,
                        round(r.prehedge_shares, 3), round(r.mark_used, 4),
                        round(r.mark_lo, 4), round(r.mark_hi, 4), round(r.net_basis, 4),
                        round(r.net_basis_lo, 4), round(r.net_basis_hi, 4),
                        round(r.naked_loss, 4), round(r.proj_alloc_basis, 4),
                        round(r.z_naked, 4), round(r.z_threshold, 4),
                        round(r.ev_per_share, 4), round(r.ev_vs_wait_per_share, 4),
                        round(r.ev_total, 2), r.armed])
    paths.append(p2)
    return paths


# --------------------------------------------------------------------------------------
# S1 live inputs: EURUSD (live, flagged) + candles (cached like snapshots)
# --------------------------------------------------------------------------------------
def fetch_eurusd(timeout: float = 15.0) -> tuple[float, str]:
    """Live EURUSD with the source flagged. Yahoo intraday first (same chart API the
    Cerebras case study uses), open.er-api.com daily as fallback."""
    import httpx

    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X?range=1d&interval=1h"
        r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        r.raise_for_status()
        meta = r.json()["chart"]["result"][0]["meta"]
        return float(meta["regularMarketPrice"]), "Yahoo EURUSD=X intraday"
    except Exception:  # noqa: BLE001 - fall through to the daily source
        pass
    r = httpx.get("https://open.er-api.com/v6/latest/EUR", timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return float(d["rates"]["USD"]), f"open.er-api.com daily ({d.get('time_last_update_utc', '?')})"


def resolve_eurusd(args: argparse.Namespace) -> tuple[float, str]:
    """--eurusd override > live fetch (cached for --offline) > cached > error."""
    fx_cache = args.out_dir / "fx_latest.json"
    if args.eurusd is not None:
        return args.eurusd, "OVERRIDE (--eurusd)"
    if not args.offline:
        try:
            rate, src = fetch_eurusd()
            args.out_dir.mkdir(parents=True, exist_ok=True)
            fx_cache.write_text(json.dumps({
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "rate": rate, "source": src}))
            return rate, src
        except Exception as exc:  # noqa: BLE001 - fall back to cache, flagged
            if fx_cache.exists():
                d = json.loads(fx_cache.read_text())
                return d["rate"], f"CACHED {d['source']} @ {d['fetched_at_utc']} (live failed: {exc})"
            raise SystemExit(f"ERROR: EURUSD fetch failed ({exc}) and no cache; pass --eurusd.")
    if fx_cache.exists():
        d = json.loads(fx_cache.read_text())
        return d["rate"], f"CACHED {d['source']} @ {d['fetched_at_utc']}"
    raise SystemExit("ERROR: --offline with no cached EURUSD; pass --eurusd.")


def get_candles(args: argparse.Namespace, coin: str,
                since_utc: datetime) -> tuple[list[datetime], list[float], str]:
    """Hourly candles for the decay fit: live fetch + cache, or cache under --offline."""
    cached = load_latest_candles(coin, "1h", args.out_dir)
    if not args.offline:
        try:
            candles = fetch_hl_candles(coin, "1h", since_utc, datetime.now(timezone.utc))
            save_candles(coin, "1h", candles, args.out_dir)
            note = "LIVE"
        except Exception as exc:  # noqa: BLE001 - fall back to cache, flagged
            if cached is None:
                raise SystemExit(f"ERROR: candle fetch failed ({exc}) and no cache.")
            candles, note = cached["candles"], f"CACHED @ {cached['fetched_at_utc']} (live failed)"
    else:
        if cached is None:
            raise SystemExit("ERROR: --offline with no cached candles; run once online.")
        candles, note = cached["candles"], f"CACHED @ {cached['fetched_at_utc']}"
    times = [datetime.fromtimestamp(c["t"] / 1000, timezone.utc) for c in candles]
    closes = [float(c["c"]) for c in candles]
    return times, closes, note


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_weighted_map(spec: str) -> list[tuple[float, float]]:
    """'0.13:1,0.26:1,0.39:1' -> [(0.13,1.0), ...] (weights normalized downstream)."""
    out = []
    for part in spec.split(","):
        if not part.strip():
            continue
        k, v = part.split(":")
        out.append((float(k), float(v)))
    return out


def run_s1(args: argparse.Namespace) -> int:
    """Drive S1 (a)+(b)+(c): grid, decay fit, decision table, per the flags given."""
    snap, snap_age = get_snapshot(args)
    contracts = contracts_from_snapshot(snap, ipo_base=args.ipo_base, xyz_base=args.xyz_base)
    key = "xyz" if args.contract == "both" else args.contract
    c = contracts.get(key)
    if c is None:
        print(f"ERROR: contract {key} not in snapshot.", file=sys.stderr)
        return 2
    if args.contract == "both":
        print(f"[S1] --contract both -> grid/decision computed on {c.name} "
              "(the gameplan's hedge instrument; pass --contract vntl to override).\n")
    now_utc = datetime.now(timezone.utc)
    pse = per_ipo_share_equiv(c.mark, c.base, c.units, args.ipo_base)
    cps = fdv_neutral_contract_count(1.0, args.ipo_base, c.base, c.units, c.mark)
    settle_utc = _parse_utc(args.settle_utc)
    hours_now = max(0.0, (settle_utc - now_utc).total_seconds() / 3600.0)
    nodes = [("NOW", now_utc), ("D1 pricing-night", _parse_utc(args.node_d1_utc)),
             ("D2 allocation", _parse_utc(args.node_d2_utc))]
    print(f"[S1] snapshot: {snap_age} | {c.name} mark {c.mark:,.2f} "
          f"(per-IPO-share ${pse:,.2f}) | funding {c.funding_hourly*100:+.6f}%/hr | "
          f"settle {settle_utc:%Y-%m-%d %H:%M} UTC ({hours_now:.0f}h away)\n")

    need_grid = args.grid or args.decision
    need_fit = args.decay_fit or args.decision

    eurusd, fx_note = (resolve_eurusd(args) if (need_grid or args.decision)
                       else (float("nan"), "n/a"))

    fit: Optional[DecayFit] = None
    if need_fit:
        since = _parse_utc(args.decay_since)
        times, closes, candle_note = get_candles(args, c.name, since)
        try:
            fit = fit_premium_decay(times, closes, args.offer, cutoff_utc=now_utc,
                                    boot_n=args.boot_n)
        except ValueError as exc:
            print(f"[S1] decay fit unavailable ({exc}); decision falls back to the live "
                  "mark at every node.", file=sys.stderr)
        if fit is not None:
            ticks = load_watch_log_marks(args.out_dir / "watch_log", c.name)
            xc = crosscheck_watchlog_vs_candles(ticks, times, closes)
            proj_nodes = nodes[1:] + [("settlement", settle_utc)]
            recent_start = now_utc - _td(seconds=args.decay_recent_days * 86400)
            try:
                recent_fit: Optional[DecayFit] = fit_premium_decay(
                    [t for t in times if t >= recent_start],
                    [cl for t, cl in zip(times, closes) if t >= recent_start],
                    args.offer, cutoff_utc=now_utc, boot_n=0)
            except ValueError:
                recent_fit = None
            print(render_decay_text(fit, proj_nodes, c.mark, now_utc, xc, since))
            print(render_recent_sensitivity(fit, recent_fit, c.mark, now_utc,
                                            nodes[-1], args.decay_recent_days))
            print(f"  candles: {candle_note}\n")
            if args.decay_chart:
                make_decay_chart(fit, times, closes, proj_nodes, c.mark, now_utc,
                                 args.decay_chart)
                print(f"[S1] wrote decay chart -> {args.decay_chart}\n")

    fee_sides = 2.0 if (c.settlement == CONVERT_IN_PLACE or args.fee_both_sides) else 1.0
    grid_cells: list[GridCell] = []
    if need_grid:
        grid_cells = build_hedge_grid(
            per_ipo_share_mark=pse, mark=c.mark, contracts_per_share=cps,
            funding_hourly=c.funding_hourly, hours_to_settle=hours_now,
            fee_bps=args.hl_fee_bps, fee_sides=fee_sides,
            subscription_eur=args.subscription_eur, eurusd=eurusd,
            margin_eur=args.margin_eur, fill_prices=fill_price_axis(),
            fill_fracs=[float(x) for x in args.grid_fills.split(",")],
            leverages=[float(x) for x in args.grid_leverages.split(",")])
        print(render_grid_text(grid_cells, c.name, c.mark, eurusd, fx_note,
                               args.margin_eur, args.subscription_eur, hours_now) + "\n")

    if args.decision:
        p_fill_map = {round(k, 4): v for k, v in _parse_weighted_map(args.p_fill)}
        meltup = _parse_weighted_map(args.meltup_dist)
        lev = max(float(x) for x in args.grid_leverages.split(","))
        rows = build_decision_table(
            fill_prices=fill_price_axis(), nodes=nodes, now_name="NOW",
            live_mark=c.mark, fit=fit, p_fill_map=p_fill_map,
            prehedge_fill_frac=args.prehedge_fill, meltup_dist=meltup,
            subscription_eur=args.subscription_eur, eurusd=eurusd,
            margin_eur=args.margin_eur, leverage=lev, contracts_per_share=cps,
            funding_hourly=c.funding_hourly, settle_utc=settle_utc,
            fee_bps=args.hl_fee_bps)
        print(render_decision_text(rows, nodes, args.prehedge_fill, p_fill_map, meltup,
                                   args.margin_eur, eurusd, fx_note, lev, c.mark,
                                   c.name, fit) + "\n")
        print(render_ev_matrix_text(
            c.mark, fit, nodes, "NOW", args.offer, [0.0, 0.10, 0.25], p_fill_map, meltup,
            args.subscription_eur, eurusd, args.margin_eur, lev, cps,
            c.funding_hourly, settle_utc, args.hl_fee_bps) + "\n")
        csv_dir = Path(__file__).resolve().parents[1] / "data" / "analysis" / "csv_outputs" / "market_maps"
        for p in write_s1_csvs(grid_cells, rows, csv_dir):
            print(f"[S1] wrote {p}")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offer", type=float, default=IPO_OFFER_DEFAULT, help="IPO offer price (default 135)")
    p.add_argument("--ipo-base", type=float, default=float(IPO_BASE_DEFAULT),
                   help="IPO share base (default 13,075,865,175)")
    p.add_argument("--xyz-base", type=float, default=float(IPO_BASE_DEFAULT),
                   help=f"xyz:SPCX per-share base (default = IPO base; pass {XYZ_SPLIT_HYPOTHESIS_BASE} "
                        "to stress the split-adjusted hypothesis)")
    p.add_argument("--contract", choices=["xyz", "vntl", "both"], default="both")
    # legs -- short sizing (pick one; default is FDV-neutral h=1)
    p.add_argument("--shares", type=float, default=1000.0, help="IPO shares filled (long)")
    p.add_argument("--hedge-ratio", type=float, default=1.0,
                   help="FDV-anchored hedge ratio h = short/FDV-neutral (default 1.0 = fully hedged; "
                        "0.5 = half-hedged net-long; 1.5 = over-hedged net-short)")
    p.add_argument("--short-contracts", type=float, default=None,
                   help="override: short this many perp contracts (sets h = contracts/FDV-neutral)")
    p.add_argument("--short-notional", type=float, default=None,
                   help="override: USD short notional (sets h from the implied contract count)")
    # margin / leverage (default UNLEVERED)
    p.add_argument("--margin", type=float, default=None, help="USD isolated margin posted")
    p.add_argument("--leverage", type=float, default=1.0,
                   help="effective leverage if --margin not given (default 1.0 = UNLEVERED)")
    p.add_argument("--hl-fee-bps", type=float, default=4.5)
    p.add_argument("--fee-both-sides", action="store_true")
    p.add_argument("--fx-cost-bps", type=float, default=0.0)
    p.add_argument("--stable-basis-bps", type=float, default=0.0)
    p.add_argument("--hours-to-settle", type=float, default=72.0)
    p.add_argument("--close-grid", default="100,135,167,222",
                   help="comma settlement-close prices ($/IPO-share) for the residual sweep")
    # scenarios
    p.add_argument("--scenarios", default="0.13,0.26,0.39",
                   help="comma adverse-move fractions (default Cerebras 13/26/39%%)")
    # data source
    p.add_argument("--offline", action="store_true", help="use last cached snapshot, do not fetch")
    p.add_argument("--snapshot", type=Path, default=None, help="explicit snapshot json to use")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--chart", type=Path, default=None, help="write the liquidation-survival chart PNG here")
    p.add_argument("--hedge-chart", type=Path, default=None,
                   help="write the P&L-vs-close-by-hedge-ratio chart PNG here (uses first contract)")
    p.add_argument("--json", action="store_true", help="also print machine-readable JSON")
    # live monitor (localhost terminal dashboard)
    p.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                   help="live-monitor mode: re-fetch + re-render every SECONDS (terminal dashboard)")
    p.add_argument("--duration-min", type=float, default=None,
                   help="auto-stop the --watch loop after this many minutes")
    p.add_argument("--live-entry", type=float, default=None,
                   help="declared short ENTRY price (enables the live liquidation-buffer panel)")
    p.add_argument("--live-short-notional", type=float, default=None,
                   help="declared live short notional at entry (for the live buffer panel)")
    p.add_argument("--live-margin", type=float, default=None,
                   help="declared live isolated margin (for the live buffer panel)")
    p.add_argument("--alert-buffer-pct", type=float, default=10.0,
                   help="WARN band + terminal bell when the live liq buffer thins below this %% (default 10)")
    p.add_argument("--parquet-log", action="store_true",
                   help="append-only Parquet tick shards under <out-dir>/watch_log/ during --watch")
    # ---- Block S1: hedge grid + basis-decay fit + pre-hedge timing rule ----
    p.add_argument("--grid", action="store_true",
                   help="S1(a): print the fill-price x fill-fraction x margin hedge grid")
    p.add_argument("--decay-fit", action="store_true",
                   help="S1(b): fit + print the basis-decay model on HL hourly candles")
    p.add_argument("--decision", action="store_true",
                   help="S1(c): print the one-page pre-registered pre-hedge rule table "
                        "(implies the grid and the decay fit; writes the S1 CSVs)")
    p.add_argument("--subscription-eur", type=float, default=10_000.0,
                   help="TR subscription size in EUR (default 10000)")
    p.add_argument("--margin-eur", type=float, default=2_000.0,
                   help="free Hyperliquid margin in EUR for the short (default 2000)")
    p.add_argument("--eurusd", type=float, default=None,
                   help="override the live EURUSD rate (else fetched + source flagged)")
    p.add_argument("--grid-fills", default="0.10,0.25,0.50,1.0",
                   help="comma fill fractions for the grid (default 10/25/50/100%%)")
    p.add_argument("--grid-leverages", default="1.0,1.5",
                   help="comma leverages for the grid / decision margin ceiling (default 1.0,1.5)")
    p.add_argument("--prehedge-fill", type=float, default=0.10,
                   help="pessimistic fill fraction the pre-hedge is sized to (default 0.10; "
                        "the gameplan caps pre-hedges at the 10%%-fill row)")
    p.add_argument("--p-fill", default="0.10:0.80,0.25:0.50",
                   help="ASSUMPTION: comma map fill_frac:P(fill>=frac) (default "
                        "'0.10:0.80,0.25:0.50')")
    p.add_argument("--meltup-dist", default="0.13:1,0.26:1,0.39:1",
                   help="ASSUMPTION: melt-up move:weight map from the Cerebras analogs "
                        "(default equal weights on +13/+26/+39%%)")
    p.add_argument("--decay-since", default="2026-05-14T00:00:00Z",
                   help="fit window start (HL may have purged earlier bars; flagged if so)")
    p.add_argument("--boot-n", type=int, default=500,
                   help="daily-block bootstrap resamples for the decay-fit CI (default 500)")
    p.add_argument("--decay-recent-days", type=float, default=7.0,
                   help="window for the regime-shift sensitivity refit (default last 7 days)")
    p.add_argument("--node-d1-utc", default="2026-06-11T20:00:00Z",
                   help="D1 pricing-night node (default Thu 22:00 CEST = 20:00 UTC)")
    p.add_argument("--node-d2-utc", default="2026-06-12T06:00:00Z",
                   help="D2 allocation node (default Fri 08:00 CEST = 06:00 UTC)")
    p.add_argument("--settle-utc", default="2026-06-12T20:00:00Z",
                   help="settlement reference for funding accrual (default Fri 16:00 ET close)")
    p.add_argument("--decay-chart", type=Path, default=None,
                   help="write the decay-fit + projection chart PNG here")
    return p.parse_args(argv)


def get_snapshot(args: argparse.Namespace, allow_fetch: bool = True) -> tuple[dict, str]:
    if args.snapshot is not None:
        snap = json.loads(args.snapshot.read_text())
        return snap, f"file {args.snapshot.name} @ {snap.get('fetched_at_utc','?')}"
    if args.offline:
        snap = load_latest_snapshot(args.out_dir)
        if snap is None:
            print("ERROR: --offline but no cached snapshot found; run once online first.",
                  file=sys.stderr)
            sys.exit(2)
        return snap, f"CACHED (dated) @ {snap.get('fetched_at_utc','?')}"
    if not allow_fetch:
        snap = load_latest_snapshot(args.out_dir)
        if snap is not None:
            return snap, f"CACHED @ {snap.get('fetched_at_utc','?')}"
    try:
        snap = fetch_hl_snapshot()
        save_snapshot(snap, args.out_dir)
        return snap, f"LIVE @ {snap['fetched_at_utc']}"
    except Exception as exc:  # noqa: BLE001 - fall back to cache, flagged as dated
        snap = load_latest_snapshot(args.out_dir)
        if snap is None:
            print(f"ERROR: live fetch failed ({exc}) and no cache available.", file=sys.stderr)
            sys.exit(2)
        return snap, f"CACHED (live fetch FAILED: {exc}) @ {snap.get('fetched_at_utc','?')}"


def build_evals(args: argparse.Namespace, snap: dict, scenarios: list[float],
                close_grid: list[float]) -> list[ContractEval]:
    contracts = contracts_from_snapshot(snap, ipo_base=args.ipo_base, xyz_base=args.xyz_base)
    want = ["xyz", "vntl"] if args.contract == "both" else [args.contract]
    evals: list[ContractEval] = []
    for key in want:
        c = contracts.get(key)
        if c is None:
            continue
        # FDV-neutral count for THIS contract drives hedge-ratio sizing
        neutral = fdv_neutral_contract_count(args.shares, args.ipo_base, c.base, c.units, c.mark)
        if args.short_contracts is not None:
            short_contracts = args.short_contracts
            short_notional = short_contracts * c.mark
        elif args.short_notional is not None:
            short_notional = args.short_notional
        else:  # default + --hedge-ratio path
            short_contracts = short_contracts_from_hedge(args.hedge_ratio, neutral)
            short_notional = short_contracts * c.mark
        margin = args.margin if args.margin is not None else short_notional / args.leverage
        legs = Legs(
            shares_long=args.shares, short_notional=short_notional, posted_margin=margin,
            hl_fee_bps=args.hl_fee_bps, fee_both_sides=args.fee_both_sides,
            fx_cost_bps=args.fx_cost_bps, stable_basis_bps=args.stable_basis_bps,
            hours_to_settle=args.hours_to_settle,
        )
        evals.append(ContractEval(contract=c, offer=args.offer, ipo_base=args.ipo_base,
                                  legs=legs, scenarios=scenarios, close_grid=close_grid))
    return evals


def render_live_buffer_panel(ev: ContractEval, args: argparse.Namespace) -> Optional[str]:
    """Front-and-center liquidation-buffer line for a DECLARED live short (entry fixed)."""
    if args.live_entry is None or args.live_margin is None or args.live_short_notional is None:
        return None
    L = args.live_short_notional / args.live_margin
    liq_px = liq_price_short(args.live_entry, L, ev.mmr)
    summ = liq_buffer_summary(ev.contract.mark, liq_px, alert_pct=args.alert_buffer_pct / 100.0)
    bell = "\a" if summ["band"] in ("WARN", "BREACH") else ""
    return (f"{bell}  LIVE SHORT  entry {args.live_entry:,.2f} @ L={L:.2f}x -> liq {liq_px:,.2f} | "
            f"mark {ev.contract.mark:,.2f} | buffer {summ['buffer_frac']*100:+.1f}%  [{summ['band']}]")


def run_watch(args: argparse.Namespace, scenarios: list[float], close_grid: list[float]) -> int:
    import time  # stdlib

    interval = max(1.0, float(args.watch))
    deadline = None
    if args.duration_min:
        # no Date.now in workflow scripts, but this is a plain CLI; time.monotonic is fine
        deadline = time.monotonic() + args.duration_min * 60.0
    tick_rows: list[dict] = []
    flush_every = 30
    parquet_dir = args.out_dir / "watch_log"
    print(f"[watch] live monitor every {interval:.0f}s "
          f"({'parquet log ON' if args.parquet_log else 'no parquet log'}); Ctrl-C to stop.\n")
    try:
        while True:
            snap, snap_age = get_snapshot(args, allow_fetch=True)
            evals = build_evals(args, snap, scenarios, close_grid)
            print("\033[2J\033[H", end="")  # clear screen + home
            for ev in evals:
                print(render_text(ev, snap_age))
                panel = render_live_buffer_panel(ev, args)
                if panel:
                    print(panel + "\n")
                if args.parquet_log:
                    tick_rows.append(flatten_eval_row(ev, snap.get("fetched_at_utc", "?")))
            if args.parquet_log and len(tick_rows) >= flush_every:
                _flush_parquet(tick_rows, parquet_dir)
                tick_rows = []
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[watch] stopped.")
    finally:
        if args.parquet_log and tick_rows:
            _flush_parquet(tick_rows, parquet_dir)
    return 0


def _flush_parquet(rows: list[dict], parquet_dir: Path) -> None:
    """Write a NEW append-only shard (never edit in place, per repo invariant)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    out = parquet_dir / f"spcx_watch_{stamp}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(f"[watch] flushed {len(rows)} rows -> {out.name}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    scenarios = [float(x) for x in args.scenarios.split(",") if x.strip()]
    close_grid = [float(x) for x in args.close_grid.split(",") if x.strip()]

    if args.watch is not None:
        return run_watch(args, scenarios, close_grid)

    if args.grid or args.decay_fit or args.decision:
        return run_s1(args)

    snap, snap_age = get_snapshot(args)
    evals = build_evals(args, snap, scenarios, close_grid)
    if not evals:
        print("WARNING: no requested contracts in snapshot.", file=sys.stderr)
    for ev in evals:
        print(render_text(ev, snap_age))

    if args.chart and evals:
        make_chart(evals, scenarios, args.chart)
        print(f"[chart] wrote {args.chart}")

    if args.hedge_chart and evals:
        make_hedge_chart(evals[0], [0.0, 0.5, 1.0, 1.5], args.hedge_chart)
        print(f"[hedge-chart] wrote {args.hedge_chart}")

    if args.json:
        print(json.dumps({"snapshot": snap_age, "evals": [eval_to_dict(e) for e in evals]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
