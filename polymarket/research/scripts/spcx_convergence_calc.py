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
