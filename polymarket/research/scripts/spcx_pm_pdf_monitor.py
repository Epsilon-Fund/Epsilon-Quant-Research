"""SPCX live PM-PDF + cross-venue implied-price monitor (Block S5).

PLAIN ENGLISH
-------------
On SPCX listing day (2026-06-12) the PEAK call and the Polymarket tail-selling need the
PM-implied closing-cap distribution recomputed continuously: when the stock rips, the
>$2.4T / >$3T tails reprice in minutes -- exactly when selling them is the trade -- and
nobody can refit a survivor curve by hand mid-session. This script polls the Polymarket
CLOB (16-strike "closing market cap above" ladder + the 7-bucket "cap between" market,
executable BID/ASK, never last-trade) and the Hyperliquid xyz:SPCX perp every 30-60s,
fits a monotone PCHIP survivor S(K) = P(close cap > K), derives the PDF and the implied
distribution stats in market-cap AND per-share terms, compares the bucket market against
the ladder (the addendum's $1.5-2.0T mispricing diagnostic, live), and shows the perp /
PM-implied gap. After the IPO cross, pass --spot to compare crowd-implied vs traded.

CONSTRUCTION CAVEAT (read spacex_pdf_construction_audit.md). Differentiating a PCHIP
interpolant through kinky survivor points manufactures spurious local PDF maxima. The
central stats printed here (P(win), mean, median, P25-P95) are construction-invariant;
MODE and the extreme tails are shape-fragile -- the monitor prints them flagged. PCHIP
is used deliberately so the numbers are comparable with Alvaro's pipeline; this is a
monitoring convention, not an endorsement of the shape.

READ-ONLY: no orders, no DB, no server. One static --html file with meta-refresh at most.

RUN
---
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py                  # one poll, terminal
    PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --watch 45       # poll every 45s
    PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --watch 45 \
        --html data/analysis/spcx_convergence/pm_pdf_monitor.html --parquet-log
    PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --spot 168.5     # post-cross compare
    PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --watch 45 --spot-ws alpaca   # auto spot
    PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --from-json snap.json   # offline replay
    PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --reconcile      # the -3.3 EV sweep
Tests: PYTHONPATH=. uv run pytest tests/test_spcx_pm_pdf_monitor.py -q

SPOT FEED (Block S5b). --spot-ws alpaca streams last trades from Alpaca's free IEX feed
(wss://stream.data.alpaca.markets/v2/iex; auth from ALPACA_KEY_ID / ALPACA_SECRET_KEY env
vars, never hardcoded). IEX is ~2% of consolidated US tape: signal-grade for "where is it
trading", not queue-grade. Manual --spot always overrides the feed. A dead/stale feed
degrades to a STALE/no-prints line and never interrupts PM/HL polling.

DASHBOARD (Block S5c). --html renders a self-contained single-file dashboard (atomic
temp+rename writes, inline-SVG charts, no server/CDN): status strip, headline tiles with
deltas, survivor + PDF + session time-series + PEAK/tail charts, bucket table, alert ledger.

PLAYBOOK PANEL (Block S5d). On the --html page, a NOW card renders the gameplan decision
tree live: feed day state with --fill / --hedged --hedge-entry / --cross HH:MM PRICE /
--sold (persisted to --playbook-state JSON, restarts resume). It quotes the frozen
gameplan/S1/S2 rules with live evaluation — checklist renderer only, no orders/signals.
    # listing-day full stack:
    PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --watch 45 --parquet-log \
        --spot-ws alpaca --html data/analysis/spcx_convergence/pm_pdf_dashboard.html \
        --fill 40 --offer 135
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Constants / conventions
# --------------------------------------------------------------------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
HL_INFO_URL = "https://api.hyperliquid.xyz/info"

LADDER_EVENT_SLUG = "spacex-ipo-closing-market-cap-above"
BUCKET_EVENT_SLUG = "spacex-ipo-closing-market-cap"

SHARES_PRIMARY = 13_075_865_175      # S-1/A no-option Class A + Class B (vault convention)
SHARES_COWORKER = 13_091_000_000     # Alvaro's rounded convention (~0.1% high) — comparison only
IPO_OFFER_DEFAULT = 135.0
PERP_MARK_2026_06_07 = 173.53        # xyz:SPCX mark at the coworker snapshot (handoff note)

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "analysis" / "spcx_convergence"
PARQUET_LOG_DIR = DEFAULT_OUT_DIR / "pm_pdf_log"

# Survivor-tail anchors, the coworker's convention (kept for comparability; the audit showed
# they inflate the far right tail — central stats are unaffected). S(0)=1 is always prepended.
TAIL_ANCHORS = ((4.5, 0.005), (5.0, 0.001))
GRID_N = 20_001
CAP_GRID_MAX = 5.0  # T

# The 2026-06-07 coworker snapshot (mids basis), embedded as the offline reproduction
# fixture: (strike_T, YES_best_ask_cents, NO_best_ask_cents). Identical arrays to
# spacex_pdf_builder_v2.py; YES bid = 100 - NO ask.
FIXTURE_2026_06_07_LADDER = [
    (1.0, 99.2, 1.4), (1.2, 98.3, 1.8), (1.4, 96.6, 3.6), (1.6, 91.5, 8.8),
    (1.8, 78.0, 23.0), (2.0, 64.0, 37.0), (2.2, 46.0, 55.0), (2.4, 30.0, 71.0),
    (2.6, 15.0, 86.0), (2.8, 11.0, 90.0), (3.0, 7.0, 94.0), (3.2, 3.6, 96.7),
    (3.4, 2.6, 97.8), (3.6, 2.4, 98.5), (3.8, 1.8, 98.5), (4.0, 1.3, 99.0),
]
FIXTURE_2026_06_07_BUCKETS = [
    ("<1.0T", 0.0, 1.0, 0.7, 99.4), ("1.0-1.5T", 1.0, 1.5, 3.9, 97.0),
    ("1.5-2.0T", 1.5, 2.0, 40.0, 62.0), ("2.0-2.5T", 2.0, 2.5, 43.0, 58.0),
    ("2.5-3.0T", 2.5, 3.0, 12.7, 88.8), ("3.0-3.5T", 3.0, 3.5, 4.1, 97.3),
    ("3.5T+", 3.5, 5.0, 1.2, 99.1),
]


def cap_t_to_per_share(cap_t: float, shares: float) -> float:
    """$/share equivalent of a market cap quoted in trillions, on a given share base."""
    return cap_t * 1e12 / shares


def per_share_to_cap_t(price: float, shares: float) -> float:
    return price * shares / 1e12


# --------------------------------------------------------------------------------------
# Monotone PCHIP (Fritsch–Carlson, scipy-compatible knot slopes) — numpy only, no scipy
# --------------------------------------------------------------------------------------
def _edge_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    """scipy PchipInterpolator one-sided endpoint derivative with shape preservation."""
    m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(m) != np.sign(d0):
        return 0.0
    if np.sign(d0) != np.sign(d1) and abs(m) > 3.0 * abs(d0):
        return 3.0 * d0
    return float(m)


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson monotone knot derivatives (the weighted-harmonic-mean rule)."""
    h = np.diff(x)
    d = np.diff(y) / h
    n = len(x)
    m = np.zeros(n)
    for k in range(1, n - 1):
        if d[k - 1] * d[k] <= 0.0:
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    if n == 2:
        m[0] = m[1] = d[0]
    else:
        m[0] = _edge_slope(h[0], h[1], d[0], d[1])
        m[-1] = _edge_slope(h[-1], h[-2], d[-1], d[-2])
    return m


def pchip_eval(x: np.ndarray, y: np.ndarray, m: np.ndarray, t: np.ndarray,
               derivative: bool = False) -> np.ndarray:
    """Evaluate the cubic Hermite interpolant (or its derivative) at points t."""
    t = np.asarray(t, dtype=float)
    idx = np.clip(np.searchsorted(x, t, side="right") - 1, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    s = (t - x[idx]) / h
    if not derivative:
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return y[idx] * h00 + h * m[idx] * h10 + y[idx + 1] * h01 + h * m[idx + 1] * h11
    d00 = 6 * s**2 - 6 * s
    d10 = 3 * s**2 - 4 * s + 1
    d01 = -6 * s**2 + 6 * s
    d11 = 3 * s**2 - 2 * s
    return (y[idx] * d00 + h * m[idx] * d10 + y[idx + 1] * d01 + h * m[idx + 1] * d11) / h


# --------------------------------------------------------------------------------------
# Survivor construction + distribution stats (pure, unit-tested)
# --------------------------------------------------------------------------------------
def enforce_monotone(strikes: list[float], probs: list[float]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Clip a survivor point set to [0,1] and non-increasing-in-strike.

    Returns (strikes, clipped_probs, violated_strikes). A 'violation' is any strike whose
    quoted survivor prob exceeds the running minimum of the strikes to its left (i.e. the
    market briefly priced P(cap>K2) > P(cap>K1) for K2 > K1, an arbitrage-inconsistent set).
    """
    k = np.asarray(strikes, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    order = np.argsort(k)
    k, p = k[order], p[order]
    clipped = np.minimum.accumulate(p)
    violated = [float(k[i]) for i in range(len(k)) if p[i] > clipped[i] + 1e-12]
    return k, clipped, violated


def fit_survivor(strikes: list[float], probs: list[float],
                 tail_anchors=TAIL_ANCHORS, grid_max: float = CAP_GRID_MAX,
                 grid_n: int = GRID_N) -> dict:
    """Monotone PCHIP survivor fit on a cap-strike ladder (cap in $T).

    Knots = (0, 1.0) + clipped ladder points + tail anchors (clipped into monotonicity).
    Returns grid, S(grid), pdf = -dS/dK (clipped at 0), and the monotonicity flags.
    """
    k, p, violated = enforce_monotone(strikes, probs)
    knots_x = [0.0] + list(k)
    knots_y = [1.0] + list(p)
    for ax, ay in tail_anchors:
        if ax > knots_x[-1]:
            knots_x.append(ax)
            knots_y.append(min(ay, knots_y[-1]))  # anchors must not break monotonicity
    x = np.asarray(knots_x)
    y = np.asarray(knots_y)
    m = pchip_slopes(x, y)
    grid = np.linspace(0.0, min(grid_max, x[-1]), grid_n)
    S = np.clip(pchip_eval(x, y, m, grid), 0.0, 1.0)
    S = np.minimum.accumulate(S)  # numerical guard; PCHIP through monotone knots is monotone
    pdf = np.maximum(-pchip_eval(x, y, m, grid, derivative=True), 0.0)
    return {
        "grid": grid, "S": S, "pdf": pdf,
        "knots_x": x, "knots_y": y, "slopes": m,
        "monotone_violations": violated,
        "tail_mass_beyond_grid": float(S[-1]),
    }


def dist_stats(fit: dict, shares: float, offer: float = IPO_OFFER_DEFAULT) -> dict:
    """Distribution stats from a survivor fit, in cap ($T) and per-share ($) terms.

    mean is the survivor integral E[cap] = ∫ S(K) dK (valid because S(0)=1, S(end)≈0);
    a cross-check against ∫ K·pdf dK is returned as mean_pdf_xcheck. Percentile Px =
    the cap below which x% of the probability mass sits, read off F = 1 - S directly.
    """
    grid, S, pdf = fit["grid"], fit["S"], fit["pdf"]
    mean_cap = float(np.trapezoid(S, grid))
    mass = float(np.trapezoid(pdf, grid))
    pdf_n = pdf / mass if mass > 0 else pdf
    mean_pdf = float(np.trapezoid(grid * pdf_n, grid))
    F = np.maximum.accumulate(1.0 - S)

    def pct(q: float) -> float:
        return float(np.interp(q, F, grid))

    mode_cap = float(grid[int(np.argmax(pdf))])
    var_cap = float(np.trapezoid((grid - mean_cap) ** 2 * pdf_n, grid))
    cap_offer = per_share_to_cap_t(offer, shares)
    p_win = float(np.interp(cap_offer, grid, S))
    # conditional means for the EV decomposition (used by the reconcile sweep)
    win = grid >= cap_offer
    p_mass_win = float(np.trapezoid(pdf_n[win], grid[win]))
    e_win_cap = (float(np.trapezoid(grid[win] * pdf_n[win], grid[win])) / p_mass_win
                 if p_mass_win > 0 else math.nan)
    lose = ~win
    p_mass_lose = float(np.trapezoid(pdf_n[lose], grid[lose]))
    e_lose_cap = (float(np.trapezoid(grid[lose] * pdf_n[lose], grid[lose])) / p_mass_lose
                  if p_mass_lose > 0 else math.nan)
    out = {
        "mean_cap_t": mean_cap, "mean_pdf_xcheck_t": mean_pdf,
        "median_cap_t": pct(0.50), "mode_cap_t": mode_cap,
        "std_cap_t": math.sqrt(var_cap),
        "p05_cap_t": pct(0.05), "p10_cap_t": pct(0.10), "p25_cap_t": pct(0.25),
        "p75_cap_t": pct(0.75), "p90_cap_t": pct(0.90), "p95_cap_t": pct(0.95),
        "p_win_offer": p_win, "e_win_cap_t": e_win_cap, "e_lose_cap_t": e_lose_cap,
        "offer": offer, "shares": shares,
    }
    for key in ("mean", "median", "mode", "std", "p05", "p10", "p25", "p75", "p90", "p95"):
        out[f"{key}_ps"] = cap_t_to_per_share(out[f"{key}_cap_t"], shares)
    out["ev_vs_offer_ps"] = out["mean_ps"] - offer
    return out


def survivor_at(fit: dict, cap_t: float) -> float:
    return float(np.interp(cap_t, fit["grid"], fit["S"]))


def linear_survivor(strikes: list[float], probs: list[float], grid_max: float = CAP_GRID_MAX):
    """Model-free linear-interp survivor through the clipped points, anchored (0,1)→(max,0).
    Used for the bucket comparison so the PCHIP's smoothing never injects fake gaps."""
    k, p, _ = enforce_monotone(strikes, probs)
    xs = np.concatenate([[0.0], k, [grid_max]])
    ys = np.concatenate([[1.0], p, [0.0]])
    return lambda c: float(np.interp(c, xs, ys))


def bucket_compare(ladder_strikes: list[float], ladder_probs: list[float],
                   buckets: list[dict]) -> list[dict]:
    """Ladder-implied vs bucket-market probability per cap bucket.

    buckets: [{label, lo, hi, mid}] with mid the bucket YES mid-quote (0-1). The bucket
    side is renormalized over the listed buckets (the No-IPO leg is excluded upstream),
    matching the audit's IPO-conditional convention. gap_pp = bucket - ladder, in points.
    """
    lin = linear_survivor(ladder_strikes, ladder_probs)
    mids = np.asarray([b["mid"] for b in buckets], dtype=float)
    norm = mids / mids.sum() if mids.sum() > 0 else mids
    out = []
    for i, b in enumerate(buckets):
        implied = lin(b["lo"]) - lin(b["hi"])
        out.append({
            "label": b["label"], "lo": b["lo"], "hi": b["hi"],
            "ladder_implied": implied, "bucket_mid_raw": float(mids[i]),
            "bucket_normalized": float(norm[i]),
            "gap_pp": (float(norm[i]) - implied) * 100.0,
        })
    return out


# One source of truth for the DIVERGENT flag (Block S5k PM tab): the terminal renderer
# and the PM-tab table both call this — the threshold can never drift between surfaces.
BUCKET_DIVERGENT_PP = 5.0


def bucket_divergent(gap_pp: float) -> bool:
    """|bucket − ladder| gap beyond the pre-registered ±5pp screen threshold."""
    return abs(gap_pp) > BUCKET_DIVERGENT_PP


# --------------------------------------------------------------------------------------
# Liquidity-weighted lognormal survivor (the audit's recommended construction) — the
# smooth "fair value" each tail strike's executable quote is faded against. numpy-only
# (no scipy): erf-based normal CDF + a two-stage grid search over (mu, sigma) in cap space,
# weighted 1/spread² so thin/wide-quoted strikes pull the curve less. Mirrors the math in
# scripts/spacex_pdf_builder_v2.py; see [[spacex_pdf_construction_audit]] Finding 3.
# --------------------------------------------------------------------------------------
_VERF = np.vectorize(math.erf)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _VERF(np.asarray(z, dtype=float) / math.sqrt(2.0)))


def lognormal_survivor(cap_t, mu: float, sig: float) -> np.ndarray:
    """S(K) = P(cap > K) for a lognormal closing cap with log-mean mu, log-sd sig (cap $T)."""
    k = np.maximum(np.asarray(cap_t, dtype=float), 1e-9)
    return 1.0 - _norm_cdf((np.log(k) - mu) / sig)


def fit_lognormal_weighted(strikes: list[float], surv_pts: list[float],
                           spreads: list[float]) -> dict:
    """1/spread²-weighted least-squares lognormal survivor fit (taker-liquidity proxy:
    tight quotes weigh more). Returns mu, sig, the weighted RMSE in cents, and a callable
    S(K). Two-stage grid search keeps it dependency-free and deterministic."""
    k = np.asarray(strikes, dtype=float)
    S = np.clip(np.asarray(surv_pts, dtype=float), 1e-6, 1 - 1e-6)
    w = 1.0 / np.maximum(np.asarray(spreads, dtype=float), 1e-3) ** 2
    w = w / w.sum()

    def loss(mu: float, sig: float) -> float:
        pred = lognormal_survivor(k, mu, sig)
        return float(np.sum(w * (pred - S) ** 2))

    mu_lo, mu_hi = math.log(1.0), math.log(4.0)
    sig_lo, sig_hi = 0.08, 0.70
    best = (mu_lo, sig_lo, loss(mu_lo, sig_lo))
    for _ in range(2):  # coarse then refine around the incumbent
        mus = np.linspace(mu_lo, mu_hi, 41)
        sigs = np.linspace(sig_lo, sig_hi, 41)
        for mu in mus:
            for sig in sigs:
                e = loss(float(mu), float(sig))
                if e < best[2]:
                    best = (float(mu), float(sig), e)
        dmu = (mu_hi - mu_lo) / 20.0
        dsig = (sig_hi - sig_lo) / 20.0
        mu_lo, mu_hi = best[0] - dmu, best[0] + dmu
        sig_lo, sig_hi = max(0.02, best[1] - dsig), best[1] + dsig

    mu, sig, _ = best
    wrmse_c = math.sqrt(loss(mu, sig)) * 100.0
    return {"mu": mu, "sig": sig, "median_cap_t": math.exp(mu),
            "mean_cap_t": math.exp(mu + sig**2 / 2.0), "wrmse_c": wrmse_c,
            "S": lambda c: float(lognormal_survivor(c, mu, sig))}


# --------------------------------------------------------------------------------------
# Tail-trade screen — the Polymarket leg of the gameplan (sell the rich upper-tail YES into
# the pop at PEAK; "cap above $K" YES bids fading from a high = PEAK signal #6). For each
# upper-tail strike it surfaces the EXECUTABLE SELL price (the YES *bid* — you hit the bid to
# sell), how rich that bid is vs the liquidity-weighted fair survivor, the short-window
# repricing velocity, and the sellable size at the bid. Gameplan §lines 196/199/246.
# --------------------------------------------------------------------------------------
def _bid_at(history: list[dict], strike: float, t_target: float) -> float | None:
    """The logged YES bid for `strike` at-or-after t_target (for the Δ-over-window read)."""
    for r in history:
        if r["ts"] >= t_target:
            v = (r.get("tails") or {}).get(strike)
            if v and v[0] is not None:
                return v[0]
    return None


def tail_trade_eval(snap: dict, fit_ln: dict | None, history: list[dict],
                    now: float, tail_strikes=None, window_s: float = 900.0,
                    rip_c: float = 2.0, fade_c: float = 1.5, rich_c: float = 1.5) -> dict:
    """Per upper-tail "cap above K" strike, the sell screen:
      sell_bid   – the executable YES *bid* (you hit it to sell), = implied P(cap>K);
      dbid_c     – Δ of that bid over `window_s` (the repricing — the gameplan signal);
      off_high_c – how far below its own window high the bid now sits (the fade read);
      bid_depth_sh – sellable size resting at the bid right now;
      richness_c – bid vs the liquidity-weighted lognormal fair (secondary dislocation read;
                   note the lognormal runs slightly rich in the deep tail per the audit, so
                   this skews negative — it flags relative dislocation, not the trigger).

    The PEAK trigger (gameplan #6) is REPRICING, not the model gap: tails ripped up into the
    pop and are now FADING from their highs. Robust to missing fit/depth/history (→ 'n/a')."""
    strikes = list(tail_strikes) if tail_strikes is not None else list(TAIL_STRIKES)
    by_strike = {r.get("strike_t"): r for r in snap.get("ladder", [])}
    rows: list[dict] = []
    for k in strikes:
        r = by_strike.get(k, {})
        bid, ask = r.get("bid"), r.get("ask")
        fair = fit_ln["S"](k) if fit_ln else None
        rich = (bid - fair) * 100.0 if (bid is not None and fair is not None) else None
        past = _bid_at(history, k, now - window_s)
        dbid = (bid - past) * 100.0 if (bid is not None and past is not None) else None
        win_bids = [(rr.get("tails") or {}).get(k, (None,))[0]
                    for rr in history if rr["ts"] >= now - window_s]
        win_bids = [b for b in win_bids if b is not None]
        if bid is not None:
            win_bids.append(bid)
        hi = max(win_bids) if win_bids else None
        off_hi = (bid - hi) * 100.0 if (bid is not None and hi is not None) else None
        ripped = (hi - fair) * 100.0 >= rip_c if (hi is not None and fair is not None) else False
        rows.append({
            "strike_t": k, "sell_bid": bid, "ask": ask, "fair": fair,
            "richness_c": rich, "dbid_c": dbid, "off_high_c": off_hi, "ripped": ripped,
            "bid_depth_sh": r.get("bid_sz"),
            "read": _tail_read(bid, rich, dbid, off_hi, ripped, rip_c, fade_c, rich_c),
        })
    n_ripping = sum(1 for r in rows if (r["dbid_c"] or -99) >= rip_c)
    fading = [r for r in rows if r["ripped"] and (r["off_high_c"] or 0) <= -fade_c]
    n_rich = sum(1 for r in rows if (r["richness_c"] or -99) >= rich_c)
    if fading:
        signal, cls = (f"PM tails FADING from highs ({len(fading)} strike(s)) "
                       "— PEAK signal #6 firing → SELL the tail", "go")
    elif n_ripping >= 2:
        signal, cls = ("PM tails RIPPING up (sell zone opening) "
                       "— ready the tail-sell, trigger on the fade", "watch")
    elif n_ripping >= 1 or n_rich >= 1:
        signal, cls = "one tail repricing up — not yet a PEAK trigger", "watch"
    else:
        signal, cls = "PM tails quiet — no tail-sell signal", "idle"
    return {"rows": rows, "signal": signal, "signal_cls": cls,
            "n_ripping": n_ripping, "n_fading": len(fading), "n_rich": n_rich,
            "window_min": window_s / 60.0, "rip_c": rip_c, "fade_c": fade_c}


def _tail_read(bid, rich, dbid, off_hi, ripped, rip_c, fade_c, rich_c) -> str:
    if bid is None:
        return "no bid — can't sell here"
    if ripped and off_hi is not None and off_hi <= -fade_c:
        return f"ripped then fading {off_hi:.1f}c off high → SELL (PEAK)"
    if dbid is not None and dbid >= rip_c:
        return f"ripping up +{dbid:.1f}c → ready, sell on the fade"
    if dbid is not None and dbid <= -rip_c:
        return f"bid bleeding {dbid:.1f}c → crowd lowering the close"
    if rich is not None and rich >= rich_c:
        return f"bid {rich:+.1f}c rich vs fair → locally dislocated"
    return "quiet"


# --------------------------------------------------------------------------------------
# The −$3.3/share EV reconciliation sweep (coworker PNG vs its own published stats)
# --------------------------------------------------------------------------------------
def ev_convention_sweep(fit: dict, shares: float = SHARES_COWORKER,
                        offer: float = IPO_OFFER_DEFAULT,
                        perp_mark: float = PERP_MARK_2026_06_07) -> list[dict]:
    """Candidate EV conventions evaluated on the SAME fitted distribution, each compared
    to the coworker PNG's 'EV: $-3.3/share' to find which (if any) reproduces it."""
    st = dist_stats(fit, shares=shares, offer=offer)
    pwin, plose = st["p_win_offer"], 1.0 - st["p_win_offer"]
    e_win_ps = cap_t_to_per_share(st["e_win_cap_t"], shares)
    e_lose_ps = cap_t_to_per_share(st["e_lose_cap_t"], shares)
    avg_gain = e_win_ps - offer
    avg_loss = offer - e_lose_ps
    grid, S = fit["grid"], fit["S"]
    trunc = grid >= 1.0  # the ladder's first strike: what E looks like if the [0,1.0T)
    ev_trunc = cap_t_to_per_share(float(np.trapezoid(S[trunc], grid[trunc])), shares) - offer
    cands = [
        ("A1 mean - offer (the correct unhedged EV)", st["mean_ps"] - offer),
        ("A2 pwin*avg_gain - plose*avg_loss (same, decomposed)", pwin * avg_gain - plose * avg_loss),
        ("B1 median - offer", st["median_ps"] - offer),
        ("B2 mode - offer", st["mode_ps"] - offer),
        ("C1 mean - perp mark (entry at perp, not offer)", st["mean_ps"] - perp_mark),
        ("C2 median - perp mark", st["median_ps"] - perp_mark),
        ("D1 pwin*avg_gain - plose*offer (total loss if lose)", pwin * avg_gain - plose * offer),
        ("D2 pwin*(median-offer) - plose*offer", pwin * (st["median_ps"] - offer) - plose * offer),
        ("D3 pwin*(mean-offer) - plose*offer", pwin * (st["mean_ps"] - offer) - plose * offer),
        ("E1 sign-flipped decomposition", plose * avg_loss - pwin * avg_gain),
        ("F1 survivor integral truncated at 1.0T - offer", ev_trunc),
    ]
    target = -3.3
    return [{"convention": name, "ev_ps": v, "abs_err_vs_png": abs(v - target)}
            for name, v in cands]


# --------------------------------------------------------------------------------------
# Snapshot assembly (fetch layer) — read-only, degrades gracefully
# --------------------------------------------------------------------------------------
_STRIKE_RE = re.compile(r"above \$([0-9.]+)T", re.IGNORECASE)
_BETWEEN_RE = re.compile(r"between \$([0-9.]+)T and \$([0-9.]+)T", re.IGNORECASE)
_LESS_RE = re.compile(r"less than \$([0-9.]+)T", re.IGNORECASE)
_ATLEAST_RE = re.compile(r"at least \$([0-9.]+)T", re.IGNORECASE)


def _yes_token(market: dict) -> str | None:
    try:
        outcomes = json.loads(market.get("outcomes", "[]"))
        tokens = json.loads(market.get("clobTokenIds", "[]"))
        for i, o in enumerate(outcomes):
            if str(o).strip().lower() == "yes":
                return tokens[i]
    except (ValueError, IndexError, TypeError):
        pass
    return None


def fetch_pm_metadata(timeout: float = 30.0) -> dict:
    """Resolve both Gamma events into {ladder: [{strike_t, token}], buckets: [...]}.
    Called once per session; token ids are stable."""
    import httpx

    meta = {"ladder": [], "buckets": [], "no_ipo": None}
    with httpx.Client(base_url=GAMMA_BASE, timeout=timeout) as client:
        for slug in (LADDER_EVENT_SLUG, BUCKET_EVENT_SLUG):
            r = client.get("/events", params={"slug": slug})
            r.raise_for_status()
            events = r.json()
            if not events:
                raise RuntimeError(f"gamma event not found: {slug}")
            for mkt in events[0].get("markets", []):
                q = mkt.get("question", "")
                tok = _yes_token(mkt)
                if tok is None:
                    continue
                if slug == LADDER_EVENT_SLUG:
                    m = _STRIKE_RE.search(q)
                    if m:
                        meta["ladder"].append({"strike_t": float(m.group(1)), "token": tok,
                                               "question": q})
                else:
                    if "not IPO" in q:
                        meta["no_ipo"] = {"label": "No-IPO", "token": tok, "question": q}
                        continue
                    b = _BETWEEN_RE.search(q)
                    if b:
                        lo, hi = float(b.group(1)), float(b.group(2))
                    elif (l := _LESS_RE.search(q)):
                        lo, hi = 0.0, float(l.group(1))
                    elif (a := _ATLEAST_RE.search(q)):
                        lo, hi = float(a.group(1)), CAP_GRID_MAX
                    else:
                        continue
                    meta["buckets"].append({"label": f"{lo:g}-{hi:g}T" if lo else f"<{hi:g}T",
                                            "lo": lo, "hi": hi, "token": tok, "question": q})
    meta["ladder"].sort(key=lambda d: d["strike_t"])
    meta["buckets"].sort(key=lambda d: d["lo"])
    return meta


def fetch_books(token_ids: list[str], timeout: float = 30.0) -> dict[str, dict]:
    """Best executable bid/ask per token from the CLOB. Batch POST /books, falling back
    to per-token GET /book. Empty sides come back as None (one-sided books are real)."""
    import httpx

    out: dict[str, dict] = {}

    def _parse(book: dict) -> dict:
        bids = [(float(x["price"]), float(x.get("size", 0) or 0))
                for x in book.get("bids", []) or []]
        asks = [(float(x["price"]), float(x.get("size", 0) or 0))
                for x in book.get("asks", []) or []]
        bb = max(bids, key=lambda t: t[0]) if bids else (None, None)
        ba = min(asks, key=lambda t: t[0]) if asks else (None, None)
        return {"bid": bb[0], "ask": ba[0], "bid_sz": bb[1], "ask_sz": ba[1]}

    with httpx.Client(base_url=CLOB_BASE, timeout=timeout) as client:
        try:
            r = client.post("/books", json=[{"token_id": t} for t in token_ids])
            r.raise_for_status()
            for book in r.json():
                out[str(book.get("asset_id"))] = _parse(book)
        except Exception:
            for t in token_ids:
                try:
                    r = client.get("/book", params={"token_id": t})
                    r.raise_for_status()
                    out[t] = _parse(r.json())
                except Exception:
                    out[t] = {"bid": None, "ask": None}
    return out


def fetch_hl_mark(timeout: float = 15.0) -> dict | None:
    """xyz:SPCX mark/mid from the Hyperliquid info endpoint (same shape as the calc)."""
    import httpx

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(HL_INFO_URL, json={"type": "metaAndAssetCtxs", "dex": "xyz"})
            r.raise_for_status()
            meta, ctxs = r.json()
            for i, a in enumerate(meta["universe"]):
                if a["name"] == "xyz:SPCX":
                    ctx = ctxs[i]
                    return {"mark": float(ctx["markPx"]),
                            "mid": float(ctx.get("midPx", ctx["markPx"])),
                            "funding_hourly": float(ctx.get("funding", 0.0)),
                            "max_leverage": float(a.get("maxLeverage", 3.0))}
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------------------
# Block S5b — optional live listed-price feed (Alpaca IEX websocket), read-only
# --------------------------------------------------------------------------------------
ALPACA_IEX_WS = "wss://stream.data.alpaca.markets/v2/iex"
SPOT_STALE_SECS_DEFAULT = 120.0
IEX_COVERAGE_LABEL = "IEX (≈2% of tape — signal-grade, not queue-grade)"

_FRAC_RE = re.compile(r"\.(\d{6})\d*")  # trim RFC3339 fractions to µs for fromisoformat


def _parse_rfc3339_epoch(ts: str | None) -> float:
    """Alpaca trade timestamps are RFC3339 with up to ns precision; trim to µs."""
    if not ts:
        return time.time()
    try:
        clean = _FRAC_RE.sub(lambda m: "." + m.group(1), ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(clean).timestamp()
    except ValueError:
        return time.time()


class AlpacaSpotFeed:
    """Background last-trade tracker on Alpaca's free IEX stream.

    Thread-safe: the websocket thread writes (price, trade_ts); each poll cycle reads a
    snapshot. Reconnects forever with capped exponential backoff (1s → 60s). All message
    handling is in handle_message() so tests can inject frames without a socket.
    """

    def __init__(self, symbol: str, key_id: str | None = None, secret: str | None = None,
                 url: str = ALPACA_IEX_WS):
        try:  # repo ships python-dotenv; pick up the research-dir .env if present
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        except ImportError:
            pass
        self.symbol = symbol
        self.url = url
        self._key = key_id or os.environ.get("ALPACA_KEY_ID")
        self._secret = secret or os.environ.get("ALPACA_SECRET_KEY")
        if not (self._key and self._secret):
            raise RuntimeError(
                "--spot-ws alpaca needs ALPACA_KEY_ID and ALPACA_SECRET_KEY env vars")
        self._lock = threading.Lock()
        self._last_price: float | None = None
        self._last_trade_ts: float | None = None
        self._vwap_num = 0.0           # Σ price·size since the FIRST received print —
        self._vwap_den = 0.0           # the anchored VWAP of gameplan §5.3 #1 (lookahead-
        self._first_trade_ts: float | None = None  # free: only prints already received)
        self._stop = threading.Event()
        self._backoff = 1.0

    # ---- message/state layer (socket-free, unit-tested) ----
    def handle_message(self, raw: str | bytes) -> None:
        try:
            msgs = json.loads(raw)
        except (ValueError, TypeError):
            return
        if isinstance(msgs, dict):
            msgs = [msgs]
        for m in msgs:
            if isinstance(m, dict) and m.get("T") == "t" and m.get("S") == self.symbol:
                ts = _parse_rfc3339_epoch(m.get("t"))
                size = float(m.get("s", 0.0) or 0.0)
                with self._lock:
                    self._last_price = float(m["p"])
                    self._last_trade_ts = ts
                    if self._first_trade_ts is None:
                        self._first_trade_ts = ts
                    self._vwap_num += float(m["p"]) * size
                    self._vwap_den += size

    def snapshot(self, stale_secs: float = SPOT_STALE_SECS_DEFAULT,
                 now: float | None = None) -> dict:
        """{'source','status','price','age_s'} — status: live | stale | no_prints."""
        now = time.time() if now is None else now
        with self._lock:
            price, ts = self._last_price, self._last_trade_ts
        if price is None:
            return {"source": "alpaca_iex", "status": "no_prints", "price": None, "age_s": None}
        age = max(0.0, now - ts)
        return {"source": "alpaca_iex", "status": "live" if age <= stale_secs else "stale",
                "price": price, "age_s": age}

    def avwap(self) -> dict | None:
        """Anchored VWAP since the first received print, or None before any print.
        IEX-only (~2% of tape): signal-grade — confirm on TradingView before acting."""
        with self._lock:
            if self._vwap_den <= 0:
                return None
            return {"avwap": self._vwap_num / self._vwap_den,
                    "anchored_at": self._first_trade_ts}

    def next_backoff(self) -> float:
        """Current reconnect delay, then double it (capped at 60s). Reset on connect."""
        b = self._backoff
        self._backoff = min(self._backoff * 2.0, 60.0)
        return b

    # ---- socket layer ----
    def _on_open(self, ws) -> None:
        self._backoff = 1.0
        ws.send(json.dumps({"action": "auth", "key": self._key, "secret": self._secret}))
        ws.send(json.dumps({"action": "subscribe", "trades": [self.symbol]}))

    def start(self) -> None:
        threading.Thread(target=self._run_loop, daemon=True,
                         name=f"alpaca-spot-{self.symbol}").start()

    def stop(self) -> None:
        self._stop.set()

    def _run_loop(self) -> None:
        import websocket  # websocket-client, already a project dependency

        while not self._stop.is_set():
            try:
                ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=lambda _ws, msg: self.handle_message(msg))
                ws.run_forever(ping_interval=15)  # returns on any disconnect/error
            except Exception:
                pass  # never let the feed thread die; backoff below
            if self._stop.is_set():
                break
            time.sleep(self.next_backoff())


def resolve_spot(manual_spot: float | None, feed, stale_secs: float = SPOT_STALE_SECS_DEFAULT,
                 now: float | None = None) -> tuple[float | None, dict | None]:
    """Spot precedence: manual --spot always wins; else the feed snapshot; else nothing.
    Returns (spot_for_crowd_compare | None, spot_meta | None) — a stale/no-prints feed
    yields meta (for rendering/logging) but no spot value."""
    if manual_spot is not None:
        return manual_spot, {"source": "manual", "status": "live",
                             "price": manual_spot, "age_s": None}
    if feed is None:
        return None, None
    snap = feed.snapshot(stale_secs, now=now)
    return (snap["price"], snap) if snap["status"] == "live" else (None, snap)


def build_snapshot(meta: dict, timeout: float = 30.0) -> dict:
    """One poll: books for every ladder/bucket token + the HL mark, timestamped."""
    tokens = [d["token"] for d in meta["ladder"]] + [d["token"] for d in meta["buckets"]]
    if meta.get("no_ipo"):
        tokens.append(meta["no_ipo"]["token"])
    books = fetch_books(tokens, timeout=timeout)
    snap = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "ladder": [], "buckets": [], "no_ipo": None, "hl": fetch_hl_mark()}
    for d in meta["ladder"]:
        b = books.get(d["token"], {"bid": None, "ask": None})
        snap["ladder"].append({"strike_t": d["strike_t"], "bid": b["bid"], "ask": b["ask"],
                               "bid_sz": b.get("bid_sz"), "ask_sz": b.get("ask_sz")})
    for d in meta["buckets"]:
        b = books.get(d["token"], {"bid": None, "ask": None})
        snap["buckets"].append({"label": d["label"], "lo": d["lo"], "hi": d["hi"],
                                "bid": b["bid"], "ask": b["ask"]})
    if meta.get("no_ipo"):
        b = books.get(meta["no_ipo"]["token"], {"bid": None, "ask": None})
        snap["no_ipo"] = {"bid": b["bid"], "ask": b["ask"]}
    return snap


def fixture_snapshot() -> dict:
    """The embedded 2026-06-07 coworker surface as a snapshot (cents → probabilities;
    YES bid = 1 - NO ask). Offline reproduction + test fixture."""
    snap = {"fetched_at_utc": "2026-06-07T15:59:00+00:00", "ladder": [], "buckets": [],
            "no_ipo": {"bid": None, "ask": 0.004}, "hl": {"mark": PERP_MARK_2026_06_07,
                                                          "mid": PERP_MARK_2026_06_07}}
    for k, yes_ask, no_ask in FIXTURE_2026_06_07_LADDER:
        snap["ladder"].append({"strike_t": k, "bid": (100 - no_ask) / 100, "ask": yes_ask / 100})
    for label, lo, hi, yes_ask, no_ask in FIXTURE_2026_06_07_BUCKETS:
        snap["buckets"].append({"label": label, "lo": lo, "hi": hi,
                                "bid": (100 - no_ask) / 100, "ask": yes_ask / 100})
    return snap


def basis_price(row: dict, basis: str) -> tuple[float | None, str | None]:
    """One row's survivor probability on the chosen price basis.

    basis: 'mid' (default; one-sided rows degrade to the available side, flagged),
    'ask' or 'bid' (rows missing that side are dropped, flagged). Returns (prob|None, flag|None)."""
    key = row.get("strike_t", row.get("label"))
    bid, ask = row.get("bid"), row.get("ask")
    if basis == "mid":
        if bid is not None and ask is not None:
            return (bid + ask) / 2, None
        if ask is not None:
            return ask, f"{key}: bid side empty, used ask"
        if bid is not None:
            return bid, f"{key}: ask side empty, used bid"
        return None, f"{key}: book empty, dropped"
    p = row.get(basis)
    if p is None:
        return None, f"{key}: no {basis} quote, dropped"
    return p, None


def extract_points(rows: list[dict], basis: str) -> tuple[list[float], list[float], list[str]]:
    """Ladder rows → (strikes, survivor probs, degradation flags) on the chosen basis."""
    xs, ps, flags = [], [], []
    for row in rows:
        p, flag = basis_price(row, basis)
        if flag:
            flags.append(flag)
        if p is not None:
            xs.append(row.get("strike_t", row.get("lo")))
            ps.append(p)
    return xs, ps, flags


# --------------------------------------------------------------------------------------
# Analysis of one snapshot → report dict (pure given the snapshot)
# --------------------------------------------------------------------------------------
def analyze(snap: dict, basis: str = "mid", offer: float = IPO_OFFER_DEFAULT,
            spot: float | None = None, spot_meta: dict | None = None) -> dict:
    strikes, probs, flags = extract_points(snap["ladder"], basis)
    if len(strikes) < 4:
        raise RuntimeError(f"only {len(strikes)} usable ladder strikes — refusing to fit")
    fit = fit_survivor(strikes, probs)
    stats_a = dist_stats(fit, shares=SHARES_PRIMARY, offer=offer)
    stats_b = dist_stats(fit, shares=SHARES_COWORKER, offer=offer)

    # liquidity-weighted lognormal "fair" survivor + the tail-trade screen (gameplan PM leg).
    # spreads align to the fitted strikes; one-sided strikes get a wide spread → low weight.
    spread_by_k = {}
    for r in snap["ladder"]:
        bid, ask = r.get("bid"), r.get("ask")
        spread_by_k[r.get("strike_t")] = (max(ask - bid, 0.001)
                                          if bid is not None and ask is not None else 0.05)
    fit_ln = fit_lognormal_weighted(strikes, probs, [spread_by_k.get(k, 0.05) for k in strikes])
    tail = tail_trade_eval(snap, fit_ln, history=[], now=0.0)  # velocity filled in by caller

    b_flags: list[str] = []
    usable: list[dict] = []
    for b in snap["buckets"]:
        p, flag = basis_price(b, basis)
        if flag:
            b_flags.append(flag)
        if p is not None:
            usable.append(dict(b, mid=p))
    bucket_rows = bucket_compare(strikes, probs, usable) if usable else []

    hl = snap.get("hl") or {}
    gap = None
    if hl.get("mark") is not None:
        gap = {"hl_mark": hl["mark"],
               "vs_mean_ps": hl["mark"] - stats_a["mean_ps"],
               "vs_median_ps": hl["mark"] - stats_a["median_ps"]}
    if spot is not None and spot_meta is None:
        spot_meta = {"source": "manual", "status": "live", "price": spot, "age_s": None}
    spot_block = None
    if spot is not None:
        spot_cap = per_share_to_cap_t(spot, SHARES_PRIMARY)
        spot_block = {"spot": spot,
                      "crowd_mean_ps": stats_a["mean_ps"],
                      "spot_minus_mean": spot - stats_a["mean_ps"],
                      "crowd_pctile_of_spot": (1.0 - survivor_at(fit, spot_cap)) * 100.0}
    return {"fetched_at_utc": snap["fetched_at_utc"], "basis": basis, "offer": offer,
            "fit": fit, "stats_primary": stats_a, "stats_coworker": stats_b,
            "lognormal": fit_ln, "tail_trade": tail,
            "buckets": bucket_rows, "no_ipo": snap.get("no_ipo"),
            "hl_gap": gap, "spot": spot_block, "spot_meta": spot_meta,
            "degradation_flags": flags + b_flags,
            "monotone_violations": fit["monotone_violations"],
            "n_strikes_used": len(strikes)}


# --------------------------------------------------------------------------------------
# Rendering (terminal + html) and parquet logging
# --------------------------------------------------------------------------------------
def render_text(rep: dict) -> str:
    a, b = rep["stats_primary"], rep["stats_coworker"]
    L = []
    L.append("=" * 78)
    L.append(f"SPCX PM-implied closing distribution   poll={rep['fetched_at_utc']}  "
             f"basis={rep['basis']}  strikes={rep['n_strikes_used']}/16")
    L.append("=" * 78)
    if rep["monotone_violations"]:
        L.append(f"!! monotonicity violations clipped at strikes (T): {rep['monotone_violations']}")
    for f in rep["degradation_flags"]:
        L.append(f"!! degraded: {f}")
    L.append(f"P(close > ${a['offer']:.0f}) = {a['p_win_offer']*100:5.1f}%    "
             f"E[cap] = {a['mean_cap_t']:.3f}T    EV vs offer = "
             f"{a['ev_vs_offer_ps']:+.1f} $/sh (13.076B base)")
    L.append("")
    L.append(f"{'stat':>8s} {'cap $T':>9s} {'$/sh 13.076B':>13s} {'$/sh 13.091B (Alvaro)':>22s}")
    for key, label in [("mean", "mean"), ("median", "median"), ("mode", "mode*"),
                       ("p25", "P25"), ("p75", "P75"), ("p90", "P90"), ("p95", "P95")]:
        L.append(f"{label:>8s} {a[f'{key}_cap_t']:9.3f} {a[f'{key}_ps']:13.1f} "
                 f"{b[f'{key}_ps']:22.1f}")
    L.append("  *mode is shape-fragile under PCHIP (see spacex_pdf_construction_audit.md);")
    L.append("   central stats (P(win)/mean/median/P25-P95) are construction-robust.")
    if rep["buckets"]:
        L.append("")
        L.append(f"{'bucket':>10s} {'ladder%':>8s} {'market%':>8s} {'gap(pp)':>8s}")
        for r in rep["buckets"]:
            flag = "  <-- DIVERGENT" if bucket_divergent(r["gap_pp"]) else ""
            L.append(f"{r['label']:>10s} {r['ladder_implied']*100:8.1f} "
                     f"{r['bucket_normalized']*100:8.1f} {r['gap_pp']:+8.1f}{flag}")
        L.append("  gap = bucket market richer (+) than the ladder-implied probability;")
        L.append("  bucket side renormalized after dropping the No-IPO leg.")
    if rep.get("no_ipo") and (rep["no_ipo"].get("ask") is not None):
        L.append(f"  No-IPO leg ask: {rep['no_ipo']['ask']*100:.1f}c")
    if rep["hl_gap"]:
        g = rep["hl_gap"]
        L.append("")
        L.append(f"HL xyz:SPCX mark ${g['hl_mark']:.2f}  vs PM mean "
                 f"{g['vs_mean_ps']:+.1f} $/sh  vs PM median {g['vs_median_ps']:+.1f} $/sh")
    meta = rep.get("spot_meta")
    if rep["spot"]:
        s = rep["spot"]
        src = ""
        if meta and meta.get("source") == "alpaca_iex":
            src = f"  [{IEX_COVERAGE_LABEL}, trade age {meta['age_s']:.0f}s]"
        L.append(f"LISTED ${s['spot']:.2f}  vs crowd mean {s['spot_minus_mean']:+.1f} $/sh  "
                 f"(spot sits at the crowd's P{s['crowd_pctile_of_spot']:.0f}){src}")
    elif meta:  # feed attached but unusable this cycle — say why, never error
        if meta["status"] == "no_prints":
            L.append("spot feed: no prints yet (normal pre-listing) — crowd-vs-traded pending")
        elif meta["status"] == "stale":
            L.append(f"spot feed STALE (last ${meta['price']:.2f}, age {meta['age_s']:.0f}s) "
                     f"— crowd-vs-traded suppressed; PM/HL polling unaffected")
    return "\n".join(L)


def render_html(rep: dict, refresh_s: int) -> str:
    body = render_text(rep).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta http-equiv='refresh' content='{refresh_s}'>"
            f"<title>SPCX PM-PDF monitor</title></head>"
            f"<body style='background:#111;color:#dfe'>"
            f"<pre style='font-size:13px'>{body}</pre></body></html>")


def log_parquet(snap: dict, rep: dict, out_dir: Path = PARQUET_LOG_DIR,
                extra: dict | None = None) -> Path:
    """Append-only: one parquet shard per poll, long format (one row per instrument),
    poll-level stats denormalized onto every row for easy DuckDB globbing. `extra` =
    additional poll-level columns (S5h: latest pasted indication) — schema-extended;
    older shards simply lack them (read with union_by_name=true)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    a = rep["stats_primary"]
    common = {
        "poll_ts": snap["fetched_at_utc"], "basis": rep["basis"],
        "p_win_offer": a["p_win_offer"], "mean_cap_t": a["mean_cap_t"],
        "median_cap_t": a["median_cap_t"], "mode_cap_t": a["mode_cap_t"],
        "mean_ps": a["mean_ps"], "median_ps": a["median_ps"],
        "p25_ps": a["p25_ps"], "p75_ps": a["p75_ps"], "p90_ps": a["p90_ps"],
        "p95_ps": a["p95_ps"], "ev_vs_offer_ps": a["ev_vs_offer_ps"],
        "hl_mark": (snap.get("hl") or {}).get("mark"),
        "n_monotone_violations": len(rep["monotone_violations"]),
        "n_degradation_flags": len(rep["degradation_flags"]),
        "spot": rep["spot"]["spot"] if rep["spot"] else None,
        # Block S5b columns (schema-extended; older shards simply lack them — read the
        # log with union_by_name=true in DuckDB)
        "spot_source": (rep.get("spot_meta") or {}).get("source"),
        "spot_status": (rep.get("spot_meta") or {}).get("status"),
        "spot_age_s": (rep.get("spot_meta") or {}).get("age_s"),
        "spot_last_price": (rep.get("spot_meta") or {}).get("price"),
    }
    if extra:
        common.update(extra)
    rows = []
    for r in snap["ladder"]:
        rows.append({**common, "kind": "ladder", "label": f">{r['strike_t']:g}T",
                     "strike_lo_t": r["strike_t"], "strike_hi_t": None,
                     "bid": r["bid"], "ask": r["ask"]})
    for r in snap["buckets"]:
        rows.append({**common, "kind": "bucket", "label": r["label"],
                     "strike_lo_t": r["lo"], "strike_hi_t": r["hi"],
                     "bid": r["bid"], "ask": r["ask"]})
    if snap.get("no_ipo"):
        rows.append({**common, "kind": "no_ipo", "label": "No-IPO",
                     "strike_lo_t": None, "strike_hi_t": None,
                     "bid": snap["no_ipo"].get("bid"), "ask": snap["no_ipo"].get("ask")})
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = snap["fetched_at_utc"].replace(":", "").replace("-", "").replace(".", "")[:15]
    path = out_dir / f"poll_{ts}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


# --------------------------------------------------------------------------------------
# Block S5c — rich self-contained HTML dashboard (display only; terminal path untouched)
# --------------------------------------------------------------------------------------
CET = None  # lazy ZoneInfo("Europe/Berlin")
HISTORY_HOURS_DEFAULT = 12.0
TAIL_STRIKES = (2.2, 2.4, 2.6, 2.8, 3.0)  # the tail-sell screen (gameplan: >$2.4T / >$3.0T,
# with 2.2/2.6/2.8 for resolution around the action) — YES "cap above K" upper tail
# Pre-registered alert levels (display-only ledger; alerting itself lives in TradingView).
ALERT_LEVELS = [
    (183.0, "pre-hedge trigger"),
    (162.0, "EU prospectus ceiling"),
    (140.0, "reassess level"),
    (135.0, "offer"),
    (125.0, "sell-everything stop"),
]
_PS_FACTOR = 1e12 / SHARES_PRIMARY  # cap $T → $/share on the 13.076B base


def _cet():
    global CET
    if CET is None:
        from zoneinfo import ZoneInfo
        CET = ZoneInfo("Europe/Berlin")
    return CET


def _epoch(iso_ts: str) -> float:
    return datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).timestamp()


def write_html_atomic(path: Path, content: str) -> None:
    """Write temp file + os.replace so the browser never reads a half-written page."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# Backfill guardrail (Block S5k follow-up, 2026-06-12). The parquet-replay paths feed
# every dashboard chart, so a single malformed or old-schema shard must never (a) crash
# the load and lose every other shard, nor (b) silently corrupt the session. Each shard
# is isolated — skipped + counted on ANY error — and a health line is logged so a broken
# backfill is LOUD, not invisible. "broken in a few places" → a few skips you can see,
# not a dead dashboard.
# --------------------------------------------------------------------------------------
_CRITICAL_HIST_FIELDS = ("mean_ps", "median_ps", "pwin")  # the session/PDF chart anchors


def _is_finite(x) -> bool:
    """A real, plottable number (rejects None, bool, NaN, inf, strings)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def backfill_health(label: str, records: list[dict], n_loaded: int, n_skipped: int,
                    reasons: dict) -> dict:
    """Summarize a parquet backfill and emit a stderr guardrail line; return the report
    dict (unit-testable). WARNs when the result is DEGENERATE — every shard skipped, or a
    session-anchor field (mean_ps/median_ps/pwin) non-finite on >=5% of loaded records —
    because that is the line between 'pre-listing, nothing logged yet' (fine) and 'backfill
    silently broke and the charts are wrong' (must be seen). perp/spot None is NOT a warning
    (legitimately absent pre-cross), so they are deliberately not in _CRITICAL_HIST_FIELDS."""
    miss = {f: sum(1 for r in records if not _is_finite(r.get(f)))
            for f in _CRITICAL_HIST_FIELDS}
    ts = sorted(r["ts"] for r in records if _is_finite(r.get("ts")))
    max_gap_min = max(((ts[i] - ts[i - 1]) / 60.0 for i in range(1, len(ts))), default=0.0)
    degenerate = (n_loaded == 0 and n_skipped > 0) or any(
        m >= 0.05 * max(n_loaded, 1) and m > 0 for m in miss.values())
    report = {"label": label, "loaded": n_loaded, "skipped": n_skipped,
              "skip_reasons": dict(reasons), "missing": miss,
              "max_gap_min": round(max_gap_min, 1), "degenerate": degenerate}
    parts = [f"{label}: loaded {n_loaded}, skipped {n_skipped}"]
    if reasons:
        parts.append("reasons " + ", ".join(f"{k}×{v}" for k, v in reasons.items()))
    bad = {f: m for f, m in miss.items() if m}
    if bad:
        parts.append("missing-anchor " + ", ".join(f"{f}×{m}" for f, m in bad.items()))
    if max_gap_min > 15:
        parts.append(f"largest hole {max_gap_min:.0f} min")
    tag = "!! BACKFILL WARNING —" if degenerate else "[backfill]"
    print(f"{tag} {' · '.join(parts)}", file=sys.stderr)
    return report


def _history_rec_from_rows(rows: list[dict]) -> dict | None:
    """Build ONE dashboard history record from a shard's long-format rows, fully guarded:
    any missing column, bad timestamp, or non-finite session-anchor stat returns None so
    the caller skips + counts that shard instead of the exception killing the whole replay.
    The bucket-gap recompute (Block S5k) goes through the SAME bucket_compare a live poll
    uses, and is independently guarded — a bad bucket leg costs the sparkline, not the row."""
    try:
        first = rows[0]
        ts = _epoch(first["poll_ts"])
        mean_ps, median_ps, pwin = first["mean_ps"], first["median_ps"], first["p_win_offer"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not all(_is_finite(x) for x in (ts, mean_ps, median_ps, pwin)):
        return None
    tails = {r["strike_lo_t"]: (r.get("bid"), r.get("ask")) for r in rows
             if r.get("kind") == "ladder" and r.get("strike_lo_t") in TAIL_STRIKES}
    gaps = None
    try:
        lrows = sorted(({"strike_t": r["strike_lo_t"], "bid": r["bid"], "ask": r["ask"]}
                        for r in rows if r.get("kind") == "ladder"),
                       key=lambda d: d["strike_t"])
        usable = []
        for r in rows:
            if r.get("kind") != "bucket":
                continue
            p, _flag = basis_price(r, "mid")
            if p is not None:
                usable.append({"label": r["label"], "lo": r["strike_lo_t"],
                               "hi": r["strike_hi_t"], "mid": p})
        ks, ps, _ = extract_points(lrows, "mid")
        if usable and len(ks) >= 4:
            gaps = {b["label"]: b["gap_pp"] for b in bucket_compare(ks, ps, usable)}
    except Exception:
        gaps = None      # a malformed bucket leg never blocks the row or the backfill
    return {"ts": ts, "mean_ps": mean_ps, "median_ps": median_ps, "pwin": pwin,
            "perp": first.get("hl_mark"), "spot": first.get("spot"),
            "spot_status": first.get("spot_status"), "tails": tails,
            "bucket_gaps": gaps,
            "n_flags": (first.get("n_monotone_violations") or 0)
            + (first.get("n_degradation_flags") or 0)}


class DashboardState:
    """In-memory poll history + session bookkeeping for the dashboard.

    History records are compact dicts (one per poll); on startup, today's parquet shards
    are backfilled so a restart keeps the session time series. Rendered history is capped
    at `history_hours` (default 12h) — older records are dropped, noted on the page.
    """

    def __init__(self, history_hours: float = HISTORY_HOURS_DEFAULT,
                 marks: list[str] | None = None, offer: float = IPO_OFFER_DEFAULT):
        self.history: list[dict] = []
        self.history_hours = history_hours
        self.offer = offer
        self.session_start_rec: dict | None = None
        self.reference: dict | None = None  # {"label", "fit"} for the Chart-A overlay
        self.last_ok = {"pm": None, "hl": None, "spot": None}  # epoch of last good fetch
        self.alert_first_touch: dict[float, tuple[float, str]] = {}  # level → (ts, source)
        self.marks: list[tuple[float, str]] = []
        self.dropped_old = 0
        now = time.time()
        for m in marks or []:
            label, ts = m, now
            if "@" in m:  # "label@HH:MM" in CET, today
                label, hhmm = m.rsplit("@", 1)
                try:
                    h, mi = map(int, hhmm.split(":"))
                    ts = datetime.now(_cet()).replace(hour=h, minute=mi, second=0,
                                                      microsecond=0).timestamp()
                except ValueError:
                    pass
            self.marks.append((ts, label))

    def set_reference(self, snap: dict, label: str, basis: str = "mid") -> None:
        xs, ps, _ = extract_points(snap["ladder"], basis)
        self.reference = {"label": label, "fit": fit_survivor(xs, ps)}

    # ---- per-poll ----
    def record(self, snap: dict, rep: dict) -> dict:
        ts = _epoch(snap["fetched_at_utc"])
        a = rep["stats_primary"]
        hl = (snap.get("hl") or {})
        meta = rep.get("spot_meta") or {}
        spot = rep["spot"]["spot"] if rep["spot"] else None
        tails = {}
        for row in snap.get("ladder", []):
            if row.get("strike_t") in TAIL_STRIKES:
                tails[row["strike_t"]] = (row.get("bid"), row.get("ask"))
        rec = {"ts": ts, "mean_ps": a["mean_ps"], "median_ps": a["median_ps"],
               "pwin": a["p_win_offer"], "perp": hl.get("mark"), "spot": spot,
               "spot_status": meta.get("status"), "tails": tails,
               "n_flags": len(rep["degradation_flags"]) + len(rep["monotone_violations"]),
               # Block S5k PM tab: per-poll bucket-vs-ladder gaps (pp) for the sparklines
               "bucket_gaps": ({r["label"]: r["gap_pp"] for r in rep["buckets"]}
                               if rep.get("buckets") else None)}
        self.last_ok["pm"] = ts
        if hl.get("mark") is not None:
            self.last_ok["hl"] = ts
        if meta.get("status") == "live":
            self.last_ok["spot"] = ts - (meta.get("age_s") or 0.0)
        self._append(rec)
        if self.session_start_rec is None:
            self.session_start_rec = rec
        if self.reference is None:
            self.set_reference(snap, f"session start "
                               f"{datetime.fromtimestamp(ts, _cet()):%H:%M} CET")
        self._update_alerts(rec)
        return rec

    def _append(self, rec: dict) -> None:
        self.history.append(rec)
        cutoff = rec["ts"] - self.history_hours * 3600.0
        before = len(self.history)
        self.history = [r for r in self.history if r["ts"] >= cutoff]
        self.dropped_old += before - len(self.history)

    def _update_alerts(self, rec: dict) -> None:
        """A level is 'crossed' in its trigger direction relative to the offer: upside
        levels (> offer) when price >= level, downside levels (<= offer) when price <= level.
        Monitored series = spot once printing, perp before."""
        price = rec["spot"] if rec["spot"] is not None else rec["perp"]
        if price is None:
            return
        source = "spot" if rec["spot"] is not None else "perp"
        for level, _label in self._levels():
            if level in self.alert_first_touch:
                continue
            crossed = price >= level if level > self.offer else price <= level
            if crossed:
                self.alert_first_touch[level] = (rec["ts"], source)

    def _levels(self) -> list[tuple[float, str]]:
        levels = list(ALERT_LEVELS)
        if self.offer != IPO_OFFER_DEFAULT:
            levels.append((self.offer, "final price"))
        return sorted(levels, reverse=True)

    # ---- startup backfill from recent shards ----
    def backfill_from_parquet(self, log_dir: Path = PARQUET_LOG_DIR,
                              days: int = 1) -> int:
        """Replay logged polls so a restart keeps the session time series. `days`=1
        loads today only (clean listing-day session); days>1 walks back N UTC calendar
        days so the run-up evolution survives a cross-day restart. Shards are append-only
        on disk regardless — this only widens the read window, not what gets cached."""
        import pyarrow.parquet as pq

        if not log_dir.exists():
            return 0
        wanted = {(datetime.now(timezone.utc) - timedelta(days=d)).strftime("%Y%m%d")
                  for d in range(max(1, days))}
        n, skipped, reasons = 0, 0, {}

        def _skip(why: str) -> None:
            nonlocal skipped
            skipped += 1
            reasons[why] = reasons.get(why, 0) + 1

        shards = sorted(s for d in wanted for s in log_dir.glob(f"poll_{d}*.parquet"))
        for shard in shards:
            try:
                t = pq.read_table(shard)
            except Exception as e:                       # corrupt/truncated file on disk
                _skip(f"unreadable:{type(e).__name__}")
                continue
            rows = t.to_pylist()
            if not rows:
                _skip("empty")
                continue
            rec = _history_rec_from_rows(rows)           # fully guarded (returns None on any fault)
            if rec is None:
                _skip("malformed")
                continue
            self._append(rec)
            n += 1
        self.history.sort(key=lambda r: r["ts"])
        if self.history and self.session_start_rec is None:
            self.session_start_rec = self.history[0]
        for rec in self.history:  # replay the alert ledger over the backfilled session
            self._update_alerts(rec)
        backfill_health("dashboard", self.history, n, skipped, reasons)
        return n


class CurveIndex:
    """Time-scrub support (Block S5h): an in-memory index of every logged poll's full
    16-strike ladder, so the dashboard's slider can refit and display the survivor/PDF
    exactly as it stood at any past timestamp. Loaded from the append-only parquet shards
    at startup (window: `days`), then appended per live poll, so the scrub range keeps
    extending while the monitor runs. Refits go through the SAME analyze() as live polls
    — one code path, identical math. Thread-safe: the engine thread appends while the
    server's executor thread reads."""

    def __init__(self, days: float = 7.0):
        self.days = days
        self.entries: list[dict] = []   # {ts, ladder:[{strike_t,bid,ask}], perp, spot}
        self._keys: list[float] = []    # entry timestamps, kept sorted alongside
        self._lock = threading.Lock()

    def load_parquet(self, log_dir: Path = PARQUET_LOG_DIR) -> int:
        """Index every shard younger than the window. Whole-table reads (the schema is
        long-format; older shards may lack newer columns, .get() handles both)."""
        import pyarrow.parquet as pq

        if not log_dir.exists():
            return 0
        cutoff = time.time() - self.days * 86400.0
        loaded = []
        skipped, reasons = 0, {}
        for shard in sorted(log_dir.glob("poll_*.parquet")):
            try:
                rows = pq.read_table(shard).to_pylist()
                if not rows:
                    continue
                ts = _epoch(rows[0]["poll_ts"])
                if ts < cutoff:
                    continue                              # out of window — a normal skip
                ladder = sorted(({"strike_t": r["strike_lo_t"], "bid": r["bid"],
                                  "ask": r["ask"]} for r in rows if r.get("kind") == "ladder"),
                                key=lambda d: d["strike_t"])
                if len(ladder) < 4:
                    skipped += 1
                    reasons["too-few-strikes"] = reasons.get("too-few-strikes", 0) + 1
                    continue
                loaded.append({"ts": ts, "ladder": ladder,
                               "perp": rows[0].get("hl_mark"), "spot": rows[0].get("spot")})
            except Exception as e:                        # guardrail: one bad shard never
                skipped += 1                              # kills the whole scrub index
                key = f"malformed:{type(e).__name__}"
                reasons[key] = reasons.get(key, 0) + 1
        if skipped:
            print(f"!! BACKFILL WARNING — time-scrub: indexed {len(loaded)}, "
                  f"skipped {skipped} ({reasons})", file=sys.stderr)
        with self._lock:
            seen = set(self._keys)
            for e in loaded:
                if e["ts"] not in seen:
                    self.entries.append(e)
            self.entries.sort(key=lambda e: e["ts"])
            self._keys = [e["ts"] for e in self.entries]
        return len(loaded)

    def append(self, snap: dict, rep: dict) -> None:
        """One live poll → one scrub point (called from the engine loop after analyze)."""
        ts = _epoch(snap["fetched_at_utc"])
        entry = {"ts": ts,
                 "ladder": [{"strike_t": r["strike_t"], "bid": r.get("bid"),
                             "ask": r.get("ask")} for r in snap["ladder"]],
                 "perp": (rep.get("hl_gap") or {}).get("hl_mark"),
                 "spot": (rep.get("spot") or {}).get("spot")}
        with self._lock:
            if self._keys and ts <= self._keys[-1]:
                return
            self.entries.append(entry)
            self._keys.append(ts)

    def range(self) -> dict | None:
        with self._lock:
            if not self._keys:
                return None
            return {"min_ts": self._keys[0], "max_ts": self._keys[-1],
                    "n": len(self._keys)}

    def nearest(self, ts: float) -> dict | None:
        from bisect import bisect_left
        with self._lock:
            if not self._keys:
                return None
            i = bisect_left(self._keys, ts)
            if i == 0:
                return self.entries[0]
            if i >= len(self._keys):
                return self.entries[-1]
            before, after = self.entries[i - 1], self.entries[i]
            return before if ts - before["ts"] <= after["ts"] - ts else after

    def report_at(self, ts: float):
        """Snap to the nearest logged poll and refit it: returns (entry, snap, rep) or
        None. The synthetic snapshot has the exact shape build_snapshot emits, so the
        report is byte-compatible with a live poll's (minus buckets, which the scrub
        view doesn't draw)."""
        e = self.nearest(ts)
        if e is None:
            return None
        snap = {"fetched_at_utc": datetime.fromtimestamp(e["ts"], timezone.utc).isoformat(),
                "ladder": [dict(r) for r in e["ladder"]], "buckets": [], "no_ipo": None,
                "hl": ({"mark": e["perp"], "mid": e["perp"], "funding_hourly": 0.0}
                       if e["perp"] is not None else None)}
        try:
            rep = analyze(snap, basis="mid", spot=e.get("spot"))
        except RuntimeError:
            return None
        return e, snap, rep


# ---- chart helpers (each returns an inline <svg>; failures degrade to a placeholder) ----
def _fig_to_svg(fig) -> str:
    import io

    import matplotlib.pyplot as plt
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.find("<svg"):]


def _placeholder(title: str, why: str) -> str:
    return (f"<div class='ph'><b>{title}</b> — unavailable this poll "
            f"({why}); page stays live, retries next poll</div>")


def _new_fig(h: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(7.2, h), dpi=100)


def chart_survivor(rep: dict, state: DashboardState, snap: dict) -> str:
    fit = rep["fit"]
    fig, ax = _new_fig(3.4)
    if state.reference:
        rf = state.reference["fit"]
        ax.plot(rf["grid"], rf["S"], "--", color="#999", lw=1.4,
                label=f"reference: {state.reference['label']}")
    ax.plot(fit["grid"], fit["S"], color="#c40", lw=2.2, label="now (PCHIP fit)")
    clipped = set(fit["monotone_violations"])
    for row in snap["ladder"]:
        k, bid, ask = row["strike_t"], row.get("bid"), row.get("ask")
        p, _ = basis_price(row, rep["basis"])
        if p is None:
            continue
        if bid is not None and ask is not None:
            ax.plot([k, k], [bid, ask], color="#36c", lw=1.0, alpha=0.7)
        ax.plot(k, p, "o", ms=4.5, color="#d00" if k in clipped else "#36c",
                zorder=5)
    ax.set_xlim(0.8, 4.2)
    ax.set_xlabel("closing market cap ($T)")
    ax.set_ylabel("P(close cap > K)")
    sec = ax.secondary_xaxis("top", functions=(lambda c: c * _PS_FACTOR,
                                               lambda p: p / _PS_FACTOR))
    sec.set_xlabel("$/share (13.076B shares)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    return _fig_to_svg(fig)


def chart_pdf(rep: dict) -> str:
    fit, a = rep["fit"], rep["stats_primary"]
    fig, ax = _new_fig(3.0)
    price = fit["grid"] * _PS_FACTOR
    dens = fit["pdf"] / _PS_FACTOR
    ax.plot(price, dens, color="#c40", lw=2.0)
    band = (price >= a["p25_ps"]) & (price <= a["p75_ps"])
    ax.fill_between(price[band], dens[band], alpha=0.25, color="#c40",
                    label="P25–P75 band")
    for x, lab, col in [(rep["offer"], f"offer ${rep['offer']:.0f}", "green"),
                        ((rep["hl_gap"] or {}).get("hl_mark"), "perp", "#e80"),
                        ((rep["spot"] or {}).get("spot"), "spot", "#06c")]:
        if x is not None:
            ax.axvline(x, color=col, lw=1.4, ls="-" if lab.startswith("offer") else "--")
            ax.text(x, ax.get_ylim()[1] * 0.96, f" {lab}", color=col, fontsize=8,
                    rotation=90, va="top")
    ax.plot(a["mode_ps"], np.interp(a["mode_ps"], price, dens), "v", color="#888", ms=7)
    ax.annotate("mode (shape-fragile, see audit)", (a["mode_ps"], 0), xytext=(0, -26),
                textcoords="offset points", ha="center", fontsize=7.5, color="#888")
    ax.set_xlim(60, 320)
    ax.set_xlabel("day-1 closing $/share (13.076B base)")
    ax.set_ylabel("density (per $)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_svg(fig)


def _hist_times(state: DashboardState):
    return [datetime.fromtimestamp(r["ts"], _cet()) for r in state.history]


def chart_timeseries(state: DashboardState) -> str:
    if len(state.history) < 2:
        return "<div class='ph'>session time series: collecting… (needs ≥2 polls)</div>"
    fig, ax = _new_fig(3.0)
    ts = _hist_times(state)

    def series(key):
        return [r[key] if r[key] is not None else float("nan") for r in state.history]

    ax.plot(ts, series("mean_ps"), color="#c40", lw=1.8, label="crowd mean $/sh")
    ax.plot(ts, series("median_ps"), color="#c40", lw=1.2, ls="--", label="crowd median")
    ax.plot(ts, series("perp"), color="#e80", lw=1.6, label="perp mark")
    if any(r["spot"] is not None for r in state.history):
        ax.plot(ts, series("spot"), color="#06c", lw=1.6, label="spot")
    ax.set_ylabel("$/share")
    ax2 = ax.twinx()
    ax2.plot(ts, [r["pwin"] * 100 for r in state.history], color="#383",
             lw=1.2, alpha=0.8, label="P(close>$135) %")
    ax2.set_ylabel("P(close>offer) %", color="#383")
    for mts, label in state.marks:
        mdt = datetime.fromtimestamp(mts, _cet())
        if ts[0] <= mdt <= ts[-1]:
            ax.axvline(mdt, color="#555", lw=1.0, ls=":")
            ax.text(mdt, ax.get_ylim()[1], f" {label}", fontsize=8, color="#555",
                    rotation=90, va="top")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlabel("time (CET)")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _fig_to_svg(fig)


def chart_tails(state: DashboardState) -> str:
    if len(state.history) < 2:
        return "<div class='ph'>tail panel: collecting… (needs ≥2 polls)</div>"
    have = [k for k in TAIL_STRIKES
            if any(r["tails"].get(k, (None, None))[0] is not None
                   or r["tails"].get(k, (None, None))[1] is not None
                   for r in state.history)]
    if not have:
        return _placeholder("PEAK/tail panel", "no tail-strike quotes in history")
    fig, ax = _new_fig(3.6)
    ts = _hist_times(state)
    colors = {2.2: "#06c", 2.4: "#c40", 2.6: "#7a3", 2.8: "#a48", 3.0: "#383"}
    for k in have:
        bids = [(r["tails"].get(k) or (None, None))[0] for r in state.history]
        asks = [(r["tails"].get(k) or (None, None))[1] for r in state.history]
        f = lambda v: [x * 100 if x is not None else float("nan") for x in v]
        col = colors.get(k, "#888")
        ax.plot(ts, f(bids), color=col, lw=2.4, label=f">${k}T bid")
        ax.plot(ts, f(asks), color=col, lw=1.4, ls=":", label=f">${k}T ask")
    ax.set_ylabel("YES price (cents)", fontsize=11)
    ax.set_xlabel("time (CET)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, ncols=3, loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _fig_to_svg(fig)


# ---- page assembly ----
def _chip(label: str, age: float | None) -> str:
    if age is None:
        cls, txt = "red", "never"
    elif age <= 90:
        cls, txt = "green", f"{age:.0f}s"
    elif age <= 300:
        cls, txt = "amber", f"{age:.0f}s"
    else:
        cls, txt = "red", f"{age:.0f}s"
    return f"<span class='chip {cls}'>{label}: {txt}</span>"


def _arrow(delta: float | None, fmt: str = "{:+.1f}") -> str:
    if delta is None:
        return "<span class='d'>·</span>"
    if abs(delta) < 1e-9:
        return "<span class='d'>· 0</span>"
    cls = "up" if delta > 0 else "dn"
    sym = "▲" if delta > 0 else "▼"
    return f"<span class='d {cls}'>{sym} {fmt.format(delta)}</span>"


def _tile(label: str, value: str, dprev: float | None, dstart: float | None,
          fmt: str = "{:+.1f}") -> str:
    return (f"<div class='tile'><div class='tl'>{label}</div>"
            f"<div class='tv'>{value}</div>"
            f"<div class='td'>{_arrow(dprev, fmt)} vs prev · "
            f"{_arrow(dstart, fmt)} vs start</div></div>")


def render_dashboard(state: DashboardState, rep: dict, snap: dict,
                     refresh_s: int = 60, playbook_html: str = "") -> str:
    a = rep["stats_primary"]
    now_ts = _epoch(snap["fetched_at_utc"])
    cur = state.history[-1] if state.history else None
    prev = state.history[-2] if len(state.history) >= 2 else None
    start = state.session_start_rec if state.session_start_rec is not cur else None

    def d(key, rec):
        if rec is None or cur is None or cur.get(key) is None or rec.get(key) is None:
            return None
        return cur[key] - rec[key]

    # 1. status strip
    ages = {k: (now_ts - v if v is not None else None) for k, v in state.last_ok.items()}
    flags = []
    if rep["monotone_violations"]:
        flags.append(f"monotonicity clipped at {rep['monotone_violations']} $T")
    flags += rep["degradation_flags"]
    meta = rep.get("spot_meta") or {}
    if meta.get("status") == "stale":
        flags.append(f"spot WS STALE (age {meta['age_s']:.0f}s)")
    cet_str = datetime.fromtimestamp(now_ts, _cet()).strftime("%H:%M:%S CET")
    utc_str = datetime.fromtimestamp(now_ts, timezone.utc).strftime("%H:%M:%S UTC")
    strip = (f"<div class='strip'>poll {cet_str} / {utc_str} &nbsp; "
             + _chip("PM", ages["pm"]) + _chip("HL", ages["hl"])
             + _chip("spot", ages["spot"])
             + (f"<div class='flags'>⚑ {' · '.join(flags)}</div>" if flags else "")
             + "</div>")

    # 2. headline tiles
    perp = (rep["hl_gap"] or {}).get("hl_mark")
    gap = (rep["hl_gap"] or {}).get("vs_mean_ps")
    if rep["spot"]:
        s = rep["spot"]
        spot_val = f"${s['spot']:.2f} (P{s['crowd_pctile_of_spot']:.0f})"
    elif meta.get("status") == "stale":
        spot_val = "STALE"
    else:
        spot_val = "no prints yet"
    tiles = "".join([
        _tile("P(close>$135)", f"{a['p_win_offer']*100:.1f}%",
              None if d("pwin", prev) is None else d("pwin", prev) * 100,
              None if d("pwin", start) is None else d("pwin", start) * 100, "{:+.1f}pp"),
        _tile("crowd mean", f"${a['mean_ps']:.1f}", d("mean_ps", prev), d("mean_ps", start)),
        _tile("crowd median", f"${a['median_ps']:.1f}",
              d("median_ps", prev), d("median_ps", start)),
        _tile("perp mark", f"${perp:.2f}" if perp is not None else "—",
              d("perp", prev), d("perp", start)),
        _tile("perp − mean", f"{gap:+.1f} $/sh" if gap is not None else "—", None, None),
        _tile("spot", spot_val, d("spot", prev), d("spot", start)),
        _tile(f"EV vs ${rep['offer']:.0f} (A1)", f"{a['mean_ps']-rep['offer']:+.1f} $/sh",
              d("mean_ps", prev), d("mean_ps", start)),
    ])

    # 3-6. charts, each degrading independently
    def panel(fn, *args_, title=""):
        try:
            return fn(*args_)
        except Exception as e:  # a failed chart must never kill the page
            return _placeholder(title, type(e).__name__)

    svg_a = panel(chart_survivor, rep, state, snap, title="survivor curve")
    svg_b = panel(chart_pdf, rep, title="implied PDF")
    svg_c = panel(chart_timeseries, state, title="session time series")
    svg_d = panel(chart_tails, state, title="PEAK/tail panel")

    # 7. bucket table
    rows = ""
    for r in rep["buckets"]:
        div = abs(r["gap_pp"]) > 5
        rows += (f"<tr class='{'divergent' if div else ''}'><td>{r['label']}</td>"
                 f"<td>{r['ladder_implied']*100:.1f}</td>"
                 f"<td>{r['bucket_normalized']*100:.1f}</td>"
                 f"<td>{r['gap_pp']:+.1f}{' DIVERGENT' if div else ''}</td></tr>")
    bucket_tbl = (f"<table><tr><th>bucket</th><th>ladder %</th><th>market %</th>"
                  f"<th>gap (pp)</th></tr>{rows}</table>") if rows else \
        "<div class='ph'>bucket market: no usable quotes this poll</div>"

    # 8. alert ledger
    led = ""
    for level, label in state._levels():
        hit = state.alert_first_touch.get(level)
        if hit:
            t_str = datetime.fromtimestamp(hit[0], _cet()).strftime("%H:%M CET")
            cell = f"<td class='hit'>CROSSED {t_str} ({hit[1]})</td>"
        else:
            cell = "<td class='nohit'>not crossed</td>"
        led += f"<tr><td>${level:.0f}</td><td>{label}</td>{cell}</tr>"
    ledger = (f"<table><tr><th>level</th><th>meaning</th><th>state (this session)</th></tr>"
              f"{led}</table>")

    hist_note = (f"history: {len(state.history)} polls (last {state.history_hours:.0f}h"
                 + (f"; {state.dropped_old} older dropped" if state.dropped_old else "")
                 + ")")
    css = """
    body{font-family:-apple-system,system-ui,sans-serif;background:#14171a;color:#e8ecef;
         margin:0;padding:10px;max-width:780px;margin-inline:auto}
    .strip{font-size:13px;padding:6px 0;border-bottom:1px solid #333}
    .chip{padding:2px 8px;border-radius:10px;margin-left:6px;font-weight:600;font-size:12px}
    .chip.green{background:#143;color:#6e6}.chip.amber{background:#430;color:#ea3}
    .chip.red{background:#411;color:#f66}
    .flags{color:#ea3;font-size:12px;margin-top:3px}
    .tiles{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
    .tile{background:#1d2126;border-radius:8px;padding:8px 12px;flex:1 1 96px;min-width:96px}
    .tl{font-size:11px;color:#9ab}.tv{font-size:21px;font-weight:700;margin:2px 0}
    .td{font-size:11px;color:#789}
    .d.up{color:#6e6}.d.dn{color:#f66}
    h3{margin:18px 0 4px;font-size:14px;color:#9ab}
    .cap{font-size:12px;color:#789;margin:2px 0 8px}
    svg{width:100%;height:auto;background:#fff;border-radius:6px}
    table{border-collapse:collapse;width:100%;font-size:13px}
    td,th{padding:4px 8px;text-align:right;border-bottom:1px solid #2a2f35}
    td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
    tr.divergent td{background:#412;color:#fa6;font-weight:700}
    td.hit{color:#f66;font-weight:700}td.nohit{color:#6e6}
    .ph{background:#1d2126;border:1px dashed #555;border-radius:6px;padding:18px;
        color:#9ab;font-size:13px;margin:6px 0}
    .foot{font-size:11px;color:#678;margin-top:14px}
    .pbcard{background:#20262c;border:1px solid #345;border-radius:10px;padding:10px 14px;
            margin:10px 0}
    .pbnode{font-size:17px}.pbsub{font-size:11px;color:#789;font-weight:400}
    .pbul{margin:8px 0;padding-left:18px;font-size:13.5px;line-height:1.55}
    .pbwatch{font-size:13px;color:#9cf;border-top:1px solid #345;padding-top:6px}
    .pbmiss{font-size:12px;color:#ea3;margin-top:4px}
    .src{font-size:10.5px;color:#678}
    .go{color:#6e6}.nogo{color:#f66}
    code{background:#14171a;padding:1px 4px;border-radius:4px}
    """
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta http-equiv='refresh' content='{refresh_s}'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>SPCX PM-PDF dashboard</title><style>{css}</style></head><body>
{strip}
{playbook_html}
<div class='tiles'>{tiles}</div>
<h3>A · survivor curve S(K) = P(close cap &gt; K)</h3>
<div class='cap'>red curve = current PCHIP fit; dots = strike quotes on the {rep['basis']} basis
(red dot = clipped monotonicity violation); blue whiskers = bid–ask; dashed grey = reference
({state.reference['label'] if state.reference else '—'}). Drift of red away from grey = the
crowd repricing. Top axis converts cap to $/share (13.076B).</div>
{svg_a}
<h3>B · implied closing-price PDF</h3>
<div class='cap'>Density of the day-1 close per $/share; shaded = central P25–P75 mass; lines:
offer (green), perp (orange dashed), spot (blue dashed). Mode marker is shape-fragile under
PCHIP — read central stats, not bumps (construction audit).</div>
{svg_b}
<h3>C · session time series</h3>
<div class='cap'>Crowd mean/median, perp and spot in $/share (left); P(close&gt;offer) in %
(right, green). Dotted verticals = --mark events. Crowd ≈ flat while perp moves = venues
disagreeing — pick a side before trading on it.</div>
{svg_c}
<h3>D · PEAK / tail panel — the in-a-hurry chart</h3>
<div class='cap'>Bid (thick) and ask (dotted) of the &gt;$2.2T / &gt;$2.4T / &gt;$3.0T YES
strikes, in cents. Tails bid up = crowd chasing the rip (sell-into window); tails fading from
highs = the PEAK signal confirming. This is Alvaro's tail-sell screen.</div>
{svg_d}
<h3>bucket vs ladder (cross-market consistency)</h3>
<div class='cap'>gap = bucket market richer (+) than the ladder-implied probability, in points;
|gap| &gt; 5pp rows highlight DIVERGENT (the 06-07 1.5–2.0T gap was +7.8pp; it has since
closed).</div>
{bucket_tbl}
<h3>alert ledger (display only — alerting lives in TradingView)</h3>
<div class='cap'>Pre-registered levels, each crossed in its trigger direction relative to the
offer (upside levels when price ≥ level, downside when ≤). Series = spot once printing, perp
before. CROSSED shows the first-crossing time this session.</div>
{ledger}
<div class='foot'>{hist_note} · auto-refresh {refresh_s}s · self-contained page (renders
offline) · read-only monitor — no orders, no alerts. Conventions:
spcx_pm_pdf_monitor_findings.md</div>
</body></html>"""


# --------------------------------------------------------------------------------------
# Block S5d — "what now" playbook panel: the gameplan decision tree rendered live.
# A CHECKLIST RENDERER for rules frozen in spcx_listing_day_gameplan.md §3/§5/§6 and the
# S2 unwind-tape findings §6. It quotes those rules and evaluates them against live
# inputs; it invents no thresholds and places/suggests no orders.
# --------------------------------------------------------------------------------------
PLAYBOOK_STATE_FILE = DEFAULT_OUT_DIR / "playbook_state.json"
SETTLE_UTC_DEFAULT = "2026-06-12T20:00:00+00:00"  # Fri 16:00 ET close (calc --settle-utc)
XYZ_FEE_BPS = 4.5          # HL taker fee, per the calc's --hl-fee-bps default
XYZ_FEE_SIDES = 2.0        # xyz converts in place → pair-close = 2 fee sides (calc rule)
BELL_CET = (15, 30)        # Nasdaq bell; D4 display-only window starts here
CLOSEOUT_CET = (21, 30)    # last 30 min: no new decisions (gameplan §6 D5)
PERP_RISKOFF_LEVEL = 140.0  # gameplan §6 D3: perp ≤ ~$140 → pop thesis weakening
COMFORT_SHARES_DEFAULT = 22.0  # operator risk input 2026-06-10: ~€2-3k happy long-only;
FORCED_FLAT_CET = (21, 0)      # the D2 hedge is an OVERFLOW VALVE above this, and the
NO_CROSS_PERP_CLOSE_CET = (20, 30)  # perp is force-flattened by 21:00 (gameplan §6 D5)
TRACKING_GAP_USD = 2.0     # gameplan §5.1: Cerebras tracked within ~$1–4 → small-gap chip
TRACKING_MINUTES = 15.0    # sustained-gap window for the pair-close chip
PAIRCLOSE_MIN_MINS = 60.0  # §5.1: mid-session after the cross settles (1–3h), not first 30m
# S6 readiness ladder (CBRS 15m-vs-1m tape, n=1 — spcx_ipo_unwind_tape_findings § Block S6):
S6_NOZONE_MIN = 45.0       # first ~45 min post-cross = genuine no-pair-close zone (±$4–8)
S6_GATE2_MIN = 46.0        # |gap| ≤ $2 sustained from +46 min
S6_GATE1_MIN = 61.0        # |gap| ≤ $1 from +61 min
S6_GATE1_USD = 1.0
BACKSTOP_COST_PS = 1.7     # S6-measured backstop cost on the CBRS analog, ~$/share gap drag
FASTTAPE_BUFFER_FRAC = 0.25   # §5.1 fast-tape exception input 1: liq buffer < 25%
FASTTAPE_MOVE_FRAC = 0.02     # input 2: >2% spot move in the last 5 minutes
FASTTAPE_WINDOW_S = 300.0
STOP_LADDER = [(140.0, "reassess"), (135.0, "offer"), (125.0, "sell-everything stop")]
# Tranche phases in minutes since cross (gameplan §5.2 re-anchored; S2 §4 calibration):
TRANCHE_PHASES = [(0, 15, "OBSERVE", 0.0), (15, 60, "T1", 40.0),
                  (60, 180, "T2", 80.0), (180, 10 ** 9, "T3", 100.0)]


class PlaybookState:
    """User-fed day state, persisted to a small JSON file so restarts resume.

    Fields are None until the user provides them; the panel renders 'awaiting --x'
    rather than assuming defaults for unset inputs."""

    FIELDS = ("offer", "fill", "hedged", "hedge_entry", "hedge_lev", "sold",
              "cross_ts", "cross_price", "sub_eur", "margin_eur", "eurusd", "comfort",
              "hedge_ts", "shape_override", "indications")
    STR_FIELDS = ("shape_override",)   # not float-castable in the /api/state setter
    LIST_FIELDS = ("indications",)

    def __init__(self, path: Path | None = None):
        self.path = path
        self.offer: float | None = None
        self.fill: float | None = None
        self.hedged: float | None = None
        self.hedge_entry: float | None = None
        self.hedge_lev: float = 1.0
        self.sold: float = 0.0
        self.cross_ts: float | None = None
        self.cross_price: float | None = None
        self.sub_eur: float = 10_000.0
        self.margin_eur: float = 2_000.0
        self.eurusd: float | None = None
        self.comfort: float = COMFORT_SHARES_DEFAULT
        self.hedge_ts: float | None = None       # epoch when hedged was first set (S5h)
        self.shape_override: str | None = None   # S7 manual pin (S5h item 3)
        self.indications: list = []              # [{ts, text}] operator-pasted (S5h item 4)
        self.avwap_lost_since: float | None = None  # runtime only, not persisted
        if path and path.exists():
            try:
                data = json.loads(path.read_text())
                for f in self.FIELDS:
                    if data.get(f) is not None:
                        setattr(self, f, data[f])
            except (ValueError, OSError):
                pass

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({f: getattr(self, f) for f in self.FIELDS}, indent=2))
            os.replace(tmp, self.path)

    def update_avwap_state(self, spot: float | None, avwap: float | None,
                           now: float) -> None:
        """Track 'price lost anchored VWAP N min ago, not reclaimed' (gameplan §5.3 #1)."""
        if spot is None or avwap is None:
            return
        if spot < avwap:
            if self.avwap_lost_since is None:
                self.avwap_lost_since = now
        else:
            self.avwap_lost_since = None


def parse_cross_arg(vals: list[str], now: float) -> tuple[float, float]:
    """--cross <HH:MM|now> <price> (CET) → (epoch, price)."""
    t_str, p_str = vals
    price = float(p_str)
    if t_str.lower() == "now":
        return now, price
    h, m = map(int, t_str.split(":"))
    dt = datetime.fromtimestamp(now, _cet()).replace(hour=h, minute=m, second=0,
                                                     microsecond=0)
    return dt.timestamp(), price


def infer_node(pb: PlaybookState, now: float) -> str:
    """D-node per gameplan §6: fill unset → PRE-ALLOC; after 21:30 CET → CLOSE-OUT;
    cross logged → D5; after the 15:30 bell → D4; else D2/D3."""
    if pb.fill is None:
        return "PRE-ALLOC"
    t = datetime.fromtimestamp(now, _cet())
    if (t.hour, t.minute) >= CLOSEOUT_CET:
        return "CLOSE-OUT"
    if pb.cross_ts is not None:
        return "D5"
    if (t.hour, t.minute) >= BELL_CET:
        return "D4"
    return "D2/D3"


def hedge_rule_eval(pb: PlaybookState, mark: float, funding_hourly: float,
                    now: float) -> dict:
    """The S1/D2 rule evaluated live, arithmetic delegated to spcx_convergence_calc
    (build_hedge_grid — no third implementation). xyz convention: R=1, per-share."""
    from scripts.spcx_convergence_calc import build_hedge_grid, shares_requested

    eurusd = pb.eurusd or 1.08
    offer = pb.offer if pb.offer is not None else IPO_OFFER_DEFAULT
    req = shares_requested(pb.sub_eur, eurusd, offer)
    frac = (pb.fill / req) if req > 0 else 0.0
    comfort = pb.comfort if pb.comfort is not None else COMFORT_SHARES_DEFAULT
    overflow = max(pb.fill - comfort, 0.0)
    frac_overflow = (overflow / req) if req > 0 else 0.0
    hours = max(0.0, (_epoch(SETTLE_UTC_DEFAULT) - now) / 3600.0)
    cells = build_hedge_grid(
        per_ipo_share_mark=mark, mark=mark, contracts_per_share=1.0,
        funding_hourly=funding_hourly, hours_to_settle=hours,
        fee_bps=XYZ_FEE_BPS, fee_sides=XYZ_FEE_SIDES,
        subscription_eur=pb.sub_eur, eurusd=eurusd, margin_eur=pb.margin_eur,
        fill_prices=[offer], fill_fracs=[frac_overflow], leverages=[1.0, 1.5])
    by_lev = {c.leverage: c for c in cells}
    nearest_row = min((10, 25, 50, 100), key=lambda x: abs(x - frac * 100))
    c15 = by_lev[1.5]
    net_basis_ps = (c15.locked_net / c15.hedged_shares) if c15.hedged_shares > 0 else \
        (mark - offer)
    return {"cells": by_lev, "req": req, "frac": frac, "nearest_fill_row": nearest_row,
            "comfort": comfort, "overflow": overflow,
            "gross_basis": mark - offer, "net_basis_ps": net_basis_ps,
            "eurusd": eurusd, "go": net_basis_ps > 0 and overflow > 0}


# --------------------------------------------------------------------------------------
# Block S5h item 1 — hedge-ops panel: display of EXISTING math while the short is on.
# Buffer/liq arithmetic is IMPORTED from spcx_convergence_calc (never reimplemented);
# funding accrual uses the public HL funding rate (read-only monitor: no account API).
# --------------------------------------------------------------------------------------
LEG_SEQ_DEFAULT = ("Execute the TR sell first (slower venue, limit order), then "
                   "immediately close the perp (fast venue). Leg risk is "
                   "seconds-to-minutes of one-sided exposure.")
LEG_SEQ_EXCEPTION = ("Fast-tape exception (pre-registered judgment, NOT measured — there "
                     "is no tape to calibrate this): if the liq buffer is < 25% or the "
                     "stock has moved > 2% in the last 5 minutes, reverse the order — "
                     "close the perp first, then the shares; a few minutes of unhedged "
                     "long is survivable, a liquidated short is not.")


def spot_move_5m(history: list[dict], now: float,
                 window_s: float = FASTTAPE_WINDOW_S) -> float | None:
    """Fractional spot move over the trailing window, from the poll history (None until
    two spot prints exist inside the window)."""
    pts = [(r["ts"], r["spot"]) for r in history
           if r.get("spot") is not None and r["ts"] >= now - window_s]
    if len(pts) < 2 or pts[0][1] == 0:
        return None
    return (pts[-1][1] - pts[0][1]) / pts[0][1]


def hedge_ops_eval(pb: PlaybookState, mark: float | None, spot: float | None,
                   funding_hourly: float, max_leverage: float | None,
                   history: list[dict], now: float) -> dict | None:
    """Everything the hedge-ops panel shows, computed from declared state + public feeds.
    None when no hedge is on. Buffer math: spcx_convergence_calc (single source)."""
    from scripts.spcx_convergence_calc import (
        liq_buffer_summary, liq_price_short, maintenance_margin_frac)

    if not pb.hedged or pb.hedge_entry is None:
        return None
    offer = pb.offer if pb.offer is not None else IPO_OFFER_DEFAULT
    eurusd = pb.eurusd or 1.08
    max_lev = max_leverage if max_leverage else 3.0
    mmr = maintenance_margin_frac(max_lev)
    liq_px = liq_price_short(pb.hedge_entry, pb.hedge_lev, mmr)
    buf = (liq_buffer_summary(mark, liq_px, alert_pct=0.10) if mark is not None
           else {"current_mark": None, "liq_px": liq_px, "buffer_frac": None,
                 "band": "NO MARK"})
    basis_ps = pb.hedge_entry - offer
    hours_on = max(0.0, (now - pb.hedge_ts) / 3600.0) if pb.hedge_ts else None
    notional = (mark if mark is not None else pb.hedge_entry) * pb.hedged
    funding_accrued = (funding_hourly * notional * hours_on
                       if hours_on is not None else None)
    # S6 readiness ladder, anchored to the cross
    mins = (now - pb.cross_ts) / 60.0 if pb.cross_ts is not None else None
    gap = (mark - spot) if (mark is not None and spot is not None) else None
    pc = pair_close_status(history, pb.cross_ts, now)
    if mins is None:
        ladder = ("PRE-CROSS", "ladder starts at the cross print")
    elif mins <= S6_NOZONE_MIN:
        ladder = ("NO-ZONE", f"no pair-close zone — {S6_GATE2_MIN - mins:.0f} min to the "
                  f"$2 gate (S6: ±$4–8 dislocations in the first ~45 min)")
    elif mins < S6_GATE1_MIN:
        ok = gap is not None and abs(gap) <= TRACKING_GAP_USD
        ladder = ("GATE-2", f"$2 gate open ({'gap OK' if ok else 'gap NOT inside $2'}) — "
                  f"{S6_GATE1_MIN - mins:.0f} min to the $1 gate")
    else:
        ok1 = gap is not None and abs(gap) <= S6_GATE1_USD
        ladder = ("GREEN" if pc["green"] else "GATE-1",
                  f"$1 gate live ({'gap ≤ $1' if ok1 else 'gap > $1'}); pair-close chip "
                  f"{'GREEN — readiness confirmed' if pc['green'] else 'not yet green'}")
    # forced-flat backstop countdown
    ff = datetime.fromtimestamp(now, _cet()).replace(
        hour=FORCED_FLAT_CET[0], minute=FORCED_FLAT_CET[1], second=0, microsecond=0)
    backstop_mins = (ff.timestamp() - now) / 60.0
    # §5.1 fast-tape exception, both inputs evaluated live
    move = spot_move_5m(history, now)
    in1 = buf["buffer_frac"] is not None and buf["buffer_frac"] < FASTTAPE_BUFFER_FRAC
    in2 = move is not None and abs(move) > FASTTAPE_MOVE_FRAC
    return {"entry": pb.hedge_entry, "lev": pb.hedge_lev, "hedged": pb.hedged,
            "basis_ps": basis_ps, "basis_total": basis_ps * pb.hedged,
            "mmr": mmr, "max_lev": max_lev, "liq_px": liq_px, "buffer": buf,
            "margin_posted_usd": pb.margin_eur * eurusd,
            "funding_hourly": funding_hourly, "funding_accrued": funding_accrued,
            "hours_on": hours_on, "ladder": ladder, "gap": gap,
            "mins_since_cross": mins, "pair_close": pc,
            "backstop_mins": backstop_mins,
            "fasttape": {"buffer_lt_25": in1, "move_5m": move, "move_gt_2pct": in2,
                         "active": in1 or in2}}


def render_hedge_ops(ops: dict | None) -> str:
    """The hedge-ops panel HTML (empty string when no hedge is on)."""
    if ops is None:
        return ""
    b = ops["buffer"]
    band = b["band"]
    bcls = {"SAFE": "g", "WARN": "a"}.get(band, "r")
    bufpct = f"{b['buffer_frac'] * 100:.1f}%" if b["buffer_frac"] is not None else "n/a"
    fund = (f"${ops['funding_accrued']:+,.2f} accrued over {ops['hours_on']:.1f}h "
            f"(≈ at the current public rate {ops['funding_hourly'] * 100:.4f}%/h)"
            if ops["funding_accrued"] is not None
            else "n/a — hedge_ts unset (set when --hedged lands)")
    gap = f"${ops['gap']:+.2f}" if ops["gap"] is not None else "· (no spot yet)"
    ft = ops["fasttape"]
    move = f"{ft['move_5m'] * 100:+.2f}%" if ft["move_5m"] is not None else "n/a (no tape)"
    seq = ("⚠ PERP LEG FIRST — fast-tape exception ACTIVE" if ft["active"]
           else "TR sell first, then perp (default sequence)")
    lname, ltxt = ops["ladder"]
    rows = [
        f"<div class='hopgrid'>"
        f"<div><span class='tl'>SHORT</span><div class='num hv'>{ops['hedged']:.0f} sh @ "
        f"${ops['entry']:.2f} · {ops['lev']:g}×</div></div>"
        f"<div><span class='tl'>BASIS LOCKED</span><div class='num hv'>"
        f"${ops['basis_ps']:+.2f}/sh · ${ops['basis_total']:+,.0f} total</div></div>"
        f"<div><span class='tl'>LIQ PRICE</span><div class='num hv'>${ops['liq_px']:.2f}"
        f"</div></div>"
        f"<div><span class='tl'>BUFFER</span><div class='num hv'>"
        f"<span class='chip {bcls}'>{band}</span> {bufpct}</div></div>"
        f"<div><span class='tl'>MARGIN POSTED</span><div class='num hv'>"
        f"${ops['margin_posted_usd']:,.0f}</div></div>"
        f"<div><span class='tl'>FUNDING</span><div class='num hv' style='font-size:14px'>"
        f"{fund}</div></div>"
        f"</div>",
        f"<div class='hopline'><b>S6 readiness ladder:</b> <span class='chip "
        f"{'g' if lname == 'GREEN' else 'a' if lname in ('GATE-1', 'GATE-2') else 'r'}'>"
        f"{lname}</span> {ltxt} · gap (perp−spot) <b class='num'>{gap}</b></div>",
        f"<div class='hopline'><b>21:00 CET backstop:</b> <span class='num'>"
        f"{ops['backstop_mins']:.0f} min</span> — if readiness has not confirmed, "
        f"pair-close anyway (S6-measured cost ≈ ${BACKSTOP_COST_PS:.1f}/sh gap drag)</div>",
        f"<div class='hopline'><b>Leg sequencing:</b> <b>{seq}</b><br>"
        f"<span class='src'>§5.1 verbatim:</span> {LEG_SEQ_DEFAULT}<br>"
        f"<span class='src'>§5.1 verbatim:</span> {LEG_SEQ_EXCEPTION}<br>"
        f"<span class='num' style='font-size:13px'>inputs live: buffer &lt; 25% → "
        f"{'YES' if ft['buffer_lt_25'] else 'no'} · 5-min move {move} (&gt; 2% → "
        f"{'YES' if ft['move_gt_2pct'] else 'no'})</span> "
        f"<span class='src'>[pre-registered judgment, not measured]</span></div>",
    ]
    return ("<div class='hopcard'><div class='hoph'>HEDGE OPS — short is ON "
            "<span class='src'>[buffer math: spcx_convergence_calc]</span></div>"
            + "".join(rows) + "</div>")


def tranche_phase(mins_since_cross: float) -> tuple[str, float]:
    """(phase chip, target cumulative residual-sold %) — S2-calibrated gameplan §5.2."""
    for lo, hi, name, target in TRANCHE_PHASES:
        if lo <= mins_since_cross < hi:
            return name, target
    return "T3", 100.0


# --------------------------------------------------------------------------------------
# Block S5h item 2 — lean tranche tables (gameplan §5.2, decided 2026-06-11) + day-shape
# display overrides. A ticket = one manual €1 limit order; windows in minutes post-cross:
# T1 = +15–60, T2 = +60–180, T3 = +180 → 21:30 CET. Overrides change the DISPLAYED
# schedule only — no order logic exists anywhere in this codebase.
# --------------------------------------------------------------------------------------
TRANCHE_ROWS = {20: [8, 8, 5], 40: [17, 17, 9], 65: [26, 26, 13]}
TRANCHE_WINDOWS = [(15, 60), (60, 180), (180, None)]   # None = until 21:30 CET


def select_tranche_plan(residual: float) -> dict:
    """Pick the §5.2 lean row from the residual-sleeve size (fill − hedge sleeve).
    ≤10 sh → two tickets 60/40 (skip T3); else the nearest of the 20/40/65 rows, shown
    verbatim with the LAST ticket absorbing the difference so totals match the actual
    residual. The 65-row carries the 'T1 immediately after the observe window' flag."""
    residual = max(0.0, residual)
    if residual <= 0:
        return {"row": None, "tickets": [], "note": "no residual sleeve"}
    if residual <= 10:
        t1 = round(residual * 0.6)
        return {"row": "≤10", "tickets": [("T1", t1), ("T2", residual - t1)],
                "note": "≤10 sh — two tickets 60/40, skip T3", "t1_immediate": False}
    row = min(TRANCHE_ROWS, key=lambda r: abs(r - residual))
    base = TRANCHE_ROWS[row]
    diff = residual - sum(base)
    tickets = [("T1", float(base[0])), ("T2", float(base[1])),
               ("T3", float(base[2]) + diff)]
    note = (f"~{row} sh row {base[0]}/{base[1]}/{base[2]}"
            + (f" (T3 {diff:+.0f} to match {residual:.0f} sh residual)" if diff else ""))
    return {"row": f"~{row}", "tickets": tickets, "note": note,
            "t1_immediate": row == 65}


def tranche_schedule(fill: float | None, hedged: float | None, sold: float,
                     cross_ts: float | None, now: float, shape: str = "FLAT",
                     session_high: float | None = None) -> dict:
    """The §5.2 schedule as displayed: per-ticket shares, window, done/active/pending vs
    cumulative `sold`, with the day-shape override applied to the WINDOWS (display only).
    FADE halves every remaining window; RALLY defers the final ticket (mental stop 10%
    below session high); CRASH hands over to the hard-stop ladder; FLAT default."""
    residual = max((fill or 0.0) - (hedged or 0.0), 0.0)
    plan = select_tranche_plan(residual)
    out = {"residual": residual, "plan": plan, "shape": shape, "rows": [],
           "flag": None, "sold": sold}
    if not plan["tickets"]:
        return out
    if shape == "CRASH":
        out["flag"] = ("CRASH — schedule void: hard-stop ladder governs "
                       "($140 reassess / $125 sell everything / below cross+$160 → out)")
        return out
    mins = (now - cross_ts) / 60.0 if cross_ts is not None else None
    cum = 0.0
    n = len(plan["tickets"])
    for i, (name, shares) in enumerate(plan["tickets"]):
        lo, hi = TRANCHE_WINDOWS[0 if name == "T1" else 1 if name == "T2" else 2]
        # ≤10 row uses the T1/T2 windows it names; window end None = 21:30 CET
        win_lo, win_hi = float(lo), (float(hi) if hi is not None else None)
        deferred = False
        if shape == "FADE" and mins is not None:
            # halve what REMAINS of each window from now (pre-registered §5.2 override)
            if win_hi is not None and win_hi > mins:
                win_hi = mins + (win_hi - mins) / 2.0
            if win_lo > mins:
                win_lo = mins + (win_lo - mins) / 2.0
        if shape == "RALLY" and i == n - 1:
            deferred = True
        cum += shares
        done = sold >= cum - 1e-9
        partial = (not done) and sold > cum - shares + 1e-9
        if mins is None:
            status = "pending (windows arm at the cross)"
        elif done:
            status = "DONE"
        elif partial:
            status = f"PARTIAL ({sold - (cum - shares):.0f}/{shares:.0f})"
        elif deferred:
            status = "DEFERRED (RALLY) — hold, 21:30 CET hard end"
        elif win_hi is not None and mins > win_hi:
            status = "OVERDUE"
        elif mins >= win_lo:
            status = "ACTIVE — place now"
        else:
            status = f"opens +{win_lo:.0f} min"
        win_txt = (f"+{win_lo:.0f}–{win_hi:.0f} min" if win_hi is not None
                   else f"+{win_lo:.0f} min → 21:30 CET")
        out["rows"].append({"name": name, "shares": shares, "window": win_txt,
                            "win_lo_min": win_lo, "win_hi_min": win_hi,  # numeric, for
                            "cum_target": cum,                           # the S5j chart
                            "status": status, "deferred": deferred})
    if shape == "FADE":
        nxt = next((r for r in out["rows"] if r["status"].startswith(("ACTIVE", "opens",
                                                                      "OVERDUE"))), None)
        out["flag"] = (f"FADE — sell the next tranche NOW ({nxt['name']} "
                       f"{nxt['shares']:.0f} sh), remaining windows HALVED" if nxt
                       else "FADE — all tickets done")
    elif shape == "RALLY":
        stop = f"${session_high * 0.9:.2f}" if session_high else "0.9 × session high"
        out["flag"] = (f"RALLY — final ticket deferred; mental stop {stop} "
                       f"(10% below session high); call the PEAK check with Alvaro")
    if plan.get("t1_immediate") and mins is not None and mins >= 15 and sold <= 0:
        out["flag"] = ((out["flag"] + " · " if out["flag"] else "")
                       + "65-row: T1 places IMMEDIATELY post-observe — risk reduction, "
                         "not price optimization")
    return out


# --------------------------------------------------------------------------------------
# Block S5j — interpretation-chart data (display only). Chart 1 consumes tranche_schedule's
# OWN numeric window bounds (the override arithmetic lives in one place and is already
# tested); chart 2's buffer series calls the calc module's liq_buffer_summary per point
# (imported, never reimplemented). Both blocks are shared verbatim by the live payload
# and the rehearsal panel, so the dress run exercises the exact production path.
# --------------------------------------------------------------------------------------
def tranche_chart_data(tr: dict, cross_ts: float | None, now: float) -> dict | None:
    """Chart-1 block from a tranche_schedule() result: numeric bands/blocks/playhead/
    cumulative targets. None when there is nothing to draw."""
    if not tr["rows"] and not tr.get("flag"):
        return None
    end_min = None
    now_min = None
    if cross_ts is not None:
        ff = datetime.fromtimestamp(now, _cet()).replace(
            hour=CLOSEOUT_CET[0], minute=CLOSEOUT_CET[1], second=0, microsecond=0)
        end_min = max((ff.timestamp() - cross_ts) / 60.0, 1.0)
        now_min = (now - cross_ts) / 60.0
    rows = [{"name": r["name"], "shares": r["shares"],
             "lo": r["win_lo_min"],
             "hi": r["win_hi_min"] if r["win_hi_min"] is not None else end_min,
             "open_ended": r["win_hi_min"] is None,
             "done": r["status"] == "DONE",
             "partial": r["status"].startswith("PARTIAL"),
             "active": r["status"].startswith(("ACTIVE", "OVERDUE")),
             "deferred": r["deferred"], "cum_target": r["cum_target"]}
            for r in tr["rows"]]
    return {"rows": rows, "sold": tr["sold"], "shape": tr["shape"],
            "flag": tr.get("flag"), "residual": tr["residual"],
            "crash": tr["shape"] == "CRASH", "now_min": now_min, "end_min": end_min}


def hedge_chart_data(ops: dict | None, history: list[dict], cross_ts: float | None,
                     now: float, green_since: float | None = None) -> dict | None:
    """Chart-2 block: per-poll gap + liq-buffer series (buffer via the calc module's
    liq_buffer_summary at the entry-fixed liq price), S6 zone boundaries, the 21:00 CET
    backstop, and the close-both-legs-now readout = (locked basis − gap) × hedged."""
    if ops is None:
        return None
    from scripts.spcx_convergence_calc import liq_buffer_summary

    liq = ops["liq_px"]
    series = []
    for r in history:
        p = r.get("perp")
        if p is None or p <= 0:
            continue
        s = r.get("spot")
        b = liq_buffer_summary(p, liq, alert_pct=0.10)
        series.append({"ts": r["ts"], "gap": (p - s) if s is not None else None,
                       "buffer": b["buffer_frac"], "band": b["band"]})
    ff = datetime.fromtimestamp(now, _cet()).replace(
        hour=FORCED_FLAT_CET[0], minute=FORCED_FLAT_CET[1], second=0, microsecond=0)
    gap = ops["gap"]
    locked_total = ops["basis_ps"] * ops["hedged"]
    drag_total = (gap or 0.0) * ops["hedged"]
    return {"series": series, "liq_px": liq, "entry": ops["entry"],
            "hedged": ops["hedged"], "basis_ps": ops["basis_ps"],
            "cross_ts": cross_ts,
            "zone2_ts": cross_ts + S6_GATE2_MIN * 60.0 if cross_ts else None,
            "zone1_ts": cross_ts + S6_GATE1_MIN * 60.0 if cross_ts else None,
            "backstop_ts": ff.timestamp(), "backstop_cost_ps": BACKSTOP_COST_PS,
            "green_since": green_since,
            "close_now": {"gap": gap, "locked_total": locked_total,
                          "drag_total": drag_total if gap is not None else None,
                          "net_total": (locked_total - drag_total) if gap is not None
                          else None}}


# --------------------------------------------------------------------------------------
# Block S5h item 3 — S7 day-shape classifier (gameplan §5.3, pre-registered 2026-06-11).
# Judgment thresholds, sanity-replayed on the CBRS tape but NOT tuned to it; NO volume
# features (IEX coverage is too thin for them). Manual override always wins. Display
# only — the human sells; this labels which §5.2 row applies.
# --------------------------------------------------------------------------------------
SHAPE_CRASH_FLOOR = 160.0
SHAPE_HARD_STOPS = (140.0, 125.0)       # §5.4: $140 reassess / $125 sell-everything
SHAPE_FADE_AVWAP_SECS = 600.0           # below anchored VWAP for ≥10 min
SHAPE_FADE_OFF_HIGH = 0.05              # AND >5% off the session high
SHAPE_RALLY_NEWHIGH_SECS = 900.0        # new session high within the last 15 min
SHAPE_HYSTERESIS_SECS = 120.0           # 2 consecutive minutes to switch (CRASH instant)

SHAPE_ACTIONS = {  # §0 decision-tree rows, verbatim (mermaid line breaks → " — ")
    "FADE": "FRONT-LOAD: sell fast — the opening price was ~89% of the day's high in "
            "past mega-IPOs",
    "RALLY": "be patient, sell in steps — mental stop 10% below high — tell Alvaro: "
             "tails getting rich (PEAK)",
    "FLAT": "spread sales evenly, done by ~20:00",
    "CRASH": "sell everything now — hard rules: $140 reassess, $125 all out",
}


class DayShapeClassifier:
    """The §5.3 state machine, stepped once per poll with the live inputs (IEX spot,
    anchored VWAP, cross print). Tracks its own session high. CRASH switches instantly;
    every other switch needs the candidate to hold SHAPE_HYSTERESIS_SECS."""

    def __init__(self):
        self.state = "FLAT"
        self.since: float | None = None
        self.session_high: float | None = None
        self.last_new_high_ts: float | None = None
        self.below_avwap_since: float | None = None
        self._pending: tuple[str, float] | None = None
        self.armed = False              # becomes True at the first spot print

    def _raw(self, now: float, spot: float, avwap: float | None,
             cross_price: float | None) -> str:
        if (cross_price is not None and spot < cross_price
                and spot < SHAPE_CRASH_FLOOR) or spot <= min(SHAPE_HARD_STOPS):
            return "CRASH"
        if spot <= max(SHAPE_HARD_STOPS):          # any hard-stop level hit
            return "CRASH"
        fade = (self.below_avwap_since is not None
                and now - self.below_avwap_since >= SHAPE_FADE_AVWAP_SECS
                and self.session_high is not None
                and spot < self.session_high * (1.0 - SHAPE_FADE_OFF_HIGH))
        if fade:
            return "FADE"
        rally = (avwap is not None and spot > avwap
                 and self.last_new_high_ts is not None
                 and now - self.last_new_high_ts <= SHAPE_RALLY_NEWHIGH_SECS)
        if rally:
            return "RALLY"
        return "FLAT"

    def step(self, now: float, spot: float | None, avwap: float | None,
             cross_price: float | None) -> str:
        if spot is None:
            return self.state                      # nothing to read — hold
        self.armed = True
        if self.session_high is None or spot > self.session_high:
            self.session_high = spot
            self.last_new_high_ts = now
        if avwap is not None:
            if spot < avwap:
                if self.below_avwap_since is None:
                    self.below_avwap_since = now
            else:
                self.below_avwap_since = None
        raw = self._raw(now, spot, avwap, cross_price)
        if raw == self.state:
            self._pending = None
            return self.state
        if raw == "CRASH":                          # instant, no hysteresis
            self.state, self.since, self._pending = "CRASH", now, None
            return self.state
        if self._pending is None or self._pending[0] != raw:
            self._pending = (raw, now)
        elif now - self._pending[1] >= SHAPE_HYSTERESIS_SECS:
            self.state, self.since, self._pending = raw, now, None
        return self.state

    def view(self, override: str | None, now: float) -> dict:
        """What the banner shows: override pins until released, computed state beneath."""
        eff = override if override in SHAPE_ACTIONS else self.state
        mins = (now - self.since) / 60.0 if self.since is not None else None
        return {"state": eff, "computed": self.state, "override": override or None,
                "armed": self.armed, "minutes_in_state": mins,
                "session_high": self.session_high,
                "action": SHAPE_ACTIONS.get(eff, ""),
                "below_avwap_min": ((now - self.below_avwap_since) / 60.0
                                    if self.below_avwap_since else None)}


def replay_classifier_on_bars(bars: list[dict]) -> list[tuple[float, str]]:
    """Sanity replay (smoke test, NOT calibration): step the classifier over 1m OHLC bars
    (keys time_utc/close/volume). Cross = first bar with volume > 0; AVWAP = running
    Σ(close·vol)/Σvol from that bar. Returns [(minutes_since_cross, state)] transitions."""
    cls = DayShapeClassifier()
    cross_price = None
    t0 = None
    pv = v = 0.0
    out: list[tuple[float, str]] = []
    prev = cls.state
    for b in bars:
        vol = float(b.get("volume") or 0.0)
        if cross_price is None:
            if vol <= 0:
                continue
            cross_price = float(b["open"] if b.get("open") else b["close"])
            t0 = _epoch(b["time_utc"])
        ts = _epoch(b["time_utc"])
        px = float(b["close"])
        pv += px * vol
        v += vol
        avwap = pv / v if v > 0 else None
        st = cls.step(ts, px, avwap, cross_price)
        if st != prev:
            out.append(((ts - t0) / 60.0, st))
            prev = st
    return out


# --------------------------------------------------------------------------------------
# Block S5h item 4a — Nasdaq trading-halts poller (cross timing). Read-only GET of the
# public nasdaqtrader.com halts RSS; the page format is NOT an API contract, so parsing
# is defensive: any failure → "poller down — watch CNBC", never stale data shown as live.
# --------------------------------------------------------------------------------------
NASDAQ_HALTS_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"
HALTS_START_CET = (15, 0)
HALTS_POLL_SECS = 60.0


def parse_nasdaq_halts(xml_text: str, symbol: str = "SPCX") -> list[dict]:
    """Defensive parse of the halts RSS for one symbol. Namespace-agnostic tag matching;
    raises on malformed XML (the poller wrapper converts that to 'down')."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    out = []
    for item in root.iter():
        if not item.tag.lower().endswith("item"):
            continue
        rec: dict = {}
        for child in item.iter():
            tag = child.tag.rsplit("}", 1)[-1].lower()
            txt = (child.text or "").strip()
            if tag in ("issuesymbol", "haltdate", "halttime", "reasoncode",
                       "resumptiondate", "resumptionquotetime", "resumptiontradetime"):
                rec[tag] = txt
        if rec.get("issuesymbol", "").upper() == symbol.upper():
            out.append(rec)
    return out


class NasdaqHaltsPoller:
    """Background daemon thread polling the halts feed every HALTS_POLL_SECS, active only
    from 15:00 CET (the pre-cross window). snapshot() is lock-protected; `alert` is set
    exactly once when a resumption TRADE time first appears (= when the cross will print)."""

    def __init__(self, symbol: str = "SPCX"):
        self.symbol = symbol
        self._lock = threading.Lock()
        self._state: dict = {"status": "waiting", "halt": None, "fetched_at": None,
                             "alert": False}
        self._seen_trade_time: str | None = None
        self._stop = False

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True,
                         name="spcx-halts-poller").start()

    def _run(self) -> None:
        import httpx
        while not self._stop:
            t = datetime.now(_cet())
            if (t.hour, t.minute) < HALTS_START_CET:
                with self._lock:
                    self._state.update({"status": "waiting", "halt": None})
                time.sleep(HALTS_POLL_SECS)
                continue
            try:
                r = httpx.get(NASDAQ_HALTS_URL, timeout=20,
                              headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                halts = parse_nasdaq_halts(r.text, self.symbol)
            except Exception:
                with self._lock:
                    self._state.update({"status": "down", "halt": None,
                                        "fetched_at": time.time()})
                time.sleep(HALTS_POLL_SECS)
                continue
            halt = halts[0] if halts else None
            alert = False
            trade_t = (halt or {}).get("resumptiontradetime") or None
            if trade_t and trade_t != self._seen_trade_time:
                self._seen_trade_time = trade_t
                alert = True            # the cross print time just appeared/changed
            with self._lock:
                self._state = {"status": "ok" if halt else "no_entry", "halt": halt,
                               "fetched_at": time.time(), "alert": alert}
            time.sleep(HALTS_POLL_SECS)

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self._state)
        s["age_s"] = (time.time() - s["fetched_at"]) if s["fetched_at"] else None
        alert, self._state["alert"] = s["alert"], False   # consume the one-shot
        s["alert"] = alert
        return s


def render_halts(hs: dict | None) -> str:
    """Cross-timing card HTML (the halts feed lists resumption quote/trade times —
    resumption TRADE time = when the cross will print)."""
    if hs is None:
        return ""
    age = f"{hs['age_s']:.0f}s ago" if hs.get("age_s") is not None else "never"
    if hs["status"] == "waiting":
        body = "polling starts 15:00 CET (pre-cross window)"
    elif hs["status"] == "down":
        body = "<b class='nogo'>poller down — watch CNBC</b>"
    elif hs["status"] == "no_entry":
        body = (f"SPCX not in the halts feed (normal before the IPO halt is posted) "
                f"<span class='src'>checked {age}</span>")
    else:
        h = hs["halt"]
        rq = h.get("resumptionquotetime") or "—"
        rt = h.get("resumptiontradetime") or "— (not yet scheduled)"
        body = (f"halt code <b>{h.get('reasoncode', '?')}</b> @ "
                f"{h.get('haltdate', '')} {h.get('halttime', '')} ET · resumption quote "
                f"<b class='num'>{rq}</b> · <b>resumption TRADE (the cross): "
                f"<span class='num'>{rt}</span></b> "
                f"<span class='src'>checked {age}</span>")
    return (f"<div class='xtcard'><div class='xth'>CROSS TIMING — Nasdaq halts feed "
            f"<span class='src'>[ET times; trade resumption = cross print]</span></div>"
            f"<div class='xtbody'>{body}</div></div>")


def render_indications(pb: PlaybookState, perp: float | None,
                       pm_mean: float | None) -> str:
    """Operator-pasted 'indicated to open' ranges (CNBC/X relays), latest vs perp + PM
    mean, with history. Manual paste only — no headline scraping exists."""
    rows = list(pb.indications or [])
    perp_s = f"${perp:.1f}" if perp is not None else "·"
    mean_s = f"${pm_mean:.1f}" if pm_mean is not None else "·"
    if not rows:
        body = ("no indication pasted yet — when CNBC/X relays one, type it here "
                "(e.g. <code>148-152</code>)")
    else:
        last = rows[-1]
        t = datetime.fromtimestamp(last["ts"], _cet()).strftime("%H:%M CET")
        hist = " · ".join(
            f"{r['text']} ({datetime.fromtimestamp(r['ts'], _cet()):%H:%M})"
            for r in rows[:-1][-4:])
        body = (f"latest <b class='num' style='font-size:19px'>{last['text']}</b> "
                f"@ {t} &nbsp;vs&nbsp; perp <b class='num'>{perp_s}</b> · PM mean "
                f"<b class='num'>{mean_s}</b>"
                + (f"<div class='src'>earlier: {hist}</div>" if hist else ""))
    return (f"<div class='xtcard'><div class='xth'>INDICATION (pasted)</div>"
            f"<div class='xtbody'>{body}</div>"
            f"<div class='mkrow' style='margin-top:8px'><input id='ind_in' "
            f"placeholder='148-152' style='width:130px'>"
            f"<button onclick='sendIndication()'>ADD</button></div></div>")


# --------------------------------------------------------------------------------------
# Block S5k — PM tab (display-only): full distribution table, bucket-vs-ladder with gap
# sparklines, perp-vs-crowd strip. Everything here renders values the engine already
# computes (analyze() + DashboardState history); no new math, no signals, no orders.
# --------------------------------------------------------------------------------------
PM_TAB_NOTE = ("Analytics for the PM leg (Alvaro). Display only — no PM trading logic, "
               "no edge claims; DIVERGENT = quote gap before spread/fees, check depth "
               "before acting.")


def _gap_sparkline_svg(series: list[tuple[float, float]], width: int = 150,
                       height: int = 26, lim: float = 12.0) -> str:
    """Inline SVG of one bucket's gap-pp path through today's polls: grey zero line,
    dashed ±5pp DIVERGENT guides, the gap polyline, latest point as a dot (red beyond
    the threshold). Gaps are clamped to ±lim pp for the y-scale."""
    if not series:
        return ("<svg width='%d' height='%d'><text x='4' y='17' fill='#5d6b78' "
                "font-size='10'>no polls yet</text></svg>" % (width, height))
    y = lambda g: height / 2 - max(-lim, min(lim, g)) / lim * (height / 2 - 2)
    t0, t1 = series[0][0], series[-1][0]
    span = max(t1 - t0, 1.0)
    x = lambda ts: 2 + (ts - t0) / span * (width - 8)
    thr = BUCKET_DIVERGENT_PP
    pts = " ".join(f"{x(ts):.1f},{y(g):.1f}" for ts, g in series)
    last_ts, last_g = series[-1]
    dot = "#f87171" if bucket_divergent(last_g) else "#38bdf8"
    return (f"<svg width='{width}' height='{height}' style='vertical-align:middle'>"
            f"<line x1='0' y1='{y(0):.1f}' x2='{width}' y2='{y(0):.1f}' "
            f"stroke='#2d3845' stroke-width='1'/>"
            f"<line x1='0' y1='{y(thr):.1f}' x2='{width}' y2='{y(thr):.1f}' "
            f"stroke='#5d6b78' stroke-width='0.6' stroke-dasharray='3,3'/>"
            f"<line x1='0' y1='{y(-thr):.1f}' x2='{width}' y2='{y(-thr):.1f}' "
            f"stroke='#5d6b78' stroke-width='0.6' stroke-dasharray='3,3'/>"
            f"<polyline points='{pts}' fill='none' stroke='#8b9aa8' stroke-width='1.4'/>"
            f"<circle cx='{x(last_ts):.1f}' cy='{y(last_g):.1f}' r='2.6' fill='{dot}'/>"
            f"</svg>")


def _bucket_gap_series(dash, label: str, now: float) -> list[tuple[float, float]]:
    """Today's (CET) gap history for one bucket from the dashboard poll log."""
    if dash is None:
        return []
    midnight = datetime.fromtimestamp(now, _cet()).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp()
    out = []
    for rec in dash.history:
        gaps = rec.get("bucket_gaps") or {}
        if rec["ts"] >= midnight and label in gaps:
            out.append((rec["ts"], gaps[label]))
    return out


def _render_arb_variant(a: dict | None, div_id: str, visible: bool) -> str:
    """One fee-setting's executable-arb readout (PM3). `a` is best_executable_arb()'s
    render-ready dict — imported math, nothing recomputed here."""
    style = "" if visible else " style='display:none'"
    if a is None:
        return (f"<div id='{div_id}'{style}><div class='ph'>arb books unavailable this "
                f"poll — retries next poll</div></div>")
    chip_cls, chip_txt = ("g", "ARB — investable") if a["investable"] else \
        (("a", "lock exists — dust") if a["exists"] else ("n", "no executable lock"))
    L = [f"<div id='{div_id}'{style}>",
         f"<span class='pm3chip {chip_cls}'>{chip_txt}</span>"]
    best = a.get("best")
    if best:
        legs = " · ".join(f"<b>{l['action']}</b> {l['market']} @ {l['price']:.2f}"
                          f" <span class='src'>({l['top_size']:.0f} sh top)</span>"
                          for l in best["legs"])
        L.append(f"<div class='num' style='margin:8px 0'>{legs}</div>")
        L.append(f"<div class='num'>pay <b>${best['pay_per_set']:.3f}/set</b> → locked "
                 f"floor <b>${best['payout_floor']:g}</b>"
                 + (f" <span class='src'>+ free upside: "
                    f"{'; '.join(str(c) for c in best['free_sliver'])}</span>"
                    if best.get("free_sliver") else "") + "</div>")
        L.append(f"<div class='num' style='font-size:17px;margin:8px 0'>EXTRACTABLE: "
                 f"<b>${best['net_usd']:.2f}</b> net over {best['net_sets']:.0f} sets "
                 f"(${best['notional_usd']:.0f} notional) — edge closes by "
                 f"{best['closed_by']}</div>")
    else:
        L.append("<div class='src' style='margin:8px 0'>no taker-only ladder↔bucket lock "
                 "clears its cost at this fee — the PM2 gaps are model/spread artifact, "
                 "not cash (S8)</div>")
    L.append(f"<div class='src'>fee = {a['fee_formula']}</div></div>")
    return "".join(L)


def render_pm_panel(rep: dict, dash, now: float | None = None, arb: dict | None = None,
                    arb_fee_default: float = 0.0) -> str:
    """The PM tab's server-rendered card (terminal numbers, dashboard layout)."""
    now = now if now is not None else time.time()
    a, b = rep["stats_primary"], rep["stats_coworker"]
    L = []
    # --- perp-vs-crowd strip (the +1.5/+3.5 read from the terminal print) ---
    g = rep.get("hl_gap")
    if g:
        L.append(
            f"<div class='pmstrip num'>perp xyz:SPCX <b>${g['hl_mark']:.2f}</b>"
            f" &nbsp;·&nbsp; vs PM mean <b>{g['vs_mean_ps']:+.1f} $/sh</b>"
            f" &nbsp;·&nbsp; vs PM median <b>{g['vs_median_ps']:+.1f} $/sh</b></div>")
    else:
        L.append("<div class='pmstrip num'>perp feed unavailable this poll</div>")
    # --- full distribution table (the terminal table, both share bases) ---
    L.append("<section><h2><span class='pn'>PM1</span><span class='ti'>Crowd "
             "distribution — full stats</span></h2>")
    L.append(f"<div class='cap num'>P(close &gt; ${a['offer']:.0f}) = "
             f"<b>{a['p_win_offer']*100:.1f}%</b> &nbsp;·&nbsp; E[cap] = "
             f"<b>{a['mean_cap_t']:.3f}T</b> &nbsp;·&nbsp; EV vs offer = "
             f"<b>{a['ev_vs_offer_ps']:+.1f} $/sh</b> (A1: mean − offer, 13.076B base)"
             f"</div>")
    L.append("<table><tr><th>stat</th><th class='n'>cap $T</th>"
             "<th class='n'>$/sh 13.076B</th><th class='n'>$/sh 13.091B (Alvaro)</th></tr>")
    for key, label in [("mean", "mean"), ("median", "median"), ("mode", "mode*"),
                       ("p25", "P25"), ("p75", "P75"), ("p90", "P90"), ("p95", "P95")]:
        L.append(f"<tr><td>{label}</td><td class='n'>{a[f'{key}_cap_t']:.3f}</td>"
                 f"<td class='n'>{a[f'{key}_ps']:.1f}</td>"
                 f"<td class='n'>{b[f'{key}_ps']:.1f}</td></tr>")
    L.append("</table>")
    L.append(f"<div class='src'>*mode is shape-fragile under PCHIP "
             f"(spacex_pdf_construction_audit) — printed for continuity, not decisions; "
             f"central stats are construction-robust · strikes used "
             f"{rep['n_strikes_used']}/16 · basis {rep['basis']}</div></section>")
    # --- bucket-vs-ladder with sparklines ---
    L.append("<section><h2><span class='pn'>PM2</span><span class='ti'>Bucket market vs "
             "ladder — consistency screen</span></h2>")
    L.append("<div class='cap'>gap = bucket market richer (+) than the ladder-implied "
             "probability (model-free linear interp of the clipped mids); bucket side "
             "renormalized after dropping the No-IPO leg. Sparkline = today's gap path "
             "(grey zero line, dashed ±5pp guides, dot = now, red = DIVERGENT) — a "
             "divergence whose line just left the zero band is NEW, one that has sat "
             "wide all session is stale.</div>")
    if rep.get("buckets"):
        L.append("<table><tr><th>bucket</th><th class='n'>ladder %</th>"
                 "<th class='n'>market %</th><th class='n'>gap (pp)</th>"
                 "<th>today</th><th></th></tr>")
        for r in rep["buckets"]:
            div = bucket_divergent(r["gap_pp"])
            series = _bucket_gap_series(dash, r["label"], now)
            ago = [gp for ts, gp in series if ts <= now - 2 * 3600]
            ago_s = f"{ago[-1]:+.1f} 2h ago" if ago else ""
            chip = "<span class='divg'>DIVERGENT</span>" if div else ""
            style = " style='background:#1c1416'" if div else ""
            L.append(f"<tr{style}><td>{r['label']}</td>"
                     f"<td class='n'>{r['ladder_implied']*100:.1f}</td>"
                     f"<td class='n'>{r['bucket_normalized']*100:.1f}</td>"
                     f"<td class='n'><b>{r['gap_pp']:+.1f}</b></td>"
                     f"<td>{_gap_sparkline_svg(series)} "
                     f"<span class='src'>{ago_s}</span></td><td>{chip}</td></tr>")
        L.append("</table>")
    else:
        L.append("<div class='ph'>bucket books unavailable this poll</div>")
    no_ipo = rep.get("no_ipo") or {}
    if no_ipo.get("ask") is not None:
        L.append(f"<div class='src num'>No-IPO leg ask: {no_ipo['ask']*100:.1f}c "
                 f"(excluded from the renormalization by convention)</div>")
    L.append("</section>")
    # --- PM3: the cash value of the PM2 gap (Block S5k-PM follow-up; math imported from
    # spcx_pm_arb_check.best_executable_arb — taker-only book walk, never mids/curve).
    # Both fee variants are server-rendered each poll; the page toggles client-side.
    L.append("<section><h2><span class='pn'>PM3</span><span class='ti'>Ladder↔bucket "
             "mismatch — what's executable now (taker-only)</span></h2>")
    L.append("<div class='cap'>This is the cash value of the gap above — the PM2 points "
             "are mostly the interpolated S(2.5T) knot + wide-bucket spread "
             "([[spcx_pm_arb_findings]]); watch this dollar figure, not the pp. Walks the "
             "live L2 books best-ask up until the marginal set goes net-negative under "
             "ONE stated fee; ladder↔bucket families only (covers + exact unions), one "
             "resolution source, never a mid or fitted curve. Read-only — no orders.</div>")
    if arb is None:
        L.append("<div class='ph'>arb books unavailable this poll — retries next poll"
                 "</div></section>")
        return "".join(L)
    def_key = "1000" if float(arb_fee_default) >= 1000 else "0"
    L.append(f"<div id='pm3wrap' data-def='{def_key}'>")
    L.append("<div class='mkrow' style='margin-bottom:8px'>"
             "<button class='pm3fee' id='pm3b0' onclick='pm3Fee(0)'>0 bps — observed "
             "CLOB fills</button>"
             "<button class='pm3fee' id='pm3b1000' onclick='pm3Fee(1000)'>1000 bps — "
             "declared taker_base_fee</button></div>")
    L.append(_render_arb_variant(arb.get("0"), "pm3v0", def_key == "0"))
    L.append(_render_arb_variant(arb.get("1000"), "pm3v1000", def_key == "1000"))
    L.append("</div></section>")
    return "".join(L)


def render_tranches(ts: dict) -> str:
    """Tranche-table HTML for the dashboard (empty when no fill/residual)."""
    if not ts["plan"]["tickets"] and ts["flag"] is None:
        return ""
    head = (f"<div class='trh'>RESIDUAL TRANCHES — {ts['residual']:.0f} sh "
            f"(fill − hedge sleeve) · row {ts['plan']['row'] or '—'} "
            f"<span class='src'>[gameplan §5.2 lean tables, 2026-06-11]</span></div>")
    flag = (f"<div class='trflag s-{ts['shape']}'>{ts['flag']}</div>"
            if ts["flag"] else "")
    if not ts["rows"]:
        return f"<div class='trcard'>{head}{flag}</div>"
    rows = "".join(
        f"<tr class='{'trdone' if r['status'] == 'DONE' else ''}'>"
        f"<td>{r['name']}</td><td class='n'>{r['shares']:.0f} sh</td>"
        f"<td class='n'>{r['window']}</td><td>{r['status']}</td></tr>"
        for r in ts["rows"])
    note = ts["plan"]["note"]
    return (f"<div class='trcard'>{head}{flag}"
            f"<table><tr><th>ticket</th><th class='n'>shares</th><th class='n'>window "
            f"(post-cross)</th><th>status</th></tr>{rows}</table>"
            f"<div class='src' style='margin-top:6px'>{note} · sold so far: "
            f"{ts['sold']:.0f} sh · limit-order-only on TR, 1–2 ticks inside the bid, "
            f"re-pegged on a 5-min timer · display only — no order logic exists</div></div>")


def pair_close_status(history: list[dict], cross_ts: float | None,
                      now: float) -> dict:
    """Gameplan §5.1 'tracking confirmed, books deep' chip: |perp − spot| ≤ $2 sustained
    ≥15 min AND ≥60 min since the cross. Computed from the dashboard poll history."""
    ok_since = None
    for rec in reversed(history):
        p, s = rec.get("perp"), rec.get("spot")
        if p is None or s is None or abs(p - s) > TRACKING_GAP_USD:
            break
        ok_since = rec["ts"]
    gap_mins = (now - ok_since) / 60.0 if ok_since is not None else 0.0
    cross_mins = (now - cross_ts) / 60.0 if cross_ts is not None else 0.0
    green = gap_mins >= TRACKING_MINUTES and cross_mins >= PAIRCLOSE_MIN_MINS
    return {"green": green, "gap_ok_minutes": gap_mins, "mins_since_cross": cross_mins}


def _src(anchor: str) -> str:
    return f"<span class='src'>[{anchor}]</span>"


def render_playbook(pb: PlaybookState, dash: "DashboardState", rep: dict, snap: dict,
                    avwap_info: dict | None, now: float | None = None) -> str:
    """The NOW card. avwap_info = {'avwap': float|None, 'stale': bool} from the S5b feed
    (None = no feed attached)."""
    now = now if now is not None else _epoch(snap["fetched_at_utc"])
    node = infer_node(pb, now)
    hl = snap.get("hl") or {}
    mark = hl.get("mark")
    spot = rep["spot"]["spot"] if rep["spot"] else None
    a = rep["stats_primary"]
    rows: list[str] = []

    def li(txt: str) -> None:
        rows.append(f"<li>{txt}</li>")

    watch = ""
    if node == "PRE-ALLOC":
        li(f"awaiting <code>--fill &lt;shares&gt;</code> — D2 unlocks at allocation, Fri "
           f"~8:00 CET {_src('gameplan §6 D2 — allocation-morning node')}")
        watch = "watch next: TR allocation notification, then feed --fill (and --offer if ≠135)."
    elif node == "D2/D3":
        if mark is None:
            li("perp mark unavailable — hedge rule cannot evaluate this poll")
        else:
            ev = hedge_rule_eval(pb, mark, hl.get("funding_hourly", 0.0), now)
            c1, c15 = ev["cells"][1.0], ev["cells"][1.5]
            if ev["overflow"] <= 0:
                verdict = (f"<b class='go'>fill ≤ comfort zone ({ev['comfort']:.0f} sh) → "
                           f"NO hedge — expected case; margin stays free; all shares run "
                           f"the sell plan</b>")
            elif ev["go"]:
                verdict = (f"<b class='go'>hedge the OVERFLOW: short "
                           f"~{c15.hedged_shares:.0f} sh at 1.5× "
                           f"(~{c1.hedged_shares:.0f} at 1×) at 8:00, not later</b>")
            else:
                verdict = ("<b class='nogo'>overflow exists but net basis ≤ 0 → NO hedge; "
                           "all shares run the sell plan</b>")
            li(f"rule: hedge = clamp(fill − comfort {ev['comfort']:.0f} sh, 0, margin cap) "
               f"iff net basis &gt; 0 → {verdict} "
               f"{_src('gameplan §6 D2 / S1 pre-registered hedge rule (overflow valve, '
               f'updated 2026-06-11)')}")
            li(f"live: gross basis ${ev['gross_basis']:+.2f}/sh, net "
               f"${ev['net_basis_ps']:+.2f}/sh after fees+funding; margin needed "
               f"€{c15.margin_used / ev['eurusd']:.0f} at 1.5× / "
               f"€{c1.margin_used / ev['eurusd']:.0f} at 1× (EURUSD {ev['eurusd']:.3f})")
            residual15 = max(pb.fill - c15.hedged_shares, 0)
            li(f"fill {pb.fill:.0f}/{ev['req']:.0f} sh requested ≈ {ev['frac'] * 100:.0f}% "
               f"→ §3 row: {ev['nearest_fill_row']}% · residual sleeve "
               f"{residual15:.0f} sh at 1.5× "
               f"{_src('gameplan §3 fill-scenario table')}")
            if residual15 > ev["comfort"]:
                li(f"<b class='nogo'>residual ({residual15:.0f} sh) still exceeds the "
                   f"comfort zone after the margin ceiling → front-load the first sell "
                   f"tranche as risk reduction</b> "
                   f"{_src('gameplan §6 D2 high-fill case')}")
        if mark is not None and mark <= PERP_RISKOFF_LEVEL:
            li(f"<b class='nogo'>perp ${mark:.2f} ≤ ${PERP_RISKOFF_LEVEL:.0f} — pop thesis "
               f"weakening; pre-agree risk-off with Alvaro</b> {_src('gameplan §6 D3 pre-open watch rule')}")
        if pb.hedged and pb.hedge_entry and mark is not None:
            from scripts.spcx_convergence_calc import (liq_buffer_summary,
                                                       liq_price_short,
                                                       maintenance_margin_frac)
            liq = liq_price_short(pb.hedge_entry, pb.hedge_lev,
                                  maintenance_margin_frac(5.0))
            buf = liq_buffer_summary(mark, liq)
            li(f"liq buffer ({pb.hedged:.0f} sh short @ ${pb.hedge_entry:.2f}, "
               f"{pb.hedge_lev:g}×): liq ${liq:.2f}, buffer "
               f"{buf['buffer_frac'] * 100:+.1f}% → {buf['band']} "
               f"{_src('liquidation math from spcx_convergence_calc / gameplan §5.1')}")
        bell = datetime.fromtimestamp(now, _cet()).replace(hour=BELL_CET[0],
                                                           minute=BELL_CET[1], second=0)
        mins = (bell.timestamp() - now) / 60.0
        watch = (f"watch next: perp level vs offer (risk-off line ${PERP_RISKOFF_LEVEL:.0f}); "
                 f"bell in {mins:.0f} min — no action before it [gameplan §6 D3 pre-open watch rule].")
    elif node == "D4":
        li(f"display-only window: proxy stack = perp + newswire indications; NOII has no EU "
           f"retail access (S3). <b>No selling into the cross or first prints.</b> "
           f"{_src('gameplan §6 D4 display-only window / S3 data-source map')}")
        if mark is not None:
            li(f"tension: perp ${mark:.2f} vs PM mean ${a['mean_ps']:.1f} "
               f"({mark - a['mean_ps']:+.1f}) — the pre-trade truth pair {_src('gameplan §5.3 day-of screen map')}")
        li("log the cross when it prints: <code>--cross HH:MM PRICE</code> — unlocks D5")
        if pb.hedged:
            li(f"<b>no cross by 20:30 CET → close the perp at market and keep only the "
               f"shares</b> (the 21:00 forced-flat backstop) "
               f"{_src('gameplan §6 D5 forced-flat / S6')}")
        watch = "watch next: first print (likely 17:00–20:00 CET); start the tranche clock at it."
    elif node == "D5":
        mins = (now - pb.cross_ts) / 60.0
        phase, target = tranche_phase(mins)
        residual_total = max((pb.fill or 0) - (pb.hedged or 0), 0)
        sold_pct = (pb.sold / residual_total * 100.0) if residual_total > 0 else None
        sold_txt = (f"sold {pb.sold:.0f}/{residual_total:.0f} sh ({sold_pct:.0f}%) vs "
                    f"target {target:.0f}%" if sold_pct is not None
                    else "no residual sleeve (fully hedged)")
        li(f"tranche clock: <b>{mins:.0f} min since cross</b> → phase <b>{phase}</b> · "
           f"{sold_txt} {_src('gameplan §5.2 tranche plan / S2 §4 mega-IPO tape calibration')}")
        if avwap_info and avwap_info.get("avwap") is not None and not avwap_info.get("stale"):
            av = avwap_info["avwap"]
            if spot is not None:
                state = "ABOVE" if spot >= av else "BELOW"
                lost = ""
                if pb.avwap_lost_since is not None:
                    lost = (f" — <b class='nogo'>lost {(now - pb.avwap_lost_since) / 60:.0f} "
                            f"min ago, not reclaimed → front-load remaining tranches</b>")
                li(f"anchored VWAP ${av:.2f} (IEX ≈2% of tape — confirm on TradingView "
                   f"before acting): spot {state}{lost} {_src('gameplan §5.3 #1 anchored-VWAP rule / S2 §6 front-load finding')}")
            else:
                li(f"anchored VWAP ${av:.2f} (IEX) — no live spot this poll")
        else:
            li("AVWAP: use TradingView (IEX feed absent or stale) "
               + _src("gameplan §5.3 #1 anchored-VWAP rule"))
        if spot is not None:
            cells = []
            for level, label in STOP_LADDER:
                crossed = spot <= level
                dist = (spot - level) / spot * 100.0
                cells.append(f"<b class='{'nogo' if crossed else 'go'}'>${level:.0f} "
                             f"{label}: {'CROSSED' if crossed else f'{dist:+.1f}%'}</b>")
            li("hard stops vs spot: " + " · ".join(cells)
               + f" (display only — execution TR, alerts TradingView) {_src('gameplan §5.4 hard risk rules')}")
        else:
            li("hard-stop ladder: awaiting live spot (S5b feed or --spot)")
        if pb.hedged:
            pc = pair_close_status(dash.history, pb.cross_ts, now)
            chip = ("<b class='go'>tracking confirmed — pair-close window OPEN</b>"
                    if pc["green"] else
                    f"not yet (gap ≤ ${TRACKING_GAP_USD:.0f} for {pc['gap_ok_minutes']:.0f}"
                    f"/{TRACKING_MINUTES:.0f} min, cross +{pc['mins_since_cross']:.0f}"
                    f"/{PAIRCLOSE_MIN_MINS:.0f} min)")
            li(f"hedge sleeve pair-close: {chip} · leg order: TR limit sell first, then "
               f"close the perp · thresholds S6-calibrated on the Cerebras tape (gap ≤ $2 "
               f"at +46 min) {_src('gameplan §5.1 pair-close mechanics / S6')}")
            ff = datetime.fromtimestamp(now, _cet()).replace(
                hour=FORCED_FLAT_CET[0], minute=FORCED_FLAT_CET[1], second=0)
            mins_ff = (ff.timestamp() - now) / 60.0
            li(f"<b>forced-flat backstop: perp must be 0 by 21:00 CET</b> "
               f"({mins_ff:+.0f} min away) — a dislocated close beats a naked short; "
               f"backstop cost ≈ $1.7/sh on the Cerebras tape "
               f"{_src('gameplan §6 D5 forced-flat / S6')}")
        li("PEAK = volume divergence + any of: AVWAP loss / first lower low / perp "
           "divergence / PM-tail fade. Live here: AVWAP (above) and PM tails (Chart D); "
           "☐ volume divergence and ☐ lower low are manual TradingView checks "
           + _src("gameplan §5.3 PEAK signal definition"))
        if dash.history:
            tail_hi = {}
            for k in (2.4, 3.0):
                bids = [(r["tails"].get(k) or (None, None))[0] for r in dash.history]
                bids = [b for b in bids if b is not None]
                if bids:
                    tail_hi[k] = (bids[-1], max(bids))
            if tail_hi:
                parts = []
                for k, (cur, hi) in tail_hi.items():
                    fading = " (fading)" if cur < hi else ""
                    parts.append(f"&gt;${k}T bid {cur * 100:.0f}c vs session high "
                                 f"{hi * 100:.0f}c{fading}")
                li("PM tails: " + " · ".join(parts) + " " + _src("gameplan §5.3 #6 PM-tail repricing signal"))
        watch = ("watch next: PEAK trigger per the box above; last 30 min (21:30 CET) = "
                 "no new decisions [gameplan §6 D5 post-cross node].")
    else:  # CLOSE-OUT
        residual_left = max((pb.fill or 0) - (pb.hedged or 0) - pb.sold, 0)
        li(f"<b>don't carry residual overnight</b> — 5/6 mega-IPO day-2 closes were lower; "
           f"remaining residual {residual_left:.0f} sh · end state to verify before 22:00: "
           f"<b>perp 0, shares 0</b> {_src('S2 §5 day-2 follow-through (5/6 closed lower) / S2 §6')}")
        if pb.hedged:
            li("hedge sleeve should already be flat (pair-close per §5.1); if vntl was used "
               f"anywhere, its close-leg is a hand-timed limit at the close — no MOC on TR "
               f"{_src('gameplan §5.1 vntl close-leg / S4: Trade Republic has no market-on-close')}")
        watch = "watch next: post-day — feed fills + parquet log into the postmortem note."

    missing = []
    if pb.fill is None:
        missing.append("--fill")
    if node in ("D4",) and pb.cross_ts is None:
        missing.append("--cross (at the print)")
    miss_txt = (f"<div class='pbmiss'>awaiting: {', '.join(missing)}</div>" if missing else "")
    return (f"<div class='pbcard' data-node='{node}'><div class='pbnode'>NOW: <b>{node}</b>"
            f"<span class='pbsub'> — checklist renderer; rules quoted from the gameplan, "
            f"no orders, no new signals</span></div>"
            f"<ul class='pbul'>{''.join(rows)}</ul>"
            f"<div class='pbwatch'>{watch}</div>{miss_txt}</div>")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def run_reconcile() -> str:
    """The Block-S5(d) sweep: which EV convention reproduces the coworker PNG's
    'IPO at 135: EV $-3.3/share' on his own 2026-06-07 surface + conventions?"""
    snap = fixture_snapshot()
    strikes, probs, _ = extract_points(snap["ladder"], "mid")
    fit = fit_survivor(strikes, probs)
    sweep = sorted(ev_convention_sweep(fit), key=lambda d: d["abs_err_vs_png"])
    st = dist_stats(fit, shares=SHARES_COWORKER)
    L = ["EV-convention reconciliation vs the coworker PNG 'EV: $-3.3/share'",
         f"(his construction: PCHIP on 2026-06-07 mids, 13.091B shares; this fit: "
         f"P(win)={st['p_win_offer']*100:.1f}%, mean=${st['mean_ps']:.1f}, "
         f"median=${st['median_ps']:.1f})", "",
         f"{'convention':<55s} {'EV $/sh':>9s} {'|err vs -3.3|':>13s}"]
    for r in sweep:
        L.append(f"{r['convention']:<55s} {r['ev_ps']:+9.2f} {r['abs_err_vs_png']:13.2f}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SPCX live PM-PDF + cross-venue monitor (read-only)")
    ap.add_argument("--watch", type=int, metavar="SEC",
                    help="poll every SEC seconds (30-60 recommended; clamped 15-300)")
    ap.add_argument("--basis", choices=["mid", "ask", "bid"], default="mid",
                    help="survivor price basis (default mid of executable bid/ask)")
    ap.add_argument("--offer", type=float, default=IPO_OFFER_DEFAULT)
    ap.add_argument("--spot", type=float,
                    help="listed price once SPCX trades (post-cross); overrides --spot-ws")
    ap.add_argument("--spot-ws", choices=["alpaca"],
                    help="auto spot feed: Alpaca free IEX trade stream "
                         "(needs ALPACA_KEY_ID/ALPACA_SECRET_KEY env vars)")
    ap.add_argument("--spot-symbol", default="SPCX",
                    help="symbol for --spot-ws (default SPCX; use a liquid one to smoke-test)")
    ap.add_argument("--spot-stale-secs", type=float, default=SPOT_STALE_SECS_DEFAULT,
                    help="trades older than this render STALE (default 120)")
    ap.add_argument("--html", type=Path,
                    help="write ONE static auto-refresh dashboard html file here")
    ap.add_argument("--serve", type=int, nargs="?", const=8642, metavar="PORT",
                    help="serve the interactive localhost dashboard (S5e) on 127.0.0.1 "
                         "(default port 8642); terminal + --html keep working alongside")
    ap.add_argument("--reference", metavar="WHAT",
                    help="Chart-A overlay: 'fixture' (2026-06-07 addendum surface) or a "
                         "saved snapshot JSON path; default = the session-start poll")
    ap.add_argument("--mark", action="append", metavar="LABEL[@HH:MM]",
                    help="drop a vertical event marker in the time-series chart "
                         "(repeatable; CET time optional, default now)")
    # Block S5d — playbook day-state (persisted; the panel renders on the --html page)
    ap.add_argument("--fill", type=float, help="shares allocated (Fri ~8:00 — unlocks D2)")
    ap.add_argument("--hedged", type=float, help="shares hedged (short on)")
    ap.add_argument("--hedge-entry", type=float, help="perp entry price of the short")
    ap.add_argument("--hedge-lev", type=float, help="leverage of the short (default 1.0)")
    ap.add_argument("--sold", type=float,
                    help="cumulative residual-sleeve shares sold (update after each tranche)")
    ap.add_argument("--cross", nargs=2, metavar=("HH:MM|now", "PRICE"),
                    help="log the IPO cross (CET time + first print) — unlocks D5")
    ap.add_argument("--sub-eur", type=float, help="subscription size in EUR (default 10000)")
    ap.add_argument("--margin-eur", type=float, help="free HL margin in EUR (default 2000)")
    ap.add_argument("--eurusd", type=float, help="EURUSD for the hedge math (else live fetch)")
    ap.add_argument("--comfort", type=float,
                    help="comfort-zone shares held long-only without hedging (default 22; "
                         "the D2 hedge covers only the overflow above this)")
    ap.add_argument("--playbook-state", type=Path, default=PLAYBOOK_STATE_FILE,
                    help="JSON day-state file (persists --fill/--cross/… across restarts)")
    ap.add_argument("--parquet-log", action="store_true",
                    help=f"append one parquet shard per poll under {PARQUET_LOG_DIR}")
    ap.add_argument("--backfill-days", type=int, default=7, metavar="N",
                    help="on startup, replay the last N UTC days of parquet shards into "
                         "the dashboard session (default 7 = match the time-scrub window). "
                         "The session/pair/tail charts then span the full week; the chart "
                         "zoom slider sets the visible lookback. Use 1 for today-only.")
    ap.add_argument("--curve-days", type=float, default=7.0, metavar="N",
                    help="(with --serve) index the last N days of logged polls for the "
                         "PDF/survivor time-scrub slider (default 7)")
    ap.add_argument("--fee-bps", type=float, default=0.0,
                    help="(with --serve) initial PM3 taker-fee setting: 0 = observed CLOB "
                         "fills, 1000 = declared taker_base_fee; the page toggles between "
                         "both live (default 0)")
    ap.add_argument("--from-json", type=Path, help="offline: analyze a saved snapshot JSON")
    ap.add_argument("--save-json", type=Path, help="save each snapshot to this JSON path")
    ap.add_argument("--fixture", action="store_true",
                    help="offline: analyze the embedded 2026-06-07 coworker surface")
    ap.add_argument("--reconcile", action="store_true",
                    help="run the -3.3 EV convention sweep on the 06-07 fixture and exit")
    args = ap.parse_args(argv)

    if args.reconcile:
        print(run_reconcile())
        return 0

    meta = None
    if not (args.from_json or args.fixture):
        meta = fetch_pm_metadata()
        print(f"[meta] ladder strikes: {len(meta['ladder'])}  buckets: {len(meta['buckets'])}"
              f"  no_ipo: {'yes' if meta.get('no_ipo') else 'no'}", file=sys.stderr)

    feed = None
    if args.spot_ws == "alpaca":
        if args.spot is not None:
            print("[spot] manual --spot given — it overrides --spot-ws", file=sys.stderr)
        try:
            feed = AlpacaSpotFeed(args.spot_symbol)
        except RuntimeError as e:
            print(f"[spot] {e}", file=sys.stderr)
            return 2
        feed.start()
        print(f"[spot] alpaca IEX feed started for {args.spot_symbol} "
              f"(stale after {args.spot_stale_secs:.0f}s)", file=sys.stderr)

    pb = None
    if args.html or args.serve is not None:
        pb = PlaybookState(args.playbook_state)
        pb.offer = args.offer
        for flag, attr in [("fill", "fill"), ("hedged", "hedged"),
                           ("hedge_entry", "hedge_entry"), ("hedge_lev", "hedge_lev"),
                           ("sold", "sold"), ("sub_eur", "sub_eur"),
                           ("margin_eur", "margin_eur"), ("eurusd", "eurusd"),
                           ("comfort", "comfort")]:
            v = getattr(args, flag)
            if v is not None:
                setattr(pb, attr, v)
        if pb.hedged and pb.hedge_ts is None:   # stamp when the short first goes on
            pb.hedge_ts = time.time()
        if args.cross:
            pb.cross_ts, pb.cross_price = parse_cross_arg(args.cross, time.time())
        if pb.eurusd is None:
            try:
                from scripts.spcx_convergence_calc import fetch_eurusd
                pb.eurusd = fetch_eurusd()[0]
            except Exception:
                pass  # hedge_rule_eval falls back to 1.08, labeled
        pb.save()

    dash = None
    if args.html or args.serve is not None:
        # retention must cover the backfill window, else the older shards load then get
        # trimmed by the 12h cap; today-only keeps the clean 12h listing-day default.
        hist_hours = max(HISTORY_HOURS_DEFAULT, args.backfill_days * 24.0 + 6.0)
        dash = DashboardState(marks=args.mark, offer=args.offer, history_hours=hist_hours)
        n_back = dash.backfill_from_parquet(days=args.backfill_days)
        if n_back:
            print(f"[dash] backfilled {n_back} polls from the last "
                  f"{max(1, args.backfill_days)} day(s) of parquet shards",
                  file=sys.stderr)
        if args.reference == "fixture":
            dash.set_reference(fixture_snapshot(), "2026-06-07 addendum fixture")
        elif args.reference:
            dash.set_reference(json.loads(Path(args.reference).read_text()),
                               Path(args.reference).name)

    server = None
    curve_idx = None
    shape_cls = DayShapeClassifier()   # S7 day-shape (steps only once spot prints exist)
    halts = None
    if args.serve is not None:
        halts = NasdaqHaltsPoller("SPCX")   # cross-timing; self-gates to ≥15:00 CET
        halts.start()
        from scripts.spcx_dashboard_server import DashboardServer
        curve_idx = CurveIndex(days=args.curve_days)
        n_curves = curve_idx.load_parquet()
        print(f"[curves] indexed {n_curves} logged polls for the time-scrub slider "
              f"(window {args.curve_days:g} days)", file=sys.stderr)
        server = DashboardServer(dash, pb, port=args.serve, curves=curve_idx)
        server.start()
        print(f"[serve] interactive dashboard: {server.url()}  (127.0.0.1 only, no auth "
              f"— localhost display; terminal + --html unaffected)", file=sys.stderr)

    # PM3 executable-arb metadata (Block S5k-PM follow-up): one-time fetch of both
    # tokens per leg + resolution keys; the per-poll book walk imports from
    # spcx_pm_arb_check (single source of math). Failure → PM3 degrades, never fatal.
    arb_meta = None
    if args.serve is not None:
        try:
            from scripts.spcx_pm_arb_check import fetch_arb_metadata
            arb_meta = fetch_arb_metadata()
            print(f"[arb] PM3 armed: {len(arb_meta['ladder'])} ladder + "
                  f"{len(arb_meta['buckets'])} bucket legs, both tokens each "
                  f"(fee default {args.fee_bps:g} bps)", file=sys.stderr)
        except Exception as e:
            print(f"[arb] metadata fetch failed ({type(e).__name__}: {e}) — "
                  f"PM3 renders unavailable", file=sys.stderr)

    interval = max(15, min(300, args.watch)) if args.watch else None
    while True:
        if args.fixture:
            snap = fixture_snapshot()
        elif args.from_json:
            snap = json.loads(args.from_json.read_text())
        else:
            snap = build_snapshot(meta)
        if args.save_json:
            args.save_json.write_text(json.dumps(snap, indent=2))
        spot_val, spot_meta = resolve_spot(args.spot, feed, args.spot_stale_secs)
        try:
            rep = analyze(snap, basis=args.basis, offer=args.offer,
                          spot=spot_val, spot_meta=spot_meta)
        except RuntimeError as e:
            print(f"[poll skipped] {e}", file=sys.stderr)
            if interval is None:
                return 1
            time.sleep(interval)
            continue
        print(render_text(rep), flush=True)  # line-buffer even when redirected to a log
        if dash is not None:
            dash.record(snap, rep)
            if curve_idx is not None:   # extend the scrub range with this poll
                curve_idx.append(snap, rep)
            avwap_info = None
            if feed is not None:
                av = feed.avwap()
                stale = (spot_meta or {}).get("status") == "stale"
                avwap_info = {"avwap": av["avwap"] if av else None, "stale": stale}
                if av and rep["spot"]:
                    pb.update_avwap_state(rep["spot"]["spot"], av["avwap"],
                                          _epoch(snap["fetched_at_utc"]))
            st = shape_cls.step(_epoch(snap["fetched_at_utc"]),
                                rep["spot"]["spot"] if rep["spot"] else None,
                                (avwap_info or {}).get("avwap"), pb.cross_price)
            # stamp shape + avwap onto this poll's record: the EXEC tab's live state
            # ribbon and AVWAP line read them from history (backfilled records lack
            # them — the ribbon starts when live polling starts, which is honest)
            if dash.history:
                dash.history[-1]["shape"] = st if shape_cls.armed else None
                dash.history[-1]["avwap"] = (avwap_info or {}).get("avwap")
            pb_html = render_playbook(pb, dash, rep, snap, avwap_info)
            if args.html:
                args.html.parent.mkdir(parents=True, exist_ok=True)
                write_html_atomic(args.html,
                                  render_dashboard(dash, rep, snap,
                                                   refresh_s=interval or 60,
                                                   playbook_html=pb_html))
            if server is not None:
                from scripts.spcx_dashboard_server import build_ws_payload
                arb = None
                if arb_meta is not None:
                    try:  # PM3: full-L2 snapshot + both fee variants (pure local math)
                        from scripts.spcx_pm_arb_check import (
                            best_executable_arb, build_snapshot as arb_snapshot)
                        s_arb = arb_snapshot(arb_meta, timeout=15.0)
                        arb = {"0": best_executable_arb(arb_meta, s_arb, 0.0),
                               "1000": best_executable_arb(arb_meta, s_arb, 1000.0)}
                    except Exception as e:  # degrade the card, never the poll
                        print(f"[arb] poll fetch failed: {type(e).__name__}: {e}",
                              file=sys.stderr)
                try:
                    server.last_ctx = (rep, snap, avwap_info)
                    server.push(build_ws_payload(dash, pb, rep, snap, avwap_info, pb_html,
                                                 curves=curve_idx, classifier=shape_cls,
                                                 halts=halts, arb=arb,
                                                 arb_fee_default=args.fee_bps))
                except Exception as e:  # a display-layer failure never stops the engine
                    print(f"[serve] push failed: {type(e).__name__}: {e}", file=sys.stderr)
        if args.parquet_log:
            last_ind = (pb.indications or [None])[-1]
            extra = ({"indication_text": last_ind["text"],
                      "indication_ts": last_ind["ts"]} if last_ind else None)
            p = log_parquet(snap, rep, extra=extra)
            print(f"[parquet] {p}", file=sys.stderr)
        if interval is None or args.from_json or args.fixture:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
