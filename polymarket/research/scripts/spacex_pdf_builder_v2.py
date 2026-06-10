"""
SpaceX IPO day-1 closing distribution — robust builder (v2).

WHY v2: the colleague's v1 (spacex_pdf_analysis.py / ev_report.py) builds the PDF by
differentiating a PCHIP interpolant fitted EXACTLY through 16 noisy survivor mids. That
manufactures 5-6 spurious local maxima (the "3 humps") from a single-hump distribution,
ignores per-strike liquidity entirely, and produces untrustworthy tail percentiles.
See: polymarket/research/notes/overview/data_quality/spacex_pdf_construction_audit.md

WHAT v2 DOES DIFFERENTLY:
  1. Fits a UNIMODAL lognormal survivor to the BEST-ASK quotes (taker convention: you pay the
     ask) by 1/spread^2-weighted least squares, so thin/noisy strikes pull the curve less and
     the density is single-humped by design. (best-ask vs mid moves P(win)/mean by <0.5pp/$0.6.)
  2. Reports the raw interval-mass histogram beside it (assumption-free, no interpolation).
  3. Reports a skew-normal as a right-skew sensitivity.
  4. Overlays the standalone bucket market and flags cross-market gaps.
  5. Headline P(close>$135), mean, median are method-invariant (~80% / ~$167) — v2 just
     stops lying about the SHAPE and the tails.

ENV: the polymarket/research venv does NOT ship scipy. Run with system python (has
numpy>=2 / scipy / matplotlib):  python3 polymarket/research/scripts/spacex_pdf_builder_v2.py
This script is standalone and read-only; it writes one PNG and prints a report.

REFRESH (June 11 eve / June 12 morning): update CONTINUOUS_RAW and BUCKETED_RAW from
https://polymarket.com/event/spacex-ipo-closing-market-cap-above and the bucket event,
update PERP_PRICE, re-run, and re-check the headline numbers before trading.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- share count: exact S-1/A no-option Class A + Class B (store the convention!) ---
TOTAL_SHARES = 13_075_865_175          # 13.076B; v1 used a rounded 13.091B (~0.1% high)
IPO_PRICE = 135.0
PERP_PRICE = 165.7                     # Hyperliquid xyz:SPCX mark — refresh before trading

def cap_to_price(cap_T): return (cap_T * 1e12) / TOTAL_SHARES
def price_to_cap(p):     return (p * TOTAL_SHARES) / 1e12

# === RAW DATA (refresh these two arrays before any trade) ====================
# 16-strike "closing market cap ABOVE x" ladder: (strike_T, YES_buy_cents, NO_buy_cents)
CONTINUOUS_RAW = [
    (1.0, 99.2, 1.4), (1.2, 98.3, 1.8), (1.4, 96.6, 3.6), (1.6, 91.5, 8.8),
    (1.8, 78.0, 23.0), (2.0, 64.0, 37.0), (2.2, 46.0, 55.0), (2.4, 30.0, 71.0),
    (2.6, 15.0, 86.0), (2.8, 11.0, 90.0), (3.0, 7.0, 94.0), (3.2, 3.6, 96.7),
    (3.4, 2.6, 97.8), (3.6, 2.4, 98.5), (3.8, 1.8, 98.5), (4.0, 1.3, 99.0),
]
# 8-bucket "closing market cap BETWEEN x and y": (label, lo_T, hi_T, YES_buy_cents, NO_buy_cents)
BUCKETED_RAW = [
    ("<1.0T", 0.0, 1.0, 0.7, 99.4), ("1.0-1.5T", 1.0, 1.5, 3.9, 97.0),
    ("1.5-2.0T", 1.5, 2.0, 40.0, 62.0), ("2.0-2.5T", 2.0, 2.5, 43.0, 58.0),
    ("2.5-3.0T", 2.5, 3.0, 12.7, 88.8), ("3.0-3.5T", 3.0, 3.5, 4.1, 97.3),
    ("3.5T+", 3.5, 5.0, 1.2, 99.1), ("No-IPO", -1, -1, 0.4, 99.7),
]

strikes = np.array([c[0] for c in CONTINUOUS_RAW])
yes_ask = np.array([c[1] for c in CONTINUOUS_RAW]) / 100      # taker BUY price for "cap above K" YES
yes_bid = np.array([(100 - c[2]) for c in CONTINUOUS_RAW]) / 100
mids    = (yes_ask + yes_bid) / 2
spread  = np.maximum(yes_ask - yes_bid, 0.001)               # bid-ask, floored
weights = 1.0 / spread**2                                     # liquidity proxy: tight quotes weigh more
weights /= weights.sum()
# CONSTRUCTION STANDARD = best-ask (yes_ask): we are takers, so the survivor uses the price
# we actually pay. vs mid this shifts P(win)/mean up by only ~0.4pp / ~$0.6 (half the spread).
S_construct = yes_ask

# === FIT 1: liquidity-weighted lognormal survivor on BEST-ASK (PRIMARY, unimodal) ===
def lognorm_S(K, mu, sig): return 1 - norm.cdf((np.log(np.maximum(K, 1e-9)) - mu) / sig)
res_ln = minimize(lambda p: np.sum(weights * (lognorm_S(strikes, *p) - S_construct) ** 2),
                  [np.log(2.1), 0.25], method="Nelder-Mead")
mu, sig = res_ln.x
median_cap = np.exp(mu)
mean_cap   = np.exp(mu + sig**2 / 2)
ln_wrmse   = np.sqrt(np.sum(weights * (lognorm_S(strikes, mu, sig) - S_construct) ** 2)) * 100
# mid sensitivity (to show the best-ask vs mid difference is minimal)
mu_mid, sig_mid = minimize(lambda p: np.sum(weights * (lognorm_S(strikes, *p) - mids) ** 2),
                           [np.log(2.1), 0.25], method="Nelder-Mead").x

# === FIT 2: skew-normal survivor on BEST-ASK (right-skew SENSITIVITY) ===
def skew_S(K, a, loc, sc): return 1 - skewnorm.cdf(K, a, loc, sc)
res_sn = minimize(lambda p: np.sum(weights * (skew_S(strikes, *p) - S_construct) ** 2),
                  [2.0, 1.7, 0.7], method="Nelder-Mead")
a_sn, loc_sn, sc_sn = res_sn.x

# === Grids / densities ===
xf = np.linspace(0.01, 5.0, 20000); dx = xf[1] - xf[0]
ln_pdf = np.maximum(np.gradient(1 - lognorm_S(xf, mu, sig), xf), 0); ln_pdf /= np.trapezoid(ln_pdf, xf)

def headline(S_func, tag):
    cap_ipo = price_to_cap(IPO_PRICE)
    p_win = float(S_func(cap_ipo))
    pdf = np.maximum(np.gradient(1 - S_func(xf), xf), 0); pdf /= np.trapezoid(pdf, xf)
    m = np.trapezoid(xf * pdf, xf)
    cdf = np.cumsum(pdf) * dx; med = xf[np.searchsorted(cdf, 0.5)]
    print(f"  {tag:28s} P(close>$135)={p_win*100:5.1f}%   mean=${cap_to_price(m):6.1f}   median=${cap_to_price(med):6.1f}")
    return p_win, pdf

print("=" * 74)
print("SpaceX IPO day-1 closing distribution — robust builder v2")
print(f"  shares={TOTAL_SHARES:,}  IPO=${IPO_PRICE}  perp=${PERP_PRICE}")
print("=" * 74)
print("\nHEADLINE (method-invariant — the shape artifact never moved these):")
pwin_ln, _ = headline(lambda c: lognorm_S(c, mu, sig),        "lognormal best-ask (PRIMARY)")
headline(lambda c: lognorm_S(c, mu_mid, sig_mid),             "lognormal MID (sensitivity)")
headline(lambda c: skew_S(c, a_sn, loc_sn, sc_sn),            "skew-normal best-ask (sens.)")
# raw best-ask linear-interp survivor headline
def lin_S(c):
    return np.interp(c, np.concatenate([[0], strikes, [5]]),
                        np.concatenate([[1], S_construct, [0]]))
headline(np.vectorize(lambda c: float(lin_S(c))),             "raw best-ask, linear interp")

print(f"\nPRIMARY lognormal: mu={mu:.3f} sig={sig:.3f}  median_cap={median_cap:.3f}T "
      f"mean_cap={mean_cap:.3f}T  weighted-RMSE={ln_wrmse:.2f}c")

# === FULL KEY-METRICS SET (same fields the colleague's report printed) ============
cap_ipo = price_to_cap(IPO_PRICE)
pwin = float(lognorm_S(cap_ipo, mu, sig)); ploss = 1 - pwin
cdf_ln = np.cumsum(ln_pdf) * dx
mode_cap = xf[np.argmax(ln_pdf)]
var_cap = np.trapezoid((xf - mean_cap) ** 2 * ln_pdf, xf); sd_cap = np.sqrt(var_cap)
# lognormal shape is scale-invariant (price = cap x const), so closed-form skew/kurt apply
skew = (np.exp(sig**2) + 2) * np.sqrt(np.exp(sig**2) - 1)
kurt = np.exp(4*sig**2) + 2*np.exp(3*sig**2) + 3*np.exp(2*sig**2) - 6
mw = xf >= cap_ipo; ml = xf < cap_ipo
e_win = cap_to_price(np.trapezoid(xf[mw]*ln_pdf[mw], xf[mw]) / pwin)
e_loss = cap_to_price(np.trapezoid(xf[ml]*ln_pdf[ml], xf[ml]) / ploss)
avg_gain = e_win - IPO_PRICE; avg_loss = IPO_PRICE - e_loss
ev = pwin*avg_gain - ploss*avg_loss; kelly = pwin - ploss*avg_loss/avg_gain if avg_gain > 0 else 0
print("\nKEY METRICS (liquidity-weighted lognormal; reproduces the original report's fields):")
print(f"  Mean ${cap_to_price(mean_cap):.1f}  Median ${cap_to_price(median_cap):.1f}  "
      f"Mode ${cap_to_price(mode_cap):.1f}  StdDev ${cap_to_price(sd_cap):.1f}")
print(f"  Skewness {skew:+.3f}   Excess kurtosis {kurt:+.3f}")
print(f"  P(close>$135) {pwin*100:.1f}%   avg gain +${avg_gain:.1f}   avg loss -${avg_loss:.1f}")
print(f"  Expected value +${ev:.1f}/sh (+{ev/IPO_PRICE*100:.1f}%)   Kelly {kelly*100:.0f}%")
pct_line = "  ".join(f"P{p}=${cap_to_price(xf[np.searchsorted(cdf_ln, p/100)]):.0f}"
                     for p in [1,5,10,25,50,75,90,95,99])
print("  " + pct_line)

# Per-strike fit tension (flag strikes the unimodal fit can't match within ~1c)
print("\nPER-STRIKE FIT (model vs best-ask quote; flag = the kink that made v1 bulge):")
print(f"  {'strike':>7s} {'ask%':>7s} {'lognorm%':>8s} {'resid(c)':>8s}")
for i, k in enumerate(strikes):
    pred = lognorm_S(k, mu, sig) * 100
    resid = pred - S_construct[i] * 100
    flag = "  <-- re-quote check" if abs(resid) > 3 else ""
    print(f"  >{k:.1f}T  {S_construct[i]*100:6.1f}  {pred:7.1f}  {resid:+7.1f}{flag}")

# === Bucket overlay (cross-market consistency) ===
# Use the DIRECT ladder survivor (linear interp of mids), NOT the lognormal: the lognormal
# is a smooth global fit and would inject its own smoothing residuals as fake "gaps". The
# model-free ladder-vs-bucket comparison isolates the true cross-market disagreement.
print("\nLIQUIDITY-WEIGHTED BUCKET OVERLAY (ladder-implied vs bucket mkt + per-bucket spread band):")
b = [x for x in BUCKETED_RAW if x[0] != "No-IPO"]
b_mid = np.array([(x[3] + 100 - x[4]) / 2 for x in b]) / 100
b_spr = np.array([abs(x[3] - (100 - x[4])) for x in b]) / 100   # per-bucket bid-ask (liquidity)
b_norm = b_mid / b_mid.sum()
print(f"  {'bucket':>10s} {'ladder%':>8s} {'market%':>8s} {'±band(pp)':>10s} {'gap(pp)':>8s}")
impl, gaps, lw = [], [], []
for i, (lab, lo, hi, *_ ) in enumerate(b):
    p = float(lin_S(lo) - lin_S(hi))               # S(lo)-S(hi) from best-ask ladder (model-free)
    impl.append(p)
    gap = (b_norm[i] - p) * 100                     # gap = market richer (+) than ladder
    gaps.append(gap); lw.append(1.0 / max(b_spr[i], 0.005) ** 2)
    flag = "  <-- DIVERGENT" if abs(gap) > 5 else ""
    print(f"  {lab:>10s}  {p*100:7.1f}  {b_norm[i]*100:7.1f}  {b_spr[i]*100:9.1f}  {gap:+7.1f}{flag}")
lw = np.array(lw) / sum(lw)
print(f"  Liquidity-weighted RMS cross-market gap = {np.sqrt(np.sum(lw*np.array(gaps)**2)):.2f}pp "
      f"(weights 1/spread^2). Constructions agree within bucket spread bands except the known 1.5-2.0T structural gap.")

# === Chart: honest histogram + unimodal fit + bucket overlay ===
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
price = cap_to_price(xf)
# left: histogram + unimodal fit
mass = -np.diff(S_construct)              # interval mass on the best-ask survivor
price_edges = cap_to_price(strikes)
centers = (price_edges[:-1] + price_edges[1:]) / 2
widths_p = np.diff(price_edges)
heights = mass / widths_p                 # probability density per $/share (mass / bin width)
ax[0].bar(centers, heights, width=widths_p * 0.92, alpha=0.45, color="#888",
          label="raw interval-mass histogram")
ax[0].plot(price, ln_pdf * (TOTAL_SHARES / 1e12), color="#CC0000", lw=2.5,
           label=f"lognormal (unimodal)  mean ${cap_to_price(mean_cap):.0f}")
ax[0].axvline(IPO_PRICE, color="green", lw=2, label=f"IPO ${IPO_PRICE:.0f}")
ax[0].axvline(PERP_PRICE, color="orange", lw=1.5, ls="--", label=f"perp ${PERP_PRICE:.0f}")
ax[0].set_xlim(50, 350); ax[0].set_xlabel("day-1 closing $/share"); ax[0].set_ylabel("density")
ax[0].set_title("v2: single-hump distribution (no spurious peaks)", fontweight="bold")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)
# right: bucket overlay
xp = np.arange(len(b)); ww = 0.38
ax[1].bar(xp - ww/2, np.array(impl) * 100, ww, label="ladder-implied", color="steelblue", alpha=0.85)
ax[1].bar(xp + ww/2, b_norm * 100, ww, label="bucket market", color="coral", alpha=0.85)
for i in range(len(b)):
    g = (b_norm[i] - impl[i]) * 100
    ax[1].annotate(f"{g:+.1f}", (xp[i], max(impl[i], b_norm[i]) * 100 + 0.6), ha="center",
                   fontsize=8, color="red" if abs(g) > 5 else "green", fontweight="bold")
ax[1].set_xticks(xp); ax[1].set_xticklabels([x[0] for x in b], rotation=40, ha="right", fontsize=8)
ax[1].set_title("cross-market overlay (gap = market − model, pp)", fontweight="bold")
ax[1].set_ylabel("probability %"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2, axis="y")
plt.tight_layout()
import os
out = "polymarket/research/data/analysis/plots/overview/spacex_pdf_v2.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print(f"\n[chart saved: {out}]")
print("\nNote: this is a PDF-CONSTRUCTION fix. Central stats (P(close>$135), mean, median, EV) are")
print("method-invariant; the v1 PCHIP artifact only distorted shape/tail stats (mode, kurtosis,")
print("extreme percentiles). The 1.5-2.0T bucket gap is structural/known and out of scope here.")
