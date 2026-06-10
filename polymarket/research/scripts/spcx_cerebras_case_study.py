"""Cerebras (CBRS) pre-IPO -> listing case study, as a real-data analog for the SPCX trade.

WHY: a user asked how Cerebras actually moved through its IPO/listing -- IPO price, open vs perp,
and the intraday path on the first trading day -- to learn what could happen to a SpaceX short.

DATA: Hyperliquid `xyz:CBRS` candles, live-fetched from the public `/info` candleSnapshot endpoint
(read-only). The CBRS pre-IPO perp converted to a listed-equity perp ~2026-05-14; its post-conversion
price tracks the listed stock, so a single candle series spans the pre-IPO run-up, the listing-day
spike, and the subsequent settle. Scenario IPO anchors (offer/short-entry) are LABELLED as such --
the authoritative series here is the HL-measured perp path.

OUTPUT: three PNGs under data/analysis/plots/spcx_convergence/ + a candle CSV.

RUN: cd polymarket/research && PYTHONPATH=. uv run python scripts/spcx_cerebras_case_study.py
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False  # render literal "$" (no mathtext)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
PLOTS = Path(__file__).resolve().parents[1] / "data" / "analysis" / "plots" / "spcx_convergence"
CSV_OUT = Path(__file__).resolve().parents[1] / "data" / "analysis" / "csv_outputs" / "market_maps"

# --- Cerebras cash-equity IPO anchors (corroborated by 2026-dated reporting: cerebras.ai pricing
#     release, CNBC, Yahoo Finance, IPOScoop). Priced 2026-05-13, debuted 2026-05-14 on Nasdaq:CBRS. ---
CBRS_IPO_OFFER = 185.0       # IPO offer price ($/share); priced above the $150-160 range, ~$5.55B raised
CBRS_FIRST_OPEN = 350.0      # first-trade open (~+89% vs offer)
CBRS_FIRST_HIGH_CASH = 385.0  # cash-equity intraday high (~$385-386); the perp printed ~$392
CBRS_FIRST_CLOSE = 311.07    # first-day close (+~68% vs offer); fell ~10% the next day
CBRS_SHORT_ENTRY = 277.0     # illustrative pre-listing PERP short entry (the spcx calc regression fixture)
CBRS_MMR = 1.0 / (2.0 * 5.0)  # pre-conversion max-lev was 5x -> maintenance fraction 0.10
LISTING = datetime(2026, 5, 14, tzinfo=timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_candles(coin: str, interval: str, start: datetime, end: datetime) -> list[dict]:
    req = {"type": "candleSnapshot",
           "req": {"coin": coin, "interval": interval, "startTime": _ms(start), "endTime": _ms(end)}}
    r = urllib.request.Request(HL_INFO_URL, data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.load(resp)


def fetch_yahoo(symbol: str, query: str):
    """Real cash-equity (spot) OHLC from the Yahoo chart API. Returns (times, o, h, l, c)
    with null bars dropped. query e.g. 'range=1mo&interval=5m' or
    'period1=..&period2=..&interval=1d'."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    r = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        d = json.load(resp)
    res = d["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    t, o, h, low, c = [], [], [], [], []
    for i, sec in enumerate(ts):
        if q["close"][i] is None:
            continue
        t.append(datetime.fromtimestamp(sec, timezone.utc))
        o.append(q["open"][i]); h.append(q["high"][i]); low.append(q["low"][i]); c.append(q["close"][i])
    return t, np.array(o), np.array(h), np.array(low), np.array(c)


def _series(candles: list[dict]):
    t = [datetime.fromtimestamp(c["t"] / 1000, timezone.utc) for c in candles]
    o = np.array([float(c["o"]) for c in candles])
    h = np.array([float(c["h"]) for c in candles])
    low = np.array([float(c["l"]) for c in candles])
    cl = np.array([float(c["c"]) for c in candles])
    return t, o, h, low, cl


def liq_price_short(entry: float, lev: float, mmr: float) -> float:
    return entry * (1.0 + 1.0 / lev) / (1.0 + mmr)


def break_on_gaps(times, vals, max_gap):
    """Insert a NaN break wherever consecutive samples are separated by more than `max_gap`,
    so a line plot does NOT stitch a straight diagonal across a non-trading gap (cash equities
    only trade regular hours; plotting them on a continuous 24/7 axis otherwise connects each
    session's close to the next open). Returns (times2, vals2) as lists."""
    from datetime import timedelta as _td  # local alias; module-level import also exists
    if len(times) == 0:
        return list(times), list(vals)
    out_t, out_v = [times[0]], [float(vals[0])]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > max_gap:
            out_t.append(times[i - 1] + (times[i] - times[i - 1]) / 2)
            out_v.append(float("nan"))
        out_t.append(times[i])
        out_v.append(float(vals[i]))
    return out_t, out_v


def plot_sessions_dotted_gaps(ax, times, vals, color, label, max_gap, lw=1.6, marker=None, ms=3):
    """Plot a series so each trading session is a SOLID line and the non-trading gaps
    (overnight / weekend closures) show as DOTTED connectors -- making the closure visible
    rather than blank. Done by drawing a faint dotted full-series line underneath, then the
    solid in-session segments (NaN-broken at gaps) on top."""
    if len(times) == 0:
        return
    ax.plot(times, vals, color=color, ls=":", lw=max(0.8, lw * 0.6), alpha=0.65, zorder=1)
    bt, bv = break_on_gaps(times, vals, max_gap)
    ax.plot(bt, bv, color=color, ls="-", lw=lw, marker=marker, ms=ms, label=label, zorder=2)


# ----------------------------------------------------------------------------------------
def chart_full_arc(t, o, h, low, cl, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t, cl, color="black", lw=1.4, label="CBRS perp price (HL 1h close)")
    ax.fill_between(t, low, h, color="tab:gray", alpha=0.18, label="1h low-high range")

    # IPO/listing anchors (cash-equity, corroborated)
    ax.axhline(CBRS_IPO_OFFER, color="tab:green", ls="--", lw=1.3,
               label=f"IPO offer ${CBRS_IPO_OFFER:.0f}")
    ax.axvline(LISTING, color="tab:purple", ls=":", lw=1.5)
    ax.text(LISTING, ax.get_ylim()[1] * 0.985, " listing 2026-05-14:\n open $350 / high ~$385 /\n close $311 (+68%)",
            color="tab:purple", fontsize=8, va="top")

    # short-entry + liquidation lines for a $277 PERP short at 1x/2x/5x (mmr 0.10)
    ax.axhline(CBRS_SHORT_ENTRY, color="tab:blue", lw=1.0, alpha=0.8)
    ax.text(t[1], CBRS_SHORT_ENTRY, f" perp short entry ${CBRS_SHORT_ENTRY:.0f}", color="tab:blue", fontsize=8, va="bottom")
    for lev, c in [(1.0, "tab:green"), (2.0, "tab:orange"), (5.0, "tab:red")]:
        lp = liq_price_short(CBRS_SHORT_ENTRY, lev, CBRS_MMR)
        ax.axhline(lp, color=c, ls="-.", lw=1.1, alpha=0.9)
        ax.text(t[-1], lp, f" {lev:.0f}x liq ${lp:.0f}", color=c, fontsize=8, va="center", ha="right")

    peak_i = int(np.argmax(h))
    ax.annotate(f"listing-day high ${h[peak_i]:,.0f} (perp)\na levered perp short is liquidated here",
                xy=(t[peak_i], h[peak_i]), xytext=(t[peak_i], h[peak_i] * 1.02),
                fontsize=8, color="tab:red", ha="center",
                arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.annotate(f"settled ~${cl[-1]:,.0f}", xy=(t[-1], cl[-1]), xytext=(t[-1], cl[-1] * 0.80),
                fontsize=8, ha="right", arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_title("Cerebras (xyz:CBRS) perp: pre-IPO run-up -> listing spike -> settle\n"
                 "Real Hyperliquid data. A $277 perp short is liquidated at the ~$392 spike at >=2x; only unlevered (1x) survived.")
    ax.set_ylabel("price ($ / share-equivalent)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_intraday(t, o, h, low, cl, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    # candlestick-ish: vertical low-high bars + close line
    for i in range(len(t)):
        ax.plot([t[i], t[i]], [low[i], h[i]], color="tab:gray", lw=0.8, alpha=0.7)
    ax.plot(t, cl, color="black", lw=1.3, label="CBRS perp (15m close)")
    ax.axhline(CBRS_SHORT_ENTRY, color="tab:blue", lw=1.0)
    ax.text(t[0], CBRS_SHORT_ENTRY, f" short entry ${CBRS_SHORT_ENTRY:.0f}", color="tab:blue", fontsize=8, va="bottom")
    for lev, c in [(2.0, "tab:orange"), (5.0, "tab:red")]:
        lp = liq_price_short(CBRS_SHORT_ENTRY, lev, CBRS_MMR)
        ax.axhline(lp, color=c, ls="-.", lw=1.1)
        ax.text(t[-1], lp, f" {lev:.0f}x liq ${lp:.0f}", color=c, fontsize=8, va="center", ha="right")
    peak_i = int(np.argmax(h))
    ax.annotate(f"high ${h[peak_i]:,.0f}\n{t[peak_i]:%b %d %H:%MZ}", xy=(t[peak_i], h[peak_i]),
                xytext=(t[peak_i], h[peak_i] * 1.01), fontsize=8, color="tab:red", ha="center",
                arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.set_title("Cerebras listing window (2026-05-12 -> 05-16), 15-minute candles\n"
                 "The transition spike to ~$392 took out levered perp shorts in minutes (2x liq $378, 5x $302).")
    ax.set_ylabel("price ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %Hh"))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_spcx_compare(out: Path) -> None:
    """Where SPCX sits today on the same journey (pre-listing run-up)."""
    try:
        c = fetch_candles("xyz:SPCX", "1h", datetime(2026, 5, 1, tzinfo=timezone.utc),
                          datetime(2026, 6, 9, tzinfo=timezone.utc))
    except Exception as exc:  # noqa: BLE001
        print(f"[spcx compare] fetch failed: {exc}; skipping")
        return
    if not c:
        print("[spcx compare] no SPCX candles; skipping")
        return
    t, o, h, low, cl = _series(c)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t, cl, color="tab:blue", lw=1.4, label="xyz:SPCX perp (1h close)")
    ax.fill_between(t, low, h, color="tab:blue", alpha=0.12)
    ax.axhline(135.0, color="tab:green", ls="--", lw=1.3, label="IPO offer $135")
    ax.text(t[1], 135.0, " IPO offer $135", color="tab:green", fontsize=8, va="bottom")
    ax.annotate(f"now ~${cl[-1]:,.0f}\n(+{(cl[-1]/135-1)*100:.0f}% vs offer)",
                xy=(t[-1], cl[-1]), xytext=(t[-1], cl[-1] * 1.04), fontsize=8, ha="right",
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_title("SpaceX (xyz:SPCX) perp today vs the $135 IPO offer -- pre-listing, same kind of run-up\n"
                 "Anticipated listing ~2026-06-12. This is the analog to Cerebras's pre-listing leg.")
    ax.set_ylabel("price ($ / share)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_perp_vs_spot_listingday(out: Path) -> None:
    """The real gap story. Pre-IPO there is NO spot -- only the perp trades, and the perp
    PRE-DISCOVERS the open: CBRS was referenced at the $185 offer (no trades) until ~16:55 UTC
    when the cash equity finally opened at ~$385, by which point the perp had ALREADY run to
    ~$392 hours earlier. Overlay perp (HL 15m -- 5m history isn't retained that far back) vs
    spot (Yahoo 5m), dropping the pre-open $185 placeholder bars."""
    day0 = datetime(2026, 5, 14, tzinfo=timezone.utc)
    day1 = datetime(2026, 5, 15, 0, tzinfo=timezone.utc)
    pt, po, ph, pl, pc = _series(fetch_candles("xyz:CBRS", "15m", day0, day1))
    st_all, _, _, _, sc_all = fetch_yahoo("CBRS", "range=1mo&interval=5m")
    # real spot trading = first bar above offer*1.4 (drops the $185 pre-open reference placeholder)
    open_thresh = CBRS_IPO_OFFER * 1.4
    st, sc = [], []
    started = False
    for x, v in zip(st_all, sc_all):
        if x.date() != day0.date():
            continue
        if not started and v < open_thresh:
            continue
        started = True
        st.append(x); sc.append(v)

    fig, ax = plt.subplots(figsize=(11, 6))
    pbt, pbv = break_on_gaps(pt, pc, timedelta(hours=1))
    ax.plot(pbt, pbv, color="tab:blue", lw=1.7, marker=".", ms=4, label="xyz:CBRS PERP (HL 15m close)")
    if st:
        plot_sessions_dotted_gaps(ax, st, sc, "tab:red",
                                  "CBRS SPOT (Nasdaq cash, Yahoo 5m; dotted = closed)", timedelta(minutes=30), lw=1.7)
    ax.axhline(CBRS_IPO_OFFER, color="tab:green", ls="--", lw=1.2, label=f"IPO offer ${CBRS_IPO_OFFER:.0f}")

    peak_i = int(np.argmax(ph))
    ax.annotate(f"perp pre-discovers the open:\n~${ph[peak_i]:,.0f} by {pt[peak_i]:%H:%MZ}\n(hours before cash opens)",
                xy=(pt[peak_i], ph[peak_i]), xytext=(day0.replace(hour=2), 360),
                color="tab:blue", fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:blue"))
    if st:
        ax.annotate(f"CASH OPENS ~${sc[0]:,.0f} at {st[0]:%H:%MZ}\n(+{(sc[0]/CBRS_IPO_OFFER-1)*100:.0f}% gap vs $185 offer;\nperp was already here)",
                    xy=(st[0], sc[0]), xytext=(st[0], sc[0] - 70), color="tab:red", fontsize=8, ha="center",
                    arrowprops=dict(arrowstyle="->", color="tab:red"))
        ax.annotate("spot referenced at $185\npre-open (no trades)", xy=(day0.replace(hour=14), CBRS_IPO_OFFER),
                    xytext=(day0.replace(hour=14, minute=10), CBRS_IPO_OFFER + 30), color="tab:gray", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="tab:gray"))
    ax.set_title("Cerebras listing day (2026-05-14): the perp pre-discovers the IPO open\n"
                 "Pre-listing only the perp trades; cash gaps from $185 reference to ~$385 open -- to where the perp already was.")
    ax.set_ylabel("price ($ / share)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%MZ"))
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)


def chart_perp_vs_spot_multiday(out: Path) -> None:
    """Perp vs spot daily once both trade, plus the basis (perp - spot). Shows how tightly
    funding ties the converted perp to the cash equity after listing."""
    perp = fetch_candles("xyz:CBRS", "1d", datetime(2026, 5, 13, tzinfo=timezone.utc),
                         datetime(2026, 6, 9, tzinfo=timezone.utc))
    pt, po, ph, pl, pc = _series(perp)
    st, so, sh, sl, sc = fetch_yahoo("CBRS", "period1=1747094400&period2=1781000000&interval=1d")
    # align on calendar date
    sp = {x.date(): v for x, v in zip(st, sc)}
    days, perp_c, spot_c = [], [], []
    for x, v in zip(pt, pc):
        if x.date() in sp:
            days.append(x); perp_c.append(v); spot_c.append(sp[x.date()])
    perp_c = np.array(perp_c); spot_c = np.array(spot_c)
    basis = perp_c - spot_c

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[3, 1])
    ax.plot(days, perp_c, color="tab:blue", lw=1.6, marker="o", ms=3, label="xyz:CBRS perp (daily close)")
    ax.plot(days, spot_c, color="tab:red", lw=1.6, marker="o", ms=3, label="CBRS spot (daily close)")
    ax.axhline(CBRS_IPO_OFFER, color="tab:green", ls="--", lw=1.0, label=f"IPO offer ${CBRS_IPO_OFFER:.0f}")
    ax.set_title("Cerebras after listing: perp vs spot daily close, and the basis (perp - spot)\n"
                 "Real HL + Yahoo data. Once both trade, funding keeps the converted perp tied to the cash equity.")
    ax.set_ylabel("price ($)"); ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    ax2.bar(days, basis, color=["tab:green" if b >= 0 else "tab:red" for b in basis], width=0.7)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("basis $\n(perp-spot)"); ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)


# ======================================================================================
# LIFECYCLE MAPPING (Phase A timeline -> Phase C phase-aligned readout + charts)
# ======================================================================================
ET_OFFSET = timedelta(hours=-4)  # EDT = UTC-4 in May 2026 (US Eastern, daylight time)


def et_of(utc_dt: datetime) -> datetime:
    """US/Eastern (EDT, UTC-4) wall-clock for a UTC datetime in the May-2026 window."""
    return utc_dt + ET_OFFSET


def nearest_at_or_before(times: list[datetime], vals, target: datetime, max_gap=None):
    """Lookahead-free pick: the value at the latest candle whose time is <= target.
    Returns (time, value), or (None, None) if target precedes the first candle OR if the
    nearest prior candle is staler than max_gap (a data-coverage gap -> flag, don't silently
    reuse a stale price). Assumes `times` is ascending."""
    chosen = None
    for t, v in zip(times, vals):
        if t <= target:
            chosen = (t, float(v))
        else:
            break
    if chosen is None:
        return (None, None)
    if max_gap is not None and (target - chosen[0]) > max_gap:
        return (None, None)
    return chosen


# Phase A -- SOURCED event timeline. UTC is authoritative; ET derived (EDT, UTC-4).
# `utc` = best-sourced wall-clock; `time_confirmed=False` => only the DATE is sourced, the
# clock time is a labeled placeholder (listing/+close times are data-derived and confirmed).
# `source` URLs are filled from the Phase-D research pass; unfilled => "pending".
CBRS_IPO_OFFER_RANGE_1 = (115.0, 125.0)   # initial S-1, 28M shares (investing.com, 2026-05-04)
CBRS_IPO_OFFER_RANGE_2 = (150.0, 160.0)   # raised + upsized to 30M (CNBC/MarketWise, 2026-05-11)
CBRS_LISTING_OPEN_UTC = datetime(2026, 5, 14, 16, 59, tzinfo=timezone.utc)  # 12:59 ET first trade (IPOScoop)
CBRS_OFFICIAL_OPEN = 350.0      # majority-reported opening print (IPOScoop/Yahoo/CNBC)
CBRS_FIRST_PRINT_5M = 385.0     # first non-placeholder 5m print (~16:55Z) / TechCrunch-reported open

# sourced URLs (Phase-D research pass)
SRC_S1 = "https://www.investing.com/news/stock-market-news/cerebras-systems-plans-ipo-with-shares-priced-between-115125-432SI-4655311"
SRC_RAISE = "https://www.cnbc.com/2026/05/11/cerebras-raises-ipo-range.html"
SRC_PRICING = "https://www.globenewswire.com/news-release/2026/05/13/3294565/0/en/cerebras-systems-announces-pricing-of-initial-public-offering.html"
SRC_LISTING = "https://www.iposcoop.com/the-ipo-buzz-cerebras-cbrs-prices-ipo-at-185-25-above-range-2/"
SRC_CLOSE = "https://finance.yahoo.com/markets/article/cerebras-stock-slides-after-near-70-surge-in-biggest-ipo-of-2026-130757084.html"


def cbrs_events() -> list[dict]:
    def ev(label, utc, price, confirmed, source, note=""):
        return {"label": label, "utc": utc, "et": et_of(utc) if utc else None,
                "price": price, "time_confirmed": confirmed, "source": source, "note": note}
    return [
        ev("S-1 initial range $115-125 (28M sh)", datetime(2026, 5, 4, 10, 26, tzinfo=timezone.utc),
           None, False, SRC_S1, "date 05-04 sourced; clock time = article publish, SEC filing time unconfirmed"),
        ev("Range raise to $150-160 + upsize to 30M sh", datetime(2026, 5, 11, 16, 0, tzinfo=timezone.utc),
           None, False, SRC_RAISE, "Monday 05-11 filing sourced; exact clock time unconfirmed"),
        ev("Pricing night: IPO priced $185", datetime(2026, 5, 13, 23, 45, tzinfo=timezone.utc),
           185.0, True, SRC_PRICING, "GlobeNewswire wire 19:45 ET / 23:45 UTC (corroborated StockTitan)"),
        ev("Allocation window (pricing -> listing open)", datetime(2026, 5, 14, 4, 0, tzinfo=timezone.utc),
           None, False, SRC_PRICING, "overnight window, not a single print"),
        ev("Listing open (first cash trade)", CBRS_LISTING_OPEN_UTC, CBRS_FIRST_PRINT_5M, True,
           SRC_LISTING, "first trade 12:59 ET (IPOScoop, single source); open DISPUTED $350 vs $385"),
        ev("+1 trading day close (Fri 05-15)", datetime(2026, 5, 15, 20, 0, tzinfo=timezone.utc),
           None, True, SRC_CLOSE, "05-16 is Saturday (no session); news 'down ~10%'"),
        ev("+2 trading days close (Mon 05-18)", datetime(2026, 5, 18, 20, 0, tzinfo=timezone.utc),
           None, True, SRC_CLOSE, "+2 trading days; Yahoo close ~$296.65 (news '+17%' does NOT reconcile)"),
    ]


def build_phase_table(events, perp_t, perp_c, spot_t, spot_c, offer, listing_open_utc,
                      max_gap=timedelta(hours=2), perp_h=None):
    """Phase-aligned, LOOKAHEAD-FREE readout. For each event: perp at-or-before, spot at-or-before
    (None pre-listing), perp->offer basis, perp->spot basis. `max_gap` flags a data-coverage gap
    as None rather than silently reusing a stale price. Returns (rows, spike_dict)."""
    rows = []
    for e in events:
        if e["utc"] is None:
            continue
        pt, pv = nearest_at_or_before(perp_t, perp_c, e["utc"], max_gap=max_gap)
        assert pt is None or pt <= e["utc"], "lookahead: perp candle postdates event"
        st, sv = (None, None)
        if e["utc"] >= listing_open_utc:
            st, sv = nearest_at_or_before(spot_t, spot_c, e["utc"], max_gap=max_gap)
            assert st is None or st <= e["utc"], "lookahead: spot candle postdates event"
        else:
            sv = None  # NO spot before listing -- never forward-fill
        rows.append({
            "label": e["label"], "utc": e["utc"], "et": e["et"],
            "time_confirmed": e["time_confirmed"],
            "perp": pv, "spot": sv,
            "perp_offer_basis_$": (pv - offer) if pv is not None else None,
            "perp_offer_basis_%": ((pv / offer - 1) * 100) if pv else None,
            "perp_spot_basis_$": (pv - sv) if (pv is not None and sv is not None) else None,
            "note": e["note"],
        })
    # spike = running max perp HIGH across the listing window (the worst point a short faced)
    series = perp_h if perp_h is not None else perp_c
    win = [(t, float(v)) for t, v in zip(perp_t, series)
           if datetime(2026, 5, 14, tzinfo=timezone.utc) <= t < datetime(2026, 5, 15, tzinfo=timezone.utc)]
    spike_t, spike_v = max(win, key=lambda x: x[1]) if win else (None, None)
    return rows, {"time": spike_t, "perp": spike_v}


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if x is None else str(x) for x in r) + "\n")


def chart_event_timeline(perp_t, perp_c, spot_t, spot_c, events, spike, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    pbt, pbv = break_on_gaps(perp_t, perp_c, timedelta(hours=1))      # perp 24/7; break any HL gap
    ax.plot(pbt, pbv, color="tab:blue", lw=1.3, label="xyz:CBRS PERP (HL 15m, 24/7)")
    if spot_t:  # solid in-session, dotted across overnight/weekend closures
        plot_sessions_dotted_gaps(ax, spot_t, spot_c, "tab:red",
                                  "CBRS SPOT (Nasdaq RTH 5m; dotted = closed)", timedelta(minutes=30), lw=1.3)
    ax.axhline(CBRS_IPO_OFFER, color="tab:green", ls="--", lw=1.1, label=f"IPO offer ${CBRS_IPO_OFFER:.0f}")
    ymax = max(perp_c) * 1.05
    alloc0 = alloc1 = None
    for e in events:
        if e["utc"] is None:
            continue
        if e["label"].startswith("Allocation"):
            continue
        ls = "-" if e["time_confirmed"] else ":"
        ax.axvline(e["utc"], color="tab:gray", ls=ls, lw=1.0, alpha=0.8)
        short = (e["label"].split(":")[0].split("(")[0].strip())
        ax.text(e["utc"], ymax, " " + short, rotation=90, va="top", ha="left", fontsize=7, color="dimgray")
        if "Pricing night" in e["label"]:
            alloc0 = e["utc"]
        if "Listing open" in e["label"]:
            alloc1 = e["utc"]
    if alloc0 and alloc1:
        ax.axvspan(alloc0, alloc1, color="tab:orange", alpha=0.12)
        ax.text(alloc0, min(perp_c), " allocation\n window", fontsize=7, color="tab:orange", va="bottom")
    if spike["time"]:
        ax.annotate(f"perp spike ${spike['perp']:,.0f}\n{et_of(spike['time']):%b %d %H:%M ET}",
                    xy=(spike["time"], spike["perp"]),
                    xytext=(datetime(2026, 5, 20, tzinfo=timezone.utc), spike["perp"] - 2),
                    fontsize=8, color="tab:red", ha="left",
                    arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.set_title("Cerebras IPO lifecycle: pre-IPO perp (continuous) vs cash spot (from listing open)\n"
                 "Solid markers = sourced times; dotted = date-only (clock unconfirmed). Real HL + Yahoo data.")
    ax.set_ylabel("price ($ / share)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)


def chart_listing_window_annotated(perp_t, perp_c, spot_t, spot_c, events, out: Path) -> None:
    lo = datetime(2026, 5, 13, 4, tzinfo=timezone.utc)   # 05-13 00:00 ET
    hi = datetime(2026, 5, 15, 4, tzinfo=timezone.utc)   # 05-15 00:00 ET
    pt = [(t, v) for t, v in zip(perp_t, perp_c) if lo <= t <= hi]
    st = [(t, v) for t, v in zip(spot_t, spot_c) if lo <= t <= hi]
    fig, ax = plt.subplots(figsize=(12, 6))
    if pt:
        pbt, pbv = break_on_gaps([t for t, _ in pt], [v for _, v in pt], timedelta(hours=1))
        ax.plot(pbt, pbv, color="tab:blue", lw=1.6, marker=".", ms=3, label="xyz:CBRS PERP (HL 15m)")
    if st:
        plot_sessions_dotted_gaps(ax, [t for t, _ in st], [v for _, v in st], "tab:red",
                                  "CBRS SPOT (RTH 5m; dotted = closed)", timedelta(minutes=30), lw=1.6)
    ax.axhline(CBRS_IPO_OFFER, color="tab:green", ls="--", lw=1.1, label=f"IPO offer ${CBRS_IPO_OFFER:.0f}")
    for e in events:
        if e["utc"] is None or not (lo <= e["utc"] <= hi):
            continue
        ls = "-" if e["time_confirmed"] else ":"
        ax.axvline(e["utc"], color="tab:gray", ls=ls, lw=1.0)
        ax.text(e["utc"], ax.get_ylim()[1], " " + e["label"].split(":")[0].split("(")[0].strip(),
                rotation=90, va="top", fontsize=7, color="dimgray")
    ax.set_title("Cerebras listing window (05-13 00:00 ET -> 05-15 00:00 ET)\n"
                 "Perp 15m is the FINEST HL retains for this date (1m/5m purged); spot 5m from the open.")
    ax.set_ylabel("price ($ / share)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%MZ"))
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)


def run_lifecycle() -> None:
    """Phase B/C: pull tape, write CSVs, build phase table + 2 annotated charts, run asserts."""
    # --- Phase B: perp 15m full arc (finest HL serves; 5m/1m purged) ---
    perp = fetch_candles("xyz:CBRS", "15m", datetime(2026, 4, 20, tzinfo=timezone.utc),
                         datetime(2026, 6, 9, tzinfo=timezone.utc))  # full arc (covers +2-day close + drift)
    pt, po, ph, pl, pc = _series(perp)
    # spot 5m (finest Yahoo serves for the listing day; 1m window has passed)
    st_all, sso, ssh, ssl, ssc = fetch_yahoo("CBRS", "range=1mo&interval=5m")
    # drop the pre-open $185 placeholder bars; spot exists only from the real open
    st, ssc2 = [], []
    started = False
    for t, v in zip(st_all, ssc):
        if t < CBRS_LISTING_OPEN_UTC - timedelta(minutes=20):
            continue
        if not started and v < CBRS_IPO_OFFER * 1.4:
            continue
        started = True
        st.append(t); ssc2.append(v)
    ssc2 = np.array(ssc2)

    # --- assert: NO spot before the listing open (never forward-fill) ---
    assert all(t >= CBRS_LISTING_OPEN_UTC - timedelta(minutes=20) for t in st), "spot present pre-listing"
    # --- assert: ET = UTC-4 in May 2026 ---
    probe = datetime(2026, 5, 14, 18, 0, tzinfo=timezone.utc)
    assert et_of(probe).hour == 14, "ET offset wrong"

    events = cbrs_events()
    rows, spike = build_phase_table(events, pt, pc, st, list(ssc2), CBRS_IPO_OFFER,
                                    CBRS_LISTING_OPEN_UTC, perp_h=ph)

    # short entered at the pricing-night perp level -> adverse move to the spike
    pricing = next(r for r in rows if "Pricing night" in r["label"])
    if pricing["perp"] and spike["perp"]:
        adverse = (spike["perp"] / pricing["perp"] - 1) * 100
        print(f"\nPricing-night perp ${pricing['perp']:.2f} -> spike ${spike['perp']:.2f} "
              f"= +{adverse:.1f}% adverse for a short entered at pricing")

    # --- print phase table ---
    print("\nPHASE-ALIGNED READOUT (lookahead-free; perp/spot = candle at-or-before each event)")
    print(f"{'event':42} {'ET':16} {'perp':>8} {'spot':>8} {'p-offer%':>9} {'p-spot$':>8}")
    for r in rows:
        et = r["et"].strftime("%m-%d %H:%M") + ("" if r["time_confirmed"] else "?")
        perp = f"{r['perp']:.2f}" if r["perp"] else "-"
        spot = f"{r['spot']:.2f}" if r["spot"] else "(none)"
        pob = f"{r['perp_offer_basis_%']:+.1f}" if r["perp_offer_basis_%"] is not None else "-"
        psb = f"{r['perp_spot_basis_$']:+.1f}" if r["perp_spot_basis_$"] is not None else "-"
        print(f"{r['label']:42} {et:16} {perp:>8} {spot:>8} {pob:>9} {psb:>8}")

    # --- write tidy CSVs (UTC + ET columns) ---
    write_csv(CSV_OUT / "cerebras_cbrs_perp_15m.csv",
              ["time_utc", "time_et", "open", "high", "low", "close"],
              [[pt[i].isoformat(), et_of(pt[i]).isoformat(), po[i], ph[i], pl[i], pc[i]]
               for i in range(len(pt))])
    write_csv(CSV_OUT / "cerebras_cbrs_spot_5m_listingwindow.csv",
              ["time_utc", "time_et", "close"],
              [[st[i].isoformat(), et_of(st[i]).isoformat(), ssc2[i]] for i in range(len(st))])
    write_csv(CSV_OUT / "cerebras_lifecycle_phase_table.csv",
              ["event", "time_utc", "time_et", "time_confirmed", "perp", "spot",
               "perp_offer_basis_usd", "perp_offer_basis_pct", "perp_spot_basis_usd", "note"],
              [[r["label"], r["utc"].isoformat(), r["et"].isoformat(), r["time_confirmed"],
                r["perp"], r["spot"], r["perp_offer_basis_$"], r["perp_offer_basis_%"],
                r["perp_spot_basis_$"], r["note"]] for r in rows])

    # --- charts ---
    chart_event_timeline(pt, pc, st, list(ssc2), events, spike, PLOTS / "cerebras_event_timeline.png")
    chart_listing_window_annotated(pt, pc, st, list(ssc2), events,
                                   PLOTS / "cerebras_listing_window_annotated.png")
    print(f"[lifecycle] wrote 2 charts + 3 CSVs; perp 15m bars={len(pt)}, spot 5m bars={len(st)}")
    print("[lifecycle] NOTE: HL 5m/1m and Yahoo 1m history are PURGED for 05-14 -> 15m perp / 5m spot are the finest available")


def main() -> int:
    c1h = fetch_candles("xyz:CBRS", "1h", datetime(2026, 4, 20, tzinfo=timezone.utc),
                        datetime(2026, 6, 9, tzinfo=timezone.utc))
    c15 = fetch_candles("xyz:CBRS", "15m", datetime(2026, 5, 12, tzinfo=timezone.utc),
                        datetime(2026, 5, 16, tzinfo=timezone.utc))
    print(f"CBRS 1h candles: {len(c1h)}; 15m: {len(c15)}")
    t, o, h, low, cl = _series(c1h)
    peak_i = int(np.argmax(h))
    print(f"high ${h[peak_i]:,.2f} at {t[peak_i].isoformat()}; latest close ${cl[-1]:,.2f}")

    chart_full_arc(t, o, h, low, cl, PLOTS / "cerebras_full_arc.png")
    t2, o2, h2, l2, cl2 = _series(c15)
    chart_intraday(t2, o2, h2, l2, cl2, PLOTS / "cerebras_intraday_listing.png")
    chart_spcx_compare(PLOTS / "spcx_vs_offer_today.png")
    for fn, name in [(chart_perp_vs_spot_listingday, "cerebras_perp_vs_spot_listingday.png"),
                     (chart_perp_vs_spot_multiday, "cerebras_perp_vs_spot_multiday.png")]:
        try:
            fn(PLOTS / name)
        except Exception as exc:  # noqa: BLE001 - spot source optional; perp charts already built
            print(f"[perp-vs-spot] {name} skipped: {exc}")
    print(f"[charts] wrote PNGs to {PLOTS}")

    # save the 1h candles for reproducibility
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_OUT / "cerebras_cbrs_perp_1h.csv"
    with csv_path.open("w") as f:
        f.write("time_utc,open,high,low,close\n")
        for i in range(len(t)):
            f.write(f"{t[i].isoformat()},{o[i]},{h[i]},{low[i]},{cl[i]}\n")
    print(f"[csv] wrote {csv_path}")

    # Phase A->C lifecycle mapping (event timeline + phase-aligned readout + annotated charts)
    run_lifecycle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
