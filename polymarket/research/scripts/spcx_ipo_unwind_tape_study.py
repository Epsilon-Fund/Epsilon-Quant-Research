"""Block S2 -- Mega-IPO listing-day unwind tape study (SPCX residual-sleeve calibration).

WHY: the SPCX listing-day gameplan (spcx_listing_day_gameplan.md §5.2/§7 Block S2) sells the
unhedged residual sleeve in tranches after the IPO cross. The tranche windows, the fade prior,
and S1's melt-up distribution were all Cerebras-only assumptions. This script measures them on
every mega-IPO day-1 tape that free sources still serve, and compares three LOOKAHEAD-FREE
unwind policies on each tape.

SAMPLE + DATA HONESTY:
  - Cerebras (CBRS, 2026-05-14): Yahoo serves a REAL 1-minute listing-day tape (within the 30d
    retention window) -> full intraday microstructure + policy simulation. HL 15m perp tape is
    already cached by spcx_cerebras_case_study.py and is used for the overlay chart only.
  - ARM 2023 / RIVN 2021 / BABA 2014 / META(FB) 2012 / RDDT 2024: Yahoo intraday is PURGED
    (422); only daily OHLCV survives. These degrade to daily-OHLC PROXY rows -- open ~ cross,
    OHLC/4 ~ day TWAP, close = close -- and every such number is labelled PROXY.
  - Everything fetched is cached under data/analysis/csv_outputs/market_maps/ipo_tapes/ so the
    study reruns offline after Yahoo purges the CBRS 1m window.

POLICIES (mechanical, pre-registered, clocks in minutes-since-cross; NO signal uses future data
-- each is a fixed schedule known at the cross):
  A "Alvaro 40/40/20": observe [0,15); sell 40% as TWAP over [15,60), 40% over [60,180),
    20% over [180, close].
  B "TWAP-from-cross": equal-weight TWAP of 1m closes from the cross bar to the close.
  C "sell-all-at-cross+15": single print at the 1m close 15 minutes after the cross.
Slippage: flat $/share budget in {0.10, 0.30, 0.50} subtracted from the gross average sell
price. Flat per-share slippage shifts all three policies identically -> it changes levels,
never rankings; rankings would only move under per-order or spread-scaled costs.

EXTRA OUTPUT (S1 hook): the empirical day-1 high-vs-offer distribution across the sample,
printed in `--meltup-dist` format (move:weight,...) so spcx_convergence_calc.py --decision can
swap its equal-weighted +13/26/39% Cerebras assumption for a measured one.

RUN: cd polymarket/research && PYTHONPATH=. uv run python scripts/spcx_ipo_unwind_tape_study.py
OUT: CSVs -> data/analysis/csv_outputs/market_maps/   PNGs -> data/analysis/plots/spcx_convergence/
"""
from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False  # render literal "$" (no mathtext)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_OUT = ROOT / "data" / "analysis" / "csv_outputs" / "market_maps"
TAPE_CACHE = CSV_OUT / "ipo_tapes"
PLOTS = ROOT / "data" / "analysis" / "plots" / "spcx_convergence"

SLIPPAGE_LEVELS = [0.10, 0.30, 0.50]  # $/share flat budget, per the S2 spec

# Mechanical 40/40/20 tranche schedule, minutes since cross: (start, end_exclusive, weight).
# end=None means "to the close". Pre-registered before looking at any policy result.
TRANCHE_SCHEDULE = [(15, 60, 0.40), (60, 180, 0.40), (180, None, 0.20)]
OBSERVE_MIN = 15  # no selling in the first 15 minutes post-cross (gameplan §5.2 Phase A)


# ---------------------------------------------------------------- IPO registry
@dataclass
class IPO:
    symbol: str            # Yahoo symbol today (META, not FB)
    label: str             # display label incl. listing year
    listing_date: str      # YYYY-MM-DD (exchange listing day, US/Eastern date)
    offer: float           # IPO offer price $/share
    oversub: float | None  # order-book oversubscription multiple (None = never published)
    oversub_note: str      # source + confidence for the oversub figure
    intraday: bool = False # True only where a real 1-5m day-1 tape still exists


REGISTRY: list[IPO] = [
    IPO("CBRS", "Cerebras 2026", "2026-05-14", 185.0, 20.0,
        "~20x (vault: TECHi/techstackipo via spcx case study); HIGH-ish confidence", True),
    IPO("ARM",  "ARM 2023",      "2023-09-14", 51.0, 10.0,
        "'oversubscribed 10x, could reach 15x' (Bloomberg 2023-09-11); books-close figure unconfirmed"),
    IPO("RIVN", "Rivian 2021",   "2021-11-10", 78.0, None,
        "reported 'oversubscribed', multiple never published -> excluded from the oversub scatter"),
    IPO("BABA", "Alibaba 2014",  "2014-09-19", 68.0, 17.0,
        "reports range 14-22x; 17x midpoint used, LOW confidence on the exact multiple"),
    IPO("META", "Facebook 2012", "2012-05-18", 38.0, 5.0,
        "institutional demand ~5x shares (Wikipedia IPO account); 15-20x claims exist -> 5x = conservative"),
    IPO("RDDT", "Reddit 2024",   "2024-03-21", 34.0, 4.5,
        "'4-5x oversubscribed' (Globe and Mail 2024-03-18); 4.5x midpoint"),
]

SPCX_OVERSUB = 4.0  # ~4x / ~$250B vs $75B as of 2026-06-10 (Reuters via Coindesk) -- plotted as a vline


# ---------------------------------------------------------------- data layer (cache-first)
def _yahoo_fetch(symbol: str, query: str) -> list[dict]:
    """Yahoo chart API -> list of bar dicts (UTC datetimes), null-close bars dropped."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        d = json.load(resp)
    res = d["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    bars = []
    for i, sec in enumerate(ts):
        if q["close"][i] is None:
            continue
        bars.append({"t": datetime.fromtimestamp(sec, timezone.utc),
                     "o": float(q["open"][i]), "h": float(q["high"][i]),
                     "l": float(q["low"][i]), "c": float(q["close"][i]),
                     "v": float(q["volume"][i] or 0)})
    return bars


def _cache_path(name: str) -> Path:
    return TAPE_CACHE / f"{name}.csv"


def _write_bars(path: Path, bars: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_utc", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b["t"].isoformat(), b["o"], b["h"], b["l"], b["c"], b["v"]])


def _read_bars(path: Path) -> list[dict]:
    bars = []
    with path.open() as f:
        for row in csv.DictReader(f):
            bars.append({"t": datetime.fromisoformat(row["time_utc"]),
                         "o": float(row["open"]), "h": float(row["high"]),
                         "l": float(row["low"]), "c": float(row["close"]),
                         "v": float(row["volume"])})
    return bars


def load_bars(cache_name: str, symbol: str, query: str) -> list[dict]:
    """Cache-first loader: the CBRS 1m window expires from Yahoo ~30d after listing, so the
    first successful fetch is persisted and every rerun is offline-stable."""
    p = _cache_path(cache_name)
    if p.exists():
        return _read_bars(p)
    bars = _yahoo_fetch(symbol, query)
    if bars:
        _write_bars(p, bars)
    return bars


def _epoch(date_str: str, days_offset: int = 0) -> int:
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int((d + timedelta(days=days_offset)).timestamp())


def load_daily_day1(ipo: IPO) -> tuple[dict, dict | None]:
    """(day-1 daily bar, day-2 daily bar or None). day-2 = next trading day."""
    bars = load_bars(f"{ipo.symbol.lower()}_daily_listing",
                     ipo.symbol,
                     f"period1={_epoch(ipo.listing_date, -1)}&period2={_epoch(ipo.listing_date, 5)}&interval=1d")
    want = datetime.strptime(ipo.listing_date, "%Y-%m-%d").date()
    day1 = next(b for b in bars if b["t"].date() == want)
    later = [b for b in bars if b["t"].date() > want]
    return day1, (later[0] if later else None)


def load_cbrs_1m() -> list[dict]:
    return load_bars("cbrs_spot_1m_listingday", "CBRS",
                     f"period1={_epoch('2026-05-14')}&period2={_epoch('2026-05-14', 1)}&interval=1m")


# ---------------------------------------------------------------- pure analysis (unit-tested)
def first_print_index(bars: list[dict], offer: float, thresh_mult: float = 1.2) -> int:
    """Index of the cross bar: first bar with real volume AND price clearly off the offer
    placeholder (pre-open bars sit exactly at the offer with v=0)."""
    for i, b in enumerate(bars):
        if b["v"] > 0 and b["c"] > offer * thresh_mult:
            return i
    raise ValueError("no cross bar found")


def anchored_vwap(bars: list[dict]) -> np.ndarray:
    """Running VWAP anchored at bars[0] (the cross), on typical price (h+l+c)/3.
    vwap[i] uses bars 0..i only -- lookahead-free by construction."""
    tp = np.array([(b["h"] + b["l"] + b["c"]) / 3 for b in bars])
    v = np.array([b["v"] for b in bars])
    cum_pv = np.cumsum(tp * v)
    cum_v = np.cumsum(v)
    out = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-12), tp)
    return out


def rolling_vwap_peak(bars: list[dict], window: int = 15) -> tuple[int, float]:
    """(index, value) of the max of the trailing `window`-bar VWAP -- the 'volume-weighted
    intraday peak': where volume-weighted price, not a single print, topped out."""
    tp = np.array([(b["h"] + b["l"] + b["c"]) / 3 for b in bars])
    v = np.array([b["v"] for b in bars])
    best_i, best_val = 0, -np.inf
    for i in range(len(bars)):
        lo = max(0, i - window + 1)
        vol = v[lo:i + 1].sum()
        val = (tp[lo:i + 1] * v[lo:i + 1]).sum() / vol if vol > 0 else tp[i]
        if val > best_val:
            best_i, best_val = i, float(val)
    return best_i, best_val


def fade_onset_minutes(bars: list[dict], drop_frac: float = 0.05) -> int | None:
    """Minutes since the cross until the close first sits `drop_frac` below the RUNNING
    intraday high (running max uses bars up to that minute only -- lookahead-free)."""
    run_hi = -np.inf
    for i, b in enumerate(bars):
        run_hi = max(run_hi, b["h"])
        if b["c"] <= run_hi * (1 - drop_frac):
            return i
    return None


def vwap_loss_minutes(bars: list[dict], persist: int = 10) -> int | None:
    """Minutes since cross of the first anchored-VWAP loss that is NOT reclaimed for
    `persist` consecutive minutes (the gameplan's 'losing VWAP and failing to reclaim')."""
    av = anchored_vwap(bars)
    below = 0
    for i, b in enumerate(bars):
        if b["c"] < av[i]:
            below += 1
            if below >= persist:
                return i - persist + 1
        else:
            below = 0
    return None


def volume_buckets_30m(bars: list[dict]) -> list[tuple[int, float]]:
    """[(bucket_start_minute, share_of_day1_volume)] in 30-min buckets since the cross."""
    tot = sum(b["v"] for b in bars)
    out: dict[int, float] = {}
    for i, b in enumerate(bars):
        out[(i // 30) * 30] = out.get((i // 30) * 30, 0.0) + b["v"]
    return [(k, v / tot) for k, v in sorted(out.items())]


def twap(bars: list[dict], start_min: int, end_min: int | None) -> float:
    """Equal-weight average of 1m closes over [start_min, end_min) minutes since cross
    (bars[0] = cross bar). end_min=None -> to the last bar."""
    seg = bars[start_min: end_min if end_min is not None else len(bars)]
    if not seg:
        seg = bars[-1:]  # window beyond the tape -> execute at the last print
    return float(np.mean([b["c"] for b in seg]))


def policy_tranche(bars: list[dict], schedule=tuple(TRANCHE_SCHEDULE)) -> float:
    """Average sell price of the mechanical 40/40/20 schedule (TWAP within each window)."""
    assert abs(sum(w for _, _, w in schedule) - 1.0) < 1e-9
    return float(sum(w * twap(bars, s, e) for s, e, w in schedule))


def policy_twap_from_cross(bars: list[dict]) -> float:
    return twap(bars, 0, None)


def policy_sell_at(bars: list[dict], minute: int = OBSERVE_MIN) -> float:
    return float(bars[min(minute, len(bars) - 1)]["c"])


def day_vwap(bars: list[dict]) -> float:
    av = anchored_vwap(bars)
    return float(av[-1])


def bootstrap_mean_ci(vals: list[float], n_boot: int = 10000, seed: int = 7) -> tuple[float, float, float]:
    """(mean, lo2.5, hi97.5) by bootstrap over the IPO sample. With n=6 this is a width
    illustration, not inference -- the findings note says so."""
    rng = np.random.default_rng(seed)
    arr = np.array(vals, float)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def meltup_dist_string(moves: list[float]) -> str:
    """Format day-1 high-vs-offer moves for spcx_convergence_calc.py --meltup-dist
    (move:weight, equal weights)."""
    return ",".join(f"{m:.3f}:1" for m in moves)


# ---------------------------------------------------------------- per-IPO rows
def cbrs_intraday_metrics(ipo: IPO) -> dict:
    bars_all = load_cbrs_1m()
    i0 = first_print_index(bars_all, ipo.offer)
    bars = bars_all[i0:]
    peak_i, peak_v = rolling_vwap_peak(bars)
    high = max(b["h"] for b in bars)
    close = bars[-1]["c"]
    return {
        "bars": bars, "cross_t": bars[0]["t"], "cross_close": bars[0]["c"],
        "official_open": bars[0]["o"],
        "high": high, "close": close,
        "vw_peak_min": peak_i, "vw_peak_val": peak_v,
        "fade_onset_min": fade_onset_minutes(bars),
        "fade_depth": (high - close) / high,
        "vwap_loss_min": vwap_loss_minutes(bars),
        "buckets": volume_buckets_30m(bars),
        "day_vwap": day_vwap(bars),
    }


def policy_row_intraday(ipo: IPO, bars: list[dict]) -> dict:
    return {"label": ipo.label, "kind": "1m tape",
            "A_tranche": policy_tranche(bars),
            "B_twap": policy_twap_from_cross(bars),
            "C_cross15": policy_sell_at(bars),
            "day_vwap": day_vwap(bars),
            "high": max(b["h"] for b in bars), "close": bars[-1]["c"], "offer": ipo.offer}


def policy_row_daily_proxy(ipo: IPO, d1: dict) -> dict:
    """Daily-OHLC PROXY: cross+15 ~ official open O; day TWAP ~ (O+H+L+C)/4; 40/40/20 ~
    0.4*O (early tranche near the open) + 0.4*OHLC4 (mid-day) + 0.2*C (trailing). These are
    stand-ins, not simulations -- labelled PROXY everywhere they appear."""
    o, h, l, c = d1["o"], d1["h"], d1["l"], d1["c"]
    ohlc4 = (o + h + l + c) / 4
    return {"label": ipo.label, "kind": "daily PROXY",
            "A_tranche": 0.4 * o + 0.4 * ohlc4 + 0.2 * c,
            "B_twap": ohlc4,
            "C_cross15": o,
            "day_vwap": ohlc4, "high": h, "close": c, "offer": ipo.offer}


# ---------------------------------------------------------------- charts
def _ann(ax, title, xlabel, ylabel, sample):
    import textwrap
    wrapped = "\n".join(textwrap.wrap(sample, width=105))
    ax.set_title(f"{title}\n{wrapped}", fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)


def chart_cbrs_day1(m: dict, out: Path) -> None:
    bars = m["bars"]
    t = [b["t"] for b in bars]
    c = [b["c"] for b in bars]
    av = anchored_vwap(bars)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax2 = ax.twinx()
    ax2.bar(t, [b["v"] / 1e6 for b in bars], width=1 / (24 * 60), color="tab:gray", alpha=0.35)
    ax2.set_ylabel("1m volume (M shares)", fontsize=9, color="tab:gray")
    ax.plot(t, c, color="black", lw=1.3, label="CBRS spot 1m close")
    ax.plot(t, av, color="tab:purple", lw=1.4, ls="--", label="anchored VWAP (from cross)")
    ax.axhline(185.0, color="tab:green", ls="--", lw=1.0, label="IPO offer $185")
    # policy execution markers
    for s, e, w in TRANCHE_SCHEDULE:
        ei = min(e if e is not None else len(bars), len(bars))
        ax.axvspan(t[min(s, len(t) - 1)], t[ei - 1], color="tab:orange", alpha=0.10)
    c15 = min(OBSERVE_MIN, len(bars) - 1)
    ax.scatter([t[c15]], [bars[c15]["c"]], color="tab:red", zorder=5, s=45,
               label=f"policy C: sell-all at cross+15 (${bars[c15]['c']:.0f})")
    if m["vwap_loss_min"] is not None:
        i = m["vwap_loss_min"]
        ax.axvline(t[i], color="tab:purple", ls=":", lw=1.2)
        ax.text(t[i], min(c) * 0.92, f" anchored-VWAP loss\n +{i} min, not reclaimed >=10m",
                fontsize=8, color="tab:purple", va="bottom")
    ax.annotate(f"cross 16:59Z (12:59 ET)\nopen ${m['official_open']:.0f} / 1m close ${m['cross_close']:.0f}",
                xy=(t[0], bars[0]["c"]), xytext=(t[25], bars[0]["c"] - 14), fontsize=8,
                arrowprops=dict(arrowstyle="->"))
    ax.annotate(f"close ${m['close']:.2f}  (fade {m['fade_depth'] * 100:.0f}% off the ${m['high']:.0f} high)",
                xy=(t[-1], c[-1]), xytext=(t[len(t) // 2], min(c) + 4), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="tab:red"), color="tab:red")
    _ann(ax, "Cerebras listing day on the 1-minute tape: price, anchored VWAP, volume, and the three unwind policies",
         "UTC time, 2026-05-14 (cross at 16:59Z; cash close 20:00Z)", "price ($/share)",
         "Sample: Yahoo CBRS 1m, cross->close (n=1 IPO -- the only mega-IPO with surviving intraday). Orange bands = 40/40/20 tranche windows (min 15-60/60-180/180+ since cross).")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%MZ"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_volume_buckets(m: dict, out: Path) -> None:
    ks = [k for k, _ in m["buckets"]]
    vs = [v * 100 for _, v in m["buckets"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([f"{k}-{k + 30}" for k in ks], vs, color="tab:blue", alpha=0.8)
    for i, v in enumerate(vs):
        ax.text(i, v + 0.4, f"{v:.0f}%", ha="center", fontsize=8)
    _ann(ax, "Cerebras day-1: share of total day-1 volume per 30-min bucket since the cross",
         "minutes since cross (16:59Z)", "% of day-1 volume",
         "Sample: Yahoo CBRS 1m, 2026-05-14 cross->close (~181 min session). Read: how fast liquidity dies after the cross -- the unwind must front-load while depth exists.")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_high_close_vs_offer(rows: list[dict], out: Path) -> None:
    """rows = micro_rows (daily OHLC vs offer, official open for every IPO incl. CBRS)."""
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    hi = [r["high_vs_offer_pct"] for r in rows]
    cl = [r["close_vs_offer_pct"] for r in rows]
    op = [r["open_vs_offer_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x - 0.25, op, 0.25, label="official open (the cross print)", color="tab:blue", alpha=0.8)
    ax.bar(x, hi, 0.25, label="day-1 high", color="tab:red", alpha=0.8)
    ax.bar(x + 0.25, cl, 0.25, label="day-1 close", color="tab:gray", alpha=0.8)
    for xi, v in zip(x, hi):
        ax.text(xi, v + 1.5, f"+{v:.0f}%", ha="center", fontsize=8, color="tab:red")
    ax.axhline(0, color="black", lw=0.8)
    _ann(ax, "Day-1 open / high / close vs the IPO offer price -- the measured melt-up distribution",
         "IPO (listing year)", "% above offer price",
         "Sample: 6 mega-IPOs, daily OHLC (CBRS from the 1m tape). The red bars feed S1's --meltup-dist; old assumption was +13/26/39% equal-weighted.")
    ax.set_xticks(x, labels, fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_fade_vs_oversub(rows: list[dict], ipos: list[IPO], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r, ipo in zip(rows, ipos):
        fade = (r["high"] - r["close"]) / r["high"] * 100
        if ipo.oversub is None:
            ax.annotate(f"{ipo.label}: fade {fade:.0f}%, oversub never published (excluded)",
                        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7, color="tab:gray")
            continue
        ax.scatter(ipo.oversub, fade, s=60, color="tab:blue")
        ax.annotate(f" {ipo.label}", (ipo.oversub, fade), fontsize=8)
    ax.axvline(SPCX_OVERSUB, color="tab:green", ls="--", lw=1.2)
    ax.text(SPCX_OVERSUB, ax.get_ylim()[1] * 0.95, " SPCX book ~4x (2026-06-10)",
            color="tab:green", fontsize=8)
    _ann(ax, "Day-1 fade depth (high -> close, % of high) vs order-book oversubscription",
         "oversubscription multiple (book / deal size; sourcing confidence varies -- see note)",
         "fade depth: (high - close) / high, %",
         "Sample: 5 IPOs with a published oversub multiple (RIVN excluded -- never published). n=5: a scatter to eyeball, NOT a regression.")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_policy_comparison(rows: list[dict], out: Path) -> None:
    labels = [f"{r['label']}\n({r['kind']})" for r in rows]
    x = np.arange(len(rows))
    a = [r["A_tranche"] / r["high"] * 100 for r in rows]
    b = [r["B_twap"] / r["high"] * 100 for r in rows]
    c = [r["C_cross15"] / r["high"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.25, a, 0.25, label="A: 40/40/20 tranches", color="tab:orange", alpha=0.85)
    ax.bar(x, b, 0.25, label="B: TWAP from cross", color="tab:blue", alpha=0.85)
    ax.bar(x + 0.25, c, 0.25, label="C: sell-all at cross+15 (daily rows: open PROXY)", color="tab:red", alpha=0.85)
    ax.axhline(100, color="black", lw=0.8, ls=":")
    _ann(ax, "Unwind policy comparison: gross average sell price as % of the day-1 high (higher = better)",
         "IPO (tape kind: real 1m simulation vs daily-OHLC PROXY)", "avg sell price, % of day-1 high",
         "Sample: 6 mega-IPOs. CBRS = real lookahead-free 1m simulation; the rest are daily-OHLC proxies (open~cross+15, OHLC/4~TWAP). Flat slippage shifts all bars equally.")
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(min(min(a), min(b), min(c)) - 5, 104)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    CSV_OUT.mkdir(parents=True, exist_ok=True)

    cbrs = REGISTRY[0]
    m = cbrs_intraday_metrics(cbrs)
    print(f"[cbrs 1m] cross {m['cross_t']:%H:%MZ} open ${m['official_open']:.0f} "
          f"1m-close ${m['cross_close']:.0f} high ${m['high']:.0f} close ${m['close']:.2f}")
    print(f"[cbrs 1m] vw-peak +{m['vw_peak_min']}min (${m['vw_peak_val']:.0f}) | "
          f"fade onset (-5% off running high) +{m['fade_onset_min']}min | "
          f"anchored-VWAP loss +{m['vwap_loss_min']}min | fade depth {m['fade_depth'] * 100:.1f}%")

    # ---- per-IPO microstructure + policy rows
    micro_rows, policy_rows, meltup_moves = [], [], []
    for ipo in REGISTRY:
        d1, d2 = load_daily_day1(ipo)
        meltup_moves.append(d1["h"] / ipo.offer - 1)
        if ipo.intraday:
            pr = policy_row_intraday(ipo, m["bars"])
            micro = {"vw_peak_min": m["vw_peak_min"], "fade_onset_min": m["fade_onset_min"],
                     "vwap_loss_min": m["vwap_loss_min"]}
        else:
            pr = policy_row_daily_proxy(ipo, d1)
            micro = {"vw_peak_min": None, "fade_onset_min": None, "vwap_loss_min": None}
        policy_rows.append(pr)
        micro_rows.append({
            "label": ipo.label, "offer": ipo.offer,
            "open": d1["o"], "high": d1["h"], "low": d1["l"], "close": d1["c"],
            "volume": d1["v"],
            "open_vs_offer_pct": (d1["o"] / ipo.offer - 1) * 100,
            "high_vs_offer_pct": (d1["h"] / ipo.offer - 1) * 100,
            "close_vs_offer_pct": (d1["c"] / ipo.offer - 1) * 100,
            "fade_depth_pct": (d1["h"] - d1["c"]) / d1["h"] * 100,
            "day2_vs_day1_close_pct": ((d2["c"] / d1["c"] - 1) * 100) if d2 else None,
            "oversub": ipo.oversub, "oversub_note": ipo.oversub_note,
            **micro,
        })

    # ---- CSV: microstructure table
    with (CSV_OUT / "ipo_day1_microstructure.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(micro_rows[0].keys()))
        w.writeheader()
        w.writerows(micro_rows)

    # ---- CSV: policy comparison (gross + net at the three slippage levels)
    with (CSV_OUT / "ipo_unwind_policy_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        hdr = ["label", "kind", "offer", "high", "close", "day_vwap",
               "A_tranche_gross", "B_twap_gross", "C_cross15_gross"]
        for s in SLIPPAGE_LEVELS:
            hdr += [f"A_net_{s:.2f}", f"B_net_{s:.2f}", f"C_net_{s:.2f}"]
        hdr += ["A_pct_of_high", "B_pct_of_high", "C_pct_of_high", "best_policy_gross"]
        w.writerow(hdr)
        for r in policy_rows:
            row = [r["label"], r["kind"], r["offer"], round(r["high"], 2), round(r["close"], 2),
                   round(r["day_vwap"], 2),
                   round(r["A_tranche"], 2), round(r["B_twap"], 2), round(r["C_cross15"], 2)]
            for s in SLIPPAGE_LEVELS:
                row += [round(r["A_tranche"] - s, 2), round(r["B_twap"] - s, 2), round(r["C_cross15"] - s, 2)]
            pcts = {k: r[k2] / r["high"] * 100 for k, k2 in
                    [("A", "A_tranche"), ("B", "B_twap"), ("C", "C_cross15")]}
            best = max(pcts, key=pcts.get)
            row += [round(pcts["A"], 1), round(pcts["B"], 1), round(pcts["C"], 1), best]
            w.writerow(row)

    # ---- CSV: volume buckets (CBRS only -- the lone intraday tape)
    with (CSV_OUT / "ipo_day1_volume_buckets_cbrs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket_start_min_since_cross", "share_of_day1_volume"])
        for k, v in m["buckets"]:
            w.writerow([k, round(v, 4)])

    # ---- CSV + stdout: melt-up distribution for S1
    moves_sorted = sorted(meltup_moves)
    dist_str = meltup_dist_string(moves_sorted)
    with (CSV_OUT / "ipo_day1_meltup_dist.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "offer", "day1_high", "high_vs_offer_move"])
        for ipo, mv in zip(REGISTRY, meltup_moves):
            d1, _ = load_daily_day1(ipo)
            w.writerow([ipo.label, ipo.offer, d1["h"], round(mv, 4)])
        w.writerow([])
        w.writerow(["--meltup-dist (equal-weight, HIGH-vs-OFFER moves; conservative vs a live mark above offer)"])
        w.writerow([dist_str])
    e_move = float(np.mean(meltup_moves))
    print(f"\n[meltup] measured day-1 high-vs-offer moves: "
          + ", ".join(f"{ipo.label} +{mv * 100:.0f}%" for ipo, mv in zip(REGISTRY, meltup_moves)))
    print(f"[meltup] E[move] = +{e_move * 100:.1f}% (old +13/26/39% assumption: E = +26.0%)")
    print(f"[meltup] S1 flag:  --meltup-dist {dist_str}")

    # ---- policy summary across the sample (bootstrap = width illustration, n=6)
    print("\n[policies] gross avg sell price as % of day-1 high (CBRS = real 1m sim; others = daily PROXY)")
    for key, name in [("A_tranche", "A 40/40/20"), ("B_twap", "B TWAP-from-cross"), ("C_cross15", "C cross+15")]:
        vals = [r[key] / r["high"] * 100 for r in policy_rows]
        mean, lo, hi = bootstrap_mean_ci(vals)
        print(f"  {name:20s} mean {mean:5.1f}%  boot95 [{lo:.1f}, {hi:.1f}]  per-IPO " +
              " ".join(f"{v:.0f}" for v in vals))

    # ---- charts
    chart_cbrs_day1(m, PLOTS / "ipo_day1_path_cbrs_1m.png")
    chart_volume_buckets(m, PLOTS / "ipo_day1_volume_buckets_cbrs.png")
    chart_high_close_vs_offer(micro_rows, PLOTS / "ipo_day1_high_close_vs_offer.png")
    chart_fade_vs_oversub(policy_rows, REGISTRY, PLOTS / "ipo_fade_vs_oversubscription.png")
    chart_policy_comparison(policy_rows, PLOTS / "ipo_unwind_policy_comparison.png")
    print(f"\n[out] 5 charts -> {PLOTS}")
    print(f"[out] 4 CSVs   -> {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
