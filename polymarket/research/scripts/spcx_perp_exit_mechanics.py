"""Block S6 -- Perp exit mechanics on the Cerebras precedent: pair-close window + backstop cost.

WHY: the SPCX gameplan (spcx_listing_day_gameplan.md section 5.1) closes the hedge-sleeve short
via a simultaneous pair-close after the IPO cross, with asserted-not-measured numbers: tracking
"within ~$1-4" (from daily closes), readiness thresholds |gap|<=$2 for 15min / >=60min post-cross,
and a 21:00 CET forced-flat backstop. This script measures the actual minute-path of
gap = perp - spot on the one real precedent (Cerebras, 2026-05-14) plus realized HL funding
across the IPOP -> equity-perp conversion.

DATA:
  - perp: cached HL xyz:CBRS 15m candles (csv_outputs/market_maps/cerebras_cbrs_perp_15m.csv,
    written by spcx_cerebras_case_study.py). 15m is the finest HL retains for the listing window
    -> every gap number carries a +/-(intra-15m drift) caveat, stated on the chart.
  - spot: cached Yahoo CBRS 1m listing-day tape (ipo_tapes/, written by the S2 study) + Yahoo 5m
    for the +1/+2 sessions (fetched once here, cached to ipo_tapes/cbrs_spot_5m_postlisting.csv).
  - funding: HL fundingHistory for xyz:CBRS 05-12 -> 05-20 (hourly; retained), cached.

LOOKAHEAD: the buy-back ledger entry is the close of the 15m perp bar CONTAINING the cross
(16:45Z bar, closing ~17:00Z) -- the first perp print a short opened "at the cross" could
realistically reference; every exit row uses the bar at-or-before its target time. Gap stats
used as decision calibration (first-sustained-inside thresholds) scan forward in time only.

RUN: cd polymarket/research && PYTHONPATH=. uv run python scripts/spcx_perp_exit_mechanics.py
OUT: CSVs -> data/analysis/csv_outputs/market_maps/   PNGs -> data/analysis/plots/spcx_convergence/
"""
from __future__ import annotations

import csv
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_OUT = ROOT / "data" / "analysis" / "csv_outputs" / "market_maps"
TAPE_CACHE = CSV_OUT / "ipo_tapes"
PLOTS = ROOT / "data" / "analysis" / "plots" / "spcx_convergence"

HL_INFO = "https://api.hyperliquid.xyz/info"
CROSS_UTC = datetime(2026, 5, 14, 16, 59, tzinfo=timezone.utc)  # CBRS first cash trade (12:59 ET)
SESSIONS = [  # (label, RTH open, RTH close) UTC; day-1 spot exists only from the cross
    ("day 1 (listing)", datetime(2026, 5, 14, 13, 30, tzinfo=timezone.utc), datetime(2026, 5, 14, 20, 0, tzinfo=timezone.utc)),
    ("day +1", datetime(2026, 5, 15, 13, 30, tzinfo=timezone.utc), datetime(2026, 5, 15, 20, 0, tzinfo=timezone.utc)),
    ("day +2 (Mon)", datetime(2026, 5, 18, 13, 30, tzinfo=timezone.utc), datetime(2026, 5, 18, 20, 0, tzinfo=timezone.utc)),
]


# ---------------------------------------------------------------- loaders
def read_csv_bars(path: Path, tcol: str = "time_utc", ccol: str = "close",
                  shift: timedelta = timedelta(0)) -> list[tuple[datetime, float]]:
    """Bars as (time, close). Both HL and Yahoo stamp bars with their OPEN time, but a bar's
    close trades at open + bar-length -- pass `shift` = bar length so (t, close) means "the
    price AT t". Without this, comparing a 15m perp close against a 1m spot close at the same
    stamp misaligns them by up to 14 minutes (= tens of $ on the listing-day tape)."""
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            out.append((datetime.fromisoformat(row[tcol]) + shift, float(row[ccol])))
    return out


def yahoo_5m_postlisting() -> list[tuple[datetime, float]]:
    """Cache-first CBRS spot 5m, 05-14 -> 05-19 (covers day-1 + the next two sessions).
    Cache keeps raw Yahoo open-time stamps; callers apply the close-time shift."""
    p = TAPE_CACHE / "cbrs_spot_5m_postlisting.csv"
    if p.exists():
        return read_csv_bars(p, shift=timedelta(minutes=5))
    p1 = int(datetime(2026, 5, 14, tzinfo=timezone.utc).timestamp())
    p2 = int(datetime(2026, 5, 19, tzinfo=timezone.utc).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/CBRS?period1={p1}&period2={p2}&interval=5m"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        d = json.load(resp)
    res = d["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    bars = []
    for i, sec in enumerate(res["timestamp"]):
        if q["close"][i] is None:
            continue
        t = datetime.fromtimestamp(sec, timezone.utc)
        if t > datetime(2026, 5, 19, tzinfo=timezone.utc):  # Yahoo appends a current-quote bar
            continue
        bars.append((t, float(q["close"][i])))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_utc", "close"])
        for t, c in bars:
            w.writerow([t.isoformat(), c])
    return [(t + timedelta(minutes=5), c) for t, c in bars]


def hl_funding_history() -> list[dict]:
    """Cache-first HL hourly funding for xyz:CBRS, 05-12 -> 05-20 (spans the conversion)."""
    p = TAPE_CACHE / "cbrs_funding_hourly.csv"
    if p.exists():
        out = []
        with p.open() as f:
            for row in csv.DictReader(f):
                out.append({"t": datetime.fromisoformat(row["time_utc"]),
                            "rate": float(row["funding_rate"]), "premium": float(row["premium"])})
        return out
    req = {"type": "fundingHistory", "coin": "xyz:CBRS",
           "startTime": int(datetime(2026, 5, 12, tzinfo=timezone.utc).timestamp() * 1000),
           "endTime": int(datetime(2026, 5, 20, tzinfo=timezone.utc).timestamp() * 1000)}
    r = urllib.request.Request(HL_INFO, data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        rows = json.load(resp)
    out = [{"t": datetime.fromtimestamp(x["time"] / 1000, timezone.utc),
            "rate": float(x["fundingRate"]), "premium": float(x["premium"])} for x in rows]
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_utc", "funding_rate", "premium"])
        for x in out:
            w.writerow([x["t"].isoformat(), x["rate"], x["premium"]])
    return out


# ---------------------------------------------------------------- pure helpers (unit-tested)
def at_or_before(series: list[tuple[datetime, float]], target: datetime,
                 max_stale: timedelta | None = None) -> tuple[datetime, float] | tuple[None, None]:
    """Latest (t, value) with t <= target; (None, None) if absent or staler than max_stale.
    Lookahead-free by construction."""
    pick = None
    for t, v in series:
        if t <= target:
            pick = (t, v)
        else:
            break
    if pick is None:
        return (None, None)
    if max_stale is not None and target - pick[0] > max_stale:
        return (None, None)
    return pick


def build_gap_series(perp: list[tuple[datetime, float]], spot: list[tuple[datetime, float]],
                     sessions: list[tuple[str, datetime, datetime]],
                     spot_start: datetime) -> list[dict]:
    """gap = perp - spot at each 15m perp mark inside a cash session (spot at-or-before the
    mark, <=15min stale). Day-1 marks before the cross are excluded (no real spot)."""
    out = []
    for label, o, c in sessions:
        for t, pv in perp:
            if not (o <= t <= c) or t < spot_start:
                continue
            st, sv = at_or_before(spot, t, max_stale=timedelta(minutes=15))
            if sv is None:
                continue
            out.append({"t": t, "session": label, "perp": pv, "spot": sv, "gap": pv - sv})
    return out


def first_sustained_inside(gaps: list[dict], thresh: float, n_consec: int = 2) -> datetime | None:
    """First mark where |gap| <= thresh and stays inside for n_consec consecutive marks
    (15m marks -> n_consec=2 means ~30 min inside). Forward scan only."""
    run = 0
    for i, g in enumerate(gaps):
        if abs(g["gap"]) <= thresh:
            run += 1
            if run >= n_consec:
                return gaps[i - n_consec + 1]["t"]
        else:
            run = 0
    return None


def short_funding_pnl(funding: list[dict], start: datetime, end: datetime,
                      price_ref: float) -> float:
    """Realized funding $/share for a SHORT over [start, end): HL longs pay shorts when
    rate > 0, so short P&L = +sum(rate) * price_ref (price_ref = approx mark over the window)."""
    return sum(f["rate"] for f in funding if start <= f["t"] < end) * price_ref


# ---------------------------------------------------------------- charts
def chart_gap_path(gaps: list[dict], out: Path) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True, height_ratios=[2, 1])
    colors = {"day 1 (listing)": "tab:red", "day +1": "tab:blue", "day +2 (Mon)": "tab:green"}
    for label in colors:
        seg = [g for g in gaps if g["session"] == label]
        if not seg:
            continue
        t = [g["t"] for g in seg]
        ax.plot(t, [g["perp"] for g in seg], color="black", lw=1.3,
                label="xyz:CBRS perp (HL 15m close)" if label == "day 1 (listing)" else None)
        ax.plot(t, [g["spot"] for g in seg], color=colors[label], lw=1.3, ls="--",
                label=f"CBRS spot, {label}")
        ax2.plot(t, [g["gap"] for g in seg], color=colors[label], lw=1.5, marker=".", ms=4)
    for _, o, c in SESSIONS:
        for axx in (ax, ax2):
            axx.axvline(c, color="tab:gray", ls=":", lw=1.0)
    ax2.axhline(0, color="black", lw=0.8)
    for y in (2, -2):
        ax2.axhline(y, color="tab:orange", ls="--", lw=1.0)
    for y in (1, -1):
        ax2.axhline(y, color="tab:green", ls="--", lw=0.9)
    ax2.text(gaps[0]["t"], 2.2, "gameplan readiness band |gap|<=$2 (orange) / $1 (green)",
             fontsize=8, color="tab:orange")
    ax.axvline(CROSS_UTC, color="tab:purple", ls="-", lw=1.2)
    ax.text(CROSS_UTC, ax.get_ylim()[0], " cross 16:59Z", color="tab:purple", fontsize=8, va="bottom")
    ax.set_title("Cerebras: perp vs spot and the gap (perp - spot), listing day + 2 sessions\n"
                 "Sample: HL xyz:CBRS 15m closes vs Yahoo spot (1m day-1, 5m after), cash sessions only; "
                 "gaps measured at 15m marks.\nCAVEAT: 15m perp granularity -> each gap point is "
                 "+/-(intra-15m drift), a few $ in the first post-cross hour. n=1 IPO.", fontsize=9.5)
    ax.set_ylabel("price ($/share)")
    ax2.set_ylabel("gap = perp - spot ($)")
    ax2.set_xlabel("UTC (x-axis compresses overnight/weekend gaps between sessions; dotted verticals = cash closes)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax2.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%MZ"))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_funding(funding: list[dict], out: Path) -> None:
    t = [f["t"] for f in funding]
    bps = [f["rate"] * 1e4 for f in funding]
    fig, ax = plt.subplots(figsize=(11.5, 5))
    ax.bar(t, bps, width=1 / 24, color=["tab:green" if b >= 0 else "tab:red" for b in bps])
    ax.axvline(CROSS_UTC, color="tab:purple", ls="-", lw=1.4)
    ax.text(CROSS_UTC, max(bps) * 0.95, " listing cross / IPOP->equity conversion 05-14 ~17:00Z",
            color="tab:purple", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("xyz:CBRS hourly funding rate through the IPOP -> equity-perp conversion\n"
                 "Sample: HL fundingHistory, 2026-05-12 -> 05-19 (hourly). Green = longs pay shorts "
                 "(a short EARNS); red = short pays.\nRead: whether the conversion changed the funding regime, "
                 "and what a short held across it actually paid/earned. n=1 IPO.", fontsize=9.5)
    ax.set_ylabel("funding rate (bps/hour)")
    ax.set_xlabel("UTC")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> int:
    perp = read_csv_bars(CSV_OUT / "cerebras_cbrs_perp_15m.csv", shift=timedelta(minutes=15))
    spot_1m = read_csv_bars(TAPE_CACHE / "cbrs_spot_1m_listingday.csv", shift=timedelta(minutes=1))
    spot_5m = yahoo_5m_postlisting()
    funding = hl_funding_history()

    # spot series: real prints only -- day 1 from the cross (1m tape already starts pre-open
    # with placeholders at $185; S2's cache kept all bars, so filter), then 5m for +1/+2
    spot_d1 = [(t, c) for t, c in spot_1m if t >= CROSS_UTC and c > 185 * 1.2]
    spot_later = [(t, c) for t, c in spot_5m if t >= datetime(2026, 5, 15, tzinfo=timezone.utc)]
    spot = sorted(spot_d1 + spot_later)

    gaps = build_gap_series(perp, spot, SESSIONS, spot_start=CROSS_UTC)
    d1 = [g for g in gaps if g["session"] == "day 1 (listing)"]

    # ---- (b) compression + end-of-session stats
    in2 = first_sustained_inside(d1, 2.0)
    in1 = first_sustained_inside(d1, 1.0)
    print("[gap] day-1 marks:", ", ".join(f"{g['t']:%H:%M}Z {g['gap']:+.1f}" for g in d1))
    for label, ts in [("|gap|<=$2 sustained (2x15m)", in2), ("|gap|<=$1 sustained (2x15m)", in1)]:
        msg = f"{ts:%H:%M}Z (+{(ts - CROSS_UTC).total_seconds() / 60:.0f} min post-cross)" if ts else "NEVER on day 1"
        print(f"[gap] first {label}: {msg}")
    for label, _, c in SESSIONS:
        seg = [g for g in gaps if g["session"] == label]
        if not seg:
            continue
        final_hr = [g for g in seg if g["t"] >= c - timedelta(hours=1)]
        worst = max((abs(g["gap"]) for g in final_hr), default=float("nan"))
        last = seg[-1]
        print(f"[gap] {label}: marks={len(seg)} worst-final-hour |gap|=${worst:.2f} "
              f"last mark {last['t']:%H:%M}Z gap {last['gap']:+.2f}")

    # ---- (c) buy-back ledger for a short entered at the cross-bar perp close
    et, entry = at_or_before(perp, CROSS_UTC + timedelta(minutes=15))  # bar containing the cross
    print(f"\n[ledger] short entry = perp 15m close at {et:%H:%M}Z = ${entry:.2f} (bar containing the 16:59Z cross)")
    targets = [
        ("cross+1h", CROSS_UTC + timedelta(hours=1)),
        ("cross+2h", CROSS_UTC + timedelta(hours=2)),
        ("cross+3h (=cash close)", datetime(2026, 5, 14, 20, 0, tzinfo=timezone.utc)),
        ("next-day open+15m", datetime(2026, 5, 15, 13, 45, tzinfo=timezone.utc)),
        ("next-day close", datetime(2026, 5, 15, 20, 0, tzinfo=timezone.utc)),
        ("+2d close (Mon)", datetime(2026, 5, 18, 20, 0, tzinfo=timezone.utc)),
    ]
    ledger = []
    for name, tt in targets:
        # evaluate BOTH legs at the same 15m perp mark (mixing a stale perp close with a
        # fresher spot print manufactures fake $20+ "gaps" in the fast first hour)
        pt, pv = at_or_before(perp, tt)
        _, sv = at_or_before(spot, pt, max_stale=timedelta(minutes=20))
        gap = (pv - sv) if (pv is not None and sv is not None) else None
        fund = short_funding_pnl(funding, et, tt, entry)
        pnl = entry - pv
        ledger.append({"exit": name, "t_utc": tt, "perp": pv, "spot": sv, "gap": gap,
                       "short_pnl_per_sh": pnl, "funding_pnl_per_sh": fund,
                       "pair_close_drag": gap})
        gs = f"{gap:+.2f}" if gap is not None else "n/a"
        print(f"  {name:22s} perp {pv:7.2f} spot {sv if sv else float('nan'):7.2f} gap {gs:>7} "
              f"short-pnl {pnl:+7.2f} funding {fund:+5.2f} $/sh")

    # ---- (d) funding regime around conversion
    pre = [f for f in funding if f["t"] < CROSS_UTC]
    post = [f for f in funding if f["t"] >= CROSS_UTC]
    for label, seg in [("pre-conversion (05-12 -> cross)", pre), ("post-conversion (cross -> 05-19)", post)]:
        rates = [f["rate"] * 1e4 for f in seg]
        prem = [abs(f["premium"]) * 1e4 for f in seg]
        print(f"[funding] {label}: n={len(seg)} mean {np.mean(rates):+.3f} bps/h "
              f"median |premium| {np.median(prem):.1f} bps "
              f"share positive (short earns) {np.mean([r > 0 for r in rates]) * 100:.0f}%")
    d1_fund = short_funding_pnl(funding, et, datetime(2026, 5, 14, 20, 0, tzinfo=timezone.utc), entry)
    print(f"[funding] short held cross -> cash close: realized funding {d1_fund:+.3f} $/sh")

    # ---- CSVs
    with (CSV_OUT / "cbrs_perp_spot_gap_path.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_utc", "session", "perp_15m_close", "spot_close", "gap_usd"])
        for g in gaps:
            w.writerow([g["t"].isoformat(), g["session"], g["perp"], g["spot"], round(g["gap"], 3)])
    with (CSV_OUT / "cbrs_short_buyback_ledger.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exit", "time_utc", "perp", "spot", "gap_usd", "short_pnl_per_sh",
                    "funding_pnl_per_sh", "pair_close_drag_usd"])
        for r in ledger:
            w.writerow([r["exit"], r["t_utc"].isoformat(), r["perp"], r["spot"],
                        None if r["gap"] is None else round(r["gap"], 3),
                        round(r["short_pnl_per_sh"], 3), round(r["funding_pnl_per_sh"], 4),
                        None if r["pair_close_drag"] is None else round(r["pair_close_drag"], 3)])

    chart_gap_path(gaps, PLOTS / "cbrs_perp_spot_gap_path.png")
    chart_funding([f for f in funding if f["t"] <= datetime(2026, 5, 19, tzinfo=timezone.utc)],
                  PLOTS / "cbrs_funding_through_conversion.png")
    print(f"\n[out] 2 charts -> {PLOTS}\n[out] 2 CSVs + funding cache -> {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
