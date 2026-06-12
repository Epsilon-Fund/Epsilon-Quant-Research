#!/usr/bin/env python3
"""Block S5i — REHEARSAL tab support: replay a real Nasdaq session through the production
pipeline, pretending the 9:30 ET open print is the IPO cross.

This is a rehearsal HARNESS, not a feature. Invariants:
  * Engine code is REUSED READ-ONLY (DayShapeClassifier, tranche_schedule, hedge_ops_eval,
    render_* — imported, never copied). Nothing here mutates production state.
  * Separate state namespace: rehearsal day-state lives in `rehearsal_state.json`;
    playbook_state.json is never touched; no parquet shards are written anywhere.
  * Level scaling without touching the classifier's constants: the SPCX dollar levels
    ($183/$162/$160/$140/$135/$125) are ratios to the $135 offer. With the simulated
    offer := cross/1.30, mapping the rehearsal price into SPCX-dollar space
    (px_spcx = px · 135/offer) makes the PRODUCTION constants exactly the scaled levels —
    the classifier runs its pre-registered rules verbatim, provably identical.

Tape sources: Alpaca historical bars (the keys already exist for the spot feed) first,
Yahoo 1m fallback, both cached as CSV under ipo_tapes/ next to the CBRS tapes.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.spcx_pm_pdf_monitor import (
    DEFAULT_OUT_DIR,
    DayShapeClassifier,
    IPO_OFFER_DEFAULT,
    PlaybookState,
    SHAPE_ACTIONS,
    _epoch,
)

TAPE_DIR = (Path(__file__).resolve().parents[1] / "data" / "analysis" / "csv_outputs"
            / "market_maps" / "ipo_tapes")
REHEARSAL_STATE_PATH = DEFAULT_OUT_DIR / "rehearsal_state.json"
NY = ZoneInfo("America/New_York")
SIM_POP = 1.30          # assumed +30% pop: simulated offer = cross / 1.30
SIM_PERP_OFFSET = 25.0  # simulated perp proxy = stock price + $25 (labeled SIMULATED)
SIM_FILL = 40.0         # rehearsal position: fill 40, hedge sleeve 18 → residual 22
SIM_HEDGED = 18.0

# every SPCX dollar level the gameplan uses, as a ratio to the $135 offer
SPCX_LEVEL_RATIOS = [
    ("pre-hedge trigger", 183.0), ("EU prospectus ceiling", 162.0),
    ("CRASH floor", 160.0), ("reassess", 140.0), ("offer", 135.0),
    ("sell everything", 125.0),
]


# --------------------------------------------------------------------------------------
# Tape fetch (Alpaca → Yahoo → cache), RTH filter
# --------------------------------------------------------------------------------------
def _cache_path(symbol: str, day: str) -> Path:
    return TAPE_DIR / f"rehearsal_{symbol.lower()}_{day}_1m.csv"


def resolve_session_date(now_utc: float | None = None) -> str:
    """Most recent completed-or-partial US session date (YYYY-MM-DD, ET calendar):
    today if a weekday with the open already past, else walk back to the last weekday."""
    t = datetime.fromtimestamp(now_utc or time.time(), NY)
    d = t.date()
    if t.weekday() >= 5 or (t.hour, t.minute) < (9, 31):
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def fetch_tape_alpaca(symbol: str, day: str, timeout: float = 30.0) -> list[dict]:
    """1m bars for one session from Alpaca's historical API (IEX feed, free tier);
    credentials from the same .env the spot feed uses. Raises on any failure."""
    import os

    import httpx
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key, sec = os.environ.get("ALPACA_KEY_ID"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("no Alpaca credentials")
    bars, token = [], None
    with httpx.Client(timeout=timeout) as client:
        while True:
            params = {"timeframe": "1Min", "start": f"{day}T08:00:00Z",
                      "end": f"{day}T23:59:00Z", "limit": 10000, "feed": "iex",
                      "adjustment": "raw"}
            if token:
                params["page_token"] = token
            r = client.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                           params=params,
                           headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
            r.raise_for_status()
            j = r.json()
            for b in j.get("bars") or []:
                bars.append({"time_utc": b["t"], "open": b["o"], "high": b["h"],
                             "low": b["l"], "close": b["c"], "volume": b["v"]})
            token = j.get("next_page_token")
            if not token:
                break
    if not bars:
        raise RuntimeError(f"Alpaca returned no bars for {symbol} {day}")
    return bars


def fetch_tape_yahoo(symbol: str, day: str, timeout: float = 30.0) -> list[dict]:
    """Yahoo 1m fallback (1m history only exists ~30 days back). Raises on failure."""
    import httpx

    d0 = datetime.fromisoformat(day).replace(tzinfo=NY)
    p1 = int(d0.timestamp())
    p2 = int((d0 + timedelta(days=1)).timestamp())
    r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                  params={"interval": "1m", "period1": p1, "period2": p2},
                  headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    bars = []
    for i, t in enumerate(ts):
        if q["close"][i] is None:
            continue
        bars.append({"time_utc": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                     "open": q["open"][i] or q["close"][i],
                     "high": q["high"][i] or q["close"][i],
                     "low": q["low"][i] or q["close"][i],
                     "close": q["close"][i], "volume": q["volume"][i] or 0})
    if not bars:
        raise RuntimeError(f"Yahoo returned no bars for {symbol} {day}")
    return bars


def load_tape(symbol: str, day: str | None = None) -> tuple[list[dict], str, str]:
    """Cache-first tape load → (bars, session_date, source)."""
    day = day or resolve_session_date()
    cp = _cache_path(symbol, day)
    if cp.exists():
        with cp.open() as f:
            return list(csv.DictReader(f)), day, "cache"
    try:
        bars, source = fetch_tape_alpaca(symbol, day), "alpaca"
    except Exception:
        bars, source = fetch_tape_yahoo(symbol, day), "yahoo"
    TAPE_DIR.mkdir(parents=True, exist_ok=True)
    with cp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time_utc", "open", "high", "low", "close",
                                          "volume"])
        w.writeheader()
        w.writerows(bars)
    return bars, day, source


def rth_bars(bars: list[dict]) -> list[dict]:
    """Regular-session bars only (9:30–16:00 ET) — pre/post-market excluded, so the first
    surviving bar is the simulated cross."""
    out = []
    for b in bars:
        t = datetime.fromtimestamp(_epoch(b["time_utc"]), NY)
        if (9, 30) <= (t.hour, t.minute) < (16, 0):
            out.append(b)
    return out


# --------------------------------------------------------------------------------------
# Replay build — production classifier on SPCX-dollar-mapped prices (lookahead-free)
# --------------------------------------------------------------------------------------
def scale_levels(offer: float) -> list[dict]:
    """SPCX gameplan levels mapped onto the rehearsal symbol via ratio-to-offer."""
    return [{"label": lab, "spcx": lv,
             "scaled": lv / IPO_OFFER_DEFAULT * offer}
            for lab, lv in SPCX_LEVEL_RATIOS]


def build_rehearsal(bars: list[dict], pop: float = SIM_POP,
                    perp_offset: float = SIM_PERP_OFFSET) -> dict:
    """The whole session, precomputed lookahead-free: per-RTH-minute close, AVWAP,
    session high, perp proxy, and the PRODUCTION classifier's state ribbon (stepped
    sequentially — the state at minute t can never see bars after t)."""
    rth = rth_bars(bars)
    if len(rth) < 5:
        raise RuntimeError(f"only {len(rth)} RTH bars — not a replayable session")
    cross_px = float(rth[0]["open"] or rth[0]["close"])
    cross_ts = _epoch(rth[0]["time_utc"])
    offer = cross_px / pop
    to_spcx = IPO_OFFER_DEFAULT / offer    # price mapping into SPCX-dollar space
    cls = DayShapeClassifier()
    pv = v = 0.0
    t_l, c_l, av_l, hi_l, ribbon = [], [], [], [], []
    hi = None
    for b in rth:
        ts = _epoch(b["time_utc"])
        px = float(b["close"])
        vol = float(b["volume"] or 0.0)
        pv += px * vol
        v += vol
        avwap = pv / v if v > 0 else None
        hi = px if hi is None or px > hi else hi
        state = cls.step(ts, px * to_spcx,
                         avwap * to_spcx if avwap is not None else None,
                         cross_px * to_spcx)
        t_l.append(ts)
        c_l.append(px)
        av_l.append(avwap)
        hi_l.append(hi)
        ribbon.append(state)
    return {"symbol": None, "cross_px": cross_px, "cross_ts": cross_ts,
            "offer": offer, "pop": pop, "to_spcx": to_spcx,
            "levels": scale_levels(offer), "perp_offset": perp_offset,
            "t": t_l, "close": c_l, "avwap": av_l, "session_high": hi_l,
            "ribbon": ribbon, "n": len(t_l)}


def rehearsal_pb(reh: dict, sold: float,
                 path: Path | None = REHEARSAL_STATE_PATH) -> PlaybookState:
    """The rehearsal-only day state (separate JSON namespace; never the production file).
    Simulated: fill 40 / hedge sleeve 18 / entry = cross-time perp proxy, 1.5×."""
    pb = PlaybookState(path=path)
    pb.offer = reh["offer"]
    pb.fill = SIM_FILL
    pb.hedged = SIM_HEDGED
    pb.hedge_entry = reh["cross_px"] + reh["perp_offset"]
    pb.hedge_lev = 1.5
    pb.hedge_ts = reh["cross_ts"]
    pb.cross_ts = reh["cross_ts"]
    pb.cross_price = reh["cross_px"]
    pb.sold = sold
    pb.eurusd = 1.08
    return pb


def panel_at(reh: dict, i: int, sold: float, state_path: Path | None = None) -> dict:
    """Everything the rehearsal panels show at bar index i, rendered by the SAME
    production functions (hedge_ops_eval/render_hedge_ops, tranche_schedule/
    render_tranches), fed only data available at-or-before bar i."""
    from scripts.spcx_pm_pdf_monitor import (
        hedge_chart_data, hedge_ops_eval, render_hedge_ops, render_tranches,
        tranche_chart_data, tranche_schedule)

    i = max(0, min(int(i), reh["n"] - 1))
    now = reh["t"][i]
    shape = reh["ribbon"][i]
    pb = rehearsal_pb(reh, sold,
                      path=state_path if state_path is not None
                      else REHEARSAL_STATE_PATH)
    # synthetic poll history up to the playhead (for the gap chip + fast-tape inputs)
    history = [{"ts": reh["t"][j], "spot": reh["close"][j],
                "perp": reh["close"][j] + reh["perp_offset"]}
               for j in range(max(0, i - 90), i + 1)]
    ops = hedge_ops_eval(pb, mark=reh["close"][i] + reh["perp_offset"],
                         spot=reh["close"][i], funding_hourly=0.0, max_leverage=5.0,
                         history=history, now=now)
    tr = tranche_schedule(pb.fill, pb.hedged, sold, pb.cross_ts, now, shape=shape,
                          session_high=reh["session_high"][i])
    mins_in = 0.0
    for j in range(i, -1, -1):
        if reh["ribbon"][j] != shape:
            break
        mins_in = (now - reh["t"][j]) / 60.0
    # S5j charts — same helpers as the live payload, fed the replayed values
    ops_green = ops["pair_close"]["green"] if ops else False
    return {"i": i, "ts": now, "shape": shape, "action": SHAPE_ACTIONS.get(shape, ""),
            "minutes_in_state": mins_in,
            "tranche_chart": tranche_chart_data(tr, pb.cross_ts, now),
            "hedge_chart": hedge_chart_data(ops, history, pb.cross_ts, now,
                                            green_since=now if ops_green else None),
            "hedge_ops_html": ("<div class='simnote'>SIMULATED hedge: entry = cross-time "
                               "perp proxy (stock + $" f"{reh['perp_offset']:.0f}), 18 sh"
                               " @ 1.5× — rehearsal only</div>" + render_hedge_ops(ops)),
            "tranche": tr, "tranche_html": render_tranches(tr),
            "spot": reh["close"][i], "perp_sim": reh["close"][i] + reh["perp_offset"],
            "avwap": reh["avwap"][i], "session_high": reh["session_high"][i]}
