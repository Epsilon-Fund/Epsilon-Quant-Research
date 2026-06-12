"""Builder for notebooks/ipo_tail_repricing_study.ipynb.

WHY a builder: same convention as scripts/build_weather_notebook.py — the notebook is a
generated artifact, so the source of truth is this script. Edit here and re-run, do not
hand-edit the .ipynb.

WHAT the notebook does: studies how the *upper/lower-tail* YES tokens repriced during the two
previously-resolved Polymarket IPO market-cap ladders — Cerebras (Nasdaq:CBRS, 2026-05-14) and
StubHub (NYSE:STUB, 2025-09-17) — to calibrate the SpaceX (SPCX) tail-selling plan for the
2026-06-12 listing.

DATA (all live-fetched, then cached to data/analysis/csv_outputs/market_maps/ipo_tapes/ so the
notebook is offline-reproducible after the first run):
  - Cerebras PM YES tokens : Polymarket CLOB /prices-history (still served for CBRS).
  - Cerebras stock         : Yahoo chart API 5m (1m is purged after ~30d; 5m is the finest left).
  - StubHub PM YES tokens  : Path A = local DuckDB view layer (raw_trades) if the parquet is
                             present; Path B = live Goldsky subgraph (same source the parquet is
                             built from). CLOB /prices-history is purged for StubHub.
  - StubHub stock          : yfinance daily OHLC (1m purged for a 2025 listing).

BUILD:  cd polymarket/research && PYTHONPATH=. python scripts/build_ipo_tail_repricing_notebook.py
RUN  :  cd polymarket/research && PYTHONPATH=. uv run jupyter lab notebooks/ipo_tail_repricing_study.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "ipo_tail_repricing_study.ipynb"

cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# =====================================================================================
# TITLE + PLAIN-ENGLISH SUMMARY
# =====================================================================================
md(r"""
# IPO Tail-Token Repricing Study — selling the long-shot YES strikes into the listing-day pop

**Plain-English headline:** SpaceX (SPCX) lists today (2026-06-12). The plan is to *sell* the
upper-tail YES tokens on Polymarket — the `>$2.8T / >$3.0T / >$3.2T / ...` market-cap strikes,
currently 1–9¢ — into the euphoria of the first trading pop, because those tails are long shots
that almost certainly resolve to **0**. This notebook studies the only two precedents we have —
the **Cerebras (CBRS)** and **StubHub (STUB)** Polymarket market-cap ladders — to learn *when* the
tail tokens were richest and *how fast* they collapsed, so we can time the SPCX sell.

## Plain-English Summary

- **What this is:** an empirical study of how the tail YES tokens on two resolved Polymarket
  IPO market-cap ladders (Cerebras, StubHub) repriced around their listing crosses, overlaid on
  the underlying stock price.
- **Why it was written:** to calibrate the SPCX tail-sell trade *before* the listing today.
  We are selling tokens that pay 0 at resolution, so the only question is **timing the exit** —
  what price the tails reach and for how long.
- **What it covers:** Cerebras PM token prices (CLOB), Cerebras stock (Yahoo 5m), StubHub PM
  token prices (on-chain Goldsky fills), StubHub stock (yfinance daily). All cached locally.
- **One-line takeaway (filled in by Cell 6 after the data loads):** the tails are richest in the
  hours *around and just before* the cross and decay quickly once the stock prints — so the sell
  window is narrow and front-loaded, not a patient hold.
- **Calibration caveat:** this is **n = 2** (really n ≈ 1.x — StubHub was a tiny ~$119K-volume
  market). The value is *intuition*, not statistical proof. Stated plainly in the synthesis.

> Hubs / related: [[spcx_listing_day_gameplan]] · [[spcx_ipo_unwind_tape_findings]] ·
> [[spacex_ipo_market_map_handoff]] · [[POLYMARKET_BRAIN]] · [[TODO]] § SPCX.
> Table terms: [[polymarket_table_dictionary]].

**Conventions:** all timestamps are **UTC**. Token prices are the YES mid in `[0, 1]`, shown in
**cents (¢ = price × 100)**. "Tail" = any strike priced **< 15¢ at market creation** (a long-shot
leg). Times are also expressed as **minutes since the listing cross** (the first stock print).
""")


# =====================================================================================
# CELL 1 — IMPORTS, CONFIG, DATA LOADING
# =====================================================================================
md(r"""
## Cell 1 — Imports, configuration, and data loading

This cell fetches **all** historical data for both IPOs and caches it to CSV so every later cell
(and every rerun) works offline. Four sources, each with a documented fallback:

| Series | Primary source | Fallback / note |
|---|---|---|
| Cerebras PM YES tokens | Polymarket CLOB `/prices-history` (`fidelity=1`, ~10-min bars) | still served for CBRS |
| Cerebras stock (CBRS) | Yahoo chart API, 5-minute bars | 1-min purged after ~30 days; 5-min is the finest left |
| StubHub PM YES tokens | **Path A:** local DuckDB `raw_trades` view; **Path B:** live Goldsky subgraph | CLOB is purged for StubHub; Path B is the *same* on-chain source the parquet is built from |
| StubHub stock (STUB) | yfinance **daily** OHLC | 1-min purged for a 2025 listing — only daily is available |

**On-chain price reconstruction (StubHub):** each `OrderFilled` event swaps USDC (asset id `"0"`,
6 decimals) for an outcome token (6 decimals). The fill price is simply `usd_amount / token_amount`
— the 6-decimal scaling cancels, leaving the YES mid in `[0, 1]`, the same semantics as the CLOB
mid. We build a per-strike last-trade price series in 10-minute windows.
""")

code(r'''
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.rcParams["text.parse_math"] = False  # render literal "$" (no mathtext)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("dark_background")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

UTC = timezone.utc
CACHE = Path("data/analysis/csv_outputs/market_maps/ipo_tapes")
CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, ".")  # so `import data_infra...` works from polymarket/research

# ---- Listing anchors (UTC). Cross = first regular-session stock print. ------------------
CBRS_CROSS = datetime(2026, 5, 14, 16, 59, tzinfo=UTC)   # 12:59 ET first trade (IPOScoop)
CBRS_OFFER = 185.0
CBRS_CLOSE_D1 = 311.07                                    # +68% vs offer
STUB_LISTING = datetime(2025, 9, 17, tzinfo=UTC)
# STUB intraday stock is purged -> the cross clock is approximate (labelled everywhere it is used).
STUB_CROSS_APPROX = datetime(2025, 9, 17, 17, 47, tzinfo=UTC)  # ~13:47 ET typical IPO cross
STUB_OFFER = 23.50   # StubHub priced at $23.50 (re-checked at run time against yfinance open)

GOLDSKY_URL = ("https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
               "/subgraphs/orderbook-subgraph/0.0.1/gn")

# ---- Cerebras YES token ids (one per strike). Winner = $50B+. ---------------------------
CBRS_TOKENS = {
    "<$20B":   "107145636477748599950685161449590141102597225144882848848176572972394807617808",
    "$20-30B": "33185816580070359531670887516327538876057633561223660613820062172367465351140",
    "$30-40B": "16324658415267021790471773791448399620635861424508421154984779721340745544115",
    "$40-50B": "692008562859574366373167471424306671270711791921101151861137181794215138833",
    "$50B+":   "30285895310864329712474926299854946979606268381032251018339749396129997815927",
}
CBRS_WINNER = "$50B+"

# ---- HTTP helpers (httpx with a browser UA + small retry; transient 403/429/5xx from
#      Yahoo's query1 host are common, so retry with backoff rather than abort the run) ---
import httpx

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def _retry(fn, tries: int = 4):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (i + 1))
    raise last

def _get_json(url: str, timeout: int = 30) -> dict:
    def go():
        r = httpx.get(url, headers=_UA, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        return r.json()
    return _retry(go)

def _post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    def go():
        r = httpx.post(url, json=payload, headers=_UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
    return _retry(go)


# ---- (1) Cerebras PM tokens via CLOB /prices-history ----------------------------------
def fetch_clob_history(token_id: str) -> pd.DataFrame:
    """YES-mid history for a CLOB token. Returns DataFrame[t (UTC), p (0-1)]."""
    url = (f"https://clob.polymarket.com/prices-history?market={token_id}"
           f"&interval=all&fidelity=1")
    hist = _get_json(url).get("history", [])
    if not hist:
        return pd.DataFrame(columns=["t", "p"])
    df = pd.DataFrame(hist)
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df[["t", "p"]].sort_values("t").reset_index(drop=True)

def load_cbrs_pm() -> dict[str, pd.DataFrame]:
    cache = CACHE / "cbrs_pm_tokens_10m.csv"
    if cache.exists():
        raw = pd.read_csv(cache, parse_dates=["t"])
        print(f"[cbrs pm] loaded {len(raw)} rows from cache {cache.name}")
        return {s: g[["t", "p"]].reset_index(drop=True) for s, g in raw.groupby("strike")}
    out, frames = {}, []
    for strike, tok in CBRS_TOKENS.items():
        d = fetch_clob_history(tok)
        out[strike] = d
        f = d.copy(); f["strike"] = strike; frames.append(f)
        print(f"[cbrs pm] {strike:8} {len(d):4} points")
        time.sleep(0.2)
    pd.concat(frames, ignore_index=True).to_csv(cache, index=False)
    print(f"[cbrs pm] cached -> {cache.name}")
    return out


# ---- (2) Cerebras stock via Yahoo chart API (5m) --------------------------------------
def fetch_yahoo_chart(symbol: str, query: str) -> pd.DataFrame:
    """OHLC from the Yahoo chart API. Returns DataFrame[t (UTC), open, high, low, close]."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    res = _get_json(url)["chart"]["result"][0]
    ts = res.get("timestamp", []) or []
    q = res["indicators"]["quote"][0]
    rows = []
    for i, sec in enumerate(ts):
        if q["close"][i] is None:
            continue
        rows.append((datetime.fromtimestamp(sec, UTC), q["open"][i], q["high"][i],
                     q["low"][i], q["close"][i]))
    return pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"])

def load_cbrs_spot() -> pd.DataFrame:
    cache = CACHE / "cbrs_spot_5m_listingday.csv"
    if cache.exists():
        print(f"[cbrs spot] loaded from cache {cache.name}")
        return pd.read_csv(cache, parse_dates=["t"])
    try:
        df = fetch_yahoo_chart("CBRS", "range=1mo&interval=5m")
    except Exception as exc:  # noqa: BLE001
        print(f"[cbrs spot] Yahoo fetch failed ({exc}); trying yfinance")
        import yfinance as yf
        y = yf.download("CBRS", start="2026-05-14", end="2026-05-16", interval="5m",
                        progress=False, auto_adjust=False)
        y = y.reset_index().rename(columns={"Datetime": "t", "Open": "open", "High": "high",
                                            "Low": "low", "Close": "close"})
        df = y[["t", "open", "high", "low", "close"]]
        df["t"] = pd.to_datetime(df["t"], utc=True)
    # keep the listing day, drop the pre-open $185 placeholder bars (first real trade gaps up huge)
    day = df[df["t"].dt.date == CBRS_CROSS.date()].copy()
    started = day["close"] >= CBRS_OFFER * 1.4
    if started.any():
        day = day[started.cummax()]
    day.to_csv(cache, index=False)
    print(f"[cbrs spot] {len(day)} 5m bars on listing day; cached -> {cache.name}")
    return day


# ---- (3) StubHub event metadata + PM tokens (Path A: DuckDB, Path B: Goldsky) ----------
def stubhub_markets() -> list[dict]:
    """Gamma metadata: one dict per strike with label, condition_id, yes_token_id."""
    url = "https://gamma-api.polymarket.com/events?slug=stubhub-ipo-closing-market-cap"
    ev = _get_json(url)[0]
    out = []
    for m in ev["markets"]:
        ids = json.loads(m["clobTokenIds"])
        out.append({"label": m.get("groupItemTitle"), "condition_id": m.get("conditionId"),
                    "yes_token_id": ids[0]})
    return out

def _goldsky_fills_for_token(token_id: str) -> pd.DataFrame:
    """Paginate both maker/taker legs (cap 1000/page), reconstruct price = usd/token."""
    seen: dict[str, dict] = {}
    for side in ("makerAssetId", "takerAssetId"):
        last = 0
        while True:
            q = ('{ orderFilledEvents(first:1000, orderBy: timestamp, orderDirection: asc, '
                 'where:{ %s:"%s", timestamp_gt:"%d"}){ id timestamp makerAmountFilled '
                 'takerAmountFilled makerAssetId } }' % (side, token_id, last))
            d = _post_json(GOLDSKY_URL, {"query": q})
            page = d.get("data", {}).get("orderFilledEvents", [])
            if not page:
                break
            for r in page:
                seen[r["id"]] = r
            if len(page) < 1000:
                break
            last = int(page[-1]["timestamp"])
    recs = []
    for r in seen.values():
        if r["makerAssetId"] == "0":            # maker paid USDC -> bought token
            usd, tok = int(r["makerAmountFilled"]), int(r["takerAmountFilled"])
        else:                                   # maker sold token -> received USDC
            tok, usd = int(r["makerAmountFilled"]), int(r["takerAmountFilled"])
        if tok == 0:
            continue
        recs.append((datetime.fromtimestamp(int(r["timestamp"]), UTC), usd / tok, usd / 1e6))
    df = pd.DataFrame(recs, columns=["t", "p", "usd"]).sort_values("t").reset_index(drop=True)
    return df

def _stubhub_pm_via_duckdb(markets: list[dict]) -> dict[str, pd.DataFrame] | None:
    """Path A: query the local DuckDB raw_trades view. Returns None if unavailable
    (duckdb missing, no local parquet, or no markets snapshot)."""
    try:
        from data_infra.duck import connect
        from data_infra.views import load_views
        con = connect()
        load_views(con)  # raises SystemExit if no markets_*.parquet snapshot on disk
        cond_ids = [m["condition_id"] for m in markets]
        ph = ",".join("?" * len(cond_ids))
        df = con.execute(
            f"SELECT timestamp, condition_id, price, usd_amount "
            f"FROM raw_trades WHERE condition_id IN ({ph}) ORDER BY timestamp", cond_ids
        ).fetchdf()
        if df.empty:
            print("[stubhub pm] Path A (DuckDB) returned 0 rows -> falling back to Goldsky")
            return None
        df["t"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        by_cond = {m["condition_id"]: m["label"] for m in markets}
        out = {}
        for cid, g in df.groupby("condition_id"):
            out[by_cond[cid]] = g.rename(columns={"price": "p"})[["t", "p", "usd_amount"]] \
                .rename(columns={"usd_amount": "usd"}).reset_index(drop=True)
        print(f"[stubhub pm] Path A (DuckDB) OK: {len(df)} fills across {len(out)} strikes")
        return out
    except SystemExit as exc:
        print(f"[stubhub pm] Path A unavailable (no local data: {exc}) -> Goldsky")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[stubhub pm] Path A unavailable ({type(exc).__name__}: {exc}) -> Goldsky")
        return None

def load_stubhub_pm(markets: list[dict]) -> dict[str, pd.DataFrame]:
    cache = CACHE / "stub_pm_trades.csv"
    if cache.exists():
        raw = pd.read_csv(cache, parse_dates=["t"])
        print(f"[stubhub pm] loaded {len(raw)} fills from cache {cache.name}")
        return {s: g[["t", "p", "usd"]].reset_index(drop=True) for s, g in raw.groupby("strike")}
    out = _stubhub_pm_via_duckdb(markets)
    if out is None:  # Path B: live Goldsky
        out = {}
        for m in markets:
            d = _goldsky_fills_for_token(m["yes_token_id"])
            out[m["label"]] = d
            print(f"[stubhub pm] Goldsky {m['label']:8} {len(d):4} fills")
        print("[stubhub pm] reconstructed from live on-chain Goldsky fills")
    frames = []
    for strike, d in out.items():
        f = d.copy(); f["strike"] = strike
        if "usd" not in f:
            f["usd"] = np.nan
        frames.append(f[["t", "p", "usd", "strike"]])
    pd.concat(frames, ignore_index=True).to_csv(cache, index=False)
    print(f"[stubhub pm] cached -> {cache.name}")
    return out


# ---- (4) StubHub stock via yfinance (daily only) --------------------------------------
def load_stub_spot() -> pd.DataFrame:
    cache = CACHE / "stub_spot_listingday.csv"
    if cache.exists():
        print(f"[stub spot] loaded from cache {cache.name}")
        return pd.read_csv(cache, parse_dates=["t"])
    df = pd.DataFrame(columns=["t", "open", "high", "low", "close", "interval"])
    # try intraday first (will usually be empty for a 2025 listing), then daily
    for q, lab in [("period1=1758000000&period2=1758585600&interval=1m", "1m"),
                   ("period1=1757980800&period2=1758672000&interval=1d", "1d")]:
        try:
            d = fetch_yahoo_chart("STUB", q)
            if not d.empty:
                d["interval"] = lab
                df = d
                print(f"[stub spot] Yahoo {lab}: {len(d)} bars")
                break
        except Exception as exc:  # noqa: BLE001
            print(f"[stub spot] Yahoo {lab} failed: {exc}")
    if df.empty:
        try:
            import yfinance as yf
            y = yf.download("STUB", start="2025-09-17", end="2025-09-24", interval="1d",
                            progress=False, auto_adjust=False)
            y = y.reset_index().rename(columns={"Date": "t", "Open": "open", "High": "high",
                                                "Low": "low", "Close": "close"})
            df = y[["t", "open", "high", "low", "close"]]
            df["t"] = pd.to_datetime(df["t"], utc=True)
            df["interval"] = "1d"
            print(f"[stub spot] yfinance daily: {len(df)} bars")
        except Exception as exc:  # noqa: BLE001
            print(f"[stub spot] yfinance failed too: {exc}")
    df.to_csv(cache, index=False)
    return df


# ======================== LOAD EVERYTHING ===============================================
cbrs_pm = load_cbrs_pm()
cbrs_spot = load_cbrs_spot()
stub_mk = stubhub_markets()
stub_pm = load_stubhub_pm(stub_mk)
stub_spot = load_stub_spot()

# ---- classify tails: any strike priced < 15c at first observation (a long-shot leg) ----
TAIL_THRESH = 0.15
def classify_tails(pm: dict[str, pd.DataFrame], winner: str | None = None) -> list[str]:
    tails = []
    for strike, d in pm.items():
        if d.empty:
            continue
        first_p = d["p"].iloc[0]
        if first_p < TAIL_THRESH and strike != winner:
            tails.append(strike)
    return tails

cbrs_tails = classify_tails(cbrs_pm, winner=CBRS_WINNER)
# StubHub winner inferred as the strike with the highest *terminal* price (resolved to 1)
def terminal_price(d: pd.DataFrame) -> float:
    return float(d["p"].iloc[-1]) if not d.empty else np.nan
stub_winner = max(stub_pm, key=lambda s: terminal_price(stub_pm[s]))
stub_tails = classify_tails(stub_pm, winner=stub_winner)

print("\n=== SUMMARY ===")
print(f"Cerebras tails (<15c at creation, winner={CBRS_WINNER}): {cbrs_tails}")
print(f"StubHub  winner (highest terminal price): {stub_winner}")
print(f"StubHub  tails (<15c at creation): {stub_tails}")
print(f"StubHub  stock rows: {len(stub_spot)} ({stub_spot['interval'].iloc[0] if len(stub_spot) else 'none'})")
''')

md(r"""
**Read:** the print-out above confirms which strikes count as *tails* (long-shot legs we would be
selling) versus the *winner* (the bucket that resolved to 1). For Cerebras the winner is the
`$50B+` strike (~91¢ near the end); its four lower strikes are the tails. For StubHub the winner
is inferred from the highest terminal price, and every strike that opened under 15¢ is a tail —
on a market-cap ladder that means **both the low end and the high end** are long shots.
""")


# =====================================================================================
# CELL 2 — CEREBRAS OVERLAY
# =====================================================================================
md(r"""
## Cell 2 — Cerebras: tail tokens overlaid on the stock price

Two panels on a shared UTC time axis with a vertical line at the **cross** (first stock print,
`2026-05-14 16:59 UTC`):

- **Top:** the CBRS stock (Yahoo 5-min), shown from ~1h before the cross through the close.
- **Bottom:** every Cerebras **tail** YES token in **cents**, colour-coded by strike. The `$50B+`
  winner is *not* a tail (it ran toward ~91¢); it is drawn as a thin grey reference line only.

The question this answers: did the long-shot strikes get *bid up* into the listing, and where were
they at the moment the stock finally printed?
""")

code(r'''
def to_cents(d):
    return d["t"], d["p"] * 100.0

STRIKE_COLORS = ["#ff6b6b", "#feca57", "#48dbfb", "#1dd1a1", "#c8d6e5", "#ff9ff3"]

fig, (axs, axt) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={"height_ratios": [2, 3]})

# --- top: stock ---
sp = cbrs_spot[cbrs_spot["t"] >= CBRS_CROSS - timedelta(hours=1)]
axs.plot(sp["t"], sp["close"], color="#54a0ff", lw=1.6, label="CBRS stock (Yahoo 5m close)")
axs.axhline(CBRS_OFFER, color="#10ac84", ls="--", lw=1.0, label=f"IPO offer ${CBRS_OFFER:.0f}")
if not sp.empty:
    hi_i = sp["close"].idxmax()
    axs.annotate(f"day-1 high ${sp['close'].max():,.0f}",
                 xy=(sp.loc[hi_i, "t"], sp["close"].max()),
                 xytext=(sp.loc[hi_i, "t"], sp["close"].max() * 1.03),
                 color="#54a0ff", fontsize=8, ha="center",
                 arrowprops=dict(arrowstyle="->", color="#54a0ff"))
axs.set_ylabel("stock price ($)")
axs.set_title("Cerebras listing day (2026-05-14): tail YES tokens vs the stock\n"
              "Vertical line = cross (first stock print, 16:59 UTC). Tails are long shots that resolve to 0.")
axs.legend(loc="upper left", fontsize=8)

# --- bottom: tail tokens in cents ---
for i, strike in enumerate(cbrs_tails):
    t, c = to_cents(cbrs_pm[strike])
    axt.plot(t, c, color=STRIKE_COLORS[i % len(STRIKE_COLORS)], lw=1.5, label=f"{strike} YES")
# winner as thin reference
tw, cw = to_cents(cbrs_pm[CBRS_WINNER])
axt.plot(tw, cw, color="#8395a7", lw=1.0, ls=":", alpha=0.8, label=f"{CBRS_WINNER} (winner, ref)")
axt.set_ylabel("YES price (¢)")
axt.set_xlabel("time (UTC)")
axt.legend(loc="upper left", fontsize=8, ncol=2)

for ax in (axs, axt):
    ax.axvline(CBRS_CROSS, color="#ee5253", ls="-", lw=1.3, alpha=0.9)
axt.text(CBRS_CROSS, axt.get_ylim()[1] * 0.95, " cross", color="#ee5253", fontsize=8, va="top")
axt.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=UTC))
fig.tight_layout()
plt.show()
''')

md(r"""
**Read:** good for the trade would be tail prices that *peak around or just before the cross* and
then fall — that is the euphoria we want to sell into. Bad (for a patient hold) would be tails that
keep drifting up after the cross. Watch where each coloured line sits at the red cross line versus
its own peak: the gap between "peak" and "value at cross" is the cost of being slow.
""")


# =====================================================================================
# CELL 3 — CEREBRAS ZOOM ON IPO DAY
# =====================================================================================
md(r"""
## Cell 3 — Cerebras: zoom on the IPO-day repricing (with the winner for contrast)

Same overlay, zoomed to the listing day only (`2026-05-14 12:00 → 22:00 UTC`), with a **third
panel** for the `$50B+` winner so we can compare how the *mode* repriced (toward 1) against how the
*tails* repriced (toward 0). Key question: did the tails spike **before, during, or after** the
stock cross, and how fast did they collapse?
""")

code(r'''
z0 = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
z1 = datetime(2026, 5, 14, 22, 0, tzinfo=UTC)
def clip(d):
    return d[(d["t"] >= z0) & (d["t"] <= z1)]

fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 3, 2]})

# stock
sp = clip(cbrs_spot)
a1.plot(sp["t"], sp["close"], color="#54a0ff", lw=1.6)
a1.set_ylabel("stock ($)")
a1.set_title("Cerebras IPO-day zoom (12:00–22:00 UTC): stock / tails / winner")

# tails
for i, strike in enumerate(cbrs_tails):
    d = clip(cbrs_pm[strike])
    a2.plot(d["t"], d["p"] * 100, color=STRIKE_COLORS[i % len(STRIKE_COLORS)], lw=1.6,
            marker=".", ms=3, label=f"{strike} YES")
    if not d.empty:
        pk = d.loc[d["p"].idxmax()]
        a2.annotate(f"{pk['p']*100:.0f}¢", xy=(pk["t"], pk["p"]*100),
                    xytext=(pk["t"], pk["p"]*100 + 1.5), fontsize=7,
                    color=STRIKE_COLORS[i % len(STRIKE_COLORS)], ha="center")
a2.set_ylabel("tail YES (¢)")
a2.legend(loc="upper right", fontsize=8, ncol=2)

# winner
dw = clip(cbrs_pm[CBRS_WINNER])
a3.plot(dw["t"], dw["p"] * 100, color="#1dd1a1", lw=1.8, label=f"{CBRS_WINNER} (winner) YES")
a3.set_ylabel("winner YES (¢)")
a3.set_xlabel("time (UTC)")
a3.legend(loc="lower right", fontsize=8)

for ax in (a1, a2, a3):
    ax.axvline(CBRS_CROSS, color="#ee5253", ls="-", lw=1.3, alpha=0.9)
a3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
fig.tight_layout()
plt.show()

# numeric read: tail price at cross vs peak, and minutes of peak relative to cross
print("Cerebras tails — peak vs cross (minutes relative to cross; +ve = after):")
for strike in cbrs_tails:
    d = cbrs_pm[strike]
    if d.empty:
        continue
    pk = d.loc[d["p"].idxmax()]
    mins = (pk["t"] - CBRS_CROSS).total_seconds() / 60
    # value at cross (last obs at/before the cross)
    at = d[d["t"] <= CBRS_CROSS]
    at_cross = at["p"].iloc[-1] if not at.empty else np.nan
    print(f"  {strike:8} peak {pk['p']*100:5.1f}¢ @ {mins:+6.0f} min | at cross {at_cross*100:5.1f}¢")
''')

md(r"""
**Read:** the printed table is the heart of the timing question. If the **peak minutes are
negative**, the tails were richest *before* the cross — i.e. the time to sell was during the
pre-open euphoria, and waiting for the stock to print already cost you. If the value-at-cross is
well below the peak, the collapse was fast. The winner panel is the mirror image: it converges
toward 100¢ as the tails bleed to 0.
""")


# =====================================================================================
# CELL 4 — CEREBRAS OPTIMAL SELL WINDOW
# =====================================================================================
md(r"""
## Cell 4 — Cerebras: the optimal tail sell window

Every tail resolved to **0**, so selling a tail YES token at price `p` locks in a profit of `p` per
share (you keep the premium; the buyer is left holding a 0). The "sell P&L" curve over time is
therefore just the **price curve** — its peak is the optimal sell moment.

**Table glossary (one row per tail strike):**

- `lifetime_max_¢` — the highest the YES token ever traded (the best possible sell price).
- `max_@min` — when that peak happened, in **minutes relative to the cross** (negative = before).
- `at_cross_¢` — price at the cross (what a "wait for the stock" seller would get).
- `%max_at_cross` — `at_cross / lifetime_max` — fraction of the best price still available at the cross.
- `+30m_¢`, `+60m_¢` — price 30 and 60 minutes after the cross (how fast it bled).
- `half_life_min` — minutes after the peak for the price to fall to half the peak (collapse speed).
""")

code(r'''
def at_or_before(d, t):
    s = d[d["t"] <= t]
    return float(s["p"].iloc[-1]) if not s.empty else np.nan

def post_peak_halflife(d):
    """Minutes from the peak until price first falls to <= 50% of the peak."""
    if d.empty:
        return np.nan
    pk_i = d["p"].idxmax()
    pk_t, pk_p = d.loc[pk_i, "t"], d.loc[pk_i, "p"]
    after = d[d["t"] > pk_t]
    hit = after[after["p"] <= pk_p * 0.5]
    if hit.empty:
        return np.nan
    return (hit["t"].iloc[0] - pk_t).total_seconds() / 60

def sell_window_table(pm, tails, cross):
    rows = []
    for strike in tails:
        d = pm[strike]
        if d.empty:
            continue
        pk_i = d["p"].idxmax()
        pk_t, pk_p = d.loc[pk_i, "t"], d.loc[pk_i, "p"]
        at_c = at_or_before(d, cross)
        rows.append({
            "strike": strike,
            "lifetime_max_¢": round(pk_p * 100, 1),
            "max_@min": round((pk_t - cross).total_seconds() / 60),
            "at_cross_¢": round(at_c * 100, 1) if not np.isnan(at_c) else np.nan,
            "%max_at_cross": round(100 * at_c / pk_p) if not np.isnan(at_c) else np.nan,
            "+30m_¢": round(at_or_before(d, cross + timedelta(minutes=30)) * 100, 1),
            "+60m_¢": round(at_or_before(d, cross + timedelta(minutes=60)) * 100, 1),
            "half_life_min": round(post_peak_halflife(d)) if not np.isnan(post_peak_halflife(d)) else None,
        })
    return pd.DataFrame(rows)

cbrs_tbl = sell_window_table(cbrs_pm, cbrs_tails, CBRS_CROSS)
print("CEREBRAS — tail sell-window table")
print(cbrs_tbl.to_string(index=False))

# P&L (=price) curves vs minutes-since-cross, with the cross and peaks marked
fig, ax = plt.subplots(figsize=(12, 6))
for i, strike in enumerate(cbrs_tails):
    d = cbrs_pm[strike]
    if d.empty:
        continue
    mins = (d["t"] - CBRS_CROSS).dt.total_seconds() / 60
    ax.plot(mins, d["p"] * 100, color=STRIKE_COLORS[i % len(STRIKE_COLORS)], lw=1.6,
            label=f"{strike} YES")
    pk_i = d["p"].idxmax()
    ax.scatter([(d.loc[pk_i, "t"] - CBRS_CROSS).total_seconds() / 60], [d.loc[pk_i, "p"] * 100],
               color=STRIKE_COLORS[i % len(STRIKE_COLORS)], s=40, zorder=5, edgecolor="white")
ax.axvline(0, color="#ee5253", lw=1.3, label="cross")
ax.set_xlabel("minutes since cross")
ax.set_ylabel("sell P&L per share (¢)  =  YES price")
ax.set_title("Cerebras: tail-token sell P&L over time (dots = lifetime peak = optimal sell)\n"
             "Selling a tail at price p locks in +p because the tail resolves to 0.")
ax.set_xlim(-1500, 400)
ax.legend(loc="upper right", fontsize=8, ncol=2)
plt.show()
''')

md(r"""
**Read:** the dots mark the optimal sell — the lifetime peak of each tail. The actionable numbers
are `%max_at_cross` and `half_life_min`: if most of the peak value was *already gone* by the cross,
the lesson for SPCX is to **pre-position the sell into the pre-listing euphoria** rather than wait
for the first stock print; if the half-life is short (tens of minutes), the exit window is narrow
and the order should be worked aggressively, not patiently.
""")


# =====================================================================================
# CELL 5 — STUBHUB
# =====================================================================================
md(r"""
## Cell 5 — StubHub: the same analysis (with honest data caveats)

StubHub (NYSE:STUB, 2025-09-17) is the second precedent, and a weaker one: it was a **~$119K
total-volume** market and the **stock's intraday history is purged** (only daily OHLC survives), so
the stock overlay is a single daily candle, not a tape. The PM token prices, however, are fully
recoverable from on-chain Goldsky fills.

StubHub's ladder is shaped differently from Cerebras: a **middle** bucket won (inferred from the
terminal prices in Cell 1 — printed there and below), so the **tails are on both ends** — the low strikes
(`<$7B`) *and* the high strikes (`≥$13B`, `$12-13B`, `$11-12B`). Those high strikes are the closest
structural analog to the SPCX `>$2.8T / >$3T / ...` upper-tail tokens we plan to sell.
""")

code(r'''
# resample each strike to a 10-min last-trade line for plotting (fills are sparse)
def last_trade_line(d, freq="10min"):
    if d.empty:
        return d
    s = d.set_index("t")["p"].resample(freq).last().ffill().dropna()
    return s.reset_index()

print(f"StubHub winner: {stub_winner} | tails: {stub_tails}")

# --- two panels: stock (daily) + tail tokens ---
fig, (axs, axt) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={"height_ratios": [1, 3]})

if len(stub_spot):
    iv = stub_spot["interval"].iloc[0]
    if iv == "1d":
        axs.plot(stub_spot["t"], stub_spot["close"], color="#54a0ff", marker="o", lw=1.2,
                 label="STUB daily close")
        axs.scatter(stub_spot["t"], stub_spot["open"], color="#feca57", s=25, label="daily open")
    else:
        axs.plot(stub_spot["t"], stub_spot["close"], color="#54a0ff", lw=1.4,
                 label=f"STUB {iv} close")
    axs.axhline(STUB_OFFER, color="#10ac84", ls="--", lw=1.0, label=f"IPO offer ${STUB_OFFER:.2f}")
    axs.legend(loc="upper left", fontsize=8)
axs.set_ylabel("stock ($)")
axs.set_title("StubHub (2025-09-17): tail YES tokens vs the stock (stock = DAILY only; cross clock approximate)")

# tail tokens (resampled lines) + raw fills as faint dots
hi_strikes = [s for s in stub_tails if s not in ("<$7B",)]
order = stub_tails
for i, strike in enumerate(order):
    d = stub_pm[strike]
    if d.empty:
        continue
    ln = last_trade_line(d)
    axt.plot(ln["t"], ln["p"] * 100, color=STRIKE_COLORS[i % len(STRIKE_COLORS)], lw=1.5,
             label=f"{strike} YES")
    axt.scatter(d["t"], d["p"] * 100, color=STRIKE_COLORS[i % len(STRIKE_COLORS)], s=6, alpha=0.35)
# winner reference
wln = last_trade_line(stub_pm[stub_winner])
axt.plot(wln["t"], wln["p"] * 100, color="#8395a7", ls=":", lw=1.0, alpha=0.8,
         label=f"{stub_winner} (winner, ref)")
axt.set_ylabel("YES price (¢)")
axt.set_xlabel("time (UTC)")
axt.legend(loc="upper left", fontsize=8, ncol=2)

for ax in (axs, axt):
    ax.axvline(STUB_CROSS_APPROX, color="#ee5253", ls="--", lw=1.2, alpha=0.8)
axt.text(STUB_CROSS_APPROX, axt.get_ylim()[1] * 0.95, " cross (approx)", color="#ee5253",
         fontsize=8, va="top")
axt.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=UTC))
fig.tight_layout()
plt.show()

# sell-window table for StubHub tails (against the approximate cross)
stub_tbl = sell_window_table(stub_pm, stub_tails, STUB_CROSS_APPROX)
print("\nSTUBHUB — tail sell-window table (cross = approx; minutes are vs the approx cross)")
print(stub_tbl.to_string(index=False))
''')

md(r"""
**Read:** treat StubHub as *corroboration, not proof*. The stock panel is a single daily candle, so
the cross line is an approximate clock — read the StubHub `max_@min` column as "roughly relative to
the listing", not to the minute. What *is* reliable is the **shape** of the tail YES curves: whether
the high-end strikes (the SPCX analog) were bid up around the listing and then bled toward 0, and
how thin the trading was. If StubHub agrees with Cerebras that tails are richest around the listing
and collapse afterward, that strengthens the n≈1.x calibration; if it is too thin to read, we say so.
""")


# =====================================================================================
# CELL 6 — SYNTHESIS
# =====================================================================================
md(r"""
## Cell 6 — Synthesis: what this means for the SPCX tail-sell today

This cell builds a side-by-side comparison of the two precedents and maps the pattern onto the live
SPCX tail tokens. **SPCX is listing today (2026-06-12).**

**SPCX upper-tail tokens to sell (from the live screen):**
`>$2.8T = 9¢ · >$3.0T = 6¢ · >$3.2T = 4¢ · >$3.4T = 3¢ · >$3.6T = 3¢ · >$3.8T = 1¢ · >$4.0T = 1¢`.

**Calibration honesty:** this is **n = 2**, and StubHub is a thin ~$119K market with daily-only
stock — so really n ≈ 1.x. The deliverable is **intuition and a pre-registered sell plan**, not a
statistical edge. Read every number below as a precedent, not a forecast.
""")

code(r'''
# ---- per-IPO summary stats ----
def stock_move_vs_offer(spot, offer):
    if spot.empty:
        return np.nan, np.nan
    hi = spot["high"].max() if "high" in spot else spot["close"].max()
    cl = spot["close"].iloc[-1]
    return (hi / offer - 1) * 100, (cl / offer - 1) * 100

def tail_summary(pm, tails, cross):
    """Aggregate tail behaviour: mean peak ¢, mean %-of-max retained at cross, mean half-life."""
    tbl = sell_window_table(pm, tails, cross)
    if tbl.empty:
        return {}
    return {
        "n_tails": len(tbl),
        "mean_peak_¢": round(tbl["lifetime_max_¢"].mean(), 1),
        "mean_%max_at_cross": round(tbl["%max_at_cross"].dropna().mean()) if tbl["%max_at_cross"].notna().any() else None,
        "median_peak_@min": int(tbl["max_@min"].median()),
        "mean_half_life_min": round(tbl["half_life_min"].dropna().mean()) if tbl["half_life_min"].notna().any() else None,
    }

cbrs_hi, cbrs_cl = stock_move_vs_offer(cbrs_spot, CBRS_OFFER)
stub_hi, stub_cl = stock_move_vs_offer(stub_spot, STUB_OFFER)
cbrs_sum = tail_summary(cbrs_pm, cbrs_tails, CBRS_CROSS)
stub_sum = tail_summary(stub_pm, stub_tails, STUB_CROSS_APPROX)

comp = pd.DataFrame([
    {"IPO": "Cerebras (CBRS)", "stock_high_vs_offer_%": round(cbrs_hi),
     "stock_close_vs_offer_%": round(cbrs_cl), **cbrs_sum},
    {"IPO": "StubHub (STUB)", "stock_high_vs_offer_%": round(stub_hi) if not np.isnan(stub_hi) else None,
     "stock_close_vs_offer_%": round(stub_cl) if not np.isnan(stub_cl) else None, **stub_sum},
])
print("=== PRECEDENT COMPARISON ===")
print(comp.to_string(index=False))
print("\nColumn meaning: stock_*_vs_offer_% = day-1 high/close move vs the IPO offer; "
      "mean_peak_¢ = avg lifetime peak of the tail YES tokens; mean_%max_at_cross = avg fraction "
      "of that peak still available at the cross; median_peak_@min = median minutes of the peak "
      "relative to the cross (negative = BEFORE the cross); mean_half_life_min = avg minutes for a "
      "tail to halve after its peak.")

# ---- map onto SPCX tails ----
SPCX = {">$2.8T": 9, ">$3.0T": 6, ">$3.2T": 4, ">$3.4T": 3, ">$3.6T": 3, ">$3.8T": 1, ">$4.0T": 1}
pct_at_cross = cbrs_sum.get("mean_%max_at_cross") or 0
print("\n=== SPCX TAIL MAP (Cerebras-like repricing) ===")
print(f"Cerebras tails retained ~{pct_at_cross}% of their peak by the cross and peaked a median "
      f"{cbrs_sum.get('median_peak_@min')} min relative to it.")
print(f"{'strike':8} {'now¢':>5} {'if it behaves like a CBRS tail':>40}")
for k, v in SPCX.items():
    # illustrative: a CBRS-like tail is richest pre/at cross; selling now vs waiting loses ~ (100-pct)
    print(f"{k:8} {v:5} {'sell into the pre/at-cross euphoria; ~' + str(100 - pct_at_cross) + '% of peak lost if you wait past cross':>40}")
''')

md(r"""
## Takeaways for the SPCX tail-sell (pre-registered intuition)

> The exact numbers are filled in by the executed cells above; this section states how to *read*
> them into a plan.

1. **The tails are a sell, and the clock matters more than the level.** Both precedents show
   long-shot strikes that resolve to 0. The only decision is timing — and the data says the richest
   prices cluster **around and just before the cross**, not after the stock has settled.
2. **Front-load the exit.** If Cerebras tails had already lost a meaningful share of their peak by
   the cross (see `%max_at_cross`) and halved within tens of minutes (`half_life_min`), the SPCX
   orders should be **resting into the pre-listing pop and worked aggressively at the cross**, not
   held for a better post-cross print.
3. **Sell the whole tail strip, weighted to the fattest strikes.** The `>$2.8T` (9¢) and `>$3.0T`
   (6¢) tokens carry the most premium per share; the 1¢ strikes are almost free options for the
   buyer and barely worth the fee to exit — prioritise the 4–9¢ strikes.
4. **n ≈ 1.x — do not over-fit.** StubHub was thin and daily-only; Cerebras is the real anchor.
   Treat the half-life and peak-timing as *order-of-magnitude* guidance, and watch the live SPCX
   tape (the `spcx_pm_pdf_monitor` dashboard) for confirmation rather than assuming the precedent.

**Next step:** wire the `%max_at_cross` / `half_life_min` reads into the listing-day playbook
([[spcx_listing_day_gameplan]]) as the tail-exit schedule, and confirm against the live PM ladder on
the [[spcx_pm_pdf_monitor_findings|PM-PDF monitor]] as SPCX crosses today.
""")


# =====================================================================================
# ASSEMBLE
# =====================================================================================
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB_PATH))
print(f"wrote {NB_PATH} ({len(cells)} cells)")
