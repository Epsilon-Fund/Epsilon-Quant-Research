#!/usr/bin/env python3
"""Block SPREAD-1 — build + validate the trade-anchored effective-spread surface.

Replaces the flat 2-3c spread assumption (data_infra/weather_analysis.py) with a
state-dependent HALF-spread surface estimated from the public trade tape:

    half_spread_est = s_dir * (P_fill - mid_t),   s_dir = +1 taker-buy / -1 taker-sell

with mid from the CLOB ``/prices-history`` midpoint endpoint (lookahead-free
as-of join, mid bar strictly BEFORE the fill) and the aggressor sign from the
tape's ``maker_side`` (passive maker side on normal legs; ``_matchOrders``
internal re-emits are dropped as duplicates — see lib/spread_surface.py).

Subcommands (run in order; each caches so reruns are cheap):
    semantics    Pre-registered step 1: confirm /prices-history `p` is a book
                 midpoint by comparing against replayed live_clob book mid()
                 (lib/clob_book.py) on captured markets. STOP if it is not.
    fetch-mids   Sample (market, day) anchors from the tape, fetch + cache
                 /prices-history at 1-min fidelity per (token, day).
    estimate     Join sampled fills to cached mids -> per-fill half-spreads.
    surface      Aggregate to cells; write surface + meta CSVs.
    validate     Pre-registered step 4: predicted vs true L1 half-spread on
                 every live_clob market x window; MedAE <= 1c AND Spearman
                 >= 0.6 gate.
    diagnose     Pre-registered step 5: negative-rate, size-quantile split,
                 bid-ask bounce + Roll covariance cross-checks.

Block SPREAD-1b (separate pre-registered gate; the surface is FROZEN):
    validate-tradetime
                 Re-gate the SAME frozen surface against the TRADE-TIME quoted
                 half-spread: at each last_trade_price print in a capture
                 window, the (ask-bid)/2 of the last best_bid_ask event
                 strictly before the print. Same captures, same window grid,
                 same features and market-cells as `validate` — only the truth
                 changes. Bars: pooled MedAE <= 1c; fast-crypto MedAE <=
                 1.25c; sign test >= 60% of non-tied cells beat the flat-3c
                 incumbent. Spearman is a diagnostic, NOT a bar. Includes the
                 contaminated-cell hybrid arm (bounce/Roll replacement levels
                 for frac_negative > 0.4 cells).
    charts-1b    Pred-vs-true chart with the gate band + trade-time vs
                 time-averaged truth-compression chart.

Run env (from polymarket/research):
    PYTHONPATH=. uv run python scripts/spread_surface_build.py <subcommand>

All API responses are cached as ZSTD parquet under data/analysis/spread_surface/
so reruns are free and the CLOB API is hit at a polite rate (~4 req/s).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.clob_book import ClobBook  # noqa: E402
from lib.spread_surface import (  # noqa: E402
    ACTIVITY_LABELS,
    CATEGORY_CASE_SQL,
    EXCHANGE_INTERNAL_LEG,
    FALLBACK_LEVELS,
    TICK_FLOOR_CENTS,
    TRAILING_RATE_WINDOW_S,
    SpreadSurface,
    activity_bucket,
    aggressor_dir,
    price_bucket,
    ttr_bucket,
)

DATA = ROOT / "data"
TRADES_GLOB = str(DATA / "trades" / "*.parquet")
MARKETS_PARQUET = sorted((DATA / "markets").glob("markets_*.parquet"))[-1]
LIVE_CLOB = DATA / "live_clob"
WORK = DATA / "analysis" / "spread_surface"
MID_CACHE = WORK / "mid_history"
CAPTURE_CACHE = WORK / "capture_l1"
CSV_OUT = DATA / "analysis" / "csv_outputs" / "copytrade"
PLOTS_OUT = DATA / "analysis" / "plots" / "copytrade"

CLOB_BASE = "https://clob.polymarket.com"
API_SLEEP_S = 0.25          # polite pacing between uncached calls
INTERNAL_SET_SQL = "(" + ", ".join(f"'{a}'" for a in EXCHANGE_INTERNAL_LEG) + ")"

# Sampling design (estimate side). Window ends at the local tape tail.
BUILD_WINDOW_START = "2026-03-01"
BUILD_WINDOW_END = "2026-05-27"      # exclusive; tape ends 2026-05-26 19:57:58
MARKETS_PER_CAT_PER_QUARTILE = 12    # stratified by market fill-count quartile
DAYS_PER_MARKET = 4                  # sampled distinct UTC fill-days per market
MAX_FILLS_PER_MARKET_DAY = 1500      # estimation anchors per market-day
MID_MAX_AGE_S = 1800                 # mid bar older than this -> estimate excluded
SEED = 20260611

# Validation design. Windows are non-overlapping; features computed at window
# start so the true spread (measured across the window) is out-of-feature-time.
VALIDATION_WINDOW_S = 1800
VALIDATION_CAPTURES = [
    "block_a0/block_a0_20260528_morning",
    "block_a0b/block_a0b_replacements_v2_20260527",
    "block_a0c/block_a0c_targeted_20260529_morning",
    "block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning",
    "mm_stage1_broad_live/mm_stage1_broad_live_awake_20260609T093741Z",
    "mm_stage1_broad_live/mm_stage1_broad_live_awake_20260610T100020Z",
]

# Block SPREAD-1b — pre-registered bars for the trade-time re-gate. The
# surface is FROZEN (no refit, no re-binning, no estimator variants); only the
# validation TARGET changes to quoted half-spread as-of trade times.
GATE_1B_POOLED_MEDAE_C = 1.0
GATE_1B_FAST_MEDAE_C = 1.25
GATE_1B_SIGNTEST_SHARE = 0.60
FAST_CATEGORIES = ("crypto_4h", "daily_crypto")
# The incumbent comparator named in the pre-registration is "flat 3c" taken
# literally as a half-spread prediction. The weather_analysis constants it
# stands in for are FULL-spread numbers (DEFAULT_SPREAD_CENTS = 2.0, bid =
# ask - spread), so flat 1.5c / 1.0c half-spread variants are also scored —
# as sensitivity diagnostics only, not bars.
FLAT_BASELINE_C = 3.0
FLAT_SENSITIVITY_C = (1.5, 1.0)
CONTAMINATION_FLAG = 0.4   # frac_negative threshold for the hybrid arm


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def ensure_dirs() -> None:
    for d in (WORK, MID_CACHE, CAPTURE_CACHE, CSV_OUT, PLOTS_OUT):
        d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# /prices-history fetch + cache
# ============================================================================
def fetch_mid_history(token: str, start_s: int, end_s: int, fidelity_min: int = 1,
                      client: httpx.Client | None = None, retries: int = 3) -> pd.DataFrame:
    """Cached fetch of one token's midpoint history. Returns columns [t, p].

    Cache key includes the full request params; reruns never re-hit the API.
    Caches ONLY on a true HTTP 200 (an empty history from a 200 is a legit
    'no quotes' answer and is cacheable); transient failures (timeouts, DNS,
    4xx/5xx) retry with backoff and are left UNCACHED so a re-run retries them
    — a failure must never poison the cache as a permanent empty.
    """
    cache = MID_CACHE / f"midhist_{token[:16]}_{start_s}_{end_s}_f{fidelity_min}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    own = client is None
    cl = client or httpx.Client(timeout=30.0)
    hist, ok = [], False
    try:
        for attempt in range(retries):
            try:
                r = cl.get(f"{CLOB_BASE}/prices-history",
                           params={"market": token, "startTs": start_s, "endTs": end_s,
                                   "fidelity": fidelity_min})
                r.raise_for_status()
                hist = r.json().get("history", [])
                ok = True
                break
            except Exception as e:  # noqa: BLE001 — transient; back off and retry
                if attempt == retries - 1:
                    log(f"  /prices-history failed for {token[:14]}… (left uncached): {e!r}")
                else:
                    time.sleep(0.5 * (attempt + 1))
    finally:
        if own:
            cl.close()
    df = pd.DataFrame({"t": [float(h["t"]) for h in hist],
                       "p": [float(h["p"]) for h in hist]})
    if ok:
        df.to_parquet(cache, compression="zstd", index=False)
    time.sleep(API_SLEEP_S)
    return df


# ============================================================================
# live_clob capture parsing (shared by `semantics` and `validate`)
# ============================================================================
def capture_files(capture_dir: Path) -> list[Path]:
    # rglob: some runs (e.g. the a0c crypto roll) shard into chunk subdirs
    return sorted(p for p in capture_dir.rglob("*.jsonl") if "capture_gaps" not in p.name)


def load_capture_markets(capture_dir: Path) -> pd.DataFrame:
    """Market metadata union across a capture run's manifests.

    Returns one row per market: market_id, slug, question, family, neg_risk,
    end_date, token_ids (list). Tolerates both manifest shapes ('markets' with
    clob_token_ids; 'tokens' flat list)."""
    rows: dict[str, dict] = {}
    for mf in sorted(capture_dir.rglob("*.manifest.json")):
        try:
            m = json.loads(mf.read_text())
        except Exception:  # noqa: BLE001
            continue
        for mk in m.get("markets", []):
            mid = str(mk.get("market_id") or mk.get("id") or "")
            if not mid:
                continue
            rows.setdefault(mid, {
                "market_id": mid,
                "slug": mk.get("event_slug") or mk.get("slug") or "",
                "question": mk.get("question") or "",
                "family": mk.get("family") or "",
                "neg_risk": bool(mk.get("neg_risk", False)),
                "end_date": mk.get("end_date"),
                "token_ids": [str(t) for t in (mk.get("clob_token_ids") or [])],
            })
        for tk in m.get("tokens", []):
            mid = str(tk.get("market_id") or "")
            if not mid:
                continue
            row = rows.setdefault(mid, {
                "market_id": mid, "slug": tk.get("slug") or "",
                "question": tk.get("question") or "", "family": tk.get("family") or "",
                "neg_risk": False, "end_date": tk.get("end_date"), "token_ids": [],
            })
            tid = str(tk.get("token_id") or "")
            if tid and tid not in row["token_ids"]:
                row["token_ids"].append(tid)
    return pd.DataFrame(rows.values())


def extract_capture_events(capture_dir: Path, run_name: str) -> dict[str, pd.DataFrame]:
    """Stream a capture run once -> cached parquets {l1, trades}.

    l1:     token_id, ts (PM epoch s), best_bid, best_ask        (best_bid_ask)
    trades: token_id, market_id, ts, price, size, side, tx_hash  (last_trade_price)
    """
    l1_pq = CAPTURE_CACHE / f"{run_name}__l1.parquet"
    tr_pq = CAPTURE_CACHE / f"{run_name}__trades.parquet"
    if l1_pq.exists() and tr_pq.exists():
        return {"l1": pd.read_parquet(l1_pq), "trades": pd.read_parquet(tr_pq)}

    l1_rows, tr_rows = [], []
    files = capture_files(capture_dir)
    for i, f in enumerate(files):
        log(f"  scanning {f.name} ({i + 1}/{len(files)})")
        with f.open() as fh:
            for line in fh:
                if '"best_bid_ask"' in line:
                    try:
                        d = json.loads(line)
                        msg = d["message"]
                        l1_rows.append((str(msg["asset_id"]), float(msg["timestamp"]) / 1e3,
                                        float(msg["best_bid"]), float(msg["best_ask"])))
                    except (KeyError, ValueError, TypeError):
                        continue
                elif '"last_trade_price"' in line:
                    try:
                        d = json.loads(line)
                        msg = d["message"]
                        mid_ = ""
                        assets = d.get("assets") or []
                        if assets:
                            mid_ = str(assets[0].get("market_id", ""))
                        tr_rows.append((str(msg["asset_id"]), mid_,
                                        float(msg["timestamp"]) / 1e3, float(msg["price"]),
                                        float(msg.get("size") or 0.0),
                                        str(msg.get("side") or ""),
                                        str(msg.get("transaction_hash") or "")))
                    except (KeyError, ValueError, TypeError):
                        continue
    l1 = pd.DataFrame(l1_rows, columns=["token_id", "ts", "best_bid", "best_ask"])
    tr = pd.DataFrame(tr_rows, columns=["token_id", "market_id", "ts", "price", "size",
                                        "side", "tx_hash"])
    l1.to_parquet(l1_pq, compression="zstd", index=False)
    tr.to_parquet(tr_pq, compression="zstd", index=False)
    return {"l1": l1, "trades": tr}


def replay_book_mid(capture_dir: Path, token_ids: set[str]) -> dict[str, pd.DataFrame]:
    """Full book replay (book + price_change through lib/clob_book.ClobBook) for
    selected tokens -> {token_id: DataFrame[ts, mid]}. Used by `semantics` only
    (best_bid_ask extraction is the cheap path used everywhere else; `semantics`
    also verifies the two agree)."""
    books = {t: ClobBook() for t in token_ids}
    out: dict[str, list[tuple[float, float]]] = {t: [] for t in token_ids}
    needles = {t: t[:32] for t in token_ids}
    for f in capture_files(capture_dir):
        with f.open() as fh:
            for line in fh:
                if not any(n in line for n in needles.values()):
                    continue
                if '"event_type":"book"' not in line and '"event_type":"price_change"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                msg = d.get("message", {})
                et = d.get("event_type")
                if et == "book":
                    tok = str(msg.get("asset_id", ""))
                    if tok not in books:
                        continue
                    ts = float(msg["timestamp"]) / 1e3
                    books[tok].replace(
                        [(float(b["price"]), float(b["size"])) for b in msg.get("bids", [])],
                        [(float(a["price"]), float(a["size"])) for a in msg.get("asks", [])])
                    m = books[tok].mid()
                    if m is not None:
                        out[tok].append((ts, m))
                elif et == "price_change":
                    ts = float(msg.get("timestamp", 0)) / 1e3
                    for ch in msg.get("changes", []) or msg.get("price_changes", []) or []:
                        tok = str(ch.get("asset_id", msg.get("asset_id", "")))
                        if tok not in books:
                            continue
                        cts = float(ch.get("timestamp", msg.get("timestamp", 0))) / 1e3 or ts
                        books[tok].update_level(str(ch.get("side", "")).upper(),
                                                float(ch["price"]), float(ch["size"]))
                        m = books[tok].mid()
                        if m is not None:
                            out[tok].append((cts, m))
    return {t: pd.DataFrame(v, columns=["ts", "mid"]) for t, v in out.items()}


def _asof_value(ts_arr: np.ndarray, val_arr: np.ndarray, t: float,
                strict: bool = False) -> float | None:
    """Last value with ts <= t (or < t when strict). None if nothing yet."""
    i = np.searchsorted(ts_arr, t, side="left" if strict else "right")
    return float(val_arr[i - 1]) if i > 0 else None


# ----------------------------------------------------------------------------
# SPREAD-1b pure helpers (unit-tested in tests/test_spread_surface.py)
# ----------------------------------------------------------------------------
def tradetime_quoted_half_spreads(l1_ts: np.ndarray, l1_half_c: np.ndarray,
                                  trade_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quoted half-spread (cents) prevailing at each trade time: the value of
    the last best_bid_ask event STRICTLY before the trade (the SPREAD-1b
    target quantity). Trades with no prior quote are dropped.

    Returns (half_spreads, quote_ages_s), aligned with each other but possibly
    shorter than trade_ts."""
    idx = np.searchsorted(l1_ts, np.asarray(trade_ts, dtype=float), side="left")
    ok = idx > 0
    tts = np.asarray(trade_ts, dtype=float)[ok]
    return np.asarray(l1_half_c, dtype=float)[idx[ok] - 1], tts - np.asarray(l1_ts, dtype=float)[idx[ok] - 1]


def sign_test_share(err_a: np.ndarray, err_b: np.ndarray) -> tuple[float, int, int]:
    """Head-to-head sign test: share of NON-TIED cells where err_a < err_b
    (exact float ties excluded from numerator and denominator, per the
    SPREAD-1b pre-registration). Returns (share, n_wins, n_nontied); share is
    NaN when every cell ties."""
    a = np.asarray(err_a, dtype=float)
    b = np.asarray(err_b, dtype=float)
    nontied = a != b
    n_nontied = int(nontied.sum())
    n_wins = int((a[nontied] < b[nontied]).sum())
    return (n_wins / n_nontied if n_nontied else float("nan")), n_wins, n_nontied


def hybrid_prediction(pred_half_c: float, cell_frac_negative: float | None,
                      category: str, price_b: str,
                      repl_lookup: dict[tuple[str, str], float]) -> tuple[float, bool]:
    """SPREAD-1b §3 hybrid arm: when the surface cell behind a prediction is
    contamination-flagged (frac_negative > CONTAMINATION_FLAG), replace the
    prediction with the tape-only bounce/Roll level for (category,
    price_bucket); keep the surface prediction elsewhere or when no
    replacement level exists. The pre-registered tick floor applies to the
    replacement. Returns (prediction, was_replaced)."""
    if cell_frac_negative is None or cell_frac_negative <= CONTAMINATION_FLAG:
        return pred_half_c, False
    v = repl_lookup.get((category, price_b))
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pred_half_c, False
    return max(float(v), TICK_FLOOR_CENTS), True


# ============================================================================
# Subcommand: semantics
# ============================================================================
def cmd_semantics(args: argparse.Namespace) -> int:
    """Compare /prices-history `p` against replayed book mid (and best_bid_ask
    mid, and last trade) on captured market x times. Writes a CSV verdict."""
    ensure_dirs()
    specs = [
        ("block_a0c/block_a0c_targeted_20260529_morning", 4),
        ("block_a0/block_a0_20260528_morning", 2),
        ("mm_stage1_broad_live/mm_stage1_broad_live_awake_20260610T100020Z", 3),
    ]
    rows = []
    for rel, n_tokens in specs:
        cdir = LIVE_CLOB / rel
        run = rel.split("/")[-1]
        if not cdir.exists():
            log(f"capture missing: {rel} — skipped")
            continue
        log(f"capture {run}")
        ev = extract_capture_events(cdir, run)
        l1, tr = ev["l1"], ev["trades"]
        if l1.empty:
            log("  no best_bid_ask events — skipped")
            continue
        mkts = load_capture_markets(cdir)
        tok2fam = {}
        for _, mk in mkts.iterrows():
            for t in mk["token_ids"]:
                tok2fam[t] = mk["family"]
        # pick the most L1-active token per family, up to n_tokens families
        counts = l1.groupby("token_id").size().sort_values(ascending=False)
        chosen, seen_fam = [], set()
        for tok in counts.index:
            fam = tok2fam.get(tok, "?")
            if fam in seen_fam:
                continue
            chosen.append(tok)
            seen_fam.add(fam)
            if len(chosen) >= n_tokens:
                break
        log(f"  chosen tokens: {[(t[:10], tok2fam.get(t, '?')) for t in chosen]}")
        book_mids = replay_book_mid(cdir, set(chosen))
        for tok in chosen:
            sub = l1[l1["token_id"] == tok].sort_values("ts")
            t0, t1 = float(sub["ts"].min()), float(sub["ts"].max())
            hist = fetch_mid_history(tok, int(t0), int(t1) + 1, fidelity_min=1)
            if hist.empty:
                log(f"  {tok[:10]}…: no /prices-history points — skipped")
                continue
            bm = book_mids.get(tok, pd.DataFrame(columns=["ts", "mid"])).sort_values("ts")
            trs = tr[tr["token_id"] == tok].sort_values("ts")
            l1_ts = sub["ts"].to_numpy()
            l1_mid = ((sub["best_bid"] + sub["best_ask"]) / 2).to_numpy()
            bm_ts, bm_mid = bm["ts"].to_numpy(), bm["mid"].to_numpy()
            tr_ts, tr_p = trs["ts"].to_numpy(), trs["price"].to_numpy()
            dev_book, dev_l1, dev_last = [], [], []
            for t, p in zip(hist["t"].to_numpy(), hist["p"].to_numpy(), strict=True):
                if t < t0 or t > t1:
                    continue
                v_l1 = _asof_value(l1_ts, l1_mid, t)
                v_bk = _asof_value(bm_ts, bm_mid, t) if len(bm_ts) else None
                v_tr = _asof_value(tr_ts, tr_p, t) if len(tr_ts) else None
                if v_l1 is not None:
                    dev_l1.append(abs(p - v_l1) * 100)
                if v_bk is not None:
                    dev_book.append(abs(p - v_bk) * 100)
                if v_tr is not None:
                    dev_last.append(abs(p - v_tr) * 100)
            if not dev_l1:
                continue
            rows.append({
                "capture": run, "token_id": tok, "family": tok2fam.get(tok, "?"),
                "n_hist_points": len(dev_l1),
                "mean_absdev_vs_book_mid_c": float(np.mean(dev_book)) if dev_book else np.nan,
                "med_absdev_vs_book_mid_c": float(np.median(dev_book)) if dev_book else np.nan,
                "mean_absdev_vs_l1_mid_c": float(np.mean(dev_l1)),
                "med_absdev_vs_l1_mid_c": float(np.median(dev_l1)),
                "mean_absdev_vs_last_trade_c": float(np.mean(dev_last)) if dev_last else np.nan,
                "med_absdev_vs_last_trade_c": float(np.median(dev_last)) if dev_last else np.nan,
            })
    out = pd.DataFrame(rows)
    path = CSV_OUT / "spread_surface_v1_semantics_check.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    if out.empty:
        print("SEMANTICS: NO DATA — cannot proceed", file=sys.stderr)
        return 1
    med_mid = float(out["med_absdev_vs_l1_mid_c"].median())
    med_last = float(out["med_absdev_vs_last_trade_c"].median())
    verdict = "MIDPOINT" if (med_mid <= 0.5 and (np.isnan(med_last) or med_mid < med_last)) else "NOT-MIDPOINT"
    print(f"\npooled median |p - L1 mid|   = {med_mid:.3f}c")
    print(f"pooled median |p - lasttrade| = {med_last:.3f}c")
    print(f"VERDICT: {verdict}  (gate: midpoint iff med dev <= 0.5c and < last-trade dev)")
    print(f"wrote {path}")
    return 0 if verdict == "MIDPOINT" else 1


# ============================================================================
# Subcommand: fetch-mids  (sampling + API cache fill)
# ============================================================================
def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"""
        CREATE OR REPLACE VIEW mkt AS
        SELECT CAST(id AS VARCHAR) AS market_id,
               lower(coalesce(slug, ''))     AS slug_l,
               coalesce(slug, '')            AS slug,
               lower(coalesce(question, '')) AS question_l,
               coalesce(neg_risk, false)     AS neg_risk,
               TRY_CAST(end_date AS TIMESTAMP) AS end_ts,
               {CATEGORY_CASE_SQL} AS category
        FROM read_parquet('{MARKETS_PARQUET}')
    """)
    return con


def build_sample(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Stratified (market, day) sample. Returns market_id, category, day,
    n_fills_day, plus market-level fill counts."""
    log("sampling markets (stratified by per-category fill-count quartile)…")
    con.execute(f"""
        CREATE OR REPLACE TABLE mkt_day AS
        SELECT t.market_id, m.category, CAST(t.timestamp AS DATE) AS day,
               COUNT(*) AS n_rows
        FROM read_parquet('{TRADES_GLOB}') t
        JOIN mkt m ON m.market_id = t.market_id
        WHERE t.timestamp >= TIMESTAMP '{BUILD_WINDOW_START}'
          AND t.timestamp <  TIMESTAMP '{BUILD_WINDOW_END}'
          AND t.maker IS NOT NULL AND t.taker IS NOT NULL AND t.maker <> t.taker
          AND t.taker NOT IN {INTERNAL_SET_SQL}
        GROUP BY 1, 2, 3
    """)
    sample = con.execute(f"""
        WITH mk AS (
            SELECT market_id, category, SUM(n_rows) AS n_fills,
                   NTILE(4) OVER (PARTITION BY category ORDER BY SUM(n_rows)) AS size_q
            FROM mkt_day GROUP BY 1, 2
        ),
        chosen_mkts AS (
            SELECT market_id, category, n_fills, size_q,
                   ROW_NUMBER() OVER (PARTITION BY category, size_q
                                      ORDER BY hash(market_id || '{SEED}')) AS rn
            FROM mk
        ),
        kept AS (
            SELECT * FROM chosen_mkts WHERE rn <= {MARKETS_PER_CAT_PER_QUARTILE}
        ),
        chosen_days AS (
            SELECT d.market_id, k.category, d.day, d.n_rows,
                   ROW_NUMBER() OVER (PARTITION BY d.market_id
                                      ORDER BY hash(CAST(d.day AS VARCHAR) || d.market_id || '{SEED}')) AS rn
            FROM mkt_day d JOIN kept k USING (market_id)
        )
        SELECT market_id, category, day, n_rows AS n_fills_day
        FROM chosen_days WHERE rn <= {DAYS_PER_MARKET}
        ORDER BY category, market_id, day
    """).df()
    return sample


def cmd_fetch_mids(args: argparse.Namespace) -> int:
    ensure_dirs()
    con = _connect()
    sample_path = WORK / "sample_market_days.parquet"
    if sample_path.exists() and not args.resample:
        sample = pd.read_parquet(sample_path)
        log(f"reusing sample: {len(sample)} market-days")
    else:
        sample = build_sample(con)
        sample.to_parquet(sample_path, compression="zstd", index=False)
        log(f"sample: {len(sample)} market-days across "
            f"{sample['market_id'].nunique()} markets")

    # token ids actually traded per sampled (market, day)
    con.register("sample_days", sample)
    tok_days = con.execute(f"""
        SELECT DISTINCT t.market_id, CAST(t.timestamp AS DATE) AS day,
               CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                    ELSE t.maker_asset_id END AS token_id
        FROM read_parquet('{TRADES_GLOB}') t
        JOIN sample_days s ON s.market_id = t.market_id
                          AND CAST(t.timestamp AS DATE) = s.day
        WHERE t.maker IS NOT NULL AND t.taker IS NOT NULL AND t.maker <> t.taker
          AND t.taker NOT IN {INTERNAL_SET_SQL}
    """).df()
    log(f"{len(tok_days)} (token, day) histories to fetch (cached ones skipped)")

    client = httpx.Client(timeout=30.0)
    n_ok = n_empty = 0
    for i, r in enumerate(tok_days.itertuples(index=False)):
        day0 = pd.Timestamp(r.day).tz_localize("UTC")
        start_s = int(day0.timestamp()) - 3600          # 1h pre-roll for ffill warmup
        end_s = int(day0.timestamp()) + 86400
        df = fetch_mid_history(str(r.token_id), start_s, end_s, fidelity_min=1, client=client)
        n_ok += 1 if len(df) else 0
        n_empty += 0 if len(df) else 1
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(tok_days)} fetched ({n_empty} empty)")
    client.close()
    log(f"done: {n_ok} non-empty, {n_empty} empty histories")
    return 0


# ============================================================================
# Subcommand: estimate
# ============================================================================
def cmd_estimate(args: argparse.Namespace) -> int:
    ensure_dirs()
    con = _connect()
    sample = pd.read_parquet(WORK / "sample_market_days.parquet")
    con.register("sample_days", sample)

    log("pulling fills for sampled market-days (with 1h feature buffer)…")
    fills = con.execute(f"""
        WITH base AS (
            SELECT t.timestamp AS fill_ts, t.market_id, s.category, s.day,
                   CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                        ELSE t.maker_asset_id END AS token_id,
                   t.maker_side, t.price, t.usd_amount, t.token_amount,
                   t.transaction_hash,
                   m.end_ts
            FROM read_parquet('{TRADES_GLOB}') t
            JOIN sample_days s ON s.market_id = t.market_id
                              AND CAST(t.timestamp AS DATE) = s.day
            JOIN mkt m ON m.market_id = t.market_id
            WHERE t.maker IS NOT NULL AND t.taker IS NOT NULL AND t.maker <> t.taker
              AND t.taker NOT IN {INTERNAL_SET_SQL}
              AND t.maker NOT IN {INTERNAL_SET_SQL}
              AND t.price > 0 AND t.price < 1
        )
        SELECT *,
               epoch(fill_ts) AS fill_epoch,
               CASE WHEN maker_side = 'SELL' THEN 1
                    WHEN maker_side = 'BUY' THEN -1 ELSE 0 END AS s_dir
        FROM base
        WHERE maker_side IN ('BUY', 'SELL')
        ORDER BY market_id, fill_ts
    """).df()
    log(f"  {len(fills):,} candidate fills")

    # trailing trade-rate: distinct tx hashes per market in prior 60 min,
    # via an O(n) sliding window per market. Windows over the SAME pulled set
    # (sampled full days): fills in a day's first trailing hour keep their
    # partial count and are flagged rate_complete = False.
    log("computing trailing trade-rate per fill (sliding window)…")
    fills = fills.sort_values(["market_id", "fill_epoch"]).reset_index(drop=True)
    trail = np.zeros(len(fills), dtype=np.int64)
    t_all = fills["fill_epoch"].to_numpy()
    tx_all = fills["transaction_hash"].to_numpy()
    for _, idx in fills.groupby("market_id", sort=False).indices.items():
        t = t_all[idx]
        tx = tx_all[idx]
        for i in range(len(idx)):
            lo = np.searchsorted(t, t[i] - TRAILING_RATE_WINDOW_S, side="left")
            hi = np.searchsorted(t, t[i], side="left")  # strictly before t_i
            trail[idx[i]] = len(set(tx[lo:hi]))
    fills["trail_tx_60m"] = trail
    day_start = pd.to_datetime(fills["day"]).map(
        lambda d: pd.Timestamp(d, tz="UTC").timestamp())
    fills["rate_complete"] = (fills["fill_epoch"] - day_start) >= TRAILING_RATE_WINDOW_S

    # cap anchors per market-day (deterministic hash order), AFTER features
    rng_key = fills["transaction_hash"].astype(str) + fills["token_id"].astype(str)
    fills["samp_rank"] = (
        pd.util.hash_pandas_object(rng_key, index=False).astype("uint64")
        .groupby([fills["market_id"], fills["day"]]).rank(method="first")
    )
    fills = fills[fills["samp_rank"] <= MAX_FILLS_PER_MARKET_DAY].copy()
    log(f"  {len(fills):,} anchors after per-market-day cap")

    # join to cached mids: per (token, day) sorted-array as-of with ts strictly
    # BEFORE the fill (avoids the fill's own bar contaminating the mid)
    log("joining to cached /prices-history mids (strictly-before as-of)…")
    mids_cache: dict[tuple[str, object], tuple[np.ndarray, np.ndarray]] = {}
    mid_vals = np.full(len(fills), np.nan)
    mid_ages = np.full(len(fills), np.nan)
    fills = fills.reset_index(drop=True)
    for idx, r in enumerate(fills.itertuples(index=False)):
        key = (r.token_id, r.day)
        if key not in mids_cache:
            day0 = pd.Timestamp(r.day).tz_localize("UTC")
            start_s = int(day0.timestamp()) - 3600
            end_s = int(day0.timestamp()) + 86400
            cache = MID_CACHE / f"midhist_{str(r.token_id)[:16]}_{start_s}_{end_s}_f1.parquet"
            if cache.exists():
                h = pd.read_parquet(cache).sort_values("t")
                mids_cache[key] = (h["t"].to_numpy(), h["p"].to_numpy())
            else:
                mids_cache[key] = (np.array([]), np.array([]))
        ts_arr, p_arr = mids_cache[key]
        if len(ts_arr) == 0:
            continue
        i = np.searchsorted(ts_arr, r.fill_epoch, side="left")  # strictly before
        if i > 0:
            mid_vals[idx] = p_arr[i - 1]
            mid_ages[idx] = r.fill_epoch - ts_arr[i - 1]
    fills["mid"] = mid_vals
    fills["mid_age_s"] = mid_ages
    fills["half_spread_c"] = fills["s_dir"] * (fills["price"] - fills["mid"]) * 100.0

    n_no_mid = int(fills["mid"].isna().sum())
    n_stale = int((fills["mid_age_s"] > MID_MAX_AGE_S).sum())
    log(f"  no-mid {n_no_mid:,} | stale(>{MID_MAX_AGE_S}s) {n_stale:,} "
        f"| usable {len(fills) - n_no_mid - n_stale:,}")

    out = WORK / "half_spreads.parquet"
    fills.drop(columns=["samp_rank"]).to_parquet(out, compression="zstd", index=False)
    log(f"wrote {out}")
    return 0


# ============================================================================
# Subcommand: surface
# ============================================================================
def load_estimates(usable_only: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(WORK / "half_spreads.parquet")
    df["ttr_hours"] = (pd.to_datetime(df["end_ts"]) - pd.to_datetime(df["fill_ts"])
                       ).dt.total_seconds() / 3600.0
    if usable_only:
        df = df[df["mid"].notna() & (df["mid_age_s"] <= MID_MAX_AGE_S)].copy()
    return df


def cmd_surface(args: argparse.Namespace) -> int:
    ensure_dirs()
    df = load_estimates()
    log(f"{len(df):,} usable per-fill estimates")

    # activity breakpoints per category (build-sample quartiles, complete-rate rows)
    brk_rows = []
    for cat, sub in df[df["rate_complete"]].groupby("category"):
        q = sub["trail_tx_60m"].quantile([0.25, 0.50, 0.75])
        brk_rows.append({"category": cat, "act_q25": q.loc[0.25],
                         "act_q50": q.loc[0.50], "act_q75": q.loc[0.75]})
    meta = pd.DataFrame(brk_rows)
    breaks = {r["category"]: (r["act_q25"], r["act_q50"], r["act_q75"])
              for _, r in meta.iterrows()}

    df["price_bucket"] = df["mid"].map(price_bucket)
    df["ttr_bucket"] = df["ttr_hours"].map(ttr_bucket)
    df["activity_bucket"] = [
        activity_bucket(t, breaks.get(c) or breaks.get("other") or (0, 1, 10))
        for t, c in zip(df["trail_tx_60m"], df["category"], strict=True)]

    levels = []
    for li, keys in enumerate(FALLBACK_LEVELS):
        g = df.groupby(list(keys))
        agg = g["half_spread_c"].agg(
            n_fills="size",
            median_cents="median",
            p25_cents=lambda s: s.quantile(0.25),
            p75_cents=lambda s: s.quantile(0.75),
            frac_negative=lambda s: float((s < 0).mean()),
        ).reset_index()
        agg["n_markets"] = g["market_id"].nunique().to_numpy()
        agg["level"] = li
        for col in ("category", "price_bucket", "ttr_bucket", "activity_bucket"):
            if col not in agg.columns:
                agg[col] = ""
        levels.append(agg)
    surface = pd.concat(levels, ignore_index=True)[
        ["level", "category", "price_bucket", "ttr_bucket", "activity_bucket",
         "n_fills", "n_markets", "median_cents", "p25_cents", "p75_cents",
         "frac_negative"]]

    surface_path = CSV_OUT / "spread_surface_v1_surface.csv"
    meta_path = CSV_OUT / "spread_surface_v1_activity_breaks.csv"
    surface.to_csv(surface_path, index=False)
    meta.to_csv(meta_path, index=False)
    full = surface[surface["level"] == 0]
    log(f"surface: {len(full)} full cells "
        f"({int((full['n_fills'] >= 50).sum())} with n>=50); wrote {surface_path}")
    print(surface[surface["level"] == 2].to_string(index=False))  # category x price overview
    return 0


# ============================================================================
# Subcommand: validate
# ============================================================================
def cmd_validate(args: argparse.Namespace) -> int:
    ensure_dirs()
    con = duckdb.connect()
    surf = SpreadSurface.load(CSV_OUT / "spread_surface_v1_surface.csv",
                              CSV_OUT / "spread_surface_v1_activity_breaks.csv")
    rows = []
    for rel in VALIDATION_CAPTURES:
        cdir = LIVE_CLOB / rel
        run = rel.split("/")[-1]
        if not cdir.exists():
            log(f"capture missing: {rel} — skipped")
            continue
        log(f"validate on {run}")
        ev = extract_capture_events(cdir, run)
        l1, tr = ev["l1"], ev["trades"]
        if l1.empty:
            continue
        mkts = load_capture_markets(cdir)
        if mkts.empty:
            continue
        # category via the SAME SQL CASE as the tape side
        mk = mkts.assign(slug_l=mkts["slug"].str.lower(),
                         question_l=mkts["question"].str.lower())
        con.register("cap_mkts", mk.drop(columns=["token_ids"]))
        cats = con.execute(
            f"SELECT market_id, {CATEGORY_CASE_SQL} AS category FROM cap_mkts").df()
        mkts = mkts.merge(cats, on="market_id")
        tok2mkt = {t: r for _, r in mkts.iterrows() for t in r["token_ids"]}

        tr = tr.sort_values("ts")
        for tok, sub in l1.groupby("token_id"):
            mk_row = tok2mkt.get(tok)
            if mk_row is None:
                continue
            sub = sub.sort_values("ts")
            t0, t1 = float(sub["ts"].min()), float(sub["ts"].max())
            if t1 - t0 < VALIDATION_WINDOW_S:
                continue
            end_ts = pd.to_datetime(mk_row["end_date"], utc=True, errors="coerce")
            # trailing rate counts the whole MARKET's trades (both tokens);
            # match by token list — event-line market_id is absent in some runs
            trs = tr[tr["token_id"].isin(mk_row["token_ids"])]
            tr_ts = trs["ts"].to_numpy()
            tr_tx = trs["tx_hash"].to_numpy()
            w = t0 + TRAILING_RATE_WINDOW_S  # first window needs a full trailing hour
            while w + VALIDATION_WINDOW_S <= t1:
                win = sub[(sub["ts"] >= w) & (sub["ts"] < w + VALIDATION_WINDOW_S)]
                if len(win) >= 3:
                    true_half_c = float(((win["best_ask"] - win["best_bid"]) / 2)
                                        .median() * 100)
                    mid_lvl = float(((win["best_ask"] + win["best_bid"]) / 2).median())
                    msk = (tr_ts >= w - TRAILING_RATE_WINDOW_S) & (tr_ts < w)
                    rate = len(set(tr_tx[msk])) if msk.any() else 0
                    ttr_h = ((end_ts.timestamp() - w) / 3600.0
                             if pd.notna(end_ts) else None)
                    pred = surf.predict(mid_lvl, ttr_h, rate, mk_row["category"])
                    rows.append({
                        "capture": run, "market_id": mk_row["market_id"],
                        "token_id": tok, "category": mk_row["category"],
                        "family": mk_row["family"], "window_start": w,
                        "price_level": mid_lvl,
                        "ttr_hours": ttr_h, "trail_tx_60m": rate,
                        "true_half_c": true_half_c,
                        "pred_half_c": pred.half_spread_cents,
                        "pred_source": pred.source_level,
                        "n_l1_events": len(win),
                    })
                w += VALIDATION_WINDOW_S
    val = pd.DataFrame(rows)
    val_path = CSV_OUT / "spread_surface_v1_validation_windows.csv"
    val.to_csv(val_path, index=False)
    if val.empty:
        print("VALIDATION: no windows — cannot evaluate gate", file=sys.stderr)
        return 1

    # market-cell aggregation (pre-registered unit): market x predicted cell
    val["cell"] = (val["category"] + "|" + val["price_level"].map(price_bucket) + "|"
                   + val["ttr_hours"].map(ttr_bucket))
    mc = (val.groupby(["capture", "market_id", "token_id", "cell", "category"])
          .agg(true_half_c=("true_half_c", "median"),
               pred_half_c=("pred_half_c", "median"),
               n_windows=("true_half_c", "size"))
          .reset_index())
    mc["abs_err_c"] = (mc["pred_half_c"] - mc["true_half_c"]).abs()
    medae = float(mc["abs_err_c"].median())
    # Spearman without scipy: Pearson on ranks
    rr = mc[["true_half_c", "pred_half_c"]].rank()
    spear = float(rr["true_half_c"].corr(rr["pred_half_c"]))
    mc_path = CSV_OUT / "spread_surface_v1_validation_market_cells.csv"
    mc.to_csv(mc_path, index=False)

    per_cat = (mc.groupby("category")
               .agg(n_cells=("abs_err_c", "size"),
                    medae_c=("abs_err_c", "median"),
                    spearman=("true_half_c",
                              lambda s: float(s.rank().corr(
                                  mc.loc[s.index, "pred_half_c"].rank()))))
               .reset_index())
    print(per_cat.to_string(index=False))
    print(f"\nmarket-cells: {len(mc)} | MedAE = {medae:.3f}c (gate <= 1c) "
          f"| Spearman = {spear:.3f} (gate >= 0.6)")
    gate = "PASS" if (medae <= 1.0 and spear >= 0.6) else "FAIL"
    print(f"VALIDATION GATE: {gate}")
    print(f"wrote {val_path}\n      {mc_path}")
    return 0


# ============================================================================
# Subcommand: validate-tradetime  (Block SPREAD-1b)
# ============================================================================
def _slice_metrics(mc: pd.DataFrame, label: str) -> dict:
    """Gate metrics for one slice of validation market-cells (SPREAD-1b)."""
    share, wins, nontied = sign_test_share(mc["abs_err_c"].to_numpy(),
                                           mc["abs_err_flat3_c"].to_numpy())
    rr = mc[["true_tt_half_c", "pred_half_c"]].rank()
    spear_all = float(rr["true_tt_half_c"].corr(rr["pred_half_c"])) if len(mc) > 2 else float("nan")
    uniq = mc[~mc["true_tt_half_c"].duplicated(keep=False)]
    if len(uniq) > 2:
        ru = uniq[["true_tt_half_c", "pred_half_c"]].rank()
        spear_uniq = float(ru["true_tt_half_c"].corr(ru["pred_half_c"]))
    else:
        spear_uniq = float("nan")
    return {
        "slice": label, "n_cells": len(mc),
        "medae_surface_c": float(mc["abs_err_c"].median()),
        "medae_flat3_c": float(mc["abs_err_flat3_c"].median()),
        "medae_flat15_c": float(mc["abs_err_flat15_c"].median()),
        "medae_flat1_c": float(mc["abs_err_flat1_c"].median()),
        "medae_hybrid_bounce_c": float(mc["abs_err_hyb_bounce_c"].median()),
        "medae_hybrid_roll_c": float(mc["abs_err_hyb_roll_c"].median()),
        "signtest_share_vs_flat3": share,
        "signtest_wins": wins, "signtest_nontied": nontied,
        "spearman_all": spear_all,
        "spearman_nontied_targets": spear_uniq,
        "n_nontied_targets": len(uniq),
    }


def cmd_validate_tradetime(args: argparse.Namespace) -> int:
    """Block SPREAD-1b: score the FROZEN surface against the trade-time quoted
    half-spread. Identical captures, window grid, features, and market-cell
    definition as `validate`; only the truth changes."""
    ensure_dirs()
    con = duckdb.connect()
    surf = SpreadSurface.load(CSV_OUT / "spread_surface_v1_surface.csv",
                              CSV_OUT / "spread_surface_v1_activity_breaks.csv")
    xck = pd.read_csv(CSV_OUT / "spread_surface_v1_diag_crosschecks.csv")
    bounce_lookup = {(r.category, r.price_bucket): float(r.bounce_half_c)
                     for r in xck.itertuples()}
    roll_lookup = {(r.category, r.price_bucket): float(r.roll_half_c)
                   for r in xck.itertuples()}

    rows = []
    n_win_total = n_win_l1ok = n_win_traded = 0
    for rel in VALIDATION_CAPTURES:
        cdir = LIVE_CLOB / rel
        run = rel.split("/")[-1]
        if not cdir.exists():
            log(f"capture missing: {rel} — skipped")
            continue
        log(f"validate-tradetime on {run}")
        ev = extract_capture_events(cdir, run)
        l1, tr = ev["l1"], ev["trades"]
        if l1.empty:
            continue
        mkts = load_capture_markets(cdir)
        if mkts.empty:
            continue
        mk = mkts.assign(slug_l=mkts["slug"].str.lower(),
                         question_l=mkts["question"].str.lower())
        con.register("cap_mkts", mk.drop(columns=["token_ids"]))
        cats = con.execute(
            f"SELECT market_id, {CATEGORY_CASE_SQL} AS category FROM cap_mkts").df()
        mkts = mkts.merge(cats, on="market_id")
        tok2mkt = {t: r for _, r in mkts.iterrows() for t in r["token_ids"]}

        tr = tr.sort_values("ts")
        for tok, sub in l1.groupby("token_id"):
            mk_row = tok2mkt.get(tok)
            if mk_row is None:
                continue
            sub = sub.sort_values("ts")
            t0, t1 = float(sub["ts"].min()), float(sub["ts"].max())
            if t1 - t0 < VALIDATION_WINDOW_S:
                continue
            end_ts = pd.to_datetime(mk_row["end_date"], utc=True, errors="coerce")
            # trailing rate counts the whole MARKET's trades (both tokens) —
            # identical to `validate`; the TRUTH below uses only this token's
            # own prints against its own quotes
            trs = tr[tr["token_id"].isin(mk_row["token_ids"])]
            tr_ts = trs["ts"].to_numpy()
            tr_tx = trs["tx_hash"].to_numpy()
            tok_tr_ts = trs.loc[trs["token_id"] == tok, "ts"].to_numpy()
            l1_ts_tok = sub["ts"].to_numpy()
            l1_half_tok = ((sub["best_ask"] - sub["best_bid"]) / 2 * 100).to_numpy()
            w = t0 + TRAILING_RATE_WINDOW_S
            while w + VALIDATION_WINDOW_S <= t1:
                n_win_total += 1
                win = sub[(sub["ts"] >= w) & (sub["ts"] < w + VALIDATION_WINDOW_S)]
                if len(win) >= 3:
                    n_win_l1ok += 1
                    tt_in_win = tok_tr_ts[(tok_tr_ts >= w)
                                          & (tok_tr_ts < w + VALIDATION_WINDOW_S)]
                    tt_half, tt_age = tradetime_quoted_half_spreads(
                        l1_ts_tok, l1_half_tok, tt_in_win)
                    if len(tt_half) >= 1:
                        n_win_traded += 1
                        true_ta_c = float(((win["best_ask"] - win["best_bid"]) / 2)
                                          .median() * 100)
                        mid_lvl = float(((win["best_ask"] + win["best_bid"]) / 2).median())
                        msk = (tr_ts >= w - TRAILING_RATE_WINDOW_S) & (tr_ts < w)
                        rate = len(set(tr_tx[msk])) if msk.any() else 0
                        ttr_h = ((end_ts.timestamp() - w) / 3600.0
                                 if pd.notna(end_ts) else None)
                        pred = surf.predict(mid_lvl, ttr_h, rate, mk_row["category"])
                        pb = price_bucket(mid_lvl)
                        hyb_b, swap_b = hybrid_prediction(
                            pred.half_spread_cents, pred.cell_frac_negative,
                            mk_row["category"], pb, bounce_lookup)
                        hyb_r, swap_r = hybrid_prediction(
                            pred.half_spread_cents, pred.cell_frac_negative,
                            mk_row["category"], pb, roll_lookup)
                        rows.append({
                            "capture": run, "market_id": mk_row["market_id"],
                            "token_id": tok, "category": mk_row["category"],
                            "family": mk_row["family"], "window_start": w,
                            "price_level": mid_lvl,
                            "ttr_hours": ttr_h, "trail_tx_60m": rate,
                            "true_tt_half_c": float(np.median(tt_half)),
                            "true_ta_half_c": true_ta_c,
                            "n_trades": int(len(tt_half)),
                            "med_quote_age_s": float(np.median(tt_age)),
                            "pred_half_c": pred.half_spread_cents,
                            "pred_source": pred.source_level,
                            "pred_frac_negative": pred.cell_frac_negative,
                            "hyb_bounce_c": hyb_b, "hyb_bounce_swapped": swap_b,
                            "hyb_roll_c": hyb_r, "hyb_roll_swapped": swap_r,
                            "n_l1_events": len(win),
                        })
                w += VALIDATION_WINDOW_S
    val = pd.DataFrame(rows)
    val_path = CSV_OUT / "spread_surface_v1b_validation_windows.csv"
    val.to_csv(val_path, index=False)
    log(f"windows: {n_win_total} on grid, {n_win_l1ok} with >=3 L1 events, "
        f"{n_win_traded} also with >=1 quoted trade print (kept)")
    if val.empty:
        print("VALIDATION-TRADETIME: no windows — cannot evaluate gate", file=sys.stderr)
        return 1

    # market-cell aggregation — same unit as `validate`
    val["cell"] = (val["category"] + "|" + val["price_level"].map(price_bucket) + "|"
                   + val["ttr_hours"].map(ttr_bucket))
    mc = (val.groupby(["capture", "market_id", "token_id", "cell", "category"])
          .agg(true_tt_half_c=("true_tt_half_c", "median"),
               true_ta_half_c=("true_ta_half_c", "median"),
               pred_half_c=("pred_half_c", "median"),
               hyb_bounce_c=("hyb_bounce_c", "median"),
               hyb_roll_c=("hyb_roll_c", "median"),
               any_swapped=("hyb_bounce_swapped", "any"),
               n_windows=("true_tt_half_c", "size"),
               n_trades=("n_trades", "sum"))
          .reset_index())
    mc["abs_err_c"] = (mc["pred_half_c"] - mc["true_tt_half_c"]).abs()
    mc["abs_err_flat3_c"] = (FLAT_BASELINE_C - mc["true_tt_half_c"]).abs()
    mc["abs_err_flat15_c"] = (FLAT_SENSITIVITY_C[0] - mc["true_tt_half_c"]).abs()
    mc["abs_err_flat1_c"] = (FLAT_SENSITIVITY_C[1] - mc["true_tt_half_c"]).abs()
    mc["abs_err_hyb_bounce_c"] = (mc["hyb_bounce_c"] - mc["true_tt_half_c"]).abs()
    mc["abs_err_hyb_roll_c"] = (mc["hyb_roll_c"] - mc["true_tt_half_c"]).abs()
    mc_path = CSV_OUT / "spread_surface_v1b_validation_market_cells.csv"
    mc.to_csv(mc_path, index=False)

    # gate summary across slices
    fast = mc[mc["category"].isin(FAST_CATEGORIES)]
    slow = mc[~mc["category"].isin(FAST_CATEGORIES)]
    summary = [_slice_metrics(mc, "pooled"),
               _slice_metrics(fast, "fast_crypto"),
               _slice_metrics(slow, "slow_categories")]
    for cat, sub in mc.groupby("category"):
        summary.append(_slice_metrics(sub, f"cat:{cat}"))
    for cap, sub in mc.groupby("capture"):
        summary.append(_slice_metrics(sub, f"capture:{cap}"))
    summ = pd.DataFrame(summary)
    summ_path = CSV_OUT / "spread_surface_v1b_gate_summary.csv"
    summ.to_csv(summ_path, index=False)

    # hybrid arm: head-to-head on the cells the swap actually touches
    hyb_rows = []
    aff = mc[mc["any_swapped"]]
    for name, err_col in (("bounce", "abs_err_hyb_bounce_c"),
                          ("roll", "abs_err_hyb_roll_c")):
        for label, sub in (("affected_cells", aff), ("pooled", mc)):
            share, wins, nontied = sign_test_share(sub[err_col].to_numpy(),
                                                   sub["abs_err_c"].to_numpy())
            hyb_rows.append({
                "replacement": name, "slice": label, "n_cells": len(sub),
                "medae_hybrid_c": float(sub[err_col].median()) if len(sub) else float("nan"),
                "medae_surface_c": float(sub["abs_err_c"].median()) if len(sub) else float("nan"),
                "hybrid_beats_surface_share": share,
                "wins": wins, "nontied": nontied,
            })
    hyb = pd.DataFrame(hyb_rows)
    hyb_path = CSV_OUT / "spread_surface_v1b_hybrid_comparison.csv"
    hyb.to_csv(hyb_path, index=False)

    # verdict
    pooled = summary[0]
    fastm = summary[1]
    bar_a = pooled["medae_surface_c"] <= GATE_1B_POOLED_MEDAE_C
    bar_b = fastm["medae_surface_c"] <= GATE_1B_FAST_MEDAE_C
    bar_c = pooled["signtest_share_vs_flat3"] >= GATE_1B_SIGNTEST_SHARE
    print(summ.to_string(index=False))
    print("\nhybrid arm:")
    print(hyb.to_string(index=False))
    print(f"\nmarket-cells: {len(mc)} (windows kept: {n_win_traded}/{n_win_l1ok} "
          f"with >=3 L1 events; {n_win_total} on grid)")
    print(f"(a) pooled MedAE       = {pooled['medae_surface_c']:.3f}c "
          f"(gate <= {GATE_1B_POOLED_MEDAE_C}c)  -> {'PASS' if bar_a else 'FAIL'}")
    print(f"(b) fast-crypto MedAE  = {fastm['medae_surface_c']:.3f}c "
          f"(gate <= {GATE_1B_FAST_MEDAE_C}c) -> {'PASS' if bar_b else 'FAIL'}")
    print(f"(c) sign test vs flat3 = {pooled['signtest_share_vs_flat3']:.3f} "
          f"({pooled['signtest_wins']}/{pooled['signtest_nontied']} non-tied; "
          f"gate >= {GATE_1B_SIGNTEST_SHARE}) -> {'PASS' if bar_c else 'FAIL'}")
    print(f"(d) Spearman diagnostic: all-cells {pooled['spearman_all']:.3f}, "
          f"non-tied targets {pooled['spearman_nontied_targets']:.3f} "
          f"(n={pooled['n_nontied_targets']}) — NOT a bar")
    gate = "PASS" if (bar_a and bar_b and bar_c) else "FAIL"
    print(f"SPREAD-1b GATE: {gate}")
    print(f"wrote {val_path}\n      {mc_path}\n      {summ_path}\n      {hyb_path}")
    return 0


# ============================================================================
# Subcommand: diagnose
# ============================================================================
def cmd_diagnose(args: argparse.Namespace) -> int:
    ensure_dirs()
    df = load_estimates()

    # (a) negative-rate overall + per price bucket x category
    df["price_bucket"] = df["mid"].map(price_bucket)
    neg_overall = float((df["half_spread_c"] < 0).mean())
    neg = (df.groupby(["category", "price_bucket"])["half_spread_c"]
           .agg(n="size", frac_negative=lambda s: float((s < 0).mean()),
                median_c="median").reset_index())

    # (b) size-quantile split (book-walking check)
    df["size_q"] = pd.qcut(df["usd_amount"], 4, labels=["sz_q1", "sz_q2", "sz_q3", "sz_q4"])
    size_split = (df.groupby(["category", "size_q"], observed=True)["half_spread_c"]
                  .agg(n="size", median_c="median",
                       p75_c=lambda s: s.quantile(0.75)).reset_index())

    # (c) tape-only cross-checks per category x price bucket:
    #     bid-ask bounce: consecutive opposite-sign fills within 60s, |dmid| < 1 tick
    #     Roll: 2*sqrt(-cov(dp_t, dp_{t-1})) on per-token fill-price changes
    bounce_rows, roll_rows = [], []
    for (cat, pb), sub in df.groupby(["category", "price_bucket"]):
        sub = sub.sort_values(["token_id", "fill_epoch"])
        by_tok = sub.groupby("token_id")
        b_est, r_est = [], []
        for _, g in by_tok:
            if len(g) < 10:
                continue
            p = g["price"].to_numpy()
            s = g["s_dir"].to_numpy()
            t = g["fill_epoch"].to_numpy()
            m = g["mid"].to_numpy()
            dt = np.diff(t)
            flip = (s[1:] * s[:-1]) == -1
            dmid_ok = np.abs(np.diff(m)) < 0.01
            ok = flip & (dt <= 60) & dmid_ok
            if ok.sum() >= 3:
                # bounce half-spread = |p_t - p_{t-1}| / 2 on qualifying pairs
                b_est.extend((np.abs(np.diff(p))[ok] / 2 * 100).tolist())
            dp = np.diff(p)
            if len(dp) >= 10:
                cov = float(np.cov(dp[1:], dp[:-1])[0, 1])
                if cov < 0:
                    r_est.append(np.sqrt(-cov) * 100)  # HALF-spread in cents
        if b_est:
            bounce_rows.append({"category": cat, "price_bucket": pb,
                                "n_pairs": len(b_est),
                                "bounce_half_c": float(np.median(b_est))})
        if r_est:
            roll_rows.append({"category": cat, "price_bucket": pb,
                              "n_tokens": len(r_est),
                              "roll_half_c": float(np.median(r_est))})

    bounce = pd.DataFrame(bounce_rows)
    roll = pd.DataFrame(roll_rows)
    surf_l2 = pd.read_csv(CSV_OUT / "spread_surface_v1_surface.csv",
                          keep_default_na=False)
    surf_l2 = surf_l2[surf_l2["level"] == 2][
        ["category", "price_bucket", "median_cents", "n_fills"]]
    xck = surf_l2.merge(bounce, on=["category", "price_bucket"], how="left") \
                 .merge(roll, on=["category", "price_bucket"], how="left")

    neg_path = CSV_OUT / "spread_surface_v1_diag_negative_rates.csv"
    size_path = CSV_OUT / "spread_surface_v1_diag_size_split.csv"
    xck_path = CSV_OUT / "spread_surface_v1_diag_crosschecks.csv"
    neg.to_csv(neg_path, index=False)
    size_split.to_csv(size_path, index=False)
    xck.to_csv(xck_path, index=False)
    print(f"overall negative-estimate rate: {neg_overall:.3f}")
    print(size_split.to_string(index=False))
    print(xck.to_string(index=False))
    print(f"wrote {neg_path}\n      {size_path}\n      {xck_path}")
    return 0


# ============================================================================
# Subcommand: charts
# ============================================================================
def cmd_charts(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs()

    # (1) predicted vs true scatter on validation market-cells
    mc = pd.read_csv(CSV_OUT / "spread_surface_v1_validation_market_cells.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    cats = sorted(mc["category"].unique())
    cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(cats):
        sub = mc[mc["category"] == cat]
        ax.scatter(sub["true_half_c"], sub["pred_half_c"], s=22, alpha=0.7,
                   color=cmap(i % 10), label=f"{cat} (n={len(sub)})")
    lim = max(float(mc["true_half_c"].quantile(0.98)),
              float(mc["pred_half_c"].max()), 3.0) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.fill_between([0, lim], [max(0, v - 1) for v in (0, lim)],
                    [v + 1 for v in (0, lim)], color="grey", alpha=0.12,
                    label="±1c gate band")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("true quoted half-spread (cents, live_clob L1 median)")
    ax.set_ylabel("surface-predicted half-spread (cents)")
    ax.set_title("SPREAD-1 validation: predicted vs true half-spread per market-cell")
    ax.legend(fontsize=8, loc="upper left")
    p1 = PLOTS_OUT / "spread_surface_v1_pred_vs_true.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # (2) surface heatmap: price-level x activity quartile (pooled cats + TTR)
    df = load_estimates()
    meta = pd.read_csv(CSV_OUT / "spread_surface_v1_activity_breaks.csv")
    breaks = {r["category"]: (r["act_q25"], r["act_q50"], r["act_q75"])
              for _, r in meta.iterrows()}
    df["price_bucket"] = df["mid"].map(price_bucket)
    df["activity_bucket"] = [
        activity_bucket(t, breaks.get(c) or breaks.get("other") or (0, 1, 10))
        for t, c in zip(df["trail_tx_60m"], df["category"], strict=True)]
    from lib.spread_surface import PRICE_BUCKET_LABELS
    piv = (df.groupby(["price_bucket", "activity_bucket"])["half_spread_c"]
           .median().unstack()
           .reindex(index=PRICE_BUCKET_LABELS, columns=ACTIVITY_LABELS))
    npiv = (df.groupby(["price_bucket", "activity_bucket"]).size().unstack()
            .reindex(index=PRICE_BUCKET_LABELS, columns=ACTIVITY_LABELS))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ACTIVITY_LABELS)),
                  ["Q1 (quiet)", "Q2", "Q3", "Q4 (busy)"])
    ax.set_yticks(range(len(PRICE_BUCKET_LABELS)), PRICE_BUCKET_LABELS)
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            v = piv.iat[r, c]
            n = npiv.iat[r, c]
            if pd.notna(v):
                ax.text(c, r, f"{v:.2f}\n(n={int(n)})", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if v < float(np.nanmax(piv.to_numpy())) * 0.6 else "black")
    fig.colorbar(im, label="median half-spread estimate (cents)")
    ax.set_xlabel("trailing 60-min activity quartile (per-category breakpoints)")
    ax.set_ylabel("price-level bucket (token mid)")
    ax.set_title("SPREAD-1 surface: median trade-anchored half-spread\n"
                 "(pooled over categories and time-to-resolution)")
    p2 = PLOTS_OUT / "spread_surface_v1_heatmap_price_activity.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"wrote {p1}\n      {p2}")
    return 0


# ============================================================================
# Subcommand: charts-1b
# ============================================================================
def cmd_charts_1b(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs()
    mc = pd.read_csv(CSV_OUT / "spread_surface_v1b_validation_market_cells.csv")
    cmap = plt.get_cmap("tab10")
    cats = sorted(mc["category"].unique())

    # (1) predicted vs TRADE-TIME true, with the ±1c gate band
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cat in enumerate(cats):
        sub = mc[mc["category"] == cat]
        ax.scatter(sub["true_tt_half_c"], sub["pred_half_c"], s=22, alpha=0.7,
                   color=cmap(i % 10), label=f"{cat} (n={len(sub)})")
    lim = max(float(mc["true_tt_half_c"].quantile(0.98)),
              float(mc["pred_half_c"].max()), 3.0) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.fill_between([0, lim], [max(0, v - 1) for v in (0, lim)],
                    [v + 1 for v in (0, lim)], color="grey", alpha=0.12,
                    label="±1c gate band")
    ax.axhline(FLAT_BASELINE_C, color="firebrick", lw=1, ls=":",
               label=f"flat {FLAT_BASELINE_C:.0f}c incumbent")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("true TRADE-TIME quoted half-spread (cents, as-of trade prints)")
    ax.set_ylabel("surface-predicted half-spread (cents)")
    ax.set_title("SPREAD-1b re-gate: predicted vs trade-time quoted half-spread\n"
                 "per market-cell (frozen surface, new target)")
    ax.legend(fontsize=8, loc="upper left")
    p1 = PLOTS_OUT / "spread_surface_v1b_pred_vs_true.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # (2) truth compression: trade-time vs time-averaged quoted half-spread
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cat in enumerate(cats):
        sub = mc[mc["category"] == cat]
        ax.scatter(sub["true_ta_half_c"], sub["true_tt_half_c"], s=22, alpha=0.7,
                   color=cmap(i % 10), label=f"{cat} (n={len(sub)})")
    lim2 = max(float(mc["true_ta_half_c"].quantile(0.98)),
               float(mc["true_tt_half_c"].quantile(0.98)), 3.0) * 1.1
    ax.plot([0, lim2], [0, lim2], "k--", lw=1, label="y = x")
    ax.set_xlim(0, lim2)
    ax.set_ylim(0, lim2)
    ax.set_xlabel("time-averaged quoted half-spread (cents, SPREAD-1 target)")
    ax.set_ylabel("trade-time quoted half-spread (cents, SPREAD-1b target)")
    ax.set_title("Same market-cells, two targets: books are tighter when trades\n"
                 "actually happen (points below y = x)")
    ax.legend(fontsize=8, loc="upper left")
    p2 = PLOTS_OUT / "spread_surface_v1b_truth_compression.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"wrote {p1}\n      {p2}")
    return 0


# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("semantics")
    p_f = sub.add_parser("fetch-mids")
    p_f.add_argument("--resample", action="store_true",
                     help="rebuild the (market, day) sample instead of reusing")
    sub.add_parser("estimate")
    sub.add_parser("surface")
    sub.add_parser("validate")
    sub.add_parser("validate-tradetime")
    sub.add_parser("diagnose")
    sub.add_parser("charts")
    sub.add_parser("charts-1b")
    args = ap.parse_args(argv)
    return {"semantics": cmd_semantics, "fetch-mids": cmd_fetch_mids,
            "estimate": cmd_estimate, "surface": cmd_surface,
            "validate": cmd_validate, "validate-tradetime": cmd_validate_tradetime,
            "diagnose": cmd_diagnose,
            "charts": cmd_charts, "charts-1b": cmd_charts_1b}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
