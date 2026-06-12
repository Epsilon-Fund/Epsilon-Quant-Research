"""SPCX Polymarket executable-arbitrage check (Block S8, extends S5).

PLAIN ENGLISH
-------------
The 2026-06-12 09:48 UTC PM-PDF poll flagged a 0.1c No-IPO ask and two buckets
diverging from the PCHIP ladder ([2-2.5T] +8.1pp, [2.5-3T] -6.3pp). This script
answers one question: is any of that a *riskless, executable* arbitrage net of
fees and book depth, and how much size does it absorb? It is a measurement tool
-- it places no orders.

Three arb classes, all computed from RAW EXECUTABLE QUOTES ONLY (level-by-level
asks to buy, bids to sell; never mids, never the PCHIP survivor):

  1. NegRisk basket on the mutually-exclusive bucket group (7 cap buckets + the
     No-IPO leg): buy-all-YES (sum of asks < $1), buy-all-NO (sum of NO asks <
     $N-1, the no-mint "sell the basket" equivalent), and NegRisk mint-and-sell
     (mint a full YES set for $1 + gas, hit the bids).
  2. Ladder monotonicity: for strikes a < b, the lock = buy YES(>a) + buy NO(>b);
     payoff floor $1, so any cost < $1 is riskless. Scans adjacent + all pairs.
  3. Ladder-vs-bucket boxes: ONLY unions whose edges exist as real ladder strikes
     ([<1T], [1-2T], [2-3T], [>=3T]; the half-T bucket edges 1.5/2.5/3.5T have NO
     ladder instrument, so single-bucket "gaps vs the PCHIP ladder" are model
     artifacts by construction, not arb). Both directions per union.

Payoff floors are not hand-asserted: every combo's payoff is brute-force
enumerated over cap states (every bracket edge exactly, midpoints between edges,
below-min, above-max) plus the No-IPO state, using the resolution semantics from
the live Gamma descriptions (ladder "above $K" strict; buckets lo-inclusive /
hi-exclusive via "exactly between brackets -> higher bracket"; No-IPO => all cap
legs NO). Exact-edge states where the payoff dips below the robust floor are
reported as measure-zero boundary caveats, never silently used.

Fees use the two repo-canonical schedules (mm_negrisk_consistency_analyze.py):
  repo  : fee/share = 0.05 * p * (1-p)        (K5-STRESS / A14c convention)
  harsh : fee/share = 0.10 * min(p, 1-p)      (Gamma-declared 1000 bps base)
CLOB taker orders are relayer-gasless on Polymarket; the only gas-bearing path
is NegRisk mint+convert (--gas-usd, default $0.10 total). Gross and net are both
reported; verdicts use net.

Co-resolution guard: every multi-leg combo asserts its legs share one resolution
key (first-trading-day-close underlying + the Dec 31, 2027 no-IPO deadline),
parsed from each market's own description. Mixed-resolution combos are reported
as RV-with-risk and excluded from arb verdicts.

RUN
---
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/spcx_pm_arb_check.py               # capture + analyze
    PYTHONPATH=. uv run python scripts/spcx_pm_arb_check.py --watch 60 --polls 5
    PYTHONPATH=. uv run python scripts/spcx_pm_arb_check.py --from-parquet latest   # offline
Tests: PYTHONPATH=. uv run pytest tests/test_spcx_pm_arb_check.py -q

OUTPUTS (append-only)
---------------------
    data/analysis/spcx_convergence/pm_arb_books/books_<ts>.parquet   full L2 books
    data/analysis/spcx_convergence/pm_arb_books/meta_<ts>.json       leg/resolution map
    data/analysis/spcx_convergence/pm_arb/summary_<ts>.csv           one row per combo
    data/analysis/spcx_convergence/pm_arb/depth_<ts>.csv             depth walk segments
    data/analysis/spcx_convergence/pm_arb/legs_<ts>.csv              per-leg TOB + mirror check
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

LADDER_EVENT_SLUG = "spacex-ipo-closing-market-cap-above"
BUCKET_EVENT_SLUG = "spacex-ipo-closing-market-cap"

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "analysis" / "spcx_convergence"
BOOKS_DIR = DEFAULT_OUT_DIR / "pm_arb_books"
RESULTS_DIR = DEFAULT_OUT_DIR / "pm_arb"

# Repo-canonical fee schedules (same constants as mm_negrisk_consistency_analyze.py).
FEE_RATE_REPO = 0.05    # fee/share = 0.05 * p * (1-p)
FEE_RATE_HARSH = 0.10   # fee/share = 0.10 * min(p, 1-p)  (Gamma-declared 1000bps base)
GAS_USD_DEFAULT = 0.10  # NegRisk mint + convert, total per *batch* (Polygon, generous)

# Single live-fee model for the "how much can I pull out of the book" walk. Polymarket's
# documented CLOB taker fee is rate * min(price, 1-price) per share, rate = feeBps/10000.
# Observed CLOB fills have been fee-free (rate 0) even though these markets DECLARE 1000bps;
# the rate is therefore one tunable knob, stated on the dashboard, not two schedules.
FEE_BPS_DEFAULT = 0.0   # what Polymarket charges in practice; 1000 = declared face value

MIN_ORDER_SHARES = 5.0  # CLOB minimum order size on every SPCX leg (dust threshold)
DUST_NET_USD = 10.0     # net profit below this => uninvestable dust, stated in output

EPS = 1e-9

_STRIKE_RE = re.compile(r"above \$([0-9.]+)T", re.IGNORECASE)
_BETWEEN_RE = re.compile(r"between \$([0-9.]+)T and \$([0-9.]+)T", re.IGNORECASE)
_LESS_RE = re.compile(r"less than \$([0-9.]+)T", re.IGNORECASE)
_ATLEAST_RE = re.compile(r"at least \$([0-9.]+)T", re.IGNORECASE)
_NOIPO_DEADLINE_RE = re.compile(r"(?:by|before)\s+(December\s+31,\s*\d{4})", re.IGNORECASE)


# --------------------------------------------------------------------------------------
# Fees
# --------------------------------------------------------------------------------------
def fee_per_share(price: float, schedule: str) -> float:
    """Taker fee in $ for one share traded at `price` under the named schedule."""
    if schedule == "repo":
        return FEE_RATE_REPO * price * (1.0 - price)
    if schedule == "harsh":
        return FEE_RATE_HARSH * min(price, 1.0 - price)
    raise ValueError(f"unknown fee schedule: {schedule}")


def taker_fee(price: float, fee_bps: float) -> float:
    """Polymarket-documented CLOB taker fee for one share at `price`:
    (fee_bps / 10000) * min(price, 1 - price). One knob, not a named schedule."""
    return (fee_bps / 10000.0) * min(price, 1.0 - price)


# --------------------------------------------------------------------------------------
# Phase 0 — metadata / market-structure map (live Gamma + CLOB; don't assume)
# --------------------------------------------------------------------------------------
def _resolution_key(description: str, condition_id: str) -> str:
    """Co-resolution fingerprint parsed from the market's own resolution text.
    Legs may only be combined risk-free when their keys are identical."""
    desc = description or ""
    first_day = bool(re.search(r"first (?:trading day|day of trading)", desc, re.IGNORECASE))
    m = _NOIPO_DEADLINE_RE.search(desc)
    deadline = re.sub(r"\s+", " ", m.group(1)) if m else None
    if not first_day or deadline is None:
        return f"UNKNOWN:{condition_id}"  # never matches anything else
    return f"first_day_close_cap|no_ipo_by:{deadline}"


def parse_leg(market: dict, event_slug: str) -> dict | None:
    """One Gamma market dict -> leg record (both tokens, structure class, res key)."""
    q = market.get("question", "")
    try:
        outcomes = json.loads(market.get("outcomes", "[]"))
        tokens = json.loads(market.get("clobTokenIds", "[]"))
    except (ValueError, TypeError):
        return None
    tok = {str(o).strip().lower(): t for o, t in zip(outcomes, tokens)}
    if "yes" not in tok or "no" not in tok:
        return None
    leg = {
        "event_slug": event_slug,
        "question": q,
        "condition_id": market.get("conditionId"),
        "neg_risk": bool(market.get("negRisk")),
        "neg_risk_market_id": market.get("negRiskMarketID"),
        "token_yes": tok["yes"],
        "token_no": tok["no"],
        "end_date_iso": market.get("endDateIso") or market.get("endDate"),
        "active": market.get("active"),
        "closed": market.get("closed"),
        "accepting_orders": market.get("acceptingOrders"),
        "res_key": _resolution_key(market.get("description", ""), market.get("conditionId") or q),
    }
    if event_slug == LADDER_EVENT_SLUG:
        m = _STRIKE_RE.search(q)
        if not m:
            return None
        leg.update(kind="ladder", strike_t=float(m.group(1)), lo=None, hi=None,
                   label=f">{float(m.group(1)):g}T")
        return leg
    if "not IPO" in q:
        leg.update(kind="no_ipo", strike_t=None, lo=None, hi=None, label="No-IPO")
        return leg
    if (b := _BETWEEN_RE.search(q)):
        lo, hi = float(b.group(1)), float(b.group(2))
    elif (l := _LESS_RE.search(q)):
        lo, hi = 0.0, float(l.group(1))
    elif (a := _ATLEAST_RE.search(q)):
        lo, hi = float(a.group(1)), math.inf
    else:
        return None
    label = (f"<{hi:g}T" if lo == 0.0 else (f">={lo:g}T" if math.isinf(hi) else f"{lo:g}-{hi:g}T"))
    leg.update(kind="bucket", strike_t=None, lo=lo, hi=hi, label=label)
    return leg


def fetch_arb_metadata(timeout: float = 30.0) -> dict:
    """Phase 0: resolve both Gamma events into a structure map with both tokens per leg,
    NegRisk classification, and per-leg resolution keys. CLOB per-condition metadata
    (tick, min order, fee fields) is merged in for the record."""
    import httpx

    legs: list[dict] = []
    with httpx.Client(base_url=GAMMA_BASE, timeout=timeout) as client:
        for slug in (LADDER_EVENT_SLUG, BUCKET_EVENT_SLUG):
            r = client.get("/events", params={"slug": slug})
            r.raise_for_status()
            events = r.json()
            if not events:
                raise RuntimeError(f"gamma event not found: {slug}")
            for mkt in events[0].get("markets", []):
                leg = parse_leg(mkt, slug)
                if leg:
                    legs.append(leg)
    with httpx.Client(base_url=CLOB_BASE, timeout=timeout) as client:
        for leg in legs:
            try:
                r = client.get(f"/markets/{leg['condition_id']}")
                r.raise_for_status()
                m = r.json()
                leg.update(tick=float(m.get("minimum_tick_size") or 0.01),
                           min_order=float(m.get("minimum_order_size") or MIN_ORDER_SHARES),
                           maker_base_fee=m.get("maker_base_fee"),
                           taker_base_fee=m.get("taker_base_fee"),
                           clob_neg_risk=bool(m.get("neg_risk")))
            except Exception:
                leg.update(tick=0.01, min_order=MIN_ORDER_SHARES,
                           maker_base_fee=None, taker_base_fee=None, clob_neg_risk=None)
    ladder = sorted([l for l in legs if l["kind"] == "ladder"], key=lambda d: d["strike_t"])
    buckets = sorted([l for l in legs if l["kind"] == "bucket"], key=lambda d: d["lo"])
    no_ipo = next((l for l in legs if l["kind"] == "no_ipo"), None)
    return {"fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "ladder": ladder, "buckets": buckets, "no_ipo": no_ipo}


# --------------------------------------------------------------------------------------
# Phase 1 — full executable books (all levels, both sides, both tokens)
# --------------------------------------------------------------------------------------
def fetch_full_books(token_ids: list[str], timeout: float = 30.0) -> dict[str, dict]:
    """Full CLOB book per token: asks ascending, bids descending, every level kept.
    Batch POST /books with per-token GET /book fallback."""
    import httpx

    def _parse(book: dict) -> dict:
        bids = sorted(((float(x["price"]), float(x.get("size", 0) or 0))
                       for x in book.get("bids", []) or []), key=lambda t: -t[0])
        asks = sorted(((float(x["price"]), float(x.get("size", 0) or 0))
                       for x in book.get("asks", []) or []), key=lambda t: t[0])
        return {"bids": bids, "asks": asks,
                "book_ts": book.get("timestamp"), "book_hash": book.get("hash")}

    out: dict[str, dict] = {}
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
                    out[t] = {"bids": [], "asks": [], "book_ts": None, "book_hash": None}
    return out


def build_snapshot(meta: dict, timeout: float = 30.0) -> dict:
    """One timestamped capture of every leg's YES and NO book, full depth."""
    legs = meta["ladder"] + meta["buckets"] + ([meta["no_ipo"]] if meta["no_ipo"] else [])
    tokens = [t for l in legs for t in (l["token_yes"], l["token_no"])]
    books = fetch_full_books(tokens, timeout=timeout)
    return {"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "books": books}


def log_books_parquet(meta: dict, snap: dict, out_dir: Path = BOOKS_DIR) -> Path:
    """Append-only: one parquet shard per capture, one row per (leg, outcome, side, level)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    legs = meta["ladder"] + meta["buckets"] + ([meta["no_ipo"]] if meta["no_ipo"] else [])
    for leg in legs:
        for outcome, tok in (("YES", leg["token_yes"]), ("NO", leg["token_no"])):
            book = snap["books"].get(tok, {"bids": [], "asks": []})
            for side in ("bids", "asks"):
                for lvl, (price, size) in enumerate(book.get(side, [])):
                    rows.append({"poll_ts": snap["fetched_at_utc"], "kind": leg["kind"],
                                 "label": leg["label"], "condition_id": leg["condition_id"],
                                 "token_id": tok, "outcome": outcome,
                                 "side": side[:-1], "level": lvl,
                                 "price": price, "size": size,
                                 "neg_risk": leg["neg_risk"], "res_key": leg["res_key"],
                                 "book_ts": str(book.get("book_ts")),
                                 "book_hash": book.get("book_hash")})
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = snap["fetched_at_utc"].replace(":", "").replace("-", "").replace(".", "")[:15]
    path = out_dir / f"books_{ts}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    (out_dir / f"meta_{ts}.json").write_text(json.dumps(meta, indent=1, default=str))
    return path


def load_snapshot_parquet(path: Path) -> tuple[dict, dict]:
    """Rebuild (meta, snap) from a books_<ts>.parquet shard + its meta_<ts>.json."""
    import pyarrow.parquet as pq

    meta = json.loads((path.parent / path.name.replace("books_", "meta_")
                       .replace(".parquet", ".json")).read_text())
    for b in meta["buckets"]:  # JSON round-trips inf as the string "inf"
        if isinstance(b.get("hi"), str):
            b["hi"] = math.inf
    tbl = pq.read_table(path).to_pylist()
    books: dict[str, dict] = {}
    poll_ts = tbl[0]["poll_ts"] if tbl else None
    for r in tbl:
        bk = books.setdefault(r["token_id"], {"bids": [], "asks": [],
                                              "book_ts": r["book_ts"], "book_hash": r["book_hash"]})
        bk[r["side"] + "s"].append((r["level"], r["price"], r["size"]))
    for bk in books.values():
        bk["bids"] = [(p, s) for _, p, s in sorted(bk["bids"])]
        bk["asks"] = [(p, s) for _, p, s in sorted(bk["asks"])]
    return meta, {"fetched_at_utc": poll_ts, "books": books}


# --------------------------------------------------------------------------------------
# Payoff floors by state enumeration (resolution semantics, incl. the No-IPO state)
# --------------------------------------------------------------------------------------
def leg_indicator(leg: dict, outcome: str, cap_t: float | None, no_ipo_state: bool) -> int:
    """$ payoff of one share of `outcome` on `leg` in the given terminal state.
    Semantics from the live Gamma descriptions:
      ladder 'above $K' : YES iff cap > K strictly; No-IPO state => NO.
      bucket [lo, hi)   : lo-inclusive / hi-exclusive ('exactly between brackets
                          resolves to the higher bracket'); No-IPO state => NO.
      no_ipo            : YES iff the No-IPO state."""
    if leg["kind"] == "no_ipo":
        yes = 1 if no_ipo_state else 0
    elif no_ipo_state:
        yes = 0
    elif leg["kind"] == "ladder":
        yes = 1 if cap_t > leg["strike_t"] + EPS else 0
    else:
        yes = 1 if (leg["lo"] - EPS <= cap_t < leg["hi"] - EPS) else 0
    return yes if outcome == "YES" else 1 - yes


def enumerate_states(meta: dict) -> list[dict]:
    """Terminal states: every bracket edge exactly (boundary=True), midpoints between
    consecutive edges, below-min, above-max, plus the No-IPO state."""
    edges = sorted({l["strike_t"] for l in meta["ladder"]}
                   | {e for b in meta["buckets"] for e in (b["lo"], b["hi"])
                      if e and not math.isinf(e)})
    states = [{"cap_t": None, "no_ipo": True, "boundary": False, "label": "No-IPO"}]
    pts = [edges[0] / 2.0] + [(a + b) / 2.0 for a, b in zip(edges, edges[1:])] + [edges[-1] + 0.5]
    for p in pts:
        states.append({"cap_t": p, "no_ipo": False, "boundary": False, "label": f"cap={p:g}T"})
    for e in edges:
        states.append({"cap_t": e, "no_ipo": False, "boundary": True,
                       "label": f"cap={e:g}T exactly"})
    return states


def payoff_floor(combo_legs: list[tuple[dict, str]], states: list[dict]) -> dict:
    """Min combo payoff over non-boundary states (robust floor) and over all states.
    Boundary states paying below the robust floor are the measure-zero caveats."""
    def pay(st):
        return sum(leg_indicator(leg, outc, st["cap_t"], st["no_ipo"])
                   for leg, outc in combo_legs)
    robust = min(pay(s) for s in states if not s["boundary"])
    worst_all = min(pay(s) for s in states)
    caveats = [f"{s['label']} pays {pay(s)}" for s in states
               if s["boundary"] and pay(s) < robust]
    return {"floor_robust": float(robust), "floor_all": float(worst_all), "caveats": caveats}


# --------------------------------------------------------------------------------------
# Depth walk — executable level-by-level lock economics
# --------------------------------------------------------------------------------------
def _price_at(levels: list[tuple[float, float]], q: float) -> float | None:
    """Price of the marginal share at cumulative position q (book order)."""
    c = 0.0
    for p, s in levels:
        c += s
        if q < c - EPS:
            return p
    return None


def walk_lock(leg_levels: list[list[tuple[float, float]]], floor: float,
              gas_per_set: float = 0.0) -> dict:
    """Walk all legs' ask ladders simultaneously (1 share per leg per set).
    Marginal cost is non-decreasing, so the walk stops when net edge <= 0 under a
    schedule, or when any leg's book is exhausted. Returns per-segment rows and
    cumulative size/$ at the gross / net-repo / net-harsh cutoffs."""
    bps: set[float] = set()
    for lv in leg_levels:
        c = 0.0
        for _, s in lv:
            c += s
            bps.add(c)
    rows, prev = [], 0.0
    exhausted = False
    for b in sorted(bps):
        if b <= prev + EPS:
            continue
        prices = [_price_at(lv, prev) for lv in leg_levels]
        if any(p is None for p in prices):
            exhausted = True
            break
        cost = sum(prices)
        gross = floor - cost - gas_per_set
        rows.append({"sets_from": prev, "sets_to": b, "sets": b - prev,
                     "cost_per_set": cost, "gross_per_set": gross,
                     "net_repo_per_set": gross - sum(fee_per_share(p, "repo") for p in prices),
                     "net_harsh_per_set": gross - sum(fee_per_share(p, "harsh") for p in prices)})
        prev = b
    else:
        exhausted = True  # consumed every breakpoint without a None: books fully walked

    def cum(key):
        sets = profit = 0.0
        closes_at = None
        for i, r in enumerate(rows):
            if r[key] <= EPS:
                closes_at = i
                break
            sets += r["sets"]
            profit += r[key] * r["sets"]
        return {"sets": sets, "profit_usd": profit, "closes_at_segment": closes_at}

    top = rows[0] if rows else None
    return {"segments": rows, "top": top, "exhausted_before_close": exhausted,
            "gross": cum("gross_per_set"), "net_repo": cum("net_repo_per_set"),
            "net_harsh": cum("net_harsh_per_set")}


def walk_mint_sell(leg_bid_levels: list[list[tuple[float, float]]], set_cost: float,
                   gas_total: float) -> dict:
    """NegRisk mint-and-sell: mint full YES sets at $1 each (+ one-off gas), hit each
    leg's bids level-by-level. Legs with no bids contribute zero revenue (the residual
    token is kept; payoff >= 0, so the lock test stays conservative)."""
    sellable = [lv for lv in leg_bid_levels if lv]
    n_residual = len(leg_bid_levels) - len(sellable)
    bps: set[float] = set()
    for lv in sellable:
        c = 0.0
        for _, s in lv:
            c += s
            bps.add(c)
    rows, prev = [], 0.0
    for b in sorted(bps):
        if b <= prev + EPS:
            continue
        prices = [_price_at(lv, prev) for lv in sellable]
        if any(p is None for p in prices):
            break
        rev = sum(prices)
        gross = rev - set_cost
        rows.append({"sets_from": prev, "sets_to": b, "sets": b - prev,
                     "revenue_per_set": rev, "gross_per_set": gross,
                     "net_repo_per_set": gross - sum(fee_per_share(p, "repo") for p in prices),
                     "net_harsh_per_set": gross - sum(fee_per_share(p, "harsh") for p in prices)})
        prev = b

    def cum(key):
        sets = profit = 0.0
        closes_at = None
        for i, r in enumerate(rows):
            if r[key] <= EPS:
                closes_at = i
                break
            sets += r["sets"]
            profit += r[key] * r["sets"]
        profit -= gas_total if sets > 0 else 0.0
        return {"sets": sets, "profit_usd": profit, "closes_at_segment": closes_at}

    return {"segments": rows, "top": rows[0] if rows else None,
            "n_residual_legs": n_residual,
            "gross": cum("gross_per_set"), "net_repo": cum("net_repo_per_set"),
            "net_harsh": cum("net_harsh_per_set")}


# --------------------------------------------------------------------------------------
# Phase 2 — combo construction + evaluation
# --------------------------------------------------------------------------------------
def _book(snap: dict, leg: dict, outcome: str) -> dict:
    tok = leg["token_yes"] if outcome == "YES" else leg["token_no"]
    return snap["books"].get(tok, {"bids": [], "asks": []})


def evaluate_combo(name: str, klass: str, combo_legs: list[tuple[dict, str]],
                   snap: dict, states: list[dict], gas_per_set: float = 0.0) -> dict:
    """One buy-side lock: buy every (leg, outcome) at ask, payoff floor from state
    enumeration, depth-walked and fee-netted. Executable-only by construction (reads
    nothing but ask levels)."""
    res_keys = {leg["res_key"] for leg, _ in combo_legs}
    mixed = len(res_keys) > 1
    fl = payoff_floor(combo_legs, states)
    leg_levels = [_book(snap, leg, outc).get("asks", []) for leg, outc in combo_legs]
    missing = [f"{leg['label']}:{outc}" for (leg, outc), lv in zip(combo_legs, leg_levels)
               if not lv]
    walk = (walk_lock(leg_levels, fl["floor_robust"], gas_per_set)
            if not missing else {"segments": [], "top": None, "exhausted_before_close": True,
                                 "gross": {"sets": 0.0, "profit_usd": 0.0, "closes_at_segment": None},
                                 "net_repo": {"sets": 0.0, "profit_usd": 0.0, "closes_at_segment": None},
                                 "net_harsh": {"sets": 0.0, "profit_usd": 0.0, "closes_at_segment": None}})
    top = walk["top"]
    net_sets = max(walk["net_repo"]["sets"], walk["net_harsh"]["sets"])
    net_usd = max(walk["net_repo"]["profit_usd"], walk["net_harsh"]["profit_usd"])
    dust = (walk["gross"]["sets"] > 0) and (net_sets < MIN_ORDER_SHARES or net_usd < DUST_NET_USD)
    if mixed:
        verdict = "RV-WITH-RISK (mixed resolution keys — not arb)"
    elif missing:
        verdict = f"NO-ARB (no ask on {','.join(missing)})"
    elif top and top["gross_per_set"] > EPS and walk["net_harsh"]["sets"] >= MIN_ORDER_SHARES \
            and walk["net_harsh"]["profit_usd"] >= DUST_NET_USD:
        verdict = "ARB (net of harsh fees, investable size)"
    elif top and top["gross_per_set"] > EPS:
        verdict = "NO-ARB (gross>0 but dust/fee-killed)" if (dust or walk["net_harsh"]["sets"] < EPS) \
            else "ARB-MARGINAL (net-repo only)"
    else:
        verdict = "NO-ARB"
    return {"name": name, "class": klass,
            "legs": " + ".join(f"{o}({l['label']})" for l, o in combo_legs),
            "n_legs": len(combo_legs), "mixed_resolution": mixed,
            "res_keys": sorted(res_keys), "floor": fl["floor_robust"],
            "boundary_caveats": fl["caveats"], "missing_asks": missing,
            "cost_top": top["cost_per_set"] if top else None,
            "gross_top": top["gross_per_set"] if top else None,
            "net_repo_top": top["net_repo_per_set"] if top else None,
            "net_harsh_top": top["net_harsh_per_set"] if top else None,
            "gross_sets": walk["gross"]["sets"], "gross_usd": walk["gross"]["profit_usd"],
            "net_repo_sets": walk["net_repo"]["sets"], "net_repo_usd": walk["net_repo"]["profit_usd"],
            "net_harsh_sets": walk["net_harsh"]["sets"], "net_harsh_usd": walk["net_harsh"]["profit_usd"],
            "capital_per_set": top["cost_per_set"] if top else None,
            "dust": dust, "gas_per_set": gas_per_set, "verdict": verdict,
            "segments": walk["segments"]}


def build_combos(meta: dict) -> list[dict]:
    """Enumerate every combo for the three arb classes (buy-side, ask-only locks)."""
    combos = []
    ladder, buckets, no_ipo = meta["ladder"], meta["buckets"], meta["no_ipo"]
    group = buckets + ([no_ipo] if no_ipo else [])

    # -- class 1: NegRisk basket --------------------------------------------------------
    combos.append({"name": "basket_buy_all_YES", "class": "negrisk_basket",
                   "legs": [(l, "YES") for l in group]})
    combos.append({"name": "basket_buy_all_NO", "class": "negrisk_basket",
                   "legs": [(l, "NO") for l in group]})

    # -- class 1 intra-market complements (YES ask + NO ask < $1), every market ---------
    for l in ladder + group:
        combos.append({"name": f"complement_{l['label']}", "class": "complement",
                       "legs": [(l, "YES"), (l, "NO")]})

    # -- class 2: ladder monotonicity (all ordered pairs; adjacent flagged in name) -----
    for i, a in enumerate(ladder):
        for j in range(i + 1, len(ladder)):
            b = ladder[j]
            tag = "adj" if j == i + 1 else "pair"
            combos.append({"name": f"mono_{tag}_{a['label']}_vs_{b['label']}",
                           "class": "ladder_monotonicity",
                           "legs": [(a, "YES"), (b, "NO")]})

    # -- class 3: ladder-vs-bucket boxes on executable unions only ----------------------
    strikes = {l["strike_t"]: l for l in ladder}
    by_lo = {b["lo"]: b for b in buckets}

    def bucket_run(lo: float, hi: float) -> list[dict] | None:
        run, x = [], lo
        while x < hi - EPS:
            b = by_lo.get(x)
            if b is None:
                return None
            run.append(b)
            x = b["hi"]
        return run if abs(x - hi) < EPS or (math.isinf(hi) and math.isinf(x)) else None

    edges = sorted(strikes)
    unions = ([(a, b) for i, a in enumerate(edges) for b in edges[i + 1:]]
              + [(e, math.inf) for e in edges] + [(0.0, e) for e in edges])
    for lo, hi in unions:
        run = bucket_run(lo, hi)
        if run is None:
            continue
        lab = f"[{lo:g},{'inf' if math.isinf(hi) else f'{hi:g}'})"
        # direction A: long the union synthetically via ladder, short it via bucket NOs.
        # Failure-partition structure: exactly one leg pays 0 in every non-boundary state.
        legs_a = [(b, "NO") for b in run]
        if lo > 0:
            legs_a.append((strikes[lo], "YES"))
        if not math.isinf(hi):
            legs_a.append((strikes[hi], "NO"))
        combos.append({"name": f"box_A_{lab}", "class": "box", "legs": legs_a})
        # direction B: long the union via bucket YESes, short it via ladder legs.
        legs_b = [(b, "YES") for b in run]
        if lo > 0:
            legs_b.append((strikes[lo], "NO"))
        if not math.isinf(hi):
            legs_b.append((strikes[hi], "YES"))
        if no_ipo is not None and lo == 0.0:
            legs_b.append((no_ipo, "YES"))  # the [0,hi) union must also pay in the No-IPO state
        combos.append({"name": f"box_B_{lab}", "class": "box", "legs": legs_b})

    # -- class 3b: containment covers. The half-T bucket edges (1.5/2.5/3.5T) have no
    # ladder strike, so the bucket can't be boxed exactly — but it CAN be traded
    # risk-free against the tightest WIDER ladder range (sell bucket, the uncovered
    # sliver is a free long) or the tightest NARROWER one (buy bucket, the sliver is
    # given away). Floors still come from state enumeration, never hand-derived.
    sk = sorted(strikes)
    for i in range(len(buckets)):
        for j in range(i, len(buckets)):
            run = buckets[i:j + 1]
            lo, hi = run[0]["lo"], run[-1]["hi"]
            lab = run[0]["label"] if i == j else f"{run[0]['label']}..{run[-1]['label']}"
            # C1 — sell the run, cover with the tightest wider ladder range
            a = max((s for s in sk if s <= lo + EPS), default=None) if lo > 0 else None
            c = min((s for s in sk if s >= hi - EPS), default=None) if not math.isinf(hi) else None
            ok = (lo == 0.0 or a is not None) and (math.isinf(hi) or c is not None)
            exact = ((lo == 0.0 or a == lo) and (math.isinf(hi) or c == hi)
                     and not (lo == 0.0 and math.isinf(hi)))  # full-run = all-NO basket, keep it
            if ok and not exact:
                legs = [(b, "NO") for b in run]
                if a is not None:
                    legs.append((strikes[a], "YES"))
                if c is not None:
                    legs.append((strikes[c], "NO"))
                cov = f"[{a if a is not None else 0:g},{'inf' if c is None else f'{c:g}'})"
                combos.append({"name": f"cover_sell_{lab}_via_{cov}",
                               "class": "box_cover", "legs": legs})
            # C2 — buy the run, short the tightest narrower ladder range inside it
            a2 = min((s for s in sk if s >= lo - EPS), default=None)
            if a2 is None:
                continue
            if math.isinf(hi):
                if a2 != lo:  # a2 == lo would duplicate box_B_[lo,inf)
                    combos.append({"name": f"cover_buy_{lab}_via_[{a2:g},inf)",
                                   "class": "box_cover",
                                   "legs": [(b, "YES") for b in run] + [(strikes[a2], "NO")]})
            else:
                c2 = max((s for s in sk if s <= hi + EPS), default=None)
                if c2 is not None and a2 < c2 - EPS and not (a2 == lo and c2 == hi):
                    combos.append({"name": f"cover_buy_{lab}_via_[{a2:g},{c2:g})",
                                   "class": "box_cover",
                                   "legs": [(b, "YES") for b in run]
                                   + [(strikes[a2], "NO"), (strikes[c2], "YES")]})
    return combos


def analyze(meta: dict, snap: dict, gas_usd: float = GAS_USD_DEFAULT) -> dict:
    """All three arb classes on one snapshot. Executable quotes only."""
    states = enumerate_states(meta)
    results = [evaluate_combo(c["name"], c["class"], c["legs"], snap, states)
               for c in build_combos(meta)]

    group = meta["buckets"] + ([meta["no_ipo"]] if meta["no_ipo"] else [])
    mint = walk_mint_sell([_book(snap, l, "YES").get("bids", []) for l in group],
                          set_cost=1.0, gas_total=gas_usd)
    mixed_mint = len({l["res_key"] for l in group}) > 1
    not_negrisk = not all(l.get("neg_risk") for l in group)
    top = mint["top"]
    if mixed_mint or not_negrisk:
        mint_verdict = "RV-WITH-RISK (group not uniformly NegRisk/co-resolving)"
    elif top and top["gross_per_set"] > EPS and mint["net_harsh"]["sets"] >= MIN_ORDER_SHARES \
            and mint["net_harsh"]["profit_usd"] >= DUST_NET_USD:
        mint_verdict = "ARB (net of harsh fees + gas, investable size)"
    elif top and top["gross_per_set"] > EPS:
        mint_verdict = "NO-ARB (gross>0 but dust/fee-killed)"
    else:
        mint_verdict = "NO-ARB"
    mint_row = {"name": "basket_mint_sell_all_YES", "class": "negrisk_basket",
                "legs": "mint $1 set -> sell " + " + ".join(
                    f"YES({l['label']})" for l in group
                    if _book(snap, l, "YES").get("bids")),
                "n_legs": len(group), "mixed_resolution": mixed_mint,
                "res_keys": sorted({l["res_key"] for l in group}),
                "floor": None, "boundary_caveats": [],
                "missing_asks": [f"{l['label']}:no-bid(residual kept)" for l in group
                                 if not _book(snap, l, "YES").get("bids")],
                "cost_top": 1.0 + gas_usd,
                "gross_top": top["gross_per_set"] if top else None,
                "net_repo_top": top["net_repo_per_set"] if top else None,
                "net_harsh_top": top["net_harsh_per_set"] if top else None,
                "gross_sets": mint["gross"]["sets"], "gross_usd": mint["gross"]["profit_usd"],
                "net_repo_sets": mint["net_repo"]["sets"],
                "net_repo_usd": mint["net_repo"]["profit_usd"],
                "net_harsh_sets": mint["net_harsh"]["sets"],
                "net_harsh_usd": mint["net_harsh"]["profit_usd"],
                "capital_per_set": 1.0, "dust": bool(top) and mint["net_harsh"]["sets"] < MIN_ORDER_SHARES,
                "gas_per_set": gas_usd, "verdict": mint_verdict, "segments": mint["segments"]}
    results.append(mint_row)
    return {"poll_ts": snap["fetched_at_utc"], "gas_usd": gas_usd, "results": results}


# --------------------------------------------------------------------------------------
# Single-fee book walk — "how much can I pull out before the mispricing is gone?"
# --------------------------------------------------------------------------------------
def walk_lock_net(leg_levels: list[list[tuple[float, float]]], floor: float,
                  fee_bps: float) -> dict:
    """Walk every leg's ask ladder together (1 share/leg per set), buying best-ask, then
    best-ask+1, ... accumulating NET profit = floor - Σ price - Σ taker_fee, and STOP at
    the first set where the marginal net <= 0 (the mispricing is gone / fees eat it) or a
    book is exhausted. Returns the one number that matters: cumulative net $ and the size
    it took, plus the top-of-book per-leg prices for display."""
    bps: set[float] = set()
    for lv in leg_levels:
        c = 0.0
        for _, s in lv:
            c += s
            bps.add(c)
    sets = profit = 0.0
    closed = "fees/spread"
    prev = 0.0
    for b in sorted(bps):
        if b <= prev + EPS:
            continue
        prices = [_price_at(lv, prev) for lv in leg_levels]
        if any(p is None for p in prices):
            closed = "book exhausted"
            break
        net = floor - sum(prices) - sum(taker_fee(p, fee_bps) for p in prices)
        if net <= EPS:
            break
        sets += b - prev
        profit += net * (b - prev)
        prev = b
    return {"net_sets": sets, "net_usd": profit, "closed_by": closed}


def best_executable_arb(meta: dict, snap: dict, fee_bps: float = FEE_BPS_DEFAULT) -> dict:
    """THE dashboard output: the single best ladder<->bucket pricing-mismatch lock that
    is executable RIGHT NOW, taker-only, sized by walking the book until the marginal set
    goes net-negative under ONE stated fee. Returns a render-ready dict (no two-schedule
    business). Considers only the ladder<->bucket families (exact unions + containment
    covers); never a mid, never a fitted survivor curve.

    fee_bps: Polymarket CLOB taker fee, fee/share = (fee_bps/10000)*min(p,1-p). 0 = what
    fills are observed to pay; 1000 = the rate these markets declare. ONE knob."""
    states = enumerate_states(meta)
    candidates = []
    for c in build_combos(meta):
        if c["class"] not in ("box", "box_cover"):
            continue
        combo_legs = c["legs"]
        if len({leg["res_key"] for leg, _ in combo_legs}) > 1:
            continue  # never combine mixed-resolution legs (no basis risk allowed)
        leg_levels = [_book(snap, leg, outc).get("asks", []) for leg, outc in combo_legs]
        if any(not lv for lv in leg_levels):
            continue
        fl = payoff_floor(combo_legs, states)
        walk = walk_lock_net(leg_levels, fl["floor_robust"], fee_bps)
        if walk["net_usd"] <= EPS:
            continue
        display_legs = []
        for (leg, outc), lv in zip(combo_legs, leg_levels):
            display_legs.append({
                "action": "BUY" if outc == "YES" else "SELL",  # NO ask = sell the YES
                "market": leg["label"], "kind": leg["kind"],
                "price": lv[0][0], "top_size": lv[0][1]})
        cost_top = sum(lv[0][0] for lv in leg_levels)
        candidates.append({
            "name": c["name"], "class": c["class"], "legs": display_legs,
            "pay_per_set": cost_top, "payout_floor": fl["floor_robust"],
            "free_sliver": fl["caveats"],  # states that pay ABOVE the floor are free upside
            "net_sets": walk["net_sets"], "notional_usd": cost_top * walk["net_sets"],
            "net_usd": walk["net_usd"], "closed_by": walk["closed_by"]})
    candidates.sort(key=lambda r: -r["net_usd"])
    best = candidates[0] if candidates else None
    investable = bool(best) and best["net_sets"] >= MIN_ORDER_SHARES and best["net_usd"] >= DUST_NET_USD
    return {
        "poll_ts": snap["fetched_at_utc"], "fee_bps": fee_bps,
        "fee_formula": f"({fee_bps:g}bps) x min(price, 1-price) per share, taker",
        "exists": best is not None, "investable": investable,
        "verdict": ("ARB" if investable else
                    "lock exists — uninvestable (dust)" if best else "no lock"),
        "best": best, "n_candidates": len(candidates)}


# --------------------------------------------------------------------------------------
# Per-leg table + YES/NO mirror diagnostic
# --------------------------------------------------------------------------------------
def leg_table(meta: dict, snap: dict) -> list[dict]:
    rows = []
    for leg in meta["ladder"] + meta["buckets"] + ([meta["no_ipo"]] if meta["no_ipo"] else []):
        y, n = _book(snap, leg, "YES"), _book(snap, leg, "NO")
        yb = y["bids"][0] if y["bids"] else (None, None)
        ya = y["asks"][0] if y["asks"] else (None, None)
        nb = n["bids"][0] if n["bids"] else (None, None)
        na = n["asks"][0] if n["asks"] else (None, None)
        mirror = (abs(na[0] - (1.0 - yb[0])) if na[0] is not None and yb[0] is not None else None)
        rows.append({"kind": leg["kind"], "label": leg["label"],
                     "yes_bid": yb[0], "yes_bid_sz": yb[1], "yes_ask": ya[0], "yes_ask_sz": ya[1],
                     "no_bid": nb[0], "no_bid_sz": nb[1], "no_ask": na[0], "no_ask_sz": na[1],
                     "yes_ask_depth_usd": sum(p * s for p, s in y["asks"]),
                     "yes_bid_depth_usd": sum(p * s for p, s in y["bids"]),
                     "mirror_no_ask_vs_1m_yes_bid": mirror,
                     "neg_risk": leg["neg_risk"], "res_key": leg["res_key"]})
    return rows


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
SUMMARY_COLS = ["name", "class", "verdict", "n_legs", "floor", "cost_top", "gross_top",
                "net_repo_top", "net_harsh_top", "gross_sets", "gross_usd",
                "net_repo_sets", "net_repo_usd", "net_harsh_sets", "net_harsh_usd",
                "capital_per_set", "gas_per_set", "dust", "mixed_resolution",
                "missing_asks", "boundary_caveats", "legs"]


def write_csvs(rep: dict, legs: list[dict], out_dir: Path = RESULTS_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = rep["poll_ts"].replace(":", "").replace("-", "").replace(".", "")[:15]
    paths = []
    p = out_dir / f"summary_{ts}.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for r in rep["results"]:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], (list, dict)) else r[k])
                        for k in SUMMARY_COLS})
    paths.append(p)
    p = out_dir / f"depth_{ts}.csv"
    with p.open("w", newline="") as f:
        cols = ["name", "sets_from", "sets_to", "sets", "cost_per_set", "revenue_per_set",
                "gross_per_set", "net_repo_per_set", "net_harsh_per_set"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rep["results"]:
            if r["gross_top"] is not None and r["gross_top"] > EPS:
                for seg in r["segments"]:
                    w.writerow({"name": r["name"], **{k: seg.get(k) for k in cols[1:]}})
    paths.append(p)
    p = out_dir / f"legs_{ts}.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(legs[0].keys()))
        w.writeheader()
        w.writerows(legs)
    paths.append(p)
    return paths


def render_text(rep: dict, legs: list[dict]) -> str:
    lines = [f"SPCX PM executable-arb check — poll {rep['poll_ts']}  (gas ${rep['gas_usd']:.2f}/set on mint path)",
             "=" * 100]
    lines.append(f"{'leg':<12}{'yes bid':>9}{'sz':>9}{'yes ask':>9}{'sz':>9}{'no ask':>9}"
                 f"{'mirror|Δ|':>10}")
    for r in legs:
        f = lambda v, d=3: ("" if v is None else f"{v:.{d}f}")
        lines.append(f"{r['label']:<12}{f(r['yes_bid']):>9}{f(r['yes_bid_sz'],0):>9}"
                     f"{f(r['yes_ask']):>9}{f(r['yes_ask_sz'],0):>9}{f(r['no_ask']):>9}"
                     f"{f(r['mirror_no_ask_vs_1m_yes_bid'],4):>10}")
    lines.append("-" * 100)
    by_class: dict[str, list[dict]] = {}
    for r in rep["results"]:
        by_class.setdefault(r["class"], []).append(r)
    for klass, rows in by_class.items():
        pos = [r for r in rows if r["gross_top"] is not None and r["gross_top"] > EPS]
        lines.append(f"[{klass}] {len(rows)} combos, {len(pos)} with positive gross at top of book")
        show = pos if pos else sorted([r for r in rows if r["cost_top"] is not None],
                                      key=lambda r: -(r["gross_top"] or -9))[:3]
        for r in show:
            lines.append(f"  {r['name']:<34} floor={r['floor']!s:>4} cost@top="
                         f"{r['cost_top']:.3f} gross/set={r['gross_top']:+.4f} "
                         f"netH/set={r['net_harsh_top']:+.4f} "
                         f"sets(netH)={r['net_harsh_sets']:.0f} $netH={r['net_harsh_usd']:+.2f} "
                         f"{'DUST ' if r['dust'] else ''}{r['verdict']}")
            for c in r["boundary_caveats"]:
                lines.append(f"        boundary caveat: {c}")
    return "\n".join(lines)


def render_best(arb: dict) -> str:
    """The single dashboard-facing output: one best executable ladder<->bucket lock,
    sized by walking the book to the fee/spread cutoff. One fee, one number."""
    out = [f"LADDER<->BUCKET MISMATCH ARB (taker-only)  fee = {arb['fee_formula']}"]
    if not arb["exists"]:
        out.append("  no executable lock right now (ladder and buckets consistent within spread)")
        return "\n".join(out)
    b = arb["best"]
    out.append(f"  TRADE: " + "  ".join(f"{l['action']} {l['market']} @ {l['price']:.3f}"
                                        for l in b["legs"]))
    out.append(f"         pay ${b['pay_per_set']:.3f}/set  -> locked payout ${b['payout_floor']:.0f}"
               + (f"  (+free upside: {b['free_sliver'][0]})" if b["free_sliver"] else ""))
    out.append(f"  EXTRACTABLE: ${b['net_usd']:.2f} net over {b['net_sets']:.0f} sets "
               f"(${b['notional_usd']:.0f} notional), edge closes by {b['closed_by']}")
    out.append(f"  VERDICT: {arb['verdict']}")
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-parquet", default=None,
                    help="offline: analyze a cached books_<ts>.parquet ('latest' allowed)")
    ap.add_argument("--watch", type=int, default=None, help="capture every N seconds")
    ap.add_argument("--polls", type=int, default=1, help="number of captures in watch mode")
    ap.add_argument("--gas-usd", type=float, default=GAS_USD_DEFAULT,
                    help="total gas for the NegRisk mint+convert path")
    ap.add_argument("--fee-bps", type=float, default=FEE_BPS_DEFAULT,
                    help="single CLOB taker fee for the extractable-$ walk "
                         "(0 = observed fills; 1000 = declared face value)")
    ap.add_argument("--no-parquet", action="store_true", help="skip book caching (debug)")
    args = ap.parse_args()

    if args.from_parquet:
        path = (sorted(BOOKS_DIR.glob("books_*.parquet"))[-1]
                if args.from_parquet == "latest" else Path(args.from_parquet))
        meta, snap = load_snapshot_parquet(path)
        rep = analyze(meta, snap, gas_usd=args.gas_usd)
        legs = leg_table(meta, snap)
        print(render_text(rep, legs))
        print(render_best(best_executable_arb(meta, snap, fee_bps=args.fee_bps)))
        for p in write_csvs(rep, legs):
            print("wrote", p)
        return

    meta = fetch_arb_metadata()
    n_polls = args.polls if args.watch else 1
    for i in range(n_polls):
        snap = build_snapshot(meta)
        if not args.no_parquet:
            print("cached", log_books_parquet(meta, snap))
        rep = analyze(meta, snap, gas_usd=args.gas_usd)
        legs = leg_table(meta, snap)
        print(render_text(rep, legs))
        print(render_best(best_executable_arb(meta, snap, fee_bps=args.fee_bps)))
        for p in write_csvs(rep, legs):
            print("wrote", p)
        if args.watch and i < n_polls - 1:
            time.sleep(args.watch)


if __name__ == "__main__":
    main()
