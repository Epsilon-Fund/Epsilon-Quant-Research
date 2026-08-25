"""Historical PM calibration — the v3.1 BIG BACKFILL of resolved politics/macro markets.

Why: the Stage-B fit (alpha) and the band knob (band_mult) ride on a thin sample —
53 pairs / 9 markets / one geopolitically extreme month (the v0 archive). This
script pulls MANY more RESOLVED Polymarket politics/macro binaries plus their
lookahead-free historical news, backfills Stage-A features via token-cheap
subagents, and refits alpha + band_mult (band coverage) on the enlarged resolved
sample. Interim scoring of unresolved markets (FV vs next-day mid) is unchanged
and lives in the dashboard.

Lookahead discipline (non-negotiable):
  - Packets are reconstructed through feeds.build_packet(now=t): Guardian filters
    by webPublicationDate <= t and Wikipedia Current Events pages are full past
    days — lookahead-free by construction. GDELT DOC is NOT used here; the
    BigQuery GKG series is day-partitioned (no end-bound leak). The mandatory
    client-side seendate filter applies wherever GDELT DOC ever re-enters.
  - Post-LLM-cutoff only: markets must RESOLVE on/after MIN_CLOSE (2026-02-15,
    past the extraction/prior models' knowledge cutoffs) so neither the prior
    subagents nor Stage-A extraction can know outcomes from training. Each prior
    subagent additionally answers an outcome-knowledge CANARY; markets where the
    model claims knowledge are excluded from the fit.
  - The market mid is context only — never a fitting target. Outcomes (y) come
    from Gamma outcomePrices on resolved markets.

Stages (out-of-band friendly, resume-safe — existing packet files are reused):
  --discover [--limit N]   resolved-universe sweep -> hist/universe.json (+ printed
                           table). Retrieval keys are AUTO-DRAFTED from question
                           text; spot-check before --fetch.
  --fetch [--markets K]    reconstruct SNAPSHOT packets + mid history per market;
                           emit hist/extract_pending.json + prior/canary prompts.
  --ingest --features-file F [--priors-file P]   validate + cache.
  --pull-gdelt             one consolidated GKG scan for backfill name-sets.
  --fit                    combined pairs (v0 archive + backfill), alpha grid-fit,
                           band_mult coverage rescale (pre-registered: first rescale
                           at >=20 resolved), params + CSVs + plot.

Run from polymarket/research/:
  PYTHONPATH=. uv run python scripts/newsagent_hist_backfill.py --discover
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from newsagent import engine, feeds, features, fvmodel, gdelt_bq, sourceweights
from newsagent.config import CSV_OUT, DATA, ROOT

HIST = ROOT / "data" / "newsagent" / "hist"
UNIVERSE = HIST / "universe.json"
HIST_PRIORS = HIST / "hist_priors.json"
PLOTS = ROOT / "data" / "analysis" / "plots" / "news_agent"

MIN_CLOSE = "2026-02-15"      # resolution must be past the LLM knowledge cutoffs
MIN_RUNWAY_DAYS = 14          # market must have lived >= this before close
MIN_VOLUME = 200_000          # resolved-market volume floor (liquidity ~0 after close)
SNAPSHOT_STEP_DAYS = 4        # snapshot cadence over the pre-close window
SNAPSHOT_WINDOW_DAYS = 30     # how far before close the snapshots start
MAX_SNAPSHOTS = 8

SPORTS_NOISE = ["fifa", "world cup", "win on 20", "vs.", "premier league", "nba",
                "nfl", "ufc", "grand prix", "wimbledon", "olympic", "college",
                "champions league", "kiss", "tweet", "say ", "mention"]

# Family cap: correlated same-event outcomes inflate n without adding information
# (the small-K cluster lesson). Coarse DECLARED buckets; max FAMILY_CAP markets
# per bucket, highest-volume first. Pairs stay clustered by family — the findings
# note must read pooled Brier with that in mind.
FAMILY_CAP = 4
FAMILY_PATTERNS = [
    ("iran", r"iran|khamenei|hormuz|uranium"),
    ("fed", r"\bfed\b|federal reserve|interest rate"),
    ("hungary", r"hungar"),
    ("ukraine_russia", r"ukraine|russia|putin|zelensk|crimea"),
    ("israel", r"israel|netanyahu|gaza"),
    ("trump_admin", r"trump|white house|maga"),
    ("china", r"china|taiwan|xi jinping"),
]


def coarse_family(question: str) -> str:
    q = (question or "").lower()
    for name, pat in FAMILY_PATTERNS:
        if re.search(pat, q):
            return name
    return "other:" + (re.findall(r"[a-z]{4,}", q) or ["misc"])[0]
POLITICS_KEYWORDS = ["election", "president", "prime minister", "senate", "house",
                     "congress", "parliament", "minister", "resign", "impeach",
                     "ceasefire", "peace", "war", "treaty", "sanction", "nato",
                     "supreme court", "nominee", "cabinet", "putin", "trump",
                     "iran", "israel", "ukraine", "tariff", "strike", "regime"]
MACRO_KEYWORDS = ["fed ", "federal reserve", "interest rate", "recession",
                  "inflation", "gdp", "shutdown", "debt ceiling"]
# crypto/equity price binaries are OD's closed territory — never in this universe
PRICE_NOISE = ["bitcoin", "ethereum", "solana", "$", "price of", "all-time high",
               "s&p", "nasdaq", "stock"]
SLOW_HINTS = ["election", "senate", "house", "congress", "governor", "fed ",
              "federal reserve", "interest rate", "parliament", "nobel"]

CANARY_SUFFIX = """

FINAL REQUIRED FIELD — OUTCOME-KNOWLEDGE CANARY: this question is from the past.
Independent of the packet, do you have training/world knowledge of how this
specific question ACTUALLY resolved? Add to your JSON object:
"outcome_knowledge": "none" | "suspected" | "known"
(none = no idea beyond the packet; suspected = you can guess from general
knowledge of later events; known = you know the specific resolution).
Answer honestly — a "known" here removes this market from the calibration sample."""


def http_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def draft_keys(question: str) -> dict:
    """Auto-draft retrieval keys from question text (same heuristic as the v3
    universe script; backfill breadth precludes hand-tuning every market —
    Stage-A relevance scoring absorbs noisy packets, and empty packets degrade
    to alpha-invariant prior-only pairs)."""
    words = re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?", question or "")
    stop = {"Will", "The", "Yes", "No", "By", "Before", "After", "In", "On", "US", "Any"}
    ents = [w for w in words if w.split()[0] not in stop][:3]
    key = ents[0].lower() if ents else (question or "").split()[-1].lower()
    return {"guardian_q": " AND ".join(f'"{e}"' if " " in e else e for e in ents[:2]) or key,
            "wp_keys": [e.lower() for e in ents[:3]] or [key],
            "gdelt_keys": [ents[0].lower()] if ents else [key]}


# ------------------------------------------------------------------ discover ---

def cmd_discover(limit: int) -> None:
    HIST.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    seen, out = set(), []
    # politics tag + general volume sweep, closed + resolved in the safe window
    queries = [{"closed": "true", "tag_id": "2", "limit": "100", "offset": str(off),
                "order": "volumeNum", "ascending": "false",
                "end_date_min": MIN_CLOSE, "end_date_max": today}
               for off in (0, 100, 200, 300)]
    queries.append({"closed": "true", "limit": "300", "order": "volumeNum",
                    "ascending": "false", "end_date_min": MIN_CLOSE,
                    "end_date_max": today})
    archive_slugs = set()
    v0_universe = CSV_OUT / "newsagent_v0_universe.csv"
    if v0_universe.exists():
        with open(v0_universe) as f:
            archive_slugs = {r["slug"] for r in csv.DictReader(f)}
    for params in queries:
        url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode(params)
        for m in http_json(url):
            slug = m.get("slug", "")
            if not slug or slug in seen or slug in archive_slugs:
                continue
            seen.add(slug)
            if m.get("umaResolutionStatus") != "resolved":
                continue
            try:
                outcomes = json.loads(m.get("outcomes") or "[]")
                prices = [float(p) for p in json.loads(m.get("outcomePrices") or "[]")]
            except Exception:
                continue
            if sorted(o.lower() for o in outcomes) != ["no", "yes"] or len(prices) != 2:
                continue
            if not ({round(prices[0]), round(prices[1])} == {0, 1}):
                continue   # ambiguous/50-50 resolutions excluded
            y = round(prices[outcomes.index(next(o for o in outcomes if o.lower() == "yes"))])
            vol = float(m.get("volumeNum") or 0)
            if vol < MIN_VOLUME:
                continue
            q = (m.get("question") or "").lower()
            if any(k in q for k in SPORTS_NOISE) or any(k in q for k in PRICE_NOISE):
                continue
            is_topic = (params.get("tag_id") == "2"
                        or any(k in q for k in POLITICS_KEYWORDS)
                        or any(k in q for k in MACRO_KEYWORDS))
            if not is_topic:
                continue
            end = (m.get("endDate") or "")[:10]
            close = (m.get("closedTime") or "")[:10] or end
            start = (m.get("startDate") or m.get("createdAt") or "")[:10]
            if not end or not start:
                continue
            if close < MIN_CLOSE:
                continue   # resolved before the safety cutoff (early resolution)
            runway = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
            if runway < MIN_RUNWAY_DAYS:
                continue
            mtype = "slow" if any(k in q for k in SLOW_HINTS) else "shock"
            out.append({"slug": slug, "question": m.get("question"),
                        "description": (m.get("description") or "")[:600],
                        "end_date": end, "close_date": close, "start_date": start,
                        "volume": round(vol), "y": y, "mtype": mtype,
                        "family": coarse_family(m.get("question")),
                        **draft_keys(m.get("question") or "")})
    # family cap (declared): highest-volume FAMILY_CAP markets per coarse family
    capped, fam_counts = [], {}
    for m in sorted(out, key=lambda x: -x["volume"]):
        fam_counts[m["family"]] = fam_counts.get(m["family"], 0) + 1
        if fam_counts[m["family"]] <= FAMILY_CAP:
            capped.append(m)
    out = capped[:limit]
    UNIVERSE.write_text(json.dumps(out, indent=1))
    n_yes = sum(1 for m in out if m["y"] == 1)
    fams = {}
    for m in out:
        fams[m["family"]] = fams.get(m["family"], 0) + 1
    print(f"{len(out)} resolved markets -> {UNIVERSE}  "
          f"(YES rate {n_yes}/{len(out)}; volume floor ${MIN_VOLUME:,}; "
          f"family cap {FAMILY_CAP}; resolutions {MIN_CLOSE}..{today})")
    print("families:", ", ".join(f"{k}×{v}" for k, v in sorted(fams.items())))
    for m in out:
        print(f"  y={m['y']} {m['mtype']:>5} {m['family']:>14} ${m['volume']:>12,}  "
              f"end={m['end_date']}  {m['question'][:70]}")


# ------------------------------------------------------- add settled forward ---

def cmd_add_settled() -> None:
    """APPEND the forward slate's own SETTLED markets to the backfill universe.

    v3.3 § 9 item 6: "fold the four settled markets into the fit sample on the next
    refit (via --discover/--fetch/--ingest), which is the honest way to make
    'refit on every settlement' literally true." They cannot arrive through
    cmd_discover: the declared FAMILY_CAP is already saturated on both of their
    families (iran ×4, fed ×4), so the sweep drops them.

    DECLARED DEVIATION, recorded here rather than in a commit message: the family
    cap is a DISCOVERY rule for sweeping the resolved universe — it exists so a
    correlated cluster cannot inflate n while adding no information. These markets
    enter by a different route: they are our OWN forward slate's settlements, and
    "refit on every settlement" is a standing commitment, not a sampling choice.
    The cap therefore does not govern them. The resulting family concentration IS
    reported (the findings note prints per-family counts), and the fit is reported
    both with and without them so the effect of the breach is visible rather than
    buried.

    Rows carry src="settled_forward" so downstream code can always separate them.
    Never overwrites universe.json wholesale — appends only, idempotently.
    """
    from newsagent.config import RETIRED_MARKETS

    universe = json.loads(UNIVERSE.read_text()) if UNIVERSE.exists() else []
    have = {m["slug"] for m in universe}
    added = []
    for slug, meta in RETIRED_MARKETS.items():
        if slug in have:
            print(f"  already present: {slug[:60]}")
            continue
        m = _gamma_resolved(slug)
        if m is None:
            print(f"  MISS gamma has nothing for {slug[:60]} (tried closed=true + search)")
            continue
        outcomes = json.loads(m.get("outcomes") or "[]")
        prices = [float(p) for p in json.loads(m.get("outcomePrices") or "[]")]
        y = round(prices[outcomes.index(next(o for o in outcomes if o.lower() == "yes"))])
        expected = 1 if meta["outcome"] == "YES" else 0
        if y != expected:
            raise SystemExit(f"outcome mismatch for {slug}: gamma says {y}, "
                             f"config.RETIRED_MARKETS says {expected} — refusing to write")
        q = (m.get("question") or "")
        end = (m.get("endDate") or "")[:10]
        start = (m.get("startDate") or m.get("createdAt") or "")[:10]
        row = {"slug": slug, "question": q,
               "description": (m.get("description") or "")[:600],
               "end_date": end,
               "close_date": (m.get("closedTime") or "")[:10] or end,
               "start_date": start,
               "volume": round(float(m.get("volumeNum") or 0)),
               "y": y, "mtype": "slow" if any(k in q.lower() for k in SLOW_HINTS) else "shock",
               "family": coarse_family(q),
               "src": "settled_forward", "sf_id": meta["sf_id"],
               **draft_keys(q)}
        universe.append(row)
        added.append(row)
    UNIVERSE.write_text(json.dumps(universe, indent=1))
    fams = {}
    for m in universe:
        fams[m["family"]] = fams.get(m["family"], 0) + 1
    print(f"\n+{len(added)} settled-forward markets -> {len(universe)} total in {UNIVERSE}")
    print("families now:", ", ".join(f"{k}×{v}" for k, v in sorted(fams.items())))
    for r in added:
        print(f"  y={r['y']} {r['mtype']:>5} {r['family']:>14} {r['sf_id']}  "
              f"end={r['end_date']} start={r['start_date']}  {r['question'][:62]}")


def _gamma_resolved(slug: str) -> dict | None:
    """Resolved-market lookup that survives Polymarket's silent re-slugs.

    Closed markets need closed=true on the slug lookup (the v3.1 gotcha); a market
    Polymarket has RENAMED returns zero rows even with the flag and has to be
    recovered through /public-search (the v3.3 gotcha). Both paths are tried here
    so a rename cannot silently drop a settlement out of the fit sample."""
    base = "https://gamma-api.polymarket.com/markets?"
    for extra in ({"slug": slug}, {"slug": slug, "closed": "true"}):
        rows = http_json(base + urllib.parse.urlencode(extra))
        if rows:
            return rows[0]
    try:
        res = http_json("https://gamma-api.polymarket.com/public-search?"
                        + urllib.parse.urlencode({"q": slug[:70], "limit_per_type": "20"}))
    except Exception:
        return None
    cands = (res or {}).get("events") or []
    for ev in cands:
        for mk in ev.get("markets") or []:
            s = mk.get("slug") or ""
            if s.startswith(slug[:60]) and mk.get("umaResolutionStatus") == "resolved":
                print(f"    re-slug recovered: {slug[:48]} -> {s[:60]}")
                return mk
    return None


# --------------------------------------------------------------------- fetch ---

def snapshot_dates(mkt: dict) -> list[str]:
    """Snapshot schedule: every SNAPSHOT_STEP_DAYS over the SNAPSHOT_WINDOW_DAYS
    before the market's scheduled end, clipped to its life and the safety floor."""
    end = datetime.fromisoformat(mkt["end_date"]).replace(tzinfo=timezone.utc)
    start = datetime.fromisoformat(mkt["start_date"]).replace(tzinfo=timezone.utc)
    lo = max(start + timedelta(days=1),
             end - timedelta(days=SNAPSHOT_WINDOW_DAYS),
             datetime.fromisoformat("2026-02-01").replace(tzinfo=timezone.utc))
    # onboard_date: markets whose prior is the LEDGER PRIOR OF RECORD (the settled
    # forward markets folded in on 2026-08-25) may not be scored before the day that
    # prior actually existed — anchoring a 2026-06-27 snapshot on a 2026-07-05 prior
    # would be a lookahead. Clip the schedule instead of silently allowing it.
    if mkt.get("onboard_date"):
        lo = max(lo, datetime.fromisoformat(mkt["onboard_date"]).replace(tzinfo=timezone.utc))
    dates, t = [], end - timedelta(days=1)
    while t >= lo and len(dates) < MAX_SNAPSHOTS:
        dates.append(t.strftime("%Y-%m-%d"))
        t -= timedelta(days=SNAPSHOT_STEP_DAYS)
    return sorted(dates)


def mid_history_range(slug: str, start: datetime, end: datetime) -> dict[str, float]:
    """Daily mids over an explicit historical window (CLOB /prices-history).

    Months-old history is NOT served by startTs/endTs windows (recent-only, plus
    a silent ~15d span cap) — the whole-lifetime series comes back only via
    interval=max at coarse fidelity. Context columns only, never a fit target."""
    url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode({"slug": slug})
    rows = http_json(url)
    if not rows:  # closed markets need the flag on slug lookups
        rows = http_json(url + "&closed=true")
    if not rows:
        return {}
    token = json.loads(rows[0].get("clobTokenIds") or "[]")
    if not token:
        return {}
    # interval=max needs an explicit fidelity to return anything (probed live)
    q = urllib.parse.urlencode({"market": token[0], "interval": "max", "fidelity": "720"})
    try:
        hist = http_json("https://clob.polymarket.com/prices-history?" + q).get("history", [])
    except Exception:
        hist = []
    lo, hi = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    pts: dict[str, dict] = {}
    for h in hist:
        ts = datetime.fromtimestamp(h["t"], tz=timezone.utc)
        d = ts.strftime("%Y-%m-%d")
        if not (lo <= d <= hi):
            continue
        dist = abs(ts.hour * 60 + ts.minute - 720)
        if d not in pts or dist < pts[d]["dist"]:
            pts[d] = {"mid": round(float(h["p"]), 4), "dist": dist}
    time.sleep(0.4)
    return {d: v["mid"] for d, v in pts.items()}


def cmd_fetch(max_markets: int | None) -> None:
    universe = json.loads(UNIVERSE.read_text())
    if max_markets:
        universe = universe[:max_markets]
    (HIST / "prior_prompts").mkdir(parents=True, exist_ok=True)
    pending_all, n_pkts = [], 0
    for m in universe:
        sdir = HIST / m["slug"][:80]
        sdir.mkdir(exist_ok=True)
        dates = snapshot_dates(m)
        if not dates:
            print(f"  SKIP no snapshot runway: {m['slug'][:60]}")
            continue
        first_pkt = None
        for d in dates:
            pf = sdir / f"{d}.packet.json"
            if pf.exists():
                pkt = json.loads(pf.read_text())
            else:
                t = datetime.fromisoformat(d).replace(hour=12, tzinfo=timezone.utc)
                cfg = {"guardian_q": m["guardian_q"], "wp_keys": m["wp_keys"]}
                try:
                    pkt = feeds.build_packet(m["slug"], cfg, now=t)
                except Exception as e:
                    print(f"  WARN packet failed {m['slug'][:45]} {d}: {e}")
                    continue
                pf.write_text(json.dumps(pkt, indent=1))
            if first_pkt is None:
                first_pkt = (d, pkt)
            pending_all.extend(features.pending_extractions(
                m["slug"], m["question"], m.get("description") or m["question"],
                pkt["articles"]))
            n_pkts += 1
        mh = sdir / "mid_history.json"
        if not mh.exists():
            end = datetime.fromisoformat(m["end_date"]).replace(tzinfo=timezone.utc)
            mids = mid_history_range(m["slug"],
                                     end - timedelta(days=SNAPSHOT_WINDOW_DAYS + 5), end)
            mh.write_text(json.dumps(mids, indent=1))
        # prior prompt from the FIRST snapshot packet only (lookahead-free) + canary
        if first_pkt is not None and m["slug"] not in json.loads(
                HIST_PRIORS.read_text() if HIST_PRIORS.exists() else "{}"):
            d, pkt = first_pkt
            market = {"question": m["question"],
                      "description": m.get("description") or m["question"],
                      "end_date": m["end_date"] + "T00:00:00Z"}
            asof = datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
            prompt = engine.build_prompt(market, pkt, asof=asof) + CANARY_SUFFIX
            (HIST / "prior_prompts" / f"{m['slug'][:80]}.txt").write_text(prompt)
        print(f"  {m['slug'][:58]}: {len(dates)} snapshots")
    seen, dedup = set(), []
    for p in pending_all:
        if p["cache_key"] in seen:
            continue
        seen.add(p["cache_key"])
        dedup.append(p)
    (HIST / "extract_pending.json").write_text(json.dumps(dedup, indent=1))
    print(f"\n{n_pkts} packets; {len(dedup)} unique (market, article) extractions "
          f"pending -> {HIST / 'extract_pending.json'}")
    print(f"prior+canary prompts -> {HIST / 'prior_prompts'}")


# -------------------------------------------------------------------- ingest ---

def cmd_ingest(features_file: str | None, priors_file: str | None,
               oob_source: str = "oob:hist_backfill") -> None:
    """Ingest out-of-band Stage-A features / priors.

    `oob_source` is stamped into every cache record's `_meta.source`. It is NOT
    cosmetic: alpha is fitted on HAIKU-era extraction, so a round produced by a
    different extractor is a provider change in everything but name and has to be
    attributable after the fact (the v3.3 § 5a caveat). Pass the real provenance."""
    if features_file:
        done = json.loads(Path(features_file).read_text())
        n = features.ingest_features(done, source=oob_source)
        print(f"ingested {n} feature records")
    if priors_file:
        raw = json.loads(Path(priors_file).read_text())
        priors = json.loads(HIST_PRIORS.read_text()) if HIST_PRIORS.exists() else {}
        for slug, rec in raw.items():
            agg = engine.aggregate(rec["estimates_pct"])
            priors[slug] = {"p0_pct": agg["p_pct"], "estimates_pct": rec["estimates_pct"],
                            "drivers": rec.get("drivers", []),
                            "outcome_knowledge": rec.get("outcome_knowledge", "none")}
        HIST_PRIORS.write_text(json.dumps(priors, indent=1))
        known = [s for s, r in priors.items() if r.get("outcome_knowledge") == "known"]
        print(f"stored {len(raw)} priors -> {HIST_PRIORS} "
              f"({len(known)} canary-KNOWN, excluded from fit: {known or '—'})")


# ---------------------------------------------------------------- gdelt scan ---

def cmd_pull_gdelt(win_from: str | None = None, win_to: str | None = None) -> None:
    """Chunked <=70-day windows: each scan stays under the 25 GB per-query guard;
    the multi-month total (~37 GB) is ~4% of the 1 TB/mo free tier.

    win_from/win_to bound the scan explicitly. Use them to TOP UP a gap rather than
    re-scanning eleven months: the four settled-forward markets were retired from
    the live slate on 2026-08-24, so their attention series stops on 2026-07-05 and
    every reconstruction snapshot after that date would silently lose the GDELT
    burst term (burst_z -> None -> amplify no-ops) while the other 40 markets keep
    it. Refreshing the window restores parity instead of degrading four markets."""
    universe = json.loads(UNIVERSE.read_text())
    name_keys = {m["slug"]: m["gdelt_keys"] for m in universe if m.get("gdelt_keys")}
    lo = win_from or min(m["start_date"] for m in universe)
    hi = win_to or max(m["end_date"] for m in universe)
    t = datetime.fromisoformat(lo) - (timedelta(0) if win_from else timedelta(days=16))
    end = datetime.fromisoformat(hi)
    cache = {}
    while t < end:
        chunk_end = min(t + timedelta(days=70), end)
        cache = gdelt_bq.pull_daily_series(name_keys, t.strftime("%Y-%m-%d"),
                                           chunk_end.strftime("%Y-%m-%d"))
        print(f"  chunk {t:%Y-%m-%d}..{chunk_end:%Y-%m-%d} done")
        t = chunk_end + timedelta(days=1)
    print(f"pulled {len(name_keys)} name-sets -> {gdelt_bq.SERIES_CACHE}")
    print("  days per market:", sorted({len(v) for s, v in cache.items() if s in name_keys}))


# ----------------------------------------------------------------------- fit ---

def build_hist_pairs(gamma: float, gdelt_series: dict, params: dict) -> list[dict]:
    universe = json.loads(UNIVERSE.read_text())
    priors = json.loads(HIST_PRIORS.read_text()) if HIST_PRIORS.exists() else {}
    pairs, skipped = [], []
    for m in universe:
        slug = m["slug"]
        pr = priors.get(slug)
        if pr is None:
            skipped.append((slug, "no prior"))
            continue
        if pr.get("outcome_knowledge") == "known":
            skipped.append((slug, "canary: outcome known"))
            continue
        sdir = HIST / slug[:80]
        mids = {}
        mh = sdir / "mid_history.json"
        if mh.exists():
            mids = json.loads(mh.read_text())
        tp = fvmodel.type_params(params, m["mtype"])
        floor = params["floor_pp"].get(m["mtype"], params["floor_pp"]["shock"])
        series = gdelt_series.get(slug, {})
        state, counted = None, set()
        for d in snapshot_dates(m):
            pf = sdir / f"{d}.packet.json"
            if not pf.exists():
                continue
            pkt = json.loads(pf.read_text())
            feats = sourceweights.annotate(features.features_for(slug, pkt["articles"]))
            new = [r for r in feats if r["cache_key"] not in counted]
            counted |= {r["cache_key"] for r in new}
            s_t = fvmodel.daily_score(new)
            bz = gdelt_bq.burst_z(series, d.replace("-", "")) if series else None
            vol_z = bz["vol_z"] if bz else None
            s_eff = fvmodel.amplify(s_t, vol_z, gamma)
            state = fvmodel.step_state(state, d, s_eff, tp["lam"], tp["a_clip"], tp["s_min"])
            comp = fvmodel.band_components(feats)
            pairs.append({"slug": slug,
                          "family": m.get("family", (m.get("gdelt_keys") or [slug])[0]),
                          "date": d, "mtype": m["mtype"], "p0_pct": pr["p0_pct"],
                          "A": state["A"], "y": int(m["y"]),
                          "mid": mids.get(d), "vol_z": vol_z,
                          "shift_clip": tp["shift_clip"],
                          "band_floor": floor, "band_raw": comp["raw"],
                          "n_missing_feats": sum(1 for r in feats if r["features"] is None),
                          "src": m.get("src", "hist")})
    if skipped:
        print(f"  skipped {len(skipped)} markets: "
              + "; ".join(f"{s[:40]} ({r})" for s, r in skipped))
    return pairs


def half_at(bm: float, floor: float, raw: float) -> float:
    return min(fvmodel.BAND_CAP_PP, max(floor, floor + bm * raw))


def band_mult_coverage(pairs: list[dict], alpha: float, nominal: float = 0.8) -> dict:
    """Grid band_mult so BUCKETED band coverage ~ nominal on resolved pairs.

    For FV buckets (5): realized YES-frequency should fall inside the bucket's
    average [fv-half, fv+half]. Pre-registered in the v3 findings: rescale the
    single band_mult knob once >= 20 resolved forecasts exist; this backfill is
    that first rescale, on (market, snapshot) pairs rather than ledger entries
    (declared; the forward ledger takes over as live markets settle)."""
    rows = []
    for p in pairs:
        fv = fvmodel.fair_value(p["p0_pct"], p["A"], alpha, p.get("shift_clip"))
        rows.append({"fv": fv, "y": p["y"], "floor": p.get("band_floor", 12.0),
                     "raw": p.get("band_raw", 0.0)})

    def coverage(bm: float) -> float:
        buckets: dict[int, list[dict]] = {}
        for r in rows:
            buckets.setdefault(min(4, int(r["fv"] / 20.0)), []).append(r)
        ok = total = 0
        for b, rs in buckets.items():
            if len(rs) < 3:
                continue   # too thin to say anything about the bucket
            total += 1
            freq = sum(r["y"] for r in rs) / len(rs) * 100.0
            lo = sum(max(1.0, r["fv"] - half_at(bm, r["floor"], r["raw"])) for r in rs) / len(rs)
            hi = sum(min(99.0, r["fv"] + half_at(bm, r["floor"], r["raw"])) for r in rs) / len(rs)
            ok += 1 if lo <= freq <= hi else 0
        return ok / total if total else 0.0

    grid = [round(0.25 * k, 2) for k in range(1, 13)]   # 0.25 .. 3.0
    scores = {bm: coverage(bm) for bm in grid}
    # Selection = "narrowest knob AT nominal" (the v3.1 findings phrasing): the
    # smallest band_mult whose coverage >= nominal. With ~3 scoreable buckets,
    # coverage is quantized to thirds, so "closest to nominal" can tie-break
    # INTO undercoverage (2/3 = 0.667 vs 0.8) on pure granularity — for a
    # public band the covering side of the tie is the honest one. Falls back
    # to closest-to-nominal only if no knob reaches nominal.
    covering = [bm for bm in grid if scores[bm] >= nominal]
    best = min(covering) if covering else \
        min(grid, key=lambda bm: (abs(scores[bm] - nominal), bm))
    return {"band_mult": best, "coverage_at_best": round(scores[best], 3),
            "nominal": nominal, "n_pairs": len(rows),
            "curve": [{"band_mult": bm, "coverage": round(c, 3)}
                      for bm, c in scores.items()]}


def cmd_fit() -> None:
    import newsagent_stageb_calibration as stageb   # sibling script (sys.path[0])
    params = fvmodel.load_params()
    gdelt_series = gdelt_bq.load_series()
    gamma = params.get("gamma", 0.0) if gdelt_series else 0.0

    arch_pairs = stageb.build_pairs(gamma=gamma, gdelt_series=gdelt_series)
    for p in arch_pairs:
        p["src"] = "archive"
        p.setdefault("band_floor", params["floor_pp"].get(p["mtype"], 12.0))
        p.setdefault("band_raw", 0.0)
    hist_pairs = build_hist_pairs(gamma, gdelt_series, params)
    pairs = arch_pairs + hist_pairs
    n_mkts = len({p["slug"] for p in pairs})
    print(f"pairs: {len(arch_pairs)} archive + {len(hist_pairs)} backfill = "
          f"{len(pairs)} over {n_mkts} markets")

    fit = fvmodel.fit_alpha(pairs)
    alpha = fit["alpha"]
    print(f"alpha* = {alpha} (pooled Brier {fit['brier_at_best']} vs prior-only "
          f"{fit['brier_prior_only']}); declared gamma = {gamma}")

    # band_mult coverage rescale — hist pairs only carry real band components;
    # archive pairs lack per-snapshot feats here, so the rescale uses hist pairs
    bm = band_mult_coverage(hist_pairs, alpha)
    print(f"band_mult* = {bm['band_mult']} (bucket coverage {bm['coverage_at_best']} "
          f"vs nominal {bm['nominal']} on {bm['n_pairs']} resolved backfill pairs)")

    missing = sum(p["n_missing_feats"] for p in pairs)
    if missing:
        print(f"  WARN {missing} article-features missing from cache across pairs")

    for p in pairs:
        fv = fvmodel.fair_value(p["p0_pct"], p["A"], alpha, p.get("shift_clip")) / 100.0
        p["fv"] = round(fv, 4)
        p["brier_fv"] = round((fv - p["y"]) ** 2, 4)
        p["brier_prior"] = round((p["p0_pct"] / 100.0 - p["y"]) ** 2, 4)
        p["brier_mid_context"] = (round((float(p["mid"]) - p["y"]) ** 2, 4)
                                  if p.get("mid") not in (None, "") else None)

    params.update({"alpha": alpha, "gamma": gamma, "band_mult": bm["band_mult"],
                   "fitted_on": "v0_archive + hist_backfill (resolved >= 2026-02-15)",
                   "n_pairs": fit["n_pairs"], "n_markets": fit["n_markets"],
                   "notes": ("alpha grid-fit on v0-archive + v3.1 historical-backfill "
                             "resolved outcomes; band_mult set by the pre-registered "
                             "bucket-coverage rescale on the backfill pairs. "
                             "lambda/floors/gamma declared, not fit.")})
    fvmodel.save_params(params)

    CSV_OUT.mkdir(parents=True, exist_ok=True)
    cols = ["slug", "family", "src", "date", "mtype", "p0_pct", "A", "fv", "y", "mid",
            "brier_fv", "brier_prior", "brier_mid_context", "vol_z",
            "band_floor", "band_raw", "n_missing_feats"]
    with open(CSV_OUT / "newsagent_hist_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(pairs)
    with open(CSV_OUT / "newsagent_hist_fit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "brier"])
        for row in fit["curve"]:
            w.writerow([row["alpha"], row["brier"]])
    with open(CSV_OUT / "newsagent_hist_bandmult.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band_mult", "bucket_coverage"])
        for row in bm["curve"]:
            w.writerow([row["band_mult"], row["coverage"]])

    by_src: dict[str, list] = {}
    for p in pairs:
        by_src.setdefault(p["src"], []).append(p)
    for src, rows in sorted(by_src.items()):
        n = len(rows)
        print(f"  {src:>8}: n={n:>3}  fv={sum(r['brier_fv'] for r in rows)/n:.3f}  "
              f"prior={sum(r['brier_prior'] for r in rows)/n:.3f}  "
              f"mid(ctx)={sum(r['brier_mid_context'] for r in rows if r['brier_mid_context'] is not None)/max(1, sum(1 for r in rows if r['brier_mid_context'] is not None)):.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        xs = [r["alpha"] for r in fit["curve"]]
        ys = [r["brier"] for r in fit["curve"]]
        axes[0].plot(xs, ys)
        axes[0].axvline(alpha, ls="--", color="tab:green", label=f"alpha*={alpha}")
        axes[0].axhline(fit["brier_prior_only"], ls=":", color="grey",
                        label=f"prior-only {fit['brier_prior_only']}")
        axes[0].set_xlabel("alpha (evidence weight)")
        axes[0].set_ylabel("pooled Brier vs resolved outcomes")
        axes[0].set_title(f"Stage-B fit — archive+backfill ({fit['n_pairs']} pairs, "
                          f"{fit['n_markets']} markets)")
        axes[0].legend()
        bxs = [r["band_mult"] for r in bm["curve"]]
        bys = [r["coverage"] for r in bm["curve"]]
        axes[1].plot(bxs, bys, marker="o")
        axes[1].axhline(bm["nominal"], ls=":", color="grey", label="nominal 0.8")
        axes[1].axvline(bm["band_mult"], ls="--", color="tab:green",
                        label=f"band_mult*={bm['band_mult']}")
        axes[1].set_xlabel("band_mult")
        axes[1].set_ylabel("bucket coverage")
        axes[1].set_title("band coverage rescale (backfill pairs)")
        axes[1].legend()
        PLOTS.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(PLOTS / "newsagent_hist_backfill_fit.png", dpi=120)
        print(f"plot -> {PLOTS / 'newsagent_hist_backfill_fit.png'}")
    except ImportError:
        print("matplotlib unavailable — skipped the fit plot")

    print(f"params -> {fvmodel.PARAMS_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--add-settled", action="store_true",
                    help="append config.RETIRED_MARKETS (the forward slate's own "
                         "settled markets) to the universe — see cmd_add_settled")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--markets", type=int, default=None,
                    help="cap markets for --fetch (Guardian budget control)")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--features-file")
    ap.add_argument("--priors-file")
    ap.add_argument("--oob-source", default="oob:hist_backfill",
                    help="provenance stamped into _meta.source of every ingested "
                         "feature record (alpha is Haiku-era — say who extracted)")
    ap.add_argument("--pull-gdelt", action="store_true")
    ap.add_argument("--gdelt-from", help="YYYY-MM-DD lower bound for --pull-gdelt (top-up)")
    ap.add_argument("--gdelt-to", help="YYYY-MM-DD upper bound for --pull-gdelt (top-up)")
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    if args.discover:
        cmd_discover(args.limit)
    if args.add_settled:
        cmd_add_settled()
    if args.fetch:
        cmd_fetch(args.markets)
    if args.ingest:
        cmd_ingest(args.features_file, args.priors_file, args.oob_source)
    if args.pull_gdelt:
        cmd_pull_gdelt(args.gdelt_from, args.gdelt_to)
    if args.fit:
        cmd_fit()
    if not (args.discover or args.add_settled or args.fetch or args.ingest
            or args.pull_gdelt or args.fit):
        print("nothing to do — pass --discover / --add-settled / --fetch / "
              "--ingest / --pull-gdelt / --fit")


if __name__ == "__main__":
    main()
