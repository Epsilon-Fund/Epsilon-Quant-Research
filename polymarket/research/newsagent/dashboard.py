"""Site-ready dashboard for the Epsilon Calibration Observatory (independent-FV).

Artifacts under data/newsagent/showcase/ (git-ignored, regenerable):
  showcase.json — everything the page renders (the website colleague lifts this
                  into the site repo and restyles; UX ownership his)
  index.html    — fully self-contained page (inline CSS + inline SVG, no CDN, no
                  webfonts fetched — local font stacks with graceful fallbacks)

Style (v3.2): READ-ONLY reproduction of the DEPLOYED site design language
(https://epsilon-site-changes.vercel.app — inspected 2026-07-05): warm charcoal
#242423, surface #49413c, cream text/accent #d1cab7, muted #b8b7ad, terracotta
highlight #cc5c44, hairline borders at text-color 10%, 8-10px radii, pill
buttons, numbered "01." sections, mono uppercase tag chips, IBM Plex Sans
(light-weight headings) / Geist Mono numerals — all via local font stacks.
Nothing is shipped into the colleague's site repo from here.

Layout (v3.2): the PAGE is two columns — left pane = the aggregated news/
evidence feed (sticky, own scroll), right pane = markets. Every market's donut
gauge sits in a full-width overview grid, visible WITHOUT expanding anything;
deep detail (time series, FV construction) stays in collapsible cards. Extras:
a big-movers strip (largest FV change since each market's previous published snapshot,
which mirrors the append-only ledger) and an evidence-quality badge per market.

Framing rules (enforced in copy):
  - Epsilon's INDEPENDENT fair value, judged against RESOLVED OUTCOMES over time;
    the Polymarket mid is discovery + context, never the benchmark;
  - the retrospective gate scoreboard (market beat the agent) stays displayed —
    that closure is permanent and part of the story;
  - numbers-only default; ANALYTICAL toggle reveals drivers/evidence/method/breakdown;
  - divergence flags are informational ("where our model most disagrees"), never
    an edge claim;
  - IP-scrub: public data + our own forecasts only; attribution for Guardian/
    Wikipedia/RSP/Iffy/AllSides in the footer.
"""
from __future__ import annotations

import csv
import html
import json
import math
from datetime import datetime, timezone

from . import sourcelean, sourceweights
from .config import (CSV_OUT, DATA, SHOWCASE, DIVERGENCE_GAP_PP,
                     DIVERGENCE_HALF_MAX_PP, DIVERGENCE_NREL_MIN)

# ---- design tokens (deployed epsilon site, read-only borrow; v3.2) -------------
BG = "#242423"          # --color-bg
PANEL = "#49413c"       # --color-surface
PANEL2 = "#34322d"      # nested surfaces
TX = "#d1cab7"          # --color-text (the site's accent IS the text color)
DIM = "#b8b7ad"         # --color-muted
LINE = "rgba(209,202,183,0.12)"   # border = text at ~10-12%
ACC = "#d1cab7"         # monochrome accent (cream)
HI = "#cc5c44"          # --color-highlight (terracotta) — flags/negatives only
NEG = "#cc5c44"
SERIF = "'Instrument Serif', Georgia, 'Times New Roman', serif"
SANS = "'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif"
MONO = "'Geist Mono', ui-monospace, 'SF Mono', Menlo, monospace"


def _read_csv(name: str) -> list[dict]:
    p = CSV_OUT / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ data model --

def _interim_scores(fv_series: dict) -> dict:
    """ForecastBench-convention interim read: FV_t scored against the NEXT day's
    mid (as a truth proxy while unresolved). Replaced by outcome scoring at
    resolution; populated from day one thanks to the labeled backfill segment."""
    per_market, rows = {}, []
    for slug, ser in fv_series.items():
        pts = [p for p in ser if p.get("mid_pct") is not None]
        briers = []
        for a, b in zip(pts, pts[1:]):
            ib = ((a["fv_pct"] - b["mid_pct"]) / 100.0) ** 2
            briers.append(ib)
            rows.append({"slug": slug, "date": a["date"], "interim_brier": round(ib, 4),
                         "segment": a.get("segment", "live")})
        if briers:
            per_market[slug] = {"n": len(briers),
                                "mean_interim_brier": round(sum(briers) / len(briers), 4)}
    overall = [r["interim_brier"] for r in rows]
    return {"note": ("Interim read: each day's FV scored against the NEXT day's market "
                     "mid (ForecastBench convention) while the market is unresolved; "
                     "replaced by true outcome scoring at resolution. Not a gate."),
            "overall_mean": round(sum(overall) / len(overall), 4) if overall else None,
            "n": len(overall), "per_market": per_market}


SIMLIVE_PATH = DATA.parent / "simlive" / "simlive_results.json"


def _simlive() -> dict | None:
    """The SIMULATED-LIVE RECONSTRUCTION payload — a DIFFERENT OBJECT from the
    forward ledger and never mixed with it.

    Produced by scripts/newsagent_simlive.py: what the current model WOULD have
    published on already-resolved markets, replayed lookahead-free. It is not a
    record of forecasts we made. Nothing here is written to the sf ledger, nothing
    here is summed with `live_track_record`, and every element it renders carries
    the word "reconstruction". Absent file -> the section simply does not render.
    """
    if not SIMLIVE_PATH.exists():
        return None
    try:
        d = json.loads(SIMLIVE_PATH.read_text())
    except (ValueError, OSError):
        return None
    return d if d.get("label") == "reconstruction" and d.get("n_markets") else None


def _simlive_bins(sl: dict) -> list[dict]:
    """Reconstruction reliability curve in the shape _svg_reliability expects."""
    out = []
    for b in sl["at_resolution"]["reliability_curve"]:
        if not b["n"]:
            continue
        out.append({"lo": 0.0, "hi": 1.0, "mean_fv": b["mean_forecast_pct"] / 100.0,
                    "outcome_rate": b["observed_yes_pct"] / 100.0, "n": b["n"]})
    return out


# Reach reality, stated on the page rather than only in the build notes. These are
# MEASURED counts from the shipped agent-reach extension, not aspirations.
REACH_STATUS_NOTE = (
    "Primary-source reach, honestly: the page channel currently reaches <b>two</b> "
    "domains that carry a trustworthy publication date — federalreserve.gov and "
    "state.gov/releases. The best instrument found for the Hormuz questions, "
    "ukmto.org, stays <b>navigation-only</b> because nothing on the page dates it. "
    "The dateline path is also <b>T+1</b>: a document published on the run day is "
    "refused rather than risked, so today's official statement reaches the packet "
    "tomorrow. Official documents show headline, source and link only; transcripts "
    "and private analysis are counted and never shown."
)


def _reliability_bins(nbins: int = 5) -> list[dict]:
    """Reliability data from the Stage-B archive calibration pairs (in-sample,
    resolved outcomes). The forward ledger takes over as markets settle."""
    pairs = _read_csv("newsagent_stageb_pairs.csv")
    if not pairs:
        return []
    bins = []
    for i in range(nbins):
        lo, hi = i / nbins, (i + 1) / nbins
        rows = [p for p in pairs if lo <= float(p["fv"]) < hi or (i == nbins - 1 and float(p["fv"]) == 1.0)]
        if not rows:
            continue
        bins.append({"lo": lo, "hi": hi,
                     "mean_fv": round(sum(float(r["fv"]) for r in rows) / len(rows), 3),
                     "outcome_rate": round(sum(int(r["y"]) for r in rows) / len(rows), 3),
                     "n": len(rows)})
    return bins


def _movers(cards: list[dict], fv_series: dict, top_n: int = 6) -> dict:
    """Big movers: largest FV change since each market's PREVIOUS PUBLISHED
    snapshot, from the live series only (backfill/reconstructed segments never
    count — they are not in the append-only sf ledger).

    v3.3 wording fix: the observatory runs ATTENDED, so consecutive snapshots can
    be days or weeks apart (2026-07-05 -> 2026-08-24 was 50 days). Calling that a
    "day-over-day" move, as v3.2 did, overstated the cadence; every row carries
    its own from/to dates and the gap in days."""
    items = []
    for c in cards:
        pts = [p for p in fv_series.get(c["slug"], [])
               if p.get("segment", "live") == "live" and p.get("fv_pct") is not None]
        if len(pts) < 2:
            continue
        prev, last = pts[-2], pts[-1]
        try:
            gap_days = (datetime.fromisoformat(last["date"])
                        - datetime.fromisoformat(prev["date"])).days
        except ValueError:
            gap_days = None
        items.append({"slug": c["slug"], "question": c["question"],
                      "fv_pct": last["fv_pct"],
                      "delta_pp": round(last["fv_pct"] - prev["fv_pct"], 1),
                      "from_date": prev["date"], "to_date": last["date"],
                      "gap_days": gap_days})
    items.sort(key=lambda m: -abs(m["delta_pp"]))
    gaps = [m["gap_days"] for m in items[:top_n] if m.get("gap_days")]
    span = (f" These snapshots are {min(gaps)}–{max(gaps)} days apart: the "
            "observatory is attended, not a daily cron, so a move is the change "
            "since the last published number, not a one-day move."
            if gaps and max(gaps) > 1 else "")
    return {"items": items[:top_n],
            "note": ("Largest changes in our fair value since each market's "
                     "previous PUBLISHED snapshot (reconstructed history never "
                     "counts). Big moves trace to the evidence shown on the "
                     "market's card." + span)
            if items else
            "Movers appear once a market has two published snapshots."}


def _extended_scorecard() -> dict | None:
    """Optional extras from the `calibrate` skill (log-loss, ECE, Spiegelhalter Z).

    calibrate is a READ-ONLY consumer of the same ledger and is the scorer of
    record; it lives behind lemma-calibrate, which the daily loop must not depend
    on — if it is not importable the page simply shows the headline Brier, which
    dashboard code computes from the ledger itself.
    """
    try:
        from lib.calibration import core as cal   # noqa: PLC0415
        sc = cal.score_ledger(book="polymarket")
    except Exception:
        return None
    # keep only JSON-serialisable scalars: score_ledger also returns a pandas
    # reliability table, and showcase.json must stay plain JSON for the website.
    return {"n": sc.get("n"), "brier": sc.get("brier"),
            "log_loss": sc.get("log_loss"), "ece": sc.get("ece"), "mce": sc.get("mce"),
            "murphy": {k: v for k, v in (sc.get("murphy") or {}).items()},
            "spiegelhalter": {k: v for k, v in (sc.get("spiegelhalter") or {}).items()},
            "calibration_in_the_large": {
                k: v for k, v in (sc.get("calibration_in_the_large") or {}).items()}}


def _live_track_record(fv_series: dict | None = None) -> dict:
    """The public FORWARD track record: every settled ledger entry, scored on the
    number we had actually published when the market resolved.

    Reported PER METHOD and never merged (DC-8, approved 2026-08-24): a Brier
    earned by the news-only model is not credited to a later news+data model. A
    method that is live but has settled nothing yet shows the honest transition
    line "new method — no settled track record yet (n=0)".
    """
    from . import config, ledger   # noqa: PLC0415
    recs = ledger.settled_records()
    fv_series = fv_series or {}
    for r in recs:
        pts = [p for p in fv_series.get(r["slug"], [])
               if p.get("segment", "live") == "live" and p.get("mid_pct") is not None]
        r["mid_pct_at_last_snapshot"] = pts[-1]["mid_pct"] if pts else None
        lag = None
        if r["resolution_date"] and r["last_update"]:
            try:
                lag = (datetime.fromisoformat(r["resolution_date"])
                       - datetime.fromisoformat(r["last_update"])).days
            except ValueError:
                lag = None
        r["stale_days"] = lag
    live_methods = {ledger.method_for(slug) for slug in config.LIVE_MARKETS}
    per_method = {}
    for m in sorted(live_methods | {r["method"] for r in recs}):
        rows = [r for r in recs if r["method"] == m]
        per_method[m] = {
            "n": len(rows),
            "brier": round(sum(r["brier"] for r in rows) / len(rows), 4) if rows else None,
            "base_rate": round(sum(r["outcome"] for r in rows) / len(rows), 3) if rows else None,
            "note": ("new method — no settled track record yet (n=0)" if not rows else
                     f"{len(rows)} settled forecast{'s' if len(rows) != 1 else ''}"),
        }
    stale = [r["stale_days"] for r in recs if r["stale_days"] is not None]
    return {
        "status": "scored" if recs else "collecting",
        "n_settled": len(recs),
        "brier": round(sum(r["brier"] for r in recs) / len(recs), 4) if recs else None,
        "per_method": per_method,
        "records": recs,
        "max_stale_days": max(stale) if stale else None,
        "extended": _extended_scorecard(),
        "note": ("Daily snapshots go to an append-only forecast ledger (anti-post-hoc: "
                 "settled entries reject edits); Brier and reliability appear here as "
                 "markets resolve." if not recs else
                 "Each row is scored on the probability we had ALREADY PUBLISHED when "
                 "the market resolved — the ledger is append-only and settled entries "
                 "reject edits, so nothing here can be improved after the fact. Scores "
                 "are kept separately per method and never merged. The market mid is "
                 "shown as context only: with a handful of resolutions a head-to-head "
                 "would be noise, and the pre-registered gates that closed the "
                 "beat-the-mid claim are the scoreboard above."),
    }


def _norm_title(t: str) -> str:
    return "".join(ch for ch in (t or "").lower() if ch.isalnum())[:80]


def _build_feed(cards: list[dict], max_items: int = 80) -> dict:
    """The left-pane news/evidence feed (v3.2): every public item the extractor
    read across all markets, deduped cross-market by (title, domain), newest
    first, each tagged with source lean + the market(s) it informed. Private
    (display=False) items never reach cards' evidence lists, so they can never
    reach this feed either — only their count is shown."""
    seen: dict[tuple, dict] = {}
    n_private = 0
    for c in cards:
        n_private += c.get("n_private_items", 0)
        for e in c.get("evidence", []):
            key = (_norm_title(e.get("title", "")), e.get("domain", ""))
            rec = seen.get(key)
            if rec is None:
                rec = {"title": e.get("title", ""), "domain": e.get("domain", ""),
                       "url": e.get("url", ""), "seendate": e.get("seendate", ""),
                       "lean": e.get("lean", "unrated"), "markets": []}
                seen[key] = rec
            if c["slug"] not in [m["slug"] for m in rec["markets"]]:
                rec["markets"].append({"slug": c["slug"],
                                       "question": c["question"][:46]})
    items = sorted(seen.values(), key=lambda r: r.get("seendate", ""), reverse=True)
    return {"items": items[:max_items], "n_total": len(items), "n_private": n_private,
            "note": ("Everything shown is headline + source + link back to the "
                     "original; lean per the curated AllSides-informed table; "
                     "private analysis items (newsletters/research) are counted, "
                     "never displayed.")}


def build_showcase(snapshots: list[dict], fv_series: dict) -> dict:
    backtests = []
    for variant, label in [("", "v0 (single-sample forecaster)"),
                           ("_v0b", "v0b (5-perspective ensemble)")]:
        rows = _read_csv(f"newsagent_v0{variant}_metrics.csv")
        if rows:
            m = rows[0]
            backtests.append({
                "label": label, "n_pairs": int(m["n_pairs"]), "n_markets": int(m["n_markets"]),
                "brier_agent": float(m["brier_ours"]), "brier_market": float(m["brier_mid"]),
                "verdict": "market wins" if float(m["brier_diff"]) > 0 else "agent wins",
            })
    cards = []
    for s in snapshots:
        mkt, fc, sb = s["market"], s["forecast"], s.get("stage_b", {})
        div = sb.get("divergence", {})
        # public evidence feed: display=False items (private newsletters) NEVER
        # appear — only their count does. Everything shown is headline+source+link.
        shown = [a for a in s["packet"]["articles"]
                 if a.get("display", True) and not _never_public(a.get("domain", ""))]
        n_private = len(s["packet"]["articles"]) - len(shown)
        half_pp = sb.get("half_pp",
                         round((fc["band_hi_pct"] - fc["band_lo_pct"]) / 2, 1))
        quality = sb.get("evidence_quality")
        if quality is None:
            from . import fvmodel
            quality = fvmodel.evidence_quality(sb.get("n_relevant", 0), half_pp,
                                               DIVERGENCE_HALF_MAX_PP,
                                               DIVERGENCE_NREL_MIN)
        cards.append({
            "slug": mkt["slug"], "question": mkt["question"], "region": s.get("region", ""),
            "mtype": sb.get("mtype", ""), "deadline": mkt["end_date"][:10],
            "tract": ("data_scored" if sb.get("p0_source") == "p_struct"
                      else sb.get("tract", "news")),
            "tract_note": sb.get("tract_note", ""),
            "method": sb.get("method", "news"),
            "p0_source": sb.get("p0_source", "onboarding_prior"),
            "p0_used_pct": sb.get("p0_used_pct"),
            "n_double_counted": sb.get("n_double_counted", 0),
            "market_implied": sb.get("market_implied"),
            "fv_pct": fc["p_pct"], "band": [fc["band_lo_pct"], fc["band_hi_pct"]],
            "market_pct": round(mkt["mid"] * 100, 1),
            "gap_pp": div.get("gap_pp", round(fc["p_pct"] - mkt["mid"] * 100, 1)),
            "divergence_flag": bool(div.get("flag")),
            "n_relevant": sb.get("n_relevant", 0),
            "evidence_quality": quality,
            "volume24h": mkt["volume24h"], "liquidity": mkt["liquidity"],
            "drivers": s.get("drivers", [])[:3],
            "gdelt": sb.get("gdelt"),
            "breakdown": sb.get("breakdown"),
            "bias": sb.get("bias"),
            "evidence": [{"title": a["title"], "domain": a["domain"],
                          "seendate": a.get("seendate", ""), "url": a.get("url", ""),
                          "lean": sourcelean.lean_label(
                              sourcelean.get_lean(a.get("domain", "")))}
                         for a in shown],
            "n_private_items": n_private,
            "n_transcripts": sum(1 for a in s["packet"]["articles"]
                                 if a.get("reach_kind") == "transcript"),
            "n_reach_docs": sum(1 for a in s["packet"]["articles"]
                                if a.get("reach_kind") == "document"),
            "series": fv_series.get(mkt["slug"], []),
            "sf_id": s.get("sf_id", ""),
        })
    cards.sort(key=lambda c: (not c["divergence_flag"], -abs(c["gap_pp"])))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framing": ("Epsilon Observatory — our independent fair value on liquid "
                    "politics and macro questions, scored in public against what "
                    "actually happens. Polymarket tells us which questions matter "
                    "and provides display context; it is not the benchmark. We hold "
                    "no inside information, so misses on shock events are expected "
                    "— and scored anyway. In two pre-registered retrospective gates "
                    "the market mid beat our earlier forecaster; that scoreboard "
                    "stays up. The craft on display is measurement discipline."),
        "method": ("Per market: a one-time onboarding prior from a five-perspective "
                   "LLM ensemble (never shown the market price), then a transparent "
                   "daily update — a cheap LLM extracts structured features from each "
                   "news article (relevance, stance, event phase, strength; cached per "
                   "article), and a fitted log-odds model turns them into the fair "
                   "value and band. The single evidence weight is calibrated on "
                   "resolved outcomes only. Each article's influence is scaled by "
                   "source reliability (Wikipedia perennial-sources tiers + Iffy "
                   "blocklist) and by political-lean extremity (curated AllSides-"
                   "informed table) — lean never sets or flips direction. A global-"
                   "attention signal (GDELT GKG article volume for the question's "
                   "entities) can amplify a day's directional evidence when world "
                   "coverage bursts; it never sets the direction. Every number "
                   "decomposes into the article-level contributions shown in "
                   "analytical mode."),
        "divergence_rule": (f"Flag shown only when |FV − mid| ≥ {DIVERGENCE_GAP_PP:g}pp "
                            f"AND band half-width ≤ {DIVERGENCE_HALF_MAX_PP:g}pp AND "
                            f"≥ {DIVERGENCE_NREL_MIN} relevant articles in 72h. "
                            "Informational: where our model most disagrees — never an "
                            "edge claim."),
        "backtests": backtests,
        "interim": _interim_scores(fv_series),
        "reliability": {"bins": _reliability_bins(),
                        "note": ("In-sample reliability of the Stage-B model on the "
                                 "resolved June-2026 archive (the calibration set). "
                                 "The forward ledger becomes the real track record "
                                 "as live markets settle.")},
        "live_track_record": _live_track_record(fv_series),
        # A DIFFERENT OBJECT from live_track_record. Kept as its own key precisely so
        # nothing downstream can accidentally sum the two.
        "simlive": _simlive(),
        "reach_status": REACH_STATUS_NOTE,
        "markets": cards,
        "movers": _movers(cards, fv_series),
        "feed": _build_feed(cards),
        "attribution": ("Headlines: The Guardian (Open Platform), BBC/Sky/Politico/"
                        "The Hill RSS, and Wikipedia Current events portal (CC BY-SA "
                        "4.0) — links go to the sources. Market data: Polymarket "
                        "public APIs. " + sourceweights.ATTRIBUTION + " "
                        + sourcelean.ATTRIBUTION),
    }


# ------------------------------------------------------------------ SVG charts --

def _x(i: int, n: int, w: int, pad: int = 34) -> float:
    return pad + (w - 2 * pad) * (i / max(1, n - 1))


def _y(pct: float, h: int, pad: int = 16) -> float:
    return pad + (h - 2 * pad) * (1 - pct / 100.0)


def _svg_series(series: list[dict], w: int = 660, h: int = 190) -> str:
    """FV + band vs mid time series. Backfill segment dashed; live solid."""
    pts = [p for p in series if p.get("fv_pct") is not None]
    if len(pts) < 2:
        return f'<div class="nochart">time series appears after a few daily runs</div>'
    n = len(pts)
    band = " ".join(f"{_x(i, n, w):.1f},{_y(p['band_hi_pct'], h):.1f}" for i, p in enumerate(pts))
    band += " " + " ".join(f"{_x(i, n, w):.1f},{_y(p['band_lo_pct'], h):.1f}"
                           for i, p in reversed(list(enumerate(pts))))

    def path(sel, key) -> str:
        idx = [(i, p) for i, p in enumerate(pts) if sel(p) and p.get(key) is not None]
        if len(idx) < 2:
            return ""
        return "M " + " L ".join(f"{_x(i, n, w):.1f} {_y(p[key], h):.1f}" for i, p in idx)

    fv_back = path(lambda p: p.get("segment") == "backfill", "fv_pct")
    # join live to the last backfill point so the line is continuous
    first_live = next((i for i, p in enumerate(pts) if p.get("segment") == "live"), None)
    live_from = max(0, (first_live or 0) - 1)
    fv_live = "M " + " L ".join(
        f"{_x(i, n, w):.1f} {_y(p['fv_pct'], h):.1f}"
        for i, p in enumerate(pts) if i >= live_from) if first_live is not None else ""
    mid = path(lambda p: True, "mid_pct")
    gl = "".join(f'<line x1="34" x2="{w-34}" y1="{_y(v, h):.1f}" y2="{_y(v, h):.1f}" '
                 f'stroke="{LINE}" stroke-width="1"/>'
                 f'<text x="6" y="{_y(v, h)+4:.1f}" fill="{DIM}" font-size="9" '
                 f'font-family="{MONO}">{v}</text>' for v in (0, 25, 50, 75, 100))
    d0, d1 = pts[0]["date"][5:], pts[-1]["date"][5:]
    return f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">
{gl}
<polygon points="{band}" fill="{ACC}" opacity="0.09"/>
{f'<path d="{fv_back}" fill="none" stroke="{ACC}" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.65"/>' if fv_back else ''}
{f'<path d="{fv_live}" fill="none" stroke="{ACC}" stroke-width="2"/>' if fv_live else ''}
{f'<path d="{mid}" fill="none" stroke="{DIM}" stroke-width="1.4" opacity="0.7"/>' if mid else ''}
<text x="34" y="{h-2}" fill="{DIM}" font-size="9" font-family="{MONO}">{d0}</text>
<text x="{w-60}" y="{h-2}" fill="{DIM}" font-size="9" font-family="{MONO}">{d1}</text>
</svg>
<div class="chartkey"><span style="color:{TX}">— FV + band</span>
<span style="color:{TX};opacity:.65">- - reconstructed (pre-launch backfill, unscored)</span>
<span style="color:{DIM}">— Polymarket mid (context)</span></div>"""


def _svg_reliability(bins: list[dict], w: int = 340, h: int = 300) -> str:
    if not bins:
        return f'<div class="nochart">reliability chart appears once calibration pairs exist</div>'
    pad = 38
    def xx(v): return pad + (w - pad - 12) * v
    def yy(v): return (h - pad) - (h - pad - 12) * v
    pts = "".join(
        f'<circle cx="{xx(b["mean_fv"]):.1f}" cy="{yy(b["outcome_rate"]):.1f}" '
        f'r="{min(14, 4 + b["n"] * 0.45):.1f}" fill="{ACC}" opacity="0.7"/>'
        f'<text x="{xx(b["mean_fv"]):.1f}" y="{yy(b["outcome_rate"]) - 12:.1f}" fill="{DIM}" '
        f'font-size="9" text-anchor="middle" font-family="{MONO}">n={b["n"]}</text>'
        for b in bins)
    ticks = "".join(
        f'<text x="{xx(v):.1f}" y="{h-pad+14}" fill="{DIM}" font-size="9" text-anchor="middle" font-family="{MONO}">{v:.0%}</text>'
        f'<text x="{pad-8}" y="{yy(v)+3:.1f}" fill="{DIM}" font-size="9" text-anchor="end" font-family="{MONO}">{v:.0%}</text>'
        for v in (0, 0.25, 0.5, 0.75, 1.0))
    return f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">
<line x1="{pad}" y1="{h-pad}" x2="{w-12}" y2="{h-pad}" stroke="{LINE}"/>
<line x1="{pad}" y1="12" x2="{pad}" y2="{h-pad}" stroke="{LINE}"/>
<line x1="{xx(0):.1f}" y1="{yy(0):.1f}" x2="{xx(1):.1f}" y2="{yy(1):.1f}" stroke="{DIM}" stroke-dasharray="4 4" stroke-width="1"/>
{pts}{ticks}
<text x="{(w+pad)/2:.0f}" y="{h-4}" fill="{DIM}" font-size="10" text-anchor="middle" font-family="{SANS}">model fair value</text>
<text x="12" y="{h/2:.0f}" fill="{DIM}" font-size="10" text-anchor="middle" font-family="{SANS}" transform="rotate(-90 12 {h/2:.0f})">observed outcome rate</text>
</svg>"""


def _svg_staleness(sl: dict, w: int = 660, h: int = 300) -> str:
    """The CENTREPIECE: pooled Brier as a function of snapshot age.

    Two series. The BALANCED one (markets carrying all eight grid snapshots) is the
    honest read — the raw one moves partly because short-lived markets drop out of
    the far horizons, and a curve that changes shape because its sample changed is
    not a finding. Lower is better; the x-axis runs old -> fresh left to right.
    """
    raw = [r for r in sl["staleness"]["raw"] if r["brier"] is not None]
    bal = [r for r in sl["staleness"]["balanced"] if r["brier"] is not None]
    if not bal:
        return '<div class="nochart">staleness curve appears once the reconstruction runs</div>'
    pad_l, pad_b, pad_t = 46, 40, 14
    xs = [r["days_before_resolution"] for r in bal]
    lo_x, hi_x = min(xs), max(xs)
    vals = [r["brier"] for r in raw + bal]
    lo_y, hi_y = 0.0, max(vals) * 1.18
    def xx(d):   # old (left) -> fresh (right)
        return pad_l + (w - pad_l - 14) * (hi_x - d) / max(1, hi_x - lo_x)
    def yy(v):
        return (h - pad_b) - (h - pad_b - pad_t) * (v - lo_y) / (hi_y - lo_y or 1)

    def path(rows, colour, dash, width):
        d = " ".join(f'{"M" if i == 0 else "L"}{xx(r["days_before_resolution"]):.1f},'
                     f'{yy(r["brier"]):.1f}' for i, r in enumerate(sorted(
                         rows, key=lambda r: -r["days_before_resolution"])))
        dots = "".join(f'<circle cx="{xx(r["days_before_resolution"]):.1f}" '
                       f'cy="{yy(r["brier"]):.1f}" r="3.4" fill="{colour}"/>' for r in rows)
        return (f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"'
                + (f' stroke-dasharray="{dash}"' if dash else "")
                + f'/>{dots}')

    gridlines, yticks = "", ""
    steps = 4
    for i in range(steps + 1):
        v = lo_y + (hi_y - lo_y) * i / steps
        y = yy(v)
        gridlines += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-14}" y2="{y:.1f}" stroke="{LINE}"/>'
        yticks += (f'<text x="{pad_l-8}" y="{y+3:.1f}" fill="{DIM}" font-size="9" '
                   f'text-anchor="end" font-family="{MONO}">{v:.2f}</text>')
    xticks = "".join(
        f'<text x="{xx(r["days_before_resolution"]):.1f}" y="{h-pad_b+15}" fill="{DIM}" '
        f'font-size="9" text-anchor="middle" font-family="{MONO}">'
        f'{r["days_before_resolution"]}d</text>'
        f'<text x="{xx(r["days_before_resolution"]):.1f}" y="{h-pad_b+27}" fill="{DIM}" '
        f'font-size="8" text-anchor="middle" font-family="{MONO}">n={r["n"]}</text>'
        for r in bal)
    legend = (f'<circle cx="{pad_l+8}" cy="{pad_t+2}" r="3.4" fill="{HI}"/>'
              f'<text x="{pad_l+18}" y="{pad_t+6}" fill="{DIM}" font-size="9.5" '
              f'font-family="{SANS}">balanced — the '
              f'{sl["staleness"]["n_markets_balanced"]} markets with all 8 snapshots '
              f'(the honest read)</text>'
              f'<circle cx="{pad_l+8}" cy="{pad_t+18}" r="3.4" fill="{DIM}"/>'
              f'<text x="{pad_l+18}" y="{pad_t+22}" fill="{DIM}" font-size="9.5" '
              f'font-family="{SANS}">raw — all {sl["n_markets"]} markets, n varies by '
              f'horizon</text>')
    return f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Reconstruction: pooled Brier by snapshot age">
{gridlines}
<line x1="{pad_l}" y1="{h-pad_b}" x2="{w-14}" y2="{h-pad_b}" stroke="{LINE}"/>
<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h-pad_b}" stroke="{LINE}"/>
{path(raw, DIM, "4 4", 1.4)}
{path(bal, HI, "", 2.2)}
{yticks}{xticks}{legend}
<text x="{(w+pad_l)/2:.0f}" y="{h-4}" fill="{DIM}" font-size="10" text-anchor="middle"
      font-family="{SANS}">snapshot age at publication — older (left) to fresher (right)</text>
<text x="13" y="{h/2:.0f}" fill="{DIM}" font-size="10" text-anchor="middle"
      font-family="{SANS}" transform="rotate(-90 13 {h/2:.0f})">pooled Brier (lower is better)</text>
</svg>"""


def _svg_gauge(fv: float, lo: float, hi: float, mid: float,
               w: int = 150, h: int = 92) -> str:
    """Per-market semicircle gauge (kept for the card's numerals row): 0-100 arc,
    band segment, FV needle, mid tick (grey, context)."""
    cx, cy, r = w / 2, h - 10, 58

    def pt(pct: float, rad: float) -> tuple[float, float]:
        a = math.pi * (1 - pct / 100.0)
        return cx + rad * math.cos(a), cy - rad * math.sin(a)

    def arc(p0: float, p1: float, rad: float) -> str:
        x0, y0 = pt(p0, rad)
        x1, y1 = pt(p1, rad)
        large = 1 if abs(p1 - p0) > 50 else 0
        return f"M {x0:.1f} {y0:.1f} A {rad} {rad} 0 {large} 1 {x1:.1f} {y1:.1f}"

    nx, ny = pt(fv, r - 8)
    mx0, my0 = pt(mid, r - 3)
    mx1, my1 = pt(mid, r + 5)
    return f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="fair value gauge">
<path d="{arc(0, 100, r)}" fill="none" stroke="{LINE}" stroke-width="7"/>
<path d="{arc(max(0.5, lo), min(99.5, hi), r)}" fill="none" stroke="{ACC}" stroke-width="7" opacity="0.30"/>
<line x1="{mx0:.1f}" y1="{my0:.1f}" x2="{mx1:.1f}" y2="{my1:.1f}" stroke="{DIM}" stroke-width="2"/>
<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{ACC}" stroke-width="2.4" stroke-linecap="round"/>
<circle cx="{cx}" cy="{cy}" r="3.2" fill="{ACC}"/>
<text x="6" y="{h-2}" fill="{DIM}" font-size="8" font-family="{MONO}">0</text>
<text x="{w-16}" y="{h-2}" fill="{DIM}" font-size="8" font-family="{MONO}">100</text>
</svg>"""


def _svg_donut(fv: float, lo: float, hi: float, mid: float,
               size: int = 120, flagged: bool = False) -> str:
    """Overview donut (v3.2): full ring — FV arc from 12 o'clock (cream), band
    segment underneath (cream, faint), mid tick (muted, context), FV numeral in
    the center. Flagged markets get a terracotta FV arc."""
    cx = cy = size / 2
    r = size / 2 - 9

    def pt(pct: float, rad: float) -> tuple[float, float]:
        a = math.radians(-90 + 360.0 * pct / 100.0)
        return cx + rad * math.cos(a), cy + rad * math.sin(a)

    def arc(p0: float, p1: float, rad: float) -> str:
        x0, y0 = pt(p0, rad)
        x1, y1 = pt(min(p1, 99.97), rad)
        large = 1 if (p1 - p0) > 50 else 0
        return f"M {x0:.1f} {y0:.1f} A {rad} {rad} 0 {large} 1 {x1:.1f} {y1:.1f}"

    mx0, my0 = pt(mid, r - 6)
    mx1, my1 = pt(mid, r + 6)
    color = HI if flagged else ACC
    return f"""<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="fair value donut">
<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{LINE}" stroke-width="7"/>
<path d="{arc(max(0.3, lo), min(99.7, hi), r)}" fill="none" stroke="{ACC}" stroke-width="7" opacity="0.22"/>
{f'<path d="{arc(0.0, fv, r)}" fill="none" stroke="{color}" stroke-width="7" stroke-linecap="round"/>' if fv > 0.3 else ''}
<line x1="{mx0:.1f}" y1="{my0:.1f}" x2="{mx1:.1f}" y2="{my1:.1f}" stroke="{DIM}" stroke-width="2"/>
<text x="{cx}" y="{cy + 7:.0f}" fill="{TX}" font-size="{size * 0.21:.0f}" text-anchor="middle"
 font-family="{MONO}" font-weight="700">{fv:g}</text>
</svg>"""


def _svg_sparkline(series: list[dict], w: int = 120, h: int = 26) -> str:
    """FV trend preview (accent line, endpoint dot) — used in the movers strip
    and overview cells; the card's full time series stays the detail view."""
    pts = [p for p in series if p.get("fv_pct") is not None]
    if len(pts) < 2:
        return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"><text x="4" y="{h-9}" fill="{DIM}" font-size="9" font-family="{MONO}">new</text></svg>'
    vals = [p["fv_pct"] for p in pts]
    vmin, vmax = min(vals), max(vals)
    span = max(4.0, vmax - vmin)

    def xy(i: int, v: float) -> str:
        x = 3 + (w - 8) * i / (len(vals) - 1)
        y = h - 4 - (h - 8) * (v - vmin + (span - (vmax - vmin)) / 2) / span
        return f"{x:.1f} {y:.1f}"

    path = "M " + " L ".join(xy(i, v) for i, v in enumerate(vals))
    lx, ly = xy(len(vals) - 1, vals[-1]).split()
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            f'<path d="{path}" fill="none" stroke="{ACC}" stroke-width="1.5" opacity="0.9"/>'
            f'<circle cx="{lx}" cy="{ly}" r="2" fill="{ACC}"/></svg>')


def _quality_chip(q: dict | None) -> str:
    if not q:
        return ""
    tier = q.get("tier", "moderate")
    bq = q.get("band_mult_q")
    qtag = f' · band ×{bq:.2f}' if bq is not None and abs(bq - 1.0) >= 0.02 else ""
    return (f'<span class="qual qual-{tier}" title="{html.escape(q.get("note", ""))}">'
            f'evidence: {tier} · {q.get("n_relevant", 0)} arts · ±{q.get("half_pp", 0):g}pp{qtag}</span>')


def _band_quality_html(q: dict | None) -> str:
    """Visible band-quality note on the card: why the band was widened/narrowed
    (source reliability level + political spectrum balance; coverage is in the
    article count). Declared multiplier — calibration is a later pass."""
    if not q:
        return ""
    reasons = q.get("band_reasons") or []
    bq = q.get("band_mult_q")
    if not reasons or bq is None:
        return ""
    verb = "widened" if bq > 1 else "narrowed"
    return (f'<p class="note bandq" style="margin-top:.35rem">band {verb} '
            f'<span class="mono">×{bq:.2f}</span> — {html.escape("; ".join(reasons))}. '
            f'<span class="dim">(evidence-quality multiplier; magnitudes declared, '
            f'calibration pending)</span></p>')


def _overview_grid_html(cards: list[dict]) -> str:
    """All markets at a glance (v3.2): a full-width DONUT grid — every market's
    gauge visible without expanding anything, ordered by the divergence layer
    (flags first, then |gap|). Each cell: donut (FV arc + band + mid tick),
    question, fv/mid/gap numerals, badges (⚑ flag, ◆ not-news-tractable,
    evidence-quality). Click a cell to open the market's detail card."""
    cells = ""
    for c in sorted(cards, key=lambda x: (not x["divergence_flag"], -abs(x["gap_pp"]))):
        q = c["question"][:96] + ("…" if len(c["question"]) > 96 else "")
        gap_cls = "pos" if c["gap_pp"] > 0 else "neg"
        flag = '<span class="gflag">⚑</span> ' if c["divergence_flag"] else ""
        dmark = (f'<span class="dmark" title="{_TRACT_TITLE[c["tract"]]}">◆</span> '
                 if c.get("tract") in _TRACT_TITLE else "")
        cells += f"""<a class="cell" href="#card-{html.escape(c["slug"])}"
 onclick="revealCard('{html.escape(c["slug"])}')" data-flag="{1 if c["divergence_flag"] else 0}">
  <div class="celldonut">{_svg_donut(c["fv_pct"], c["band"][0], c["band"][1], c["market_pct"], flagged=c["divergence_flag"])}</div>
  <div class="cellq">{flag}{dmark}{html.escape(q)}</div>
  <div class="cellnums mono">fv <b>{c["fv_pct"]:g}%</b> · mid {c["market_pct"]:g}% ·
   <span class="{gap_cls}">{c["gap_pp"]:+g}pp</span></div>
  <div class="cellband mono">band {c["band"][0]:g}–{c["band"][1]:g}%</div>
  {_quality_chip(c.get("evidence_quality"))}
</a>"""
    return f'<div class="donutgrid">{cells}</div>'


def _movers_html(movers: dict) -> str:
    items = movers.get("items", [])
    if not items:
        return f'<p class="note">{html.escape(movers.get("note", ""))}</p>'
    cells = ""
    for m in items:
        up = m["delta_pp"] > 0
        arrow = "▲" if up else ("▼" if m["delta_pp"] < 0 else "•")
        cls = "mv-up" if up else ("mv-dn" if m["delta_pp"] < 0 else "")
        q = m["question"][:64] + ("…" if len(m["question"]) > 64 else "")
        cells += f"""<a class="mover" href="#card-{html.escape(m["slug"])}"
 onclick="revealCard('{html.escape(m["slug"])}')">
  <div class="mvdelta mono {cls}">{arrow} {m["delta_pp"]:+g}pp</div>
  <div class="mvq">{html.escape(q)}</div>
  <div class="mvnow mono">now {m["fv_pct"]:g}% <span class="dim">({m["from_date"][5:]} → {m["to_date"][5:]})</span></div>
</a>"""
    return (f'<div class="movers">{cells}</div>'
            f'<p class="note">{html.escape(movers.get("note", ""))}</p>')


def _lean_chip(lean_lbl: str) -> str:
    if not lean_lbl or lean_lbl == "unrated":
        return ""
    short = {"left": "L", "lean-left": "LL", "center": "C",
             "lean-right": "LR", "right": "R"}.get(lean_lbl, "")
    return f'<span class="lean lean-{html.escape(lean_lbl)}" title="source lean: {html.escape(lean_lbl)} (AllSides-informed)">{short}</span>'


def _feed_html(feed: dict, reach_note: str = "") -> str:
    items = ""
    for e in feed.get("items", []):
        refs = "".join(
            f'<a class="mref" href="#card-{html.escape(m["slug"])}" '
            f'onclick="revealCard(\'{html.escape(m["slug"])}\')">{html.escape(m["question"])}…</a>'
            for m in e.get("markets", [])[:2])
        link_open = f'<a href="{html.escape(e["url"])}" target="_blank" rel="noopener">' if e.get("url") else ""
        link_close = "</a>" if e.get("url") else ""
        items += f"""<li>
  <div class="fmeta"><span class="dom">{html.escape(e["domain"])}</span>{_lean_chip(e.get("lean", ""))}
   <span class="ts">{html.escape(e.get("seendate", "")[:8])}</span></div>
  <div class="ftitle">{link_open}{html.escape(e["title"])}{link_close}</div>
  <div class="frefs">{refs}</div>
</li>"""
    if feed.get("n_private"):
        items += (f'<li class="dim fpriv">+ {feed["n_private"]} private analysis item'
                  f'{"s" if feed["n_private"] != 1 else ""} (newsletters/research — '
                  'used internally, never displayed)</li>')
    return f"""<div class="feedhead">
  <div class="sub" style="margin:0">news &amp; evidence feed</div>
  <div class="note" style="margin:.3rem 0 .6rem">{html.escape(feed.get("note", ""))}</div>
</div>
<ul class="ev">{items or '<li class="dim">no public items yet today</li>'}</ul>
<div class="note reachnote">{reach_note}</div>"""


def _svg_divergence(cards: list[dict], w: int = 660) -> str:
    """Divergence layer: markets sorted by |gap|. Fixed columns so long question
    labels never collide with bars: labels left, centered bars middle, pp right."""
    if not cards:
        return ""
    rows, h_row = [], 34
    h = len(cards) * h_row + 30
    max_gap = max(20.0, max(abs(c["gap_pp"]) for c in cards))
    cx = w * 0.64                       # bar column center
    half_span = w * 0.21                # bar column half-width
    scale = half_span / max_gap
    val_x = cx + half_span + 14         # fixed value column
    for i, c in enumerate(cards):
        y = 18 + i * h_row
        g = c["gap_pp"]
        x0 = cx + min(0, g) * scale
        color = HI if c["divergence_flag"] else (DIM if abs(g) < DIVERGENCE_GAP_PP else TX)
        label = c["question"][:40] + ("…" if len(c["question"]) > 40 else "")
        rows.append(
            f'<text x="8" y="{y+4}" fill="{TX}" font-size="10.5" font-family="{SANS}">{html.escape(label)}</text>'
            f'<rect x="{x0:.1f}" y="{y-8}" width="{max(1.5, abs(g)*scale):.1f}" height="14" fill="{color}" '
            f'opacity="{1.0 if c["divergence_flag"] else 0.55}" rx="2"/>'
            f'<text x="{val_x:.1f}" y="{y+3}" fill="{color}" font-size="10" '
            f'font-family="{MONO}">{g:+.1f}pp{" ⚑" if c["divergence_flag"] else ""}</text>')
    thr = "".join(
        f'<line x1="{cx + s*DIVERGENCE_GAP_PP*scale:.1f}" y1="8" x2="{cx + s*DIVERGENCE_GAP_PP*scale:.1f}" '
        f'y2="{h-18}" stroke="{LINE}" stroke-dasharray="3 4"/>' for s in (-1, 1))
    return f"""<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">
<line x1="{cx}" y1="8" x2="{cx}" y2="{h-18}" stroke="{LINE}"/>{thr}{''.join(rows)}
<text x="{cx}" y="{h-4}" fill="{DIM}" font-size="9" text-anchor="middle" font-family="{MONO}">FV − mid (pp); dashed = ±{DIVERGENCE_GAP_PP:g}pp flag threshold</text>
</svg>"""


# ------------------------------------------------------------------ HTML render --

def _bd_row_public(a: dict) -> tuple[str, str]:
    """Breakdown row -> (public domain label, public title).

    THE PAGE BOUNDARY. Any item flagged display=False shows its generic source
    label and nothing else — never a title, never text, never a link. Keying off
    the `display` flag rather than a `newsletter:` domain prefix is what makes
    this cover reach transcripts as well: their titles are internal, and one of
    them is a content-hash-suffixed cache key that must never be public.

    Belt and braces on purpose: the flag is checked AND the domain is re-checked
    against the never-public prefixes. A breakdown record written before the flag
    existed would default to display=True, and this boundary must not depend on
    an upstream field being present to keep a secret."""
    dom = a.get("domain", "")
    if not a.get("display", True) or _never_public(dom):
        return _public_source_label(dom), "private analysis item — not displayed"
    return dom, a.get("title", "")


# Source families whose ITEM TEXT may never reach the page, whatever a record's
# display flag happens to say: private newsletters and any video transcript.
NEVER_PUBLIC_PREFIXES = ("newsletter:", "youtube.com/@")


def _never_public(domain: str) -> bool:
    return (domain or "").startswith(NEVER_PUBLIC_PREFIXES)


def _public_source_label(domain: str) -> str:
    """Generic, public-safe name for a never-displayed source."""
    from . import fvmodel
    return fvmodel._source_label(domain)


def _breakdown_html(bd: dict | None) -> str:
    if not bd:
        return ""
    steps = ""
    for a in bd["articles"][:8]:
        dom, title = _bd_row_public(a)
        steps += (f'<tr><td class="dom">{html.escape(dom)}</td>'
                  f'<td>{html.escape(title)}</td>'
                  f'<td class="mono {"pos" if a["pp_effect"] > 0 else "neg"}">'
                  f'{a["pp_effect"]:+.2f}pp</td></tr>')
    if not steps:
        steps = '<tr><td colspan="3" class="dim">no new qualifying evidence today</td></tr>'
    return f"""<div class="sub">fv construction (prior → evidence → number)</div>
<table class="bd"><tr><td class="dim">onboarding prior</td><td></td><td class="mono">{bd["p0_pct"]:.1f}%</td></tr>
<tr><td class="dim">decayed carry from prior days</td><td></td><td class="mono">{bd["carry_pp"]:+.2f}pp</td></tr>
{steps}
<tr class="tot"><td>fair value</td><td></td><td class="mono">{bd["fv_pct"]:.1f}%</td></tr></table>"""


def _gdelt_html(g: dict | None) -> str:
    if not g:
        return ""
    tone = (f' · tone {g["tone"]:+.1f}'
            + (f' (shift {g["tone_shift"]:+.1f} vs 14d)' if g.get("tone_shift") is not None else "")
            if g.get("tone") is not None else "")
    return (f'<div class="sub">global attention (GDELT GKG)</div>'
            f'<p class="note" style="margin-top:.1rem">{g["n"]:,} matched articles today '
            f'(z = {g["vol_z"]:+.1f} vs 14-day mean {g["n_trailing_mean"]:,.0f}){tone}. '
            f'Volume bursts amplify the day\'s directional evidence; tone is shown for '
            f'transparency and is not wired into the number.</p>')


def _bias_html(bias: dict | None) -> str:
    """Ratings explainer: how the number formed from sources of differing bias —
    lean (AllSides-informed curated table) × reliability (RSP tiers + Iffy)."""
    if not bias:
        return ""
    leans = bias.get("leans", {})
    tier_rows = ""
    for lean, tiers in (("YES", bias.get("yes_tiers", {})), ("NO", bias.get("no_tiers", {}))):
        for tier, srcs in tiers.items():
            names = ", ".join(
                (f"{html.escape(s)} ×{k}" if k > 1 else html.escape(s))
                + _lean_chip(leans.get(s, ""))
                for s, k in srcs)
            tier_rows += (f'<tr><td class="mono">{lean}</td>'
                          f'<td class="dim">{html.escape(tier)}</td><td>{names}</td></tr>')
    table = (f'<table class="bd"><tr><th>lean</th><th>reliability tier</th><th>sources</th></tr>'
             f'{tier_rows}</table>') if tier_rows else ""
    cov = bias.get("coverage") or {}
    cov_html = ""
    if any(cov.values()):
        cov_html = ('<p class="note" style="margin-top:.3rem">packet coverage mix '
                    '(what we read, by source lean): '
                    + " · ".join(f"{k} {v}" for k, v in cov.items() if v)
                    + " — our own packet's distribution, not any third party's "
                      "per-story figure.</p>")
    return (f'<div class="sub">how the number formed (source lean × reliability)</div>'
            f'<p class="note" style="margin-top:.1rem">{html.escape(bias["sentence"])} '
            f'Reliability per the published Scheme-A table (Wikipedia RSP tiers + Iffy '
            f'blocklist); lean per the curated AllSides-informed table.</p>{table}{cov_html}')


# Tractability chips (v3.1 tag, split 2026-08-24 into data-driven vs poll-driven —
# see newsagent/config.py). "news" markets carry no chip.
_TRACT_LABEL = {"data": "data-driven", "poll": "poll-driven",
                "data_scored": "data-channel scored"}
_TRACT_TITLE = {
    "data": "data-driven — official statistics drive this, not news text",
    "poll": "poll-driven — private/issue polling drives this, not news text",
    "data_scored": ("data-channel scored — this number is built from official "
                    "statistics, not from news text alone"),
}
# APPROVED display treatment, sign-off row 10. A market only shows the
# `data_scored` copy once it is really in config.DATA_CHANNEL_MARKETS — the
# labelling follows the method, never the intention.
_TRACT_LEAD = {
    "data_scored": ("◆ data-channel scored — this number is built from official "
                    "statistics (CPI/PCE/labour + the Cleveland Fed nowcast + FOMC "
                    "projections), entering as the anchor the news evidence then "
                    "moves. Market-implied odds are shown for context and are "
                    "never used as an input."),
    "data": ("◆ data-driven — our news-FV is structurally blind here. The question "
             "turns on official statistics rather than news text; a data-evidence "
             "channel built from public statistics is designed and pre-registered "
             "but NOT yet built, so this number is still news-only."),
    "poll": ("◆ poll-driven — our news-FV is structurally blind here. The question "
             "turns on private/campaign or ballot-issue polling, which is not an "
             "official statistic and is not available to us at any price."),
}


DC_RESULTS_PATH = DATA.parent / "datachannel" / "v3_results.json"
# The regime the LIVE data-channel market sits in. Declared, not inferred: the regime
# labels in the v3 retro-test are hand-assigned for reporting and are never a model
# input, so the mapping from "today's market" to "which row of that table applies"
# is a human call and belongs in code where it can be read.
DC_LIVE_REGIME = "holding"


def _dc_regime_html() -> str:
    """The § 5 regime split, on the card rather than only in the findings note.

    The retro-test that authorised this channel ALSO measured that the method is
    worse than doing nothing in the holding regime — and the live market is in that
    regime. Showing the pooled pass without that row would be the flattering half of
    a result we already know both halves of.
    """
    if not DC_RESULTS_PATH.exists():
        return ""
    try:
        by = json.loads(DC_RESULTS_PATH.read_text()).get("by_regime") or {}
    except (ValueError, OSError):
        return ""
    if not by:
        return ""
    order = ["hiking", "cutting", "normalisation", "holding"]
    rows = ""
    for k in [r for r in order if r in by] + [r for r in by if r not in order]:
        v = by[k]
        beats = v["brier"] < v["base"]
        here = k == DC_LIVE_REGIME
        rows += (f'<tr{" class=\"rowhi\"" if here else ""}>'
                 f'<td>{html.escape(k)}{" — <b>this market</b>" if here else ""}</td>'
                 f'<td class="mono">{v["n"]}</td>'
                 f'<td class="mono">{v["brier"]:.4f}</td>'
                 f'<td class="mono dim">{v["base"]:.4f}</td>'
                 f'<td class="{"" if beats else "neg"}">'
                 f'{"beats doing nothing" if beats else "WORSE than doing nothing"}</td></tr>')
    return (
        '<div class="sub">Method accuracy by rate regime — including where it loses</div>'
        '<table><tr><th>regime</th><th>meetings</th><th>Brier</th>'
        '<th>base-rate-only</th><th></th></tr>' + rows + '</table>'
        '<p class="note callout">Measured on 40 resolved FOMC decisions before this '
        'channel was switched on. It earns its keep where there is pressure to read, '
        'and <b>in the quiet holding regime it is worse than doing nothing</b> — which '
        'is the regime this market is in. That is stated here rather than buried: the '
        'pre-registered bars were met on the pooled set, and this row is the part of '
        'the same result that argues for caution.</p>')


def _tract_html(c: dict) -> str:
    t = c.get("tract", "news")
    if t not in _TRACT_LEAD:
        return ""
    if t == "data_scored":
        # The transition line is mandatory (row 10): a new method starts with no
        # settled forecasts of its own, and the two tracks never merge.
        extra = ('<br><b>New method — no settled track record yet (n=0).</b> The '
                 'Brier shown elsewhere on this page belongs to the news-only '
                 'method and is attributed to it; this market\'s data-channel '
                 'track starts today and is scored separately from here on.')
        mi = c.get("market_implied") or {}
        if not mi.get("available"):
            extra += ('<br><span class="dim">Market-implied context: '
                      + html.escape(mi.get("why", "unavailable")) + '.</span>')
        if c.get("n_double_counted"):
            n = c["n_double_counted"]
            extra += (f'<br><span class="dim">{n} article'
                      f'{"s" if n != 1 else ""} in this packet report a release the '
                      'data channel already counted; they are shown below but '
                      'contribute zero evidence, so the same print cannot move the '
                      'number twice.</span>')
        # When the news half contributes nothing, the published number IS the
        # structural anchor. Saying so is the difference between a reader thinking
        # two channels agreed and knowing only one of them spoke.
        p0u = c.get("p0_used_pct")
        if p0u is not None and abs(c["fv_pct"] - p0u) < 0.05:
            extra += (f'<br><b>Today the published number equals the structural '
                      f'anchor exactly ({p0u}%)</b>: no news in this packet cleared '
                      'the decisive-evidence threshold for a slow market, so the '
                      'evidence term is zero and the anchor is the answer. The news '
                      'half of "news+data" has yet to do any work on this market.')
        return (f'<div class="tractnote">{_TRACT_LEAD[t]} '
                f'{html.escape(c.get("tract_note", ""))}{extra}</div>'
                + _dc_regime_html())
    return (f'<div class="tractnote">{_TRACT_LEAD[t]} '
            f'{html.escape(c.get("tract_note", ""))} Scored in public anyway; '
            f'expect the market to carry information our packet cannot see.</div>')


_TRACT_PLAIN = {"news": "news-driven", "data": "data-driven", "poll": "poll-driven"}


def _simlive_html(sl: dict | None) -> str:
    """§ 07 — the simulated-live RECONSTRUCTION panel.

    Deliberately separated from § 06 in both senses the brief asked for: visually
    (its own `.recon` treatment and a standing "reconstruction" chip on every
    element) and verbally (the first sentence says what it is NOT). The forward
    ledger stays the headline; this is context.
    """
    if not sl:
        return ""
    a = sl["at_resolution"]
    m, sp = a["murphy"], a["spiegelhalter"]
    calib = ("<b>and it fails a calibration test</b>" if abs(sp["z"]) > 1.96
             else "and a calibration test does not reject it")
    tract_rows = "".join(
        f'<tr><td>{html.escape(_TRACT_PLAIN.get(r["tract"], r["tract"]))}</td>'
        f'<td class="mono">{r["n"]}</td><td class="mono">{r["brier"]:.4f}</td>'
        f'<td class="mono">{r["log_loss"]:.3f}</td>'
        f'<td class="mono dim">{r["base_rate_pct"]:.0f}%</td>'
        f'<td class="mono dim">{r["mean_forecast_pct"]:.0f}%</td>'
        f'<td class="dim">{"thin — read as a hint, not a result" if r["thin"] else ""}</td></tr>'
        for r in sl["splits"]["tract"])
    fam = [r for r in sl["splits"]["family"] if r["n"] >= 4]
    fam_rows = "".join(
        f'<tr><td>{html.escape(r["family"])}</td><td class="mono">{r["n"]}</td>'
        f'<td class="mono">{r["brier"]:.4f}</td>'
        f'<td class="mono dim">{r["base_rate_pct"]:.0f}%</td>'
        f'<td class="mono dim">{r["mean_forecast_pct"]:.0f}%</td></tr>' for r in fam)
    cad_rows = "".join(
        f'<tr><td>every <span class="mono">{c["cadence_days"]}</span> days</td>'
        f'<td class="mono">{c["expected_brier"]:.4f}</td></tr>'
        for c in sl["staleness"]["derived_cadence"])
    d = sl["divergence"]
    return f"""
      <div class="panel recon" style="margin-top:1.3rem">
        {_sechead("07", "Simulated-live reconstruction — what the model WOULD have shown")}
        <p class="reconlead"><span class="reconchip">reconstruction</span>
        <b>These are not forecasts we published.</b> This section replays the current
        model over {sl["n_markets"]} markets that have already resolved, and scores
        what it would have shown on each day — lookahead-free throughout. The forward
        record in § 06 above is the real one: numbers published before the fact into an
        append-only ledger. Reconstruction rows are never added to it, never averaged
        with it, and are labelled reconstruction wherever they appear.</p>

        <p class="note"><b>{sl["n_markets"]} resolved markets</b> ·
        {sl["n_snapshots"]} reconstructed snapshots · at resolution:
        Brier <span class="mono">{a["brier"]:.4f}</span> ·
        log-loss <span class="mono">{a["log_loss"]:.3f}</span> ·
        base rate <span class="mono">{a["base_rate_pct"]:.1f}%</span> ·
        mean forecast <span class="mono">{a["mean_forecast_pct"]:.1f}%</span>.
        Murphy: reliability <span class="mono">{m["reliability"]:.4f}</span> −
        resolution <span class="mono">{m["resolution"]:.4f}</span> +
        uncertainty <span class="mono">{m["uncertainty"]:.4f}</span> ·
        Spiegelhalter Z <span class="mono">{sp["z"]:+.2f}</span>
        (p&nbsp;{sp["p"]:.3f}).</p>

        <p class="note callout"><b>Read this before the Brier.</b> The reconstruction
        forecasts <span class="mono">{a["mean_forecast_pct"]:.1f}%</span> on average
        against a <span class="mono">{a["base_rate_pct"]:.1f}%</span> base rate — it
        leans systematically too far toward NO — {calib} at |Z|&nbsp;&gt;&nbsp;1.96.
        A low Brier here is mostly the base rate, not skill: the Murphy split puts
        resolution at just <span class="mono">{m["resolution"]:.4f}</span> against
        uncertainty <span class="mono">{m["uncertainty"]:.4f}</span>.</p>

        <p class="note callout"><b>This is in-sample, and that is not a detail.</b>
        The single fitted parameter (the evidence weight α, currently
        <span class="mono">{sl["params"]["alpha"]}</span>) was calibrated on these same
        resolved markets. The reconstruction is therefore a replay of a model that has
        seen this sample, not an out-of-sample test, and its Brier should be read as an
        upper bound on what the same model would score on markets it was not fitted to.
        The only genuinely out-of-sample numbers on this page are the settled forecasts
        in § 06.</p>

        <div class="sub">Brier vs snapshot age — the cadence question, answered</div>
        <div class="scrollwrap">{_svg_staleness(sl)}</div>
        <p class="note">Each point scores the number the model would have been showing
        that many days before resolution. <b>The curve is flat.</b> A fresher snapshot
        did not score better than a stale one on this sample, and the mechanism is
        visible in the model rather than mysterious: at resolution the reconstructed
        fair value sits a mean of
        <span class="mono">{a["mean_abs_shift_from_prior_pp"]:.2f}pp</span> from its
        onboarding prior, and only
        <span class="mono">{a["share_moved_ge_1pp"]:.0%}</span> of markets moved as much
        as 1pp off it — so prior-only scores
        <span class="mono">{a["brier_prior_only"]:.4f}</span> against the model's
        <span class="mono">{a["brier"]:.4f}</span>. <b>In this reconstruction, cadence is
        not the binding constraint; how much the evidence can move the number is.</b>
        That conclusion is bounded by the caveat below — the reconstruction runs in an
        evidence-poorer world than the live loop.</p>

        <div class="grid2">
          <div>
            <div class="sub">Derived — expected Brier by publish cadence
              <span class="reconchip">derived</span></div>
            <table><tr><th>run the loop</th><th>expected Brier</th></tr>{cad_rows}</table>
            <p class="note">Derived, not measured: a number published under an N-day
            cadence has an age uniform on [0,&nbsp;N), so this averages the balanced
            curve over the horizons inside that window.</p>
          </div>
          <div>
            <div class="sub">Reliability at n={sl["n_markets"]}
              <span class="reconchip">reconstruction</span></div>
            {_svg_reliability(_simlive_bins(sl), w=320, h=250)}
            <p class="note">Dot size = markets in the bin. Points sitting above the
            diagonal are the same under-forecasting the Z-score picks up.</p>
          </div>
        </div>

        <div class="sub">Where does this actually work? — accuracy by market type</div>
        <table><tr><th>market type</th><th>n</th><th>Brier</th><th>log-loss</th>
          <th>base rate</th><th>mean forecast</th><th></th></tr>{tract_rows}</table>
        <p class="note">This is the honest answer to "where does this work", and it is
        not the flattering one: the reconstruction scores <b>worse</b> on the
        news-driven markets the method is built for than on the ones our own tags call
        structurally blind. Base rates differ sharply across the three groups, so much
        of the gap is composition rather than skill — which is exactly why the base-rate
        and mean-forecast columns are shown beside every Brier.</p>

        <div class="sub">By event family (families with n ≥ 4)</div>
        <table><tr><th>family</th><th>n</th><th>Brier</th><th>base rate</th>
          <th>mean forecast</th></tr>{fam_rows}</table>
        <p class="note">Markets are clustered by event family and snapshots within a
        market are serially dependent, so no confidence interval is quoted anywhere in
        this section.</p>

        <div class="sub">Divergence flags in reconstruction</div>
        <p class="note">Of {d["n_snapshots_with_a_mid"]} reconstructed snapshots that had
        a market mid to compare against, <span class="mono">{d["n_flagged_snapshots"]}</span>
        cleared the divergence rule, across
        <span class="mono">{d["n_flagged_markets"]}</span> markets;
        {d["flagged_markets_resolved_yes"]} of those markets resolved YES, and
        {d["fv_below_mid_snapshots"]} of the flagged snapshots had our number below the
        mid. <b>{html.escape(d["closure"])}</b></p>

        <p class="note callout"><b>An evidence-poorer world than the live page.</b>
        {html.escape(sl["exclusions_note"])}</p>
      </div>"""


def _card_html(c: dict, expanded: bool) -> str:
    drivers = "".join(f"<li>{html.escape(d)}</li>" for d in c["drivers"])
    n_ev = len(c.get("evidence", []))
    ev_note = (f'{n_ev} public item{"s" if n_ev != 1 else ""} in the shared news feed '
               f'(left pane), tagged to this market')
    if c.get("n_private_items"):
        kinds = "newsletters/research"
        if c.get("n_transcripts"):
            kinds = "newsletters/research/transcripts"
        ev_note += (f' · + {c["n_private_items"]} private analysis item'
                    f'{"s" if c["n_private_items"] != 1 else ""} ({kinds} '
                    '— used internally, never displayed)')
    gap_cls = "pos" if c["gap_pp"] > 0 else "neg"
    flag = ('<span class="flag">⚑ divergence — high-confidence disagreement</span>'
            if c["divergence_flag"] else "")
    dchip = (f' · <span class="dmark">◆ {_TRACT_LABEL[c["tract"]]}</span>'
             if c.get("tract") in _TRACT_LABEL else "")
    return f"""
    <div class="card{'' if expanded else ' collapsed'}" id="card-{html.escape(c["slug"])}"
         data-slug="{html.escape(c["slug"])}" data-flag="{1 if c["divergence_flag"] else 0}">
      <div class="cardhead" onclick="toggleCard(this.parentElement)">
        <div class="headleft">
          <div class="chip">{html.escape(c["region"])} · {html.escape(c["mtype"])} · closes {c["deadline"]}{dchip}</div>
          <div class="q">{html.escape(c["question"])}</div>
        </div>
        <div class="headnums mono">fv <span class="acc">{c["fv_pct"]}%</span>
          <span class="dim">mid {c["market_pct"]}%</span>
          <span class="{gap_cls}">{c["gap_pp"]:+}pp</span>
          {'<span class="gflag">⚑</span>' if c["divergence_flag"] else ''}
          <span class="caret">▾</span></div>
      </div>
      <div class="cardbody">
        {flag}
        {_tract_html(c)}
        <div class="cardcols">
          <div class="colcharts">
            <div class="nums">
              <div class="gauge">{_svg_gauge(c["fv_pct"], c["band"][0], c["band"][1], c["market_pct"])}</div>
              <div class="num"><div class="lbl">epsilon fv</div><div class="val acc">{c["fv_pct"]}%</div>
                <div class="band">band {c["band"][0]}–{c["band"][1]}%</div></div>
              <div class="num"><div class="lbl">market</div><div class="val">{c["market_pct"]}%</div>
                <div class="band">mid (context)</div></div>
              <div class="num"><div class="lbl">gap</div><div class="val {gap_cls}">{c["gap_pp"]:+}pp</div>
                <div class="band">fv − mid</div></div>
            </div>
            <div class="chart">{_svg_series(c["series"], w=560)}</div>
          </div>
          <div class="colnews">
            {_quality_chip(c.get("evidence_quality"))}
            {_band_quality_html(c.get("evidence_quality"))}
            {_bias_html(c.get("bias"))}
            <div class="sub">cited drivers</div><ul>{drivers or "<li class='dim'>none today</li>"}</ul>
            <div class="sub">evidence</div><p class="note" style="margin-top:.1rem">{ev_note}</p>
          </div>
        </div>
        <div class="analytical">
          {_breakdown_html(c.get("breakdown"))}
          {_gdelt_html(c.get("gdelt"))}
          <div class="sub">ledger {html.escape(c["sf_id"] or "—")} · 24h vol ${c["volume24h"]:,} · {c["n_relevant"]} relevant articles/72h</div>
        </div>
      </div>
    </div>"""


def _sechead(num: str, title: str) -> str:
    return (f'<div class="sechead"><span class="secnum">{num}.</span>'
            f'<h2>{html.escape(title)}</h2></div>')


def render_html(sc: dict) -> str:
    # default-visible subset: every flagged market + the top gaps, capped at 8 —
    # the rest stay hidden behind "show all" / the picker (kills the long scroll)
    ordered = sc["markets"]
    default_visible = [c["slug"] for c in ordered if c["divergence_flag"]]
    for c in ordered:
        if len(default_visible) >= 8:
            break
        if c["slug"] not in default_visible:
            default_visible.append(c["slug"])
    cards = ""
    for c in ordered:
        cards += _card_html(c, expanded=bool(c["divergence_flag"]))
    picker = "".join(
        f'<label><input type="checkbox" data-slug="{html.escape(c["slug"])}" '
        f'onchange="pickChanged()"> {html.escape(c["question"][:70])}</label>'
        for c in ordered)

    bt_rows = "".join(
        f'<tr><td>{html.escape(b["label"])}</td><td class="mono">{b["n_pairs"]} pairs / {b["n_markets"]} mkts</td>'
        f'<td class="mono">{b["brier_agent"]:.3f}</td><td class="mono">{b["brier_market"]:.3f}</td>'
        f'<td class="verdict">{html.escape(b["verdict"])}</td></tr>'
        for b in sc["backtests"])

    ltr = sc["live_track_record"]
    if ltr.get("n_settled"):
        tr_rows = ""
        for r in ltr["records"]:
            mid = (f'{r["mid_pct_at_last_snapshot"]:g}%'
                   if r.get("mid_pct_at_last_snapshot") is not None else "—")
            stale = (f'{r["stale_days"]}d' if r.get("stale_days") is not None else "—")
            tr_rows += (
                f'<tr><td>{html.escape(r["question"][:78])}</td>'
                f'<td class="mono">{html.escape(r["resolution_date"])}</td>'
                f'<td class="mono">{r["final_p"] * 100:g}%</td>'
                f'<td class="mono">{"YES" if r["outcome"] else "NO"}</td>'
                f'<td class="mono">{r["brier"]:.3f}</td>'
                f'<td class="mono dim">{mid}</td>'
                f'<td class="mono dim">{stale}</td>'
                f'<td class="mono dim">{html.escape(r["method"])}</td></tr>')
        meth_bits = " · ".join(
            f'{html.escape(m)}: ' + (f'n={v["n"]}, Brier {v["brier"]:.4f}'
                                     if v["n"] else html.escape(v["note"]))
            for m, v in ltr["per_method"].items())
        ext = ltr.get("extended") or {}
        ext_bits = ""
        if ext.get("n"):
            ext_bits = (f' Log-loss <span class="mono">{ext["log_loss"]:.4f}</span> · '
                        f'ECE <span class="mono">{ext["ece"]:.4f}</span> · '
                        f"Spiegelhalter Z <span class=\"mono\">{ext['spiegelhalter']['z']:+.2f}</span> "
                        "(|Z| &gt; 1.96 would reject calibration-in-the-large).")
        stale_note = ""
        if ltr.get("max_stale_days"):
            stale_note = (
                f' <b>Read this before the Brier:</b> the observatory ran attended, not '
                f'daily — the scored numbers are the last ones published before each '
                f'market resolved, up to <span class="mono">{ltr["max_stale_days"]}</span> '
                f'days stale. That staleness is part of the score, not an excuse '
                f'removed from it.')
        track_html = f"""
      <div class="panel" style="margin-top:1.3rem">{_sechead("06", "Public track record — settled forecasts (first published 2026-08-24)")}
        <p class="note"><b>{ltr["n_settled"]} settled</b> · mean Brier
          <span class="mono">{ltr["brier"]:.4f}</span> · by method: {meth_bits}.{ext_bits}</p>
        <table><tr><th>question</th><th>resolved</th><th>our published p</th><th>outcome</th>
          <th>Brier</th><th>mid (context)</th><th>snapshot age</th><th>method</th></tr>{tr_rows}</table>
        <p class="note">{html.escape(ltr["note"])}{stale_note}</p>
      </div>"""
    else:
        track_html = f"""
      <div class="panel" style="margin-top:1.3rem">{_sechead("06", "Public track record — settled forecasts")}
        <p class="note">{html.escape(ltr["note"])}</p>
      </div>"""

    inter = sc["interim"]
    inter_html = ""
    if inter.get("overall_mean") is not None:
        inter_html = (f'<p class="note">Interim mean Brier <span class="mono">{inter["overall_mean"]:.4f}</span> '
                      f'over <span class="mono">{inter["n"]}</span> forecast-days. {html.escape(inter["note"])}</p>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Epsilon Observatory</title>
<style>
  * {{ box-sizing:border-box; margin:0; }}
  svg {{ max-width:100%; height:auto; }}
  body {{ background:{BG}; color:{TX}; font:15px/1.65 {SANS}; padding:1.6rem 1.2rem 3rem; }}
  .wrap {{ max-width:1360px; margin:0 auto; }}
  .masthead {{ display:flex; align-items:baseline; gap:1rem; border-bottom:1px solid {LINE};
               padding-bottom:.9rem; margin-bottom:1.6rem; flex-wrap:wrap; }}
  .brand {{ font-size:1.25rem; font-weight:600; letter-spacing:-0.02em; }}
  .mastnav {{ font-size:.85rem; color:{DIM}; letter-spacing:.02em; }}
  .masthead .toggle {{ margin-left:auto; }}
  h1 {{ font-weight:300; font-size:clamp(1.9rem,4vw,2.7rem); letter-spacing:-0.025em; line-height:1.15; }}
  .kicker {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.2em; color:{DIM}; margin-bottom:.4rem; }}
  .framing {{ color:{DIM}; margin:.9rem 0 1.6rem; max-width:74ch; }}
  .sechead {{ display:flex; align-items:baseline; gap:.6rem; margin:0 0 .9rem; }}
  .secnum {{ color:{DIM}; font-weight:300; font-size:1.25rem; }}
  .sechead h2 {{ font-size:1.35rem; font-weight:300; letter-spacing:-0.025em; }}
  .layout {{ display:grid; grid-template-columns:330px minmax(0,1fr); gap:1.4rem; align-items:start; }}
  .feedpane {{ position:sticky; top:1rem; max-height:calc(100vh - 2rem); overflow-y:auto;
               background:{PANEL}; border:1px solid {LINE}; border-radius:10px;
               padding:1.1rem 1.2rem; scrollbar-width:thin; }}
  .main {{ min-width:0; }}
  .feedpane {{ min-width:0; }}
  .scrollwrap {{ overflow-x:auto; }}
  /* RECONSTRUCTION: deliberately a different surface from the forward panels, so a
     reader skimming cannot mistake replayed numbers for published ones. */
  .panel.recon {{ background:transparent; border:1px dashed {LINE}; }}
  .panel.recon .sechead {{ opacity:.95; }}
  .reconchip {{ font:600 .62rem/1 {MONO}; letter-spacing:.09em; text-transform:uppercase;
                border:1px solid {LINE}; border-radius:99px; padding:.24rem .5rem;
                color:{DIM}; margin-right:.5rem; white-space:nowrap; vertical-align:middle; }}
  .reconlead {{ font-size:.92rem; color:{TX}; border-left:2px solid {LINE};
                padding-left:.85rem; margin:.2rem 0 1rem; }}
  .note.callout {{ border-left:2px solid {HI}; padding-left:.85rem; }}
  tr.rowhi td {{ background:rgba(204,92,68,0.10); }}
  .reachnote {{ border-top:1px solid {LINE}; margin-top:.9rem; padding-top:.7rem;
                font-size:.78rem; }}
  @media (max-width:820px) {{
    .layout {{ grid-template-columns:minmax(0,1fr); }}
    .feedpane {{ position:static; max-height:400px; order:2; }}
    .main {{ order:1; }}
  }}
  .panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:10px; padding:1.3rem 1.5rem; margin-bottom:1.4rem; }}
  table {{ width:100%; border-collapse:collapse; }}
  td,th {{ padding:.45rem .6rem; border-bottom:1px solid {LINE}; text-align:left; font-size:.9rem; }}
  th {{ font-size:.66rem; text-transform:uppercase; letter-spacing:.1em; color:{DIM}; font-weight:500; }}
  .mono {{ font-family:{MONO}; font-variant-numeric:tabular-nums; }}
  .verdict {{ color:{DIM}; text-transform:uppercase; font-size:.75rem; letter-spacing:.08em; }}
  .toggle {{ background:none; border:1px solid {LINE}; color:{TX}; border-radius:50px;
             padding:8px 28px; cursor:pointer; font:500 .75rem {SANS}; text-transform:uppercase;
             letter-spacing:.1em; transition:background 150ms ease,color 150ms ease; }}
  .toggle:hover, .toggle.on {{ background:{TX}; color:{BG}; border-color:{TX}; }}
  /* -------- overview donut grid -------- */
  .donutgrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:.9rem; }}
  .cell {{ display:block; background:{PANEL2}; border:1px solid {LINE}; border-radius:8px;
           padding:.8rem .8rem .9rem; text-decoration:none; color:{TX};
           transition:border-color 150ms ease; }}
  .cell:hover {{ border-color:rgba(209,202,183,0.45); }}
  .cell[data-flag="1"] {{ border-color:rgba(204,92,68,0.55); }}
  .celldonut svg {{ width:104px; height:104px; display:block; margin:0 auto .5rem; }}
  .cellq {{ font-size:.78rem; line-height:1.35; min-height:3.1em; color:{TX};
            display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }}
  .cellnums {{ font-size:.72rem; color:{DIM}; margin-top:.4rem; }}
  .cellnums b {{ color:{TX}; }}
  .cellband {{ font-size:.66rem; color:{DIM}; }}
  .qual {{ display:inline-block; font-family:{MONO}; font-size:.62rem; text-transform:uppercase;
           letter-spacing:.06em; border:1px solid {LINE}; border-radius:50px; padding:.14rem .55rem;
           margin-top:.45rem; color:{DIM}; }}
  .qual-strong {{ color:{TX}; border-color:rgba(209,202,183,0.45); }}
  .qual-thin {{ color:{HI}; border-color:rgba(204,92,68,0.5); }}
  /* -------- movers strip -------- */
  .movers {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:.8rem; }}
  .mover {{ background:{PANEL2}; border:1px solid {LINE}; border-radius:8px; padding:.7rem .9rem;
            text-decoration:none; color:{TX}; transition:border-color 150ms ease; }}
  .mover:hover {{ border-color:rgba(209,202,183,0.45); }}
  .mvdelta {{ font-size:1.15rem; font-weight:700; }}
  .mv-up {{ color:{TX}; }} .mv-dn {{ color:{HI}; }}
  .mvq {{ font-size:.76rem; color:{DIM}; line-height:1.35; margin:.25rem 0; }}
  .mvnow {{ font-size:.7rem; color:{DIM}; }}
  /* -------- cards -------- */
  .cards {{ display:grid; grid-template-columns:1fr; gap:1.1rem; }}
  .card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:10px; padding:1.1rem 1.4rem;
           transition:border-color 150ms ease; }}
  .card:hover {{ border-color:rgba(209,202,183,0.4); }}
  .card.hiddencard {{ display:none; }}
  .cardhead {{ display:flex; justify-content:space-between; gap:1rem; align-items:center;
               cursor:pointer; }}
  .headnums {{ white-space:nowrap; font-size:.95rem; display:flex; gap:.8rem; align-items:baseline; }}
  .headnums .acc {{ color:{TX}; font-weight:700; }}
  .caret {{ color:{DIM}; transition:transform 150ms ease; display:inline-block; }}
  .card.collapsed .caret {{ transform:rotate(-90deg); }}
  .card.collapsed .cardbody {{ display:none; }}
  .cardbody {{ margin-top:.9rem; }}
  .cardcols {{ display:grid; grid-template-columns:1.05fr 0.95fr; gap:1.5rem; align-items:start; }}
  @media (max-width:1100px) {{ .cardcols {{ grid-template-columns:1fr; }} }}
  @media (max-width:820px) {{ .headnums {{ display:none; }} }}
  .controls {{ display:flex; gap:.7rem; align-items:center; flex-wrap:wrap; margin:0 0 1rem; }}
  .controls button {{ background:none; border:1px solid {LINE}; color:{DIM}; border-radius:50px;
             padding:.35rem 1rem; cursor:pointer; font:500 .7rem {SANS}; text-transform:uppercase;
             letter-spacing:.09em; transition:background 150ms ease,color 150ms ease; }}
  .controls button:hover, .controls button.on {{ background:{TX}; color:{BG}; border-color:{TX}; }}
  .picker summary {{ cursor:pointer; color:{DIM}; font-size:.7rem; text-transform:uppercase;
                     letter-spacing:.09em; border:1px solid {LINE}; border-radius:50px;
                     padding:.35rem 1rem; list-style:none; }}
  .picker[open] summary {{ background:{PANEL2}; }}
  .picklist {{ position:absolute; z-index:5; background:{PANEL2}; border:1px solid {LINE};
               border-radius:8px; padding: .8rem 1rem; margin-top:.4rem; max-height:340px;
               overflow-y:auto; display:flex; flex-direction:column; gap:.25rem; }}
  .picklist label {{ font-size:.8rem; color:{TX}; cursor:pointer; }}
  .picker {{ position:relative; }}
  .tractnote {{ border:1px dashed {LINE}; border-radius:8px; color:{DIM}; font-size:.82rem;
                padding:.6rem .9rem; margin-bottom:.8rem; }}
  .dmark {{ color:{DIM}; }}
  .chip {{ font-size:.68rem; font-family:{MONO}; text-transform:uppercase; letter-spacing:.08em; color:{DIM}; }}
  .q {{ font-size:1.22rem; font-weight:300; letter-spacing:-0.015em; line-height:1.3; margin:.35rem 0 .1rem; }}
  .flag {{ display:inline-block; color:{HI}; border:1px solid rgba(204,92,68,.5); border-radius:50px;
           font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; padding:.2rem .7rem; margin-bottom:.7rem; }}
  .gflag {{ color:{HI}; }}
  .nums {{ display:flex; gap:1.1rem; margin-bottom:.9rem; align-items:center; flex-wrap:wrap; }}
  .num {{ flex:1; min-width:88px; }}
  .gauge {{ flex:0 0 130px; }} .gauge svg {{ width:130px; height:auto; display:block; }}
  .lbl {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.12em; color:{DIM}; }}
  .val {{ font-family:{MONO}; font-size:2rem; font-weight:700; font-variant-numeric:tabular-nums; }}
  .val.acc {{ color:{TX}; }} .pos {{ color:{TX}; }} .neg {{ color:{HI}; }}
  td.pos {{ color:{TX}; }} td.neg {{ color:{HI}; }}
  .band {{ font-size:.72rem; color:{DIM}; }}
  .chart svg {{ width:100%; height:auto; display:block; }}
  .chartkey {{ font-size:.68rem; color:{DIM}; display:flex; gap:1.1rem; margin-top:.3rem; flex-wrap:wrap; }}
  .nochart {{ color:{DIM}; font-size:.8rem; border:1px dashed {LINE}; border-radius:8px; padding:1rem; text-align:center; }}
  .analytical {{ display:none; margin-top:1rem; border-top:1px dashed {LINE}; padding-top:.8rem; }}
  body.analytical .analytical {{ display:block; }}
  .sub {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.12em; color:{DIM}; margin:.7rem 0 .3rem; }}
  ul {{ padding-left:1.1rem; font-size:.85rem; }}
  /* -------- left feed pane -------- */
  .ev {{ list-style:none; padding:0; }}
  .ev li {{ padding:.55rem 0; border-bottom:1px solid {LINE}; }}
  .ev li:last-child {{ border-bottom:none; }}
  .fmeta {{ display:flex; align-items:baseline; gap:.4rem; }}
  .ftitle {{ font-size:.84rem; line-height:1.4; margin:.15rem 0; }}
  .ftitle a {{ color:{TX}; text-decoration:none; }} .ftitle a:hover {{ color:{HI}; }}
  .frefs {{ display:flex; flex-wrap:wrap; gap:.35rem; }}
  .mref {{ font-size:.66rem; font-family:{MONO}; color:{DIM}; text-decoration:none;
           border:1px solid {LINE}; border-radius:50px; padding:.08rem .5rem; }}
  .mref:hover {{ color:{TX}; border-color:rgba(209,202,183,0.45); }}
  .fpriv {{ font-size:.76rem; }}
  .lean {{ display:inline-block; font-family:{MONO}; font-size:.6rem; border:1px solid {LINE};
           border-radius:3px; padding:0 .3rem; margin-left:.3rem; color:{DIM}; vertical-align:1px; }}
  .lean-left, .lean-right {{ color:{HI}; border-color:rgba(204,92,68,.45); }}
  .lean-lean-left, .lean-lean-right {{ color:{TX}; }}
  .dom {{ color:{TX}; font-size:.68rem; font-family:{MONO}; text-transform:uppercase; letter-spacing:.04em; }}
  .ts {{ color:{DIM}; font-size:.66rem; font-family:{MONO}; margin-left:auto; }}
  .dim {{ color:{DIM}; }}
  .bd td {{ font-size:.83rem; }} .bd .tot td {{ border-bottom:none; font-weight:600; }}
  .note {{ color:{DIM}; font-size:.82rem; margin-top:.6rem; max-width:80ch; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:1.3rem; }}
  @media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  .qlink {{ color:{TX}; text-decoration:none; }} .qlink:hover {{ color:{HI}; }}
  footer {{ color:{DIM}; font-size:.76rem; margin-top:2.2rem; max-width:96ch; line-height:1.7; }}
</style></head><body><div class="wrap">
  <div class="masthead">
    <span class="brand">εpsilon</span>
    <span class="mastnav">Research · Polymarket · Observatory</span>
    <button class="toggle" id="tg" onclick="document.body.classList.toggle('analytical');this.classList.toggle('on');this.textContent=document.body.classList.contains('analytical')?'analytical':'numbers only';">numbers only</button>
  </div>
  <div class="kicker">Public measurement loop</div>
  <h1>Observatory — our fair value, scored in public</h1>
  <p class="framing">{html.escape(sc["framing"])}</p>

  <div class="layout">
    <aside class="feedpane">{_feed_html(sc.get("feed", {}), sc.get("reach_status", ""))}</aside>
    <div class="main">

      <div class="panel">{_sechead("01", "All markets at a glance")}
        {_overview_grid_html(sc["markets"])}
        <p class="note">Ordered by disagreement (flags first, then |gap|). Ring = our
        fair value; faint segment = the band; grey tick = the Polymarket mid (context).
        Click any tile for the full time series, evidence and FV construction.</p>
      </div>

      <div class="panel">{_sechead("02", "Big movers — largest change since the previous published snapshot")}
        {_movers_html(sc.get("movers", {}))}
      </div>

      <div class="panel">{_sechead("03", "Where our model most disagrees (divergence layer)")}
        <div class="scrollwrap">{_svg_divergence(sc["markets"])}</div>
        <p class="note">{html.escape(sc["divergence_rule"])}</p>
      </div>

      {_sechead("04", "Market detail")}
      <div class="controls">
        <span class="mono dim" id="viscount"></span>
        <button id="btn-default" onclick="showDefault()">default view</button>
        <button id="btn-all" onclick="showAll()">show all</button>
        <button id="btn-flags" onclick="showFlags()">flags only</button>
        <details class="picker"><summary>choose markets</summary>
          <div class="picklist">{picker}</div>
        </details>
        <span class="note" style="margin:0">cards ordered flags-first, then |gap|; click a
        card header to expand/collapse; ◆ = not news-tractable (data-driven or poll-driven)</span>
      </div>
      <div class="cards" id="cards">{cards}</div>

      <div class="panel" style="margin-top:1.3rem">{_sechead("05", "Honest scoreboard — retrospective gates (Brier, lower is better)")}
        <table><tr><th>experiment</th><th>sample</th><th>agent</th><th>market</th><th>verdict</th></tr>{bt_rows}</table>
        <p class="note">These two pre-registered gates closed the claim that our number beats the market mid — permanently.
        What runs now is different: an independent fair value judged against resolved outcomes, with the mid as context.</p>
        {inter_html}
      </div>
{track_html}
{_simlive_html(sc.get("simlive"))}

      <div class="grid2">
        <div class="panel">{_sechead("08", "Reliability — model FV vs observed outcomes")}
          {_svg_reliability(sc["reliability"]["bins"])}
          <p class="note">{html.escape(sc["reliability"]["note"])} Dot size = number of forecast pairs in the bin; the dashed diagonal is perfect calibration.</p>
        </div>
        <div class="panel analytical">{_sechead("09", "Method")}
          <p style="font-size:.88rem">{html.escape(sc["method"])}</p>
        </div>
      </div>

      <footer>{html.escape(sc["attribution"])} · Generated {sc["generated_at"][:16]}Z ·
      Not investment advice; not a trading signal; a public measurement experiment. Reconstructed
      (pre-launch) segments are marked and never enter the scored ledger.</footer>
    </div>
  </div>
  <script>
    var DEFAULT_VISIBLE = {json.dumps(default_visible)};
    var ALL = Array.from(document.querySelectorAll('.card')).map(function(c) {{
      return c.getAttribute('data-slug'); }});
    function currentVisible() {{
      try {{
        var s = localStorage.getItem('obs_visible');
        if (s) return JSON.parse(s);
      }} catch (e) {{}}
      return DEFAULT_VISIBLE;
    }}
    function apply(vis, persist) {{
      document.querySelectorAll('.card').forEach(function(c) {{
        c.classList.toggle('hiddencard', vis.indexOf(c.getAttribute('data-slug')) < 0);
      }});
      document.querySelectorAll('.picklist input').forEach(function(i) {{
        i.checked = vis.indexOf(i.getAttribute('data-slug')) >= 0;
      }});
      document.getElementById('viscount').textContent =
        'showing ' + vis.length + ' of ' + ALL.length;
      if (persist) {{
        try {{ localStorage.setItem('obs_visible', JSON.stringify(vis)); }} catch (e) {{}}
      }}
    }}
    function showDefault() {{
      try {{ localStorage.removeItem('obs_visible'); }} catch (e) {{}}
      apply(DEFAULT_VISIBLE, false);
    }}
    function showAll() {{ apply(ALL, true); }}
    function showFlags() {{
      var f = Array.from(document.querySelectorAll('.card[data-flag="1"]')).map(function(c) {{
        return c.getAttribute('data-slug'); }});
      apply(f.length ? f : DEFAULT_VISIBLE, true);
    }}
    function pickChanged() {{
      var vis = Array.from(document.querySelectorAll('.picklist input'))
        .filter(function(i) {{ return i.checked; }})
        .map(function(i) {{ return i.getAttribute('data-slug'); }});
      apply(vis, true);
    }}
    function toggleCard(el) {{ el.classList.toggle('collapsed'); }}
    function revealCard(slug) {{
      var vis = currentVisible();
      if (vis.indexOf(slug) < 0) {{ vis = vis.concat([slug]); }}
      apply(vis, true);
      var el = document.getElementById('card-' + slug);
      if (el) {{ el.classList.remove('collapsed'); }}
    }}
    apply(currentVisible(), false);
  </script>
</div></body></html>"""


def publish(snapshots: list[dict], fv_series: dict | None = None) -> tuple[str, str]:
    SHOWCASE.mkdir(parents=True, exist_ok=True)
    sc = build_showcase(snapshots, fv_series or {})
    jpath = SHOWCASE / "showcase.json"
    hpath = SHOWCASE / "index.html"
    jpath.write_text(json.dumps(sc, indent=1))
    hpath.write_text(render_html(sc))
    return str(jpath), str(hpath)
