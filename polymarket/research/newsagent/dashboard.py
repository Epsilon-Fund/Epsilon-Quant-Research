"""Site-ready dashboard for the Epsilon Calibration Observatory (independent-FV v2).

Artifacts under data/newsagent/showcase/ (git-ignored, regenerable):
  showcase.json — everything the page renders (the website colleague lifts this
                  into Epsilon-Fund/epsilon-webs1te and restyles; UX ownership his)
  index.html    — fully self-contained page (inline CSS + inline SVG, no CDN, no
                  webfonts fetched — local font stacks with graceful fallbacks)

Style: READ-ONLY borrow of the epsilon-webs1te design language (dark #0a0a0a,
off-white #F0EFE9, single acid-lime #C8FF00 accent, serif display / sans body /
mono numerals, 24px-radius panels, uppercase micro-labels). Nothing is shipped
into that repo from here.

Framing rules (enforced in copy):
  - Epsilon's INDEPENDENT fair value, judged against RESOLVED OUTCOMES over time;
    the Polymarket mid is discovery + context, never the benchmark;
  - the retrospective gate scoreboard (market beat the agent) stays displayed —
    that closure is permanent and part of the story;
  - numbers-only default; ANALYTICAL toggle reveals drivers/evidence/method/breakdown;
  - divergence flags are informational ("where our model most disagrees"), never
    an edge claim;
  - IP-scrub: public data + our own forecasts only; attribution for Guardian/Wikipedia.
"""
from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone

from .config import (CSV_OUT, SHOWCASE, DIVERGENCE_GAP_PP, DIVERGENCE_HALF_MAX_PP,
                     DIVERGENCE_NREL_MIN)

# ---- design tokens (epsilon-webs1te, read-only borrow) ------------------------
BG = "#0a0a0a"; PANEL = "#111111"; PANEL2 = "#1c1c1c"; TX = "#F0EFE9"
DIM = "#777777"; LINE = "rgba(255,255,255,0.07)"; ACC = "#C8FF00"
NEG = "#d4645c"
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
        shown = [a for a in s["packet"]["articles"] if a.get("display", True)]
        n_private = len(s["packet"]["articles"]) - len(shown)
        cards.append({
            "slug": mkt["slug"], "question": mkt["question"], "region": s.get("region", ""),
            "mtype": sb.get("mtype", ""), "deadline": mkt["end_date"][:10],
            "tract": sb.get("tract", "news"), "tract_note": sb.get("tract_note", ""),
            "fv_pct": fc["p_pct"], "band": [fc["band_lo_pct"], fc["band_hi_pct"]],
            "market_pct": round(mkt["mid"] * 100, 1),
            "gap_pp": div.get("gap_pp", round(fc["p_pct"] - mkt["mid"] * 100, 1)),
            "divergence_flag": bool(div.get("flag")),
            "n_relevant": sb.get("n_relevant", 0),
            "volume24h": mkt["volume24h"], "liquidity": mkt["liquidity"],
            "drivers": s.get("drivers", [])[:3],
            "gdelt": sb.get("gdelt"),
            "breakdown": sb.get("breakdown"),
            "bias": sb.get("bias"),
            "evidence": [{"title": a["title"], "domain": a["domain"],
                          "seendate": a.get("seendate", ""), "url": a.get("url", "")}
                         for a in shown],
            "n_private_items": n_private,
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
                   "resolved outcomes only. A global-attention signal (GDELT GKG "
                   "article volume for the question's entities) can amplify a day's "
                   "directional evidence when world coverage bursts; it never sets "
                   "the direction. Every number decomposes into the article-level "
                   "contributions shown in analytical mode."),
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
        "live_track_record": {"status": "collecting", "note": (
            "Daily snapshots go to an append-only forecast ledger (anti-post-hoc: "
            "settled entries reject edits); Brier/reliability appear here as markets "
            "resolve.")},
        "markets": cards,
        "attribution": ("Headlines: The Guardian (Open Platform) and Wikipedia Current "
                        "events portal (CC BY-SA 4.0) — links go to the sources. Market "
                        "data: Polymarket public APIs."),
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
<polygon points="{band}" fill="{ACC}" opacity="0.07"/>
{f'<path d="{fv_back}" fill="none" stroke="{ACC}" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.75"/>' if fv_back else ''}
{f'<path d="{fv_live}" fill="none" stroke="{ACC}" stroke-width="2"/>' if fv_live else ''}
{f'<path d="{mid}" fill="none" stroke="{DIM}" stroke-width="1.4"/>' if mid else ''}
<text x="34" y="{h-2}" fill="{DIM}" font-size="9" font-family="{MONO}">{d0}</text>
<text x="{w-60}" y="{h-2}" fill="{DIM}" font-size="9" font-family="{MONO}">{d1}</text>
</svg>
<div class="chartkey"><span style="color:{ACC}">— FV + band</span>
<span style="color:{ACC};opacity:.7">- - reconstructed (pre-launch backfill, unscored)</span>
<span style="color:{DIM}">— Polymarket mid (context)</span></div>"""


def _svg_reliability(bins: list[dict], w: int = 340, h: int = 300) -> str:
    if not bins:
        return f'<div class="nochart">reliability chart appears once calibration pairs exist</div>'
    pad = 38
    def xx(v): return pad + (w - pad - 12) * v
    def yy(v): return (h - pad) - (h - pad - 12) * v
    pts = "".join(
        f'<circle cx="{xx(b["mean_fv"]):.1f}" cy="{yy(b["outcome_rate"]):.1f}" '
        f'r="{min(14, 4 + b["n"] * 0.45):.1f}" fill="{ACC}" opacity="0.75"/>'
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


def _svg_gauge(fv: float, lo: float, hi: float, mid: float,
               w: int = 150, h: int = 92) -> str:
    """Per-market gauge — the % as a visual: 0-100 arc, band segment, FV needle
    (accent), mid tick (grey, context). Complements the numerals; the time series
    below stays the trend view (no duplication of either)."""
    import math as _m
    cx, cy, r = w / 2, h - 10, 58

    def pt(pct: float, rad: float) -> tuple[float, float]:
        a = _m.pi * (1 - pct / 100.0)
        return cx + rad * _m.cos(a), cy - rad * _m.sin(a)

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


def _svg_sparkline(series: list[dict], w: int = 120, h: int = 26) -> str:
    """Grid-row FV trend preview (accent line, endpoint dot). The full FV+band-vs-mid
    time series in the market card is the detail view — this is deliberately tiny."""
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


def _overview_grid_html(cards: list[dict]) -> str:
    """All markets at a glance, ordered by the divergence layer (flags first, then
    |gap|): FV + mid + gap + band + sparkline preview per row. The question links
    to the market card (revealing it if hidden); ◆ marks data-driven markets
    (news-FV structurally blind — see the card note)."""
    rows = ""
    for c in sorted(cards, key=lambda x: (not x["divergence_flag"], -abs(x["gap_pp"]))):
        q = c["question"][:64] + ("…" if len(c["question"]) > 64 else "")
        gap_cls = "pos" if c["gap_pp"] > 0 else "neg"
        flag = ' <span class="gflag">⚑</span>' if c["divergence_flag"] else ""
        dmark = ' <span class="dmark" title="data-driven — not news-tractable">◆</span>' \
            if c.get("tract") == "data" else ""
        rows += f"""<tr>
<td class="gq"><a class="qlink" href="#card-{html.escape(c["slug"])}"
 onclick="revealCard('{html.escape(c["slug"])}')">{html.escape(q)}</a>{flag}{dmark}</td>
<td class="mono acc">{c["fv_pct"]}%</td>
<td class="mono">{c["market_pct"]}%</td>
<td class="mono {gap_cls}">{c["gap_pp"]:+}</td>
<td class="mono dim">{c["band"][0]}–{c["band"][1]}</td>
<td class="spark">{_svg_sparkline(c.get("series", []))}</td></tr>"""
    return f"""<table class="grid">
<tr><th>market</th><th>fv</th><th>mid</th><th>gap pp</th><th>band %</th><th>trend</th></tr>
{rows}</table>"""


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
        color = ACC if c["divergence_flag"] else (DIM if abs(g) < DIVERGENCE_GAP_PP else TX)
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
    """Breakdown row -> (public domain label, public title). Private newsletter
    items show their generic source label only — never title/text/link."""
    dom = a.get("domain", "")
    if dom.startswith("newsletter:"):
        label = dom.split(":", 1)[1].strip()
        return f"{label} (newsletter)", "private analysis item — not displayed"
    return dom, a.get("title", "")


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
    """Ratings explainer: how the number formed from sources of differing bias."""
    if not bias:
        return ""
    tier_rows = ""
    for lean, tiers in (("YES", bias.get("yes_tiers", {})), ("NO", bias.get("no_tiers", {}))):
        for tier, srcs in tiers.items():
            names = ", ".join(f"{html.escape(s)} ×{k}" if k > 1 else html.escape(s)
                              for s, k in srcs)
            tier_rows += (f'<tr><td class="mono">{lean}</td>'
                          f'<td class="dim">{html.escape(tier)}</td><td>{names}</td></tr>')
    table = (f'<table class="bd"><tr><th>lean</th><th>reliability tier</th><th>sources</th></tr>'
             f'{tier_rows}</table>') if tier_rows else ""
    return (f'<div class="sub">how the number formed (source lean × reliability)</div>'
            f'<p class="note" style="margin-top:.1rem">{html.escape(bias["sentence"])} '
            f'Weights per the published Scheme-A table (Wikipedia RSP tiers + Iffy '
            f'blocklist).</p>{table}')


def _tract_html(c: dict) -> str:
    if c.get("tract") != "data":
        return ""
    return (f'<div class="tractnote">◆ not news-tractable — our news-FV is structurally '
            f'blind here. {html.escape(c.get("tract_note", ""))} Scored in public anyway; '
            f'expect the market to carry information our packet cannot see.</div>')


def _card_html(c: dict, expanded: bool) -> str:
    drivers = "".join(f"<li>{html.escape(d)}</li>" for d in c["drivers"])
    ev = "".join(
        f'<li><span class="dom">{html.escape(e["domain"])}</span> '
        + (f'<a href="{html.escape(e["url"])}" target="_blank" rel="noopener">' if e.get("url") else "")
        + html.escape(e["title"]) + ("</a>" if e.get("url") else "")
        + f' <span class="ts">{html.escape(e.get("seendate", "")[:8])}</span></li>'
        for e in c["evidence"][:8])
    if c.get("n_private_items"):
        ev += (f'<li class="dim">+ {c["n_private_items"]} private analysis item'
               f'{"s" if c["n_private_items"] != 1 else ""} (newsletters — used '
               'internally, never displayed)</li>')
    gap_cls = "pos" if c["gap_pp"] > 0 else "neg"
    flag = ('<span class="flag">⚑ divergence — high-confidence disagreement</span>'
            if c["divergence_flag"] else "")
    dchip = ' · <span class="dmark">◆ data-driven</span>' if c.get("tract") == "data" else ""
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
            {_bias_html(c.get("bias"))}
            <div class="sub">cited drivers</div><ul>{drivers or "<li class='dim'>none today</li>"}</ul>
            <div class="sub">evidence feed (what the extractor read — headline + link only)</div><ul class="ev">{ev}</ul>
          </div>
        </div>
        <div class="analytical">
          {_breakdown_html(c.get("breakdown"))}
          {_gdelt_html(c.get("gdelt"))}
          <div class="sub">ledger {html.escape(c["sf_id"] or "—")} · 24h vol ${c["volume24h"]:,} · {c["n_relevant"]} relevant articles/72h</div>
        </div>
      </div>
    </div>"""


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
  body {{ background:{BG}; color:{TX}; font:15px/1.65 {SANS}; padding:2.2rem 1.2rem 3rem; }}
  .wrap {{ max-width:1120px; margin:0 auto; }}
  h1 {{ font-family:{SERIF}; font-weight:400; font-size:clamp(2rem,4.5vw,3rem); letter-spacing:-0.02em; }}
  .kicker {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; color:{DIM}; margin-bottom:.4rem; }}
  .framing {{ color:{DIM}; margin:.9rem 0 1.8rem; max-width:74ch; }}
  .panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:24px; padding:1.4rem 1.6rem; margin-bottom:1.3rem; }}
  .panel h2 {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; color:{DIM}; margin-bottom:.9rem; font-weight:500; }}
  table {{ width:100%; border-collapse:collapse; }}
  td,th {{ padding:.45rem .6rem; border-bottom:1px solid {LINE}; text-align:left; font-size:.9rem; }}
  .mono {{ font-family:{MONO}; font-variant-numeric:tabular-nums; }}
  .verdict {{ color:{DIM}; text-transform:uppercase; font-size:.75rem; letter-spacing:.08em; }}
  .toggle {{ float:right; background:none; border:1px solid {LINE}; color:{DIM}; border-radius:999px;
             padding:.5rem 1.6rem; cursor:pointer; font:.72rem {SANS}; text-transform:uppercase;
             letter-spacing:.1em; transition:all 150ms ease; }}
  .toggle:hover, .toggle.on {{ background:{ACC}; color:{BG}; border-color:{ACC}; }}
  .cards {{ display:grid; grid-template-columns:1fr; gap:1.1rem; }}
  .card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:24px; padding:1.1rem 1.4rem;
           transition:border-color 150ms ease; }}
  .card:hover {{ border-color:rgba(200,255,0,0.35); }}
  .card.hiddencard {{ display:none; }}
  .cardhead {{ display:flex; justify-content:space-between; gap:1rem; align-items:center;
               cursor:pointer; }}
  .headnums {{ white-space:nowrap; font-size:.95rem; display:flex; gap:.8rem; align-items:baseline; }}
  .headnums .acc {{ color:{ACC}; font-weight:700; }}
  .caret {{ color:{DIM}; transition:transform 150ms ease; display:inline-block; }}
  .card.collapsed .caret {{ transform:rotate(-90deg); }}
  .card.collapsed .cardbody {{ display:none; }}
  .cardbody {{ margin-top:.9rem; }}
  .cardcols {{ display:grid; grid-template-columns:1.05fr 0.95fr; gap:1.5rem; align-items:start; }}
  @media (max-width:820px) {{ .cardcols {{ grid-template-columns:1fr; }}
    .headnums {{ display:none; }} }}
  .controls {{ display:flex; gap:.7rem; align-items:center; flex-wrap:wrap; margin:0 0 1rem; }}
  .controls button {{ background:none; border:1px solid {LINE}; color:{DIM}; border-radius:999px;
             padding:.35rem 1rem; cursor:pointer; font:.7rem {SANS}; text-transform:uppercase;
             letter-spacing:.09em; transition:all 150ms ease; }}
  .controls button:hover, .controls button.on {{ background:{ACC}; color:{BG}; border-color:{ACC}; }}
  .picker summary {{ cursor:pointer; color:{DIM}; font-size:.7rem; text-transform:uppercase;
                     letter-spacing:.09em; border:1px solid {LINE}; border-radius:999px;
                     padding:.35rem 1rem; list-style:none; }}
  .picker[open] summary {{ background:{PANEL2}; }}
  .picklist {{ position:absolute; z-index:5; background:{PANEL2}; border:1px solid {LINE};
               border-radius:14px; padding: .8rem 1rem; margin-top:.4rem; max-height:340px;
               overflow-y:auto; display:flex; flex-direction:column; gap:.25rem; }}
  .picklist label {{ font-size:.8rem; color:{TX}; cursor:pointer; }}
  .picker {{ position:relative; }}
  .tractnote {{ border:1px dashed {LINE}; border-radius:12px; color:{DIM}; font-size:.82rem;
                padding:.6rem .9rem; margin-bottom:.8rem; }}
  .dmark {{ color:{DIM}; }}
  .qlink {{ color:{TX}; text-decoration:none; }} .qlink:hover {{ color:{ACC}; }}
  .chip {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; color:{DIM}; }}
  .q {{ font-family:{SERIF}; font-size:1.3rem; line-height:1.25; margin:.35rem 0 .1rem; }}
  .flag {{ display:inline-block; color:{ACC}; border:1px solid rgba(200,255,0,.4); border-radius:999px;
           font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; padding:.2rem .7rem; margin-bottom:.7rem; }}
  .nums {{ display:flex; gap:1.1rem; margin-bottom:.9rem; align-items:center; flex-wrap:wrap; }}
  .num {{ flex:1; min-width:88px; }}
  .gauge {{ flex:0 0 130px; }} .gauge svg {{ width:130px; height:auto; display:block; }}
  table.grid td, table.grid th {{ padding:.32rem .55rem; font-size:.85rem; white-space:nowrap; }}
  table.grid th {{ font-size:.66rem; text-transform:uppercase; letter-spacing:.1em; color:{DIM}; font-weight:500; }}
  .gq {{ white-space:normal !important; min-width:220px; }}
  .gridwrap {{ overflow-x:auto; }}
  .gflag {{ color:{ACC}; }}
  td.acc {{ color:{ACC}; font-weight:600; }}
  .spark svg {{ display:block; width:120px; height:26px; }}
  .lbl {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; color:{DIM}; }}
  .val {{ font-family:{MONO}; font-size:2rem; font-weight:700; font-variant-numeric:tabular-nums; }}
  .val.acc {{ color:{ACC}; }} .pos {{ color:{TX}; }} .neg {{ color:{DIM}; }}
  td.pos {{ color:{ACC}; }} td.neg {{ color:{NEG}; }}
  .band {{ font-size:.72rem; color:{DIM}; }}
  .chart svg {{ width:100%; height:auto; display:block; }}
  .chartkey {{ font-size:.68rem; color:{DIM}; display:flex; gap:1.1rem; margin-top:.3rem; flex-wrap:wrap; }}
  .nochart {{ color:{DIM}; font-size:.8rem; border:1px dashed {LINE}; border-radius:8px; padding:1rem; text-align:center; }}
  .analytical {{ display:none; margin-top:1rem; border-top:1px dashed {LINE}; padding-top:.8rem; }}
  body.analytical .analytical {{ display:block; }}
  .sub {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; color:{DIM}; margin:.7rem 0 .3rem; }}
  ul {{ padding-left:1.1rem; font-size:.85rem; }}
  .ev li {{ color:{DIM}; margin-bottom:.15rem; }} .ev a {{ color:{TX}; text-decoration:none; }}
  .ev a:hover {{ color:{ACC}; }}
  .dom {{ color:{ACC}; font-size:.7rem; margin-right:.3rem; font-family:{MONO}; }}
  .ts {{ color:{DIM}; font-size:.7rem; font-family:{MONO}; }}
  .dim {{ color:{DIM}; }}
  .bd td {{ font-size:.83rem; }} .bd .tot td {{ border-bottom:none; font-weight:600; }}
  .note {{ color:{DIM}; font-size:.82rem; margin-top:.6rem; max-width:80ch; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:1.3rem; }}
  @media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  footer {{ color:{DIM}; font-size:.76rem; margin-top:2.2rem; max-width:86ch; line-height:1.7; }}
</style></head><body><div class="wrap">
  <button class="toggle" id="tg" onclick="document.body.classList.toggle('analytical');this.classList.toggle('on');this.textContent=document.body.classList.contains('analytical')?'analytical':'numbers only';">numbers only</button>
  <div class="kicker">Epsilon Research · public measurement loop</div>
  <h1>Observatory — our fair value, scored in public</h1>
  <p class="framing">{html.escape(sc["framing"])}</p>

  <div class="panel"><h2>All markets at a glance</h2>
    <div class="gridwrap">{_overview_grid_html(sc["markets"])}</div>
    <p class="note">Ordered by disagreement (flags first, then |gap|). The trend column
    previews our FV path; each market's card below carries the full FV + band vs mid
    time series, the gauge, and the evidence.</p>
  </div>

  <div class="panel"><h2>Where our model most disagrees (divergence layer)</h2>
    {_svg_divergence(sc["markets"])}
    <p class="note">{html.escape(sc["divergence_rule"])}</p>
  </div>

  <div class="controls">
    <span class="mono dim" id="viscount"></span>
    <button id="btn-default" onclick="showDefault()">default view</button>
    <button id="btn-all" onclick="showAll()">show all</button>
    <button id="btn-flags" onclick="showFlags()">flags only</button>
    <details class="picker"><summary>choose markets</summary>
      <div class="picklist">{picker}</div>
    </details>
    <span class="note" style="margin:0">cards ordered flags-first, then |gap|; click a
    card header to expand/collapse; ◆ = data-driven (not news-tractable)</span>
  </div>
  <div class="cards" id="cards">{cards}</div>
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

  <div class="panel" style="margin-top:1.3rem"><h2>Honest scoreboard — retrospective gates (Brier, lower is better)</h2>
    <table><tr><th>experiment</th><th>sample</th><th>agent</th><th>market</th><th>verdict</th></tr>{bt_rows}</table>
    <p class="note">These two pre-registered gates closed the claim that our number beats the market mid — permanently.
    What runs now is different: an independent fair value judged against resolved outcomes, with the mid as context.</p>
    {inter_html}
    <p class="note">{html.escape(sc["live_track_record"]["note"])}</p>
  </div>

  <div class="grid2">
    <div class="panel"><h2>Reliability — model FV vs observed outcomes</h2>
      {_svg_reliability(sc["reliability"]["bins"])}
      <p class="note">{html.escape(sc["reliability"]["note"])} Dot size = number of forecast pairs in the bin; the dashed diagonal is perfect calibration.</p>
    </div>
    <div class="panel analytical"><h2>Method</h2>
      <p style="font-size:.88rem">{html.escape(sc["method"])}</p>
    </div>
  </div>

  <footer>{html.escape(sc["attribution"])} · Generated {sc["generated_at"][:16]}Z ·
  Not investment advice; not a trading signal; a public measurement experiment. Reconstructed
  (pre-launch) segments are marked and never enter the scored ledger.</footer>
</div></body></html>"""


def publish(snapshots: list[dict], fv_series: dict | None = None) -> tuple[str, str]:
    SHOWCASE.mkdir(parents=True, exist_ok=True)
    sc = build_showcase(snapshots, fv_series or {})
    jpath = SHOWCASE / "showcase.json"
    hpath = SHOWCASE / "index.html"
    jpath.write_text(json.dumps(sc, indent=1))
    hpath.write_text(render_html(sc))
    return str(jpath), str(hpath)
