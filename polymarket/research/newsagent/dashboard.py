"""Site-ready local dashboard generator for the Epsilon Calibration Observatory.

Produces two artifacts under data/newsagent/showcase/ (git-ignored, regenerable):
  showcase.json — everything the page renders (a colleague can lift this straight
                  into Epsilon-Fund/epsilon-webs1te and restyle; UX ownership is his)
  index.html    — fully self-contained page (inline CSS/JS, no CDN, offline-renderable)

Framing rules (IP-scrub + honesty, enforced here):
  - masthead declares the EXPERIMENT framing; the scoreboard shows the market
    beating the agent on the retrospective gates — that is the content, not a bug;
  - numbers-only default; ANALYTICAL toggle reveals drivers/evidence/method;
  - public data + our own forecasts only — no wallet data, no strategy thresholds;
  - attribution: Guardian (headline+link), Wikipedia Current Events (CC BY-SA).
"""
from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone

from .config import CSV_OUT, SHOWCASE


def _read_csv(name: str) -> list[dict]:
    p = CSV_OUT / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def build_showcase(snapshots: list[dict]) -> dict:
    """snapshots: [{market, packet, forecast, drivers, decisive_evidence, sf_id}]"""
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
        mkt, fc = s["market"], s["forecast"]
        cards.append({
            "slug": mkt["slug"], "question": mkt["question"], "region": s.get("region", ""),
            "deadline": mkt["end_date"][:10],
            "agent_pct": fc["p_pct"], "band": [fc["band_lo_pct"], fc["band_hi_pct"]],
            "market_pct": round(mkt["mid"] * 100, 1),
            "gap_pp": round(fc["p_pct"] - mkt["mid"] * 100, 1),
            "volume24h": mkt["volume24h"], "liquidity": mkt["liquidity"],
            "drivers": s.get("drivers", [])[:3],
            "decisive_evidence": bool(s.get("decisive_evidence")),
            "evidence": [{"title": a["title"], "domain": a["domain"],
                          "seendate": a.get("seendate", ""), "url": a.get("url", "")}
                         for a in s["packet"]["articles"]],
            "sf_id": s.get("sf_id", ""),
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framing": ("Epsilon Calibration Observatory — a public forecasting experiment. "
                    "We publish an LLM news-agent's independent probability next to the "
                    "Polymarket mid and score both with proper scoring rules as markets "
                    "resolve. This is NOT a claim that our number is better: in two "
                    "pre-registered retrospective gates the market beat the agent. "
                    "The craft on display is the measurement discipline."),
        "backtests": backtests,
        "live_track_record": {"status": "collecting", "note": (
            "Live snapshots are logged to an append-only forecast ledger; Brier scores "
            "appear here as markets resolve. Unresolved forecasts are interim-read "
            "against the prior-day market mid (ForecastBench convention).")},
        "markets": cards,
        "method": ("Per market: curated news headlines (Guardian Open Platform + Wikipedia "
                   "Current Events; timestamped, market price never shown to the model) -> "
                   "five perspective-diverse LLM estimates (base-rate, evidence-forward, "
                   "skeptic, reference-class, adversarial) -> trimmed mean; 80% band from "
                   "ensemble spread, floored at +/-8pp. Daily snapshots; append-only ledger; "
                   "anti-post-hoc enforced by the ledger state machine."),
        "attribution": ("Headlines: The Guardian (Open Platform) and Wikipedia Current "
                        "events portal (CC BY-SA 4.0) — links go to the sources. Market "
                        "data: Polymarket public APIs."),
    }


def render_html(sc: dict) -> str:
    cards = ""
    for c in sc["markets"]:
        drivers = "".join(f"<li>{html.escape(d)}</li>" for d in c["drivers"])
        ev = "".join(
            f'<li><span class="dom">{html.escape(e["domain"])}</span> '
            + (f'<a href="{html.escape(e["url"])}" target="_blank" rel="noopener">' if e.get("url") else "")
            + html.escape(e["title"]) + ("</a>" if e.get("url") else "") + "</li>"
            for e in c["evidence"][:8])
        gap_cls = "pos" if c["gap_pp"] > 0 else "neg"
        cards += f"""
    <div class="card">
      <div class="q">{html.escape(c["question"])}</div>
      <div class="meta">{html.escape(c["region"])} · deadline {c["deadline"]} · 24h vol ${c["volume24h"]:,}</div>
      <div class="nums">
        <div class="num"><div class="lbl">agent</div><div class="val">{c["agent_pct"]}%</div>
          <div class="band">80% band {c["band"][0]}–{c["band"][1]}%</div></div>
        <div class="num"><div class="lbl">market</div><div class="val">{c["market_pct"]}%</div>
          <div class="band">Polymarket mid</div></div>
        <div class="num"><div class="lbl">gap</div><div class="val {gap_cls}">{c["gap_pp"]:+}pp</div>
          <div class="band">agent − market</div></div>
      </div>
      <div class="analytical">
        <div class="sub">why (agent's cited drivers)</div><ul>{drivers}</ul>
        <div class="sub">evidence packet (what the agent saw)</div><ul class="ev">{ev}</ul>
        <div class="sub">ledger id: {html.escape(c["sf_id"] or "—")}</div>
      </div>
    </div>"""

    bt_rows = "".join(
        f'<tr><td>{html.escape(b["label"])}</td><td>{b["n_pairs"]} pairs / {b["n_markets"]} mkts</td>'
        f'<td>{b["brier_agent"]:.3f}</td><td>{b["brier_market"]:.3f}</td>'
        f'<td class="verdict">{html.escape(b["verdict"])}</td></tr>'
        for b in sc["backtests"])

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Epsilon Calibration Observatory</title>
<style>
  :root {{ --bg:#0e1117; --panel:#171c26; --line:#2a3140; --tx:#dbe2ee; --dim:#8b95a7;
           --acc:#39c48f; --warn:#e0a437; --neg:#d4645c; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--bg); color:var(--tx); font:15px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif; padding:2rem 1rem; }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  h1 {{ font-size:1.5rem; letter-spacing:.02em; }}
  .framing {{ color:var(--dim); margin:.8rem 0 1.6rem; max-width:70ch; }}
  .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:1.1rem 1.3rem; margin-bottom:1.2rem; }}
  .panel h2 {{ font-size:.95rem; text-transform:uppercase; letter-spacing:.08em; color:var(--dim); margin-bottom:.7rem; }}
  table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
  td,th {{ padding:.4rem .6rem; border-bottom:1px solid var(--line); text-align:left; }}
  .verdict {{ color:var(--warn); }}
  .toggle {{ float:right; background:none; border:1px solid var(--line); color:var(--dim);
             border-radius:6px; padding:.35rem .8rem; cursor:pointer; }}
  .toggle.on {{ color:var(--acc); border-color:var(--acc); }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:1rem; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:1rem 1.1rem; }}
  .q {{ font-weight:600; min-height:3em; }}
  .meta {{ color:var(--dim); font-size:.8rem; margin:.3rem 0 .8rem; }}
  .nums {{ display:flex; gap:1rem; }}
  .num {{ flex:1; }}
  .lbl {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:var(--dim); }}
  .val {{ font-size:1.6rem; font-weight:700; font-variant-numeric:tabular-nums; }}
  .val.pos {{ color:var(--acc); }} .val.neg {{ color:var(--neg); }}
  .band {{ font-size:.72rem; color:var(--dim); }}
  .analytical {{ display:none; margin-top:.9rem; border-top:1px dashed var(--line); padding-top:.7rem; }}
  body.analytical .analytical {{ display:block; }}
  .sub {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); margin:.5rem 0 .2rem; }}
  ul {{ padding-left:1.1rem; font-size:.85rem; }}
  .ev li {{ color:var(--dim); }} .ev a {{ color:var(--tx); text-decoration:none; }}
  .dom {{ color:var(--acc); font-size:.75rem; margin-right:.3rem; }}
  footer {{ color:var(--dim); font-size:.78rem; margin-top:2rem; max-width:80ch; }}
</style></head><body><div class="wrap">
  <button class="toggle" id="tg" onclick="document.body.classList.toggle('analytical');this.classList.toggle('on');this.textContent=document.body.classList.contains('analytical')?'analytical mode':'numbers only';">numbers only</button>
  <h1>Epsilon Calibration Observatory</h1>
  <p class="framing">{html.escape(sc["framing"])}</p>
  <div class="panel"><h2>Scoreboard — retrospective gates (Brier, lower is better)</h2>
    <table><tr><th>experiment</th><th>sample</th><th>agent</th><th>market</th><th>verdict</th></tr>{bt_rows}</table>
    <p class="band" style="margin-top:.5rem">{html.escape(sc["live_track_record"]["note"])}</p>
  </div>
  <div class="cards">{cards}</div>
  <div class="panel analytical"><h2>Method</h2><p style="font-size:.88rem">{html.escape(sc["method"])}</p></div>
  <footer>{html.escape(sc["attribution"])} · Generated {sc["generated_at"][:16]}Z · Not investment advice; not a trading signal; a public measurement experiment.</footer>
</div></body></html>"""


def publish(snapshots: list[dict]) -> tuple[str, str]:
    SHOWCASE.mkdir(parents=True, exist_ok=True)
    sc = build_showcase(snapshots)
    jpath = SHOWCASE / "showcase.json"
    hpath = SHOWCASE / "index.html"
    jpath.write_text(json.dumps(sc, indent=1))
    hpath.write_text(render_html(sc))
    return str(jpath), str(hpath)
