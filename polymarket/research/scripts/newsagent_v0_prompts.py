"""News-agent v0 gate: render one self-contained forecast prompt file per (market, snapshot).

Pre-registration: notes/news_agent/newsagent_v0_gate_preregistration.md
The prompt NEVER contains: the PM mid, price history, the outcome, or anything dated
after the snapshot. The forecaster is instructed to use only the packet + base rates.

Outputs: data/newsagent/v0/prompts/<slug>__<date>.txt  (+ 3 empty-packet canaries)
         data/newsagent/v0/prompt_manifest.json
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "newsagent" / "v0"
NEWS = RAW / "news"
PROMPTS = RAW / "prompts"

SNAPSHOT_DATES = ["2026-06-08", "2026-06-11", "2026-06-14", "2026-06-17",
                  "2026-06-20", "2026-06-23", "2026-06-26", "2026-06-29"]

TEMPLATE = """You are a calibrated superforecaster producing an INDEPENDENT fair-value probability for a prediction-market question. Today is {date}. Rely ONLY on the evidence packet below plus general knowledge of how the world works. Your training data predates these events — do NOT assume you know anything that happened after {date}. You have no market price and must not guess one; produce your own view.

QUESTION: {question}

RESOLUTION CRITERIA: {description}

DEADLINE: {deadline} ({days_left} days from today)

EVIDENCE PACKET (news headlines retrieved as of today, most relevant first):
{packet}

Method (reason step by step privately, then output only JSON):
1. Outside view first: base rate for this class of event resolving YES within {days_left} days.
2. Status-quo weighting: the world changes slowly; absent decisive evidence, weight the status quo.
3. Adjust on the packet: what do these headlines actually establish as of {date}? Distinguish speculation from confirmed fact.
4. Give an 80% credible interval on the fair probability — honestly wide if evidence is thin.

Reply with a single JSON object only, no prose:
{{"p_pct": <0-100>, "band_lo_pct": <0-100>, "band_hi_pct": <0-100>, "drivers": ["<up to 3 short strings citing packet items or base-rate logic>"]}}
"""

CANARIES = [  # (slug, date) pairs re-rendered with an empty packet — contamination check
    ("aleksandar-vui-out-as-serbian-president-by-june-30-2026-398", "2026-06-14"),
    ("starmer-out-by-june-30-2026-862-594-548-219-739-726-569-741-645", "2026-06-17"),
    ("us-x-iran-permanent-peace-deal-by-june-15-2026-734-856-129", "2026-06-11"),
]


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
    return urllib.request.urlopen(req, timeout=30).read()


def full_description(slug: str) -> tuple[str, str]:
    url = ("https://gamma-api.polymarket.com/markets?"
           + urllib.parse.urlencode({"slug": slug, "closed": "true"}))
    rows = json.loads(http_get(url))
    m = rows[0]
    return (m.get("description") or "").strip(), m.get("endDate", "")


def render(question: str, description: str, deadline: str, date: str, packet_articles: list[dict]) -> str:
    # Lookahead enforcement (GDELT end-bound is sloppy): drop articles seen after the snapshot.
    cutoff = f"{date.replace('-', '')}T120000Z"
    packet_articles = [a for a in packet_articles if (a.get("seendate") or "9") <= cutoff]
    days_left = max(0, (datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                        - datetime.fromisoformat(f"{date}T12:00:00+00:00")).days)
    if packet_articles:
        packet = "\n".join(f"- [{a.get('seendate','?')}] {a.get('domain','?')}: {a['title']}"
                           for a in packet_articles)
    else:
        packet = "(no articles retrieved for this date)"
    return TEMPLATE.format(date=date, question=question, description=description,
                           deadline=deadline[:10], days_left=days_left, packet=packet)


def main() -> None:
    PROMPTS.mkdir(parents=True, exist_ok=True)
    selected = json.loads((RAW / "universe_selected.json").read_text())
    manifest = []
    descs: dict[str, tuple[str, str]] = {}
    for m in selected:
        descs[m["slug"]] = full_description(m["slug"])
        time.sleep(0.3)

    for m in selected:
        desc, deadline = descs[m["slug"]]
        for d in SNAPSHOT_DATES:
            if m.get(f"mid_{d}", "") == "":
                continue
            pf = NEWS / f"{m['slug']}__{d}.json"
            if not pf.exists():
                print(f"  missing packet: {pf.name} — run newsagent_v0_news.py first")
                continue
            packet = json.loads(pf.read_text())
            out = PROMPTS / f"{m['slug']}__{d}.txt"
            out.write_text(render(m["question"], desc, deadline, d, packet["articles"]))
            manifest.append({"slug": m["slug"], "date": d, "prompt_file": str(out),
                             "n_articles": len(packet["articles"]), "kind": "scored"})

    for slug, d in CANARIES:
        if slug not in descs:
            continue
        desc, deadline = descs[slug]
        q = next(m["question"] for m in selected if m["slug"] == slug)
        out = PROMPTS / f"CANARY__{slug[:40]}__{d}.txt"
        out.write_text(render(q, desc, deadline, d, []))
        manifest.append({"slug": slug, "date": d, "prompt_file": str(out),
                         "n_articles": 0, "kind": "canary"})

    (RAW / "prompt_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} prompts rendered "
          f"({sum(1 for x in manifest if x['kind']=='scored')} scored + "
          f"{sum(1 for x in manifest if x['kind']=='canary')} canaries)")


if __name__ == "__main__":
    main()
