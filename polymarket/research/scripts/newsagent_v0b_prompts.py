"""News-agent v0b gate: render the redesigned (Amendment 2) forecast prompts.

Pre-registration: notes/news_agent/newsagent_v0_gate_preregistration.md § Amendment 2.
Changes vs v0 (the ONLY changes): status-quo clause removed -> evidence-weighting
instruction; 5 perspective-diverse estimates per call (trimmed-mean aggregation is
done by the orchestrator, never by the agent). Packets, universe, dates unchanged.

Outputs: data/newsagent/v0/prompts_v0b/<slug>__<date>.txt (+3 empty-packet canaries)
         data/newsagent/v0/prompt_manifest_v0b.json
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "newsagent" / "v0"
NEWS = RAW / "news"
PROMPTS = RAW / "prompts_v0b"

SNAPSHOT_DATES = ["2026-06-08", "2026-06-11", "2026-06-14", "2026-06-17",
                  "2026-06-20", "2026-06-23", "2026-06-26", "2026-06-29"]

TEMPLATE = """You are a calibrated superforecaster producing an INDEPENDENT fair-value probability for a prediction-market question. Today is {date}. Rely ONLY on the evidence packet below plus general knowledge of how the world works. Your training data predates these events — do NOT assume you know anything that happened after {date}. You have no market price and must not guess one; produce your own view.

QUESTION: {question}

RESOLUTION CRITERIA: {description}

DEADLINE: {deadline} ({days_left} days from today)

EVIDENCE PACKET (news headlines retrieved as of today, most relevant first):
{packet}

Method — produce FIVE independent estimates of P(YES), each from a different angle. Reason privately; report all five numbers:
1. BASE-RATE: the outside-view base rate for this class of event resolving YES within {days_left} days, ignoring the packet entirely.
2. EVIDENCE-FORWARD: take the packet at face value. If credible reporting indicates an imminent, in-progress, or completed qualifying event, this estimate MUST move materially toward it (cite the driving headline in drivers). Distinguish confirmed fact from speculation — but do not dismiss a strong, specific signal merely because the event has not completed yet.
3. SKEPTIC: what would have to be true for the packet headlines to mislead? Estimate under that reading.
4. REFERENCE-CLASS: the closest concrete historical analogs to this exact situation (not the generic class); estimate from their outcomes.
5. ADVERSARIAL: assume a smart, informed person disagrees with your evidence-forward number; steelman their case, then give your all-things-considered estimate.

Set "decisive_evidence" to true only if the packet contains credible reporting that the qualifying event is imminent, in progress, or completed.

Reply with a single JSON object only, no prose:
{{"estimates_pct": [<e1>, <e2>, <e3>, <e4>, <e5>], "drivers": ["<up to 3 short strings citing packet items or base-rate logic>"], "decisive_evidence": <true|false>}}
"""

CANARIES = [
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
    m = json.loads(http_get(url))[0]
    return (m.get("description") or "").strip(), m.get("endDate", "")


def render(question: str, description: str, deadline: str, date: str, articles: list[dict]) -> str:
    cutoff = f"{date.replace('-', '')}T120000Z"
    articles = [a for a in articles if (a.get("seendate") or "9") <= cutoff]
    days_left = max(0, (datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                        - datetime.fromisoformat(f"{date}T12:00:00+00:00")).days)
    packet = ("\n".join(f"- [{a.get('seendate','?')}] {a.get('domain','?')}: {a['title']}"
                        for a in articles)
              or "(no articles retrieved for this date)")
    return TEMPLATE.format(date=date, question=question, description=description,
                           deadline=deadline[:10], days_left=days_left, packet=packet)


def main() -> None:
    PROMPTS.mkdir(parents=True, exist_ok=True)
    selected = json.loads((RAW / "universe_selected.json").read_text())
    manifest, descs = [], {}
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
                print(f"  missing packet: {pf.name}")
                continue
            packet = json.loads(pf.read_text())
            out = PROMPTS / f"{m['slug']}__{d}.txt"
            out.write_text(render(m["question"], desc, deadline, d, packet["articles"]))
            manifest.append({"slug": m["slug"], "date": d, "prompt_file": str(out),
                             "kind": "scored"})

    for slug, d in CANARIES:
        if slug not in descs:
            continue
        desc, deadline = descs[slug]
        q = next(m["question"] for m in selected if m["slug"] == slug)
        out = PROMPTS / f"CANARY__{slug[:40]}__{d}.txt"
        out.write_text(render(q, desc, deadline, d, []))
        manifest.append({"slug": slug, "date": d, "prompt_file": str(out), "kind": "canary"})

    (RAW / "prompt_manifest_v0b.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} v0b prompts rendered "
          f"({sum(1 for x in manifest if x['kind'] == 'scored')} scored + "
          f"{sum(1 for x in manifest if x['kind'] == 'canary')} canaries)")


if __name__ == "__main__":
    main()
