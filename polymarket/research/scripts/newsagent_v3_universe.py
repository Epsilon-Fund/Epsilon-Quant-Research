"""Slate-refresh discovery for the Observatory (v3): candidate live markets.

Polymarket is the DISCOVERY layer — this script surfaces which liquid US+UK
politics/macro questions people currently care about; the model never sees or
targets the mid. Prints config-ready candidate entries (guardian_q / wp_keys /
gdelt_keys are auto-drafted from the question text and are meant to be HAND-TUNED
before adoption into config.LIVE_MARKETS — curation is deliberate, not automatic).

Filters (mirrors the v0 universe rules, live variant):
  binary Yes/No; not closed; informative mid (5c-95c); liquidity >= $100k;
  resolution within ~16 months; politics/geopolitics/macro by tag or keyword.

Run: PYTHONPATH=. uv run python scripts/newsagent_v3_universe.py [--limit 40]
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request

MACRO_KEYWORDS = ["fed ", "federal reserve", "interest rate", "recession", "inflation",
                  "tariff", "gdp", "shutdown", "debt ceiling"]
POLITICS_KEYWORDS = ["election", "president", "prime minister", "senate", "house",
                     "congress", "parliament", "minister", "resign", "impeach",
                     "ceasefire", "peace", "war", "treaty", "sanction", "nato",
                     "supreme court", "nominee", "cabinet", "putin", "trump"]


def http_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


SPORTS_NOISE = ["fifa", "world cup", "win on 20", "vs.", "premier league", "nba",
                "nfl", "ufc", "grand prix", "wimbledon", "olympic"]


def fetch_candidates(limit: int) -> list[dict]:
    out, seen = [], set()
    # politics tag (tag_id=2) paginated + a liquidity-ranked general sweep for
    # macro questions that live outside the politics tag
    queries = [{"closed": "false", "tag_id": "2", "limit": "100", "offset": str(off),
                "order": "liquidityNum", "ascending": "false",
                "liquidity_num_min": "50000"} for off in (0, 100, 200)]
    queries.append({"closed": "false", "limit": "200", "order": "liquidityNum",
                    "ascending": "false", "liquidity_num_min": "100000"})
    for params in queries:
        url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode(params)
        for m in http_json(url):
            slug = m.get("slug", "")
            if not slug or slug in seen:
                continue
            seen.add(slug)
            try:
                outcomes = json.loads(m.get("outcomes") or "[]")
            except Exception:
                outcomes = []
            if sorted(o.lower() for o in outcomes) != ["no", "yes"]:
                continue
            bid, ask = float(m.get("bestBid") or 0), float(m.get("bestAsk") or 1)
            mid = (bid + ask) / 2
            liq = float(m.get("liquidityNum") or m.get("liquidity") or 0)
            q = (m.get("question") or "").lower()
            if not (0.05 <= mid <= 0.95) or liq < 50_000:
                continue
            end = m.get("endDate", "") or ""
            if end[:4] not in ("2026", "2027"):
                continue
            if any(k in q for k in SPORTS_NOISE):
                continue
            is_topic = (params.get("tag_id") == "2"
                        or any(k in q for k in POLITICS_KEYWORDS)
                        or any(k in q for k in MACRO_KEYWORDS))
            if not is_topic:
                continue
            out.append({"slug": slug, "question": m.get("question"), "mid": round(mid, 3),
                        "liquidity": round(liq), "volume24h": round(float(m.get("volume24hr") or 0)),
                        "end": end[:10]})
    return sorted(out, key=lambda x: -x["liquidity"])[:limit]


def draft_keys(question: str) -> dict:
    """Auto-draft retrieval keys from the question text (HAND-TUNE before adopting)."""
    words = re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?", question or "")
    stop = {"Will", "The", "Yes", "No", "By", "Before", "After", "In", "On", "US", "Any"}
    ents = [w for w in words if w.split()[0] not in stop][:3]
    key = ents[0].lower() if ents else (question or "").split()[-1].lower()
    return {"guardian_q": " AND ".join(f'"{e}"' if " " in e else e for e in ents[:2]) or key,
            "wp_keys": [e.lower() for e in ents[:3]] or [key],
            "gdelt_keys": [ents[0].lower()] if ents else [key]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    cands = fetch_candidates(args.limit)
    print(f"# {len(cands)} candidates (binary, open, informative mid, liq>=100k)\n")
    for c in cands:
        keys = draft_keys(c["question"])
        print(f"# {c['question']}  | mid={c['mid']}  liq=${c['liquidity']:,}  "
              f"24h=${c['volume24h']:,}  ends={c['end']}")
        print(f'"{c["slug"]}": {json.dumps(keys, indent=None)},\n')


if __name__ == "__main__":
    main()
