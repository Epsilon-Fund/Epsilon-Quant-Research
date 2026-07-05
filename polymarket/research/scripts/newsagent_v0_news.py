"""News-agent v0 gate: fetch GDELT news packets per (market, snapshot date).

Pre-registration: notes/news_agent/newsagent_v0_gate_preregistration.md
Packet rule: GDELT DOC 2.0 artlist, market-specific query (recorded verbatim below),
window [t-72h, t] widened to [t-7d, t] if < 3 hits, English, sort=hybridrel,
dedupe by (domain,title), keep max 12 of (title, seendate, domain).
Throttle >= 6s/request (GDELT hard-limits ~1 req/5s). Cache append-only.

Outputs:
  data/newsagent/v0/news/<slug>__<date>.json
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "newsagent" / "v0"
NEWS = RAW / "news"

SNAPSHOT_DATES = ["2026-06-08", "2026-06-11", "2026-06-14", "2026-06-17",
                  "2026-06-20", "2026-06-23", "2026-06-26", "2026-06-29"]
SNAPSHOT_HOUR_UTC = 12

# Verbatim per-market GDELT queries (pre-registered artifact: recorded, not tuned on results).
# GDELT DOC syntax: space=AND, OR only inside parens, quoted phrases allowed.
QUERIES: dict[str, str] = {
    "us-x-iran-permanent-peace-deal-by-june-15-2026-734-856-129":
        'iran ("peace deal" OR "peace agreement" OR "nuclear deal" OR negotiations) sourcelang:english',
    "us-and-iran-sign-an-agreement-by-june-15-2026-20260611221049851":
        'iran (agreement OR deal OR sign OR talks) (us OR washington OR trump) sourcelang:english',
    "us-iran-nuclear-deal-by-june-30":
        'iran ("nuclear deal" OR enrichment OR uranium OR talks) sourcelang:english',
    "iran-agrees-to-end-enrichment-of-uranium-by-june-30":
        'iran (enrichment OR uranium OR "nuclear program") sourcelang:english',
    "will-the-iranian-regime-fall-by-june-30":
        '("iranian regime" OR "iran regime" OR khamenei) sourcelang:english',
    "iran-leadership-change-by-june-30-689-922":
        'iran (khamenei OR "supreme leader" OR "regime change" OR succession) sourcelang:english',
    "will-there-be-no-change-in-fed-interest-rates-after-the-june-2026-fed-meeting":
        '("federal reserve" OR fomc OR "fed meeting") ("interest rate" OR "rate cut" OR "rate hike" OR powell) sourcelang:english',
    "will-the-fed-decrease-interest-rates-by-25-bps-after-the-june-2026-fed-meeting":
        '("federal reserve" OR fomc) ("rate cut" OR "basis points" OR "interest rates") sourcelang:english',
    "starmer-out-by-june-30-2026-862-594-548-219-739-726-569-741-645":
        'starmer ("prime minister" OR resign OR "leadership challenge" OR "no confidence" OR labour) sourcelang:english',
    "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election":
        'colombia (cepeda OR "presidential election" OR "presidential race") sourcelang:english',
    "will-abelardo-de-la-espriella-win-the-2026-colombian-presidential-election":
        'colombia ("de la espriella" OR "presidential election" OR "presidential race") sourcelang:english',
    "will-rafael-lpez-aliaga-win-the-2026-peruvian-presidential-election":
        'peru ("lopez aliaga" OR "presidential election" OR "presidential race") sourcelang:english',
    "will-jorge-nieto-win-the-2026-peruvian-presidential-election":
        'peru (nieto OR "presidential election" OR "presidential race") sourcelang:english',
    "netanyahu-out-by-june-30-383-244-575":
        'netanyahu ("prime minister" OR resign OR coalition OR election OR out) sourcelang:english',
    "will-benjamin-netanyahu-enter-iran-by-june-30":
        'netanyahu (iran OR visit OR tehran) sourcelang:english',
    "trump-out-as-president-by-june-30":
        'trump (impeach OR resign OR "25th amendment" OR removal) sourcelang:english',
    "aleksandar-vui-out-as-serbian-president-by-june-30-2026-398":
        '(vucic OR serbia) (protest OR resign OR president) sourcelang:english',
    "will-russia-test-a-nuclear-weapon-by-june-30-2026":
        'russia ("nuclear test" OR "nuclear weapon" OR novaya) sourcelang:english',
    "putin-out-as-president-of-russia-by-june-30":
        'putin (resign OR succession OR "out of power" OR coup) sourcelang:english',
    "will-trump-agree-to-withdraw-troops-from-the-iranian-region-by-june-30":
        '(trump OR us) (troops OR withdrawal) (iran OR "middle east" OR gulf) sourcelang:english',
    "us-announces-new-iran-agreementceasefire-extension-by-june-13":
        'iran (ceasefire OR agreement OR extension OR truce) sourcelang:english',
    "israel-closes-its-airspace-by-june-15-687-594-783-732-455-613-653":
        'israel (airspace OR "flight suspension" OR aviation) sourcelang:english',
    "iran-closes-its-airspace-by-june-30-432-786-462-866-468":
        'iran (airspace OR flights OR aviation) sourcelang:english',
    "will-the-text-of-the-us-iran-agreement-be-released-by-june-16-2026":
        'iran agreement (text OR released OR publish OR details) sourcelang:english',
    "us-x-iran-diplomatic-meeting-by-june-21-2026-631-919-131-645-654-4":
        'iran (us OR washington) (meeting OR talks OR diplomatic) sourcelang:english',
    "israel-closes-its-airspace-by-june-30":
        'israel (airspace OR "flight suspension" OR aviation) sourcelang:english',
    "will-donald-trump-announce-that-the-united-states-blockade-of-the-strait-of-hormuz":
        '"strait of hormuz" (blockade OR lifted OR shipping OR traffic) sourcelang:english',
}


def query_for(slug: str) -> str | None:
    """Prefix match: Gamma appends numeric suffixes to slugs."""
    if slug in QUERIES:
        return QUERIES[slug]
    best = None
    for key, q in QUERIES.items():
        if slug.startswith(key) and (best is None or len(key) > len(best[0])):
            best = (key, q)
    return best[1] if best else None


def http_get(url: str, retries: int = 3) -> bytes:
    """GDELT 429 blocks appear to refresh on every hit: retry RARELY and wait LONG."""
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
            return urllib.request.urlopen(req, timeout=45).read()
        except Exception as exc:
            if i == retries - 1:
                raise
            print(f"    429/err -> quiet 150s ({exc})", flush=True)
            time.sleep(150)
    raise RuntimeError("unreachable")


def gdelt_artlist(query: str, start: datetime, end: datetime, maxrecords: int = 25) -> list[dict]:
    params = {
        "query": query, "mode": "artlist", "format": "json",
        "maxrecords": str(maxrecords), "sort": "hybridrel",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
    raw = http_get(url)
    try:
        return json.loads(raw).get("articles", [])
    except json.JSONDecodeError:
        # rate-limit text or transient html; retry once after a long sleep
        time.sleep(20)
        raw = http_get(url)
        try:
            return json.loads(raw).get("articles", [])
        except json.JSONDecodeError:
            print(f"    warn: non-JSON GDELT response persisted: {raw[:120]!r}")
            return []


def build_packet(slug: str, date: str) -> dict:
    q = query_for(slug)
    assert q is not None
    t = datetime.fromisoformat(f"{date}T{SNAPSHOT_HOUR_UTC:02d}:00:00+00:00")
    arts = gdelt_artlist(q, t - timedelta(hours=72), t)
    window = "72h"
    if len(arts) < 3:
        time.sleep(10.0)
        arts = gdelt_artlist(q, t - timedelta(days=7), t)
        window = "7d"
    seen, items = set(), []
    cutoff = t.strftime("%Y%m%dT%H%M%SZ")
    for a in arts:
        # GDELT's enddatetime bound is sloppy (can leak ~24h past); enforce client-side.
        if (a.get("seendate") or "99999999") > cutoff:
            continue
        key = (a.get("domain", ""), (a.get("title") or "").strip().lower())
        if key in seen or not a.get("title"):
            continue
        seen.add(key)
        items.append({"title": a["title"].strip(), "seendate": a.get("seendate"),
                      "domain": a.get("domain")})
        if len(items) >= 12:
            break
    return {"slug": slug, "snapshot_date": date, "query": q, "window": window,
            "fetched_at": datetime.now(timezone.utc).isoformat(), "articles": items}


def main() -> None:
    NEWS.mkdir(parents=True, exist_ok=True)
    selected = json.loads((RAW / "universe_selected.json").read_text())
    jobs = []
    for m in selected:
        if query_for(m["slug"]) is None:
            print(f"  MISSING QUERY for selected market: {m['slug']}  -- add to QUERIES and rerun")
            continue
        for d in SNAPSHOT_DATES:
            # only snapshots where the market was informative-or-open per universe CSV mid columns
            if m.get(f"mid_{d}", "") == "":
                continue
            jobs.append((m["slug"], d))
    print(f"{len(jobs)} (market, date) packets to fetch")
    failed = []
    for slug, d in jobs:
        out = NEWS / f"{slug}__{d}.json"
        if out.exists():
            continue
        print(f"  {d}  {slug[:55]}", flush=True)
        try:
            packet = build_packet(slug, d)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed.append((slug, d))
            time.sleep(30)
            continue
        out.write_text(json.dumps(packet, indent=1))
        print(f"    -> {len(packet['articles'])} articles ({packet['window']})", flush=True)
        time.sleep(16.0)
    if failed:
        print(f"\n{len(failed)} packets FAILED — rerun to resume (cache skips done ones)")


if __name__ == "__main__":
    main()
