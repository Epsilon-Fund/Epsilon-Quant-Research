"""News-agent v0 gate: packet fetcher v2 — Guardian Open Platform + Wikipedia Current Events.

Pre-registration Amendment 1 (2026-07-05): GDELT DOC entered a persistent 429 state
(reproduced from two unrelated IPs); packets switch to two sources that are
time-stamped BY CONSTRUCTION, before any scored forecast existed:

  (a) Guardian content API, date-bounded search; per-item webPublicationDate <= t enforced.
  (b) Wikipedia Current Events portal daily pages — only FULL PAST days (t-3 .. t-1);
      day t itself is excluded because the daily page aggregates the whole day
      (would leak events after the 12:00 UTC snapshot).

Packet rule unchanged: window [t-72h, t] (widen to 7d if < 3 items), dedupe, max 12
items of (title, seendate, domain). Up to 8 Guardian + up to 4 Wikipedia items.

Outputs: data/newsagent/v0/news/<slug>__<date>.json   (same schema as v1 fetcher)
         data/newsagent/v0/wp_currentevents_cache/<YYYY-MM-DD>.json
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "newsagent" / "v0"
NEWS = RAW / "news"
WPCACHE = RAW / "wp_currentevents_cache"

SNAPSHOT_DATES = ["2026-06-08", "2026-06-11", "2026-06-14", "2026-06-17",
                  "2026-06-20", "2026-06-23", "2026-06-26", "2026-06-29"]
SNAPSHOT_HOUR_UTC = 12
GUARDIAN_KEY = "test"  # public demo key; v1 must use a registered free dev key (flag for Justin)

# Verbatim per-market retrieval config (recorded, not tuned on results).
# guardian_q uses Guardian search syntax (AND/OR, quotes); wp_keys is a case-insensitive
# any-of filter over Current Events bullet lines.
CONFIG: dict[str, dict] = {
    "us-x-iran-permanent-peace-deal-by-june-15-2026":
        {"guardian_q": 'iran AND (deal OR agreement OR talks OR peace)', "wp_keys": ["iran"]},
    "iran-agrees-to-end-enrichment-of-uranium-by-june-30":
        {"guardian_q": 'iran AND (enrichment OR uranium OR nuclear)', "wp_keys": ["iran"]},
    "starmer-out-by-june-30-2026":
        {"guardian_q": 'starmer', "wp_keys": ["starmer", "united kingdom"]},
    "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election":
        {"guardian_q": 'colombia AND (election OR cepeda)', "wp_keys": ["colombia"]},
    "will-abelardo-de-la-espriella-win-the-2026-colombian-presidential-election":
        {"guardian_q": 'colombia AND (election OR espriella)', "wp_keys": ["colombia"]},
    "will-trump-agree-to-withdraw-troops-from-the-iranian-region-by-june-30":
        {"guardian_q": 'iran AND troops AND (withdraw OR withdrawal OR pullout)', "wp_keys": ["iran"]},
    "israel-closes-its-airspace-by-june-15":
        {"guardian_q": 'israel AND (airspace OR flights OR aviation)', "wp_keys": ["israel"]},
    "aleksandar-vui-out-as-serbian-president-by-june-30-2026":
        {"guardian_q": 'serbia OR vucic', "wp_keys": ["serbia", "vučić", "vucic"]},
    "israel-closes-its-airspace-by-june-30":
        {"guardian_q": 'israel AND (airspace OR flights OR aviation)', "wp_keys": ["israel"]},
    "will-donald-trump-announce-that-the-united-states-blockade-of-the-strait-of-hormuz":
        {"guardian_q": '"strait of hormuz"', "wp_keys": ["hormuz"]},
}


def config_for(slug: str) -> dict | None:
    if slug in CONFIG:
        return CONFIG[slug]
    best = None
    for key, c in CONFIG.items():
        if slug.startswith(key) and (best is None or len(key) > len(best[0])):
            best = (key, c)
    return best[1] if best else None


def http_json(url: str, retries: int = 4) -> dict:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as exc:
            if i == retries - 1:
                raise
            print(f"    retry in {6 * (i + 1)}s ({exc})", flush=True)
            time.sleep(6 * (i + 1))
    raise RuntimeError("unreachable")


def guardian_search(q: str, start: datetime, end: datetime, page_size: int = 20) -> list[dict]:
    params = {
        "q": q, "from-date": start.strftime("%Y-%m-%d"), "to-date": end.strftime("%Y-%m-%d"),
        "order-by": "newest", "page-size": str(page_size), "api-key": GUARDIAN_KEY,
    }
    url = "https://content.guardianapis.com/search?" + urllib.parse.urlencode(params)
    res = http_json(url).get("response", {})
    out = []
    for r in res.get("results", []):
        pub = r.get("webPublicationDate", "")  # ISO, e.g. 2026-06-14T09:31:00Z
        if not pub or pub > end.strftime("%Y-%m-%dT%H:%M:%SZ"):
            continue  # strict per-item cutoff at the snapshot instant
        if pub < start.strftime("%Y-%m-%dT%H:%M:%SZ"):
            continue
        out.append({
            "title": r.get("webTitle", "").strip(),
            "seendate": pub.replace("-", "").replace(":", ""),
            "domain": "theguardian.com",
        })
    return out


WIKI_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def wp_day_bullets(day: datetime) -> list[str]:
    """Cached fetch of one Current Events daily page -> cleaned bullet lines."""
    WPCACHE.mkdir(parents=True, exist_ok=True)
    cache = WPCACHE / f"{day.strftime('%Y-%m-%d')}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    page = f"Portal:Current_events/{day.year}_{WIKI_MONTHS[day.month - 1]}_{day.day}"
    url = ("https://en.wikipedia.org/w/api.php?"
           + urllib.parse.urlencode({"action": "parse", "page": page, "prop": "wikitext",
                                     "format": "json", "formatversion": "2"}))
    try:
        wikitext = http_json(url)["parse"]["wikitext"]
    except Exception as exc:
        print(f"    warn: WP page {page} failed: {exc}")
        wikitext = ""
    bullets = []
    for line in wikitext.splitlines():
        if not line.startswith("*"):
            continue
        txt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", line.lstrip("* "))
        txt = re.sub(r"\{\{[^}]*\}\}|<[^>]+>|''+|\[https?://\S+\s?([^\]]*)\]", r"\1", txt).strip()
        if len(txt) > 20:
            bullets.append(txt)
    cache.write_text(json.dumps(bullets))
    time.sleep(0.6)
    return bullets


def build_packet(slug: str, question_cfg: dict, date: str) -> dict:
    t = datetime.fromisoformat(f"{date}T{SNAPSHOT_HOUR_UTC:02d}:00:00+00:00")
    g_items = guardian_search(question_cfg["guardian_q"], t - timedelta(hours=72), t)
    window = "72h"
    if len(g_items) < 3:
        time.sleep(0.7)
        g_items = guardian_search(question_cfg["guardian_q"], t - timedelta(days=7), t)
        window = "7d"
    time.sleep(0.7)

    lookback = 3 if window == "72h" else 7
    wp_items = []
    for k in range(lookback, 0, -1):  # full past days only, oldest first; day t excluded
        day = t - timedelta(days=k)
        for b in wp_day_bullets(day):
            if any(kw.lower() in b.lower() for kw in question_cfg["wp_keys"]):
                wp_items.append({"title": b[:200], "seendate": day.strftime("%Y%m%dT235900Z"),
                                 "domain": "en.wikipedia.org (Current events)"})

    seen, items = set(), []
    for a in g_items[:8] + wp_items[-4:]:
        key = a["title"].strip().lower()[:80]
        if key in seen or not a["title"]:
            continue
        seen.add(key)
        items.append(a)
        if len(items) >= 12:
            break
    return {"slug": slug, "snapshot_date": date, "query": question_cfg["guardian_q"],
            "wp_keys": question_cfg["wp_keys"], "window": window, "source": "guardian+wp_currentevents",
            "fetched_at": datetime.now(timezone.utc).isoformat(), "articles": items}


def main() -> None:
    NEWS.mkdir(parents=True, exist_ok=True)
    selected = json.loads((RAW / "universe_selected.json").read_text())
    jobs = []
    for m in selected:
        cfg = config_for(m["slug"])
        if cfg is None:
            print(f"  MISSING CONFIG: {m['slug']}")
            continue
        for d in SNAPSHOT_DATES:
            if m.get(f"mid_{d}", "") != "":
                jobs.append((m["slug"], cfg, d))
    print(f"{len(jobs)} packets to fetch (guardian+wp)")
    failed = []
    for slug, cfg, d in jobs:
        out = NEWS / f"{slug}__{d}.json"
        if out.exists():
            continue
        print(f"  {d}  {slug[:55]}", flush=True)
        try:
            packet = build_packet(slug, cfg, d)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed.append((slug, d))
            continue
        out.write_text(json.dumps(packet, indent=1))
        print(f"    -> {len(packet['articles'])} items ({packet['window']})", flush=True)
    if failed:
        print(f"\n{len(failed)} FAILED — rerun to resume")


if __name__ == "__main__":
    main()
