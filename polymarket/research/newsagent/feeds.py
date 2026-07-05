"""Market metadata + news-packet retrieval for the live observatory.

Sources (Scheme B curated set, radar-verified licences):
  - Gamma API: market question/criteria/mid.
  - Guardian Open Platform: date-bounded headline search (headline+timestamp only on
    the public page; attribution required). Uses GUARDIAN_API_KEY or the demo key.
  - Wikipedia Current Events daily pages (CC BY-SA, attribution): full past days only.
  - GDELT DOC client retained from the v0 scripts but NOT called by default —
    re-enable when its API leaves the aggressive-throttle state.

All items carry timestamps; live packets have no lookahead concern (t = now) but we
keep the per-item cutoff for symmetry with the gate pipeline.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from .config import DATA, GUARDIAN_KEY_ENV

WIKI_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def http_json(url: str, retries: int = 4) -> dict:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise RuntimeError("unreachable")


def market_state(slug: str) -> dict:
    """Question, resolution criteria, deadline, and current mid for a live market."""
    url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode({"slug": slug})
    rows = http_json(url)
    if not rows:  # closed markets need the flag
        rows = http_json(url + "&closed=true")
    m = rows[0]
    best_bid = float(m.get("bestBid") or 0)
    best_ask = float(m.get("bestAsk") or 1)
    return {
        "slug": slug, "question": m.get("question"),
        "description": (m.get("description") or "").strip(),
        "end_date": m.get("endDate", ""), "closed": m.get("closed", False),
        "mid": round((best_bid + best_ask) / 2, 4),
        "best_bid": best_bid, "best_ask": best_ask,
        "volume24h": round(float(m.get("volume24hr") or 0)),
        "liquidity": round(float(m.get("liquidity") or 0)),
    }


def guardian_search(q: str, start: datetime, end: datetime, page_size: int = 20) -> list[dict]:
    key = os.environ.get(GUARDIAN_KEY_ENV, "").strip() or "test"
    params = {"q": q, "from-date": start.strftime("%Y-%m-%d"), "to-date": end.strftime("%Y-%m-%d"),
              "order-by": "newest", "page-size": str(page_size), "api-key": key}
    url = "https://content.guardianapis.com/search?" + urllib.parse.urlencode(params)
    out = []
    for r in http_json(url).get("response", {}).get("results", []):
        pub = r.get("webPublicationDate", "")
        if not pub or pub > end.strftime("%Y-%m-%dT%H:%M:%SZ"):
            continue
        out.append({"title": r.get("webTitle", "").strip(),
                    "seendate": pub.replace("-", "").replace(":", ""),
                    "domain": "theguardian.com", "url": r.get("webUrl", "")})
    return out


def wp_day_bullets(day: datetime) -> list[str]:
    cache_dir = DATA / "wp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{day.strftime('%Y-%m-%d')}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    page = f"Portal:Current_events/{day.year}_{WIKI_MONTHS[day.month - 1]}_{day.day}"
    url = ("https://en.wikipedia.org/w/api.php?"
           + urllib.parse.urlencode({"action": "parse", "page": page, "prop": "wikitext",
                                     "format": "json", "formatversion": "2"}))
    try:
        wikitext = http_json(url)["parse"]["wikitext"]
    except Exception:
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


def build_packet(slug: str, cfg: dict, now: datetime | None = None, max_items: int = 12) -> dict:
    """Same packet shape as the gate pipeline: (title, seendate, domain) x<=12."""
    t = now or datetime.now(timezone.utc)
    g_items = guardian_search(cfg["guardian_q"], t - timedelta(hours=72), t)
    window = "72h"
    if len(g_items) < 3:
        time.sleep(0.7)
        g_items = guardian_search(cfg["guardian_q"], t - timedelta(days=7), t)
        window = "7d"
    time.sleep(0.7)
    lookback = 3 if window == "72h" else 7
    wp_items = []
    for k in range(lookback, 0, -1):
        day = t - timedelta(days=k)
        for b in wp_day_bullets(day):
            if any(kw.lower() in b.lower() for kw in cfg["wp_keys"]):
                wp_items.append({"title": b[:200], "seendate": day.strftime("%Y%m%dT235900Z"),
                                 "domain": "en.wikipedia.org (Current events)"})
    seen, items = set(), []
    for a in g_items[:8] + wp_items[-4:]:
        key = a["title"].strip().lower()[:80]
        if key in seen or not a["title"]:
            continue
        seen.add(key)
        items.append(a)
        if len(items) >= max_items:
            break
    return {"slug": slug, "asof": t.isoformat(), "query": cfg["guardian_q"],
            "wp_keys": cfg["wp_keys"], "window": window,
            "source": "guardian+wp_currentevents", "articles": items}
