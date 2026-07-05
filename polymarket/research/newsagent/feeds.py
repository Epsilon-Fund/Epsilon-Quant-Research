"""Market metadata + news-packet retrieval for the live observatory.

Sources (curated set, radar-verified licences; Scheme-A weights applied in Stage B):
  - Gamma API: market question/criteria/mid.
  - Guardian Open Platform: date-bounded search (headline+link only on the public
    page; attribution required). Uses GUARDIAN_API_KEY or the demo key.
  - Free RSS set (v3): BBC + Sky + Politico + The Hill politics/world feeds —
    fetched ONCE per run (day-cached), keyword-filtered per market; public display
    is headline + link only. Reuters/AP have no public RSS; Google News RSS is
    personal-use-only — both skipped (radar).
  - Wikipedia Current Events daily pages (CC BY-SA, attribution): full past days only.
  - Newsletters (v3, via email_ingest): analysis-grade text for Stage A ONLY —
    NEVER displayed on the public page (private/licensed content).
  - GDELT DOC client retained from the v0 scripts but NOT called by default —
    the BigQuery path (gdelt_bq) supplies the attention series instead.

All items carry timestamps; live packets have no lookahead concern (t = now) but we
keep the per-item cutoff for symmetry with the gate pipeline. Cross-source dedupe is
by normalized-title hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from .config import DATA, GUARDIAN_KEY_ENV

# Free feeds, live-probed 200 (2026-07-05). Display rule for every RSS source:
# headline + link only on the public page.
RSS_FEEDS = {
    "bbc.co.uk": ["https://feeds.bbci.co.uk/news/politics/rss.xml",
                  "https://feeds.bbci.co.uk/news/world/rss.xml"],
    "news.sky.com": ["https://feeds.skynews.com/feeds/rss/politics.xml",
                     "https://feeds.skynews.com/feeds/rss/world.xml"],
    "politico.com": ["https://rss.politico.com/politics-news.xml",
                     "https://rss.politico.com/congress.xml"],
    "thehill.com": ["https://thehill.com/homenews/feed/"],
}

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


def _lede_last(body: str, cap: int = 420) -> tuple[str, str]:
    """First and last substantive sentence-run of an article body, length-capped.

    Full text is used INTERNALLY for feature extraction only (Guardian licence:
    the public page displays headline + link only)."""
    body = (body or "").strip()
    if not body:
        return "", ""
    paras = [p.strip() for p in re.split(r"\n{2,}", body) if len(p.strip()) > 60]
    if not paras:
        return body[:cap], (body[-cap:] if len(body) > 2 * cap else "")
    if len(paras) == 1:  # bodyText often arrives as one flat block — head and tail
        return paras[0][:cap], (paras[0][-cap:] if len(paras[0]) > 2 * cap else "")
    return paras[0][:cap], paras[-1][:cap]


def guardian_search(q: str, start: datetime, end: datetime, page_size: int = 20,
                    full_text: bool = True) -> list[dict]:
    key = os.environ.get(GUARDIAN_KEY_ENV, "").strip() or "test"
    params = {"q": q, "from-date": start.strftime("%Y-%m-%d"), "to-date": end.strftime("%Y-%m-%d"),
              "order-by": "newest", "page-size": str(page_size), "api-key": key}
    if full_text:
        params["show-fields"] = "trailText,bodyText"
    url = "https://content.guardianapis.com/search?" + urllib.parse.urlencode(params)
    out = []
    for r in http_json(url).get("response", {}).get("results", []):
        pub = r.get("webPublicationDate", "")
        if not pub or pub > end.strftime("%Y-%m-%dT%H:%M:%SZ"):
            continue
        fields = r.get("fields") or {}
        lede, last = _lede_last(fields.get("bodyText", ""))
        trail = re.sub(r"<[^>]+>", "", fields.get("trailText", "") or "").strip()
        out.append({"title": r.get("webTitle", "").strip(),
                    "seendate": pub.replace("-", "").replace(":", ""),
                    "domain": "theguardian.com", "url": r.get("webUrl", ""),
                    "trail": trail[:300], "lede": lede, "last_para": last})
    return out


def title_hash(title: str) -> str:
    """Cross-source dedupe key: normalized-title hash (case/space/punct-insensitive)."""
    norm = re.sub(r"[^a-z0-9 ]", "", title.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def _rss_dt(text: str) -> str:
    """RSS/Atom date -> compact seendate (best effort; empty on failure)."""
    text = (text or "").strip()
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc)
            return dt.strftime("%Y%m%dT%H%M%SZ")
        except ValueError:
            continue
    return ""


def fetch_rss_items(day: str | None = None) -> list[dict]:
    """All RSS items across the feed set, fetched once per day (cached).

    Returns article-shaped dicts (title/seendate/domain/url/trail); packets filter
    them per market by keyword + time window. Parse tolerates RSS2 <item> and Atom
    <entry>; a dead feed degrades to zero items, never an exception."""
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_dir = DATA / "rss_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{day}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    items = []
    for domain, urls in RSS_FEEDS.items():
        for u in urls:
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
                root = ET.fromstring(urllib.request.urlopen(req, timeout=20).read())
            except Exception:
                continue
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            nodes = root.findall(".//item") or root.findall(".//atom:entry", ns)
            for it in nodes:
                title = (it.findtext("title") or it.findtext("atom:title", "", ns) or "").strip()
                if not title:
                    continue
                link = (it.findtext("link") or "").strip()
                if not link:
                    ln = it.find("atom:link", ns)
                    link = ln.get("href", "") if ln is not None else ""
                desc = re.sub(r"<[^>]+>", "", it.findtext("description")
                              or it.findtext("atom:summary", "", ns) or "").strip()
                seen = _rss_dt(it.findtext("pubDate") or it.findtext("atom:updated", "", ns))
                items.append({"title": title, "seendate": seen, "domain": domain,
                              "url": link, "trail": desc[:300]})
            time.sleep(0.3)
    cache.write_text(json.dumps(items))
    return items


def relevance_rank(items: list[dict], query: str, keys: list[str]) -> list[dict]:
    """Cheap keyword pre-rank before the top-k cap (Halawi rank-then-summarize step).

    Only ORDERS candidates; the calibrated per-article relevance comes from Stage A
    (features.py). Stable sort keeps newest-first within ties."""
    terms = {w for w in re.findall(r"[a-z]{3,}", query.lower())
             if w not in {"and", "or", "not", "the"}}
    terms |= {k.lower() for k in keys}

    def score(a: dict) -> int:
        text = " ".join([a.get("title", ""), a.get("trail", ""), a.get("lede", "")]).lower()
        return sum(1 for t in terms if t in text)

    return sorted(items, key=score, reverse=True)


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


def _keyword_filter(items: list[dict], query: str, keys: list[str],
                    start: datetime, end: datetime) -> list[dict]:
    """Window + keyword filter for broad-source items (RSS/newsletters)."""
    lo, hi = start.strftime("%Y%m%dT%H%M%SZ"), end.strftime("%Y%m%dT%H%M%SZ")
    ranked = relevance_rank(items, query, keys)
    out = []
    for a in ranked:
        sd = a.get("seendate", "")
        if sd and not (lo <= sd <= hi):
            continue
        text = " ".join([a.get("title", ""), a.get("trail", ""), a.get("lede", "")]).lower()
        if not any(k.lower() in text for k in keys):
            continue
        out.append(a)
    return out


def build_packet(slug: str, cfg: dict, now: datetime | None = None, max_items: int = 12,
                 rss_items: list[dict] | None = None,
                 newsletter_items: list[dict] | None = None) -> dict:
    """Packet = newsletters (analysis-grade, Stage-A-only, never displayed) +
    relevance-ranked Guardian items (title + trail + lede + last paragraph,
    internal-only text) + keyword-matched RSS headlines + Wikipedia Current Events
    bullets, deduped cross-source by normalized-title hash.

    Per-slot caps (of max_items=12): newsletters <=2, Guardian <=6, RSS <=4,
    WP fills the remainder. `now` in the past reconstructs a lookahead-free
    HISTORICAL packet from Guardian+WP only — RSS/newsletters are live-only
    sources (feeds carry current state; no timestamped archive), so they are
    excluded from any reconstruction by construction."""
    t = now or datetime.now(timezone.utc)
    g_items = guardian_search(cfg["guardian_q"], t - timedelta(hours=72), t)
    window = "72h"
    if len(g_items) < 3:
        time.sleep(0.7)
        g_items = guardian_search(cfg["guardian_q"], t - timedelta(days=7), t)
        window = "7d"
    time.sleep(0.7)
    g_items = relevance_rank(g_items, cfg["guardian_q"], cfg["wp_keys"])
    win_start = t - (timedelta(hours=72) if window == "72h" else timedelta(days=7))

    rss_sel = _keyword_filter(rss_items or [], cfg["guardian_q"], cfg["wp_keys"],
                              win_start, t)
    nl_sel = _keyword_filter(newsletter_items or [], cfg["guardian_q"], cfg["wp_keys"],
                             win_start, t)

    lookback = 3 if window == "72h" else 7
    wp_items = []
    for k in range(lookback, 0, -1):
        day = t - timedelta(days=k)
        for b in wp_day_bullets(day):
            if any(kw.lower() in b.lower() for kw in cfg["wp_keys"]):
                wp_items.append({"title": b[:200], "seendate": day.strftime("%Y%m%dT235900Z"),
                                 "domain": "en.wikipedia.org (Current events)"})
    seen, items = set(), []
    for a in nl_sel[:2] + g_items[:6] + rss_sel[:4] + wp_items[-4:]:
        key = title_hash(a.get("title", ""))
        if key in seen or not a.get("title"):
            continue
        seen.add(key)
        items.append(a)
        if len(items) >= max_items:
            break
    return {"slug": slug, "asof": t.isoformat(), "query": cfg["guardian_q"],
            "wp_keys": cfg["wp_keys"], "window": window,
            "source": "newsletters+guardian+rss+wp" if (nl_sel or rss_sel)
                      else "guardian+wp_currentevents",
            "articles": items}


def mid_history(slug: str, days: int = 21) -> list[dict]:
    """Daily mid history for the display time-series (CLOB /prices-history).

    Chunked <=10d per request (the API silently caps span ~15d); fidelity=60
    (hourly) downsampled to one 12:00-UTC-nearest point per day. Display context
    only — never a gate input."""
    url = "https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode({"slug": slug})
    rows = http_json(url)
    if not rows:
        return []
    token = json.loads(rows[0].get("clobTokenIds") or "[]")
    if not token:
        return []
    now = datetime.now(timezone.utc)
    pts: dict[str, dict] = {}
    start_all = now - timedelta(days=days)
    chunk_start = start_all
    while chunk_start < now:
        chunk_end = min(chunk_start + timedelta(days=10), now)
        q = urllib.parse.urlencode({"market": token[0],
                                    "startTs": int(chunk_start.timestamp()),
                                    "endTs": int(chunk_end.timestamp()), "fidelity": "60"})
        try:
            hist = http_json("https://clob.polymarket.com/prices-history?" + q).get("history", [])
        except Exception:
            hist = []
        for h in hist:
            ts = datetime.fromtimestamp(h["t"], tz=timezone.utc)
            d = ts.strftime("%Y-%m-%d")
            # keep the point nearest 12:00 UTC per day
            dist = abs(ts.hour * 60 + ts.minute - 720)
            if d not in pts or dist < pts[d]["dist"]:
                pts[d] = {"date": d, "mid": round(float(h["p"]), 4), "dist": dist}
        chunk_start = chunk_end
        time.sleep(0.4)
    return [{"date": v["date"], "mid": v["mid"]} for v in sorted(pts.values(), key=lambda x: x["date"])]
