"""Scheme A — source reliability weights for Stage-B aggregation (v3).

Two licence-clean public datasets (radar-approved pattern, running default was
Scheme B flat weights until now):

  1. Wikipedia perennial-sources tiers (CC BY-SA 4.0, attribution): the RSP table
     encodes each entry's community-consensus status as a CSS class on its row
     (`<tr class="s-gr" id="BBC">`). Status -> weight mapping is DECLARED below.
  2. Iffy Index of unreliable sources (CC BY 4.0, attribution): domain blocklist
     -> weight 0.0, overriding everything.

The weight scales an article's influence in STAGE-B aggregation (contribution and
band dispersion) — never Stage-A extraction. Scope rule: the RSP tier mapping
applies to NEWS OUTLETS. Sources that are not news outlets (Wikipedia Current
Events digests, private research newsletters) are OUT OF SCOPE of the tiers and
run at the neutral 1.0 (identical to the previous flat Scheme B) until Justin
signs off the proposed tier mapping in the v3 findings note — nothing is
hard-adopted for uncovered sources.

Refresh cadence: weekly, ATTENDED (`refresh(force=True)` — this round has no
cron by mandate). The cache carries fetched_at; a stale cache warns but still
serves (a weights outage must never break the daily loop).
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from .config import DATA

CACHE = DATA / "sourceweights.json"
STALE_DAYS = 10   # warn past this; weekly refresh is the documented cadence

# DECLARED status -> weight mapping (not fitted; n far too small to fit).
RSP_STATUS_W = {
    "s-gr": 1.0,   # generally reliable
    "s-nc": 0.7,   # no consensus / marginal / additional considerations
    "s-gu": 0.3,   # generally unreliable
    "s-d": 0.0,    # deprecated
    "s-b": 0.0,    # blacklisted
}
UNCOVERED_W = 1.0  # neutral (= previous flat Scheme B) pending sign-off

# Our packet domains -> RSP row ids (parsed case-insensitively from the table).
DOMAIN_RSP_ID = {
    "theguardian.com": "the guardian",
    "bbc.co.uk": "bbc",
    "news.sky.com": "sky news uk",   # RSP splits Sky News UK vs Sky News Australia
    "politico.com": "politico",
    "thehill.com": "the hill",
}

IFFY_CSV_URL = ("https://docs.google.com/spreadsheets/d/"
                "1ck1_FZC-97uDLIlvRJDTrGqBk0FuDe9yHkluROgpGS8/gviz/tq?tqx=out:csv")

ATTRIBUTION = ("Source-reliability weights: Wikipedia perennial-sources list "
               "(CC BY-SA 4.0) and the Iffy Index of unreliable sources "
               "(iffy.news, CC BY 4.0).")


def _fetch_rsp_statuses() -> dict[str, str]:
    """RSP row-id (lowercased, underscores->spaces) -> status class."""
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "parse", "page": "Wikipedia:Reliable sources/Perennial sources",
        "prop": "text", "format": "json", "formatversion": "2"})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
    html = json.loads(urllib.request.urlopen(req, timeout=90).read())["parse"]["text"]
    out = {}
    for m in re.finditer(r'<tr class="(s-\w+)[^"]*" id="([^"]+)"', html):
        status, row_id = m.group(1), m.group(2)
        out[urllib.parse.unquote(row_id).replace("_", " ").lower()] = status
    return out


def _fetch_iffy_domains() -> list[str]:
    req = urllib.request.Request(IFFY_CSV_URL, headers={"User-Agent": "Mozilla/5.0 (epsilon-research)"})
    text = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    return sorted({r["Domain"].strip().lower() for r in rows if r.get("Domain", "").strip()})


def refresh(force: bool = False) -> dict:
    """Fetch + cache both datasets. Attended weekly; failures keep the old cache."""
    if CACHE.exists() and not force:
        return json.loads(CACHE.read_text())
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    try:
        rsp = _fetch_rsp_statuses()
        time.sleep(0.5)
        iffy = _fetch_iffy_domains()
        cache = {"fetched_at": datetime.now(timezone.utc).isoformat(),
                 "rsp_status_by_id": rsp, "iffy_domains": iffy,
                 "n_rsp": len(rsp), "n_iffy": len(iffy)}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1))
    except Exception as e:
        print(f"  sourceweights: refresh failed ({e}) — serving previous cache")
    return cache


_WEIGHTS: dict[str, float] | None = None
_IFFY: set[str] = set()


def _build_weights(cache: dict) -> dict[str, float]:
    rsp = cache.get("rsp_status_by_id", {})
    iffy = set(cache.get("iffy_domains", []))
    w = {}
    for domain, rsp_id in DOMAIN_RSP_ID.items():
        status = rsp.get(rsp_id)
        w[domain] = RSP_STATUS_W.get(status, UNCOVERED_W)
        if domain in iffy:
            w[domain] = 0.0
    return w


def get_weight(domain: str) -> float:
    """Stage-B weight for an article's source domain (1.0 for uncovered/absent)."""
    global _WEIGHTS, _IFFY
    if _WEIGHTS is None:
        cache = refresh(force=False)
        if cache.get("fetched_at"):
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(cache["fetched_at"])).days
            if age > STALE_DAYS:
                print(f"  sourceweights: cache is {age}d old — run an attended "
                      "refresh (sourceweights.refresh(force=True))")
        _WEIGHTS = _build_weights(cache)
        _IFFY = set(cache.get("iffy_domains", []))
    d = (domain or "").lower()
    if d.startswith("newsletter:") or d.startswith("en.wikipedia.org"):
        return UNCOVERED_W   # out of RSP scope; proposal pending sign-off
    if d in _WEIGHTS:
        return _WEIGHTS[d]
    if d in _IFFY:
        return 0.0
    return UNCOVERED_W


def annotate(feats: list[dict]) -> list[dict]:
    """Attach source_w to features_for() rows (Stage-B consumers read it)."""
    for r in feats:
        r["source_w"] = get_weight(r.get("article", {}).get("domain", ""))
    return feats


def summary() -> dict:
    cache = refresh(force=False)
    return {"fetched_at": cache.get("fetched_at"),
            "weights": _build_weights(cache),
            "n_rsp_entries": cache.get("n_rsp", 0),
            "n_iffy_domains": cache.get("n_iffy", 0),
            "uncovered_default": UNCOVERED_W,
            "attribution": ATTRIBUTION}
