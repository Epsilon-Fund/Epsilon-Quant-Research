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
Events digests, private research newsletters, bank research desks) are OUT OF
SCOPE of the tiers and take a DECLARED weight from UNCOVERED_SOURCE_W below.

Sign-off status: the uncovered-source weights proposed in v3/v3.1 were APPROVED
by Justin on 2026-08-24 and are LIVE from that date (ING 0.9 / Wikipedia Current
Events 0.8 / Bloomberg 0.9 / bank research desks 0.9 / genuinely unknown 0.5).
Before that date every uncovered source ran at the neutral 1.0, so alpha fitted
under the old neutral default is not comparable — activating these weights
changes Stage-B inputs and REQUIRES a refit through
scripts/newsagent_hist_backfill.py --fit (done the same day; see
newsagent_observatory_v33_findings). The values are declared judgments, not
fits — n is far too small to fit them.

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
# DECLARED weights for sources outside the RSP news-outlet scope (Justin
# APPROVED 2026-08-24; ran at the neutral 1.0 before that date).
#   ING / Bloomberg newsletters, bank research desks -> 0.9: professional
#     analysis with a named house behind it, but not a news outlet the RSP
#     community has ruled on, so a small discount against a generally-reliable
#     wire rather than parity.
#   Wikipedia Current Events -> 0.8: a curated, sourced digest, one editorial
#     step removed from the reporting it summarises.
#   genuinely unknown -> 0.5: no basis to trust or distrust; it should be able
#     to move the number, but only half as far as a source we have vetted.
UNKNOWN_W = 0.5    # anything with no declared row and no RSP tier
WP_CE_W = 0.8      # Wikipedia Current Events digests (domain prefix match)
UNCOVERED_SOURCE_W: dict[str, float] = {
    "newsletter:ing think": 0.9,
    "newsletter:ing research": 0.9,
    "newsletter:bloomberg": 0.9,
    "bloomberg.com": 0.9,
    # bank research desks — the pdf_ingest macro set (pdf_ingest.PDF_SOURCES)
    "am.jpmorgan.com": 0.9,
    "jpmorganfunds.com": 0.9,
    "am.gs.com": 0.9,
    "ml.com": 0.9,
}
UNCOVERED_W = UNKNOWN_W  # back-compat alias: the default for uncovered sources

# ---------------------------------------------------------------------------
# agent-reach source types (v3.4). APPROVED as proposed by Justin 2026-08-24
# ([[newsagent_agentreach_scoping]] § Sign-off block 1); inert until this build,
# because no reach item could carry a weight before there were reach items.
# ---------------------------------------------------------------------------
# The split below is the substantive point and it is a DECLARED judgment, not a
# fit: **"official" is not the same as "reliable."** A Federal Reserve statement
# about the Federal Reserve's own decision is a PROCEDURAL fact about the issuing
# institution — as close to ground truth as a source gets. A Kremlin readout about
# Russian intentions is a primary document AND an interested party's contested
# claim about itself. Collapsing both into one "official = 1.0" row would import
# propaganda at maximum weight, so they get separate rows.
OFFICIAL_PRIMARY_W = 1.0        # procedural fact about the issuing institution
OFFICIAL_STATE_CLAIM_W = 0.5    # a state actor's contested claim about itself
OFFICIAL_TRANSCRIPT_W = 1.0     # an official body's own video, official channel
UNKNOWN_VIDEO_W = 0.5           # any other uploader (or its RSP tier if wired)

# Domains curated in newsagent/reach_sources.json. Kept HERE rather than read from
# the JSON so the weight table stays a declared constant in code that tests pin —
# a data-file edit must never be able to move a source into a heavier weight row.
OFFICIAL_PRIMARY_DOMAINS = frozenset({
    "federalreserve.gov", "congress.gov", "state.gov", "war.gov", "centcom.mil",
    "nato.int", "ukmto.org", "imo.org", "iaea.org", "nobelprize.org",
    "sos.ca.gov", "conseil-constitutionnel.fr", "tse.jus.br", "interieur.gouv.fr",
    "whitehouse.gov",
})
OFFICIAL_STATE_CLAIM_DOMAINS = frozenset({
    "kremlin.ru", "en.kremlin.ru", "mfa.gov.ir", "president.gov.ua", "mod.ru",
    "gov.il", "cec.gov.ru",
})
# Official-body channels, as they appear in an item's `domain` field
# ("youtube.com/@handle"). Anything else on youtube.com takes UNKNOWN_VIDEO_W.
OFFICIAL_TRANSCRIPT_CHANNELS = frozenset({
    "youtube.com/@federalreserve", "youtube.com/@statedept",
    "youtube.com/@centcom", "youtube.com/@nobelprize",
})
_VIDEO_PREFIX = "youtube.com/@"


def _reach_weight(domain: str) -> float | None:
    """Declared weight for a reach source domain, or None if it is not one.

    NOTE the caller checks the Iffy blocklist BEFORE this — an official domain
    that ever lands on the blocklist still resolves to 0.0. That precedence is
    test-enforced (`test_blocklist_beats_an_official_reach_domain`).
    """
    d = (domain or "").lower().strip()
    if d.startswith(_VIDEO_PREFIX):
        return (OFFICIAL_TRANSCRIPT_W if d in OFFICIAL_TRANSCRIPT_CHANNELS
                else UNKNOWN_VIDEO_W)
    # state-claim is checked first: a domain must never fall through to the
    # procedural 1.0 row just because it also looks official.
    if d in OFFICIAL_STATE_CLAIM_DOMAINS:
        return OFFICIAL_STATE_CLAIM_W
    if d in OFFICIAL_PRIMARY_DOMAINS:
        return OFFICIAL_PRIMARY_W
    return None


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
    """Stage-B weight for an article's source domain.

    Resolution order (blocklist first — Iffy always wins, test-enforced):
      1. Iffy blocklist            -> 0.0
      2. RSP tier (news outlets)   -> RSP_STATUS_W
      3. declared uncovered row    -> UNCOVERED_SOURCE_W / WP_CE_W
      4. declared reach row (v3.4) -> official primary / state claim / transcript
      5. anything else             -> UNKNOWN_W (0.5)
    """
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
    d = (domain or "").lower().strip()
    if d in _IFFY:
        return 0.0
    if d in _WEIGHTS:
        return _WEIGHTS[d]
    if d.startswith("en.wikipedia.org"):
        return WP_CE_W
    if d in UNCOVERED_SOURCE_W:
        return UNCOVERED_SOURCE_W[d]
    reach_w = _reach_weight(d)
    if reach_w is not None:
        return reach_w
    return UNKNOWN_W


def annotate(feats: list[dict]) -> list[dict]:
    """Attach source weighting to features_for() rows (Stage-B consumers read it).

    v3.2: the effective Stage-B weight composes RELIABILITY (Scheme A: RSP tiers
    + Iffy blocklist) with the POLITICAL-LEAN extremity multiplier (sourcelean —
    AllSides-seeded curated table; declared mults). Kept separately on the row so
    the ratings explainer can show each axis on its own:
      source_w_rel  — reliability-only weight (0.0–1.0; blocklist lives here)
      source_lean   — -2..+2 or None (unrated)
      source_w      — source_w_rel × lean_mult (what contribution/band consume)
    """
    from . import sourcelean
    for r in feats:
        dom = r.get("article", {}).get("domain", "")
        rel = get_weight(dom)
        lean = sourcelean.get_lean(dom)
        r["source_w_rel"] = rel
        r["source_lean"] = lean
        r["source_w"] = rel * sourcelean.lean_mult(lean)
    return feats


def summary() -> dict:
    cache = refresh(force=False)
    return {"fetched_at": cache.get("fetched_at"),
            "weights": _build_weights(cache),
            "n_rsp_entries": cache.get("n_rsp", 0),
            "n_iffy_domains": cache.get("n_iffy", 0),
            "uncovered_default": UNKNOWN_W,
            "uncovered_declared": dict(UNCOVERED_SOURCE_W, **{"en.wikipedia.org": WP_CE_W}),
            "reach_declared": {"official_primary": OFFICIAL_PRIMARY_W,
                               "official_state_claim": OFFICIAL_STATE_CLAIM_W,
                               "official_transcript": OFFICIAL_TRANSCRIPT_W,
                               "unknown_video": UNKNOWN_VIDEO_W},
            "attribution": ATTRIBUTION}
