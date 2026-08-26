"""Evidence provenance and the declared display bands (dashboard v3.5).

WHAT THIS IS. A read-only DISPLAY layer that answers one question per market:
*what evidence actually reached this number?* It groups the overview grid by the
evidence source that really fed each market, states each card's provenance in
words, and makes "no relevant evidence found" impossible to miss.

WHAT THIS IS NOT — and this is the load-bearing sentence in the file. Nothing
here is a model input. The bands, the window and the threshold below are
DISPLAY BANDS: they decide how a card is presented, never what Stage B computes.
No function in this module is imported by `fvmodel`, `run_daily` or the ledger,
and the published number is byte-identical whether this module exists or not.

WHY IT EXISTS. Two observations from Justin on the v3.4 page:

  1. The overview grid was grouped by nothing — 24 donuts in one wall, ordered by
     disagreement. A reader could not tell which numbers were built on a month of
     coverage and which had read nothing at all, because the tract tag ("news-
     driven" / "data-driven" / "poll-driven") describes what the question is
     ABOUT, not what actually arrived. A news-driven market with zero relevant
     articles looks exactly like a news-driven market with thirteen.

  2. A card showing a fair value on a market with zero relevant articles is
     misleading as presented. The California-governor market publishes 14.0%
     against a 95.2% mid; that 14.0% is the onboarding prior of 2026-07-05,
     untouched by evidence since. Presented in the same type as a researched
     estimate, it reads as a considered disagreement. It is not one.

THE FOUR DECLARED GROUPS

  news+data       the structural data channel is anchoring the number, i.e. the
                  published FV is re-anchored on `p_struct` rather than on the
                  onboarding prior. Membership follows the METHOD, never the
                  intention: it is exactly `p0_source == "p_struct"`, which is
                  exactly `config.DATA_CHANNEL_MARKETS` at publish time.
  news, well-fed  >= WELL_FED_MIN distinct relevant articles in the trailing
                  window.
  news, thin      1 .. WELL_FED_MIN-1 distinct relevant articles.
  prior-only      zero. The published number IS the onboarding prior (possibly
                  plus decayed carry from evidence that has since aged out of
                  the window), untouched by anything the loop has read lately.

THE THRESHOLDS, AND WHY THEY ARE THESE NUMBERS

  WINDOW_DAYS = 30. The observatory runs ATTENDED, not daily: published snapshots
  have been up to 50 days apart, and the packet's own retrieval window is 72h
  (7d on a thin day). A 72h window would therefore say "no evidence" about a
  market that was well fed a week ago, and a 90d window would call a market
  well-fed on coverage that predates its last two publishes. 30 days is the
  shortest span that contains at least one published packet for every live market
  at the current cadence, and it is inside the horizon over which the v3.4
  reconstruction's Brier-vs-staleness curve is flat — i.e. the span over which a
  number is still meaningfully "current" on this evidence diet.

  WELL_FED_MIN = 5. This is NOT a new number: it is `config.DIVERGENCE_NREL_MIN`,
  the confidence leg the page already uses to decide whether a disagreement is
  trustworthy enough to flag. Reusing it means the page has ONE bar for "enough
  evidence to take this seriously", not two that a reader has to reconcile. Note
  what it means over 30 days rather than 72 hours: it is a deliberately LOW bar,
  and that is the point. A market that cannot clear five relevant articles in a
  month is not being fed, and "well-fed" here means "cleared the floor", never
  "well researched".

  Both are declared here, in code, so they can be read and argued with. Changing
  either changes only which box a card renders in.

WHAT COUNTS AS EVIDENCE THAT "REACHED" A MARKET

  Distinct articles — deduped by normalized-title hash, so the same story
  appearing in three consecutive packets counts once — that were in a packet
  behind a PUBLISHED snapshot inside the window. Two exclusions, both deliberate:

  - Reconstructed pre-launch packets (`data/newsagent/live/backfill/`) are NOT
    counted. They never produced a published number, so they cannot have fed one.
    This is the same rule the movers strip already applies to the FV series, and
    it is applied here for the same reason.
  - An article with no Stage-A feature record counts as an item but never as a
    RELEVANT item. An uncached article contributes zero evidence to the model,
    so counting it as evidence on the page would overstate the diet.

  "Relevant" is `fvmodel.RELEVANT_MIN` (0.4) — the same threshold the band, the
  divergence flag and the evidence-quality badge already use.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from . import config, features, feeds, fvmodel

# ---------------------------------------------------------------- thresholds --
# DISPLAY BANDS. Never model inputs. Rationale in the module docstring.
WINDOW_DAYS = 30
WELL_FED_MIN = config.DIVERGENCE_NREL_MIN

# An FV within this many pp of its prior is displayed as "unchanged". 0.05pp is
# half the page's own display precision (numbers render to 0.1pp), so it means
# "identical at every digit the reader can see", not "close enough".
UNCHANGED_TOL_PP = 0.05

# ------------------------------------------------------------------- channels --
# The evidence channels the page names in words. Order is the order they are read
# out in a provenance line: primary documents first, then news, then the
# structural channel, then the amplifier.
CH_REACH_DOC = "reach-document"
CH_REACH_TRANSCRIPT = "reach-transcript"
CH_NEWSLETTER = "newsletter"
CH_PDF = "pdf"
CH_GUARDIAN = "guardian"
CH_RSS = "rss"
CH_WIKIPEDIA = "wikipedia"
CH_DATA = "data-channel"
CH_GDELT = "gdelt"
CH_OTHER = "other"

CHANNEL_ORDER = (CH_REACH_DOC, CH_REACH_TRANSCRIPT, CH_DATA, CH_GUARDIAN, CH_RSS,
                 CH_WIKIPEDIA, CH_NEWSLETTER, CH_PDF, CH_GDELT, CH_OTHER)

# Public-facing words. These are what a reader sees; the constants above are what
# the code passes around.
CHANNEL_LABEL = {
    CH_GUARDIAN: "Guardian",
    CH_RSS: "RSS (BBC/Sky/Politico/The Hill)",
    CH_WIKIPEDIA: "Wikipedia Current Events",
    CH_NEWSLETTER: "newsletters (private, counted never shown)",
    CH_PDF: "macro-research PDFs",
    CH_REACH_DOC: "reach — official document",
    CH_REACH_TRANSCRIPT: "reach — official transcript (private)",
    CH_DATA: "data channel (official statistics)",
    CH_GDELT: "GDELT attention burst",
    CH_OTHER: "other",
}

# Domains of the macro-research PDF slot (pdf_ingest.PDF_SOURCES). Kept as a set
# here rather than imported, because importing pdf_ingest pulls a PDF parser into
# every dashboard render for four strings.
_PDF_DOMAINS = frozenset({"am.jpmorgan.com", "jpmorganfunds.com", "am.gs.com", "ml.com"})


def channel_of(article: dict) -> str:
    """Which evidence channel one packet article arrived on.

    Reach items are self-identifying (`reach_kind` is set by `newsagent.reach`
    from the curated worklist, never from the fetched payload). Everything else
    is identified by the domain the fetcher stamped, which is the same field the
    display boundary and the source-weight table key on.
    """
    kind = article.get("reach_kind")
    if kind == "document":
        return CH_REACH_DOC
    if kind == "transcript":
        return CH_REACH_TRANSCRIPT
    dom = (article.get("domain") or "").strip().lower()
    if dom.startswith("newsletter:"):
        return CH_NEWSLETTER
    if dom.startswith("youtube.com/@"):
        # a caption track reaching the packet without a reach_kind stamp
        return CH_REACH_TRANSCRIPT
    if dom.startswith("theguardian.com"):
        return CH_GUARDIAN
    if dom.startswith("en.wikipedia.org"):
        return CH_WIKIPEDIA
    if dom in feeds.RSS_FEEDS:
        return CH_RSS
    if dom in _PDF_DOMAINS:
        return CH_PDF
    return CH_OTHER


# ---------------------------------------------------------------- the groups --
GROUP_NEWS_DATA = "news+data"
GROUP_WELL_FED = "news, well-fed"
GROUP_THIN = "news, thin"
GROUP_PRIOR_ONLY = "prior-only"
GROUP_ORDER = (GROUP_NEWS_DATA, GROUP_WELL_FED, GROUP_THIN, GROUP_PRIOR_ONLY)

GROUP_LEAD = {
    GROUP_NEWS_DATA: (
        "The structural data channel is anchoring these numbers: the published "
        "value starts from a probability built out of official statistics "
        "(CPI/PCE/labour + the Cleveland Fed nowcast + FOMC projections) rather "
        "than from the onboarding prior, and news evidence moves it from there. "
        "This method has its own calibration track and is never merged with the "
        "news-only one."),
    GROUP_WELL_FED: (
        f"At least {WELL_FED_MIN} distinct relevant articles reached these "
        f"markets' packets in the last {WINDOW_DAYS} days. That is the same "
        "evidence bar the divergence flag already uses, applied over a month "
        "instead of 72 hours — a floor, not a badge: it means the loop was "
        "reading something, not that the question is well researched."),
    GROUP_THIN: (
        f"Between 1 and {WELL_FED_MIN - 1} distinct relevant articles in the "
        f"last {WINDOW_DAYS} days. Something arrived, but not enough to clear "
        "the page's own confidence bar. Read these numbers as a prior that has "
        "been nudged, not as a researched estimate."),
    GROUP_PRIOR_ONLY: (
        f"<b>Zero relevant articles in the last {WINDOW_DAYS} days.</b> The "
        "published number here IS the onboarding prior — a one-time "
        "five-perspective LLM estimate made when the market was added, never "
        "shown the market price — carried forward untouched. It is not a "
        "researched disagreement with the market, and the gap column should not "
        "be read as one. These cards are marked so they cannot be mistaken for "
        "the rest at a glance."),
}


def group_for(n_relevant_window: int, p0_source: str = "onboarding_prior") -> str:
    """The declared display band for one market.

    Order matters: the data channel is checked FIRST, because a market whose
    number is anchored on official statistics belongs in `news+data` regardless
    of how much news also reached it — the anchor is the thing a reader most
    needs to know about that number.
    """
    if p0_source == "p_struct":
        return GROUP_NEWS_DATA
    if n_relevant_window <= 0:
        return GROUP_PRIOR_ONLY
    if n_relevant_window >= WELL_FED_MIN:
        return GROUP_WELL_FED
    return GROUP_THIN


# ------------------------------------------------------- packet history on disk --

def _day_dirs(root: Path | None = None) -> list[Path]:
    """The dated day directories holding PUBLISHED packets, oldest first.

    `data/newsagent/live/<YYYY-MM-DD>/`. `backfill/` is not a dated directory and
    is therefore never picked up — the reconstructed pre-launch packets it holds
    never produced a published number (see the module docstring).
    """
    base = root or config.DATA
    if not base.exists():
        return []
    out = []
    for p in base.iterdir():
        n = p.name
        if p.is_dir() and len(n) == 10 and n[4] == "-" and n[7] == "-" and n[:4].isdigit():
            out.append(p)
    return sorted(out)


def _packet_articles(day: Path, slug: str) -> list[dict]:
    """Today's packet for one market out of one day directory, or []. The `[:80]`
    truncation mirrors `run_daily._load_day` — that is how these files are named."""
    p = day / f"{slug[:80]}.packet.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("articles") or []
    except (ValueError, OSError):
        return []


def _relevance(slug: str, title: str) -> float | None:
    """Stage-A relevance for one (market, article), or None when uncached."""
    if not title:
        return None
    rec = features.cached(features.cache_key(slug, title))
    if not rec:
        return None
    try:
        return float(rec["relevance"])
    except (KeyError, TypeError, ValueError):
        return None


def evidence_window(slug: str, asof: str, articles_today: list[dict] | None = None,
                    window_days: int = WINDOW_DAYS,
                    root: Path | None = None) -> dict:
    """Distinct evidence that reached this market's published packets in the
    trailing window ending on `asof`.

    `articles_today` is the current snapshot's packet, passed in rather than read
    from disk so a caller that is holding the live snapshot (or a test holding a
    synthetic one) is scored on exactly what it is holding. On-disk history is
    merged in on top of it, deduped by normalized-title hash.

    Returns counts and the channel sets; degrades to the passed-in packet alone
    when nothing is on disk, and to an empty read when there is neither.
    """
    try:
        end = datetime.fromisoformat(asof[:10])
    except (TypeError, ValueError):
        end = None
    start = (end - timedelta(days=window_days)).date().isoformat() if end else ""
    end_s = end.date().isoformat() if end else ""

    seen: dict[str, tuple[str, float | None]] = {}
    for a in articles_today or []:
        title = a.get("title", "")
        seen[feeds.title_hash(title)] = (channel_of(a), _relevance(slug, title))
    # No parseable as-of date means no window, and a window we cannot date is not
    # one we may fill from disk: score exactly what was handed in. (Anchoring to
    # the wall clock instead would make the page's own history time-dependent —
    # the defect class that produced all three v3.4 data-channel bugs.)
    for day in (_day_dirs(root) if end else []):
        if not (start <= day.name <= end_s):
            continue
        for a in _packet_articles(day, slug):
            title = a.get("title", "")
            key = feeds.title_hash(title)
            if key in seen:
                continue
            seen[key] = (channel_of(a), _relevance(slug, title))

    rel = [(ch, r) for ch, r in seen.values()
           if r is not None and r >= fvmodel.RELEVANT_MIN]
    order = {c: i for i, c in enumerate(CHANNEL_ORDER)}
    return {
        "window_days": window_days,
        "window_from": start,
        "window_to": end_s,
        "n_items": len(seen),
        "n_relevant": len(rel),
        "channels": sorted({ch for ch, _ in rel}, key=lambda c: order.get(c, 99)),
        "channels_all": sorted({ch for ch, _ in seen.values()},
                               key=lambda c: order.get(c, 99)),
    }


# ------------------------------------------------------------- per-card record --

def _priors(root: Path | None = None) -> dict:
    """The ledger priors of record: {slug: {p0_pct, set_on, ...}}. Absent -> {}."""
    p = (root or config.DATA) / "priors.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return {}


def for_card(card: dict, articles_today: list[dict] | None = None, asof: str = "",
             priors: dict | None = None, root: Path | None = None) -> dict:
    """The full provenance record the page renders for one market card.

    Every field is descriptive. `moved_pp` in particular is a DISPLAY of how far
    evidence has carried the number away from where it started; it is not fed
    back into anything.
    """
    slug = card.get("slug", "")
    win = evidence_window(slug, asof, articles_today, root=root)
    pri = (priors if priors is not None else _priors(root)).get(slug) or {}

    channels = list(win["channels"])
    p0_source = card.get("p0_source", "onboarding_prior")
    if p0_source == "p_struct":
        channels.append(CH_DATA)
    # GDELT is an AMPLIFIER, not a source of articles: `fvmodel.amplify` scales a
    # day's directional evidence when vol_z > 0 and does nothing at all otherwise
    # (and nothing to a zero-evidence day). It is therefore named as a contributor
    # only when it could actually have moved the number — listing it on every card
    # because a series exists would overstate the diet on 24 cards out of 24.
    g = card.get("gdelt") or {}
    vol_z = g.get("vol_z")
    gdelt_amplified = bool(vol_z is not None and vol_z > 0 and win["n_relevant"] > 0)
    if gdelt_amplified:
        channels.append(CH_GDELT)

    fv = card.get("fv_pct")
    p0 = pri.get("p0_pct")
    anchor = card.get("p0_used_pct") if p0_source == "p_struct" else None
    moved = round(fv - p0, 1) if (fv is not None and p0 is not None) else None
    # For a news+data market the honest "did news move it" number is measured from
    # the STRUCTURAL ANCHOR, not from the onboarding prior: the anchor is what the
    # news term is applied to, so anchor->published is the news half's whole
    # contribution, and prior->anchor is the data channel's.
    moved_from_anchor = (round(fv - anchor, 1)
                         if (fv is not None and anchor is not None) else None)

    unchanged = bool(moved is not None and abs(moved) < UNCHANGED_TOL_PP)
    return {
        **win,
        "group": group_for(win["n_relevant"], p0_source),
        "channels": channels,
        "gdelt_amplified": gdelt_amplified,
        "prior_pct": p0,
        "prior_set_on": pri.get("set_on") or "",
        "anchor_pct": anchor,
        "moved_pp": moved,
        "moved_from_anchor_pp": moved_from_anchor,
        "unchanged": unchanged,
    }


def movement_summary(cards: list[dict]) -> dict:
    """How far evidence has moved these published numbers off their priors.

    The live-page counterpart to the v3.4 reconstruction's central mechanism
    finding (mean |FV - prior| 3.35pp; 66% of markets never move 1pp). Computed
    over exactly the markets on the page, so a reader can check it against the
    per-card indicators rather than take it on trust.
    """
    d = [abs(c["provenance"]["moved_pp"]) for c in cards
         if (c.get("provenance") or {}).get("moved_pp") is not None]
    if not d:
        return {"n": 0, "mean_abs_pp": None, "n_moved_ge_1pp": 0,
                "share_moved_ge_1pp": None, "share_still_ge_1pp": None}
    moved = sum(1 for x in d if x >= 1.0)
    return {"n": len(d),
            "mean_abs_pp": round(sum(d) / len(d), 2),
            "n_moved_ge_1pp": moved,
            "share_moved_ge_1pp": round(moved / len(d), 3),
            "share_still_ge_1pp": round(1 - moved / len(d), 3)}


# ---------------------------------------------------- per-group settled scores --

def settled_by_group(records: list[dict] | None = None,
                     root: Path | None = None) -> dict:
    """Settled forward-ledger Briers, split by the evidence group each market was
    in when it resolved.

    The group is recomputed for the settled market over the window ending on its
    RESOLUTION date — what fed it while it was live, not what a retired slug looks
    like today. Every cell is tiny (the whole forward ledger is n=4), so each row
    carries its own n and the page marks it as such: this is a description of four
    settled markets, not a result.
    """
    if records is None:
        try:
            from . import ledger   # noqa: PLC0415  (optional at render time)
            records = ledger.settled_records()
        except Exception:
            return {}
    out: dict[str, dict] = {}
    for r in records:
        slug, end = r.get("slug", ""), (r.get("resolution_date") or "")[:10]
        if not slug or not end:
            continue
        win = evidence_window(slug, end, [], root=root)
        p0_source = "p_struct" if r.get("method") == "news+data" else "onboarding_prior"
        g = group_for(win["n_relevant"], p0_source)
        row = out.setdefault(g, {"n": 0, "briers": []})
        row["n"] += 1
        row["briers"].append(float(r.get("brier", 0.0)))
    for g, row in out.items():
        row["brier"] = round(sum(row["briers"]) / len(row["briers"]), 4)
        del row["briers"]
    return out


def group_rows(cards: list[dict], settled: dict | None = None) -> list[dict]:
    """The overview grid's group headers: one row per NON-EMPTY declared group,
    in declared order, each with its market count and — where at least one market
    in that group has settled — that group's own forward Brier."""
    settled = settled if settled is not None else settled_by_group()
    rows = []
    for g in GROUP_ORDER:
        members = [c for c in cards if (c.get("provenance") or {}).get("group") == g]
        if not members:
            continue
        s = settled.get(g) or {}
        rows.append({"group": g, "n": len(members), "lead": GROUP_LEAD[g],
                     "settled_n": s.get("n", 0), "settled_brier": s.get("brier"),
                     "slugs": [c["slug"] for c in members]})
    return rows
