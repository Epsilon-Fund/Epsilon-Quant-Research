"""Data-evidence channel — the READ side (Option C, live from 2026-08-24).

This module reads the JSON that `scripts/newsagent_datachannel_snapshot.py`
writes and hands `p_struct` to Stage B as a re-anchorable prior. It computes
nothing, fetches nothing, and — load-bearing — **imports no OpenBB**.

**The AGPL boundary.** OpenBB is AGPL-3.0-only. It is installed in the research
venv (sign-off row 1, "when the build starts") and may be imported by the offline
snapshot script, whose output is data. It must never be imported by `newsagent/*`,
which renders a public page. `tests/test_newsagent_datachannel.py` asserts that
boundary over the whole package rather than trusting a convention.

**Option C, and what it does NOT touch (DC-4).** The structural probability enters
through `p0` — the market's prior — and never through `A_t`, the decayed evidence
score. α, λ, the shift caps and the band are all untouched, so activating this
channel does not silently rescale anything the news path fitted. DC-5's blend
weight is **1.0 (full replacement)**, declared and signed off: on a data-channel
market the structural number IS the anchor, and the news evidence moves the fair
value away from it exactly as it always moved it away from the onboarding prior.

**Re-anchorable, not re-anchored blindly.** A stale snapshot is worse than none:
if the JSON is older than MAX_SNAPSHOT_AGE_DAYS the market silently falls back to
its stored onboarding prior and the run says so. Failing back to the news method
is always safe; failing forward on a stale macro state is not.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .config import ROOT

SNAPSHOT_DIR = ROOT / "data" / "newsagent" / "datachannel"
LATEST = SNAPSHOT_DIR / "p_struct_latest.json"

# A macro snapshot goes stale quickly around a release. Declared, not fitted.
MAX_SNAPSHOT_AGE_DAYS = 7

# ---------------------------------------------------------------------------
# § 4d — the double-count guard. DECLARED, test-enforced, no new LLM call.
# ---------------------------------------------------------------------------
# The news packet already contains articles ABOUT the data ("core PCE came in at
# 0.19%"). Once the print itself moves p_struct, the same information would move
# A_t again and be counted twice. The guard is a rule over fields we already
# have: an article on a data-channel market that Stage A tagged `economic` AND
# whose visible text names one of the mapped releases contributes **0** to the
# day's evidence score. It is still fetched, still displayed and still counted in
# the evidence list — it is labelled, not hidden, because suppressing it from the
# page would misrepresent what the model read.
DOUBLE_COUNT_EVENT_TYPES = frozenset({"economic"})
DOUBLE_COUNT_TERMS = (
    "core pce", "pce price", "pce inflation", "cpi", "consumer price",
    "inflation rate", "inflation print", "inflation data", "inflation report",
    "payrolls", "nonfarm", "non-farm", "jobs report", "jobs data",
    "unemployment rate", "labour market report", "labor market report",
    "employment report", "jobless", "fomc projections",
    "summary of economic projections", "dot plot",
)
DOUBLE_COUNT_LABEL = "already counted in the data channel"


def load_snapshot(path: Path | None = None) -> dict:
    """The latest snapshot, or {} when there is none (never raises)."""
    p = path or LATEST
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def p_struct_for(slug: str, run_date: str, snapshot: dict | None = None
                 ) -> tuple[float | None, str]:
    """(p_struct in %, why-not) for a data-channel market on `run_date`.

    Returns (None, reason) for every failure mode — no snapshot, market absent,
    snapshot too old — so the caller can degrade to the stored onboarding prior
    and print the reason, exactly as GDELT and the newsletters degrade.
    """
    snap = snapshot if snapshot is not None else load_snapshot()
    if not snap:
        return None, "no data-channel snapshot on disk (run the snapshot script)"
    rec = (snap.get("markets") or {}).get(slug)
    if not rec:
        return None, f"no p_struct for {slug[:40]} in the snapshot"
    try:
        age = (date.fromisoformat(run_date) - date.fromisoformat(rec["asof"])).days
    except Exception:
        return None, "snapshot carries an unparseable asof date"
    if age > MAX_SNAPSHOT_AGE_DAYS:
        return None, (f"snapshot is {age}d old (max {MAX_SNAPSHOT_AGE_DAYS}) — "
                      "falling back to the onboarding prior")
    if age < 0:
        return None, "snapshot is dated after the run date"
    return float(rec["p_struct_pct"]), ""


def is_double_count(article: dict, features: dict | None) -> bool:
    """§ 4d — does this article merely report a release the data channel already ate?

    Both legs must hold: Stage A tagged it `economic`, AND its visible text names
    a mapped release. The event-type leg alone would swallow every macro story
    (including ones carrying genuinely new information, like a Fed official's
    speech); the keyword leg alone would catch a political story that mentions
    inflation in passing.
    """
    if not features or features.get("event_type") not in DOUBLE_COUNT_EVENT_TYPES:
        return False
    text = " ".join([article.get("title", ""), article.get("trail", ""),
                     article.get("lede", "")]).lower()
    return any(term in text for term in DOUBLE_COUNT_TERMS)


def apply_double_count_guard(feats: list[dict], slug: str) -> tuple[list[dict], int]:
    """Zero the Stage-B weight of double-counting rows on a data-channel market.

    Returns (rows, n_excluded). The row keeps its features and its place in the
    evidence list — only `source_w` goes to 0 and `double_counted` is set, so the
    card can label it. Non-data-channel markets are returned untouched.
    """
    from .config import DATA_CHANNEL_MARKETS
    if slug not in DATA_CHANNEL_MARKETS:
        return feats, 0
    n = 0
    for r in feats:
        if is_double_count(r.get("article", {}), r.get("features")):
            r["double_counted"] = True
            r["source_w_before_double_count"] = r.get("source_w", 1.0)
            r["source_w"] = 0.0
            n += 1
    return feats, n


def market_implied_note(slug: str, snapshot: dict | None = None) -> dict | None:
    """Display-only market-implied context (DC-6), or a structured absence.

    Never an input. The Atlanta Fed MPT stops quoting a 3-month window once that
    window opens, so absence is the NORMAL state near a meeting and the card must
    render it in words rather than break.
    """
    snap = snapshot if snapshot is not None else load_snapshot()
    rec = (snap.get("markets") or {}).get(slug)
    return (rec or {}).get("market_implied")
