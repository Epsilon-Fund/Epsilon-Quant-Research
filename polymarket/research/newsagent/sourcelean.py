"""Political-lean axis for Stage-B source weighting + the ratings explainer (v3.2).

Justin's pick was Ground News for the lean axis. Access verdict (2026-07-05, ToS
read attended): NOT VIABLE as a data source —
  - no official public API or developer program exists;
  - ground.news Terms & Conditions expressly prohibit both "any robot, spider, or
    other automatic device, process, or means to access the Site for any purpose,
    including monitoring or copying any of the material on the Site" AND "any
    manual process to monitor or copy any of the material" (License Restrictions,
    read 2026-07-05). Being non-monetised does not cure a ToS prohibition.

So this module implements the plan's named fallback: a CURATED LEAN TABLE seeded
from AllSides Media Bias Ratings (the easier-access alternative; CC BY-NC 4.0
with attribution — acceptable for this non-monetised public showcase, and the
radar's "written NC OK" request stays open as a Justin item). Ground News's own
per-source ratings are averages of AllSides + Ad Fontes + MBFC, so for the small
outlet set we actually read, the AllSides axis is the same axis.

Scale (AllSides categories): -2 Left · -1 Lean Left · 0 Center · +1 Lean Right ·
+2 Right · None unrated. Only VERIFIED ratings enter the table (checked against
AllSides' published ratings 2026-07-05); everything else is honestly unrated.
Out-of-scope sources (newsletters, bank research PDFs, Wikipedia digests) are
unrated by construction — same scope rule as the RSP reliability tiers.

Routing (the "same Stage-B source weighting" ask): lean EXTREMITY maps to a
DECLARED contribution multiplier that composes with the Scheme-A reliability
weight in sourceweights.annotate():

    source_w = reliability_w × LEAN_EXTREMITY_MULT[|lean|]

i.e. more-biased outlets count a bit less toward the number and toward band
corroboration; direction of lean NEVER flips or directs evidence (a left outlet's
YES item counts the same as a right outlet's YES item at equal extremity).
Reliability (RSP + Iffy) keeps the blocklist role — reliability != lean.
The multiplier values are DECLARED (n far too small to fit) and listed for
Justin's sign-off in the v3.2 findings note; alpha is refit under the composed
weights via scripts/newsagent_hist_backfill.py --fit so the evidence scale stays
calibrated on resolved outcomes.
"""
from __future__ import annotations

# Verified AllSides ratings as of 2026-07-05 (rating moves are noted because
# AllSides revises: Guardian Lean Left -> Left (Nov 2024 editorial review);
# Daily Mail Right -> Lean Right (Sep 2025 editorial review).
DOMAIN_LEAN: dict[str, int] = {
    "theguardian.com": -2,    # Left
    "bbc.co.uk": 0,           # Center
    "politico.com": -1,       # Lean Left
    "thehill.com": 0,         # Center
    "bloomberg.com": -1,      # Lean Left
    "foxnews.com": 2,         # Right
    "dailymail.co.uk": 1,     # Lean Right
    # news.sky.com: no AllSides rating found -> deliberately absent (unrated)
}

# DECLARED extremity -> contribution multiplier (composes with reliability w).
# Center/unrated 1.0 keeps previous behavior; |1| and |2| are gentle discounts —
# the axis should inform, not dominate (reliability already carries the 0.0-1.0
# blocklist range). For Justin's sign-off; runs at these values, refit-covered.
LEAN_EXTREMITY_MULT = {0: 1.0, 1: 0.9, 2: 0.75}

LEAN_LABEL = {-2: "left", -1: "lean-left", 0: "center", 1: "lean-right", 2: "right"}

ATTRIBUTION = ("Political lean: curated table informed by AllSides Media Bias "
               "Ratings (allsides.com, CC BY-NC 4.0, non-commercial use with "
               "attribution); outlets without a published rating are shown as "
               "unrated and carry no lean adjustment.")


def get_lean(domain: str) -> int | None:
    """Lean for an article's source domain; None = unrated (no adjustment).
    Newsletters/PDF research/Wikipedia digests are out of scope -> unrated."""
    d = (domain or "").lower()
    if d.startswith("newsletter:") or d.startswith("en.wikipedia.org"):
        return None
    return DOMAIN_LEAN.get(d)


def lean_mult(lean: int | None) -> float:
    """Declared extremity multiplier on Stage-B contribution (1.0 when unrated)."""
    if lean is None:
        return 1.0
    return LEAN_EXTREMITY_MULT.get(abs(lean), 1.0)


def lean_label(lean: int | None) -> str:
    return LEAN_LABEL.get(lean, "unrated") if lean is not None else "unrated"


def lean_bucket(lean: int | None) -> str:
    """Coarse L/C/R bucket for the per-story coverage distribution."""
    if lean is None:
        return "unrated"
    if lean <= -1:
        return "left"
    if lean >= 1:
        return "right"
    return "center"
