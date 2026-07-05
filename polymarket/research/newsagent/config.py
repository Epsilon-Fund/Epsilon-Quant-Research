"""Live market slate + retrieval config for the calibration observatory.

Slate rule (v1): live binary politics/macro markets, highest liquidity first, US+UK
first, near-to-mid horizon. Polymarket is the DISCOVERY layer only (which liquid
questions people care about) and display context — never a benchmark the model is
fit to. Curated hand-list, reviewed when markets resolve.

Market types (mtype) drive Stage-B decay + band floors (fvmodel.py):
  slow  — structural questions (elections, scheduled policy decisions): evidence
          decays slowly, FV should be stable between genuine developments.
  shock — event-driven geopolitics/personnel: evidence decays fast, band floor is
          wider; we hold no insider information, so shock misses are expected and
          honestly scored.

Source weighting: Scheme B (flat curated whitelist via Guardian + Wikipedia Current
Events + RSS) pending sign-off on Scheme A — see newsagent_repo_data_radar_findings.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "newsagent" / "live"
SHOWCASE = ROOT / "data" / "newsagent" / "showcase"
CSV_OUT = ROOT / "data" / "analysis" / "csv_outputs" / "news_agent"

GUARDIAN_KEY_ENV = "GUARDIAN_API_KEY"   # falls back to the public 'test' demo key
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"  # API stages need this (or the out-of-band files)

# Divergence flag (public, informational): |FV - mid| >= GAP AND band half-width <=
# HALF AND >= NREL relevant articles in 72h. 15pp ~= 2x the v0 median |gap|; the
# confidence leg keeps the flag off thin/uncertain evidence. Never an edge claim.
DIVERGENCE_GAP_PP = 15.0
DIVERGENCE_HALF_MAX_PP = 12.0
DIVERGENCE_NREL_MIN = 5

# v3.1 tractability tagging (Justin's call: keep ALL 24 markets, do NOT drop).
# A market is DATA-DRIVEN when its dominant information channel is not news text:
# rate decisions price off Fed-funds futures/options-implied odds; primary races
# ride private/campaign polling. Our news-FV is structurally blind there — the
# public page says so on those cards ("not news-tractable") and they stay scored
# in public anyway (the honest-measurement point). Everything else = news-driven.
# BACKLOG (revisit): restrict the news-FV universe to news-driven markets + add a
# rates-via-options/futures econ track as a separate, labeled method.
DATA_DRIVEN: dict[str, str] = {
    "will-there-be-no-change-in-fed-interest-rates-after-the-july-2026-meeting":
        "Rate decisions are priced off Fed-funds futures/options-implied odds; "
        "news text adds little beyond what those markets already carry.",
    "will-there-be-no-change-in-fed-interest-rates-after-the-september-2026-meeting-615":
        "Rate decisions are priced off Fed-funds futures/options-implied odds; "
        "news text adds little beyond what those markets already carry.",
    "will-xavier-becerra-win-the-california-governor-election-in-2026":
        "State-primary races move on private/campaign polling that never reaches "
        "the news packet; the market carries polling information we cannot see.",
    "billionaire-one-time-wealth-tax-passes-in-california-election-2026":
        "Ballot-measure odds ride issue polling, not news coverage; our packet "
        "sees the campaign noise, not the poll numbers.",
}


def tract(slug: str) -> str:
    """news | data — v3.1 tractability tag (see DATA_DRIVEN above)."""
    return "data" if slug in DATA_DRIVEN else "news"


# slug -> retrieval config (guardian_q Guardian search syntax; wp_keys any-of filter;
# gdelt_keys AND-substring match on GKG AllNames; mtype slow/shock; region display tag)
# v3 slate (2026-07-05, discovery via scripts/newsagent_v3_universe.py): 24 markets,
# informative mids (5-95c), liquidity >= $100k, <= 2 per event family, curated by hand.
# US+UK first — still NO informative UK binary live (Starmer family resolved; the
# discovery sweep found none above the liquidity floor); French/Brazil/Russia
# elections add slow-type diversity meanwhile. Revisit each slate refresh.
LIVE_MARKETS: dict[str, dict] = {
    "putin-out-before-2027": {
        "guardian_q": "putin AND (resign OR succession OR power OR health)",
        "wp_keys": ["putin", "russia"],
        "gdelt_keys": ["vladimir putin"],
        "region": "geopolitics", "mtype": "shock",
    },
    "strait-of-hormuz-traffic-returns-to-normal-by-july-31": {
        "guardian_q": "\"strait of hormuz\" AND (shipping OR traffic OR reopen OR tanker)",
        "wp_keys": ["hormuz"],
        "gdelt_keys": ["strait of hormuz"],
        "region": "geopolitics", "mtype": "shock",
    },
    "will-there-be-no-change-in-fed-interest-rates-after-the-july-2026-meeting": {
        "guardian_q": "\"federal reserve\" AND (rates OR cut OR powell OR fomc)",
        "wp_keys": ["federal reserve", "interest rate"],
        "gdelt_keys": ["federal reserve"],
        "region": "US", "mtype": "slow",
    },
    "us-x-iran-diplomatic-meeting-by-july-17-2026-20260625223459704": {
        "guardian_q": "iran AND (meeting OR talks OR diplomatic OR negotiation)",
        "wp_keys": ["iran"],
        "gdelt_keys": ["iran", "united states"],
        "region": "geopolitics", "mtype": "shock",
    },
    "will-the-democratic-party-control-the-house-after-the-2026-midterm-elections": {
        "guardian_q": "(democrats OR republicans) AND (house OR midterm OR congress)",
        "wp_keys": ["united states house", "midterm", "democratic party"],
        "gdelt_keys": ["democratic party"],
        "region": "US", "mtype": "slow",
    },
    # ---- v3 additions (2026-07-05) --------------------------------------------
    "trump-out-as-president-before-2027": {
        "guardian_q": "trump AND (impeachment OR removal OR resign OR \"25th amendment\")",
        "wp_keys": ["trump"],
        "gdelt_keys": ["donald trump"],
        "region": "US", "mtype": "shock",
    },
    "will-the-iranian-regime-fall-by-the-end-of-2026": {
        "guardian_q": "iran AND (regime OR protest OR uprising OR khamenei)",
        "wp_keys": ["iran"],
        "gdelt_keys": ["iran"],
        "region": "geopolitics", "mtype": "shock",
    },
    "will-the-us-invade-iran-before-2027": {
        "guardian_q": "iran AND (strike OR military OR invasion OR troops)",
        "wp_keys": ["iran"],
        "gdelt_keys": ["iran", "united states"],
        "region": "geopolitics", "mtype": "shock",
    },
    "will-iran-announce-withdrawal-from-mou-negotiations-by-july-31-20260622191733846": {
        "guardian_q": "iran AND (negotiations OR talks OR agreement OR withdrawal)",
        "wp_keys": ["iran"],
        "gdelt_keys": ["iran", "united states"],
        "region": "geopolitics", "mtype": "shock",
    },
    "strait-of-hormuz-traffic-returns-to-normal-by-december-31": {
        "guardian_q": "\"strait of hormuz\" AND (shipping OR traffic OR reopen OR tanker)",
        "wp_keys": ["hormuz"],
        "gdelt_keys": ["strait of hormuz"],
        "region": "geopolitics", "mtype": "shock",
    },
    "will-ukraine-recapture-crimean-territory-by-december-31-2026": {
        "guardian_q": "crimea OR (ukraine AND counteroffensive)",
        "wp_keys": ["crimea", "ukraine"],
        "gdelt_keys": ["crimea"],
        "region": "geopolitics", "mtype": "shock",
    },
    "russia-x-ukraine-ceasefire-agreement-by-december-31-2026": {
        "guardian_q": "russia AND ukraine AND (ceasefire OR peace OR talks)",
        "wp_keys": ["ukraine", "russia"],
        "gdelt_keys": ["ukraine", "russia"],
        "region": "geopolitics", "mtype": "shock",
    },
    "zelenskyy-out-as-ukraine-president-before-2027": {
        "guardian_q": "zelenskyy AND (election OR resign OR succession OR president)",
        "wp_keys": ["zelenskyy", "ukraine"],
        "gdelt_keys": ["zelensky"],   # substring matches both spellings
        "region": "geopolitics", "mtype": "shock",
    },
    "will-china-invade-taiwan-by-december-31-2027": {
        "guardian_q": "taiwan AND (china OR invasion OR military OR blockade)",
        "wp_keys": ["taiwan"],
        "gdelt_keys": ["taiwan"],
        "region": "geopolitics", "mtype": "shock",
    },
    "will-benjamin-netanyahu-be-the-next-prime-minister-of-israel": {
        "guardian_q": "netanyahu AND (coalition OR election OR government)",
        "wp_keys": ["netanyahu", "israel"],
        "gdelt_keys": ["netanyahu"],
        "region": "geopolitics", "mtype": "shock",
    },
    "will-the-us-invade-cuba-in-2026": {
        "guardian_q": "cuba AND (military OR invasion OR blockade OR sanctions)",
        "wp_keys": ["cuba"],
        "gdelt_keys": ["cuba"],
        "region": "geopolitics", "mtype": "shock",
    },
    "will-the-democratic-party-control-the-senate-after-the-2026-midterm-elections": {
        "guardian_q": "(democrats OR republicans) AND (senate OR midterm)",
        "wp_keys": ["united states senate", "midterm", "democratic party"],
        "gdelt_keys": ["democratic party"],
        "region": "US", "mtype": "slow",
    },
    "will-there-be-no-change-in-fed-interest-rates-after-the-september-2026-meeting-615": {
        "guardian_q": "\"federal reserve\" AND (rates OR cut OR powell OR fomc)",
        "wp_keys": ["federal reserve", "interest rate"],
        "gdelt_keys": ["federal reserve"],
        "region": "US", "mtype": "slow",
    },
    "will-xavier-becerra-win-the-california-governor-election-in-2026": {
        "guardian_q": "becerra OR \"california governor\"",
        "wp_keys": ["becerra", "california"],
        "gdelt_keys": ["becerra"],
        "region": "US", "mtype": "slow",
    },
    "billionaire-one-time-wealth-tax-passes-in-california-election-2026": {
        "guardian_q": "california AND (\"wealth tax\" OR ballot OR proposition)",
        "wp_keys": ["california", "wealth tax"],
        "gdelt_keys": ["california"],
        "region": "US", "mtype": "slow",
    },
    "will-donald-trump-win-the-nobel-peace-prize-in-2026-382": {
        "guardian_q": "\"nobel peace prize\"",
        "wp_keys": ["nobel"],
        "gdelt_keys": ["nobel"],
        "region": "US", "mtype": "slow",
    },
    "will-luiz-incio-lula-da-silva-win-the-2026-brazilian-presidential-election": {
        "guardian_q": "lula AND brazil",
        "wp_keys": ["lula", "brazil"],
        "gdelt_keys": ["lula"],
        "region": "elections", "mtype": "slow",
    },
    "will-jordan-bardella-win-the-2027-french-presidential-election": {
        "guardian_q": "bardella OR (france AND presidential)",
        "wp_keys": ["bardella", "france"],
        "gdelt_keys": ["bardella"],
        "region": "elections", "mtype": "slow",
    },
    "will-united-russia-er-gain-the-most-seats-in-the-next-russian-parliamentary-election": {
        "guardian_q": "russia AND (election OR duma OR parliament)",
        "wp_keys": ["russia", "duma", "election"],
        "gdelt_keys": ["united russia"],
        "region": "elections", "mtype": "slow",
    },
}
