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

Source weighting: Scheme A is live (RSP reliability tiers + Iffy blocklist, composed
with the AllSides-seeded lean multiplier). The uncovered-source weights were APPROVED
2026-08-24 and are live from that date — see newsagent/sourceweights.py.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "newsagent" / "live"
SHOWCASE = ROOT / "data" / "newsagent" / "showcase"
CSV_OUT = ROOT / "data" / "analysis" / "csv_outputs" / "news_agent"

GUARDIAN_KEY_ENV = "GUARDIAN_API_KEY"   # falls back to the public 'test' demo key
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"  # API stages need this (or the out-of-band files)

# ---------------------------------------------------------------------------
# Credential loading (approved 2026-08-24, data-channel scoping sign-off rows 3-4)
# ---------------------------------------------------------------------------
# Keys live in the git-ignored polymarket/research/.env (see .gitignore line
# "polymarket/research/.env") so an attended run does not depend on what happens
# to be exported in the calling shell. The real environment always WINS over the
# file — export it and the file is ignored for that key.
#
# NEVER print, log, or write a key value. load_env() returns NAMES only, and the
# callers below print names only. The OpenBB keys are read by the (not yet built)
# offline data-channel ingest script, which keeps OpenBB out of newsagent/* per
# the AGPL boundary rule Justin acknowledged on 2026-08-24.
ENV_FILE = ROOT / ".env"
ENV_KEYS = (
    "GUARDIAN_API_KEY",            # Guardian Open Platform (else the 'test' demo key)
    "ANTHROPIC_API_KEY",           # Stage-A extraction / priors via API
    "GEMINI_API_KEY",              # Stage-A extraction, Gemini 2.5 Flash provider flag
    "GOOGLE_APPLICATION_CREDENTIALS",   # GDELT BigQuery service account (path, not a key)
    "OPENBB_FRED_API_KEY",         # data channel (offline ingest only)
    "OPENBB_BLS_API_KEY",          # data channel (offline ingest only)
)


def load_env(path: Path | None = None, override: bool = False) -> list[str]:
    """Load KEY=VALUE lines from the git-ignored .env into os.environ.

    Returns the NAMES of the keys it set (never values). Lines that are blank,
    commented (#), or malformed are skipped; surrounding quotes are stripped.
    An absent .env is not an error — every consumer already degrades (Guardian
    falls back to the demo key, GDELT skips, Stage A writes a pending file).
    """
    f = path or ENV_FILE
    if not f.exists():
        return []
    loaded = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if val.startswith("#"):      # "KEY=   # not pasted yet" -> treat as unset
            val = ""
        val = val.strip('"').strip("'")
        if not key or not val:
            continue
        if key in os.environ and not override:
            continue
        # A credential PATH in .env is written relative to the .env itself, so the
        # run works from any cwd (GDELT would otherwise silently skip).
        if key.endswith("_CREDENTIALS") and val and not Path(val).is_absolute():
            val = str((f.parent / val).resolve())
        os.environ[key] = val
        loaded.append(key)
    return loaded


# Loaded at import so every entry point (run_daily, the backfill/fit script, the
# dashboard re-render, tests) sees the same credentials without a wrapper.
_LOADED_ENV_KEYS = load_env()

# Divergence flag (public, informational): |FV - mid| >= GAP AND band half-width <=
# HALF AND >= NREL relevant articles in 72h. 15pp ~= 2x the v0 median |gap|; the
# confidence leg keeps the flag off thin/uncertain evidence. Never an edge claim.
DIVERGENCE_GAP_PP = 15.0
DIVERGENCE_HALF_MAX_PP = 12.0
DIVERGENCE_NREL_MIN = 5

# ---------------------------------------------------------------------------
# Tractability tagging — v3.1 tag, SPLIT 2026-08-24 (Justin APPROVED row 7 of the
# data-channel scoping sign-off, [[newsagent_data_channel_scoping]] § 7).
# ---------------------------------------------------------------------------
# A market is NOT news-tractable when its dominant information channel is not
# news text. The single v3.1 DATA_DRIVEN tag lumped two genuinely different kinds
# of blindness together; the split is an honesty fix and is independent of
# whether the data channel is ever built:
#
#   data-driven  — official statistics exist for it: free, complete, and
#       vintage-stamped (FRED/BLS + the Cleveland Fed nowcast + FOMC
#       projections). A second evidence channel is BUILDABLE at $0 and is
#       pre-registered (DC-1…DC-8, Option C: structural probability as a
#       re-anchorable prior; market-implied odds display-only). NOT BUILT YET —
#       until it is, these cards stay honestly blind.
#   poll-driven  — the dominant channel is private/campaign or ballot-issue
#       POLLING. That is not an official statistic and is not purchasable at any
#       tier (checked 2026-08-24). No fix is promised for these.
#
# Both stay scored in public — that is the honest-measurement point. Their cards
# say so. BACKLOG (unchanged): a rates-via-options econ track as a separate,
# labeled method.
DATA_DRIVEN: dict[str, str] = {
    "will-there-be-no-change-in-fed-interest-rates-after-the-september-2026-meeting-615":
        "Rate decisions are priced off Fed-funds futures/options-implied odds; "
        "news text adds little beyond what those markets already carry. Official "
        "statistics (CPI/PCE/labour + Cleveland Fed nowcast + FOMC projections) "
        "do map onto this question — a data-evidence channel is designed and "
        "pre-registered but not yet built.",
}

POLL_DRIVEN: dict[str, str] = {
    "will-xavier-becerra-win-the-california-governor-election-in-2026":
        "General-election polling drives this race — private/campaign polling "
        "that never reaches the news packet. California publishes no official "
        "statistic that tracks it, and no free or paid provider carries the "
        "polling either; the market carries information we cannot see.",
    "billionaire-one-time-wealth-tax-passes-in-california-election-2026":
        "Ballot-measure odds ride issue polling, not news coverage; our packet "
        "sees the campaign noise, not the poll numbers. No official statistic "
        "and no data provider covers it.",
}

# Merged view — every not-news-tractable market and its public note.
NOT_NEWS_TRACTABLE: dict[str, str] = {**DATA_DRIVEN, **POLL_DRIVEN}


# Markets whose published number is produced by the news+data method (the
# data-evidence channel of [[newsagent_data_channel_scoping]] § 4). EMPTY until
# that channel is actually built: DC-8 requires every ledger entry to record the
# method that produced it, and per-method calibration tracks must never merge, so
# a market only joins this set on the day its published number really changes
# method — never in advance.
# LIVE from 2026-08-24: the September Fed market joined on the day the v3
# retro-test passed its pre-registered bars (GO on 40 resolved FOMC decisions
# across three rate regimes — see newsagent_data_channel_v3_findings). Its
# published number is now anchored on p_struct rather than on its onboarding
# prior, so `ledger.method_for` labels its snapshots `news+data` from today and
# its calibration track starts at n=0, separate from the news track forever.
DATA_CHANNEL_MARKETS: frozenset[str] = frozenset({
    "will-there-be-no-change-in-fed-interest-rates-after-the-september-2026-meeting-615",
})


def tract_note(slug: str) -> str:
    """Public card note for a not-news-tractable market ('' when news-driven)."""
    return NOT_NEWS_TRACTABLE.get(slug, "")


def tract(slug: str) -> str:
    """news | data | poll — tractability tag (see the split above)."""
    if slug in DATA_DRIVEN:
        return "data"
    if slug in POLL_DRIVEN:
        return "poll"
    return "news"


# slug -> retrieval config (guardian_q Guardian search syntax; wp_keys any-of filter;
# gdelt_keys AND-substring match on GKG AllNames; mtype slow/shock; region display tag)
# v3 slate (2026-07-05, discovery via scripts/newsagent_v3_universe.py): 24 markets,
# informative mids (5-95c), liquidity >= $100k, <= 2 per event family, curated by hand.
# US+UK first — still NO informative UK binary live (Starmer family resolved; the
# discovery sweep found none above the liquidity floor); French/Brazil/Russia
# elections add slow-type diversity meanwhile. Revisit each slate refresh.
LIVE_MARKETS: dict[str, dict] = {
    "putin-out-before-2027": {
        # Polymarket RE-SLUGGED this market to "putin-out-before-2027-346" some time
        # after 2026-07-05 (the old slug now returns 0 rows from Gamma, closed=true
        # included). The KEY here stays the original slug because every piece of our
        # state is keyed by it — sf-2026-001 in the ledger registry, the stored prior,
        # fv_state/fv_series, the GDELT series and the Stage-A feature cache. Only the
        # Gamma lookup follows the rename, via gamma_slug.
        "gamma_slug": "putin-out-before-2027-346",
        "guardian_q": "putin AND (resign OR succession OR power OR health)",
        "wp_keys": ["putin", "russia"],
        "gdelt_keys": ["vladimir putin"],
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
    # ---- 2026-08-24 slate refresh: replacements for the four July resolutions --
    # Rule as before: informative mid (5-95c), liquidity >= $100k, <= 2 per event
    # family, curated by hand from scripts/newsagent_v3_universe.py. Hormuz-Oct is
    # the direct successor of the resolved Hormuz-Jul (family: hormuz 2/2 with the
    # December market); the Clarity Act opens a new family (US legislation) and is
    # the slate's first non-election, non-geopolitics question; Le Pen joins
    # Bardella (france 2/2) and Bolsonaro joins Lula (brazil 2/2) — same election,
    # different candidate, so neither pair is a 1-p mirror.
    "strait-of-hormuz-traffic-returns-to-normal-by-october-31-20260810151043583": {
        "guardian_q": "\"strait of hormuz\" AND (shipping OR traffic OR reopen OR tanker)",
        "wp_keys": ["hormuz"],
        "gdelt_keys": ["strait of hormuz"],
        "region": "geopolitics", "mtype": "shock",
    },
    "clarity-act-signed-into-law-in-2026": {
        "guardian_q": "\"clarity act\" OR (cryptocurrency AND (congress OR senate OR legislation OR bill))",
        "wp_keys": ["clarity act", "cryptocurrency", "congress"],
        "gdelt_keys": ["clarity act"],
        "region": "US", "mtype": "slow",
    },
    "will-marine-le-pen-win-the-2027-french-presidential-election": {
        "guardian_q": "\"le pen\" OR (france AND presidential)",
        "wp_keys": ["le pen", "france"],
        "gdelt_keys": ["marine le pen"],
        "region": "elections", "mtype": "slow",
    },
    "will-flvio-bolsonaro-win-the-2026-brazilian-presidential-election": {
        "guardian_q": "bolsonaro AND brazil",
        "wp_keys": ["bolsonaro", "brazil"],
        "gdelt_keys": ["bolsonaro"],
        "region": "elections", "mtype": "slow",
    },
}


# Markets retired from the slate because they RESOLVED. Kept here as provenance:
# the ledger entry is settled and scored (append-only, never edited), the market
# is gone from LIVE_MARKETS, and the row below records why. Verified on Gamma
# (closed-market slug lookups need closed=true) on the retirement date.
RETIRED_MARKETS: dict[str, dict] = {
    "us-x-iran-diplomatic-meeting-by-july-17-2026-20260625223459704": {
        "sf_id": "sf-2026-004", "resolved": "2026-07-18", "outcome": "NO",
        "retired": "2026-08-24"},
    "will-there-be-no-change-in-fed-interest-rates-after-the-july-2026-meeting": {
        "sf_id": "sf-2026-003", "resolved": "2026-07-29", "outcome": "YES",
        "retired": "2026-08-24"},
    "strait-of-hormuz-traffic-returns-to-normal-by-july-31": {
        "sf_id": "sf-2026-002", "resolved": "2026-08-04", "outcome": "NO",
        "retired": "2026-08-24"},
    "will-iran-announce-withdrawal-from-mou-negotiations-by-july-31-20260622191733846": {
        "sf_id": "sf-2026-009", "resolved": "2026-08-01", "outcome": "NO",
        "retired": "2026-08-24",
        "note": "Polymarket also re-slugged this one; the resolved market is "
                "…-20260622191733846-586-829-787."},
}
