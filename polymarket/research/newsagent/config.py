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

# slug -> retrieval config (guardian_q Guardian search syntax; wp_keys any-of filter)
# Slate rule applied 2026-07-05: informative mids (10-90c), liquidity > $250k, diverse
# families. US+UK first — no informative UK binary was live at slate time (the UK
# leadership question had just resolved; revisit when the next UK market appears).
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
}
