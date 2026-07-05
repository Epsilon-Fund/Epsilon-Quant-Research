"""Live market slate + retrieval config for the calibration observatory.

Slate rule (v1): live binary politics markets, highest liquidity first, US+UK first,
near-to-mid horizon. Curated hand-list, reviewed when markets resolve. Source
weighting: Scheme B (flat curated whitelist via Guardian + Wikipedia Current Events
+ RSS) pending sign-off on Scheme A — see newsagent_repo_data_radar_findings.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "newsagent" / "live"
SHOWCASE = ROOT / "data" / "newsagent" / "showcase"
CSV_OUT = ROOT / "data" / "analysis" / "csv_outputs" / "news_agent"

GUARDIAN_KEY_ENV = "GUARDIAN_API_KEY"   # falls back to the public 'test' demo key
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"  # forecast stage requires this (or --forecasts-file)

# slug -> retrieval config (guardian_q Guardian search syntax; wp_keys any-of filter)
# Slate rule applied 2026-07-05: informative mids (10-90c), liquidity > $250k, diverse
# families. US+UK first — no informative UK binary was live at slate time (the UK
# leadership question had just resolved; revisit when the next UK market appears).
LIVE_MARKETS: dict[str, dict] = {
    "putin-out-before-2027": {
        "guardian_q": "putin AND (resign OR succession OR power OR health)",
        "wp_keys": ["putin", "russia"],
        "region": "geopolitics",
    },
    "strait-of-hormuz-traffic-returns-to-normal-by-july-31": {
        "guardian_q": "\"strait of hormuz\" AND (shipping OR traffic OR reopen OR tanker)",
        "wp_keys": ["hormuz"],
        "region": "geopolitics",
    },
    "will-there-be-no-change-in-fed-interest-rates-after-the-july-2026-meeting": {
        "guardian_q": "\"federal reserve\" AND (rates OR cut OR powell OR fomc)",
        "wp_keys": ["federal reserve", "interest rate"],
        "region": "US",
    },
    "us-x-iran-diplomatic-meeting-by-july-17-2026-20260625223459704": {
        "guardian_q": "iran AND (meeting OR talks OR diplomatic OR negotiation)",
        "wp_keys": ["iran"],
        "region": "geopolitics",
    },
    "will-the-democratic-party-control-the-house-after-the-2026-midterm-elections": {
        "guardian_q": "(democrats OR republicans) AND (house OR midterm OR congress)",
        "wp_keys": ["united states house", "midterm", "democratic party"],
        "region": "US",
    },
}
