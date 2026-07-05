"""News-agent v0b: aggregate 5 perspective-diverse estimates into (p, band) per Amendment 2.

Reads a staging JSON of raw agent outputs {key: {estimates_pct:[5], drivers, decisive_evidence}}
and writes forecasts_v0b/<key>.json with:
  p_pct       = trimmed mean (drop min & max, average middle 3) — computed HERE, never by the agent
  band        = [min, max] of the middle 3, widened to at least +/-8pp around p, clipped [1, 99]

Usage: PYTHONPATH=. uv run python scripts/newsagent_v0b_aggregate.py <staging.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FC = ROOT / "data" / "newsagent" / "v0" / "forecasts_v0b"


def aggregate(estimates: list[float]) -> tuple[float, float, float]:
    assert len(estimates) == 5, f"need 5 estimates, got {len(estimates)}"
    mid3 = sorted(float(e) for e in estimates)[1:4]
    p = sum(mid3) / 3.0
    half = max(8.0, (mid3[2] - mid3[0]) / 2.0)
    lo, hi = max(1.0, p - half), min(99.0, p + half)
    return round(p, 1), round(lo, 1), round(hi, 1)


def main() -> None:
    FC.mkdir(parents=True, exist_ok=True)
    staging = json.loads(Path(sys.argv[1]).read_text())
    for key, raw in staging.items():
        p, lo, hi = aggregate(raw["estimates_pct"])
        out = {
            "p_pct": p, "band_lo_pct": lo, "band_hi_pct": hi,
            "estimates_pct": raw["estimates_pct"],
            "decisive_evidence": raw.get("decisive_evidence"),
            "drivers": raw.get("drivers", []), "model": "claude-sonnet (v0b 5-perspective)",
        }
        (FC / f"{key}.json").write_text(json.dumps(out))
        print(f"  {key[:70]:<70} p={p:>5} [{lo},{hi}] est={raw['estimates_pct']}")
    print(f"{len(staging)} aggregated -> {FC}")


if __name__ == "__main__":
    main()
