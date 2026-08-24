"""Append-only ledger integration (vendored superforecasting skill, SF_BOOK=polymarket).

One sf forecast per market: created on first sighting (new -> scope -> set-prob),
then `sf update` on subsequent daily snapshots. The sf state machine enforces
anti-post-hoc (settled forecasts reject edits); settlement happens via `sf settle`
when the market resolves. This module never edits ledger files directly — all
writes go through the sf CLI. Local registry maps market slug -> sf id.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .config import DATA, ROOT

SF = ROOT.parents[1] / ".agents" / "skills" / "superforecasting" / "scripts" / "sf.py"
REGISTRY = DATA / "ledger_registry.json"


def _run(*args: str) -> str:
    env = dict(os.environ, SF_BOOK="polymarket")
    res = subprocess.run(["python3", str(SF), *args], capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"sf {' '.join(args[:2])} failed: {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout


def _registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def log_snapshot(market: dict, forecast: dict, drivers: list[str]) -> str:
    """Create-or-update the sf forecast for this market with today's snapshot."""
    DATA.mkdir(parents=True, exist_ok=True)
    reg = _registry()
    p = forecast["p_pct"] / 100.0
    lo, hi = forecast["band_lo_pct"] / 100.0, forecast["band_hi_pct"] / 100.0
    reason = "news-agent daily snapshot; drivers: " + " | ".join(drivers[:3])

    if market["slug"] in reg:
        sf_id = reg[market["slug"]]
        _run("update", sf_id, "--evidence", reason, "--p", f"{p:.3f}",
             "--range", f"{lo:.3f}", f"{hi:.3f}")
        return sf_id

    out = _run("new", market["question"])
    m = re.search(r"\bsf-[A-Za-z0-9_-]+\b", out)
    if not m:
        raise RuntimeError(f"could not parse sf id from: {out[:200]!r}")
    sf_id = m.group(0)
    _run("scope", sf_id, "--canonical", market["question"],
         "--resolution-date", market["end_date"][:10],
         "--criterion", (market["description"][:400] or "per Polymarket resolution"),
         "--outcome-type", "binary", "--data-source", f"polymarket:{market['slug']}")
    _run("set-prob", sf_id, "--p", f"{p:.3f}", "--range", f"{lo:.3f}", f"{hi:.3f}",
         "--reason", reason)
    reg[market["slug"]] = sf_id
    REGISTRY.write_text(json.dumps(reg, indent=1))
    return sf_id


def settle(slug: str, outcome_yes: int) -> None:
    reg = _registry()
    _run("settle", reg[slug], "--outcome", str(outcome_yes))
