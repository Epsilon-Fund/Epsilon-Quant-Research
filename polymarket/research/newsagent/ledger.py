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

from . import config
from .config import DATA, ROOT

SF = ROOT.parents[1] / ".agents" / "skills" / "superforecasting" / "scripts" / "sf.py"
REGISTRY = DATA / "ledger_registry.json"
METHODS = DATA / "ledger_methods.json"
ACTIVE = ROOT / "data" / "superforecast" / "forecasts" / "active.json"
EVENTS = ROOT / "data" / "superforecast" / "forecasts" / "events.jsonl"

# Per-snapshot METHOD label — DC-8 of the data-channel pre-registration, approved
# by Justin 2026-08-24. Every ledger snapshot records which method produced it,
# and the public calibration tracks are reported PER METHOD and never merged: a
# Brier earned by the news-only model must not be silently credited to a later
# news+data model (or vice versa). Today every market is METHOD_NEWS, because the
# data channel is designed and pre-registered but not built.
METHOD_NEWS = "news"
METHOD_NEWS_DATA = "news+data"


def method_for(slug: str) -> str:
    """The method label under which this market's number is currently produced."""
    return (METHOD_NEWS_DATA if slug in config.DATA_CHANNEL_MARKETS
            else METHOD_NEWS)


def _methods() -> dict:
    return json.loads(METHODS.read_text()) if METHODS.exists() else {}


def record_method(sf_id: str, slug: str, date: str) -> str:
    """Append today's method for this forecast (no-op when unchanged).

    Stored as a per-forecast history so a mid-life method switch keeps its date:
      {sf_id: {"slug": …, "history": [{"from_date": …, "method": …}, …]}}
    """
    m = method_for(slug)
    book = _methods()
    rec = book.setdefault(sf_id, {"slug": slug, "history": []})
    if not rec["history"] or rec["history"][-1]["method"] != m:
        rec["history"].append({"from_date": date, "method": m})
        METHODS.parent.mkdir(parents=True, exist_ok=True)
        METHODS.write_text(json.dumps(book, indent=1))
    return m


def method_of(sf_id: str, default: str = METHOD_NEWS) -> str:
    """Latest recorded method for a forecast (default for pre-labelling entries)."""
    rec = _methods().get(sf_id)
    if not rec or not rec.get("history"):
        return default
    return rec["history"][-1]["method"]


def _last_forecast_dates() -> dict[str, str]:
    """sf id -> date of its LAST probability-bearing event (YYYY-MM-DD).

    NOT `updated_at` in active.json: settling rewrites that field to the
    settlement instant, which would make a stale forecast look fresh (and the
    published "snapshot age" negative). The append-only event log is the truth.
    """
    if not EVENTS.exists():
        return {}
    out = {}
    for line in EVENTS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") in ("probability_set", "evidence_update"):
            out[e["id"]] = e.get("timestamp", "")[:10]
    return out


def settled_records() -> list[dict]:
    """Read-only view of every SETTLED/SCORED ledger entry, method-labelled.

    Never writes. Entries created before method labelling existed (2026-08-24)
    default to METHOD_NEWS — which is what actually produced them.
    """
    if not ACTIVE.exists():
        return []
    active = json.loads(ACTIVE.read_text())
    reg = {v: k for k, v in _registry().items()}
    last_fc = _last_forecast_dates()
    out = []
    for sf_id, r in active.items():
        if r.get("outcome") is None or r.get("final_probability") is None:
            continue
        p_, y = float(r["final_probability"]), int(r["outcome"])
        out.append({
            "sf_id": sf_id, "slug": reg.get(sf_id, ""),
            "question": r.get("canonical_question") or r.get("raw_question", ""),
            "final_p": p_, "outcome": y,
            "brier": round(r.get("brier", (p_ - y) ** 2), 6),
            "resolution_date": r.get("resolution_date", ""),
            "settled_at": (r.get("settled_at") or "")[:10],
            "last_update": last_fc.get(sf_id, (r.get("updated_at") or "")[:10]),
            "method": method_of(sf_id),
        })
    return sorted(out, key=lambda r: (r["resolution_date"], r["sf_id"]))


def _run(*args: str) -> str:
    env = dict(os.environ, SF_BOOK="polymarket")
    res = subprocess.run(["python3", str(SF), *args], capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"sf {' '.join(args[:2])} failed: {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout


def _registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def log_snapshot(market: dict, forecast: dict, drivers: list[str],
                 date: str = "") -> str:
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
        record_method(sf_id, market["slug"], date)
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
    record_method(sf_id, market["slug"], date)
    return sf_id


def settle(slug: str, outcome_yes: int) -> None:
    reg = _registry()
    _run("settle", reg[slug], "--outcome", str(outcome_yes))
