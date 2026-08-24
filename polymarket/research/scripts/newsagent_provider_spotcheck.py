"""Spot-check Gemini 2.5 Flash Stage-A extraction against the existing cache.

Why: the Stage-B evidence weight alpha was fit on Haiku-era extraction (API
Haiku + Haiku-class attended subagents). Before the Gemini provider flag is
trusted for daily extraction, its features must agree with the cache on the
same articles — otherwise the fitted alpha silently rides on a different
feature distribution.

The check re-extracts a sample of ALREADY-CACHED (market, article) records via
Gemini into a comparison table (the cache is NEVER overwritten) and reports:
stance agreement, event-phase agreement, and mean absolute differences on
relevance / strength / clarity / tone. Output: CSV + printed summary.

Run from polymarket/research/ (needs GEMINI_API_KEY):
  PYTHONPATH=. uv run python scripts/newsagent_provider_spotcheck.py --date 2026-07-05 --n 36
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

from newsagent import config, features
from newsagent.config import CSV_OUT
from newsagent.run_daily import _load_day, day_dir


def collect_cached_items(date: str) -> list[dict]:
    """(slug, question, criteria, article) rows whose features are already cached."""
    d = day_dir(date)
    rows = []
    for slug in config.LIVE_MARKETS:
        loaded = _load_day(d, slug)
        if loaded is None:
            continue
        mkt, pkt = loaded
        for a in pkt["articles"]:
            if not a.get("title"):
                continue
            key = features.cache_key(slug, a["title"])
            if features.cached(key) is None:
                continue
            rows.append({"slug": slug, "question": mkt["question"],
                         "criteria": (mkt["description"] or "")[:500],
                         "cache_key": key, "id": key,
                         "text": features._item_text(a),
                         "domain": a.get("domain", "")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--model", default=features.GEMINI_MODEL)
    args = ap.parse_args()

    if not os.environ.get(features.GEMINI_KEY_ENV, "").strip():
        raise SystemExit(f"{features.GEMINI_KEY_ENV} not set — Justin: export the "
                         "free-tier Gemini key first.")

    rows = collect_cached_items(args.date)
    if not rows:
        raise SystemExit(f"no cached articles found for {args.date} — run the daily "
                         "fetch/extract first")
    random.seed(0)   # deterministic sample
    random.shuffle(rows)
    sample = rows[:args.n]

    by_slug: dict[str, list[dict]] = {}
    for r in sample:
        by_slug.setdefault(r["slug"], []).append(r)

    out_rows = []
    for slug, chunk_all in by_slug.items():
        for i in range(0, len(chunk_all), features.BATCH):
            chunk = chunk_all[i:i + features.BATCH]
            got = features._gemini_rows(chunk[0]["question"], chunk[0]["criteria"],
                                        chunk, args.model)
            for p in chunk:
                raw = got.get(p["cache_key"])
                if raw is None:
                    continue
                try:
                    g = features.validate(raw)
                except Exception:
                    continue
                h = features.cached(p["cache_key"])
                out_rows.append({
                    "slug": slug, "cache_key": p["cache_key"], "domain": p["domain"],
                    "stance_cache": h["stance"], "stance_gemini": g["stance"],
                    "phase_cache": h["event_phase"], "phase_gemini": g["event_phase"],
                    "d_relevance": round(abs(h["relevance"] - g["relevance"]), 3),
                    "d_strength": round(abs(h["strength"] - g["strength"]), 3),
                    "d_clarity": round(abs(h.get("clarity", 0.5) - g["clarity"]), 3),
                    "d_tone": round(abs(h.get("tone", 0.0) - g["tone"]), 3),
                })

    if not out_rows:
        raise SystemExit("Gemini returned no matchable rows — inspect the raw replies")

    CSV_OUT.mkdir(parents=True, exist_ok=True)
    path = CSV_OUT / "newsagent_provider_spotcheck.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    stance_agree = sum(r["stance_cache"] == r["stance_gemini"] for r in out_rows) / n
    phase_agree = sum(r["phase_cache"] == r["phase_gemini"] for r in out_rows) / n
    summary = {
        "n_compared": n, "model": args.model,
        "stance_agreement": round(stance_agree, 3),
        "phase_agreement": round(phase_agree, 3),
        "mean_abs_d_relevance": round(sum(r["d_relevance"] for r in out_rows) / n, 3),
        "mean_abs_d_strength": round(sum(r["d_strength"] for r in out_rows) / n, 3),
        "mean_abs_d_clarity": round(sum(r["d_clarity"] for r in out_rows) / n, 3),
        "mean_abs_d_tone": round(sum(r["d_tone"] for r in out_rows) / n, 3),
    }
    print(json.dumps(summary, indent=1))
    print(f"rows -> {path}")
    print("read: stance agreement is the load-bearing number (direction drives Stage B); "
          "alpha was fit on the cached features, so low agreement = do NOT flip the "
          "provider without a refit on Gemini-extracted history.")


if __name__ == "__main__":
    main()
