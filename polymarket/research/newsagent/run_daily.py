"""Daily observatory run: markets -> packets -> prompts -> forecasts -> ledger -> dashboard.

Stages (composable; state persists under data/newsagent/live/<YYYY-MM-DD>/):
  fetch      pull market state + news packets for the LIVE_MARKETS slate
  prompts    render the five-perspective prompt per market (for out-of-band forecasting)
  forecast   call the Anthropic API per market (requires ANTHROPIC_API_KEY), or ingest
             --forecasts-file with raw {slug: {estimates_pct, drivers, decisive_evidence}}
  publish    aggregate -> ledger snapshot (SF_BOOK=polymarket, append-only) -> dashboard

Typical daily cron:  PYTHONPATH=. uv run python -m newsagent.run_daily --stage all
Out-of-band flow:    --stage fetch && --stage prompts   (agents produce raw JSON)
                     --stage publish --forecasts-file raw.json
Pass --no-ledger to skip ledger writes (e.g. re-render the dashboard only).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config, dashboard, engine, feeds, ledger


def day_dir(date: str) -> Path:
    d = config.DATA / date
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_fetch(date: str) -> None:
    d = day_dir(date)
    for slug, cfg in config.LIVE_MARKETS.items():
        mkt = feeds.market_state(slug)
        if mkt["closed"]:
            print(f"  SKIP (closed): {slug}")
            continue
        packet = feeds.build_packet(slug, cfg)
        (d / f"{slug[:80]}.market.json").write_text(json.dumps(mkt, indent=1))
        (d / f"{slug[:80]}.packet.json").write_text(json.dumps(packet, indent=1))
        print(f"  {slug[:60]}  mid={mkt['mid']:.3f}  articles={len(packet['articles'])}")


def stage_prompts(date: str) -> None:
    d = day_dir(date)
    for slug in config.LIVE_MARKETS:
        mf, pf = d / f"{slug[:80]}.market.json", d / f"{slug[:80]}.packet.json"
        if not (mf.exists() and pf.exists()):
            continue
        prompt = engine.build_prompt(json.loads(mf.read_text()), json.loads(pf.read_text()))
        (d / f"{slug[:80]}.prompt.txt").write_text(prompt)
    print(f"  prompts rendered in {d}")


def stage_forecast(date: str) -> None:
    d = day_dir(date)
    raw = {}
    for slug in config.LIVE_MARKETS:
        pf = d / f"{slug[:80]}.prompt.txt"
        if not pf.exists():
            continue
        print(f"  forecasting {slug[:60]} ...")
        raw[slug] = engine.anthropic_forecast(pf.read_text())
    (d / "raw_forecasts.json").write_text(json.dumps(raw, indent=1))
    print(f"  {len(raw)} forecasts -> raw_forecasts.json")


def stage_publish(date: str, forecasts_file: str | None, write_ledger: bool) -> None:
    d = day_dir(date)
    raw_path = Path(forecasts_file) if forecasts_file else d / "raw_forecasts.json"
    raw = json.loads(raw_path.read_text())
    snapshots = []
    for slug, cfg in config.LIVE_MARKETS.items():
        mf, pf = d / f"{slug[:80]}.market.json", d / f"{slug[:80]}.packet.json"
        if slug not in raw or not mf.exists():
            continue
        market = json.loads(mf.read_text())
        packet = json.loads(pf.read_text())
        fc = engine.aggregate(raw[slug]["estimates_pct"])
        drivers = raw[slug].get("drivers", [])
        sf_id = ""
        if write_ledger:
            sf_id = ledger.log_snapshot(market, fc, drivers)
        snapshots.append({"market": market, "packet": packet, "forecast": fc,
                          "drivers": drivers, "region": cfg.get("region", ""),
                          "decisive_evidence": raw[slug].get("decisive_evidence"),
                          "sf_id": sf_id})
        print(f"  {slug[:55]}  agent={fc['p_pct']}% [{fc['band_lo_pct']},{fc['band_hi_pct']}]"
              f"  mid={market['mid']*100:.1f}%  ledger={sf_id or 'skipped'}")
    jpath, hpath = dashboard.publish(snapshots)
    print(f"  dashboard -> {hpath}\n  data      -> {jpath}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["fetch", "prompts", "forecast", "publish", "all"],
                    default="all")
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--forecasts-file", default=None,
                    help="raw forecasts JSON produced out-of-band (skips the API)")
    ap.add_argument("--no-ledger", action="store_true",
                    help="skip sf ledger writes (dashboard re-render only)")
    args = ap.parse_args()

    if args.stage in ("fetch", "all"):
        stage_fetch(args.date)
    if args.stage in ("prompts", "all"):
        stage_prompts(args.date)
    if args.stage in ("forecast", "all") and not args.forecasts_file:
        stage_forecast(args.date)
    if args.stage in ("publish", "all"):
        stage_publish(args.date, args.forecasts_file, not args.no_ledger)


if __name__ == "__main__":
    sys.exit(main())
