"""Daily observatory run — hybrid FV pipeline (Stage A extract -> Stage B model).

Stages (composable; day state under data/newsagent/live/<YYYY-MM-DD>/):
  fetch       market state + full-text news packet per LIVE_MARKETS slate
  onboard     render five-perspective prior prompts for markets with no stored
              prior (out-of-band or API); ingest with --priors-file
  extract     Stage A: cache-miss article features. API path with ANTHROPIC_API_KEY
              (cheap model, batched) or out-of-band: writes extract_pending.json,
              ingest results with --features-file
  publish     Stage B fair value + band + divergence flag + FV breakdown ->
              append-only sf ledger (SF_BOOK=polymarket) -> dashboard
  backfill-fetch    reconstruct lookahead-free daily packets for the last N days
                    (Guardian/WP are timestamped by construction) + pending list
  backfill-compute  evolve the FV series over the reconstructed days (display
                    time-series; NEVER written to the ledger — the ledger is
                    forward-only)

Typical daily cron:  PYTHONPATH=. uv run python -m newsagent.run_daily --stage all
Out-of-band flow:    --stage fetch && --stage extract        (writes pending file)
                     <agents produce features>               (cheap-LLM extraction)
                     --stage extract --features-file done.json && --stage publish
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import (config, dashboard, email_ingest, engine, features, feeds, fvmodel,
               gdelt_bq, ledger, pdf_ingest, sourceweights)


def _gdelt_burst(slug: str, date: str, series_all: dict) -> dict | None:
    """Burst feature for one market-day from the cached GDELT series (None = absent)."""
    series = series_all.get(slug)
    if not series:
        return None
    return gdelt_bq.burst_z(series, date.replace("-", ""))


def _refresh_gdelt(date: str) -> None:
    """Incremental daily pull (last 20 days, one small scan). NEVER breaks the live
    path: any failure (no credential, quota, network) leaves the cached series as-is
    and the burst feature degrades to None (gamma term inert)."""
    ok, why = gdelt_bq.bigquery_available()
    if not ok:
        print(f"  gdelt: skipped ({why})")
        return
    name_keys = {s: c["gdelt_keys"] for s, c in config.LIVE_MARKETS.items()
                 if c.get("gdelt_keys")}
    if not name_keys:
        return
    start = (datetime.fromisoformat(date) - timedelta(days=20)).strftime("%Y-%m-%d")
    try:
        gdelt_bq.pull_daily_series(name_keys, start, date)
        print(f"  gdelt: series refreshed {start} -> {date} ({len(name_keys)} markets)")
    except Exception as e:
        print(f"  gdelt: refresh failed, using cached series ({e})")


def day_dir(date: str) -> Path:
    d = config.DATA / date
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_day(d: Path, slug: str) -> tuple[dict, dict] | None:
    mf, pf = d / f"{slug[:80]}.market.json", d / f"{slug[:80]}.packet.json"
    if not (mf.exists() and pf.exists()):
        return None
    return json.loads(mf.read_text()), json.loads(pf.read_text())


def stage_fetch(date: str) -> None:
    d = day_dir(date)
    _refresh_gdelt(date)
    rss_items = feeds.fetch_rss_items(date)
    print(f"  rss: {len(rss_items)} items across the feed set")
    nl_ok, nl_why = email_ingest.available()
    nl_items = email_ingest.fetch_newsletters(day=date) if nl_ok else []
    print(f"  newsletters: {len(nl_items)} items ({nl_why})" if nl_ok
          else f"  newsletters: skipped ({nl_why})")
    pdf_items = pdf_ingest.fetch_pdf_reports(date)
    print(f"  macro pdfs: {len(pdf_items)}/{len(pdf_ingest.PDF_SOURCES)} reports")
    for slug, cfg in config.LIVE_MARKETS.items():
        mkt = feeds.market_state(slug)
        if mkt["closed"]:
            print(f"  SKIP (closed): {slug} — settle its ledger entry (sf settle) and "
                  "refresh the slate in config.py")
            continue
        packet = feeds.build_packet(slug, cfg, rss_items=rss_items,
                                    newsletter_items=nl_items, pdf_items=pdf_items)
        (d / f"{slug[:80]}.market.json").write_text(json.dumps(mkt, indent=1))
        (d / f"{slug[:80]}.packet.json").write_text(json.dumps(packet, indent=1))
        print(f"  {slug[:60]}  mid={mkt['mid']:.3f}  articles={len(packet['articles'])}")


def stage_onboard(date: str, priors_file: str | None) -> None:
    """Prior p0 per market: five-perspective ensemble, set ONCE at onboarding.
    Re-anchoring later requires a documented trigger (see fvmodel docstring)."""
    d = day_dir(date)
    if priors_file:
        raw = json.loads(Path(priors_file).read_text())
        for slug, rec in raw.items():
            agg = engine.aggregate(rec["estimates_pct"])
            fvmodel.save_prior(slug, agg["p_pct"], method="five_perspective_ensemble",
                               rationale=" | ".join(rec.get("drivers", []))[:400],
                               estimates_pct=rec["estimates_pct"])
            print(f"  prior stored: {slug[:55]} p0={agg['p_pct']}%")
        return
    priors = fvmodel.load_priors()
    todo = [s for s in config.LIVE_MARKETS if s not in priors]
    if not todo:
        print("  all live markets have priors")
        return
    pdir = d / "prior_prompts"
    pdir.mkdir(exist_ok=True)
    for slug in todo:
        loaded = _load_day(d, slug)
        if loaded is None:
            print(f"  WARN no fetch for {slug} — run --stage fetch first")
            continue
        mkt, pkt = loaded
        (pdir / f"{slug[:80]}.txt").write_text(engine.build_prompt(mkt, pkt))
    print(f"  {len(todo)} prior prompts -> {pdir} (answer out-of-band, then "
          "--stage onboard --priors-file <raw.json>)")


def stage_extract(date: str, features_file: str | None,
                  provider: str | None = None) -> None:
    d = day_dir(date)
    if features_file:
        done = json.loads(Path(features_file).read_text())
        n = features.ingest_features(done, source="oob:daily")
        print(f"  ingested {n} feature records")
        return
    pending_all = []
    for slug in config.LIVE_MARKETS:
        loaded = _load_day(d, slug)
        if loaded is None:
            continue
        mkt, pkt = loaded
        pending_all.extend(features.pending_extractions(
            slug, mkt["question"], mkt["description"], pkt["articles"]))
    if not pending_all:
        print("  nothing to extract — all articles cached")
        return
    prov = provider if provider not in (None, "auto") else features.pick_provider()
    prov = None if prov == "oob" else prov
    if prov:
        by_slug: dict[str, list] = {}
        for p in pending_all:
            by_slug.setdefault(p["slug"], []).append(p)
        n = 0
        for slug, pend in by_slug.items():
            n += features.extract_via_api(pend[0]["question"], pend[0]["criteria"],
                                          pend, provider=prov)
        model = features.GEMINI_MODEL if prov == "gemini" else features.EXTRACT_MODEL
        print(f"  extracted {n} article features via API ({model})")
    else:
        out = d / "extract_pending.json"
        out.write_text(json.dumps(pending_all, indent=1))
        print(f"  {len(pending_all)} extractions pending -> {out} "
              "(no API key; process out-of-band, ingest with --features-file)")


def _compute_market(slug: str, cfg: dict, mkt: dict, pkt: dict, date: str,
                    state: dict, params: dict, p0_pct: float,
                    gdelt_series: dict | None = None) -> dict:
    """One market's Stage-B pass: S_t from NEW articles (GDELT-burst amplified),
    state step, FV, band, flag."""
    mtype = cfg.get("mtype", "shock")
    tp = fvmodel.type_params(params, mtype)
    feats = sourceweights.annotate(features.features_for(slug, pkt["articles"]))
    st = state.get(slug)
    counted = set(st.get("counted", [])) if st else set()
    new = [r for r in feats if r["cache_key"] not in counted]
    prev = {"date": st["date"], "A": st["A"]} if st else None

    burst = _gdelt_burst(slug, date, gdelt_series or {})
    vol_z = burst["vol_z"] if burst else None
    bd = fvmodel.breakdown(p0_pct, prev, date, new, mtype, params, vol_z=vol_z)
    s_t = fvmodel.amplify(fvmodel.daily_score(new), vol_z, params.get("gamma", 0.0))
    nxt = fvmodel.step_state(prev, date, s_t, tp["lam"], tp["a_clip"], tp["s_min"])
    state[slug] = {"date": nxt["date"], "A": nxt["A"],
                   "counted": (list(counted) + [r["cache_key"] for r in new])[-300:]}

    fv = fvmodel.fair_value(p0_pct, nxt["A"], params["alpha"], tp["shift_clip"])
    half = fvmodel.band_half_pp(feats, mtype, params)
    band_q = fvmodel.band_quality(feats)
    n_rel = fvmodel.n_relevant(feats)
    flag = fvmodel.divergence_flag(fv, mkt["mid"] * 100, half, n_rel,
                                   config.DIVERGENCE_GAP_PP,
                                   config.DIVERGENCE_HALF_MAX_PP,
                                   config.DIVERGENCE_NREL_MIN)
    missing = sum(1 for r in feats if r["features"] is None)
    return {"fv_pct": round(fv, 1),
            "band_lo_pct": round(max(1.0, fv - half), 1),
            "band_hi_pct": round(min(99.0, fv + half), 1),
            "half_pp": half, "n_relevant": n_rel, "mtype": mtype,
            "A": round(nxt["A"], 4), "s_t": round(s_t, 3), "divergence": flag,
            "gdelt": burst, "breakdown": bd, "missing_features": missing,
            "bias": fvmodel.source_bias_breakdown(feats),
            "band_q": band_q,
            "evidence_quality": fvmodel.evidence_quality(
                n_rel, half, config.DIVERGENCE_HALF_MAX_PP, config.DIVERGENCE_NREL_MIN,
                band_q=band_q),
            "tract": config.tract(slug),
            "tract_note": config.DATA_DRIVEN.get(slug, "")}


def stage_publish(date: str, write_ledger: bool) -> None:
    d = day_dir(date)
    params = fvmodel.load_params()
    priors = fvmodel.load_priors()
    state = fvmodel.load_state()
    gdelt_series = gdelt_bq.load_series()
    series_path = config.DATA / "fv_series.json"
    fv_series = json.loads(series_path.read_text()) if series_path.exists() else {}
    snapshots = []
    for slug, cfg in config.LIVE_MARKETS.items():
        loaded = _load_day(d, slug)
        if loaded is None:
            continue
        if slug not in priors:
            print(f"  WARN no prior for {slug} — run --stage onboard first; skipped")
            continue
        mkt, pkt = loaded
        rec = _compute_market(slug, cfg, mkt, pkt, date, state, params,
                              priors[slug]["p0_pct"], gdelt_series)
        if rec["missing_features"]:
            print(f"  WARN {rec['missing_features']} uncached articles for {slug[:50]} "
                  "(run --stage extract) — they contribute 0 evidence today")
        fc = {"p_pct": rec["fv_pct"], "band_lo_pct": rec["band_lo_pct"],
              "band_hi_pct": rec["band_hi_pct"]}
        # drivers are public copy: newsletter items appear as their generic source
        # label only (title/text/link never leave the internal packet)
        drivers = [(fvmodel._source_label(a["domain"]) + " — private analysis item")
                   if a.get("domain", "").startswith("newsletter:") else a["title"]
                   for a in rec["breakdown"]["articles"][:3]] or \
                  [f"no new qualifying evidence; prior {rec['breakdown']['p0_pct']}% "
                   "with decayed carry"]
        sf_id = ledger.log_snapshot(mkt, fc, drivers) if write_ledger else ""
        ser = [p for p in fv_series.get(slug, []) if p["date"] != date]
        ser.append({"date": date, "fv_pct": rec["fv_pct"],
                    "band_lo_pct": rec["band_lo_pct"], "band_hi_pct": rec["band_hi_pct"],
                    "mid_pct": round(mkt["mid"] * 100, 1), "segment": "live"})
        fv_series[slug] = sorted(ser, key=lambda p: p["date"])
        snapshots.append({"market": mkt, "packet": pkt, "forecast": fc,
                          "drivers": drivers, "region": cfg.get("region", ""),
                          "stage_b": rec, "sf_id": sf_id})
        print(f"  {slug[:52]}  FV={rec['fv_pct']}% [{rec['band_lo_pct']},{rec['band_hi_pct']}]"
              f"  mid={mkt['mid']*100:.1f}%  gap={rec['divergence']['gap_pp']:+}pp"
              f"  flag={'YES' if rec['divergence']['flag'] else 'no'}  ledger={sf_id or 'skipped'}")
    fvmodel.save_state(state)
    series_path.write_text(json.dumps(fv_series, indent=1))
    jpath, hpath = dashboard.publish(snapshots, fv_series)
    print(f"  dashboard -> {hpath}\n  data      -> {jpath}")


def stage_backfill_fetch(date: str, days: int) -> None:
    """Reconstruct lookahead-free daily packets for the display time-series."""
    bdir = config.DATA / "backfill"
    bdir.mkdir(parents=True, exist_ok=True)
    end = datetime.fromisoformat(date).replace(hour=12, tzinfo=timezone.utc)
    pending_all = []
    for slug, cfg in config.LIVE_MARKETS.items():
        sdir = bdir / slug[:80]
        sdir.mkdir(exist_ok=True)
        try:
            mkt = feeds.market_state(slug)
        except Exception as e:
            print(f"  WARN market_state failed for {slug}: {e}")
            continue
        for k in range(days, 0, -1):
            t = end - timedelta(days=k)
            pf = sdir / f"{t.strftime('%Y-%m-%d')}.packet.json"
            if pf.exists():
                pkt = json.loads(pf.read_text())
            else:
                pkt = feeds.build_packet(slug, cfg, now=t)
                pf.write_text(json.dumps(pkt, indent=1))
            pending_all.extend(features.pending_extractions(
                slug, mkt["question"], mkt["description"], pkt["articles"]))
        print(f"  {slug[:60]}: {days} daily packets reconstructed")
        try:
            hist = feeds.mid_history(slug, days=days + 7)
            (sdir / "mid_history.json").write_text(json.dumps(hist, indent=1))
        except Exception as e:
            print(f"  WARN mid_history failed for {slug}: {e}")
    seen, dedup = set(), []
    for p in pending_all:
        if p["cache_key"] not in seen:
            seen.add(p["cache_key"])
            dedup.append(p)
    out = bdir / "backfill_extract_pending.json"
    out.write_text(json.dumps(dedup, indent=1))
    print(f"  {len(dedup)} unique extractions pending -> {out}")


def stage_backfill_compute(date: str, days: int) -> None:
    """Evolve the FV series across reconstructed days. Display-only ('backfill'
    segment, drawn dashed + labeled on the page); the ledger never sees these
    values. Leaves fv_state positioned so today's publish continues the series."""
    bdir = config.DATA / "backfill"
    params = fvmodel.load_params()
    priors = fvmodel.load_priors()
    gdelt_series = gdelt_bq.load_series()
    end = datetime.fromisoformat(date)
    state = fvmodel.load_state()
    series_path = config.DATA / "fv_series.json"
    fv_series = json.loads(series_path.read_text()) if series_path.exists() else {}
    for slug, cfg in config.LIVE_MARKETS.items():
        if slug not in priors:
            print(f"  WARN no prior for {slug} — skipped")
            continue
        sdir = bdir / slug[:80]
        mids = {}
        mh = sdir / "mid_history.json"
        if mh.exists():
            mids = {p["date"]: p["mid"] for p in json.loads(mh.read_text())}
        mtype = cfg.get("mtype", "shock")
        tp = fvmodel.type_params(params, mtype)
        st, counted = None, set()
        series = []
        for k in range(days, 0, -1):
            t = (end - timedelta(days=k)).strftime("%Y-%m-%d")
            pf = sdir / f"{t}.packet.json"
            if not pf.exists():
                continue
            pkt = json.loads(pf.read_text())
            feats = sourceweights.annotate(features.features_for(slug, pkt["articles"]))
            new = [r for r in feats if r["cache_key"] not in counted]
            counted |= {r["cache_key"] for r in new}
            burst = _gdelt_burst(slug, t, gdelt_series)
            s_t = fvmodel.amplify(fvmodel.daily_score(new),
                                  burst["vol_z"] if burst else None,
                                  params.get("gamma", 0.0))
            st = fvmodel.step_state(st, t, s_t, tp["lam"], tp["a_clip"], tp["s_min"])
            fv = fvmodel.fair_value(priors[slug]["p0_pct"], st["A"], params["alpha"],
                                    tp["shift_clip"])
            half = fvmodel.band_half_pp(feats, mtype, params)
            entry = {"date": t, "fv_pct": round(fv, 1),
                     "band_lo_pct": round(max(1.0, fv - half), 1),
                     "band_hi_pct": round(min(99.0, fv + half), 1),
                     "segment": "backfill"}
            if t in mids:
                entry["mid_pct"] = round(mids[t] * 100, 1)
            series.append(entry)
        live_part = [p for p in fv_series.get(slug, []) if p.get("segment") == "live"]
        fv_series[slug] = series + live_part
        if st is not None:
            state[slug] = {"date": st["date"], "A": st["A"], "counted": list(counted)[-300:]}
        print(f"  {slug[:55]}: {len(series)} backfill points")
    fvmodel.save_state(state)
    series_path.write_text(json.dumps(fv_series, indent=1))
    print(f"  series -> {series_path} (backfill segment is display-only, never ledgered)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["fetch", "onboard", "extract", "publish", "all",
                                        "backfill-fetch", "backfill-compute"],
                    default="all")
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--features-file", default=None,
                    help="Stage-A features JSON produced out-of-band {cache_key: features}")
    ap.add_argument("--provider", default="auto",
                    choices=["auto", "anthropic", "gemini", "oob"],
                    help="Stage-A extraction provider (auto = env keys decide; "
                         "gemini = 2.5 Flash free tier; oob = write pending file)")
    ap.add_argument("--priors-file", default=None,
                    help="onboarding priors JSON {slug: {estimates_pct, drivers, ...}}")
    ap.add_argument("--days", type=int, default=14, help="backfill window length")
    ap.add_argument("--no-ledger", action="store_true",
                    help="skip sf ledger writes (dashboard re-render only)")
    args = ap.parse_args()

    if args.stage in ("fetch", "all"):
        stage_fetch(args.date)
    if args.stage == "onboard" or (args.stage == "all" and args.priors_file):
        stage_onboard(args.date, args.priors_file)
    if args.stage in ("extract", "all"):
        stage_extract(args.date, args.features_file, args.provider)
    if args.stage == "backfill-fetch":
        stage_backfill_fetch(args.date, args.days)
    if args.stage == "backfill-compute":
        stage_backfill_compute(args.date, args.days)
    if args.stage in ("publish", "all"):
        stage_publish(args.date, not args.no_ledger)


if __name__ == "__main__":
    sys.exit(main())
