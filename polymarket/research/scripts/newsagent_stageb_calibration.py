"""Stage-B calibration on the v0 gate archive (resolved outcomes, lookahead-free).

The v0 gate left behind ~56 news packets across 10 RESOLVED politics markets
(data/newsagent/v0/news/<slug>__<date>.json) plus the universe table with outcomes
and snapshot mids. That is a real — small — calibration set for the Stage-B evidence
weight alpha: features are extracted from the archived packets (Stage A, cached),
the evidence state A_t is evolved across snapshots, and alpha is grid-fit to
minimize pooled Brier against resolved outcomes.

Honesty: n ≈ 50 pairs / 10 markets / ~6 event families in ONE geopolitically
extreme month. The fit is a starting point, refit as the forward ledger settles;
it is NOT a validated edge and the market mid is never a fitting target.

Flow (out-of-band friendly, no API key needed):
  --emit-work   write stageb_extract_pending.json + stageb_prior_prompts/ for
                subagent processing (features per article; five-perspective prior
                per market from its FIRST snapshot packet only).
  --ingest      --features-file F [--priors-file P]: validate + cache features;
                store archive priors (separate from the live priors file).
  --fit         evolve A_t per market, fit alpha, write fv_params.json + CSVs + plot.

Run from polymarket/research/:  PYTHONPATH=. uv run python scripts/newsagent_stageb_calibration.py --emit-work
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from newsagent import config, engine, features, fvmodel, gdelt_bq
from newsagent.config import CSV_OUT, DATA, ROOT

V0 = ROOT / "data" / "newsagent" / "v0"
WORK = DATA / "stageb_work"
ARCHIVE_PRIORS = WORK / "archive_priors.json"
PLOTS = ROOT / "data" / "analysis" / "plots" / "news_agent"

# Market type by family (declared, mirrors config.LIVE_MARKETS tagging):
# election questions = slow/structural; everything else in this archive is
# event-driven geopolitics/personnel = shock.
SLOW_FAMILIES = {"colombia_election"}

# GDELT AllNames AND-substring keys per ARCHIVE market (live keys live in
# config.LIVE_MARKETS["gdelt_keys"]). Iran-family markets share one attention
# series by design — the burst feature is a per-market z-score vs its own
# trailing baseline, so shared series still burst correctly.
ARCHIVE_GDELT_KEYS = {
    "us-x-iran-permanent-peace-deal-by-june-15-2026-734-856-129": ["iran", "united states"],
    "iran-agrees-to-end-enrichment-of-uranium-by-june-30": ["iran"],
    "starmer-out-by-june-30-2026-862-594-548-219-739-726-569-741-645": ["keir starmer"],
    "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election": ["cepeda"],
    "will-abelardo-de-la-espriella-win-the-2026-colombian-presidential-election": ["espriella"],
    "will-trump-agree-to-withdraw-troops-from-the-iranian-region-by-june-30": ["iran", "united states"],
    "israel-closes-its-airspace-by-june-15-687-594-783-732-455-613-653": ["israel"],
    "aleksandar-vui-out-as-serbian-president-by-june-30-2026-398": ["vucic"],
    "israel-closes-its-airspace-by-june-30-324-253-464-332-827-713-671-846": ["israel"],
}

SNAPSHOTS = ["2026-06-08", "2026-06-11", "2026-06-14", "2026-06-17",
             "2026-06-20", "2026-06-23", "2026-06-26", "2026-06-29"]


def load_universe() -> list[dict]:
    rows = []
    with open(CSV_OUT / "newsagent_v0_universe.csv") as f:
        for r in csv.DictReader(f):
            if r["selected"] != "True":
                continue
            rows.append(r)
    return rows


def packet_path(slug: str, date: str) -> Path:
    return V0 / "news" / f"{slug[:80]}__{date}.json"


def load_packet(slug: str, date: str) -> dict | None:
    p = packet_path(slug, date)
    return json.loads(p.read_text()) if p.exists() else None


def cmd_emit_work() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "prior_prompts").mkdir(exist_ok=True)
    universe = load_universe()
    pending_all, n_cached = [], 0
    for m in universe:
        slug, question = m["slug"], m["question"]
        first_packet = None
        for d in SNAPSHOTS:
            pkt = load_packet(slug, d)
            if pkt is None:
                continue
            if first_packet is None:
                first_packet = (d, pkt)
            pend = features.pending_extractions(slug, question, question, pkt["articles"])
            n_cached += len(pkt["articles"]) - len(pend)
            pending_all.extend(pend)
        # onboarding prior: five-perspective prompt from the FIRST snapshot packet
        # only (lookahead-free: the prior may not see later news).
        if first_packet is not None:
            d, pkt = first_packet
            market = {"question": question, "description": question,
                      "end_date": m["end_date"] + "T00:00:00Z"}
            from datetime import datetime, timezone
            asof = datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
            prompt = engine.build_prompt(market, pkt, asof=asof)
            (WORK / "prior_prompts" / f"{slug[:80]}.txt").write_text(prompt)
    # dedupe by cache_key (same article across snapshots/windows)
    seen, dedup = set(), []
    for p in pending_all:
        if p["cache_key"] in seen:
            continue
        seen.add(p["cache_key"])
        dedup.append(p)
    (WORK / "stageb_extract_pending.json").write_text(json.dumps(dedup, indent=1))
    print(f"work: {len(dedup)} unique (market, article) extractions pending "
          f"({n_cached} already cached); {len(universe)} prior prompts in {WORK / 'prior_prompts'}")


def cmd_ingest(features_file: str | None, priors_file: str | None) -> None:
    if features_file:
        done = json.loads(Path(features_file).read_text())
        n = features.ingest_features(done, source="oob:stageb_calibration")
        print(f"ingested {n} feature records into the cache")
    if priors_file:
        raw = json.loads(Path(priors_file).read_text())
        priors = {}
        for slug, rec in raw.items():
            agg = engine.aggregate(rec["estimates_pct"])
            priors[slug] = {"p0_pct": agg["p_pct"], "estimates_pct": rec["estimates_pct"],
                            "drivers": rec.get("drivers", [])}
        ARCHIVE_PRIORS.write_text(json.dumps(priors, indent=1))
        print(f"stored {len(priors)} archive priors -> {ARCHIVE_PRIORS}")


def cmd_pull_gdelt(end: str) -> None:
    """One consolidated scan for archive + live name-sets (dry-run guarded)."""
    name_keys = dict(ARCHIVE_GDELT_KEYS)
    for slug, cfg in config.LIVE_MARKETS.items():
        if cfg.get("gdelt_keys"):
            name_keys[slug] = cfg["gdelt_keys"]
    # start early enough for a 14d trailing baseline before the first archive snapshot
    cache = gdelt_bq.pull_daily_series(name_keys, "2026-05-25", end)
    n_days = {s: len(v) for s, v in cache.items() if s in name_keys}
    print(f"pulled {len(name_keys)} name-sets -> {gdelt_bq.SERIES_CACHE}")
    print("  days per market:", sorted(set(n_days.values())))


def build_pairs(gamma: float = 0.0, gdelt_series: dict | None = None) -> list[dict]:
    universe = load_universe()
    priors = json.loads(ARCHIVE_PRIORS.read_text())
    params = fvmodel.load_params()
    pairs = []
    for m in universe:
        slug = m["slug"]
        if slug not in priors:
            print(f"  WARN no prior for {slug} — skipped")
            continue
        p0 = priors[slug]["p0_pct"]
        mtype = "slow" if m["family"] in SLOW_FAMILIES else "shock"
        tp = fvmodel.type_params(params, mtype)
        y = int(m["outcome_yes"])
        series = (gdelt_series or {}).get(slug, {})
        state, counted = None, set()
        for d in SNAPSHOTS:
            pkt = load_packet(slug, d)
            if pkt is None:
                continue
            feats = features.features_for(slug, pkt["articles"])
            new = [r for r in feats if r["cache_key"] not in counted]
            counted |= {r["cache_key"] for r in new}
            s_t = fvmodel.daily_score(new)
            bz = gdelt_bq.burst_z(series, d.replace("-", "")) if series else None
            vol_z = bz["vol_z"] if bz else None
            s_eff = fvmodel.amplify(s_t, vol_z, gamma)
            state = fvmodel.step_state(state, d, s_eff, tp["lam"], tp["a_clip"], tp["s_min"])
            mid = m.get(f"mid_{d}", "")
            if mid in ("", None):  # market already closed at this snapshot
                continue
            pairs.append({"slug": slug, "family": m["family"], "date": d,
                          "mtype": mtype, "p0_pct": p0, "A": state["A"],
                          "y": y, "mid": float(mid),
                          "vol_z": vol_z, "shift_clip": tp["shift_clip"],
                          "n_missing_feats": sum(1 for r in feats if r["features"] is None)})
    return pairs


def cmd_fit() -> None:
    # gamma is DECLARED (fv_params/DEFAULT), never grid-chosen: on this small
    # shock-month sample the in-sample Brier improves MONOTONICALLY with gamma
    # (0.459 -> 0.42 at gamma=4, no plateau) — that is burst-saturation overfit,
    # not a measurable elasticity. Only alpha is fitted; the gamma sensitivity
    # curve is printed as a diagnostic.
    gdelt_series = gdelt_bq.load_series()
    gamma = fvmodel.load_params().get("gamma", 0.0) if gdelt_series else 0.0
    pairs = build_pairs(gamma=gamma, gdelt_series=gdelt_series)
    fit = fvmodel.fit_alpha(pairs)
    alpha = fit["alpha"]
    if gdelt_series:
        diag = []
        for g in (0.0, 0.5, 1.0, 1.5, 2.0):
            f = fvmodel.fit_alpha(build_pairs(gamma=g, gdelt_series=gdelt_series))
            diag.append(f"g={g}: alpha*={f['alpha']} brier={f['brier_at_best']}")
        print("gamma sensitivity (diagnostic only, gamma stays declared):")
        for d in diag:
            print("  " + d)
    print(f"declared gamma={gamma}; fitted alpha*={alpha}, brier={fit['brier_at_best']}")
    missing = sum(p["n_missing_feats"] for p in pairs)
    if missing:
        print(f"  WARN {missing} article-features missing from cache across pairs")

    # context columns (the mid is context, NEVER the fitting target)
    by_family = defaultdict(list)
    for p in pairs:
        fv = fvmodel.fair_value(p["p0_pct"], p["A"], alpha, p.get("shift_clip")) / 100.0
        p["fv"] = round(fv, 4)
        p["brier_fv"] = round((fv - p["y"]) ** 2, 4)
        p["brier_prior"] = round((p["p0_pct"] / 100.0 - p["y"]) ** 2, 4)
        p["brier_mid_context"] = round((p["mid"] - p["y"]) ** 2, 4)
        by_family[p["family"]].append(p)

    params = fvmodel.load_params()
    params.update({"alpha": alpha, "gamma": gamma, "fitted_on": "v0_archive_2026-06",
                   "n_pairs": fit["n_pairs"], "n_markets": fit["n_markets"],
                   "notes": ("alpha+gamma grid-fit on v0-archive resolved outcomes "
                             "(one geopolitically extreme month; refit as the "
                             "forward ledger settles). lambda/floors declared, not fit; "
                             "gamma=GDELT burst amplification (0 when series absent).")})
    fvmodel.save_params(params)

    CSV_OUT.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT / "newsagent_stageb_pairs.csv", "w", newline="") as f:
        cols = ["slug", "family", "date", "mtype", "p0_pct", "A", "fv", "y", "mid",
                "brier_fv", "brier_prior", "brier_mid_context", "vol_z", "n_missing_feats"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(pairs)
    with open(CSV_OUT / "newsagent_stageb_fit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "brier"])
        for row in fit["curve"]:
            w.writerow([row["alpha"], row["brier"]])

    fam_rows = []
    for fam, rows in sorted(by_family.items()):
        fam_rows.append({"family": fam, "n": len(rows),
                         "brier_fv": round(sum(r["brier_fv"] for r in rows) / len(rows), 4),
                         "brier_prior": round(sum(r["brier_prior"] for r in rows) / len(rows), 4),
                         "brier_mid_context": round(sum(r["brier_mid_context"] for r in rows) / len(rows), 4)})
    with open(CSV_OUT / "newsagent_stageb_family.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fam_rows[0].keys()))
        w.writeheader()
        w.writerows(fam_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["alpha"] for r in fit["curve"]]
        ys = [r["brier"] for r in fit["curve"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys)
        ax.axvline(alpha, ls="--", color="tab:green", label=f"alpha*={alpha}")
        ax.axhline(fit["brier_prior_only"], ls=":", color="grey",
                   label=f"prior-only {fit['brier_prior_only']}")
        ax.set_xlabel("alpha (evidence weight)")
        ax.set_ylabel("pooled Brier vs resolved outcomes")
        ax.set_title(f"Stage-B fit — v0 archive ({fit['n_pairs']} pairs, "
                     f"{fit['n_markets']} markets), gamma*={gamma}")
        ax.legend()
        PLOTS.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(PLOTS / "newsagent_stageb_alpha_fit.png", dpi=120)
        print(f"plot -> {PLOTS / 'newsagent_stageb_alpha_fit.png'}")
    except ImportError:
        print("matplotlib unavailable — skipped the fit plot")

    print(json.dumps({k: v for k, v in fit.items() if k != "curve"}, indent=1))
    print(f"params -> {fvmodel.PARAMS_PATH}")
    for r in fam_rows:
        print(f"  {r['family']:>22}  n={r['n']:>2}  fv={r['brier_fv']:.3f}  "
              f"prior={r['brier_prior']:.3f}  mid(ctx)={r['brier_mid_context']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-work", action="store_true")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--features-file")
    ap.add_argument("--priors-file")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--pull-gdelt", action="store_true",
                    help="one consolidated GKG scan for archive+live name-sets")
    ap.add_argument("--end", default=None, help="end date for --pull-gdelt (YYYY-MM-DD)")
    args = ap.parse_args()
    if args.emit_work:
        cmd_emit_work()
    if args.ingest:
        cmd_ingest(args.features_file, args.priors_file)
    if args.pull_gdelt:
        from datetime import datetime, timezone
        cmd_pull_gdelt(args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    if args.fit:
        cmd_fit()
    if not (args.emit_work or args.ingest or args.fit or args.pull_gdelt):
        print("nothing to do — pass --emit-work / --ingest / --pull-gdelt / --fit")


if __name__ == "__main__":
    main()
