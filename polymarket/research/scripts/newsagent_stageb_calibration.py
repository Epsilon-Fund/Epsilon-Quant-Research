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

from newsagent import engine, features, fvmodel
from newsagent.config import CSV_OUT, DATA, ROOT

V0 = ROOT / "data" / "newsagent" / "v0"
WORK = DATA / "stageb_work"
ARCHIVE_PRIORS = WORK / "archive_priors.json"
PLOTS = ROOT / "data" / "analysis" / "plots" / "news_agent"

# Market type by family (declared, mirrors config.LIVE_MARKETS tagging):
# election questions = slow/structural; everything else in this archive is
# event-driven geopolitics/personnel = shock.
SLOW_FAMILIES = {"colombia_election"}

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


def build_pairs() -> list[dict]:
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
        state, counted = None, set()
        for d in SNAPSHOTS:
            pkt = load_packet(slug, d)
            if pkt is None:
                continue
            feats = features.features_for(slug, pkt["articles"])
            new = [r for r in feats if r["cache_key"] not in counted]
            counted |= {r["cache_key"] for r in new}
            s_t = fvmodel.daily_score(new)
            state = fvmodel.step_state(state, d, s_t, tp["lam"], tp["a_clip"], tp["s_min"])
            mid = m.get(f"mid_{d}", "")
            if mid in ("", None):  # market already closed at this snapshot
                continue
            pairs.append({"slug": slug, "family": m["family"], "date": d,
                          "mtype": mtype, "p0_pct": p0, "A": state["A"],
                          "y": y, "mid": float(mid),
                          "n_missing_feats": sum(1 for r in feats if r["features"] is None)})
    return pairs


def cmd_fit() -> None:
    pairs = build_pairs()
    missing = sum(p["n_missing_feats"] for p in pairs)
    if missing:
        print(f"  WARN {missing} article-features missing from cache across pairs")
    fit = fvmodel.fit_alpha(pairs)
    alpha = fit["alpha"]

    # context columns (the mid is context, NEVER the fitting target)
    by_family = defaultdict(list)
    for p in pairs:
        fv = fvmodel.fair_value(p["p0_pct"], p["A"], alpha) / 100.0
        p["fv"] = round(fv, 4)
        p["brier_fv"] = round((fv - p["y"]) ** 2, 4)
        p["brier_prior"] = round((p["p0_pct"] / 100.0 - p["y"]) ** 2, 4)
        p["brier_mid_context"] = round((p["mid"] - p["y"]) ** 2, 4)
        by_family[p["family"]].append(p)

    params = fvmodel.load_params()
    params.update({"alpha": alpha, "fitted_on": "v0_archive_2026-06",
                   "n_pairs": fit["n_pairs"], "n_markets": fit["n_markets"],
                   "notes": ("alpha grid-fit on v0-archive resolved outcomes "
                             "(one geopolitically extreme month; refit as the "
                             "forward ledger settles). lambda/floors declared, not fit.")})
    fvmodel.save_params(params)

    CSV_OUT.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT / "newsagent_stageb_pairs.csv", "w", newline="") as f:
        cols = ["slug", "family", "date", "mtype", "p0_pct", "A", "fv", "y", "mid",
                "brier_fv", "brier_prior", "brier_mid_context", "n_missing_feats"]
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
        ax.set_title(f"Stage-B alpha fit — v0 archive ({fit['n_pairs']} pairs, "
                     f"{fit['n_markets']} markets)")
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
    args = ap.parse_args()
    if args.emit_work:
        cmd_emit_work()
    if args.ingest:
        cmd_ingest(args.features_file, args.priors_file)
    if args.fit:
        cmd_fit()
    if not (args.emit_work or args.ingest or args.fit):
        print("nothing to do — pass --emit-work / --ingest / --fit")


if __name__ == "__main__":
    main()
