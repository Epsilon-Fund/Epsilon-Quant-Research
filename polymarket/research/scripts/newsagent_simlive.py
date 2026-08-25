"""Simulated-live reconstruction — what the Observatory WOULD have published on
every resolved market in the backfill universe, scored against the outcome.

    THIS IS NOT THE FORWARD TRACK RECORD.

The forward record is the append-only superforecasting ledger (SF_BOOK=polymarket):
numbers published before the fact, settled by `sf settle`, currently n=4. This
script builds a DIFFERENT and WEAKER object — a lookahead-free RECONSTRUCTION of
what the current model would have shown on markets that have already resolved.
The two are never summed, never averaged together, and never written to the same
file. Reconstruction rows are labelled "reconstruction" in every artifact they
reach, and nothing here is written to the ledger.

Machinery reused as-is from scripts/newsagent_hist_backfill.py (the v3.1 302-packet
lookahead-free backfill): the resolved universe, the reconstructed snapshot packets,
the canaried onboarding priors, the GDELT-GKG attention series and the CLOB mid
history. This script adds the trajectory replay and the scoring readout.

===========================================================================
PRE-REGISTRATION — LOCKED 2026-08-25, BEFORE ANY NUMBER IN THIS PASS EXISTED
===========================================================================
Written in full before the first reconstruction row was computed. Nothing below
was chosen after seeing an output; deviations, if any are forced, are declared in
the findings note as deviations rather than silently applied.

PR-1 · SAMPLE. Every market in data/newsagent/hist/universe.json that carries a
    lookahead-free onboarding prior whose outcome-knowledge canary is not "known":
    the 40 markets discovered in v3.1 plus the 4 July-2026 markets settled in v3.3
    (sf-2026-002/003/004/009), folded in this round through --discover/--fetch/
    --ingest. Markets with no usable first packet produce no prior and are EXCLUDED;
    every exclusion is reported with its reason. Expected n ~ 44.

PR-2 · TRAJECTORY. Per market, snapshots every 4 days over the last 30 days before
    the scheduled end (<= 8 snapshots — the v3.1 grid, UNCHANGED). At each snapshot
    the published number is fair_value(p0, A_t, alpha, shift_clip) under the CURRENT
    live params (alpha 2.85, band_mult 0.5, gamma 1.0, declared lambda/floors/clips)
    and the LIVE Scheme-A reliability weights composed with the AllSides-seeded lean
    multiplier. A_t is evolved by fvmodel.step_state exactly as the live loop does.
    Between snapshots the published number is the LAST one published — the same
    step-function semantics the forward ledger has (v3.3 scored the last number
    standing before resolution, staleness included).

PR-3 · HEADLINE SCORES AT RESOLUTION. Pooled Brier and log-loss of the LAST
    published FV before each market's end, one row per market (n ~ 44). Reported
    with n attached. Log-loss uses the model's own [1, 99] FV clip, so no separate
    epsilon is introduced.

PR-4 · BRIER vs DAYS-BEFORE-RESOLUTION — the centrepiece, and the cadence question.
    Two curves, both pre-registered, both reported:
      (a) RAW: for each grid horizon h in {1,5,9,13,17,21,25,29} days before the
          end, the pooled Brier over every market that has a snapshot at h, with
          n(h) printed beside it. n(h) falls with h because short-lived markets
          have no far snapshots.
      (b) BALANCED: the same curve restricted to markets that carry ALL eight
          snapshots, so the shape cannot be an artifact of which markets drop out.
          (b) is the honest read; (a) is reported for completeness.
    DERIVED, and labelled derived: expected Brier under a publish cadence of N days
    = the mean of the balanced curve over the horizons falling in the first N days,
    i.e. what Justin should expect to score if he runs the loop every N days.

PR-5 · MURPHY DECOMPOSITION of the at-resolution scores: reliability + resolution
    - uncertainty on the standard three-component split, with DECLARED binning of
    5 equal bins of width 20pp (the same bins the band-coverage rescale uses).
    Reported with the caveat that at n ~ 44 across 5 bins the decomposition is a
    shape read, not a test.

PR-6 · RELIABILITY CURVE at n ~ 44: the same 5 bins, forecast mean vs realised
    frequency, with per-bin counts printed. Spiegelhalter Z and calibration-in-the-
    large reported alongside. Declared in advance: bins with fewer than 3 markets
    are plotted but explicitly marked thin, and no bin is dropped after the fact.

PR-7 · SPLITS.
    (a) PER FAMILY — the coarse v3.1 event-family buckets already stored on the
        universe rows. Pooled Brier + n per family.
    (b) PER TRACT — news / data / poll, the v3.3 tractability split. Assignment
        rule, DECLARED HERE BEFORE COMPUTING: config.tract(slug) when the slug is
        known to the live or retired slate; otherwise, for historical-only markets,
        family "fed" => "data" (rate decisions price off futures/options-implied
        odds, the identical reason the live Fed cards carry the tag), a question
        matching the declared election/primary/ballot-measure pattern => "poll",
        and everything else => "news". Applied mechanically, never per-market.

PR-8 · DIVERGENCE-FLAG OUTCOMES. Of the reconstruction snapshots where
    fvmodel.divergence_flag(...)["flag"] is True, tabulate how those markets
    resolved and the sign of the gap (FV below mid vs above). THIS IS A
    DESCRIPTION, NOT AN EDGE CLAIM. The v0 gate closure stands: two designs failed
    the pre-registered "our % beats the mid" bars and the fair-value-vs-mid framing
    is CLOSED (newsagent_v0_gate_findings). No hit rate, no PnL, no mid-relative
    score is offered here as evidence of skill, and that closure is restated
    wherever this table appears — note and page alike.

PR-9 · THE EVIDENCE-POORER CAVEAT, stated on the page and in the note.
    Reconstruction uses ONLY channels with timestamped archives: Guardian (date-
    bounded search), Wikipedia Current Events (whole past days) and the GDELT-GKG
    day-partitioned attention series. RSS feeds, newsletters, macro-research PDFs
    and agent-reach official documents carry no timestamped archive and are
    LIVE-ONLY: they are absent from every reconstruction packet by construction.
    The reconstruction therefore runs in an EVIDENCE-POORER world than the live
    loop. Which way that biases the score is NOT claimed in either direction.

PR-10 · SEPARATION AND METHOD SCOPE. Reconstruction is news-method only. The
    data-evidence channel (news+data, live on the September Fed market since
    2026-08-24) has ZERO settled forecasts and no reconstruction history, so it
    does not appear in this track at all. alpha is NOT refit on anything here:
    the honest refit is a separate --fit run of the backfill script over the
    enlarged sample, reported separately.

NO VERDICT RULE. This pass has no GO/NO-GO gate; it is a measurement, and its
purpose is to make the accuracy story on the public page real rather than
promised. Nothing here licenses an edge claim.
===========================================================================

Stages:
  --reconstruct   replay every market's trajectory -> simlive_trajectory.csv
  --score         compute the pre-registered readout -> CSVs + JSON + plot
  --chart         emit the staleness/reliability figure (implied by --score)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import newsagent_hist_backfill as hb                       # noqa: E402
from newsagent import config, features, fvmodel, gdelt_bq, sourceweights   # noqa: E402
from newsagent.config import CSV_OUT, ROOT                 # noqa: E402

OUT_JSON = ROOT / "data" / "newsagent" / "simlive" / "simlive_results.json"
TRAJ_CSV = CSV_OUT / "newsagent_simlive_trajectory.csv"
PLOTS = ROOT / "data" / "analysis" / "plots" / "news_agent"

# PR-4: the reconstruction grid is 4-day, so horizons land on these ages in days.
HORIZONS = [1, 5, 9, 13, 17, 21, 25, 29]
# PR-5/PR-6: declared binning — 5 equal bins of width 20pp, the band-coverage bins.
N_BINS = 5
# PR-7(b): declared election/ballot pattern for historical-only markets.
POLL_PATTERN = re.compile(
    r"\b(election|elected|primary|ballot measure|referendum|wins? the|"
    r"next (prime minister|president|chancellor)|governor)\b", re.I)


# ------------------------------------------------------------------ tract ---

def tract_for(m: dict) -> str:
    """news | data | poll — PR-7(b), applied mechanically.

    config.tract() is authoritative for any slug the live or retired slate knows.
    Historical-only markets get the declared family/question rule: the coarse
    'fed' family is data-driven for the same reason the live Fed cards are (rate
    odds price off futures/options-implied probabilities, which no news packet
    sees); an election/primary/ballot question is poll-driven; everything else is
    news-driven."""
    known = set(config.LIVE_MARKETS) | set(config.RETIRED_MARKETS)
    if m["slug"] in known:
        return config.tract(m["slug"])
    if m.get("family") == "fed":
        return "data"
    if POLL_PATTERN.search(m.get("question") or ""):
        return "poll"
    return "news"


# ---------------------------------------------------------- reconstruct ---

def cmd_reconstruct() -> None:
    params = fvmodel.load_params()
    alpha = params["alpha"]
    series_all = gdelt_bq.load_series()
    gamma = params.get("gamma", 0.0) if series_all else 0.0
    universe = json.loads(hb.UNIVERSE.read_text())
    priors = json.loads(hb.HIST_PRIORS.read_text())

    rows, excluded = [], []
    for m in universe:
        slug = m["slug"]
        pr = priors.get(slug)
        if pr is None:
            excluded.append((slug, "no lookahead-free onboarding prior"))
            continue
        if pr.get("outcome_knowledge") == "known":
            excluded.append((slug, "outcome-knowledge canary: known"))
            continue
        sdir = hb.HIST / slug[:80]
        mh = sdir / "mid_history.json"
        mids = json.loads(mh.read_text()) if mh.exists() else {}
        tp = fvmodel.type_params(params, m["mtype"])
        series = series_all.get(slug, {})
        end = datetime.fromisoformat(m["end_date"])
        state, counted, n_snap = None, set(), 0
        for d in hb.snapshot_dates(m):
            pf = sdir / f"{d}.packet.json"
            if not pf.exists():
                continue
            pkt = json.loads(pf.read_text())
            feats = sourceweights.annotate(features.features_for(slug, pkt["articles"]))
            new = [r for r in feats if r["cache_key"] not in counted]
            counted |= {r["cache_key"] for r in new}
            bz = gdelt_bq.burst_z(series, d.replace("-", "")) if series else None
            vol_z = bz["vol_z"] if bz else None
            s_eff = fvmodel.amplify(fvmodel.daily_score(new), vol_z, gamma)
            state = fvmodel.step_state(state, d, s_eff, tp["lam"], tp["a_clip"], tp["s_min"])
            fv = fvmodel.fair_value(pr["p0_pct"], state["A"], alpha, tp["shift_clip"])
            half = fvmodel.band_half_pp(feats, m["mtype"], params)
            n_rel = fvmodel.n_relevant(feats)
            mid_pct = (float(mids[d]) * 100.0) if d in mids else None
            flag = (fvmodel.divergence_flag(fv, mid_pct, half, n_rel)
                    if mid_pct is not None else None)
            n_snap += 1
            rows.append({
                "slug": slug, "question": m["question"], "family": m["family"],
                "tract": tract_for(m), "mtype": m["mtype"],
                "src": m.get("src", "hist"), "date": d,
                "days_to_end": (end - datetime.fromisoformat(d)).days,
                "p0_pct": pr["p0_pct"], "A": round(state["A"], 4),
                "fv_pct": round(fv, 2), "band_half_pp": half, "n_rel": n_rel,
                "mid_pct": None if mid_pct is None else round(mid_pct, 2),
                "gap_pp": None if flag is None else flag["gap_pp"],
                "flag": None if flag is None else int(flag["flag"]),
                "vol_z": vol_z, "y": int(m["y"]),
                "n_articles": len(feats),
                "n_missing_feats": sum(1 for r in feats if r["features"] is None),
            })
        if n_snap == 0:
            excluded.append((slug, "no reconstructable packet on the grid"))

    TRAJ_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAJ_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_mkt = len({r["slug"] for r in rows})
    miss = sum(r["n_missing_feats"] for r in rows)
    tot = sum(r["n_articles"] for r in rows)
    print(f"reconstruction: {len(rows)} snapshots over {n_mkt} markets "
          f"(alpha {alpha}, band_mult {params['band_mult']}, gamma {gamma})")
    print(f"  Stage-A coverage: {tot - miss}/{tot} article-features present "
          f"({miss} missing -> contribute 0 by construction)")
    if excluded:
        print(f"  EXCLUDED {len(excluded)}: "
              + "; ".join(f"{s[:44]} ({r})" for s, r in excluded))
    print(f"  -> {TRAJ_CSV}")


# --------------------------------------------------------------- scoring ---

def brier(ps: list[float], ys: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def log_loss(ps: list[float], ys: list[int]) -> float:
    """PR-3: no extra epsilon — the model's own [1, 99] FV clip already bounds p."""
    return -sum(y * math.log(p) + (1 - y) * math.log(1 - p)
                for p, y in zip(ps, ys)) / len(ps)


def murphy(ps: list[float], ys: list[int], n_bins: int = N_BINS) -> dict:
    """Three-component decomposition: Brier = reliability - resolution + uncertainty."""
    base = sum(ys) / len(ys)
    buckets: dict[int, list[int]] = {}
    for i, p in enumerate(ps):
        buckets.setdefault(min(n_bins - 1, int(p * n_bins)), []).append(i)
    rel = res = 0.0
    for idxs in buckets.values():
        nk = len(idxs)
        pbar = sum(ps[i] for i in idxs) / nk
        obar = sum(ys[i] for i in idxs) / nk
        rel += nk * (pbar - obar) ** 2
        res += nk * (obar - base) ** 2
    n = len(ps)
    return {"reliability": round(rel / n, 4), "resolution": round(res / n, 4),
            "uncertainty": round(base * (1 - base), 4), "base_rate": round(base, 4),
            "n_bins_populated": len(buckets)}


def spiegelhalter_z(ps: list[float], ys: list[int]) -> dict:
    """Z for calibration-in-the-small. |Z| > 1.96 rejects calibration at 5%."""
    num = sum((y - p) * (1 - 2 * p) for p, y in zip(ps, ys))
    var = sum(((1 - 2 * p) ** 2) * p * (1 - p) for p in ps)
    z = num / math.sqrt(var) if var > 0 else 0.0
    # two-sided normal p-value without scipy
    p_val = math.erfc(abs(z) / math.sqrt(2))
    return {"z": round(z, 3), "p": round(p_val, 3)}


def reliability_curve(ps: list[float], ys: list[int], n_bins: int = N_BINS) -> list[dict]:
    """PR-6: fixed 5 bins; thin bins are MARKED, never dropped after the fact."""
    buckets: dict[int, list[int]] = {}
    for i, p in enumerate(ps):
        buckets.setdefault(min(n_bins - 1, int(p * n_bins)), []).append(i)
    out = []
    for b in range(n_bins):
        idxs = buckets.get(b, [])
        out.append({
            "bin": f"{b * 100 // n_bins}-{(b + 1) * 100 // n_bins}%", "n": len(idxs),
            "mean_forecast_pct": (round(100 * sum(ps[i] for i in idxs) / len(idxs), 1)
                                  if idxs else None),
            "observed_yes_pct": (round(100 * sum(ys[i] for i in idxs) / len(idxs), 1)
                                 if idxs else None),
            "thin": len(idxs) < 3})
    return out


def split_table(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    out = []
    for k, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ps = [r["p"] for r in rs]
        ys = [r["y"] for r in rs]
        out.append({key: k, "n": len(rs), "brier": round(brier(ps, ys), 4),
                    "log_loss": round(log_loss(ps, ys), 4),
                    "base_rate_pct": round(100 * sum(ys) / len(ys), 1),
                    "mean_forecast_pct": round(100 * sum(ps) / len(ps), 1),
                    "thin": len(rs) < 5})
    return out


def cmd_score(chart: bool = True) -> None:
    with open(TRAJ_CSV) as f:
        traj = list(csv.DictReader(f))
    for r in traj:
        r["days_to_end"] = int(r["days_to_end"])
        r["fv_pct"] = float(r["fv_pct"])
        r["y"] = int(r["y"])
        r["p"] = r["fv_pct"] / 100.0
        r["flag"] = int(r["flag"]) if r["flag"] not in ("", "None") else None
        r["gap_pp"] = float(r["gap_pp"]) if r["gap_pp"] not in ("", "None") else None
        r["mid_pct"] = float(r["mid_pct"]) if r["mid_pct"] not in ("", "None") else None

    by_slug: dict[str, list[dict]] = {}
    for r in traj:
        by_slug.setdefault(r["slug"], []).append(r)
    for rs in by_slug.values():
        rs.sort(key=lambda r: r["days_to_end"])

    # PR-3 — the last number standing before resolution, one row per market.
    last = [rs[0] for rs in by_slug.values()]
    ps = [r["p"] for r in last]
    ys = [r["y"] for r in last]
    res = {
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "label": "reconstruction",
        "n_markets": len(last), "n_snapshots": len(traj),
        "params": {k: v for k, v in fvmodel.load_params().items()
                   if k in ("alpha", "band_mult", "gamma", "lambda", "floor_pp")},
        "at_resolution": {
            "brier": round(brier(ps, ys), 4), "log_loss": round(log_loss(ps, ys), 4),
            # prior-only = alpha 0: what the onboarding prior alone would have scored.
            # The gap between the two IS the news channel's measured contribution.
            "brier_prior_only": round(
                brier([float(r["p0_pct"]) / 100.0 for r in last], ys), 4),
            "mean_abs_shift_from_prior_pp": round(
                sum(abs(r["fv_pct"] - float(r["p0_pct"])) for r in last) / len(last), 2),
            "share_moved_ge_1pp": round(
                sum(1 for r in last if abs(r["fv_pct"] - float(r["p0_pct"])) >= 1.0)
                / len(last), 2),
            "base_rate_pct": round(100 * sum(ys) / len(ys), 1),
            "mean_forecast_pct": round(100 * sum(ps) / len(ps), 1),
            "mean_age_days": round(sum(r["days_to_end"] for r in last) / len(last), 1),
            "murphy": murphy(ps, ys), "spiegelhalter": spiegelhalter_z(ps, ys),
            "reliability_curve": reliability_curve(ps, ys),
        },
    }

    # PR-4 — the staleness curve, raw and balanced.
    full = {s for s, rs in by_slug.items() if len(rs) >= len(HORIZONS)}
    def curve(slugs: set[str] | None) -> list[dict]:
        out = []
        for h in HORIZONS:
            sel = [r for r in traj if r["days_to_end"] == h
                   and (slugs is None or r["slug"] in slugs)]
            if not sel:
                out.append({"days_before_resolution": h, "n": 0, "brier": None,
                            "log_loss": None})
                continue
            p = [r["p"] for r in sel]
            y = [r["y"] for r in sel]
            out.append({"days_before_resolution": h, "n": len(sel),
                        "brier": round(brier(p, y), 4),
                        "log_loss": round(log_loss(p, y), 4),
                        "mean_forecast_pct": round(100 * sum(p) / len(p), 1)})
        return out

    raw_c, bal_c = curve(None), curve(full)
    # Diagnostic that makes a FLAT staleness curve interpretable: a curve can be flat
    # because staleness is cheap, or because the model barely updates off its prior.
    # These two columns separate those readings and are reported with the curve.
    for c in bal_c:
        sel = [r for r in traj if r["days_to_end"] == c["days_before_resolution"]
               and r["slug"] in full]
        if not sel:
            continue
        moves = [abs(r["fv_pct"] - float(r["p0_pct"])) for r in sel]
        c["mean_abs_shift_from_prior_pp"] = round(sum(moves) / len(moves), 2)
        c["share_moved_ge_1pp"] = round(sum(1 for m in moves if m >= 1.0) / len(moves), 2)
    # DERIVED (labelled derived): expected Brier if the loop publishes every N days.
    # A number published under an N-day cadence has an age uniform on [0, N), so the
    # expected score is the mean of the curve over the horizons inside that window.
    cadence = []
    bal_by_h = {c["days_before_resolution"]: c["brier"] for c in bal_c
                if c["brier"] is not None}
    for n_days in (4, 8, 12, 16, 20, 28):
        hs = [h for h in bal_by_h if h <= n_days]
        if hs:
            cadence.append({"cadence_days": n_days, "horizons_averaged": sorted(hs),
                            "expected_brier": round(
                                sum(bal_by_h[h] for h in hs) / len(hs), 4)})
    res["staleness"] = {"raw": raw_c, "balanced": bal_c,
                        "n_markets_balanced": len(full),
                        "derived_cadence": cadence}

    # PR-7 — splits, scored on the at-resolution row.
    res["splits"] = {"family": split_table(last, "family"),
                     "tract": split_table(last, "tract")}

    # PR-8 — divergence-flag outcomes. DESCRIPTION ONLY. The v0 closure stands.
    flagged = [r for r in traj if r["flag"] == 1]
    fl_slugs = {r["slug"] for r in flagged}
    below = [r for r in flagged if r["gap_pp"] < 0]
    res["divergence"] = {
        "n_flagged_snapshots": len(flagged),
        "n_flagged_markets": len(fl_slugs),
        "n_snapshots_with_a_mid": sum(1 for r in traj if r["mid_pct"] is not None),
        "flagged_markets_resolved_yes": len({r["slug"] for r in flagged if r["y"] == 1}),
        "fv_below_mid_snapshots": len(below),
        "fv_above_mid_snapshots": len(flagged) - len(below),
        "closure": ("The v0 fair-value-vs-mid gate is CLOSED (two pre-registered designs "
                    "failed) and is NOT reopened here. This table describes where the "
                    "model most disagreed with the market and how those questions "
                    "resolved. It is not a hit rate, not an edge claim, and no "
                    "mid-relative score is computed from it."),
    }

    # Mid, CONTEXT ONLY. Reported in the findings note for the same reason every prior
    # note reports it — so the reader can see the market's own number — and never as a
    # benchmark. The "our % beats the mid" gate is CLOSED and is not re-litigated here;
    # this figure is deliberately kept OFF the public page's reconstruction panel.
    with_mid = [r for r in last if r["mid_pct"] is not None]
    res["mid_context"] = {
        "n": len(with_mid),
        "brier": (round(brier([r["mid_pct"] / 100.0 for r in with_mid],
                              [r["y"] for r in with_mid]), 4) if with_mid else None),
        "ours_on_same_subset": (round(brier([r["p"] for r in with_mid],
                                            [r["y"] for r in with_mid]), 4)
                                if with_mid else None),
        "closure": ("Context only. The v0 gate closed the fair-value-vs-mid claim after "
                    "two pre-registered designs failed it; this row is not a comparison "
                    "we are making, and it is not shown on the public page."),
    }

    res["exclusions_note"] = (
        "Reconstruction sees ONLY channels with a timestamped archive: Guardian, "
        "Wikipedia Current Events and the GDELT-GKG attention series. RSS feeds, "
        "newsletters, macro-research PDFs and agent-reach official documents are "
        "live-only and are absent by construction — an evidence-POORER world than the "
        "live loop. Which way that biases the score is not claimed.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    _write_csvs(res)
    _print(res)
    if chart:
        _chart(res)


def _write_csvs(res: dict) -> None:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT / "newsagent_simlive_staleness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["days_before_resolution", "n_raw", "brier_raw",
                    "n_balanced", "brier_balanced"])
        for a, b in zip(res["staleness"]["raw"], res["staleness"]["balanced"]):
            w.writerow([a["days_before_resolution"], a["n"], a["brier"],
                        b["n"], b["brier"]])
    with open(CSV_OUT / "newsagent_simlive_splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "group", "n", "brier", "log_loss",
                    "base_rate_pct", "mean_forecast_pct", "thin"])
        for kind in ("family", "tract"):
            for r in res["splits"][kind]:
                w.writerow([kind, r[kind], r["n"], r["brier"], r["log_loss"],
                            r["base_rate_pct"], r["mean_forecast_pct"], r["thin"]])
    with open(CSV_OUT / "newsagent_simlive_reliability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "n", "mean_forecast_pct", "observed_yes_pct", "thin"])
        for r in res["at_resolution"]["reliability_curve"]:
            w.writerow([r["bin"], r["n"], r["mean_forecast_pct"],
                        r["observed_yes_pct"], r["thin"]])


def _print(res: dict) -> None:
    a = res["at_resolution"]
    print(f"\n=== SIMULATED-LIVE RECONSTRUCTION (label: {res['label']}) ===")
    print(f"n = {res['n_markets']} resolved markets, {res['n_snapshots']} snapshots, "
          f"alpha {res['params']['alpha']}")
    print(f"AT RESOLUTION  Brier {a['brier']}  log-loss {a['log_loss']}  "
          f"base rate {a['base_rate_pct']}%  mean forecast {a['mean_forecast_pct']}%  "
          f"mean age {a['mean_age_days']}d")
    m = a["murphy"]
    print(f"  Murphy: reliability {m['reliability']} - resolution {m['resolution']} "
          f"+ uncertainty {m['uncertainty']}  ({m['n_bins_populated']}/{N_BINS} bins)")
    print(f"  Spiegelhalter Z {a['spiegelhalter']['z']} (p {a['spiegelhalter']['p']})")
    print("\nBRIER vs DAYS BEFORE RESOLUTION")
    print(f"  {'days':>5} | {'n':>3} {'raw':>8} | {'n':>3} {'balanced':>9}")
    for r, b in zip(res["staleness"]["raw"], res["staleness"]["balanced"]):
        print(f"  {r['days_before_resolution']:>5} | {r['n']:>3} {str(r['brier']):>8} "
              f"| {b['n']:>3} {str(b['brier']):>9}")
    print(f"  (balanced = the {res['staleness']['n_markets_balanced']} markets with all "
          f"{len(HORIZONS)} snapshots)")
    print("  evidence movement on the balanced set (mean |FV-prior| pp, share moved >=1pp):")
    print("    " + "  ".join(
        f"{c['days_before_resolution']}d:{c.get('mean_abs_shift_from_prior_pp')}/"
        f"{c.get('share_moved_ge_1pp')}" for c in res["staleness"]["balanced"]))
    print(f"  prior-only Brier {a['brier_prior_only']} | mean |FV-prior| "
          f"{a['mean_abs_shift_from_prior_pp']}pp | share moved >=1pp "
          f"{a['share_moved_ge_1pp']}")
    print("\nDERIVED — expected Brier by publish cadence")
    for c in res["staleness"]["derived_cadence"]:
        print(f"  every {c['cadence_days']:>2}d -> {c['expected_brier']}")
    for kind in ("tract", "family"):
        print(f"\nSPLIT by {kind}")
        for r in res["splits"][kind]:
            print(f"  {r[kind]:>16} n={r['n']:>2} Brier {r['brier']:<7} "
                  f"base {r['base_rate_pct']:>5}%  mean fv {r['mean_forecast_pct']:>5}%"
                  + ("  [thin]" if r["thin"] else ""))
    print("\nRELIABILITY (n per bin)")
    for r in a["reliability_curve"]:
        print(f"  {r['bin']:>8} n={r['n']:>2} forecast {str(r['mean_forecast_pct']):>6} "
              f"observed {str(r['observed_yes_pct']):>6}" + ("  [thin]" if r["thin"] else ""))
    d = res["divergence"]
    print(f"\nDIVERGENCE FLAGS (description only — the v0 closure stands): "
          f"{d['n_flagged_snapshots']} flagged snapshots across {d['n_flagged_markets']} "
          f"markets; {d['flagged_markets_resolved_yes']} of those markets resolved YES; "
          f"{d['fv_below_mid_snapshots']} below-mid / {d['fv_above_mid_snapshots']} above-mid")
    print(f"\n-> {OUT_JSON}")


def _chart(res: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                   # pragma: no cover
        print(f"  (chart skipped: {e})")
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    raw, bal = res["staleness"]["raw"], res["staleness"]["balanced"]
    xs = [r["days_before_resolution"] for r in raw]
    ax1.plot(xs, [r["brier"] for r in raw], "o--", color="#9aa0a6",
             label=f"raw (n varies, {res['n_markets']} markets)")
    ax1.plot(xs, [r["brier"] for r in bal], "o-", color="#c0563c", linewidth=2,
             label=f"balanced (n={res['staleness']['n_markets_balanced']} markets)")
    ax1.set_xlabel("days before resolution (snapshot age)")
    ax1.set_ylabel("pooled Brier")
    ax1.set_title("Reconstruction — Brier vs staleness")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)
    rc = res["at_resolution"]["reliability_curve"]
    fx = [r["mean_forecast_pct"] for r in rc if r["n"]]
    fy = [r["observed_yes_pct"] for r in rc if r["n"]]
    sz = [30 + 18 * r["n"] for r in rc if r["n"]]
    ax2.plot([0, 100], [0, 100], "--", color="#9aa0a6", linewidth=1)
    ax2.scatter(fx, fy, s=sz, color="#c0563c", zorder=3)
    for r in rc:
        if r["n"]:
            ax2.annotate(f"n={r['n']}", (r["mean_forecast_pct"], r["observed_yes_pct"]),
                         textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("mean reconstructed forecast (%)")
    ax2.set_ylabel("observed YES frequency (%)")
    ax2.set_title(f"Reconstruction — reliability, n={res['n_markets']}")
    ax2.grid(alpha=0.25)
    fig.suptitle("Simulated-live RECONSTRUCTION — not the forward ledger", fontsize=10)
    fig.tight_layout()
    out = PLOTS / "newsagent_simlive.png"
    fig.savefig(out, dpi=140)
    print(f"  chart -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reconstruct", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args()
    if args.reconstruct:
        cmd_reconstruct()
    if args.score:
        cmd_score(chart=not args.no_chart)
    if not (args.reconstruct or args.score):
        print("nothing to do — pass --reconstruct and/or --score")


if __name__ == "__main__":
    main()
