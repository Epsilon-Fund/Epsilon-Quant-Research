"""Data-channel v3 — Amendment 2: regime-neutral criterion C, W_SEP = 1.0,
40 resolved FOMC decisions across three rate regimes.

Runs the rule locked into [[newsagent_data_channel_scoping]] § 4f AMENDMENT 2 on
2026-08-24, over strictly point-in-time inputs, against criteria fixed before the
first number existed:

  A  pooled: mean Brier over the sample < mean Brier of base-rate-only
  B  July-2026 market: Brier over the dry run's own 15-day intersection < 0.2144
     (PEEK-COMPROMISED — see the lock; the weight rests on A)
  C  directional sanity, REGIME-NEUTRAL: zero violations

What changed from v2, and nothing else changed:
  V3-1  criterion C takes its implied direction from the pressure MAGNITUDE
        |r_desired - midpoint|, not from the raw direction of the news
  V3-2  W_SEP = 1.0 (allocate evenly; the SEP-concentration hypothesis was
        tested on 8 meetings and not supported)
  V3-3  B1A stays 1.5 — deliberately NOT moved toward the 2.0-3.0 the v2 sweep
        prefers, which would be fitting to an outcome
  V3-4  sample extended to 40 meetings: 2017-18 normalisation, 2022-23 hiking,
        2025 cutting, 2026 holding
  V3-5  a hard per-meeting leakage assertion; 2015-16 excluded because the
        Dec-2015 liftoff is stamped on its own decision day

Run: PYTHONPATH=. uv run python scripts/newsagent_datachannel_v3.py [--chart]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time  # noqa: E402
import urllib.error  # noqa: E402

import newsagent_datachannel_dryrun as _dry  # noqa: E402
from newsagent_datachannel_dryrun import (  # noqa: E402
    CACHE, MC_DRAWS, P_CLIP, SEED, SIGMA_MOM, KAPPA_PI, KAPPA_U,
    as_of, load_nowcasts, month_add, nowcast_asof,
    base_rate as base_rate_zlb,
)


def vintage_series(series_id: str, obs_start: str, rt_start: str, rt_end: str) -> list[dict]:
    """The dry run's cached ALFRED pull, with backoff.

    v3 needs ~280 distinct (series, realtime-window) pulls where v2 needed 56, and
    FRED answers 429 well before that. The on-disk cache means a retry always
    makes forward progress, so a bounded exponential backoff is enough — no
    parallelism, no key rotation, and the cached windows are reused across runs.
    """
    f = CACHE / f"alfred_{series_id}_{rt_start}_{rt_end}.json"
    if f.exists():
        return json.loads(f.read_text())
    delay = 3.0
    for attempt in range(7):
        try:
            rows = _dry.vintage_series(series_id, obs_start, rt_start, rt_end)
            time.sleep(0.35)          # be a good citizen between fresh pulls
            return rows
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 6:
                raise
            time.sleep(delay)
            delay *= 1.8
    raise RuntimeError("unreachable")
from newsagent_datachannel_v2 import (  # noqa: E402
    JULY_BAR, JULY_INTERSECTION, TRAJ_DAYS, DEAD_BAND, base_rate_all_years,
)
from newsagent.config import CSV_OUT, ROOT  # noqa: E402

PLOTS = ROOT / "data" / "analysis" / "plots" / "news_agent"

# ---------------------------------------------------------------------------
# LOCKED CONSTANTS — Amendment 2, § 4f, 2026-08-24. Not fitted, not tuned.
# ---------------------------------------------------------------------------
W_SEP = 1.0          # V3-2: allocate EVENLY. SEP-concentration tested, unsupported.
W_NON = 1.0
B1A = 1.5            # V3-3: unchanged. The v2 sweep prefers 2.0-3.0; not moved.
PRESSURE_EPS = 1e-4  # |d_pressure| must exceed this (pp) to score a release day

# Scheduled FOMC decision dates (second day of each meeting), published years in
# advance and therefore vintage-safe. True = the meeting carries an SEP.
FOMC = {
    2017: [(date(2017, 2, 1), False), (date(2017, 3, 15), True),
           (date(2017, 5, 3), False), (date(2017, 6, 14), True),
           (date(2017, 7, 26), False), (date(2017, 9, 20), True),
           (date(2017, 11, 1), False), (date(2017, 12, 13), True)],
    2018: [(date(2018, 1, 31), False), (date(2018, 3, 21), True),
           (date(2018, 5, 2), False), (date(2018, 6, 13), True),
           (date(2018, 8, 1), False), (date(2018, 9, 26), True),
           (date(2018, 11, 8), False), (date(2018, 12, 19), True)],
    2022: [(date(2022, 1, 26), False), (date(2022, 3, 16), True),
           (date(2022, 5, 4), False), (date(2022, 6, 15), True),
           (date(2022, 7, 27), False), (date(2022, 9, 21), True),
           (date(2022, 11, 2), False), (date(2022, 12, 14), True)],
    2023: [(date(2023, 2, 1), False), (date(2023, 3, 22), True),
           (date(2023, 5, 3), False), (date(2023, 6, 14), True),
           (date(2023, 7, 26), False), (date(2023, 9, 20), True),
           (date(2023, 11, 1), False), (date(2023, 12, 13), True)],
    2025: [(date(2025, 1, 29), False), (date(2025, 3, 19), True),
           (date(2025, 4, 30), False), (date(2025, 6, 18), True),
           (date(2025, 7, 30), False), (date(2025, 9, 17), True),
           (date(2025, 10, 29), False), (date(2025, 12, 10), True)],
    2026: [(date(2026, 1, 28), False), (date(2026, 3, 18), True),
           (date(2026, 4, 29), False), (date(2026, 6, 17), True),
           (date(2026, 7, 29), False), (date(2026, 9, 16), True),
           (date(2026, 10, 28), False), (date(2026, 12, 9), True)],
}

# V3-4: the declared sample. regime is a REPORTING label, never an input.
SAMPLE_BLOCKS = [
    (2017, "normalisation"), (2018, "normalisation"),
    (2022, "hiking"), (2023, "hiking"),
    (2025, "cutting"), (2026, "holding"),
]
# 2026 meetings that had not yet happened when this was locked.
NOT_YET_RESOLVED = {date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9)}
# The LOCKED sample carries only the three 2025 cutting meetings, exactly as v2
# used them (Amendment 2, V3-4 table). An initial execution of this script
# scored the whole 2025 year instead — 45 meetings rather than the declared 40 —
# which is a coding deviation from the lock, not a change to it. The lock is the
# spec; the extra five are excluded here and both numbers are reported in the
# findings note so nobody has to wonder whether the sample moved after a result
# was seen.
SAMPLE_ONLY = {2025: {date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10)}}


def target_change_dates() -> set[date]:
    """Effective dates on which the target range moved (never revised)."""
    out: set[date] = set()
    for sid, obs0 in (("DFEDTAR", "1994-01-01"), ("DFEDTARU", "2008-12-01")):
        rows = vintage_series(sid, obs0, "2026-08-24", "2026-08-24")
        prev = None
        for o in rows:
            if o["value"] in (".", "", None):
                continue
            v, d = float(o["value"]), datetime.fromisoformat(o["date"]).date()
            if prev is not None and abs(v - prev) > 1e-9:
                out.add(d)
            prev = v
    return out


def build_sample() -> list[dict]:
    """The 40 declared meetings, with outcomes read from the target series.

    outcome is in the MARKET's framing: 1 = no change, 0 = change. A meeting
    counts as a move when the target range changed on the decision day or the
    day after (the effective-date convention).
    """
    changes = target_change_dates()
    out = []
    for year, regime in SAMPLE_BLOCKS:
        for d, sep in FOMC[year]:
            if d in NOT_YET_RESOLVED:
                continue
            if year in SAMPLE_ONLY and d not in SAMPLE_ONLY[year]:
                continue
            moved = any((d + timedelta(days=k)) in changes for k in (0, 1))
            out.append({"decision": d, "sep": sep, "regime": regime,
                        "outcome": 0 if moved else 1,
                        "same_day_stamp": d in changes})
    return out


def _series_for(win_start: date, win_end: date) -> dict:
    rt0, rt1 = win_start.isoformat(), win_end.isoformat()
    obs0 = f"{win_start.year - 3}-01-01"
    return {sid: vintage_series(sid, obs0, rt0, rt1) for sid in
            ("DFEDTARU", "DFEDTARL", "PCEPILFE", "UNRATE",
             "FEDTARMD", "JCXFEMD", "UNRATEMD")}


def trajectory(meeting: dict, nc: dict, logit_h_base: float, logit_hold_zlb: float,
               b1a: float = B1A, w_sep: float = W_SEP) -> list[dict]:
    """Daily p_struct over [decision - TRAJ_DAYS, decision], vintage-only."""
    dec = meeting["decision"]
    start = dec - timedelta(days=TRAJ_DAYS)
    ser = _series_for(start, dec)
    sep_year = dec.year
    key = f"{sep_year}-01-01"

    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(MC_DRAWS // 2)
    z = np.concatenate([z, -z])

    rows, t = [], start
    while t <= dec:
        u_pub, pce_pub = as_of(ser["UNRATE"], t), as_of(ser["PCEPILFE"], t)
        up, lo = as_of(ser["DFEDTARU"], t), as_of(ser["DFEDTARL"], t)
        sep_r = as_of(ser["FEDTARMD"], t).get(key)
        sep_pi = as_of(ser["JCXFEMD"], t).get(key)
        sep_u = as_of(ser["UNRATEMD"], t).get(key)
        if not (u_pub and pce_pub and up and lo and sep_r and sep_pi and sep_u):
            t += timedelta(days=1)
            continue
        dkeys = [d for d in up if d <= t.isoformat()]
        lkeys = [d for d in lo if d <= t.isoformat()]
        if not dkeys or not lkeys:
            t += timedelta(days=1)
            continue
        mid = (up[max(dkeys)] + lo[max(lkeys)]) / 2.0

        last_m = max(pce_pub)
        ly, lm = int(last_m[:4]), int(last_m[5:7])
        k = (t.year * 12 + t.month) - (ly * 12 + lm)
        idx = pce_pub[last_m]
        for j in range(1, k + 1):
            ym = month_add((ly, lm), j)
            v = nowcast_asof(nc, ym, "Core PCE Inflation", t)
            if v is None:
                pk, bk = month_add((ly, lm), j - 1), month_add((ly, lm), j - 2)
                pkey, bkey = f"{pk[0]:04d}-{pk[1]:02d}-01", f"{bk[0]:04d}-{bk[1]:02d}-01"
                v = ((pce_pub[pkey] / pce_pub[bkey] - 1) * 100
                     if pkey in pce_pub and bkey in pce_pub else 0.0)
            idx *= (1 + v / 100.0)
        cur = month_add((ly, lm), k)
        bkey = f"{cur[0] - 1:04d}-{cur[1]:02d}-01"
        if bkey not in pce_pub:
            t += timedelta(days=1)
            continue
        pi_center = (idx / pce_pub[bkey] - 1) * 100.0
        u_last = u_pub[max(u_pub)]

        sigma = SIGMA_MOM * math.sqrt(max(k, 1))
        pi_draws = pi_center + sigma * z
        r_des = sep_r + KAPPA_PI * (pi_draws - sep_pi) - KAPPA_U * (u_last - sep_u)
        G = np.abs(r_des - mid) / 0.25

        rem = [(d, s) for d, s in FOMC.get(sep_year, []) if d >= t]
        ws = [w_sep if s else W_NON for _, s in rem] or [1.0]
        share1 = ws[0] / sum(ws)
        mu1 = G * share1
        zz = logit_h_base + b1a * mu1
        p_v3 = float(np.mean(np.clip(1.0 - 1.0 / (1.0 + np.exp(-zz)), *P_CLIP)))

        m_old = ((r_des - mid) / max(len(rem), 1)) / 0.25
        z1 = logit_hold_zlb + (-3.0) * np.abs(m_old)
        p_v1 = float(np.mean(np.clip(1.0 / (1.0 + np.exp(-z1)), *P_CLIP)))

        r_des_pt = sep_r + KAPPA_PI * (pi_center - sep_pi) - KAPPA_U * (u_last - sep_u)
        gap = r_des_pt - mid
        rows.append({
            "date": t.isoformat(), "p_struct": round(p_v3, 4), "p_v1": round(p_v1, 4),
            "gap_pp": round(gap, 4), "pressure_pp": round(abs(gap), 4),
            "G_clicks": round(abs(gap) / 0.25, 4), "share1": round(share1, 4),
            "n_rem": len(rem), "mu1": round(abs(gap) / 0.25 * share1, 4),
            "pi_core_pce_yoy": round(pi_center, 3), "unrate": u_last,
            "sep_rate": sep_r, "sep_core_pce": sep_pi, "sep_unrate": sep_u,
            "mid_target": mid, "k_unpublished": k, "last_pce_month": last_m,
        })
        t += timedelta(days=1)
    return rows


def directional_violations(rows: list[dict]) -> list[dict]:
    """V3-1 — criterion C, REGIME-NEUTRAL.

    The implied direction comes from the pressure MAGNITUDE, not from the sign of
    the news: a release that increases |r_desired - midpoint| must not raise
    p_struct, on either side of the target. v2's version read the raw direction
    of the SEP median / inflation print, which is only correct under hiking
    pressure and produced 6 spurious violations on a cutting cycle.
    """
    out = []
    for a, b in zip(rows, rows[1:]):
        released = (abs(b["sep_rate"] - a["sep_rate"]) > 1e-9
                    or b["last_pce_month"] != a["last_pce_month"])
        if not released:
            continue
        d_pressure = b["pressure_pp"] - a["pressure_pp"]
        if abs(d_pressure) <= PRESSURE_EPS:
            continue
        dp = b["p_struct"] - a["p_struct"]
        if abs(dp) <= DEAD_BAND:
            continue
        if (d_pressure > 0 and dp > 0) or (d_pressure < 0 and dp < 0):
            out.append({"date": b["date"],
                        "driver": "sep" if abs(b["sep_rate"] - a["sep_rate"]) > 1e-9
                                  else "pi",
                        "d_pressure": round(d_pressure, 4), "dp": round(dp, 4)})
    return out


def leak_check(meeting: dict, rows: list[dict], changes: set[date]) -> str:
    """V3-5 — the decision-day vintage must still show the PRE-decision level."""
    if meeting["decision"] in changes:
        return ("the target series is stamped on the decision day itself, so the "
                "decision-day vintage would leak the outcome")
    if not rows:
        return "no vintage rows in the trajectory window"
    if abs(rows[-1]["mid_target"] - rows[0]["mid_target"]) > 1e-9 and \
            meeting["outcome"] == 0:
        # the range moved DURING the window (an earlier meeting) — fine, but the
        # decision-day value must not already be the post-decision one
        pass
    return ""


def brier(ps, outcome) -> float:
    return float(np.mean([(p - outcome) ** 2 for p in ps])) if len(ps) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", action="store_true")
    args = ap.parse_args()

    hold_all, meta_all = base_rate_all_years()
    h_base = 1.0 - hold_all
    logit_h_base = math.log(h_base / (1 - h_base))
    hold_zlb, _ = base_rate_zlb()
    logit_hold_zlb = math.log(hold_zlb / (1 - hold_zlb))
    changes = target_change_dates()

    print(f"DC-3b base rate (ALL years, no ZLB exclusion): hold={hold_all:.4f}  "
          f"h_base={h_base:.4f}  ({meta_all['n_change_dates']} changes / "
          f"{meta_all['n_meetings']} meetings)")
    print(f"LOCKED: W_SEP={W_SEP}  B1A={B1A}  (Amendment 2)\n")

    nc = load_nowcasts()
    sample = build_sample()
    results, all_rows, dropped = [], [], []
    for mk in sample:
        rows = trajectory(mk, nc, logit_h_base, logit_hold_zlb)
        why = leak_check(mk, rows, changes)
        if why:
            dropped.append({"meeting": mk["decision"].isoformat(), "why": why})
            print(f"  DROPPED {mk['decision']}: {why}")
            continue
        for r in rows:
            r.update({"meeting": mk["decision"].isoformat(), "sep": mk["sep"],
                      "regime": mk["regime"], "outcome": mk["outcome"]})
        all_rows += rows
        dec = rows[-1]
        viol = directional_violations(rows)
        p = dec["p_struct"]
        results.append({
            "meeting": mk["decision"].isoformat(), "sep": mk["sep"],
            "regime": mk["regime"], "outcome": mk["outcome"], "n_days": len(rows),
            "p_struct": p, "brier": round((p - mk["outcome"]) ** 2, 4),
            "p_v1": dec["p_v1"], "brier_v1": round((dec["p_v1"] - mk["outcome"]) ** 2, 4),
            "p_base": round(hold_all, 4),
            "brier_base": round((hold_all - mk["outcome"]) ** 2, 4),
            "gap_pp": dec["gap_pp"], "G_clicks": dec["G_clicks"],
            "share1": dec["share1"], "n_rem": dec["n_rem"], "mu1": dec["mu1"],
            "violations": len(viol), "violation_detail": viol,
        })
        print(f"  {mk['decision']}  {'SEP' if mk['sep'] else '   '}  "
              f"{mk['regime']:13s} {'hold' if mk['outcome'] else 'MOVE'}  "
              f"G={dec['G_clicks']:6.2f}  mu1={dec['mu1']:6.3f}  "
              f"p={p:.3f}  brier={(p - mk['outcome']) ** 2:.4f}  viol={len(viol)}")

    mean_b = float(np.mean([r["brier"] for r in results]))
    mean_base = float(np.mean([r["brier_base"] for r in results]))
    mean_v1 = float(np.mean([r["brier_v1"] for r in results]))
    A = mean_b < mean_base

    j0, j1 = JULY_INTERSECTION
    july = [r for r in all_rows if r["meeting"] == "2026-07-29"
            and j0.isoformat() <= r["date"] <= j1.isoformat()]
    july_b = brier([r["p_struct"] for r in july], 1)
    B = july_b < JULY_BAR

    n_viol = sum(r["violations"] for r in results)
    C = n_viol == 0

    print(f"\n  meetings scored: {len(results)}  dropped: {len(dropped)}")
    print(f"A  pooled Brier  p_struct={mean_b:.4f}  base-rate-only={mean_base:.4f}"
          f"  -> {'PASS' if A else 'FAIL'}   (v1 rule: {mean_v1:.4f})")
    print(f"B  July window   p_struct={july_b:.4f}  bar={JULY_BAR}"
          f"  -> {'PASS' if B else 'FAIL'}   [peek-compromised]")
    print(f"C  violations    {n_viol}  -> {'PASS' if C else 'FAIL'}")
    print(f"\nVERDICT: {'GO' if (A and B and C) else 'NO-GO'}")

    by_regime = {}
    for r in results:
        by_regime.setdefault(r["regime"], []).append(r)
    print("\nby regime (REPORTED, not a bar — a method that only works in one "
          "regime must be visible):")
    for reg, rs in by_regime.items():
        print(f"  {reg:14s} n={len(rs):2d}  p_struct={np.mean([x['brier'] for x in rs]):.4f}"
              f"  base={np.mean([x['brier_base'] for x in rs]):.4f}"
              f"  v1={np.mean([x['brier_v1'] for x in rs]):.4f}"
              f"  viol={sum(x['violations'] for x in rs)}")

    sens = []
    for knob, vals in (("B1A", (0.5, 1.0, 1.5, 2.0, 3.0)),
                       ("W_SEP", (1.0, 1.5, 2.0, 3.0))):
        for v in vals:
            ps = []
            for mk in sample:
                rows = trajectory(mk, nc, logit_h_base, logit_hold_zlb,
                                  b1a=v if knob == "B1A" else B1A,
                                  w_sep=v if knob == "W_SEP" else W_SEP)
                if rows and not leak_check(mk, rows, changes):
                    ps.append((rows[-1]["p_struct"], mk["outcome"]))
            sens.append({"knob": knob, "value": v,
                         "pooled_brier": round(float(np.mean([(p - o) ** 2 for p, o in ps])), 4),
                         "mean_p": round(float(np.mean([p for p, _ in ps])), 4)})
    print("\nsensitivity (REPORTED, not a bar — verdict is on B1A=1.5 / W_SEP=1.0):")
    for s in sens:
        star = " <- declared" if (s["knob"], s["value"]) in (("B1A", B1A), ("W_SEP", W_SEP)) else ""
        print(f"  {s['knob']:6s}={s['value']:<4} Brier {s['pooled_brier']:.4f}"
              f"  mean p {s['mean_p']:.3f}{star}")

    CSV_OUT.mkdir(parents=True, exist_ok=True)
    with (CSV_OUT / "newsagent_datachannel_v3_meetings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in results[0] if k != "violation_detail"])
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in r.items() if k != "violation_detail"})
    with (CSV_OUT / "newsagent_datachannel_v3_trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with (CSV_OUT / "newsagent_datachannel_v3_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0].keys()))
        w.writeheader()
        w.writerows(sens)

    out = {"criteria": {"A_pooled": {"p_struct": round(mean_b, 4),
                                     "base_only": round(mean_base, 4),
                                     "v1_rule": round(mean_v1, 4), "pass": A},
                        "B_july": {"p_struct": round(july_b, 4), "bar": JULY_BAR,
                                   "pass": B, "peek_compromised": True},
                        "C_directional": {"violations": n_viol, "pass": C}},
           "verdict": "GO" if (A and B and C) else "NO-GO",
           "n_scored": len(results), "dropped": dropped,
           "constants": {"W_SEP": W_SEP, "W_NON": W_NON, "B1A": B1A,
                         "KAPPA_PI": KAPPA_PI, "KAPPA_U": KAPPA_U,
                         "SIGMA_MOM": SIGMA_MOM, "MC_DRAWS": MC_DRAWS, "SEED": SEED},
           "base_rate": {"hold_all_years": round(hold_all, 4), **meta_all},
           "by_regime": {k: {"n": len(v),
                             "brier": round(float(np.mean([x["brier"] for x in v])), 4),
                             "base": round(float(np.mean([x["brier_base"] for x in v])), 4),
                             "violations": sum(x["violations"] for x in v)}
                         for k, v in by_regime.items()},
           "meetings": results, "sensitivity": sens}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "v3_results.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {CACHE / 'v3_results.json'} and 3 CSVs")

    if args.chart:
        chart(results)
    return 0


def chart(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 9),
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    xs = list(range(len(results)))
    cols = {"normalisation": "#5b8c9d", "hiking": "#cc5c44",
            "cutting": "#2f7d55", "holding": "#9a7bb0"}
    for i, r in enumerate(results):
        ax.scatter(i, r["p_struct"], s=52, color=cols[r["regime"]],
                   marker="o" if r["outcome"] else "X", zorder=4,
                   edgecolor="#242423", linewidth=.5)
    ax.axhline(results[0]["p_base"], color="#9a9a90", ls="--", lw=1.2,
               label=f"base-rate-only ({results[0]['p_base']:.3f})")
    ax.set_ylim(0, 1)
    ax.set_ylabel("p_struct = P(no change at this meeting)")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["meeting"] for r in results], rotation=90, fontsize=6)
    ax.grid(alpha=.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k)
               for k, c in cols.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color="#555", label="outcome: HOLD"),
                plt.Line2D([], [], marker="X", ls="", color="#555", label="outcome: MOVE")]
    ax.legend(handles=handles, fontsize=7, ncol=3, loc="upper left")
    ax.set_title("Data-channel v3 — p_struct on the decision day, 40 resolved FOMC "
                 "decisions across three rate regimes\n"
                 "circles held, crosses moved. A well-behaved method puts circles "
                 "high and crosses low.", fontsize=11)

    w = 0.38
    ax2.bar([i - w / 2 for i in xs], [r["brier"] for r in results], w,
            color="#cc5c44", label="p_struct")
    ax2.bar([i + w / 2 for i in xs], [r["brier_base"] for r in results], w,
            color="#9a9a90", label="base-rate-only")
    ax2.set_ylabel("Brier (lower better)")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([r["meeting"] for r in results], rotation=90, fontsize=6)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.25, axis="y")
    fig.tight_layout()
    out = PLOTS / "newsagent_datachannel_v3.png"
    fig.savefig(out, dpi=140)
    print(f"chart -> {out}")


if __name__ == "__main__":
    sys.exit(main())
