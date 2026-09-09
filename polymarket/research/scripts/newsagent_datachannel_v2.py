"""Data-channel v2 — the amended reaction function (DC-3a + DC-3b) re-tested on
eight resolved FOMC decisions.

Runs the AMENDMENT-1 rule locked into [[newsagent_data_channel_scoping]] § 4f on
2026-08-24, over strictly point-in-time inputs, against criteria fixed before the
first number existed:

  A  pooled: mean Brier over the 8 meetings < mean Brier of base-rate-only
  B  July-2026 market: Brier over the dry run's own 15-day intersection < 0.2144
  C  directional sanity: zero violations, 0.5pp dead-band

The plumbing (ALFRED realtime vintages, the Cleveland daily nowcast vintages, the
PCE chaining, the never-revised target series) is imported unchanged from the dry
run, which proved it lookahead-free — see newsagent_data_channel_dryrun_findings
§ 2. Nothing here is fitted; every constant comes from the locked amendment.

Run: PYTHONPATH=. uv run python scripts/newsagent_datachannel_v2.py [--chart]
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

from newsagent_datachannel_dryrun import (  # noqa: E402
    CACHE, MC_DRAWS, P_CLIP, SEED, SIGMA_MOM, KAPPA_PI, KAPPA_U,
    as_of, load_nowcasts, month_add, nowcast_asof, vintage_series,
)
from newsagent.config import CSV_OUT, ROOT  # noqa: E402

PLOTS = ROOT / "data" / "analysis" / "plots" / "news_agent"

# ---------------------------------------------------------------------------
# LOCKED CONSTANTS — Amendment 1, § 4f, 2026-08-24. Not fitted, not tuned.
# ---------------------------------------------------------------------------
W_SEP = 2.0          # DC-3a hazard weight, SEP meeting
W_NON = 1.0          # DC-3a hazard weight, non-SEP meeting
B1A = 1.5            # DC-3a slope on allocated clicks, acting on P(MOVE)
TRAJ_DAYS = 45       # daily trajectory window before each decision
DEAD_BAND = 0.005    # 0.5pp directional dead-band (criterion C)
JULY_INTERSECTION = (date(2026, 6, 21), date(2026, 7, 5))   # the dry run's window
JULY_BAR = 0.2144    # news-FV == prior-only on that market
BASE_YEARS_ALL = list(range(1994, 2026))   # DC-3b: ALL years, no ZLB exclusion
MEETINGS_PER_YEAR = 8

# Scheduled FOMC decision dates, published years in advance (vintage-safe).
# SEP = meeting carrying a Summary of Economic Projections.
FOMC = {
    2025: [(date(2025, 1, 29), False), (date(2025, 3, 19), True),
           (date(2025, 4, 30), False), (date(2025, 6, 18), True),
           (date(2025, 7, 30), False), (date(2025, 9, 17), True),
           (date(2025, 10, 29), False), (date(2025, 12, 10), True)],
    2026: [(date(2026, 1, 28), False), (date(2026, 3, 18), True),
           (date(2026, 4, 29), False), (date(2026, 6, 17), True),
           (date(2026, 7, 29), False), (date(2026, 9, 16), True),
           (date(2026, 10, 28), False), (date(2026, 12, 9), True)],
}

# The pre-registered test set: 4 SEP / 4 non-SEP, 3 moves / 5 holds.
# outcome is in the MARKET's framing: 1 = no change, 0 = change.
TEST_SET = [
    {"decision": date(2025, 9, 17), "sep": True, "outcome": 0, "note": "cut 4.50 -> 4.25"},
    {"decision": date(2025, 10, 29), "sep": False, "outcome": 0, "note": "cut 4.25 -> 4.00"},
    {"decision": date(2025, 12, 10), "sep": True, "outcome": 0, "note": "cut 4.00 -> 3.75"},
    {"decision": date(2026, 1, 28), "sep": False, "outcome": 1, "note": "hold"},
    {"decision": date(2026, 3, 18), "sep": True, "outcome": 1, "note": "hold"},
    {"decision": date(2026, 4, 29), "sep": False, "outcome": 1, "note": "hold"},
    {"decision": date(2026, 6, 17), "sep": True, "outcome": 1, "note": "hold"},
    {"decision": date(2026, 7, 29), "sep": False, "outcome": 1,
     "note": "hold — the market that killed v1 (sf-2026-003)"},
]


def base_rate_all_years() -> tuple[float, dict]:
    """DC-3b: per-meeting NO-CHANGE rate over ALL scheduled meetings 1994-2025.

    No ZLB exclusion — that exclusion removed exactly the stretches where the Fed
    held at every meeting, which is the wrong reference class for a per-meeting
    question. Recomputed here rather than hardcoded.
    """
    rows_u = vintage_series("DFEDTARU", "2008-12-01", "2026-08-24", "2026-08-24")
    rows_old = vintage_series("DFEDTAR", "1994-01-01", "2026-08-24", "2026-08-24")
    changes: set[date] = set()
    for rows in (rows_old, rows_u):
        prev = None
        for o in rows:
            if o["value"] in (".", "", None):
                continue
            v, d = float(o["value"]), datetime.fromisoformat(o["date"]).date()
            if prev is not None and abs(v - prev) > 1e-9 and d.year in BASE_YEARS_ALL:
                changes.add(d)
            prev = v
    n_meetings = MEETINGS_PER_YEAR * len(BASE_YEARS_ALL)
    hold = 1.0 - len(changes) / n_meetings
    return hold, {"n_change_dates": len(changes), "n_meetings": n_meetings,
                  "years": len(BASE_YEARS_ALL), "excluded": []}


def remaining_meetings(t: date, sep_year: int) -> list[tuple[date, bool]]:
    """DC-3a horizon: scheduled meetings from t through the end of the SEP year."""
    return [(d, s) for d, s in FOMC.get(sep_year, []) if d >= t]


def allocate(G: float, rem: list[tuple[date, bool]]) -> tuple[float, float, int]:
    """DC-3a Step 3a: (mu_1, share_1, N) — pressure allocated to the NEXT meeting."""
    if not rem:
        return 0.0, 0.0, 0
    ws = [W_SEP if s else W_NON for _, s in rem]
    share1 = ws[0] / sum(ws)
    return G * share1, share1, len(rem)


def p_hold_v2(mu1, logit_h_base: float):
    """DC-3a Step 4a. NOTE the sign: B1A is POSITIVE and acts on P(MOVE)."""
    z = logit_h_base + B1A * np.asarray(mu1)
    p_move = 1.0 / (1.0 + np.exp(-z))
    return np.clip(1.0 - p_move, *P_CLIP)


def p_hold_v1(m_clicks, logit_hold_base_zlb: float):
    """The ORIGINAL DC-3 rule, re-scored on the same meetings so the amendment's
    contribution is separable from the reference-class change (reported, not a bar)."""
    z = logit_hold_base_zlb + (-3.0) * np.abs(np.asarray(m_clicks))
    return np.clip(1.0 / (1.0 + np.exp(-z)), *P_CLIP)


def _series_for(win_start: date, win_end: date) -> dict:
    rt0, rt1 = win_start.isoformat(), win_end.isoformat()
    obs0 = f"{win_start.year - 3}-01-01"
    return {
        "DFEDTARU": vintage_series("DFEDTARU", obs0, rt0, rt1),
        "DFEDTARL": vintage_series("DFEDTARL", obs0, rt0, rt1),
        "PCEPILFE": vintage_series("PCEPILFE", obs0, rt0, rt1),
        "UNRATE": vintage_series("UNRATE", obs0, rt0, rt1),
        "FEDTARMD": vintage_series("FEDTARMD", obs0, rt0, rt1),
        "JCXFEMD": vintage_series("JCXFEMD", obs0, rt0, rt1),
        "UNRATEMD": vintage_series("UNRATEMD", obs0, rt0, rt1),
    }


def trajectory(meeting: dict, nc: dict, logit_h_base: float,
               logit_hold_zlb: float, b1a: float = B1A,
               w_sep: float = W_SEP) -> list[dict]:
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
        if not dkeys:
            t += timedelta(days=1)
            continue
        mid = (up[max(dkeys)] + lo[max(dkeys)]) / 2.0

        # core PCE y/y: published index chained forward through the unpublished
        # months with Cleveland nowcast vintages dated <= t (DC-2, unchanged)
        last_m = max(pce_pub)
        ly, lm = int(last_m[:4]), int(last_m[5:7])
        k = (t.year * 12 + t.month) - (ly * 12 + lm)
        idx = pce_pub[last_m]
        for j in range(1, k + 1):
            ym = month_add((ly, lm), j)
            v = nowcast_asof(nc, ym, "Core PCE Inflation", t)
            if v is None:
                pk = month_add((ly, lm), j - 1)
                bk = month_add((ly, lm), j - 2)
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

        rem = remaining_meetings(t, sep_year)
        ws = [w_sep if s else W_NON for _, s in rem] or [1.0]
        share1 = ws[0] / sum(ws)
        mu1 = G * share1
        p_v2 = float(np.mean(p_hold_v2(mu1, logit_h_base) if b1a == B1A
                             else np.clip(1.0 - 1.0 / (1.0 + np.exp(
                                 -(logit_h_base + b1a * mu1))), *P_CLIP)))

        # the ORIGINAL rule on identical inputs (reported, never a bar)
        m_old = ((r_des - mid) / max(len(rem), 1)) / 0.25
        p_v1 = float(np.mean(p_hold_v1(m_old, logit_hold_zlb)))

        r_des_pt = sep_r + KAPPA_PI * (pi_center - sep_pi) - KAPPA_U * (u_last - sep_u)
        rows.append({
            "date": t.isoformat(), "p_struct": round(p_v2, 4), "p_v1": round(p_v1, 4),
            "gap_pp": round(r_des_pt - mid, 4),
            "G_clicks": round(abs(r_des_pt - mid) / 0.25, 4),
            "share1": round(share1, 4), "n_rem": len(rem),
            "mu1": round(abs(r_des_pt - mid) / 0.25 * share1, 4),
            "pi_core_pce_yoy": round(pi_center, 3), "unrate": u_last,
            "sep_rate": sep_r, "sep_core_pce": sep_pi, "sep_unrate": sep_u,
            "mid_target": mid, "k_unpublished": k, "last_pce_month": last_m,
        })
        t += timedelta(days=1)
    return rows


def directional_violations(rows: list[dict]) -> list[dict]:
    """Criterion C. A release that raises pressure must not raise p_struct.

    Release days are detected as days where a vintage value CHANGED versus the
    previous computed day: the SEP median, or the last published core-PCE month.
    Unemployment carries no pre-registered direction and is excluded, unchanged
    from the dry run.
    """
    out = []
    for a, b in zip(rows, rows[1:]):
        sep_up = b["sep_rate"] - a["sep_rate"]
        pi_new = b["last_pce_month"] != a["last_pce_month"]
        pi_up = b["pi_core_pce_yoy"] - a["pi_core_pce_yoy"]
        implied = 0.0
        if abs(sep_up) > 1e-9:
            implied = -sep_up                      # SEP median up -> hold less likely
        elif pi_new and abs(pi_up) > 1e-9:
            implied = -pi_up
        if implied == 0.0:
            continue
        dp = b["p_struct"] - a["p_struct"]
        if abs(dp) <= DEAD_BAND:
            continue
        if (implied > 0) != (dp > 0):
            out.append({"date": b["date"], "driver": "sep" if abs(sep_up) > 1e-9 else "pi",
                        "implied_sign": "up" if implied > 0 else "down",
                        "dp": round(dp, 4)})
    return out


def brier(ps: list[float], outcome: int) -> float:
    return float(np.mean([(p - outcome) ** 2 for p in ps])) if ps else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", action="store_true")
    args = ap.parse_args()

    h_hold_all, meta_all = base_rate_all_years()
    h_base = 1.0 - h_hold_all
    logit_h_base = math.log(h_base / (1 - h_base))
    # the v1 reference class, kept only to re-score the original rule
    from newsagent_datachannel_dryrun import base_rate as base_rate_zlb
    hold_zlb, meta_zlb = base_rate_zlb()
    logit_hold_zlb = math.log(hold_zlb / (1 - hold_zlb))

    print(f"DC-3b base rate (ALL years 1994-2025, no ZLB exclusion): "
          f"hold={h_hold_all:.4f}  h_base={h_base:.4f}  "
          f"({meta_all['n_change_dates']} change dates / {meta_all['n_meetings']} meetings)")
    print(f"v1 reference class (ZLB-excluded, for the re-score only): hold={hold_zlb:.4f}")
    print()

    nc = load_nowcasts()
    results, all_rows = [], []
    for mk in TEST_SET:
        rows = trajectory(mk, nc, logit_h_base, logit_hold_zlb)
        if not rows:
            print(f"  {mk['decision']}: NO DATA")
            continue
        for r in rows:
            r["meeting"] = mk["decision"].isoformat()
            r["sep"] = mk["sep"]
            r["outcome"] = mk["outcome"]
        all_rows += rows
        dec_row = rows[-1]
        viol = directional_violations(rows)
        p = dec_row["p_struct"]
        results.append({
            "meeting": mk["decision"].isoformat(), "sep": mk["sep"],
            "outcome": mk["outcome"], "note": mk["note"], "n_days": len(rows),
            "p_struct": p, "brier": round((p - mk["outcome"]) ** 2, 4),
            "p_v1": dec_row["p_v1"],
            "brier_v1": round((dec_row["p_v1"] - mk["outcome"]) ** 2, 4),
            "p_base": round(h_hold_all, 4),
            "brier_base": round((h_hold_all - mk["outcome"]) ** 2, 4),
            "gap_pp": dec_row["gap_pp"], "G_clicks": dec_row["G_clicks"],
            "share1": dec_row["share1"], "n_rem": dec_row["n_rem"],
            "mu1": dec_row["mu1"], "violations": len(viol),
            "violation_detail": viol,
        })
        print(f"  {mk['decision']}  {'SEP' if mk['sep'] else '   '}  "
              f"outcome={'hold' if mk['outcome'] else 'MOVE'}  "
              f"G={dec_row['G_clicks']:5.2f}cl  share1={dec_row['share1']:.3f}  "
              f"mu1={dec_row['mu1']:5.3f}  p_struct={p:.3f}  "
              f"brier={(p - mk['outcome']) ** 2:.4f}   (v1 {dec_row['p_v1']:.3f} / "
              f"{(dec_row['p_v1'] - mk['outcome']) ** 2:.4f})  viol={len(viol)}")

    # ---- criterion A: pooled -------------------------------------------------
    mean_b = float(np.mean([r["brier"] for r in results]))
    mean_base = float(np.mean([r["brier_base"] for r in results]))
    mean_v1 = float(np.mean([r["brier_v1"] for r in results]))
    A = mean_b < mean_base

    # ---- criterion B: the July market on the dry run's own window -------------
    j0, j1 = JULY_INTERSECTION
    july = [r for r in all_rows if r["meeting"] == "2026-07-29"
            and j0.isoformat() <= r["date"] <= j1.isoformat()]
    july_b = brier([r["p_struct"] for r in july], 1)
    july_v1 = brier([r["p_v1"] for r in july], 1)
    B = july_b < JULY_BAR

    # ---- criterion C: directional -------------------------------------------
    n_viol = sum(r["violations"] for r in results)
    C = n_viol == 0

    print()
    print(f"A  pooled mean Brier  p_struct={mean_b:.4f}  base-rate-only={mean_base:.4f}"
          f"   -> {'PASS' if A else 'FAIL'}      (v1 rule on same set: {mean_v1:.4f})")
    print(f"B  July intersection  p_struct={july_b:.4f}  bar={JULY_BAR}"
          f"   -> {'PASS' if B else 'FAIL'}      (v1 rule same window: {july_v1:.4f}; "
          f"dry run recorded 0.4223)")
    print(f"C  directional violations={n_viol}"
          f"   -> {'PASS' if C else 'FAIL'}")
    print()
    print(f"VERDICT: {'GO' if (A and B and C) else 'NO-GO'}")

    # ---- reported, never a bar: sensitivity ---------------------------------
    sens = []
    for b1a in (0.5, 1.0, 1.5, 2.0, 3.0):
        ps = []
        for mk in TEST_SET:
            rows = trajectory(mk, nc, logit_h_base, logit_hold_zlb, b1a=b1a)
            if rows:
                ps.append((rows[-1]["p_struct"], mk["outcome"]))
        sens.append({"knob": "B1A", "value": b1a,
                     "pooled_brier": round(float(np.mean([(p - o) ** 2 for p, o in ps])), 4),
                     "mean_p": round(float(np.mean([p for p, _ in ps])), 4)})
    for w in (1.0, 1.5, 2.0, 3.0):
        ps = []
        for mk in TEST_SET:
            rows = trajectory(mk, nc, logit_h_base, logit_hold_zlb, w_sep=w)
            if rows:
                ps.append((rows[-1]["p_struct"], mk["outcome"]))
        sens.append({"knob": "W_SEP", "value": w,
                     "pooled_brier": round(float(np.mean([(p - o) ** 2 for p, o in ps])), 4),
                     "mean_p": round(float(np.mean([p for p, _ in ps])), 4)})
    print("\nsensitivity (REPORTED, not a bar — the verdict is on B1A=1.5 / W_SEP=2.0):")
    for s in sens:
        star = " <- declared" if (s["knob"] == "B1A" and s["value"] == B1A) or \
                                 (s["knob"] == "W_SEP" and s["value"] == W_SEP) else ""
        print(f"  {s['knob']:6s}={s['value']:<4} pooled Brier {s['pooled_brier']:.4f}"
              f"  mean p {s['mean_p']:.3f}{star}")

    CSV_OUT.mkdir(parents=True, exist_ok=True)
    with (CSV_OUT / "newsagent_datachannel_v2_meetings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in results[0] if k != "violation_detail"])
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in r.items() if k != "violation_detail"})
    with (CSV_OUT / "newsagent_datachannel_v2_trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with (CSV_OUT / "newsagent_datachannel_v2_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0].keys()))
        w.writeheader()
        w.writerows(sens)

    out = {"criteria": {"A_pooled": {"p_struct": round(mean_b, 4),
                                     "base_only": round(mean_base, 4),
                                     "v1_rule": round(mean_v1, 4), "pass": A},
                        "B_july": {"p_struct": round(july_b, 4), "bar": JULY_BAR,
                                   "v1_rule": round(july_v1, 4), "pass": B},
                        "C_directional": {"violations": n_viol, "pass": C}},
           "verdict": "GO" if (A and B and C) else "NO-GO",
           "base_rate": {"hold_all_years": round(h_hold_all, 4), **meta_all},
           "constants": {"W_SEP": W_SEP, "W_NON": W_NON, "B1A": B1A,
                         "KAPPA_PI": KAPPA_PI, "KAPPA_U": KAPPA_U,
                         "SIGMA_MOM": SIGMA_MOM, "MC_DRAWS": MC_DRAWS, "SEED": SEED},
           "meetings": results, "sensitivity": sens}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "v2_results.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {CACHE / 'v2_results.json'} and 3 CSVs")

    if args.chart:
        chart(all_rows, results)
    return 0


def chart(all_rows: list[dict], results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharey=True)
    for ax, r in zip(axes.ravel(), results):
        rows = [x for x in all_rows if x["meeting"] == r["meeting"]]
        xs = [datetime.fromisoformat(x["date"]).date() for x in rows]
        ax.plot(xs, [x["p_struct"] for x in rows], color="#cc5c44", lw=2,
                label="p_struct (v2)")
        ax.plot(xs, [x["p_v1"] for x in rows], color="#7a7a72", lw=1.2, ls=":",
                label="v1 rule")
        ax.axhline(r["p_base"], color="#9a9a90", lw=1, ls="--", label="base-rate only")
        ax.scatter([xs[-1]], [r["outcome"]], marker="*", s=160,
                   color="#2f7d55" if r["outcome"] else "#a33", zorder=5,
                   label="outcome")
        ax.set_title(f"{r['meeting']} {'SEP' if r['sep'] else ''}\n"
                     f"{'hold' if r['outcome'] else 'MOVE'} · Brier {r['brier']:.3f}",
                     fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        ax.grid(alpha=.25)
    axes.ravel()[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Data-channel v2 — amended reaction function (DC-3a allocate + "
                 "DC-3b all-years base rate) on 8 resolved FOMC decisions\n"
                 "p_struct = P(no change at THIS meeting); star = realised outcome "
                 "(1 = no change). Vintage-only inputs.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = PLOTS / "newsagent_datachannel_v2.png"
    fig.savefig(out, dpi=140)
    print(f"chart -> {out}")


if __name__ == "__main__":
    sys.exit(main())
