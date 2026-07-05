"""News-agent v0 gate: score forecasts vs mids per the pre-registered metrics.

Pre-registration: notes/news_agent/newsagent_v0_gate_preregistration.md
  M1  pooled Brier(ours) - Brier(mid) <= +0.05   (family-clustered bootstrap CI reported)
  M2a median |p_ours - mid| >= 2pp
  M2b p90 |p_ours - mid| <= 35pp AND pooled Brier(ours) <= 0.25
  M2c sign-agreement >= 65% on consecutive-snapshot pairs where |mid move| >= 8pp
  M3  band stats (descriptive only)

Inputs:  data/newsagent/v0/universe_selected.json
         data/newsagent/v0/forecasts/<slug>__<date>.json  ({p_pct, band_lo_pct, band_hi_pct, ...})
Outputs: data/analysis/csv_outputs/news_agent/newsagent_v0_pairs.csv
         data/analysis/csv_outputs/news_agent/newsagent_v0_metrics.csv
         data/analysis/csv_outputs/news_agent/newsagent_v0_family_table.csv
         data/analysis/plots/news_agent/newsagent_v0_timeseries.png
         data/analysis/plots/news_agent/newsagent_v0_gap_hist.png
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "newsagent" / "v0"
FC = RAW / "forecasts"
CSV_OUT = ROOT / "data" / "analysis" / "csv_outputs" / "news_agent"
PLOTS = ROOT / "data" / "analysis" / "plots" / "news_agent"

SNAPSHOT_DATES = ["2026-06-08", "2026-06-11", "2026-06-14", "2026-06-17",
                  "2026-06-20", "2026-06-23", "2026-06-26", "2026-06-29"]


def load_pairs() -> list[dict]:
    selected = json.loads((RAW / "universe_selected.json").read_text())
    pairs = []
    for m in selected:
        for d in SNAPSHOT_DATES:
            mid = m.get(f"mid_{d}", "")
            f = FC / f"{m['slug']}__{d}.json"
            if mid == "" or not f.exists():
                continue
            fc = json.loads(f.read_text())
            p = float(fc["p_pct"]) / 100.0
            mid = float(mid)
            y = int(m["outcome_yes"])
            pairs.append({
                "slug": m["slug"], "family": m["family"], "date": d,
                "question": m["question"][:80], "outcome_yes": y,
                "p_ours": round(p, 4), "mid": mid,
                "band_lo": round(float(fc["band_lo_pct"]) / 100.0, 4),
                "band_hi": round(float(fc["band_hi_pct"]) / 100.0, 4),
                "gap": round(p - mid, 4),
                "brier_ours": round((p - y) ** 2, 5),
                "brier_mid": round((mid - y) ** 2, 5),
            })
    return pairs


def family_bootstrap_ci(pairs: list[dict], n_draws: int = 1000, seed: int = 7) -> tuple[float, float]:
    """CI on mean(brier_ours - brier_mid), resampling event families with replacement."""
    rng = random.Random(seed)
    fams: dict[str, list[float]] = {}
    for r in pairs:
        fams.setdefault(r["family"], []).append(r["brier_ours"] - r["brier_mid"])
    names = list(fams)
    stats = []
    for _ in range(n_draws):
        draw = [x for f in (rng.choice(names) for _ in names) for x in fams[f]]
        stats.append(sum(draw) / len(draw))
    stats.sort()
    return stats[int(0.025 * n_draws)], stats[int(0.975 * n_draws)]


def main() -> None:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    n = len(pairs)
    print(f"{n} scored (market, snapshot) pairs across "
          f"{len({r['family'] for r in pairs})} families / {len({r['slug'] for r in pairs})} markets")

    brier_ours = sum(r["brier_ours"] for r in pairs) / n
    brier_mid = sum(r["brier_mid"] for r in pairs) / n
    diff = brier_ours - brier_mid
    ci_lo, ci_hi = family_bootstrap_ci(pairs)
    abs_gaps = sorted(abs(r["gap"]) for r in pairs)
    med_gap = abs_gaps[n // 2]
    p90_gap = abs_gaps[int(0.9 * (n - 1))]

    # M2c tracking: consecutive snapshots within a market where the mid moved >= 8pp
    moves, agree = 0, 0
    by_slug: dict[str, list[dict]] = {}
    for r in pairs:
        by_slug.setdefault(r["slug"], []).append(r)
    for rows in by_slug.values():
        rows.sort(key=lambda r: r["date"])
        for a, b in zip(rows, rows[1:]):
            dmid = b["mid"] - a["mid"]
            if abs(dmid) >= 0.08:
                moves += 1
                dours = b["p_ours"] - a["p_ours"]
                if dours * dmid > 0:
                    agree += 1
    track = agree / moves if moves else float("nan")

    # M3 descriptive band stats
    widths = sorted(r["band_hi"] - r["band_lo"] for r in pairs)
    last_rows = [max(rows, key=lambda r: r["date"]) for rows in by_slug.values()]
    band_hits = sum(1 for r in last_rows
                    if (r["band_lo"] >= 0.5) == bool(r["outcome_yes"]) or r["band_lo"] < 0.5 < r["band_hi"])

    metrics = {
        "n_pairs": n, "n_markets": len(by_slug), "n_families": len({r['family'] for r in pairs}),
        "brier_ours": round(brier_ours, 4), "brier_mid": round(brier_mid, 4),
        "brier_diff": round(diff, 4), "brier_diff_ci_lo": round(ci_lo, 4),
        "brier_diff_ci_hi": round(ci_hi, 4),
        "M1_pass_diff_le_0.05": diff <= 0.05,
        "median_abs_gap": round(med_gap, 4), "M2a_pass_ge_2pp": med_gap >= 0.02,
        "p90_abs_gap": round(p90_gap, 4),
        "M2b_pass": p90_gap <= 0.35 and brier_ours <= 0.25,
        "n_big_mid_moves": moves, "tracking_agreement": round(track, 4) if moves else "n/a",
        "M2c_pass_ge_65pct": (track >= 0.65) if moves else "no-moves",
        "band_width_median": round(widths[len(widths) // 2], 4),
        "band_last_snapshot_consistent": f"{band_hits}/{len(last_rows)}",
    }
    metrics["GATE_PASS"] = bool(metrics["M1_pass_diff_le_0.05"] and metrics["M2a_pass_ge_2pp"]
                                and metrics["M2b_pass"]
                                and (metrics["M2c_pass_ge_65pct"] is True or moves == 0))

    with open(CSV_OUT / "newsagent_v0_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)
    with open(CSV_OUT / "newsagent_v0_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)

    fam_rows = []
    for fam in sorted({r["family"] for r in pairs}):
        sub = [r for r in pairs if r["family"] == fam]
        fam_rows.append({
            "family": fam, "n_pairs": len(sub),
            "n_markets": len({r['slug'] for r in sub}),
            "brier_ours": round(sum(r["brier_ours"] for r in sub) / len(sub), 4),
            "brier_mid": round(sum(r["brier_mid"] for r in sub) / len(sub), 4),
            "median_abs_gap": round(sorted(abs(r["gap"]) for r in sub)[len(sub) // 2], 4),
        })
    with open(CSV_OUT / "newsagent_v0_family_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fam_rows[0].keys()))
        w.writeheader()
        w.writerows(fam_rows)

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        slugs = sorted(by_slug)
        fig, axes = plt.subplots(5, 2, figsize=(13, 16), sharex=False)
        for ax, slug in zip(axes.flat, slugs):
            rows = sorted(by_slug[slug], key=lambda r: r["date"])
            xs = [r["date"][5:] for r in rows]
            ax.plot(xs, [r["mid"] for r in rows], "o-", color="#444", label="PM mid")
            ax.plot(xs, [r["p_ours"] for r in rows], "s-", color="#0b6", label="news-agent p")
            ax.fill_between(xs, [r["band_lo"] for r in rows], [r["band_hi"] for r in rows],
                            color="#0b6", alpha=0.15, label="80% band")
            ax.axhline(rows[0]["outcome_yes"], color="#c33", lw=1, ls="--", label="outcome")
            ax.set_title(rows[0]["question"][:60], fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=7)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle("News-agent v0: our fair value (band) vs PM mid vs outcome", fontsize=11)
        fig.tight_layout()
        fig.savefig(PLOTS / "newsagent_v0_timeseries.png", dpi=110)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist([r["gap"] * 100 for r in pairs], bins=25, color="#0b6", alpha=0.8)
        ax2.set_xlabel("our % − PM mid (pp)")
        ax2.set_ylabel("pairs")
        ax2.set_title("v0 gap distribution")
        fig2.tight_layout()
        fig2.savefig(PLOTS / "newsagent_v0_gap_hist.png", dpi=110)
        print("plots written")
    except ImportError:
        print("matplotlib unavailable — plots skipped")


if __name__ == "__main__":
    main()
