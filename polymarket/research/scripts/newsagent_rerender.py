"""Re-render the Observatory page from a snapshot ALREADY on disk. Presentation only.

    PYTHONPATH=. uv run python scripts/newsagent_rerender.py --date 2026-08-25

WHY THIS EXISTS. A presentation-only round (v3.5 was one) changes how the page
displays numbers it does not change. The obvious way to rebuild the page —
`run_daily.py --stage publish` — is NOT that: it re-runs Stage B, and Stage B is
not idempotent for a date it has already published. `fv_state[slug]["counted"]`
already holds today's article cache keys, so a second pass sees **zero new
articles**, and the rebuilt page would show an empty FV-construction breakdown
and a "no new qualifying evidence" driver list on every card. The published
number would survive; the explanation of it would not.

So this script never recomputes anything. It reads:

  data/newsagent/live/<date>/<slug>.{market,packet}.json   — what the loop fetched
  data/newsagent/showcase/showcase.json                    — what the loop computed

reassembles the exact `snapshots` list `dashboard.publish` was given on the day,
and re-runs only the rendering. It does not touch `fv_state`, `fv_series`,
`priors.json`, the feature cache or the append-only ledger — it opens none of them
for writing. It refuses to run when the stored showcase's numbers are not the ones
published for `--date` (checked against `fv_series`, not against a timestamp), and
it re-checks after rendering that no published number moved: silently rendering one
day's numbers under another day's heading is exactly the quiet-failure class this
project keeps finding.

Output: `data/newsagent/showcase/{index.html,showcase.json}`, overwritten in place.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newsagent import config, dashboard   # noqa: E402


def _assert_showcase_is_for(date: str, stored: dict) -> None:
    """Refuse to render one day's numbers under another day's date.

    NOT checked against `generated_at`: that is the moment the page was RENDERED,
    which for a presentation-only round is today, not the snapshot's date. The
    honest marker is the numbers themselves — every stored card's fair value must
    be the live `fv_series` point for `date`. Silently rendering yesterday's
    numbers under today's heading is exactly the quiet-failure class this project
    keeps finding, so it is an error rather than a warning.
    """
    series_path = config.DATA / "fv_series.json"
    series = json.loads(series_path.read_text()) if series_path.exists() else {}
    wrong = []
    for c in stored.get("markets", []):
        pt = next((p for p in reversed(series.get(c["slug"], []))
                   if p.get("date") == date and p.get("segment", "live") == "live"), None)
        if pt is None or pt.get("fv_pct") != c["fv_pct"]:
            wrong.append(c["slug"])
    if wrong:
        raise SystemExit(
            f"the stored showcase does not match the published {date} numbers on "
            f"{len(wrong)} market(s) — e.g. {wrong[0]}. It was built from a "
            "different day's run; re-run the daily loop rather than re-rendering.")


def _snapshots_from_disk(date: str) -> tuple[list[dict], dict]:
    """Rebuild the snapshot list `dashboard.publish` saw, from stored artifacts."""
    day = config.DATA / date
    if not day.is_dir():
        raise SystemExit(f"no day directory for {date}: {day}")
    sc_path = config.SHOWCASE / "showcase.json"
    if not sc_path.exists():
        raise SystemExit(f"no stored showcase to re-render: {sc_path}")
    stored = json.loads(sc_path.read_text())
    _assert_showcase_is_for(date, stored)

    snapshots = []
    for c in stored["markets"]:
        slug = c["slug"]
        mf, pf = day / f"{slug[:80]}.market.json", day / f"{slug[:80]}.packet.json"
        if not (mf.exists() and pf.exists()):
            print(f"  WARN no stored packet/market for {slug[:52]} — skipped")
            continue
        mkt, pkt = json.loads(mf.read_text()), json.loads(pf.read_text())
        # stage_b, exactly as build_showcase reads it back out of the card
        sb = {
            "mtype": c.get("mtype", ""), "n_relevant": c.get("n_relevant", 0),
            "tract": c.get("tract", "news"), "tract_note": c.get("tract_note", ""),
            "method": c.get("method", "news"),
            "p0_source": c.get("p0_source", "onboarding_prior"),
            "p0_used_pct": c.get("p0_used_pct"),
            "n_double_counted": c.get("n_double_counted", 0),
            "market_implied": c.get("market_implied"),
            "evidence_quality": c.get("evidence_quality"),
            "breakdown": c.get("breakdown"), "bias": c.get("bias"),
            "gdelt": c.get("gdelt"),
            "divergence": {"gap_pp": c["gap_pp"], "flag": c["divergence_flag"]},
        }
        q = c.get("evidence_quality") or {}
        if q.get("half_pp") is not None:
            sb["half_pp"] = q["half_pp"]
        snapshots.append({
            "market": mkt, "packet": pkt,
            "forecast": {"p_pct": c["fv_pct"], "band_lo_pct": c["band"][0],
                         "band_hi_pct": c["band"][1]},
            "drivers": c.get("drivers", []), "region": c.get("region", ""),
            "stage_b": sb, "sf_id": c.get("sf_id", ""),
        })
    return snapshots, stored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="day directory to re-render, YYYY-MM-DD")
    args = ap.parse_args()

    snapshots, stored = _snapshots_from_disk(args.date)
    series_path = config.DATA / "fv_series.json"
    fv_series = json.loads(series_path.read_text()) if series_path.exists() else {}

    jpath, hpath = dashboard.publish(snapshots, fv_series)
    fresh = json.loads(Path(jpath).read_text())

    # A re-render that changes a published number is a bug, not a re-render. Check
    # it here rather than trusting the read-only intent of the code above.
    before = {c["slug"]: (c["fv_pct"], tuple(c["band"])) for c in stored["markets"]}
    moved = [c["slug"] for c in fresh["markets"]
             if c["slug"] in before and before[c["slug"]] != (c["fv_pct"], tuple(c["band"]))]
    if moved:
        raise SystemExit("re-render CHANGED published numbers on: " + ", ".join(moved))

    groups = fresh.get("evidence_groups") or []
    print(f"  {len(fresh['markets'])} markets re-rendered, 0 published numbers changed")
    for g in groups:
        s = (f"settled Brier {g['settled_brier']:.4f} (n={g['settled_n']})"
             if g.get("settled_n") else "no settled markets yet")
        print(f"    {g['group']:<16} {g['n']:>2} markets   {s}")
    mv = fresh.get("movement") or {}
    if mv.get("n"):
        print(f"  evidence has moved the published number a mean of "
              f"{mv['mean_abs_pp']}pp off prior; "
              f"{mv['share_still_ge_1pp']:.0%} still within 1pp")
    print(f"  dashboard -> {hpath}\n  data      -> {jpath}")


if __name__ == "__main__":
    main()
