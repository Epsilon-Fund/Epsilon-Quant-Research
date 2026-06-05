"""Phase 5 Stage 1.5 orchestrator.

3 cohorts × 4 buckets × 2 sizing × 3 windows = 72 runs.
  - Primary: 2025 (Jan–Dec), 2026 (Jan–Apr 23)
  - Sidebar (low-confidence): 2024 (Jan–Dec) — flagged for data sparsity

Outputs to data/backtests/stage15/.
"""
import time
from datetime import date

from scripts.backtest.cohort_filters_stage15 import COHORTS_STAGE15
from scripts.backtest.walkforward_stage15 import run_backtest_stage15, BACKTESTS_DIR

WINDOWS = [
    # (label, start, end, primary)
    ("2024", date(2024, 1, 1), date(2025, 1, 1), False),     # sidebar
    ("2025", date(2025, 1, 1), date(2026, 1, 1), True),
    ("2026", date(2026, 1, 1), date(2026, 4, 23), True),
]
BUCKETS = ["2d", "7d", "30d", "60d"]
SIZING = ["fixed_pct", "leader_proportional"]


def main() -> None:
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

    matrix = []
    for cohort_name in COHORTS_STAGE15:
        for bucket in BUCKETS:
            for sizing in SIZING:
                for win_label, win_start, win_end, primary in WINDOWS:
                    matrix.append((cohort_name, bucket, sizing,
                                   win_label, win_start, win_end, primary))

    total = len(matrix)
    print(f"Stage 1.5 matrix: {total} runs "
          f"({len(COHORTS_STAGE15)} cohorts × {len(BUCKETS)} buckets × "
          f"{len(SIZING)} sizing × {len(WINDOWS)} windows)\n")

    t_all = time.time()
    n_run = 0
    n_skip = 0
    for i, (cohort_name, bucket, sizing, win_label, win_start, win_end, primary) in enumerate(matrix, 1):
        run_id = f"{cohort_name}__{bucket}__{sizing}__{win_label}__stage15"
        audit = BACKTESTS_DIR / f"{run_id}.parquet"
        summary = BACKTESTS_DIR / f"{run_id}_summary.json"
        if audit.exists() and summary.exists():
            n_skip += 1
            print(f"[{i}/{total}] {run_id} — cached, skip")
            continue
        tag = "PRIMARY" if primary else "sidebar"
        print(f"\n[{i}/{total}] starting ({tag})...")
        try:
            run_backtest_stage15(
                cohort_name=cohort_name,
                cohort_fn=COHORTS_STAGE15[cohort_name],
                resolution_bucket=bucket,
                sizing_rule=sizing,
                test_window_start=win_start,
                test_window_end=win_end,
                window_label=win_label,
            )
            n_run += 1
        except Exception as e:
            print(f"  !! FAILED: {e!r}")

    elapsed = time.time() - t_all
    print(f"\n=== Stage 1.5 complete ===")
    print(f"  ran:    {n_run}")
    print(f"  cached: {n_skip}")
    print(f"  elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
