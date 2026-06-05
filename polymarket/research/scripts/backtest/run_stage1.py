"""Phase 5 Stage 1 orchestrator — iterates 72-run matrix.

3 cohorts × 4 resolution buckets × 2 sizing rules × 3 windows = 72 runs.

Each writes:
  data/backtests/{run_id}.parquet      — per-signal audit log
  data/backtests/{run_id}_summary.json — headline metrics

Estimated total runtime: ~90 min on 4 threads. Re-running is idempotent —
skips run_ids whose audit + summary already exist.
"""
import json
import time
from datetime import date
from pathlib import Path

from scripts.backtest.cohort_filters import COHORTS
from scripts.backtest.walkforward import run_backtest, BACKTESTS_DIR

WINDOWS = [
    ("2024", date(2024, 1, 1), date(2024, 12, 1)),
    ("2025", date(2025, 1, 1), date(2025, 12, 1)),
    ("2026", date(2026, 1, 1), date(2026, 5, 1)),
]
BUCKETS = ["2d", "7d", "30d", "60d"]
SIZING = ["fixed_pct", "leader_proportional"]


def main() -> None:
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

    matrix: list[tuple] = []
    for cohort_name in COHORTS:
        for bucket in BUCKETS:
            for sizing in SIZING:
                for win_label, win_start, win_end in WINDOWS:
                    matrix.append((cohort_name, bucket, sizing, win_label, win_start, win_end))

    total = len(matrix)
    print(f"Stage 1 matrix: {total} runs "
          f"({len(COHORTS)} cohorts × {len(BUCKETS)} buckets × "
          f"{len(SIZING)} sizing × {len(WINDOWS)} windows)\n")

    t_all = time.time()
    n_skipped = 0
    n_run = 0
    for i, (cohort_name, bucket, sizing, win_label, win_start, win_end) in enumerate(matrix, 1):
        # Skip-if-exists
        run_id = (f"{cohort_name}__{bucket}__{sizing}__{win_label}__"
                  f"b2t4n2")  # default slippage params for Stage 1
        audit = BACKTESTS_DIR / f"{run_id}.parquet"
        summary = BACKTESTS_DIR / f"{run_id}_summary.json"
        if audit.exists() and summary.exists():
            n_skipped += 1
            print(f"[{i}/{total}] {run_id} — cached, skip")
            continue

        print(f"\n[{i}/{total}] starting...")
        try:
            run_backtest(
                cohort_name=cohort_name,
                cohort_fn=COHORTS[cohort_name],
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
    print(f"\n=== Stage 1 complete ===")
    print(f"  ran:     {n_run}")
    print(f"  cached:  {n_skipped}")
    print(f"  elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
