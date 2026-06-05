"""Summarize Block A0 capture shards and gap logs."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "data" / "live_clob" / "block_a0"
DEFAULT_OUT = ROOT / "notes" / "block_a0_capture_status.md"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def iter_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def latest_run_dir() -> Path:
    runs = sorted([p for p in DEFAULT_RUN_ROOT.glob("*") if p.is_dir()])
    if not runs:
        raise SystemExit(f"no run dirs under {DEFAULT_RUN_ROOT}")
    return runs[-1]


def summarize(run_dir: Path) -> tuple[pd.DataFrame, Counter[str], dict[str, Counter[str]], list[dict[str, Any]]]:
    rows = []
    total_counts: Counter[str] = Counter()
    market_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.name == "capture_gaps.jsonl":
            continue
        shard_counts: Counter[str] = Counter()
        first = None
        last = None
        for rec in iter_jsonl(path):
            event_type = str(rec.get("event_type") or "unknown")
            shard_counts[event_type] += 1
            total_counts[event_type] += 1
            ts = rec.get("received_at")
            first = ts if first is None else min(first, ts)
            last = ts if last is None else max(last, ts)
            for asset in rec.get("assets") or []:
                market_counts[str(asset.get("slug") or asset.get("market_id") or "")][event_type] += 1
        rows.append(
            {
                "shard": path.name,
                "first_received_at": first,
                "last_received_at": last,
                **dict(shard_counts),
            }
        )
    gaps = []
    gap_path = run_dir / "capture_gaps.jsonl"
    if gap_path.exists():
        gaps = list(iter_jsonl(gap_path))
    shards = pd.DataFrame(rows)
    if not shards.empty:
        shards["first_received_dt"] = pd.to_datetime(shards["first_received_at"], utc=True, errors="coerce")
        shards["last_received_dt"] = pd.to_datetime(shards["last_received_at"], utc=True, errors="coerce")
        shards = shards.sort_values("first_received_dt")
        shards["prev_last_received_dt"] = shards["last_received_dt"].shift(1)
        shards["inter_shard_gap_seconds"] = (
            shards["first_received_dt"] - shards["prev_last_received_dt"]
        ).dt.total_seconds()
        shards = shards.drop(columns=["first_received_dt", "last_received_dt", "prev_last_received_dt"])
    return shards, total_counts, market_counts, gaps


def write_note(run_dir: Path, out: Path) -> None:
    shards, total_counts, market_counts, gaps = summarize(run_dir)
    gap_counts = Counter(str(row.get("event_type")) for row in gaps)
    market_rows = []
    for slug, counts in market_counts.items():
        row = {"market": slug, **dict(counts)}
        row["total"] = sum(counts.values())
        market_rows.append(row)
    market_df = pd.DataFrame(market_rows).sort_values("total", ascending=False) if market_rows else pd.DataFrame()
    out.write_text(
        "# Block A0 Capture Status\n\n"
        f"Run dir: `{display_path(run_dir)}`\n\n"
        "## Total Event Counts\n\n"
        f"```json\n{json.dumps(dict(total_counts), indent=2, sort_keys=True)}\n```\n\n"
        "## Gap Log Counts\n\n"
        f"```json\n{json.dumps(dict(gap_counts), indent=2, sort_keys=True)}\n```\n\n"
        "## Shards\n\n"
        + (shards.fillna(0).to_csv(index=False) if not shards.empty else "No shards found.\n")
        + "\n## Per-Market Counts\n\n"
        + (market_df.fillna(0).to_csv(index=False) if not market_df.empty else "No market events found.\n"),
        encoding="utf-8",
    )
    print(f"wrote {display_path(out)}")
    print(f"total counts: {dict(total_counts)}")
    print(f"gap counts: {dict(gap_counts)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir or latest_run_dir()
    write_note(run_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
