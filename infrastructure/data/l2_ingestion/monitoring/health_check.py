"""Health check: verifies the ingestion chain is alive - capture files are being
written, no excessive WS gaps, compression is keeping up, and the cloud sync is
current - emitting a GREEN/YELLOW/RED status report for monitoring.

Read-only. Imports nothing from the other l2_ingestion packages and uses only
the standard library. Talks to the rest of the system purely by inspecting files
on disk (the same "communication through files" pattern as everything else).

Usage:
    python monitoring/health_check.py            # human-readable report
    python monitoring/health_check.py --json     # machine-readable (for alerts)
    python monitoring/health_check.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --- Paths -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../l2_ingestion/monitoring
PKG_ROOT = HERE.parent                          # .../l2_ingestion
DEFAULT_DATA_DIR = PKG_ROOT / "data"

# --- Status levels ---------------------------------------------------------
GREEN, YELLOW, RED = "GREEN", "YELLOW", "RED"
_RANK = {GREEN: 0, YELLOW: 1, RED: 2}

# --- Thresholds (seconds unless noted) -------------------------------------
CAPTURE_YELLOW_S = 60 * 60          # newest shard older than 1h -> warn
CAPTURE_RED_S = 2 * 60 * 60         # ... older than 2h -> capture is down
COMPRESS_AGE_S = 2 * 60 * 60        # raw shards older than this should be processed
SYNC_YELLOW_S = 2 * 60 * 60         # sync log staler than this -> warn
SYNC_RED_S = 6 * 60 * 60            # ... much staler -> sync is down
GAP_WINDOW_S = 24 * 60 * 60         # look back this far in capture_gaps.jsonl


def now_ts() -> float:
    return time.time()


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 90 * 60:
        return f"{seconds // 60}m"
    return f"{seconds / 3600:.1f}h"


def newest_mtime(paths: list[Path]) -> float | None:
    return max((p.stat().st_mtime for p in paths), default=None)


# ---------------------------------------------------------------------------
# Component checks. Each returns dict(component, status, reason, details).
# ---------------------------------------------------------------------------
def check_capture(raw_dir: Path) -> dict[str, Any]:
    shards = list(raw_dir.glob("*/*.jsonl.gz")) if raw_dir.exists() else []
    newest = newest_mtime(shards)
    if newest is None:
        return _result("capture", RED, "no raw shards found - capture has never written data",
                       {"shard_count": 0, "newest_age": None})
    age = now_ts() - newest
    details = {"shard_count": len(shards), "newest_age": fmt_age(age),
               "newest_age_seconds": round(age)}
    if age >= CAPTURE_RED_S:
        return _result("capture", RED, f"newest shard is {fmt_age(age)} old (>2h) - capture appears DOWN", details)
    if age >= CAPTURE_YELLOW_S:
        return _result("capture", YELLOW, f"newest shard is {fmt_age(age)} old - slower than expected", details)
    return _result("capture", GREEN, f"writing - newest shard {fmt_age(age)} old", details)


def check_gaps(gap_log: Path) -> dict[str, Any]:
    if not gap_log.exists():
        return _result("gaps", YELLOW, "no capture_gaps.jsonl found - capture may not have run",
                       {"disconnects_24h": 0, "stale_warnings_24h": 0})
    cutoff = now_ts() - GAP_WINDOW_S
    disconnects = stale = bad = 0
    last_disconnect = None
    for line in gap_log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            ts = datetime.fromisoformat(str(rec.get("ts", "")).replace("Z", "+00:00")).timestamp()
        except Exception:
            bad += 1
            continue
        if ts < cutoff:
            continue
        et = rec.get("event_type")
        if et == "disconnect_or_error":
            disconnects += 1
            last_disconnect = rec.get("ts")
        elif et == "stale_warning":
            stale += 1
    details = {"disconnects_24h": disconnects, "stale_warnings_24h": stale,
               "last_disconnect": last_disconnect, "unparseable_lines": bad}
    if disconnects > 5 or stale > 2:
        return _result("gaps", RED, f"{disconnects} disconnects / {stale} stale warnings in 24h - unstable", details)
    if disconnects > 0 or stale > 0:
        return _result("gaps", YELLOW, f"{disconnects} disconnects / {stale} stale warnings in 24h", details)
    return _result("gaps", GREEN, "no disconnects or stale warnings in 24h", details)


def check_compression(raw_dir: Path, parquet_dir: Path) -> dict[str, Any]:
    processed_file = parquet_dir / "_processed.txt"
    processed: set[str] = set()
    if processed_file.exists():
        processed = {ln.strip() for ln in processed_file.read_text(encoding="utf-8").splitlines() if ln.strip()}

    shards = list(raw_dir.glob("*/*.jsonl.gz")) if raw_dir.exists() else []
    cutoff = now_ts() - COMPRESS_AGE_S
    backlog = [p for p in shards
               if (now_ts() - p.stat().st_mtime) >= COMPRESS_AGE_S
               and str(p.resolve()) not in processed]
    details = {"backlog_count": len(backlog),
               "backlog_files": [p.name for p in backlog[:10]],
               "processed_count": len(processed)}
    if len(backlog) > 2:
        return _result("compression", RED, f"{len(backlog)} completed shards unprocessed (>2h old) - pipeline behind", details)
    if len(backlog) >= 1:
        return _result("compression", YELLOW, f"{len(backlog)} completed shard(s) not yet processed", details)
    return _result("compression", GREEN, "no processing backlog", details)


def check_sync(data_dir: Path) -> dict[str, Any]:
    sync_log = data_dir / "logs" / "sync_cloud.log"
    if not sync_log.exists():
        return _result("sync", YELLOW, "no sync log found - cloud sync may never have run",
                       {"sync_log_age": None})
    age = now_ts() - sync_log.stat().st_mtime
    details = {"sync_log_age": fmt_age(age), "sync_log_age_seconds": round(age)}
    if age >= SYNC_RED_S:
        return _result("sync", RED, f"sync log {fmt_age(age)} old (>6h) - cloud sync appears DOWN", details)
    if age >= SYNC_YELLOW_S:
        return _result("sync", YELLOW, f"sync log {fmt_age(age)} old - sync may be lagging", details)
    return _result("sync", GREEN, f"last sync {fmt_age(age)} ago", details)


def check_inventory(raw_dir: Path, parquet_dir: Path) -> dict[str, Any]:
    raw_days = sorted(p.name for p in raw_dir.iterdir() if p.is_dir()) if raw_dir.exists() else []
    parquet_days = sorted(p.name for p in parquet_dir.iterdir() if p.is_dir() and not p.name.startswith("_")) if parquet_dir.exists() else []
    raw_bytes = sum(p.stat().st_size for p in raw_dir.rglob("*.jsonl.gz")) if raw_dir.exists() else 0
    parquet_bytes = sum(p.stat().st_size for p in parquet_dir.rglob("*.parquet")) if parquet_dir.exists() else 0
    details = {
        "raw_days": len(raw_days), "parquet_days": len(parquet_days),
        "raw_mb": round(raw_bytes / 1e6, 1), "parquet_mb": round(parquet_bytes / 1e6, 1),
        "date_range": f"{parquet_days[0]}..{parquet_days[-1]}" if parquet_days else "none",
    }
    if not raw_days and not parquet_days:
        return _result("inventory", RED, "no data captured at all", details)
    return _result("inventory", GREEN,
                   f"{len(parquet_days)} day(s) parquet, {details['parquet_mb']}MB parquet / {details['raw_mb']}MB raw",
                   details)


def _result(component: str, status: str, reason: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"component": component, "status": status, "reason": reason, "details": details}


# ---------------------------------------------------------------------------
# Report assembly + rendering
# ---------------------------------------------------------------------------
def run_checks(data_dir: Path) -> dict[str, Any]:
    raw_dir = data_dir / "raw"
    parquet_dir = data_dir / "parquet"
    gap_log = data_dir / "capture_gaps.jsonl"
    components = [
        check_capture(raw_dir),
        check_gaps(gap_log),
        check_compression(raw_dir, parquet_dir),
        check_sync(data_dir),
        check_inventory(raw_dir, parquet_dir),
    ]
    overall = max((c["status"] for c in components), key=lambda s: _RANK[s])
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "data_dir": str(data_dir),
        "overall": overall,
        "components": components,
    }


def render_text(report: dict[str, Any], color: bool) -> str:
    marker = {GREEN: "[OK]", YELLOW: "[! ]", RED: "[XX]"}
    ansi = {GREEN: "\033[32m", YELLOW: "\033[33m", RED: "\033[31m", "reset": "\033[0m"}

    def tag(status: str) -> str:
        label = f"{marker[status]} {status:<6}"
        return f"{ansi[status]}{label}{ansi['reset']}" if color else label

    lines = [
        f"L2 Ingestion Health - {report['generated_at']}",
        f"Overall: {tag(report['overall'])}",
        "-" * 64,
    ]
    for c in report["components"]:
        lines.append(f"{tag(c['status'])} {c['component']:<12} {c['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L2 ingestion health check (read-only)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--exit-code", action="store_true",
                        help="exit 0=GREEN, 1=YELLOW, 2=RED (for alerting)")
    args = parser.parse_args(argv)

    report = run_checks(args.data_dir)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report, color=sys.stdout.isatty()))

    return _RANK[report["overall"]] if args.exit_code else 0


if __name__ == "__main__":
    raise SystemExit(main())
