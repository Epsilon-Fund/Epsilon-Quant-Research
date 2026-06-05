"""Wait for A0c captures to finish, write final summaries, then stop caffeinate."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAIN_RUN_ID = "block_a0c_targeted_20260529_morning"
CRYPTO_ROLL_ID = "block_a0c_crypto_roll_20260529_morning"
MAIN_RUN_DIR = ROOT / "data" / "live_clob" / "block_a0c" / MAIN_RUN_ID
CRYPTO_ROLL_DIR = ROOT / "data" / "live_clob" / "block_a0c_crypto_roll" / CRYPTO_ROLL_ID
LOG_PATH = ROOT / "notes" / "block_a0c_finalize_when_done.log"
MAIN_FINAL_NOTE = ROOT / "notes" / "block_a0c_capture_status_final.md"
ROLL_FINAL_NOTE = ROOT / "notes" / "block_a0c_crypto_roll_status_final.md"
COMBINED_FINAL_NOTE = ROOT / "notes" / "block_a0c_auto_final_summary.md"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now()} {message}"
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def run_cmd(cmd: list[str], *, cwd: Path = ROOT, check: bool = False) -> subprocess.CompletedProcess[str]:
    log("run: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.stdout.strip():
        log("stdout: " + proc.stdout.strip()[-2000:])
    if proc.stderr.strip():
        log("stderr: " + proc.stderr.strip()[-2000:])
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed {proc.returncode}: {' '.join(cmd)}")
    return proc


def active_capture_processes() -> list[str]:
    proc = subprocess.run(
        ["ps", "-axo", "pid,ppid,stat,etime,command"],
        text=True,
        capture_output=True,
        check=False,
    )
    lines = []
    for line in proc.stdout.splitlines():
        if "dali_a0c_finalize_when_done.py" in line:
            continue
        if MAIN_RUN_ID in line and "dali_block_a0_capture.py" in line:
            lines.append(line)
            continue
        if CRYPTO_ROLL_ID in line and (
            "dali_block_a0c_crypto_roll_capture.py" in line
            or "dali_block_a0_capture.py" in line
        ):
            lines.append(line)
    return lines


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_jsonl_tree(root: Path) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    market_counts: dict[str, Counter[str]] = defaultdict(Counter)
    gap_counts: Counter[str] = Counter()
    first_received = None
    last_received = None
    jsonl_files = 0
    data_files = 0
    for path in sorted(root.rglob("*.jsonl")):
        jsonl_files += 1
        if path.name == "capture_gaps.jsonl":
            for rec in iter_jsonl(path):
                gap_counts[str(rec.get("event_type") or "unknown")] += 1
            continue
        if path.name == "roll_supervisor.jsonl":
            for rec in iter_jsonl(path):
                gap_counts[f"roll_{rec.get('event_type') or 'unknown'}"] += 1
            continue
        data_files += 1
        for rec in iter_jsonl(path):
            event_type = str(rec.get("event_type") or "unknown")
            event_counts[event_type] += 1
            received_at = rec.get("received_at")
            if received_at:
                first_received = received_at if first_received is None else min(first_received, received_at)
                last_received = received_at if last_received is None else max(last_received, received_at)
            seen_slugs = set()
            for asset in rec.get("assets") or []:
                slug = str(asset.get("slug") or asset.get("market_id") or "unknown")
                if slug in seen_slugs:
                    continue
                seen_slugs.add(slug)
                market_counts[slug][event_type] += 1
    per_market = []
    for slug, counts in market_counts.items():
        total = sum(counts.values())
        per_market.append({"market": slug, "total": total, **dict(counts)})
    per_market.sort(key=lambda row: int(row.get("last_trade_price") or 0), reverse=True)
    return {
        "root": str(root.relative_to(ROOT)),
        "jsonl_files": jsonl_files,
        "data_files": data_files,
        "first_received_at": first_received,
        "last_received_at": last_received,
        "event_counts": dict(event_counts),
        "gap_counts": dict(gap_counts),
        "per_market": per_market,
    }


def markdown_summary(title: str, summary: dict[str, Any], top_n: int = 30) -> str:
    event_counts = summary["event_counts"]
    gap_counts = summary["gap_counts"]
    lines = [
        f"# {title}",
        "",
        f"Run dir: `{summary['root']}`",
        f"Data JSONL files: `{summary['data_files']}`",
        f"First received: `{summary['first_received_at']}`",
        f"Last received: `{summary['last_received_at']}`",
        "",
        "## Event Counts",
        "",
        "```json",
        json.dumps(event_counts, indent=2, sort_keys=True),
        "```",
        "",
        "## Gap / Control Counts",
        "",
        "```json",
        json.dumps(gap_counts, indent=2, sort_keys=True),
        "```",
        "",
        "## Top Markets By Trades",
        "",
        "market,total,last_trade_price,book,price_change,best_bid_ask",
    ]
    for row in summary["per_market"][:top_n]:
        lines.append(
            ",".join(
                [
                    str(row.get("market", "")),
                    str(row.get("total", 0)),
                    str(row.get("last_trade_price", 0)),
                    str(row.get("book", 0)),
                    str(row.get("price_change", 0)),
                    str(row.get("best_bid_ask", 0)),
                ]
            )
        )
    lines.append("")
    return "\n".join(lines)


def disk_usage(path: Path) -> str:
    proc = subprocess.run(["du", "-sh", str(path)], text=True, capture_output=True, check=False)
    return proc.stdout.strip().split()[0] if proc.stdout.strip() else "unknown"


def write_combined(main_summary: dict[str, Any], roll_summary: dict[str, Any]) -> None:
    lines = [
        "# Block A0c Auto Final Summary",
        "",
        f"Finalized at: `{utc_now()}`",
        "",
        "## Outputs",
        "",
        f"- Main A0c final note: `{MAIN_FINAL_NOTE.relative_to(ROOT)}`",
        f"- Crypto roll final note: `{ROLL_FINAL_NOTE.relative_to(ROOT)}`",
        f"- Finalizer log: `{LOG_PATH.relative_to(ROOT)}`",
        "",
        "## Disk",
        "",
        f"- Main A0c: `{disk_usage(MAIN_RUN_DIR)}`",
        f"- Crypto roll: `{disk_usage(CRYPTO_ROLL_DIR)}`",
        "",
        "## Headline Counts",
        "",
        "| run | book | price_change | best_bid_ask | last_trade_price | first | last |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for name, summary in (("main_a0c", main_summary), ("crypto_roll", roll_summary)):
        c = summary["event_counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(c.get("book", 0)),
                    str(c.get("price_change", 0)),
                    str(c.get("best_bid_ask", 0)),
                    str(c.get("last_trade_price", 0)),
                    str(summary.get("first_received_at")),
                    str(summary.get("last_received_at")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `killall caffeinate` was run after final notes were written.",
            "- WebSocket reconnects are expected; inspect gap/control counts in the final notes.",
            "- Use the per-market trade tables to decide which markets are viable for A1 analysis.",
            "",
        ]
    )
    COMBINED_FINAL_NOTE.write_text("\n".join(lines), encoding="utf-8")


def finalize() -> None:
    run_cmd(
        [
            "uv",
            "run",
            "python",
            "scripts/dali_block_a0_capture_audit.py",
            "--run-dir",
            str(MAIN_RUN_DIR.relative_to(ROOT)),
            "--out",
            str(MAIN_FINAL_NOTE.relative_to(ROOT)),
        ],
        check=False,
    )
    main_summary = summarize_jsonl_tree(MAIN_RUN_DIR)
    roll_summary = summarize_jsonl_tree(CRYPTO_ROLL_DIR)
    ROLL_FINAL_NOTE.write_text(
        markdown_summary("Block A0c Crypto Roll Final Status", roll_summary),
        encoding="utf-8",
    )
    write_combined(main_summary, roll_summary)
    log(f"wrote {MAIN_FINAL_NOTE.relative_to(ROOT)}")
    log(f"wrote {ROLL_FINAL_NOTE.relative_to(ROOT)}")
    log(f"wrote {COMBINED_FINAL_NOTE.relative_to(ROOT)}")
    run_cmd(["killall", "caffeinate"], check=False)
    log("requested killall caffeinate after final summaries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--max-wait-hours", type=float, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deadline = time.monotonic() + args.max_wait_hours * 3600
    log("finalizer started")
    while time.monotonic() < deadline:
        active = active_capture_processes()
        if not active:
            log("no active capture processes found; finalizing")
            finalize()
            return 0
        log(f"waiting; active_capture_processes={len(active)}")
        for line in active[:10]:
            log("active: " + line)
        time.sleep(args.poll_seconds)
    log("deadline reached before captures ended; not killing caffeinate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
