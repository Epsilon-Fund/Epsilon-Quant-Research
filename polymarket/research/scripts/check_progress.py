"""Compute real per-shard rate, aggregate rate, and ETA from the parallel sync log."""
import re
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "sync_parallel.log"

SHARDS_REMAINING_TARGETS = {
    "1": 25_000_000, "2": 44_000_000, "3": 51_000_000,
    "4b": 31_000_000, "4c": 31_000_000, "4d": 34_000_000,
    "5b": 45_000_000, "5c": 45_000_000, "5d": 51_000_000,
    "6b": 34_000_000, "6c": 34_000_000, "6d": 34_000_000,
    "6e": 34_000_000, "6f": 34_000_000, "6g": 34_000_000, "6h": 38_000_000,
    "7b": 45_000_000, "7c": 45_000_000, "7d": 45_000_000,
}

LINE = re.compile(r"shard(\S+):\s+(\S+)\s+rows=\s*([\d,]+)\s+latest=(\S+ \S+)")
HEADER = re.compile(r"^--- (\S+ \S+) ---")


def parse_blocks(text: str) -> list[tuple[str, dict]]:
    blocks: list[tuple[str, dict]] = []
    cur_time = None
    cur: dict[str, dict] = {}
    for raw in text.splitlines():
        h = HEADER.match(raw)
        if h:
            if cur:
                blocks.append((cur_time, cur))
            cur_time = h.group(1)
            cur = {}
            continue
        m = LINE.search(raw)
        if m and cur_time:
            idx, status, rows_s, latest = m.groups()
            cur[idx] = {
                "status": status,
                "rows": int(rows_s.replace(",", "")),
                "latest": latest,
            }
    if cur:
        blocks.append((cur_time, cur))
    return blocks


def main() -> None:
    if not LOG.exists():
        sys.exit(f"log not found: {LOG}")
    blocks = parse_blocks(LOG.read_text())
    if len(blocks) < 2:
        sys.exit(f"need at least 2 progress blocks; have {len(blocks)}")
    t1, b1 = blocks[-2]
    t2, b2 = blocks[-1]

    from datetime import datetime
    fmt = "%H:%M:%S"
    try:
        d1 = datetime.strptime(t1.split()[0], fmt)
        d2 = datetime.strptime(t2.split()[0], fmt)
        gap_s = max(int((d2 - d1).total_seconds()), 1)
        if gap_s < 0:
            gap_s += 86400
    except Exception:
        gap_s = 60

    print(f"--- between {t1} and {t2} ({gap_s}s window) ---")
    if gap_s > 120:
        print(f"!! WARNING: gap is {gap_s}s, suggests system sleep or stalled reporter !!")
    print(f"{'shard':<8}{'status':<8}{'rows':>14}{'Δ rows':>10}{'r/s':>8}{'latest':>22}")
    total_now = 0
    total_delta = 0
    fetching = 0
    for idx in sorted(b2.keys()):
        cur = b2[idx]
        prev = b1.get(idx, {"rows": cur["rows"]})
        delta = cur["rows"] - prev["rows"]
        rate = delta // gap_s
        total_now += cur["rows"]
        total_delta += delta
        if delta > 0:
            fetching += 1
        print(
            f"shard{idx:<7}{cur['status']:<8}"
            f"{cur['rows']:>14,}{delta:>10,}{rate:>8}{cur['latest']:>22}"
        )

    aggregate_rate = total_delta // gap_s
    print()
    print(f"workers fetching        : {fetching}")
    print(f"aggregate rate (last 60s): {aggregate_rate:,} r/s")

    expected_total = sum(SHARDS_REMAINING_TARGETS.values())
    pct = 100 * total_now / expected_total if expected_total else 0
    rows_remaining = max(expected_total - total_now, 0)
    eta_s = rows_remaining / max(aggregate_rate, 1) if aggregate_rate > 0 else 0
    print()
    print(f"rows fetched (active)   : {total_now:,}")
    print(f"rows expected (~)       : {expected_total:,}")
    print(f"approx % complete       : {pct:.1f}%")
    print(f"approx rows remaining   : {rows_remaining:,}")
    print(f"approx ETA              : {eta_s/3600:.1f} h")


if __name__ == "__main__":
    main()
