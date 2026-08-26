"""Salvage a truncated / multi-member capture shard into a clean JSONL.gz.

WHY THIS EXISTS
---------------
An unclean shutdown (hard reboot) truncates whatever gzip shard the capture daemon
had open. On restart the daemon re-opens the SAME hourly file in append mode and
writes a fresh gzip *member* after the truncated one. The result is a single
`*.jsonl.gz` that is:

    [ member 0 : pre-reboot, TRUNCATED mid-stream ][ member 1 : post-reboot, COMPLETE ]

A plain `gzip -t` fails on member 0 and never reaches member 1, and the naive
`zcat | head -n -1` salvage stops at member 0's break — silently discarding the
(usually much larger) post-reboot member. This tool recovers EVERY readable member:
member 0 up to its break (dropping only the final partial line), then each following
complete member in full. It then re-validates the result as line-delimited JSON.

This is the companion to the pipeline's quarantine-and-continue behaviour:
`compression/pipeline.py` sets a bad shard aside in `data/parquet/_quarantine.txt`
(a *to-salvage* list — data pending recovery, NOT written off) and keeps processing
the rest. Run this tool on a quarantined path to recover it, then let the next
compression run pick up the repaired shard.

It NEVER edits in place without a backup: the original is moved to `<path>.corrupt`
and the rebuilt clean stream is written to `<path>`.

USAGE
-----
    # dry-run report on one shard or a whole directory (no writes):
    python compression/salvage_truncated_shard.py data/raw/2026-08-21/esports_21.jsonl.gz
    python compression/salvage_truncated_shard.py data/raw/2026-08-21/

    # actually repair (backs up original to *.corrupt, writes clean *.jsonl.gz):
    python compression/salvage_truncated_shard.py --apply data/raw/2026-08-21/esports_21.jsonl.gz
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import zlib
from pathlib import Path
from typing import Any

MAGIC = b"\x1f\x8b\x08"  # gzip magic + DEFLATE compression method


def _decode_member(data: bytes, offset: int) -> tuple[bytes, bool, int]:
    """Decode one gzip member starting at `offset`.

    Returns (decoded_bytes, clean_eof, next_offset). Small input chunks are fed so
    that if the member is truncated, the raising call discards <=4 KB of compressed
    input near the break — we keep everything decoded before that point.
    """
    d = zlib.decompressobj(16 + 15)  # 16 => gzip framing
    out = bytearray()
    pos = offset
    clean = False
    try:
        while pos < len(data):
            out += d.decompress(data[pos:pos + 4096])
            pos += 4096
            if d.eof:
                clean = True
                break
        if not d.eof:
            out += d.flush()
    except zlib.error:
        clean = False  # truncated member; keep what decoded so far
    if clean and d.eof:
        # unused_data holds every byte after this member's trailer.
        next_off = len(data) - len(d.unused_data)
    else:
        next_off = -1  # unknown; caller falls back to a magic scan
    return bytes(out), clean, next_off


def _next_magic(data: bytes, start: int) -> int:
    i = data.find(MAGIC, start)
    while i != -1:
        # Validate: a real member header decodes at least some bytes.
        try:
            probe = zlib.decompressobj(16 + 15).decompress(data[i:i + 65536])
            if probe:
                return i
        except zlib.error:
            pass
        i = data.find(MAGIC, i + 1)
    return -1


def recover_members(data: bytes) -> list[tuple[bytes, bool]]:
    """Recover every readable gzip member, in file order.

    Returns a list of (decoded_bytes, clean_eof). A member with clean_eof=False was
    truncated and its decoded tail may end mid-line.
    """
    members: list[tuple[bytes, bool]] = []
    offset = 0
    while offset != -1 and offset < len(data):
        out, clean, next_off = _decode_member(data, offset)
        members.append((out, clean))
        if clean and next_off != -1 and next_off < len(data):
            offset = next_off
        elif clean:
            break  # decoded cleanly to EOF
        else:
            offset = _next_magic(data, offset + 2)  # truncated: find the next member
    return members


def rebuild_lines(members: list[tuple[bytes, bool]]) -> list[bytes]:
    """Concatenate member outputs into a line list. The final line of any TRUNCATED
    member is dropped (it may be a partial record cut off by the shutdown)."""
    lines: list[bytes] = []
    for out, clean in members:
        parts = out.split(b"\n")
        if parts and parts[-1] == b"":
            parts = parts[:-1]          # trailing newline -> drop the empty tail
        elif not clean and parts:
            parts = parts[:-1]          # truncated & no final newline -> drop partial
        lines.extend(parts)
    return lines


def _ts(line: bytes) -> int | None:
    try:
        rec = json.loads(line)
        msg = rec.get("message")
        if isinstance(msg, dict) and msg.get("timestamp") not in (None, ""):
            return int(msg["timestamp"])
    except (ValueError, TypeError, KeyError):
        return None
    return None


def analyse(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    members = recover_members(data)
    raw_lines = rebuild_lines(members)

    good: list[bytes] = []
    dropped_bad_json = 0
    last_ts: int | None = None
    ts_regressions = 0
    boundary_ts: list[int | None] = []  # first valid ts of each member (for boundary check)
    seen_member_first = False
    member_line_counts = [len(rebuild_lines([m])) for m in members]

    idx = 0
    member_bounds = []
    run = 0
    for c in member_line_counts:
        member_bounds.append((run, run + c))
        run += c

    for i, ln in enumerate(raw_lines):
        try:
            json.loads(ln)
        except ValueError:
            dropped_bad_json += 1
            continue
        good.append(ln)
        t = _ts(ln)
        if t is not None:
            if last_ts is not None and t < last_ts:
                ts_regressions += 1
            last_ts = t

    # per-member timestamp_ms range (first,last valid) — for the boundary-ordering check
    member_ts_range: list[tuple[int | None, int | None]] = []
    for (lo, hi) in member_bounds:
        first_t = last_t = None
        for ln in raw_lines[lo:hi]:
            t = _ts(ln)
            if t is not None:
                if first_t is None:
                    first_t = t
                last_t = t
        boundary_ts.append(first_t)
        member_ts_range.append((first_t, last_t))

    # boundary monotonicity: does each member start at/after the previous ended?
    boundary_ok = True
    for i in range(1, len(member_ts_range)):
        prev_last = member_ts_range[i - 1][1]
        cur_first = member_ts_range[i][0]
        if prev_last is not None and cur_first is not None and cur_first < prev_last:
            boundary_ok = False

    return {
        "path": str(path),
        "size": len(data),
        "members": [(len(o), clean) for o, clean in members],
        "member_line_counts": member_line_counts,
        "total_lines_raw": len(raw_lines),
        "dropped_bad_json": dropped_bad_json,
        "good_lines": len(good),
        "ts_regressions": ts_regressions,
        "boundary_ts": boundary_ts,
        "member_ts_range": member_ts_range,
        "boundary_monotonic": boundary_ok,
        "good": good,
    }


def apply_repair(path: Path, good_lines: list[bytes]) -> Path:
    corrupt = path.with_suffix(path.suffix + ".corrupt") if not str(path).endswith(".corrupt") \
        else path
    backup = Path(str(path) + ".corrupt")
    if not backup.exists():
        os.replace(path, backup)  # keep the original bytes
    tmp = Path(str(path) + ".salv")
    with gzip.open(tmp, "wb") as fh:
        fh.write(b"\n".join(good_lines) + (b"\n" if good_lines else b""))
    os.replace(tmp, path)
    # sanity: the rebuilt file must now be a single clean gzip stream
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        n = sum(1 for _ in fh)
    if n != len(good_lines):
        raise RuntimeError(f"rebuilt {path}: read {n} lines, expected {len(good_lines)}")
    return backup


def _iter_targets(target: str) -> list[Path]:
    p = Path(target)
    if p.is_dir():
        # any shard that fails gzip -t is a candidate; here just take all shards
        return sorted(Path(x) for x in glob.glob(str(p / "*.jsonl.gz")))
    return [p]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="a *.jsonl.gz shard, or a directory of them")
    ap.add_argument("--apply", action="store_true", help="repair in place (backs up original to *.corrupt)")
    args = ap.parse_args(argv)

    exit_code = 0
    for path in _iter_targets(args.target):
        try:
            info = analyse(path)
        except Exception as exc:  # noqa: BLE001 — report and continue over a batch
            print(f"ERROR analysing {path}: {type(exc).__name__}: {exc}")
            exit_code = 1
            continue
        mem = ", ".join(f"m{i}({'clean' if c else 'TRUNC'},{lc}L)"
                        for i, ((_, c), lc) in enumerate(zip(info["members"], info["member_line_counts"])))
        print(f"{path}")
        print(f"  members: {mem}")
        print(f"  recovered lines: {info['good_lines']}  (dropped bad-json: {info['dropped_bad_json']})")
        print(f"  per-member timestamp_ms range (first,last): {info['member_ts_range']}")
        print(f"  member-boundary monotonic (each starts >= prev end): {info['boundary_monotonic']}")
        print(f"  timestamp_ms regressions across whole file (expected >0; not pre-sorted): {info['ts_regressions']}")
        if len(info["members"]) < 2 and all(c for _, c in info["members"]):
            print("  note: single clean member — nothing to salvage")
        if args.apply:
            backup = apply_repair(path, info["good"])
            print(f"  APPLIED: original -> {backup.name}, rebuilt {path.name} "
                  f"({info['good_lines']} clean lines)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
