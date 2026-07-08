"""Convert the historical gappy JSONL fixture (R2 ``research-live-clob/``) to typed Parquet.

Pulls the envelope-JSONL fixture from Cloudflare R2, converts each shard to the typed Parquet
layout the engine's Parquet replay adapter reads (book/trades/price_change/bba +
``capture_gaps.parquet``), and optionally pushes the result back to R2 under a clearly-labeled
fixture prefix. Conversion is the EXACT VPS schema (see [[mm_vps_capture_setup]]); the
heavy lifting is :func:`mm_engine.feeds.parquet_convert.jsonl_to_parquet`.

Run from polymarket/research:
    # convert already-local shards (no network):
    PYTHONPATH=. uv run python scripts/mm_jsonl_to_parquet_fixture.py \
        --no-pull --local-dir data/live_clob/fixture --out-dir data/l2_parquet_fixture

    # pull from R2, convert, push back under parquet-fixture/:
    PYTHONPATH=. uv run python scripts/mm_jsonl_to_parquet_fixture.py \
        --r2-prefix research-live-clob --push-prefix parquet-fixture --limit 5

Replay a converted shard:
    from mm_engine.feeds.replay_parquet import replay_parquet
    replay_parquet("data/l2_parquet_fixture/<date>/<universe>")
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from mm_engine.feeds.parquet_convert import jsonl_to_parquet


ROOT = Path(__file__).resolve().parents[1]
_STAMP_RE = re.compile(r"(\d{4})(\d{2})(\d{2})T\d{6}Z")
_GAP_LOG_STEM = "capture_gaps"   # capture_gaps.jsonl or capture_gaps.jsonl.gz


def rclone_copy(src: str, dst: str, *, extra: list[str] | None = None) -> None:
    cmd = ["rclone", "copy", src, dst, "--transfers", "8", "--checkers", "16", "--stats-one-line"]
    cmd += extra or []
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _date_from(path: Path, fallback: str | None) -> str:
    m = _STAMP_RE.search(path.name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return fallback or "unknown-date"


def discover_shards(local_dir: Path) -> list[tuple[Path, Path | None]]:
    """Find JSONL shards (+ each shard's sibling ``capture_gaps.jsonl``), excluding control files."""
    out: list[tuple[Path, Path | None]] = []
    for p in sorted(local_dir.rglob("*.jsonl")) + sorted(local_dir.rglob("*.jsonl.gz")):
        if p.name.startswith(_GAP_LOG_STEM) or p.name.endswith(".manifest.json"):
            continue   # capture_gaps.jsonl[.gz] is a control log, not a data shard
        gaps = next((p.parent / n for n in ("capture_gaps.jsonl", "capture_gaps.jsonl.gz")
                     if (p.parent / n).exists()), None)
        out.append((p, gaps))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="r2")
    ap.add_argument("--bucket", default="epsilon-polymarket-data")
    ap.add_argument("--r2-prefix", default="research-live-clob",
                    help="R2 sub-path of the JSONL fixture to pull (narrow it to limit size)")
    ap.add_argument("--local-dir", type=Path, default=ROOT / "data" / "live_clob" / "fixture")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "l2_parquet_fixture")
    ap.add_argument("--universe", default="fixture", help="universe label for the output layout")
    ap.add_argument("--date", default=None, help="override date partition (else derived from shard name)")
    ap.add_argument("--limit", type=int, default=None, help="max shards to convert")
    ap.add_argument("--no-pull", action="store_true", help="convert already-local shards (skip rclone)")
    ap.add_argument("--push-prefix", default=None,
                    help="if set, rclone copy the Parquet output to r2:<bucket>/<push-prefix>/")
    args = ap.parse_args()

    if not args.no_pull:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        rclone_copy(f"{args.remote}:{args.bucket}/{args.r2_prefix}/", str(args.local_dir))

    shards = discover_shards(args.local_dir)
    if args.limit is not None:
        shards = shards[: args.limit]
    if not shards:
        raise SystemExit(f"no JSONL shards under {args.local_dir}")

    print(f"converting {len(shards)} shard(s) -> {args.out_dir}")
    for shard, gaps in shards:
        date = _date_from(shard, args.date)
        # Qualify the shard label by its path UNDER local_dir, so two shards with the same
        # filename in different run dirs get distinct {table}_{shard}.parquet (no silent
        # overwrite / data loss).
        try:
            rel = str(shard.relative_to(args.local_dir))
        except ValueError:
            rel = shard.name
        for suf in (".jsonl.gz", ".jsonl"):
            if rel.endswith(suf):
                rel = rel[: -len(suf)]
                break
        label = _safe(rel.replace("/", "_"))
        result = jsonl_to_parquet(shard, args.out_dir, date=date, universe=args.universe,
                                  shard=label, gaps_path=gaps)
        print(f"  {shard.name} -> {result['dir']}  {result['counts']}")

    if args.push_prefix:
        rclone_copy(str(args.out_dir), f"{args.remote}:{args.bucket}/{args.push_prefix.rstrip('/')}/")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
