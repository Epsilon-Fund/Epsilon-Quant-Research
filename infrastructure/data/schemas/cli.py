"""
CLI for the crypto data-contract layer.

    python -m infrastructure.data.schemas.cli list
    python -m infrastructure.data.schemas.cli validate crypto_ohlcv_daily --symbols BTCUSDT ETHUSDT
    python -m infrastructure.data.schemas.cli validate crypto_ohlcv_daily --paths /abs/file.parquet --report
    python -m infrastructure.data.schemas.cli set-reference crypto_ohlcv_daily

Exit code is non-zero when any unit fails its contract (so CI / shell gates can
rely on it). Run from the repo root.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from . import CONTRACTS, MONITOR_DIR, validate_dataset
from .core import render_report, write_report


def _parse_as_of(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc) if "T" in s else \
        datetime.fromisoformat(s + "T00:00:00+00:00")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="data-contract (crypto)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list known crypto datasets")

    for cmd in ("validate", "set-reference"):
        p = sub.add_parser(cmd)
        p.add_argument("dataset")
        p.add_argument("--symbols", nargs="*", default=None)
        p.add_argument("--paths", nargs="*", default=None)
        p.add_argument("--as-of", default=None, help="ISO date/datetime (point-in-time)")
        p.add_argument("--report", action="store_true", help="also write a markdown report per unit")

    args = ap.parse_args(argv)

    if args.cmd == "list":
        for name, c in CONTRACTS.items():
            print(f"{name:24s} {c.description}")
        return 0

    results = validate_dataset(
        args.dataset, symbols=args.symbols, paths=args.paths,
        as_of=_parse_as_of(args.as_of), set_reference=(args.cmd == "set-reference"),
    )
    contract = CONTRACTS[args.dataset]
    any_fail = False
    for label, r in results.items():
        tag = "PASS" if r.passed else "FAIL"
        any_fail |= not r.passed
        moved = [d for d in r.drift if d.flag in ("moderate", "large", "ks_significant")]
        n_err = len([v for v in r.violations if v.severity == "error"])
        n_warn = len([v for v in r.violations if v.severity == "warn"])
        print(f"[{tag}] {args.dataset}/{label}: {r.n_rows:,} rows, "
              f"{n_err} blocking, {n_warn} warning(s), "
              f"{len(moved)} drift flag(s), {r.elapsed_s:.3f}s")
        for v in r.violations[:8]:
            print(f"    • [{v.severity}|{v.check}] {v.column or '-'}: {v.message} (n={v.n_offending})")
        if args.report:
            rep = write_report(r, contract, MONITOR_DIR / args.dataset / label)
            print(f"    report: {rep}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
