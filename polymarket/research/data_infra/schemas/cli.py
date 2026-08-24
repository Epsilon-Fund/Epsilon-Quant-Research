"""
CLI for the Polymarket data-contract layer. Run with uv from polymarket/research/:

    PYTHONPATH=. uv run python -m data_infra.schemas.cli list
    PYTHONPATH=. uv run python -m data_infra.schemas.cli validate pm_trades
    PYTHONPATH=. uv run python -m data_infra.schemas.cli validate pm_l2_bba --paths /abs/bba.parquet --report
    PYTHONPATH=. uv run python -m data_infra.schemas.cli set-reference pm_trades

Exit code is non-zero when the dataset fails its contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from . import CONTRACTS, MONITOR_DIR, validate_dataset
from .core import write_report


def _parse_as_of(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc) if "T" in s else \
        datetime.fromisoformat(s + "T00:00:00+00:00")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="data-contract (polymarket)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list known PM datasets")
    for cmd in ("validate", "set-reference"):
        p = sub.add_parser(cmd)
        p.add_argument("dataset")
        p.add_argument("--paths", nargs="*", default=None)
        p.add_argument("--as-of", default=None)
        p.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for name, c in CONTRACTS.items():
            print(f"{name:22s} {c.description}")
        return 0

    results = validate_dataset(args.dataset, paths=args.paths, as_of=_parse_as_of(args.as_of),
                               set_reference=(args.cmd == "set-reference"))
    contract = CONTRACTS[args.dataset]
    any_fail = False
    for label, r in results.items():
        tag = "PASS" if r.passed else "FAIL"
        any_fail |= not r.passed
        n_err = len([v for v in r.violations if v.severity == "error"])
        n_warn = len([v for v in r.violations if v.severity == "warn"])
        moved = [d for d in r.drift if d.flag in ("moderate", "large", "ks_significant")]
        print(f"[{tag}] {label}: {r.coverage.files_row_scanned}/{r.coverage.files_total} file(s) scanned, "
              f"{r.coverage.rows_scanned:,} rows, {n_err} blocking, {n_warn} warning(s), "
              f"{len(moved)} drift flag(s), {r.elapsed_s:.3f}s")
        if r.coverage.skipped_note:
            print(f"    coverage: {r.coverage.skipped_note}")
        for v in r.violations[:8]:
            print(f"    • [{v.severity}|{v.check}] {v.column or '-'}: {v.message} (n={v.n_offending})")
        if args.report:
            print(f"    report: {write_report(r, contract, MONITOR_DIR / args.dataset)}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
