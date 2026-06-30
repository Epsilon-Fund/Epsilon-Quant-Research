"""
data_infra.schemas — executable data-contract layer for the POLYMARKET research
project.

Public API:
    validate_dataset(name, ...) -> dict[unit_label, ValidationResult]
    guard_dataset(name, ...)    -> dict[...]   # fail-closed; aborts on violation
    CONTRACTS                                   # name -> Contract

Usage (research/backtest bootstrap, fail-closed) — run with
`PYTHONPATH=. uv run python ...` from polymarket/research/:
    from data_infra.schemas import guard_dataset
    guard_dataset("pm_trades")

Env override `EPSILON_DATA_CONTRACT`: enforce (default) | warn | off.

Never cross-import the crypto schemas package — separate venvs, separate data.
The engine (core.py) is a vendored identical copy of the crypto one.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from . import core
from .contracts import CONTRACTS
from .core import (
    Contract,
    DataContractError,
    ValidationResult,
    enforce,
    render_report,
    run_contract,
    write_report,
)

# polymarket/research root: schemas -> data_infra -> research
_RESEARCH_ROOT = Path(__file__).resolve().parents[2]
MONITOR_DIR = Path(__file__).resolve().parent / "data_monitoring"

# Path layout per dataset (relative to polymarket/research/).
_SPECS: dict[str, dict] = {
    "pm_trades": {"globs": ["data/trades/trades_seed.parquet",
                            "data/trades/trades_delta_shard*.parquet"]},
    "pm_closed_positions": {"globs": ["data/closed_positions.parquet"]},
    "pm_traders": {"globs": ["data/traders.parquet"]},
    "pm_l2_book": {"globs": ["l2_data/*/*/book_*.parquet"]},
    "pm_l2_trades": {"globs": ["l2_data/*/*/trades_*.parquet"]},
    "pm_l2_price_change": {"globs": ["l2_data/*/*/price_change_*.parquet"]},
    "pm_l2_bba": {"globs": ["l2_data/*/*/bba_*.parquet"]},
}


def _resolve(name: str, paths: list[str] | None) -> list[Path]:
    if paths:
        return [Path(p) for p in paths]
    spec = _SPECS.get(name, {})
    out: list[Path] = []
    for g in spec.get("globs", []):
        out.extend(sorted(_RESEARCH_ROOT.glob(g)))
    return out


def validate_dataset(name: str, *, paths: list[str] | None = None,
                     as_of: datetime | None = None,
                     set_reference: bool = False) -> dict[str, ValidationResult]:
    if name not in CONTRACTS:
        raise KeyError(f"unknown PM dataset '{name}'. Known: {sorted(CONTRACTS)}")
    contract = CONTRACTS[name]
    files = _resolve(name, paths)
    mdir = MONITOR_DIR / name
    # one logical unit per dataset (the engine handles shard sets + coverage budget).
    res = run_contract(contract, files, as_of=as_of, monitor_dir=mdir,
                       set_reference=set_reference)
    return {name: res}


def guard_dataset(name: str, *, paths: list[str] | None = None,
                  as_of: datetime | None = None,
                  mode: str | None = None) -> dict[str, ValidationResult]:
    """Fail-closed gate for a research/backtest bootstrap. Validates the dataset;
    on any contract violation, writes a markdown report and aborts (raises
    DataContractError) unless EPSILON_DATA_CONTRACT is warn/off."""
    results = validate_dataset(name, paths=paths, as_of=as_of)
    contract = CONTRACTS[name]
    eff_mode = (mode or os.environ.get("EPSILON_DATA_CONTRACT", "enforce")).lower()
    failures = {lbl: r for lbl, r in results.items() if not r.passed}

    if not failures:
        for lbl, r in results.items():
            warns = len([v for v in r.violations if v.severity == "warn"])
            flags = len([d for d in r.drift if d.flag in ("moderate", "large", "ks_significant")])
            print(f"[data-contract] PASS {name}: {r.coverage.files_row_scanned}/{r.coverage.files_total} "
                  f"file(s) scanned, {r.coverage.rows_scanned:,} rows, {warns} warning(s), {flags} drift flag(s).")
        return results

    lines = [f"\n{'='*72}", f" DATA-CONTRACT FAILURE — {name}", "=" * 72]
    for lbl, r in failures.items():
        rep = write_report(r, contract, MONITOR_DIR / name)
        lines.append(f" {len([v for v in r.violations if v.severity=='error'])} blocking "
                     f"violation group(s) -> {rep}")
        for v in r.violations[:8]:
            if v.severity == "error":
                lines.append(f"   • [{v.check}] {v.column or '-'}: {v.message} (n={v.n_offending})")
    lines.append("=" * 72)
    banner = "\n".join(lines)

    if eff_mode == "off":
        return results
    if eff_mode == "warn":
        print(banner + "\n  EPSILON_DATA_CONTRACT=warn -> proceeding despite failure (UNSAFE).")
        return results
    raise DataContractError(banner)


__all__ = [
    "CONTRACTS", "MONITOR_DIR", "Contract", "ValidationResult", "DataContractError",
    "validate_dataset", "guard_dataset", "run_contract", "render_report",
    "write_report", "enforce", "core",
]
