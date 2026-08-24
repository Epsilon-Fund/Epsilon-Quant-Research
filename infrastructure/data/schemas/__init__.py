"""
infrastructure.data.schemas — executable data-contract layer for the CRYPTO
live-trading project.

Public API:
    validate_dataset(name, ...) -> dict[unit_label, ValidationResult]
    guard_dataset(name, ...)    -> dict[...]   # fail-closed; aborts on violation
    CONTRACTS                                   # name -> Contract

Usage (research/backtest bootstrap, fail-closed):
    from infrastructure.data.schemas import guard_dataset
    guard_dataset("crypto_ohlcv_daily", symbols=["BTCUSDT", "ETHUSDT"])

Env override `EPSILON_DATA_CONTRACT`: enforce (default) | warn | off.

Never cross-import the Polymarket schemas package — separate venvs, separate
data. The engine (core.py) is a vendored identical copy of the PM one.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import core
from .contracts import CONTRACTS, LIVE_UNIVERSE
from .core import (
    Contract,
    DataContractError,
    ValidationResult,
    enforce,
    render_report,
    run_contract,
    write_report,
)

# repo root: schemas -> data -> infrastructure -> <root>
_ROOT = Path(__file__).resolve().parents[3]
MONITOR_DIR = Path(__file__).resolve().parent / "data_monitoring"

# How each dataset name maps to files on disk (project-specific path layout).
_SPECS: dict[str, dict] = {
    "crypto_ohlcv_daily": {
        "template": "live_trading/cache/daily/{symbol}_daily.parquet",
        "per_symbol": True,
        "default_symbols": LIVE_UNIVERSE,
    },
    "crypto_ohlcv_hourly": {
        "template": "live_trading/cache/hourly/{symbol}_hourly.parquet",
        "per_symbol": True,
        "default_symbols": LIVE_UNIVERSE,
    },
}


def _units(name: str, symbols: list[str] | None, paths: list[str] | None):
    """Return [(unit_label, [Path,...]), ...]. Each unit is one logical series
    validated independently (per-symbol for OHLCV, so monotonicity/cadence are
    not corrupted by concatenating different symbols)."""
    if paths:
        return [("custom", [Path(p) for p in paths])]
    spec = _SPECS[name]
    syms = symbols or spec["default_symbols"]
    if spec.get("per_symbol"):
        return [(s, [_ROOT / spec["template"].format(symbol=s)]) for s in syms]
    return [(name, [_ROOT / spec["template"]])]


def validate_dataset(name: str, *, symbols: list[str] | None = None,
                     paths: list[str] | None = None, as_of: datetime | None = None,
                     set_reference: bool = False) -> dict[str, ValidationResult]:
    if name not in CONTRACTS:
        raise KeyError(f"unknown crypto dataset '{name}'. Known: {sorted(CONTRACTS)}")
    contract = CONTRACTS[name]
    out: dict[str, ValidationResult] = {}
    for label, ps in _units(name, symbols, paths):
        mdir = MONITOR_DIR / name / label  # per-unit references/manifests/drift
        out[label] = run_contract(contract, ps, as_of=as_of, monitor_dir=mdir,
                                  set_reference=set_reference)
    return out


def guard_dataset(name: str, *, symbols: list[str] | None = None,
                  paths: list[str] | None = None, as_of: datetime | None = None,
                  mode: str | None = None) -> dict[str, ValidationResult]:
    """Fail-closed gate for a research/backtest bootstrap. Validates every unit;
    if ANY unit fails, writes a per-unit markdown report and aborts (raises
    DataContractError) unless EPSILON_DATA_CONTRACT is warn/off."""
    import os

    results = validate_dataset(name, symbols=symbols, paths=paths, as_of=as_of)
    contract = CONTRACTS[name]
    failures = {lbl: r for lbl, r in results.items() if not r.passed}
    eff_mode = (mode or os.environ.get("EPSILON_DATA_CONTRACT", "enforce")).lower()

    if not failures:
        scanned = sum(r.coverage.rows_scanned for r in results.values())
        flags = sum(1 for r in results.values() for d in r.drift
                    if d.flag in ("moderate", "large", "ks_significant"))
        warns = sum(1 for r in results.values() for v in r.violations if v.severity == "warn")
        print(f"[data-contract] PASS {name}: {len(results)} unit(s), {scanned:,} rows, "
              f"{warns} warning(s), {flags} drift flag(s).")
        return results

    lines = [f"\n{'='*72}", f" DATA-CONTRACT FAILURE — {name} ({len(failures)} unit(s) failed)",
             "=" * 72]
    for lbl, r in failures.items():
        mdir = MONITOR_DIR / name / lbl
        rep = write_report(r, contract, mdir)
        lines.append(f" unit '{lbl}': {len([v for v in r.violations])} violation group(s) -> {rep}")
        for v in r.violations[:6]:
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
    "CONTRACTS", "LIVE_UNIVERSE", "MONITOR_DIR", "Contract", "ValidationResult",
    "DataContractError", "validate_dataset", "guard_dataset", "run_contract",
    "render_report", "write_report", "enforce", "core",
]
