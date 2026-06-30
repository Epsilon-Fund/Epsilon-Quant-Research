"""
core.py — Executable data-contract & drift validation engine.

============================================================================
VENDORED PER-PROJECT COPY — keep byte-identical between:
  - infrastructure/data/schemas/core.py                (crypto live-trading)
  - polymarket/research/data_infra/schemas/core.py     (Polymarket research)
The two projects run in SEPARATE venvs and must NEVER cross-import
(brain/CODEX.md invariant). This engine is project-agnostic; the actual
dataset contracts live in each project's own contracts.py. If you edit this
file, edit BOTH copies.
============================================================================

What it does (each is enforced as code, not assumed):
  * schema/contract  — column presence, dtype, nullability, per-column ranges
                       (pandera.polars), plus cross-column row rules.
  * lookahead-free   — no row time-stamped after `as_of` (default now), and
                       (for captured streams) no server-ts after receive-ts.
  * append-only      — sharded parquet families never mutate in place; a
                       single growing file is a row-superset of its prior self.
  * monotone ts      — strictly increasing (bars) or non-decreasing (streams).
  * cadence          — no duplicate / missing bars vs an expected step.
  * finite values    — no NaN / Inf in declared numeric columns.
  * lowercase 0x     — address / hash columns match ^0x[0-9a-f]+$.
  * drift            — PSI + two-sample KS of each monitored column vs a stored
                       reference window; appended to data_monitoring/*.parquet.

Deps: polars, numpy, pandera (pandera.polars). No external service.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

try:  # pandera split its namespaces; we use the polars backend.
    import pandera.polars as pa
    from pandera.errors import SchemaError, SchemaErrors

    _PANDERA_OK = True
except Exception as _e:  # pragma: no cover - exercised only when pandera missing
    pa = None  # type: ignore
    SchemaError = SchemaErrors = Exception  # type: ignore
    _PANDERA_OK = False
    _PANDERA_IMPORT_ERR = _e


# ── PSI thresholds (population stability index) ─────────────────────────────
# Industry convention: <0.10 stable · 0.10–0.25 moderate shift · >0.25 large.
PSI_MODERATE = 0.10
PSI_LARGE = 0.25
# KS p-value below this flags a distribution change. KS is hyper-sensitive on
# huge n, so we also report the KS statistic (effect size) and never abort on
# drift — drift is a WATCH signal, only contract/invariant breaks are gates.
KS_ALPHA = 0.05


# ════════════════════════════════════════════════════════════════════════════
# Contract specification (declared per dataset in each project's contracts.py)
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TimestampRule:
    """Ordering + cadence guard for the time column.

    column     : the timestamp column name.
    order      : "strict_increasing" (bars — duplicates illegal) or
                 "non_decreasing" (event streams — ties allowed).
    epoch_unit : None if the column is a real datetime; else the int-epoch unit
                 ("s" | "ms" | "us" | "ns") so we interpret it correctly.
    cadence    : expected fixed step for bar series ("1d" | "1h" | "1m" | None).
                 None disables the no-missing/no-duplicate-bar check (correct
                 for irregular event streams).
    per_file   : check ordering within each file rather than across the whole
                 concatenation (right for date-sharded families).
    """

    column: str
    order: str = "strict_increasing"
    epoch_unit: str | None = None
    cadence: str | None = None
    per_file: bool = True
    # Missing bars (clean multiples of the step) are a completeness WARNING unless
    # they exceed this fraction of expected bars — real exchanges (e.g. Binance) have
    # rare maintenance gaps, and a hard abort over them would be theatrically strict.
    # Duplicate / sub-step / off-grid intervals are always hard errors (corruption).
    cadence_gap_tolerance: float = 0.01


@dataclass(frozen=True)
class LookaheadRule:
    """Point-in-time integrity. Scope: the DATA layer only — it catches
    future-stamped / corrupt rows that would leak the future into any
    as-of replay. It does NOT verify feature-vs-label leakage inside strategy
    code; that stays the strategy's responsibility."""

    column: str
    # rows with `column` > as_of + grace are future leakage (as_of defaults to now).
    grace_seconds: float = 90.0
    # optional: rows where `column` > this receive-clock column are server-ahead-of-receive
    not_after_column: str | None = None
    # epoch unit of `column` if it is an int-epoch (e.g. "ms") rather than a datetime.
    epoch_unit: str | None = None


@dataclass(frozen=True)
class AppendOnlyRule:
    """append-only parquet invariant.

    mode = "shard"        : a date/hour-sharded family — no existing shard may
                            change bytes (in-place mutation), lose rows, or
                            vanish; brand-new shards are allowed.
    mode = "row_superset" : a single growing file — current rows must be a
                            superset (by row-hash over `key_columns`) of the
                            previously-seen rows, and row-count non-decreasing.
    """

    mode: str = "shard"
    key_columns: tuple[str, ...] = ()
    # deep content hash is exact but O(rows); skipped above this (fingerprint only).
    deep_hash_max_rows: int = 5_000_000


@dataclass(frozen=True)
class RowRule:
    """A cross-column boolean invariant that must hold for every row.
    `expr` returns a polars expression that is True on GOOD rows."""

    name: str
    expr: Callable[[], "pl.Expr"]
    description: str = ""


@dataclass
class Contract:
    name: str
    description: str
    # pandera.polars DataFrameSchema (presence / dtype / nullable / per-col range)
    schema: object | None = None
    timestamp: TimestampRule | None = None
    lookahead: LookaheadRule | None = None
    append_only: AppendOnlyRule | None = None
    address_columns: tuple[str, ...] = ()
    # numeric columns that must be finite (no NaN/Inf). Empty -> all Float cols.
    finite_columns: tuple[str, ...] = ()
    row_rules: tuple[RowRule, ...] = ()
    drift_columns: tuple[str, ...] = ()
    # coverage control for huge sharded families
    scan_strategy: str = "all"  # "all" | "recent_shards" | "sample"
    recent_shards: int = 3
    max_scan_rows: int = 5_000_000
    # plain-English notes that land in the report's design section
    notes: str = ""


# ════════════════════════════════════════════════════════════════════════════
# Result types
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Violation:
    check: str  # schema | monotonic_ts | cadence | finite | address | append_only | lookahead | row_rule | engine
    column: str | None
    message: str
    n_offending: int
    examples: list = field(default_factory=list)
    severity: str = "error"  # "error" (fails the gate) | "warn" (reported, non-blocking)


@dataclass
class DriftRow:
    column: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    n_ref: int
    n_cur: int
    flag: str  # stable | moderate | large | ks_significant | baseline | insufficient


@dataclass
class Coverage:
    files_total: int = 0
    files_metadata_only: int = 0
    files_row_scanned: int = 0
    rows_scanned: int = 0
    skipped_note: str = ""


@dataclass
class ValidationResult:
    dataset: str
    paths: list[str]
    n_rows: int
    passed: bool
    violations: list[Violation]
    drift: list[DriftRow]
    coverage: Coverage
    elapsed_s: float
    as_of: str
    ran_at: str


# ════════════════════════════════════════════════════════════════════════════
# Small helpers
# ════════════════════════════════════════════════════════════════════════════
_CADENCE_NS = {
    "1m": 60 * 1_000_000_000,
    "5m": 5 * 60 * 1_000_000_000,
    "1h": 3600 * 1_000_000_000,
    "1d": 86400 * 1_000_000_000,
}
_EPOCH_TO_NS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_as_ns(df: pl.DataFrame, rule: TimestampRule) -> pl.Series:
    """Return the timestamp column as int64 nanoseconds, regardless of storage."""
    col = df.get_column(rule.column)
    if rule.epoch_unit is not None:
        return (col.cast(pl.Int64) * _EPOCH_TO_NS[rule.epoch_unit]).cast(pl.Int64)
    # real datetime -> ns
    return col.cast(pl.Datetime("ns")).cast(pl.Int64)


def _examples(df: pl.DataFrame, mask: pl.Series, cols: list[str], k: int = 5) -> list:
    bad = df.filter(mask)
    if bad.height == 0:
        return []
    keep = [c for c in cols if c in bad.columns][:6] or bad.columns[:6]
    return bad.select(keep).head(k).to_dicts()


# ════════════════════════════════════════════════════════════════════════════
# Invariant checks (each returns a Violation or None). Operate on an in-memory
# polars DataFrame (the "scan set"). Caller controls how much is loaded.
# ════════════════════════════════════════════════════════════════════════════
def check_schema(contract: Contract, df: pl.DataFrame) -> list[Violation]:
    if contract.schema is None:
        return []
    if not _PANDERA_OK:
        return [Violation("engine", None,
                          f"pandera unavailable ({_PANDERA_IMPORT_ERR}); cannot run schema contract. "
                          f"Install with `uv add pandera` (PM) or `uv pip install pandera` (crypto).",
                          1)]
    try:
        contract.schema.validate(df, lazy=True)
        return []
    except SchemaErrors as e:
        fc = e.failure_cases  # polars DataFrame
        out: list[Violation] = []
        try:
            rows = fc.select([c for c in ("check", "column", "failure_case") if c in fc.columns]).to_dicts()
        except Exception:
            rows = []
        # group by (column, check)
        seen: dict[tuple, dict] = {}
        for r in rows:
            key = (r.get("column"), r.get("check"))
            d = seen.setdefault(key, {"n": 0, "ex": []})
            d["n"] += 1
            if len(d["ex"]) < 5:
                d["ex"].append(r.get("failure_case"))
        for (col, chk), d in seen.items():
            out.append(Violation("schema", col, f"failed check `{chk}`", d["n"], d["ex"]))
        if not out:
            out.append(Violation("schema", None, str(e)[:300], 1))
        return out
    except SchemaError as e:  # single error (e.g. missing column)
        return [Violation("schema", getattr(e, "column_name", None), str(e)[:300], 1)]


def check_monotonic(contract: Contract, df: pl.DataFrame) -> list[Violation]:
    rule = contract.timestamp
    if rule is None or df.height < 2:
        return []
    if rule.column not in df.columns:
        return [Violation("monotonic_ts", rule.column, "timestamp column missing", 1)]
    ns = _ts_as_ns(df, rule)
    diffs = ns.diff().drop_nulls()
    if rule.order == "strict_increasing":
        bad = (diffs <= 0)
        kind = "non-increasing (duplicate or out-of-order) timestamps"
    else:  # non_decreasing
        bad = (diffs < 0)
        kind = "out-of-order (decreasing) timestamps"
    n = int(bad.sum())
    if n == 0:
        return []
    return [Violation("monotonic_ts", rule.column,
                      f"{n} {kind} (order={rule.order})", n,
                      _examples(df, _bad_diff_mask(ns, rule), [rule.column]))]


def _bad_diff_mask(ns: pl.Series, rule: TimestampRule) -> pl.Series:
    d = ns.diff()
    if rule.order == "strict_increasing":
        return (d <= 0).fill_null(False)
    return (d < 0).fill_null(False)


def check_cadence(contract: Contract, df: pl.DataFrame) -> list[Violation]:
    rule = contract.timestamp
    if rule is None or rule.cadence is None or df.height < 2:
        return []
    if rule.column not in df.columns:
        return []  # missing-column already reported by check_monotonic / schema
    step = _CADENCE_NS.get(rule.cadence)
    if step is None:
        return [Violation("cadence", rule.column, f"unknown cadence '{rule.cadence}'", 1)]
    ns = _ts_as_ns(df, rule).sort()
    diffs = ns.diff().drop_nulls().to_numpy().astype("int64")
    dups = int((diffs == 0).sum())
    rem = np.where(diffs > 0, diffs % step, 0)
    off_grid = int(((diffs > 0) & (rem != 0)).sum())          # sub-step / misaligned -> corruption
    gap_mask = (diffs > step) & (rem == 0)                      # clean multiple of step -> missing bars
    missing_bars = int(np.sum(diffs[gap_mask] // step - 1))     # number of absent bars
    n_gaps = int(gap_mask.sum())
    expected = df.height + missing_bars
    out: list[Violation] = []
    # hard errors: duplicates and off-grid intervals (true corruption)
    if dups or off_grid:
        out.append(Violation("cadence", rule.column,
                             f"cadence {rule.cadence}: {dups} duplicate-step interval(s), "
                             f"{off_grid} off-grid (sub-step/misaligned) interval(s)",
                             dups + off_grid, severity="error"))
    # completeness: missing bars -> warn unless beyond tolerance
    if missing_bars:
        frac = missing_bars / max(expected, 1)
        sev = "error" if frac > rule.cadence_gap_tolerance else "warn"
        out.append(Violation("cadence", rule.column,
                             f"cadence {rule.cadence}: {missing_bars} missing bar(s) across {n_gaps} gap(s) "
                             f"({frac:.3%} of expected; tolerance {rule.cadence_gap_tolerance:.2%})",
                             missing_bars, severity=sev))
    return out


def check_finite(contract: Contract, df: pl.DataFrame) -> list[Violation]:
    cols = list(contract.finite_columns)
    if not cols:
        cols = [c for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)]
    out: list[Violation] = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df.get_column(c)
        if s.dtype not in (pl.Float32, pl.Float64):
            s = s.cast(pl.Float64, strict=False)
        bad = s.is_nan() | s.is_infinite()
        n = int(bad.fill_null(False).sum())
        if n:
            out.append(Violation("finite", c, f"{n} NaN/Inf values", n,
                                  _examples(df, bad.fill_null(False), [c])))
    return out


def check_addresses(contract: Contract, df: pl.DataFrame) -> list[Violation]:
    out: list[Violation] = []
    for c in contract.address_columns:
        if c not in df.columns:
            out.append(Violation("address", c, "declared address column missing", 1))
            continue
        s = df.get_column(c).cast(pl.Utf8, strict=False)
        # GOOD: matches ^0x[0-9a-f]+$ . nulls ignored.
        good = s.str.contains(r"^0x[0-9a-f]+$")
        bad = (~good.fill_null(True)) & s.is_not_null()
        n = int(bad.sum())
        if n:
            out.append(Violation("address", c,
                                  f"{n} values not lowercase 0x-hex (^0x[0-9a-f]+$)", n,
                                  _examples(df, bad, [c])))
    return out


def check_row_rules(contract: Contract, df: pl.DataFrame) -> list[Violation]:
    out: list[Violation] = []
    for rule in contract.row_rules:
        try:
            good = df.select(rule.expr().alias("__ok")).get_column("__ok")
        except Exception as e:
            out.append(Violation("row_rule", None, f"rule '{rule.name}' could not evaluate: {e}", 1))
            continue
        bad = (~good.fill_null(False))
        n = int(bad.sum())
        if n:
            out.append(Violation("row_rule", None,
                                  f"row rule '{rule.name}' violated ({rule.description})".strip(), n,
                                  _examples(df, bad, df.columns)))
    return out


def check_lookahead(contract: Contract, df: pl.DataFrame, as_of_ns: int) -> list[Violation]:
    rule = contract.lookahead
    if rule is None or df.height == 0:
        return []
    out: list[Violation] = []
    if rule.column not in df.columns:
        return [Violation("lookahead", rule.column, "declared lookahead column missing", 1)]
    # interpret the lookahead column with its own epoch unit (falls back to the
    # timestamp rule's unit only when they are the same column).
    unit = rule.epoch_unit
    if unit is None and contract.timestamp and contract.timestamp.column == rule.column:
        unit = contract.timestamp.epoch_unit
    eff = TimestampRule(column=rule.column, epoch_unit=unit)
    ns = _ts_as_ns(df, eff)
    grace = int(rule.grace_seconds * 1_000_000_000)
    future = (ns > (as_of_ns + grace)).fill_null(False)
    nf = int(future.sum())
    if nf:
        out.append(Violation("lookahead", rule.column,
                             f"{nf} rows time-stamped after as_of (+{rule.grace_seconds:.0f}s grace) "
                             f"— future leakage", nf, _examples(df, future, [rule.column])))
    if rule.not_after_column and rule.not_after_column in df.columns:
        recv = df.get_column(rule.not_after_column)
        try:
            recv_ns = recv.str.to_datetime(strict=False, time_zone="UTC").cast(pl.Datetime("ns")).cast(pl.Int64)
        except Exception:
            recv_ns = None
        if recv_ns is not None:
            ahead = ((ns - recv_ns) > grace).fill_null(False)
            na = int(ahead.sum())
            if na:
                out.append(Violation("lookahead", rule.column,
                                     f"{na} rows where {rule.column} is ahead of {rule.not_after_column} "
                                     f"by >{rule.grace_seconds:.0f}s (server-ahead-of-receive)", na))
    return out


# ════════════════════════════════════════════════════════════════════════════
# Append-only manifest (file-set integrity)
# ════════════════════════════════════════════════════════════════════════════
def _file_fingerprint(path: Path, deep: bool) -> dict:
    st = path.stat()
    rec = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "dir": str(path.parent)}
    try:
        lf = pl.scan_parquet(path)
        rec["rows"] = int(lf.select(pl.len()).collect().item())
    except Exception:
        rec["rows"] = -1
    if deep and rec["rows"] >= 0:
        try:
            df = pl.read_parquet(path)
            h = df.hash_rows().sort().to_numpy().tobytes()
            rec["hash"] = hashlib.sha256(h).hexdigest()
        except Exception:
            rec["hash"] = None
    else:
        rec["hash"] = None
    return rec


def check_append_only(contract: Contract, paths: list[Path], monitor_dir: Path) -> list[Violation]:
    rule = contract.append_only
    if rule is None or rule.mode == "none":
        return []
    out: list[Violation] = []
    manifest_path = monitor_dir / f"append_only_{contract.name}.json"
    prior = {}
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text())
        except Exception:
            prior = {}

    if rule.mode == "shard":
        cur: dict[str, dict] = {}
        for p in paths:
            deep = True
            try:
                rows_quick = int(pl.scan_parquet(p).select(pl.len()).collect().item())
                deep = rows_quick <= rule.deep_hash_max_rows
            except Exception:
                pass
            cur[p.name] = _file_fingerprint(p, deep=deep)
        cur_dirs = {str(p.parent) for p in paths}
        for name, prec in prior.items():
            crec = cur.get(name)
            if crec is None:
                # Only flag disappearance for shards that lived in a directory we
                # are actually validating now. This prevents partial validation
                # (e.g. a single --paths file in /tmp) from false-flagging the
                # whole production family as "missing".
                if prec.get("dir") in cur_dirs:
                    out.append(Violation("append_only", None,
                                         f"previously-seen shard '{name}' is missing (deleted/renamed)", 1))
                continue
            if prec.get("rows", -1) >= 0 and crec.get("rows", -1) >= 0 and crec["rows"] < prec["rows"]:
                out.append(Violation("append_only", None,
                                     f"shard '{name}' lost rows ({prec['rows']} -> {crec['rows']}) — "
                                     f"in-place truncation", 1))
            ph, ch = prec.get("hash"), crec.get("hash")
            if ph and ch and ph != ch:
                out.append(Violation("append_only", None,
                                     f"shard '{name}' content changed (hash mismatch) — in-place mutation", 1))
            elif (ph is None or ch is None) and (
                prec.get("size") != crec.get("size") and prec.get("rows") == crec.get("rows")):
                out.append(Violation("append_only", None,
                                     f"shard '{name}' bytes changed at equal row-count "
                                     f"(size {prec.get('size')} -> {crec.get('size')}) — likely in-place edit", 1))
        # persist the union (new shards recorded, old kept)
        merged = {**prior, **cur}
        _atomic_write(manifest_path, json.dumps(merged, indent=0))

    elif rule.mode == "row_superset":
        # exact superset over key columns (suited to small/medium growing files).
        df = pl.read_parquet(paths)
        keys = [c for c in rule.key_columns if c in df.columns] or df.columns
        hashes = set(df.select(keys).hash_rows().to_list())
        prior_hashes = set(prior.get("hashes", []))
        prior_count = int(prior.get("count", 0))
        if prior_hashes:
            missing = prior_hashes - hashes
            if missing:
                out.append(Violation("append_only", None,
                                     f"{len(missing)} previously-present rows are gone — file is not "
                                     f"append-only (row_superset over {keys})", len(missing)))
            if df.height < prior_count:
                out.append(Violation("append_only", None,
                                     f"row-count dropped ({prior_count} -> {df.height})", 1))
        # cap stored hashes to keep manifest bounded
        store = list(hashes)
        if len(store) <= 2_000_000:
            _atomic_write(manifest_path, json.dumps({"count": df.height, "hashes": store}))
        else:
            _atomic_write(manifest_path, json.dumps({"count": df.height, "hashes": []}))
    return out


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════════════
# Drift — PSI + two-sample KS (pure numpy; no scipy dependency)
# ════════════════════════════════════════════════════════════════════════════
def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < 2 or cur.size < 2:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:  # ~constant reference
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0].astype(float)
    c = np.histogram(cur, edges)[0].astype(float)
    r = np.clip(r / r.sum(), eps, None)
    c = np.clip(c / c.sum(), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _probks(lam: float) -> float:
    """Kolmogorov survival function Q(lam) = asymptotic two-sided p-value.
    Numerical-Recipes `probks`: the early-convergence test makes Q(0)=1
    (identical samples -> p=1, NOT the naive-series degenerate 0)."""
    if lam <= 0.0:
        return 1.0
    a2 = -2.0 * lam * lam
    fac, total, termbf = 2.0, 0.0, 0.0
    for j in range(1, 101):
        term = fac * np.exp(a2 * j * j)
        total += term
        if abs(term) <= 1e-3 * termbf or abs(term) <= 1e-8 * total:
            return float(min(1.0, max(0.0, total)))
        fac = -fac
        termbf = abs(term)
    return 1.0  # non-convergence (tiny lam) -> no evidence of difference


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / n
    cdf_b = np.searchsorted(b, allv, side="right") / m
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = np.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    return d, _probks(lam)


def compute_drift(df: pl.DataFrame, reference: pl.DataFrame | None,
                  columns: list[str]) -> list[DriftRow]:
    out: list[DriftRow] = []
    for c in columns:
        if c not in df.columns:
            continue
        cur = df.get_column(c).cast(pl.Float64, strict=False).to_numpy()
        if reference is None or c not in reference.columns:
            out.append(DriftRow(c, 0.0, 0.0, 1.0, 0, cur.size, "baseline"))
            continue
        ref = reference.get_column(c).cast(pl.Float64, strict=False).to_numpy()
        p = psi(ref, cur)
        d, pv = ks_2samp(ref, cur)
        if not np.isfinite(p):
            flag = "insufficient"
        elif p >= PSI_LARGE:
            flag = "large"
        elif p >= PSI_MODERATE:
            flag = "moderate"
        elif np.isfinite(pv) and pv < KS_ALPHA:
            flag = "ks_significant"
        else:
            flag = "stable"
        out.append(DriftRow(c, round(p, 5), round(d, 5), round(pv, 5),
                            ref.size, cur.size, flag))
    return out


# ════════════════════════════════════════════════════════════════════════════
# Orchestration: load files, run all checks under a coverage budget
# ════════════════════════════════════════════════════════════════════════════
def _select_scan_files(contract: Contract, paths: list[Path]) -> tuple[list[Path], list[Path], str]:
    """Return (row_scan_files, metadata_only_files, note)."""
    if not paths:
        return [], [], ""
    if contract.scan_strategy == "all":
        return paths, [], ""
    if contract.scan_strategy == "recent_shards":
        ordered = sorted(paths, key=lambda p: p.name)
        scan = ordered[-contract.recent_shards:]
        meta = ordered[:-contract.recent_shards]
        note = (f"row-level invariants scanned the {len(scan)} most-recent shard(s); "
                f"{len(meta)} older shard(s) got schema + append-only checks only "
                f"(coverage capped for speed).")
        return scan, meta, note
    return paths[:1], paths[1:], "sample strategy: scanned first file only."


def run_contract(contract: Contract, paths: list[str | Path], *,
                 as_of: datetime | None = None,
                 monitor_dir: str | Path,
                 set_reference: bool = False) -> ValidationResult:
    """Full pipeline for one dataset: schema + invariants + append-only + drift.
    Loads parquet via polars. `paths` may be a list of shard files."""
    import time as _time
    t0 = _time.perf_counter()
    paths = [Path(p) for p in paths]
    monitor_dir = Path(monitor_dir)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    as_of = as_of or datetime.now(timezone.utc)
    as_of_ns = int(as_of.timestamp() * 1_000_000_000)

    violations: list[Violation] = []
    missing = [p for p in paths if not p.exists()]
    paths = [p for p in paths if p.exists()]
    for p in missing:
        violations.append(Violation("engine", None, f"declared input not found: {p}", 1))
    if not paths:
        return ValidationResult(contract.name, [str(p) for p in missing], 0, False,
                                violations or [Violation("engine", None, "no input files", 1)],
                                [], Coverage(), round(_time.perf_counter() - t0, 4),
                                as_of.isoformat(), _now_iso())

    # 1) schema-from-metadata across ALL files (presence+dtype, O(1) per file)
    violations += _schema_metadata_sweep(contract, paths)

    # 2) append-only across the full file set
    violations += check_append_only(contract, paths, monitor_dir)

    # 3) choose the row-scan set under the coverage budget. Use a lazy scan +
    # tail so huge single files (e.g. 270M-row closed_positions) never get fully
    # materialised — memory stays bounded by max_scan_rows.
    scan_files, meta_only, note = _select_scan_files(contract, paths)
    if scan_files:
        srcs = [str(p) for p in scan_files]
        lf = pl.scan_parquet(srcs)
        try:
            total_rows = int(pl.scan_parquet(srcs).select(pl.len()).collect().item())
        except Exception:
            total_rows = None
        if total_rows is not None and total_rows > contract.max_scan_rows:
            lf = lf.tail(contract.max_scan_rows)  # most-recent stored rows
            note = (note + " " if note else "") + (
                f"row scan capped at {contract.max_scan_rows:,}/{total_rows:,} rows (most recent).")
        df = lf.collect()
    else:
        df = pl.DataFrame()

    cov = Coverage(files_total=len(paths), files_metadata_only=len(meta_only),
                   files_row_scanned=len(scan_files), rows_scanned=df.height, skipped_note=note)

    # 4) row-level invariants on the scan set
    if df.height:
        violations += check_schema(contract, df)
        violations += check_monotonic(contract, df)
        violations += check_cadence(contract, df)
        violations += check_finite(contract, df)
        violations += check_addresses(contract, df)
        violations += check_row_rules(contract, df)
        violations += check_lookahead(contract, df, as_of_ns)

    # 5) drift vs reference
    drift: list[DriftRow] = []
    ref_path = monitor_dir / f"reference_{contract.name}.parquet"
    if contract.drift_columns and df.height:
        dcols = [c for c in contract.drift_columns if c in df.columns]
        if set_reference or not ref_path.exists():
            df.select(dcols).write_parquet(ref_path)
            drift = compute_drift(df, None, dcols)
        else:
            reference = pl.read_parquet(ref_path)
            drift = compute_drift(df, reference, dcols)
            _append_drift_log(monitor_dir, contract.name, drift, as_of)

    passed = not any(v.severity == "error" for v in violations)
    return ValidationResult(contract.name, [str(p) for p in paths], df.height, passed,
                            violations, drift, cov, round(_time.perf_counter() - t0, 4),
                            as_of.isoformat(), _now_iso())


def _schema_metadata_sweep(contract: Contract, paths: list[Path]) -> list[Violation]:
    """Cheaply confirm every shard exposes the contract's columns (presence).
    Dtype/range are confirmed on the row-scan set via pandera."""
    if contract.schema is None:
        return []
    try:
        required = set(contract.schema.columns.keys())
    except Exception:
        return []
    out: list[Violation] = []
    for p in paths:
        try:
            cols = set(pl.scan_parquet(p).collect_schema().names())
        except Exception as e:
            out.append(Violation("schema", None, f"cannot read schema of {p.name}: {e}", 1))
            continue
        miss = required - cols
        if miss:
            out.append(Violation("schema", None,
                                 f"shard '{p.name}' missing contract column(s): {sorted(miss)}", len(miss)))
    return out


def _append_drift_log(monitor_dir: Path, name: str, drift: list[DriftRow], as_of: datetime) -> None:
    if not drift:
        return
    log_path = monitor_dir / f"drift_{name}.parquet"
    rows = [{
        "run_ts": as_of,
        "dataset": name,
        "column": d.column,
        "psi": d.psi,
        "ks_stat": d.ks_stat,
        "ks_pvalue": d.ks_pvalue,
        "n_ref": d.n_ref,
        "n_cur": d.n_cur,
        "flag": d.flag,
    } for d in drift]
    new = pl.DataFrame(rows)
    if log_path.exists():  # append-only
        try:
            old = pl.read_parquet(log_path)
            new = pl.concat([old, new], how="diagonal_relaxed")
        except Exception:
            pass
    tmp = log_path.with_suffix(".parquet.tmp")
    new.write_parquet(tmp)
    os.replace(tmp, log_path)


# ════════════════════════════════════════════════════════════════════════════
# Markdown report (CODEX markdown standard) + fail-closed guard
# ════════════════════════════════════════════════════════════════════════════
def render_report(result: ValidationResult, contract: Contract) -> str:
    L: list[str] = []
    status = "PASS ✅" if result.passed else "FAIL ❌"
    errs = [v for v in result.violations if v.severity == "error" and v.check != "engine"]
    warns = [v for v in result.violations if v.severity == "warn"]
    eng = [v for v in result.violations if v.check == "engine"]
    L.append(f"# Data-contract report — {contract.name} ({status})")
    L.append("")
    L.append("## Summary")
    L.append(f"- **Dataset:** `{contract.name}` — {contract.description}")
    L.append(f"- **Verdict:** {status} — {len(errs)} blocking violation group(s), "
             f"{len(warns)} warning(s), "
             f"{len([d for d in result.drift if d.flag in ('moderate','large','ks_significant')])} drift flag(s).")
    L.append(f"- **Scanned:** {result.coverage.files_row_scanned}/{result.coverage.files_total} file(s) "
             f"row-level, {result.coverage.rows_scanned:,} rows; "
             f"{result.coverage.files_metadata_only} file(s) metadata-only.")
    L.append(f"- **as_of:** {result.as_of} · **ran_at:** {result.ran_at} · "
             f"**elapsed:** {result.elapsed_s:.3f}s")
    if result.coverage.skipped_note:
        L.append(f"- **Coverage note:** {result.coverage.skipped_note}")
    L.append("")
    L.append("## What was checked")
    L.append("This is the executable data contract — every item below is enforced as code, "
             "not assumed. A FAIL here is a fail-closed gate: the calling research/backtest run aborts.")
    L.append("")
    checks = []
    if contract.schema is not None:
        checks.append("**schema** — column presence, dtype, nullability, per-column ranges (pandera)")
    if contract.timestamp:
        checks.append(f"**timestamp** — `{contract.timestamp.column}` {contract.timestamp.order}"
                      + (f", cadence {contract.timestamp.cadence}" if contract.timestamp.cadence else ""))
    if contract.lookahead:
        checks.append(f"**lookahead-free** — no row past as_of on `{contract.lookahead.column}`")
    if contract.append_only:
        checks.append(f"**append-only** — mode `{contract.append_only.mode}`")
    if contract.address_columns:
        checks.append(f"**lowercase 0x** — {', '.join(contract.address_columns)}")
    checks.append("**finite** — no NaN/Inf in numeric columns")
    for r in contract.row_rules:
        checks.append(f"**row rule** — {r.name}: {r.description}")
    for c in checks:
        L.append(f"- {c}")
    if contract.notes:
        L.append("")
        L.append(f"> {contract.notes}")
    L.append("")

    L.append("## Contract violations (blocking)")
    if not errs:
        L.append("None — the dataset satisfies every blocking clause of the contract.")
    else:
        L.append("Unit of observation: one violation **group** (a check × column), with the number of "
                 "offending rows/shards and up to 3 examples. Every row here is fail-closed — the "
                 "downstream run aborts.")
        L.append("")
        L.append("| check | column | offending | message | examples |")
        L.append("|---|---|---:|---|---|")
        for v in errs:
            ex = json.dumps(v.examples[:3], default=str)[:220].replace("|", "\\|") if v.examples else "—"
            msg = v.message.replace("|", "\\|")
            L.append(f"| {v.check} | {v.column or '—'} | {v.n_offending} | {msg} | `{ex}` |")
        L.append("")
        L.append("**Column glossary:** `check` = which contract clause failed "
                 "(schema/monotonic_ts/cadence/finite/address/append_only/lookahead/row_rule); "
                 "`offending` = rows (or shards, for append_only) that broke it; "
                 "`examples` = sample offending rows for triage.")
    L.append("")
    L.append("## Warnings (non-blocking)")
    if not warns:
        L.append("None.")
    else:
        L.append("Reported but the run is allowed to proceed (e.g. rare exchange-maintenance gaps "
                 "under tolerance). Inspect, but they do not by themselves invalidate a result.")
        L.append("")
        L.append("| check | column | count | message |")
        L.append("|---|---|---:|---|")
        for v in warns:
            esc = v.message.replace('|', '\\|')
            L.append(f"| {v.check} | {v.column or '—'} | {v.n_offending} | {esc} |")
    if eng:
        L.append("")
        L.append("### Engine / input issues")
        for v in eng:
            L.append(f"- {v.message}")
    L.append("")

    L.append("## Drift (PSI + KS vs reference window)")
    if not result.drift:
        L.append("No monitored columns, or no reference established yet.")
    else:
        L.append("PSI convention: `<0.10` stable · `0.10–0.25` moderate shift · `>0.25` large shift. "
                 "KS p-value `<0.05` flags a distribution change (but KS is hyper-sensitive at large n — "
                 "read the KS statistic as the effect size). **Drift never aborts the run; it is a watch signal.**")
        L.append("")
        L.append("| column | PSI | KS stat | KS p | n_ref | n_cur | flag |")
        L.append("|---|---:|---:|---:|---:|---:|---|")
        for d in result.drift:
            L.append(f"| {d.column} | {d.psi} | {d.ks_stat} | {d.ks_pvalue} | {d.n_ref} | {d.n_cur} | {d.flag} |")
    L.append("")

    L.append("## Read & decision")
    if result.passed:
        L.append("- **Read:** the dataset is contract-clean; the downstream run may proceed.")
        moved = [d for d in result.drift if d.flag in ("moderate", "large", "ks_significant")]
        if moved:
            L.append(f"- **Watch:** {len(moved)} column(s) drifted ({', '.join(d.column for d in moved)}). "
                     "Not a gate, but inspect before trusting freshly-tuned parameters.")
        L.append("- **Next step:** none required.")
    else:
        L.append("- **Read:** the dataset violates its contract; any backtest/research result built on it "
                 "would be unsafe. The gate aborted the run (fail-closed).")
        L.append("- **Next step:** fix the offending rows/shards at the source (do not edit parquet in place — "
                 "regenerate the shard), then re-run. To bypass in an emergency set "
                 "`EPSILON_DATA_CONTRACT=warn` (logs but does not abort) — never bypass for a real run.")
    L.append("")
    return "\n".join(L)


class DataContractError(RuntimeError):
    pass


def write_report(result: ValidationResult, contract: Contract, monitor_dir: str | Path) -> Path:
    monitor_dir = Path(monitor_dir)
    rep_dir = monitor_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.ran_at.replace(":", "").replace("-", "").replace(".", "")[:15]
    path = rep_dir / f"{contract.name}_{stamp}.md"
    path.write_text(render_report(result, contract))
    return path


def enforce(result: ValidationResult, contract: Contract, monitor_dir: str | Path,
            *, mode: str | None = None) -> ValidationResult:
    """Fail-closed gate. mode: 'enforce' (default, abort on failure) | 'warn' | 'off'.
    Env var EPSILON_DATA_CONTRACT overrides when mode is None."""
    mode = mode or os.environ.get("EPSILON_DATA_CONTRACT", "enforce").lower()
    if mode == "off":
        return result
    if not result.passed:
        rep = write_report(result, contract, monitor_dir)
        errs = [v for v in result.violations if v.check != "engine"]
        header = (f"\n{'='*72}\n DATA-CONTRACT FAILURE — {contract.name}\n{'='*72}\n"
                  f" {len(result.violations)} violation group(s). Report: {rep}\n")
        for v in result.violations[:12]:
            header += f"   • [{v.check}] {v.column or '-'}: {v.message} (n={v.n_offending})\n"
        header += "=" * 72
        if mode == "warn":
            print(header + "\n  EPSILON_DATA_CONTRACT=warn -> proceeding despite failure (UNSAFE).")
            return result
        raise DataContractError(header)
    return result
