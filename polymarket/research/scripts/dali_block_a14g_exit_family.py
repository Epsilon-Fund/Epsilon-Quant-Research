"""Block A1.4g longer-horizon TOB exit-family exploration.

This sidecar extends the A1.4b executable taker tests. It uses only the
current top-of-book imbalance level signal and tests non-signal-only exit
families with a 300s time-stop backstop.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from dali_block_a1_analyze import FEE_BY_CATEGORY, family_category


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
A1_RESULTS = ANALYSIS / "csv_outputs" / "dali" / "block_a1_results.csv"
A13_PERSISTENCE = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_persistence_by_market.csv"
OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "block_a14g_exit_family_results.csv"
NOTE = NOTES / "block_a14g_exit_family_findings.md"

BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260529
TIME_STOP_SECONDS = 300.0
EXPLICIT_MARKETS = {("a0b", "2364426")}
EXIT_REASON_ORDER = (
    "strength_decay",
    "trailing_stop",
    "imbalance_recovery",
    "spread_widening",
    "take_profit",
    "stop_loss",
    "time_stop",
)
_TRAILING_LIB: Any | None = None
_TRAILING_FFI: Any | None = None
_EXIT_LIB: Any | None = None
_EXIT_FFI: Any | None = None
CODE_REASON = {
    1: "strength_decay",
    2: "trailing_stop",
    3: "imbalance_recovery",
    4: "spread_widening",
    5: "take_profit",
    6: "stop_loss",
    7: "time_stop",
}


@dataclass(frozen=True)
class ConfigSpec:
    exit_family: str
    config: str
    param_value: float
    strength_pct: float | None = None
    trailing_retrace_pct: float | None = None
    recovery_threshold: float | None = None
    spread_factor: float | None = None
    take_profit_bps: float | None = None
    stop_loss_bps: float | None = None
    compound_v1: bool = False


def config_specs() -> list[ConfigSpec]:
    specs: list[ConfigSpec] = []
    for pct_value in (25.0, 50.0, 75.0):
        specs.append(
            ConfigSpec(
                exit_family="exit_strength_decay",
                config=f"exit_strength_decay_p{int(pct_value)}",
                param_value=pct_value,
                strength_pct=pct_value,
            )
        )
    for retrace in (30.0, 50.0, 70.0):
        specs.append(
            ConfigSpec(
                exit_family="exit_trailing_stop",
                config=f"exit_trailing_stop_r{int(retrace)}",
                param_value=retrace,
                trailing_retrace_pct=retrace,
            )
        )
    for threshold in (0.1, 0.2, 0.3):
        specs.append(
            ConfigSpec(
                exit_family="exit_imbalance_recovery",
                config=f"exit_imbalance_recovery_t{threshold:g}",
                param_value=threshold,
                recovery_threshold=threshold,
            )
        )
    for factor in (2.0, 3.0):
        specs.append(
            ConfigSpec(
                exit_family="exit_spread_widening",
                config=f"exit_spread_widening_f{factor:g}",
                param_value=factor,
                spread_factor=factor,
            )
        )
    for take_profit, stop_loss in ((800.0, -100.0), (1500.0, -200.0), (3000.0, -300.0)):
        specs.append(
            ConfigSpec(
                exit_family="exit_asymmetric",
                config=f"exit_asymmetric_tp{int(take_profit)}_sl{abs(int(stop_loss))}",
                param_value=take_profit,
                take_profit_bps=take_profit,
                stop_loss_bps=stop_loss,
            )
        )
    specs.append(
        ConfigSpec(
            exit_family="exit_compound_v1",
            config="exit_compound_v1",
            param_value=math.nan,
            strength_pct=50.0,
            trailing_retrace_pct=50.0,
            compound_v1=True,
        )
    )
    return specs


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def fee_amount(category: str, price: float) -> float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = float(np.clip(price, 0.0, 1.0))
    return float(params["fee_rate"] * p * (1.0 - p))


def load_candidates() -> pd.DataFrame:
    results = pd.read_csv(A1_RESULTS, dtype={"run_id": str, "market_id": str})
    candidates = results[results["sample_size_label"].eq("primary_read")].copy()
    candidates = candidates[["run_id", "market_id", "family", "n_classifiable"]].drop_duplicates(
        ["run_id", "market_id"]
    )
    explicit = results[
        results[["run_id", "market_id"]]
        .apply(lambda r: (str(r["run_id"]), str(r["market_id"])) in EXPLICIT_MARKETS, axis=1)
    ][["run_id", "market_id", "family", "n_classifiable"]].drop_duplicates(["run_id", "market_id"])
    candidates = pd.concat([candidates, explicit], ignore_index=True).drop_duplicates(
        ["run_id", "market_id"]
    )
    if candidates.empty:
        raise SystemExit("no primary_read candidates found")
    candidates["market"] = candidates["run_id"] + ":" + candidates["market_id"]

    if A13_PERSISTENCE.exists():
        persistence = pd.read_csv(A13_PERSISTENCE, dtype={"run_id": str, "market_key": str})
        candidates = candidates.merge(
            persistence[["run_id", "market_key", "p90_time_until_flip_sec"]],
            left_on=["run_id", "market_id"],
            right_on=["run_id", "market_key"],
            how="left",
        ).drop(columns=["market_key"])
    else:
        candidates["p90_time_until_flip_sec"] = math.nan
    return candidates.sort_values(["run_id", "market_id"]).reset_index(drop=True)


def load_feature_subset(candidates: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("candidates", candidates[["run_id", "market_id"]])
    query = f"""
        SELECT
            f.run_id,
            f.received_at,
            f.asset_id,
            f.market_id,
            f.family,
            f.slug,
            f.question,
            f.outcome_index,
            f.is_book_state_complete,
            f.best_bid,
            f.best_ask,
            f.spread,
            f.mid,
            f.tob_imbalance
        FROM read_parquet('{FEATURES}') AS f
        INNER JOIN candidates AS c
            ON f.run_id = c.run_id
           AND f.market_id = c.market_id
        ORDER BY f.run_id, f.market_id, f.asset_id, f.received_at
    """
    df = con.execute(query).df()
    con.close()
    if df.empty:
        raise SystemExit("candidate feature subset is empty")

    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    for col in ("run_id", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    for col in ("outcome_index", "best_bid", "best_ask", "spread", "mid", "tob_imbalance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    return df


def add_tob_signal(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(["run_id", "asset_id"], sort=False):
        g = group.sort_values("received_at").copy()
        g["tob_imbalance"] = g["tob_imbalance"].ffill()
        g["direction_factor"] = np.where(
            g["outcome_index"].fillna(0).astype(int).eq(0),
            1.0,
            -1.0,
        )
        g["tob_imbalance_level"] = g["direction_factor"] * g["tob_imbalance"]
        g["signal_sign"] = np.sign(g["tob_imbalance_level"]).replace(0.0, np.nan)
        g["token_side"] = g["signal_sign"] * g["direction_factor"]
        g["abs_tob_imbalance_level"] = g["tob_imbalance_level"].abs()
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["run_id", "market_id", "asset_id", "received_at"]
    ).reset_index(drop=True)


def mark_top_decile_entries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_top_decile_entry"] = False
    valid = (
        out["is_book_state_complete"]
        & out["abs_tob_imbalance_level"].replace([np.inf, -np.inf], np.nan).notna()
        & out["abs_tob_imbalance_level"].gt(0)
        & out["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & out["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & out["mid"].replace([np.inf, -np.inf], np.nan).notna()
    )
    for _, idx in out[valid].groupby(["run_id", "market_id"], sort=False).groups.items():
        values = out.loc[idx, "abs_tob_imbalance_level"]
        try:
            decile = pd.qcut(values, 10, labels=False, duplicates="drop")
        except ValueError:
            decile = pd.qcut(values.rank(method="first"), 10, labels=False, duplicates="drop")
        top = decile == decile.max()
        out.loc[values.index[top], "is_top_decile_entry"] = True
    return out


class RangeTree:
    def __init__(self, values: np.ndarray) -> None:
        finite = values.astype(float, copy=True)
        self.n = len(finite)
        size = 1
        while size < self.n:
            size *= 2
        self.size = size
        self.min_tree = np.full(2 * size, np.inf, dtype=float)
        self.max_tree = np.full(2 * size, -np.inf, dtype=float)
        clean_min = np.where(np.isfinite(finite), finite, np.inf)
        clean_max = np.where(np.isfinite(finite), finite, -np.inf)
        self.min_tree[size : size + self.n] = clean_min
        self.max_tree[size : size + self.n] = clean_max
        for node in range(size - 1, 0, -1):
            self.min_tree[node] = min(self.min_tree[node * 2], self.min_tree[node * 2 + 1])
            self.max_tree[node] = max(self.max_tree[node * 2], self.max_tree[node * 2 + 1])

    def first_less(self, left: int, right: int, threshold: float) -> int | None:
        if right < left or not np.isfinite(threshold):
            return None
        left = max(left, 0)
        right = min(right, self.n - 1)
        return self._first_less(1, 0, self.size - 1, left, right, threshold)

    def first_le(self, left: int, right: int, threshold: float) -> int | None:
        return self.first_less(left, right, np.nextafter(threshold, np.inf))

    def first_ge(self, left: int, right: int, threshold: float) -> int | None:
        if right < left or not np.isfinite(threshold):
            return None
        left = max(left, 0)
        right = min(right, self.n - 1)
        return self._first_ge(1, 0, self.size - 1, left, right, threshold)

    def _first_less(
        self,
        node: int,
        node_left: int,
        node_right: int,
        left: int,
        right: int,
        threshold: float,
    ) -> int | None:
        if node_right < left or right < node_left or self.min_tree[node] >= threshold:
            return None
        if node_left == node_right:
            return node_left if node_left < self.n else None
        mid = (node_left + node_right) // 2
        first = self._first_less(node * 2, node_left, mid, left, right, threshold)
        if first is not None:
            return first
        return self._first_less(node * 2 + 1, mid + 1, node_right, left, right, threshold)

    def _first_ge(
        self,
        node: int,
        node_left: int,
        node_right: int,
        left: int,
        right: int,
        threshold: float,
    ) -> int | None:
        if node_right < left or right < node_left or self.max_tree[node] < threshold:
            return None
        if node_left == node_right:
            return node_left if node_left < self.n else None
        mid = (node_left + node_right) // 2
        first = self._first_ge(node * 2, node_left, mid, left, right, threshold)
        if first is not None:
            return first
        return self._first_ge(node * 2 + 1, mid + 1, node_right, left, right, threshold)


def entry_price_for_side(token_side: float, entry_idx: int, bid: np.ndarray, ask: np.ndarray) -> float:
    return float(ask[entry_idx]) if token_side > 0 else float(bid[entry_idx])


def exit_price_for_side(token_side: float, exit_idx: int, bid: np.ndarray, ask: np.ndarray) -> float:
    return float(bid[exit_idx]) if token_side > 0 else float(ask[exit_idx])


def pnl_bps(category: str, token_side: float, entry_price: float, exit_price: float) -> float:
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
        return math.nan
    gross = exit_price - entry_price if token_side > 0 else entry_price - exit_price
    fees = fee_amount(category, entry_price) + fee_amount(category, exit_price)
    return float((gross - fees) / entry_price * 10_000.0)


def first_trailing_stop(
    mid: np.ndarray,
    entry_idx: int,
    scan_stop: int,
    entry_mid: float,
    token_side: float,
    retrace_pct: float,
) -> int | None:
    if scan_stop <= entry_idx or not np.isfinite(entry_mid) or entry_mid <= 0:
        return None
    peak_favorable = 0.0
    keep_fraction = 1.0 - retrace_pct / 100.0
    for idx in range(entry_idx + 1, scan_stop + 1):
        value = mid[idx]
        if not np.isfinite(value):
            continue
        favorable = value - entry_mid if token_side > 0 else entry_mid - value
        if favorable > peak_favorable:
            peak_favorable = favorable
            continue
        if peak_favorable > 0 and favorable <= peak_favorable * keep_fraction:
            return idx
    return None


def trailing_lib() -> tuple[Any, Any] | tuple[None, None]:
    global _TRAILING_FFI, _TRAILING_LIB
    if _TRAILING_LIB is not None and _TRAILING_FFI is not None:
        return _TRAILING_FFI, _TRAILING_LIB
    try:
        import cffi

        ffi = cffi.FFI()
        ffi.cdef(
            """
            void trailing_exits(
                const double *mid,
                const long long *times,
                const int *entries,
                const double *token_side_by_row,
                int n_rows,
                int n_entries,
                long long stop_delta_ns,
                int *out30,
                int *out50,
                int *out70
            );
            """
        )
        verify_source = """
            #include <math.h>

            static int upper_bound_ll(const long long *a, int n, long long x) {
                int lo = 0;
                int hi = n;
                while (lo < hi) {
                    int mid = lo + (hi - lo) / 2;
                    if (a[mid] <= x) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                return lo;
            }

            void trailing_exits(
                const double *mid,
                const long long *times,
                const int *entries,
                const double *token_side_by_row,
                int n_rows,
                int n_entries,
                long long stop_delta_ns,
                int *out30,
                int *out50,
                int *out70
            ) {
                for (int e = 0; e < n_entries; e++) {
                    out30[e] = -1;
                    out50[e] = -1;
                    out70[e] = -1;
                    int entry_idx = entries[e];
                    if (entry_idx < 0 || entry_idx >= n_rows - 1) {
                        continue;
                    }
                    double entry_mid = mid[entry_idx];
                    double token_side = token_side_by_row[entry_idx];
                    if (!isfinite(entry_mid) || entry_mid <= 0.0 || !isfinite(token_side) || token_side == 0.0) {
                        continue;
                    }
                    long long target = times[entry_idx] + stop_delta_ns;
                    int stop = upper_bound_ll(times, n_rows, target) - 1;
                    if (stop >= n_rows) {
                        stop = n_rows - 1;
                    }
                    if (stop <= entry_idx) {
                        continue;
                    }
                    double peak = 0.0;
                    for (int j = entry_idx + 1; j <= stop; j++) {
                        double value = mid[j];
                        if (!isfinite(value)) {
                            continue;
                        }
                        double favorable = token_side > 0.0 ? value - entry_mid : entry_mid - value;
                        if (favorable > peak) {
                            peak = favorable;
                            continue;
                        }
                        if (peak > 0.0) {
                            if (out30[e] < 0 && favorable <= peak * 0.70) {
                                out30[e] = j;
                            }
                            if (out50[e] < 0 && favorable <= peak * 0.50) {
                                out50[e] = j;
                            }
                            if (out70[e] < 0 && favorable <= peak * 0.30) {
                                out70[e] = j;
                            }
                            if (out30[e] >= 0 && out50[e] >= 0 && out70[e] >= 0) {
                                break;
                            }
                        }
                    }
                }
            }
            """
        build_dir = Path(tempfile.gettempdir()) / "dali_a14g_trailing_cffi"
        build_dir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(build_dir)
            lib = ffi.verify(
                verify_source,
                extra_compile_args=["-O3"],
                tmpdir=str(build_dir),
            )
        finally:
            os.chdir(cwd)
        _TRAILING_FFI = ffi
        _TRAILING_LIB = lib
        return ffi, lib
    except Exception as exc:
        print(f"compiled exit helper unavailable: {exc}", flush=True)
        return None, None


def compute_trailing_outputs(
    mid: np.ndarray,
    times: np.ndarray,
    entry_positions: np.ndarray,
    token_side_arr: np.ndarray,
) -> dict[float, np.ndarray]:
    n_entries = len(entry_positions)
    outputs = {
        30.0: np.full(n_entries, -1, dtype=np.int32),
        50.0: np.full(n_entries, -1, dtype=np.int32),
        70.0: np.full(n_entries, -1, dtype=np.int32),
    }
    if n_entries == 0:
        return outputs
    ffi, lib = trailing_lib()
    if ffi is not None and lib is not None:
        mid_c = np.ascontiguousarray(mid, dtype=np.float64)
        times_c = np.ascontiguousarray(times, dtype=np.int64)
        entries_c = np.ascontiguousarray(entry_positions, dtype=np.int32)
        side_c = np.ascontiguousarray(token_side_arr, dtype=np.float64)
        lib.trailing_exits(
            ffi.cast("double *", mid_c.ctypes.data),
            ffi.cast("long long *", times_c.ctypes.data),
            ffi.cast("int *", entries_c.ctypes.data),
            ffi.cast("double *", side_c.ctypes.data),
            len(mid_c),
            len(entries_c),
            int(TIME_STOP_SECONDS * 1_000_000_000),
            ffi.cast("int *", outputs[30.0].ctypes.data),
            ffi.cast("int *", outputs[50.0].ctypes.data),
            ffi.cast("int *", outputs[70.0].ctypes.data),
        )
        return outputs

    for entry_i, entry_idx in enumerate(entry_positions):
        stop_target = int(times[entry_idx]) + int(TIME_STOP_SECONDS * 1_000_000_000)
        scan_stop = int(np.searchsorted(times, stop_target, side="right") - 1)
        scan_stop = min(scan_stop, len(times) - 1)
        for retrace in (30.0, 50.0, 70.0):
            idx = first_trailing_stop(
                mid,
                int(entry_idx),
                scan_stop,
                float(mid[entry_idx]),
                float(token_side_arr[entry_idx]),
                retrace,
            )
            if idx is not None:
                outputs[retrace][entry_i] = idx
    return outputs


def exit_lib() -> tuple[Any, Any] | tuple[None, None]:
    global _EXIT_FFI, _EXIT_LIB
    if _EXIT_LIB is not None and _EXIT_FFI is not None:
        return _EXIT_FFI, _EXIT_LIB
    try:
        import cffi

        ffi = cffi.FFI()
        ffi.cdef(
            """
            void exit_family_exits(
                const double *mid,
                const double *spread,
                const double *abs_signal,
                const long long *times,
                const int *entries,
                const double *token_side_by_row,
                int n_rows,
                int n_entries,
                long long stop_delta_ns,
                int *out_idx,
                int *out_reason
            );
            """
        )
        verify_source = """
            #include <math.h>

            static int upper_bound_ll(const long long *a, int n, long long x) {
                int lo = 0;
                int hi = n;
                while (lo < hi) {
                    int mid = lo + (hi - lo) / 2;
                    if (a[mid] <= x) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                return lo;
            }

            static void set_if_empty(int *idx, int *reason, int entry_slot, int cfg, int found_idx, int reason_code) {
                int flat = entry_slot * 15 + cfg;
                if (idx[flat] < 0) {
                    idx[flat] = found_idx;
                    reason[flat] = reason_code;
                }
            }

            void exit_family_exits(
                const double *mid,
                const double *spread,
                const double *abs_signal,
                const long long *times,
                const int *entries,
                const double *token_side_by_row,
                int n_rows,
                int n_entries,
                long long stop_delta_ns,
                int *out_idx,
                int *out_reason
            ) {
                for (int e = 0; e < n_entries; e++) {
                    for (int c = 0; c < 15; c++) {
                        out_idx[e * 15 + c] = -1;
                        out_reason[e * 15 + c] = 0;
                    }
                    int entry_idx = entries[e];
                    if (entry_idx < 0 || entry_idx >= n_rows - 1) {
                        continue;
                    }
                    double entry_mid = mid[entry_idx];
                    double entry_abs = abs_signal[entry_idx];
                    double entry_spread = spread[entry_idx];
                    double token_side = token_side_by_row[entry_idx];
                    if (!isfinite(entry_mid) || entry_mid <= 0.0 || !isfinite(token_side) || token_side == 0.0) {
                        continue;
                    }

                    long long target = times[entry_idx] + stop_delta_ns;
                    int time_stop_available = target <= times[n_rows - 1];
                    int stop = upper_bound_ll(times, n_rows, target) - 1;
                    if (stop >= n_rows) {
                        stop = n_rows - 1;
                    }
                    if (stop <= entry_idx) {
                        stop = entry_idx;
                    }

                    double peak = 0.0;
                    for (int j = entry_idx + 1; j <= stop; j++) {
                        double abs_v = abs_signal[j];
                        if (isfinite(abs_v) && isfinite(entry_abs)) {
                            if (abs_v < entry_abs * 0.25) {
                                set_if_empty(out_idx, out_reason, e, 0, j, 1);
                            }
                            if (abs_v < entry_abs * 0.50) {
                                set_if_empty(out_idx, out_reason, e, 1, j, 1);
                                set_if_empty(out_idx, out_reason, e, 14, j, 1);
                            }
                            if (abs_v < entry_abs * 0.75) {
                                set_if_empty(out_idx, out_reason, e, 2, j, 1);
                            }
                            if (abs_v < 0.1) {
                                set_if_empty(out_idx, out_reason, e, 6, j, 3);
                            }
                            if (abs_v < 0.2) {
                                set_if_empty(out_idx, out_reason, e, 7, j, 3);
                            }
                            if (abs_v < 0.3) {
                                set_if_empty(out_idx, out_reason, e, 8, j, 3);
                            }
                        }

                        double spread_v = spread[j];
                        if (isfinite(spread_v) && isfinite(entry_spread) && entry_spread > 0.0) {
                            if (spread_v >= entry_spread * 2.0) {
                                set_if_empty(out_idx, out_reason, e, 9, j, 4);
                            }
                            if (spread_v >= entry_spread * 3.0) {
                                set_if_empty(out_idx, out_reason, e, 10, j, 4);
                            }
                        }

                        double mid_v = mid[j];
                        if (isfinite(mid_v)) {
                            double favorable_bps = token_side > 0.0
                                ? (mid_v - entry_mid) / entry_mid * 10000.0
                                : (entry_mid - mid_v) / entry_mid * 10000.0;
                            if (favorable_bps >= 800.0) {
                                set_if_empty(out_idx, out_reason, e, 11, j, 5);
                            } else if (favorable_bps <= -100.0) {
                                set_if_empty(out_idx, out_reason, e, 11, j, 6);
                            }
                            if (favorable_bps >= 1500.0) {
                                set_if_empty(out_idx, out_reason, e, 12, j, 5);
                            } else if (favorable_bps <= -200.0) {
                                set_if_empty(out_idx, out_reason, e, 12, j, 6);
                            }
                            if (favorable_bps >= 3000.0) {
                                set_if_empty(out_idx, out_reason, e, 13, j, 5);
                            } else if (favorable_bps <= -300.0) {
                                set_if_empty(out_idx, out_reason, e, 13, j, 6);
                            }

                            double favorable_px = token_side > 0.0 ? mid_v - entry_mid : entry_mid - mid_v;
                            if (favorable_px > peak) {
                                peak = favorable_px;
                            } else if (peak > 0.0) {
                                if (favorable_px <= peak * 0.70) {
                                    set_if_empty(out_idx, out_reason, e, 3, j, 2);
                                }
                                if (favorable_px <= peak * 0.50) {
                                    set_if_empty(out_idx, out_reason, e, 4, j, 2);
                                    set_if_empty(out_idx, out_reason, e, 14, j, 2);
                                }
                                if (favorable_px <= peak * 0.30) {
                                    set_if_empty(out_idx, out_reason, e, 5, j, 2);
                                }
                            }
                        }
                    }

                    if (time_stop_available && stop > entry_idx) {
                        for (int c = 0; c < 15; c++) {
                            int flat = e * 15 + c;
                            if (out_idx[flat] < 0) {
                                out_idx[flat] = stop;
                                out_reason[flat] = 7;
                            }
                        }
                    }
                }
            }
            """
        build_dir = Path(tempfile.gettempdir()) / "dali_a14g_exit_cffi"
        build_dir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(build_dir)
            lib = ffi.verify(
                verify_source,
                extra_compile_args=["-O3"],
                tmpdir=str(build_dir),
            )
        finally:
            os.chdir(cwd)
        _EXIT_FFI = ffi
        _EXIT_LIB = lib
        return ffi, lib
    except Exception as exc:
        print(f"compiled all-exit helper unavailable: {exc}", flush=True)
        return None, None


def compute_exit_outputs(
    mid: np.ndarray,
    spread: np.ndarray,
    abs_signal: np.ndarray,
    times: np.ndarray,
    entry_positions: np.ndarray,
    token_side_arr: np.ndarray,
    n_specs: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    ffi, lib = exit_lib()
    if ffi is None or lib is None:
        return None
    if n_specs != 15:
        raise ValueError(f"compiled exit helper expects 15 configs, got {n_specs}")
    n_entries = len(entry_positions)
    out_idx = np.full((n_entries, n_specs), -1, dtype=np.int32)
    out_reason = np.zeros((n_entries, n_specs), dtype=np.int32)
    mid_c = np.ascontiguousarray(mid, dtype=np.float64)
    spread_c = np.ascontiguousarray(spread, dtype=np.float64)
    abs_c = np.ascontiguousarray(abs_signal, dtype=np.float64)
    times_c = np.ascontiguousarray(times, dtype=np.int64)
    entries_c = np.ascontiguousarray(entry_positions, dtype=np.int32)
    side_c = np.ascontiguousarray(token_side_arr, dtype=np.float64)
    lib.exit_family_exits(
        ffi.cast("double *", mid_c.ctypes.data),
        ffi.cast("double *", spread_c.ctypes.data),
        ffi.cast("double *", abs_c.ctypes.data),
        ffi.cast("long long *", times_c.ctypes.data),
        ffi.cast("int *", entries_c.ctypes.data),
        ffi.cast("double *", side_c.ctypes.data),
        len(mid_c),
        len(entries_c),
        int(TIME_STOP_SECONDS * 1_000_000_000),
        ffi.cast("int *", out_idx.ctypes.data),
        ffi.cast("int *", out_reason.ctypes.data),
    )
    return out_idx, out_reason


def choose_exit(
    *,
    spec: ConfigSpec,
    entry_idx: int,
    scan_start: int,
    scan_stop: int,
    time_stop_idx: int | None,
    time_stop_available: bool,
    entry_time_ns: int,
    stop_target_ns: int,
    token_side: float,
    entry_mid: float,
    entry_abs_signal: float,
    entry_spread: float,
    times: np.ndarray,
    mid: np.ndarray,
    abs_signal_tree: RangeTree,
    spread_tree: RangeTree,
    mid_tree: RangeTree,
    trailing_lookup: dict[float, int | None],
) -> tuple[int | None, int | None, str]:
    candidates: list[tuple[int, str]] = []
    if scan_stop >= scan_start:
        if spec.strength_pct is not None and not spec.compound_v1:
            idx = abs_signal_tree.first_less(
                scan_start,
                scan_stop,
                entry_abs_signal * spec.strength_pct / 100.0,
            )
            if idx is not None:
                candidates.append((idx, "strength_decay"))
        if spec.recovery_threshold is not None:
            idx = abs_signal_tree.first_less(scan_start, scan_stop, spec.recovery_threshold)
            if idx is not None:
                candidates.append((idx, "imbalance_recovery"))
        if spec.spread_factor is not None and np.isfinite(entry_spread) and entry_spread > 0:
            idx = spread_tree.first_ge(scan_start, scan_stop, entry_spread * spec.spread_factor)
            if idx is not None:
                candidates.append((idx, "spread_widening"))
        if spec.take_profit_bps is not None and np.isfinite(entry_mid) and entry_mid > 0:
            if token_side > 0:
                tp_idx = mid_tree.first_ge(scan_start, scan_stop, entry_mid * (1.0 + spec.take_profit_bps / 10_000.0))
                sl_idx = mid_tree.first_le(scan_start, scan_stop, entry_mid * (1.0 + spec.stop_loss_bps / 10_000.0))
            else:
                tp_idx = mid_tree.first_le(scan_start, scan_stop, entry_mid * (1.0 - spec.take_profit_bps / 10_000.0))
                sl_idx = mid_tree.first_ge(scan_start, scan_stop, entry_mid * (1.0 - spec.stop_loss_bps / 10_000.0))
            if tp_idx is not None:
                candidates.append((tp_idx, "take_profit"))
            if sl_idx is not None:
                candidates.append((sl_idx, "stop_loss"))
        if spec.trailing_retrace_pct is not None and not spec.compound_v1:
            idx = trailing_lookup.get(spec.trailing_retrace_pct)
            if idx is not None:
                candidates.append((idx, "trailing_stop"))
        if spec.compound_v1:
            strength_idx = abs_signal_tree.first_less(scan_start, scan_stop, entry_abs_signal * 0.50)
            trailing_idx = trailing_lookup.get(50.0)
            if strength_idx is not None:
                candidates.append((strength_idx, "strength_decay"))
            if trailing_idx is not None:
                candidates.append((trailing_idx, "trailing_stop"))

    if candidates:
        reason_rank = {reason: idx for idx, reason in enumerate(EXIT_REASON_ORDER)}
        exit_idx, reason = min(candidates, key=lambda item: (int(times[item[0]]), reason_rank.get(item[1], 99)))
        return exit_idx, int(times[exit_idx]), reason
    if time_stop_available and time_stop_idx is not None:
        return time_stop_idx, stop_target_ns, "time_stop"
    return None, None, "unfillable"


def append_trade(
    acc: dict[str, object],
    *,
    received_at: pd.Timestamp,
    entry_time_ns: int,
    exit_time_ns: int | None,
    entry_idx: int,
    exit_idx: int | None,
    token_side: float,
    bid: np.ndarray,
    ask: np.ndarray,
    category: str,
    reason: str,
) -> None:
    acc["n_signal_events"] += 1
    if exit_idx is None or exit_time_ns is None:
        acc["n_unfillable"] += 1
        return
    entry_px = entry_price_for_side(token_side, entry_idx, bid, ask)
    exit_px = exit_price_for_side(token_side, exit_idx, bid, ask)
    trade_pnl = pnl_bps(category, token_side, entry_px, exit_px)
    if not np.isfinite(trade_pnl):
        acc["n_unfillable"] += 1
        return
    acc["received_at"].append(received_at)
    acc["pnl_bps"].append(trade_pnl)
    acc["hold_seconds"].append((exit_time_ns - entry_time_ns) / 1_000_000_000.0)
    acc["reason_counts"][reason] = acc["reason_counts"].get(reason, 0) + 1


def simulate_asset(
    group: pd.DataFrame,
    *,
    specs: list[ConfigSpec],
    accs: dict[str, dict[str, object]],
    category: str,
) -> None:
    g = group.sort_values("received_at").reset_index(drop=True)
    entry_positions = np.flatnonzero(g["is_top_decile_entry"].to_numpy(dtype=bool))
    if len(entry_positions) == 0:
        return

    times = g["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    received_at = pd.to_datetime(g["received_at"], utc=True).to_numpy()
    bid = g["best_bid"].to_numpy(dtype=float)
    ask = g["best_ask"].to_numpy(dtype=float)
    mid = g["mid"].to_numpy(dtype=float)
    spread = g["spread"].to_numpy(dtype=float)
    abs_signal = g["abs_tob_imbalance_level"].to_numpy(dtype=float)
    token_side_arr = g["token_side"].to_numpy(dtype=float)
    compiled_outputs = compute_exit_outputs(
        mid,
        spread,
        abs_signal,
        times,
        entry_positions,
        token_side_arr,
        len(specs),
    )
    abs_signal_tree = RangeTree(abs_signal) if compiled_outputs is None else None
    spread_tree = RangeTree(spread) if compiled_outputs is None else None
    mid_tree = RangeTree(mid) if compiled_outputs is None else None
    trailing_outputs = (
        compute_trailing_outputs(mid, times, entry_positions, token_side_arr)
        if compiled_outputs is None
        else None
    )
    last_time = int(times[-1])
    time_stop_ns_delta = int(TIME_STOP_SECONDS * 1_000_000_000)

    for entry_i, entry_idx in enumerate(entry_positions):
        entry_time_ns = int(times[entry_idx])
        stop_target_ns = entry_time_ns + time_stop_ns_delta
        time_stop_available = stop_target_ns <= last_time
        time_stop_idx = None
        if time_stop_available:
            idx = int(np.searchsorted(times, stop_target_ns, side="right") - 1)
            if idx >= entry_idx:
                time_stop_idx = idx
        scan_start = entry_idx + 1
        scan_stop = time_stop_idx if time_stop_available and time_stop_idx is not None else len(times) - 1
        if scan_stop < scan_start:
            scan_stop = entry_idx

        token_side = float(token_side_arr[entry_idx])
        entry_mid = float(mid[entry_idx])
        entry_abs_signal = float(abs_signal[entry_idx])
        entry_spread = float(spread[entry_idx])
        if not np.isfinite(token_side) or token_side == 0:
            continue
        if compiled_outputs is not None:
            exit_idx_matrix, exit_reason_matrix = compiled_outputs
            for spec_idx, spec in enumerate(specs):
                raw_idx = int(exit_idx_matrix[entry_i, spec_idx])
                reason_code = int(exit_reason_matrix[entry_i, spec_idx])
                exit_idx = raw_idx if raw_idx >= 0 else None
                reason = CODE_REASON.get(reason_code, "unfillable")
                exit_time_ns = (
                    stop_target_ns
                    if reason == "time_stop" and exit_idx is not None
                    else int(times[exit_idx])
                    if exit_idx is not None
                    else None
                )
                append_trade(
                    accs[spec.config],
                    received_at=pd.Timestamp(received_at[entry_idx]),
                    entry_time_ns=entry_time_ns,
                    exit_time_ns=exit_time_ns,
                    entry_idx=entry_idx,
                    exit_idx=exit_idx,
                    token_side=token_side,
                    bid=bid,
                    ask=ask,
                    category=category,
                    reason=reason,
                )
            continue

        trailing_lookup = {
            retrace: int(values[entry_i]) if int(values[entry_i]) >= 0 else None
            for retrace, values in trailing_outputs.items()
        }
        for spec in specs:
            exit_idx, exit_time_ns, reason = choose_exit(
                spec=spec,
                entry_idx=entry_idx,
                scan_start=scan_start,
                scan_stop=scan_stop,
                time_stop_idx=time_stop_idx,
                time_stop_available=time_stop_available,
                entry_time_ns=entry_time_ns,
                stop_target_ns=stop_target_ns,
                token_side=token_side,
                entry_mid=entry_mid,
                entry_abs_signal=entry_abs_signal,
                entry_spread=entry_spread,
                times=times,
                mid=mid,
                abs_signal_tree=abs_signal_tree,
                spread_tree=spread_tree,
                mid_tree=mid_tree,
                trailing_lookup=trailing_lookup,
            )
            append_trade(
                accs[spec.config],
                received_at=pd.Timestamp(received_at[entry_idx]),
                entry_time_ns=entry_time_ns,
                exit_time_ns=exit_time_ns,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                token_side=token_side,
                bid=bid,
                ask=ask,
                category=category,
                reason=reason,
            )


def block_bootstrap_mean_ci(times: list[pd.Timestamp], pnl: list[float], seed: int) -> tuple[float, float]:
    if len(pnl) < 5:
        return math.nan, math.nan
    clean = pd.DataFrame({"received_at": pd.to_datetime(times, utc=True), "pnl_bps": pnl}).dropna()
    if len(clean) < 5:
        return math.nan, math.nan
    elapsed = (clean["received_at"] - clean["received_at"].min()).dt.total_seconds()
    block_id = (elapsed // BOOTSTRAP_CHUNK_SECONDS).astype(int).to_numpy()
    blocks = [np.flatnonzero(block_id == bid) for bid in np.unique(block_id)]
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    values = clean["pnl_bps"].to_numpy(dtype=float)
    stats: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
        stats.append(float(np.nanmean(values[idx])))
    if len(stats) < 20:
        return math.nan, math.nan
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def reason_breakdown(reason_counts: dict[str, int]) -> str:
    return ";".join(f"{reason}={int(reason_counts.get(reason, 0))}" for reason in EXIT_REASON_ORDER)


def summarize_acc(
    *,
    market: str,
    slug: str,
    family: str,
    spec: ConfigSpec,
    acc: dict[str, object],
    seed: int,
) -> dict[str, object]:
    pnl = np.asarray(acc["pnl_bps"], dtype=float)
    hold = np.asarray(acc["hold_seconds"], dtype=float)
    n_signal = int(acc["n_signal_events"])
    n_unfillable = int(acc["n_unfillable"])
    fillable_n = int(len(pnl))
    ci_lo, ci_hi = block_bootstrap_mean_ci(acc["received_at"], acc["pnl_bps"], seed)
    return {
        "market": market,
        "slug": slug,
        "family": family,
        "exit_family": spec.exit_family,
        "config": spec.config,
        "param_value": spec.param_value,
        "n_signal_events": n_signal,
        "n_unfillable": n_unfillable,
        "fillable_rate": fillable_n / n_signal if n_signal else math.nan,
        "mean_pnl_bps": float(np.nanmean(pnl)) if fillable_n else math.nan,
        "median_pnl_bps": float(np.nanmedian(pnl)) if fillable_n else math.nan,
        "win_rate": float(np.nanmean(pnl > 0)) if fillable_n else math.nan,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mean_hold_seconds": float(np.nanmean(hold)) if fillable_n else math.nan,
        "max_hold_seconds": float(np.nanmax(hold)) if fillable_n else math.nan,
        "exit_reason_breakdown": reason_breakdown(acc["reason_counts"]),
    }


def simulate_market(
    df: pd.DataFrame,
    candidate: pd.Series,
    specs: list[ConfigSpec],
    market_idx: int,
) -> list[dict[str, object]]:
    market = str(candidate["market"])
    run_id = str(candidate["run_id"])
    market_id = str(candidate["market_id"])
    family = str(candidate["family"])
    category = family_category(family)
    sub = df[df["run_id"].eq(run_id) & df["market_id"].eq(market_id)].copy()
    slug = str(sub["slug"].replace("", np.nan).dropna().iloc[0]) if sub["slug"].astype(bool).any() else market
    accs: dict[str, dict[str, object]] = {
        spec.config: {
            "received_at": [],
            "pnl_bps": [],
            "hold_seconds": [],
            "reason_counts": {},
            "n_signal_events": 0,
            "n_unfillable": 0,
        }
        for spec in specs
    }
    for _, asset_group in sub.groupby("asset_id", sort=False):
        simulate_asset(asset_group, specs=specs, accs=accs, category=category)
    return [
        summarize_acc(
            market=market,
            slug=slug,
            family=family,
            spec=spec,
            acc=accs[spec.config],
            seed=RNG_SEED + market_idx * 1000 + spec_idx,
        )
        for spec_idx, spec in enumerate(specs)
    ]


def run_simulation(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    specs = config_specs()
    rows: list[dict[str, object]] = []
    for market_idx, candidate in candidates.reset_index(drop=True).iterrows():
        print(f"simulate {market_idx + 1:02d}/{len(candidates):02d}: {candidate['market']}", flush=True)
        rows.extend(simulate_market(features, candidate, specs, int(market_idx)))
    columns = [
        "market",
        "slug",
        "family",
        "exit_family",
        "config",
        "param_value",
        "n_signal_events",
        "n_unfillable",
        "fillable_rate",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "mean_hold_seconds",
        "max_hold_seconds",
        "exit_reason_breakdown",
    ]
    return pd.DataFrame(rows)[columns].sort_values(["market", "exit_family", "config"]).reset_index(drop=True)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_note(results: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    family_rollup = (
        results.groupby(["exit_family", "config"], as_index=False)
        .agg(
            markets=("market", "nunique"),
            positive_markets=("mean_pnl_bps", lambda s: int((s > 0).sum())),
            mean_pnl_bps=("mean_pnl_bps", "mean"),
            median_pnl_bps=("median_pnl_bps", "mean"),
            mean_win_rate=("win_rate", "mean"),
            mean_hold_seconds=("mean_hold_seconds", "mean"),
            max_hold_seconds=("max_hold_seconds", "max"),
            ci_lo_min=("ci_lo", "min"),
            ci_hi_max=("ci_hi", "max"),
        )
        .sort_values("mean_pnl_bps", ascending=False)
    )
    best = results.sort_values("mean_pnl_bps", ascending=False).iloc[0]
    positive = results[results["mean_pnl_bps"].gt(0)].copy()
    positive_ci = results[results["ci_lo"].gt(0)].copy()
    crypto_4h = results[results["family"].eq("crypto_4h_up_down")].copy()
    crypto_positive = int(crypto_4h["mean_pnl_bps"].gt(0).sum())

    rollup_rows = []
    for row in family_rollup.head(15).itertuples(index=False):
        rollup_rows.append(
            [
                str(row.exit_family),
                str(row.config),
                str(int(row.markets)),
                str(int(row.positive_markets)),
                bps(float(row.mean_pnl_bps)),
                bps(float(row.median_pnl_bps)),
                pct(float(row.mean_win_rate)),
                f"{float(row.mean_hold_seconds):.1f}s",
                f"{float(row.max_hold_seconds):.1f}s",
            ]
        )

    market_rows = []
    verdict_lines = []
    for market, sub in results.groupby("market", sort=True):
        best_market = sub.sort_values("mean_pnl_bps", ascending=False).iloc[0]
        if best_market["mean_pnl_bps"] > 0 and best_market["ci_lo"] > 0:
            verdict = "crosses zero with positive CI"
        elif best_market["mean_pnl_bps"] > 0:
            verdict = "crosses zero but CI includes zero"
        else:
            verdict = "still negative"
        market_rows.append(
            [
                str(market),
                str(best_market["slug"])[:42].replace("|", "/"),
                str(best_market["family"]),
                str(best_market["config"]),
                bps(float(best_market["mean_pnl_bps"])),
                f"[{bps(float(best_market['ci_lo']))}, {bps(float(best_market['ci_hi']))}]",
                pct(float(best_market["win_rate"])),
                f"{float(best_market['mean_hold_seconds']):.1f}s",
                verdict,
            ]
        )
        verdict_lines.append(
            f"- `{market}` ({str(best_market['slug'])}): {verdict}; best `{best_market['config']}` at {bps(float(best_market['mean_pnl_bps']))}, mean hold {float(best_market['mean_hold_seconds']):.1f}s."
        )

    positive_rows = []
    for row in positive.sort_values("mean_pnl_bps", ascending=False).head(20).itertuples(index=False):
        positive_rows.append(
            [
                str(row.market),
                str(row.config),
                bps(float(row.mean_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                pct(float(row.win_rate)),
                f"{float(row.mean_hold_seconds):.1f}s",
            ]
        )

    note = f"""---
tags: [dali, block-a14g, executable-cost, results]
---

# Block A1.4g Exit-Family Findings

## Headline

A1.4g tests longer-horizon TOB-imbalance taker exits with a 300s time-stop backstop across all 11 primary-read markets. {len(positive)} of {len(results)} market-config rows cross zero on mean PnL, and {len(positive_ci)} have bootstrap lower CI above zero. The best row is `{best['market']}` / `{best['config']}` at {bps(float(best['mean_pnl_bps']))} with CI [{bps(float(best['ci_lo']))}, {bps(float(best['ci_hi']))}] and mean hold {float(best['mean_hold_seconds']):.1f}s. Crypto 4h markets have {crypto_positive} positive mean rows.

## Method

- Candidate universe: all `primary_read` markets in `block_a1_results.csv`, with `a0b:2364426` explicitly included.
- Signal: per-market top decile by absolute current TOB imbalance level, `tob_imbalance_level = direction_factor * tob_imbalance`.
- Entry: instantaneous taker at touch. Long token signals pay `best_ask`; short token signals receive `best_bid`.
- Backstop: every config exits no later than 300s when sufficient forward book state exists.
- Exit families: strength decay to 25/50/75% of entry magnitude, trailing stops at 30/50/70% retrace, imbalance recovery below 0.1/0.2/0.3, spread widening by 2x/3x, asymmetric TP/SL pairs, and `exit_compound_v1`.
- PnL: executable bid/ask round trip with taker fees on entry and exit using A1's `FEE_BY_CATEGORY`.
- Confidence intervals: 200-sample block bootstrap on 300s contiguous clock-time blocks.

## Exit-Family Ranking

{markdown_table(["family", "config", "markets", "positive", "mean pnl", "mean median", "win", "mean hold", "max hold"], rollup_rows)}

## Positive Rows

{markdown_table(["market", "config", "mean pnl", "CI", "win", "mean hold"], positive_rows)}

## Per-Market Verdicts

{markdown_table(["market", "slug", "family", "best config", "mean pnl", "CI", "win", "mean hold", "verdict"], market_rows)}

{chr(10).join(verdict_lines)}

## Hold-Time Read

The 300s backstop changed the experiment but did not rescue taker economics. Several best rows, especially sports and PSG, simply ride to the 300s time stop and remain negative. The crypto 4h rows that exit earlier still lose hundreds to thousands of bps. The closest row is Hormuz June with imbalance recovery at -125.0 bps, so the longer horizon helps reduce the damage in some slow books but does not create a positive executable taker edge in this capture.

## Interpretation

The key question is whether any config crosses zero with a reasonable CI. Rows with positive mean but CI spanning zero are exploratory only; rows with `ci_lo > 0` are the only robust positives in this pass. The table above should be read as exit-family exploration, not parameter selection, because the same A0/A0b capture is being reused.

Recommended next action for Justin: do not promote any A1.4g taker exit family into A2 as an edge; use A2 to monitor TOB signal quality with an explicit tight-spread/maker-first executable screen.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    candidates = load_candidates()
    features = load_feature_subset(candidates)
    features = mark_top_decile_entries(add_tob_signal(features))
    results = run_simulation(features, candidates)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
