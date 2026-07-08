"""Task-5.1 evaluation methodology — whole-market nested CPCV + leakage-safe cohorts.

This module supersedes the Task-5 single-calendar-cut protocol (`mm_eval.protocol`
`SPLIT_TS_MS`) for cross-config gating. The Task-5 audit found that cut confounded
generalization with the τ regime: the same markets sat on both sides, so IS = calm
mid-life and OOS = the pre-resolution endgame (the IS→OOS drop was near-uniform across
configs and scaled with cap — a regime signature, not overfit). The fixes pinned here:

* **Split unit = the whole market (event / NegRisk group).** A group is entirely in a
  test fold or entirely out — its full τ arc (mid-life + endgame) stays together. τ is a
  CONDITIONING dimension (the regime axis of the reported surface), never the split axis.
* **Combinatorial purged CV over groups** — :func:`generate_group_cpcv_splits` is a PORT
  of ``infrastructure/walkforward/cpcv_engine.py::generate_cpcv_splits`` (López de Prado
  CPCV), adapted from time-bars to event groups (the module itself is never imported —
  it drags the crypto backtester, violating the never-cross-import invariant). Groups
  are ordered by lifecycle start; ``purge_groups`` drops order-adjacent training groups
  at each test boundary. NOTE the honest caveat: politics groups run CONCURRENTLY, so
  calendar-time purging between groups is not fully possible — the residual cross-market
  channel (shared macro news) is reported (`overlap diagnostic`), not hidden. Labels are
  30 s markouts + per-group costed PnL, so label overlap across groups is negligible.
* **Nested selection (inner-select / outer-estimate):** for every CPCV split, knobs are
  chosen by inner CV on the outer-TRAINING groups only, and the chosen config is scored
  ONCE on the held-out groups. This kills the Task-5 selection seam (non-nested selection
  is optimistically biased).
* **Leakage-safe cohort features** — computed ONLY from each group's lead-in window
  (:func:`lead_in_features`), never from the evaluation window, so fold assignment cannot
  peek at the outcome it is judged on. Folds are balanced on those cohorts
  (:func:`assign_folds`).
* **Overfitting apparatus (shared infra, reused wholesale):**
  ``infrastructure/validation/overfitting_audit`` supplies ``pbo_cscv`` (real CSCV PBO —
  blocks = event groups), ``deflated_sharpe_ratio`` (+ ``effective_n_trials``), and
  ``whites_reality_check``. Fed with per-config per-GROUP returns.

Everything is deterministic (seeded) and lookahead-free.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mm_eval.metrics import CENTS
from mm_eval.protocol import _liq_mark, _touch_series, _load_audit

# ──────────────────────────────────────────────────────────────────────────────
# 1. The ported CPCV split generator (unit = event group)
# ──────────────────────────────────────────────────────────────────────────────


def generate_group_cpcv_splits(n_groups: int, n_folds: int, k_test: int,
                               purge_groups: int = 0,
                               fold_sizes: list[int] | None = None) -> dict:
    """Enumerate every C(N, k) CPCV split of ``n_groups`` ordered groups + complete paths.

    PORT of ``cpcv_engine.generate_cpcv_splits`` with the unit changed bars → groups
    (group indices refer to a caller-supplied ordering). ``fold_sizes`` (summing to
    ``n_groups``) lets the caller pass a cohort-balanced fold partition instead of the
    original's equal contiguous slices — the combinatorial split/path structure operates
    on FOLDS, so unequal folds change nothing else. Requires ``n_folds % k_test == 0`` so
    complete out-of-sample paths exist (each path = a partition of all folds into
    test-sets, giving every group exactly one honest OOS estimate per path).

    ``purge_groups`` removes order-adjacent training groups at each train/test fold
    boundary — the group-axis analogue of the bar purge. NOTE on this stack it defaults
    to 0 by design: labels here (30 s markouts, per-group costed PnL) never span group
    boundaries — each group is a separate engine run — so the López-de-Prado purge lives
    at the markout level inside each run, and whole-market groups run CONCURRENTLY in
    calendar time (an ordering purge between them is symbolic). The residual concurrency
    channel is quantified by :func:`fold_overlap_diagnostic` instead of pretended away.

    Returns {"folds": [(start, end)...] group-index ranges, "splits": [...], "paths": [...]}
    with the same shapes as the crypto original.
    """
    if k_test < 1 or n_folds < k_test:
        raise ValueError(f"require 1 <= k_test <= n_folds; got {n_folds=}, {k_test=}")
    if n_folds % k_test != 0:
        raise ValueError(f"n_folds must be divisible by k_test; got {n_folds=}, {k_test=}")
    if n_groups < n_folds:
        raise ValueError(f"need at least one group per fold; got {n_groups=}, {n_folds=}")

    if fold_sizes is not None:
        if len(fold_sizes) != n_folds or sum(fold_sizes) != n_groups or min(fold_sizes) < 1:
            raise ValueError(f"fold_sizes must be {n_folds} positive ints summing to {n_groups}")
        folds = []
        s = 0
        for sz in fold_sizes:
            folds.append((s, s + sz))
            s += sz
    else:
        base = n_groups // n_folds
        folds = []
        for i in range(n_folds):
            s = i * base
            e = s + base if i < n_folds - 1 else n_groups
            folds.append((s, e))

    splits = []
    for split_id, test_combo in enumerate(itertools.combinations(range(n_folds), k_test)):
        test_set = set(test_combo)
        test_groups_by_fold = {f: list(range(folds[f][0], folds[f][1])) for f in test_combo}
        train_groups: list[int] = []
        for f in range(n_folds):
            if f in test_set:
                continue
            f_start, f_end = folds[f]
            eff_start, eff_end = f_start, f_end
            if (f - 1) in test_set:
                eff_start = min(f_start + purge_groups, f_end)
            if (f + 1) in test_set:
                eff_end = max(eff_start, f_end - purge_groups)
            train_groups.extend(range(eff_start, eff_end))
        splits.append({
            "split_id": split_id,
            "test_fold_indices": test_combo,
            "train_groups": np.array(train_groups, dtype=int),
            "test_groups_by_fold": test_groups_by_fold,
        })

    # complete OOS paths: partitions of all folds into disjoint test sets, one per split
    fold_to_split_ids: dict[int, list[int]] = {f: [] for f in range(n_folds)}
    for sp in splits:
        for f in sp["test_fold_indices"]:
            fold_to_split_ids[f].append(sp["split_id"])

    paths: list[dict] = []

    def _recurse(remaining: set, used: set, current: list) -> None:
        if not remaining:
            paths.append({"path_id": len(paths),
                          "split_assignments": sorted(current, key=lambda x: x[0])})
            return
        first = min(remaining)
        for sid in fold_to_split_ids[first]:
            tf = set(splits[sid]["test_fold_indices"])
            if tf & used:
                continue
            for f in tf:
                current.append((f, sid))
            _recurse(remaining - tf, used | tf, current)
            for _ in tf:
                current.pop()

    _recurse(set(range(n_folds)), set(), [])
    return {"folds": folds, "splits": splits, "paths": paths}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Leakage-safe cohort features (lead-in window only) + balanced fold assignment
# ──────────────────────────────────────────────────────────────────────────────

# Lead-in = the first min(LEAD_IN_CAP_H, LEAD_IN_FRAC × observed span) of each GROUP's
# capture. Pre-registered before any scoring; features never touch the evaluation window.
LEAD_IN_FRAC = 0.25
LEAD_IN_CAP_H = 24.0
LEAD_IN_MIN_H = 1.0


def lead_in_span(t_first: int, t_last: int) -> tuple[int, int]:
    """(start, end) ms of the leakage-safe lead-in window for a group's observed span."""
    span_h = (t_last - t_first) / 3.6e6
    h = min(max(span_h * LEAD_IN_FRAC, LEAD_IN_MIN_H), LEAD_IN_CAP_H)
    return t_first, t_first + int(h * 3.6e6)


def lead_in_features(group_id: str, token_dirs: list[Path], con) -> dict:
    """Structural cohort features of one group, from its lead-in window ONLY.

    Features (each computed per token then volume-pooled over the group's tokens):

    * ``flow_rate``   — traded contracts per hour (liquidity / flow volume).
    * ``depth``       — median top-of-book depth (book depth).
    * ``sweep_share`` — fraction of lead-in trade volume executing BEYOND the touch
      (participant aggressiveness — the LOTECH Lens-1 axis).
    * ``spread_c``    — median touch spread in cents.
    * ``vol_c``       — std of 1-minute mid changes in cents (spike propensity).
    * ``avg_price``   — mean traded price (the ~50¢-parked diagnostic).
    * ``turnover``    — trades per bba checkpoint (queue-speed proxy).
    """
    tfiles = [str(d / "trades_x.parquet") for d in token_dirs if (d / "trades_x.parquet").exists()]
    bfiles = [str(d / "bba_x.parquet") for d in token_dirs if (d / "bba_x.parquet").exists()]
    if not tfiles:
        return {"group_id": group_id, "n_tokens": len(token_dirs)}
    t0, t1 = con.execute(
        "SELECT min(timestamp_ms), max(timestamp_ms) FROM read_parquet(?)", [tfiles]).fetchone()
    if t0 is None:
        return {"group_id": group_id, "n_tokens": len(token_dirs)}
    a, b = lead_in_span(int(t0), int(t1))

    n_tr, vol, avg_px = con.execute(
        "SELECT count(*), coalesce(sum(size),0), avg(price) FROM read_parquet(?) "
        "WHERE timestamp_ms BETWEEN ? AND ?", [tfiles, a, b]).fetchone()
    flow_rate = float(vol) / max((b - a) / 3.6e6, 1e-9)

    depth = spread_c = vol_c = turnover = sweep_share = float("nan")
    n_bba = 0
    if bfiles:
        row = con.execute(
            "SELECT median((coalesce(best_bid,0)+coalesce(best_ask,0))/2.0), "
            "median(best_ask - best_bid), count(*) FROM read_parquet(?) "
            "WHERE timestamp_ms BETWEEN ? AND ? AND best_bid IS NOT NULL AND best_ask IS NOT NULL",
            [bfiles, a, b]).fetchone()
        spread_c = float(row[1]) * CENTS if row[1] is not None else float("nan")
        n_bba = int(row[2])
        turnover = float(n_tr) / max(n_bba, 1)
        # 1-minute mid vol (lead-in only)
        v = con.execute(
            "WITH m AS (SELECT timestamp_ms//60000 AS minute, "
            " avg((best_bid+best_ask)/2.0) AS mid FROM read_parquet(?) "
            " WHERE timestamp_ms BETWEEN ? AND ? AND best_bid IS NOT NULL AND best_ask IS NOT NULL "
            " GROUP BY 1 ORDER BY 1), "
            "d AS (SELECT mid - lag(mid) OVER (ORDER BY minute) AS dm FROM m) "
            "SELECT stddev_samp(dm) FROM d",
            [bfiles, a, b]).fetchone()[0]
        vol_c = float(v) * CENTS if v is not None else float("nan")
        # sweep share: trade volume executing beyond the as-of touch / all trade volume.
        # depth: median top size proxy is unavailable in the bba slice (live schema ships
        # spread, not sizes) — use the book table when present.
        sw = con.execute(
            "WITH tr AS (SELECT timestamp_ms, price, size, side FROM read_parquet(?) "
            "            WHERE timestamp_ms BETWEEN ? AND ?), "
            "     bb AS (SELECT timestamp_ms, best_bid, best_ask FROM read_parquet(?) "
            "            WHERE best_bid IS NOT NULL AND best_ask IS NOT NULL) "
            "SELECT coalesce(sum(CASE WHEN (t.side='BUY' AND t.price > b.best_ask + 1e-9) "
            "                        OR (t.side='SELL' AND t.price < b.best_bid - 1e-9) "
            "                   THEN t.size END), 0), coalesce(sum(t.size), 0) "
            "FROM tr t ASOF JOIN bb b ON t.timestamp_ms >= b.timestamp_ms",
            [tfiles, a, b, bfiles]).fetchone()
        sweep_share = float(sw[0]) / max(float(sw[1]), 1e-9)
    bookf = [str(d / "book_x.parquet") for d in token_dirs if (d / "book_x.parquet").exists()]
    if bookf:
        try:
            d = con.execute(
                "SELECT median(coalesce(json_extract(bids,'$[0][1]')::DOUBLE,0)"
                " + coalesce(json_extract(asks,'$[0][1]')::DOUBLE,0)) "
                "FROM read_parquet(?) WHERE timestamp_ms BETWEEN ? AND ?",
                [bookf, a, b]).fetchone()[0]
            depth = float(d) / 2.0 if d is not None else float("nan")
        except Exception:
            depth = float("nan")

    return {"group_id": group_id, "n_tokens": len(token_dirs), "lead_in_a": a, "lead_in_b": b,
            "t_first": int(t0), "t_last": int(t1),
            "flow_rate": flow_rate, "depth": depth, "sweep_share": sweep_share,
            "spread_c": spread_c, "vol_c": vol_c, "avg_price": float(avg_px or 0.0),
            "turnover": turnover, "n_trades_leadin": int(n_tr)}


def assign_cohorts(feat: pd.DataFrame) -> pd.DataFrame:
    """Two pre-registered cohort axes per category, median-split on lead-in features.

    * ``cohort_aggr``  — 'aggressive' | 'benign' by the category-median ``sweep_share``
      (participant aggressiveness — the primary surface axis, Justin-confirmed).
    * ``cohort_liq``   — 'thick' | 'thin' by the category-median ``flow_rate``
      (liquidity/flow — the secondary axis; used with cohort_aggr to balance folds).
    """
    out = feat.copy()
    med_s = out["sweep_share"].median()
    med_f = out["flow_rate"].median()
    out["cohort_aggr"] = np.where(out["sweep_share"].fillna(0) > med_s, "aggressive", "benign")
    out["cohort_liq"] = np.where(out["flow_rate"].fillna(0) > med_f, "thick", "thin")
    return out


def assign_folds(feat: pd.DataFrame, n_folds: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic cohort-balanced fold assignment (stratified snake by start time).

    Groups are ordered inside each (cohort_aggr × cohort_liq) stratum by lifecycle start
    and dealt to folds snake-wise, so every fold gets a comparable cohort mix and a
    spread of start times. Returns ``feat`` + ``fold`` + ``order_idx`` (the group's index
    in the global start-time ordering — the axis ``purge_groups`` purges along).
    """
    out = feat.sort_values("t_first").reset_index(drop=True)
    out["order_idx"] = np.arange(len(out))
    out["fold"] = -1
    counter = 0
    for _, stratum in out.groupby(["cohort_aggr", "cohort_liq"], sort=True):
        for j, gi in enumerate(stratum.index):
            k = (counter + j) % (2 * n_folds)
            fold = k if k < n_folds else 2 * n_folds - 1 - k
            out.loc[gi, "fold"] = fold
        counter += len(stratum)
    return out


def fold_overlap_diagnostic(feat: pd.DataFrame, splits: list[dict],
                            order_to_group: dict[int, str]) -> pd.DataFrame:
    """Per split: calendar-time overlap between train and test group lifecycles.

    Whole-market CPCV cannot time-purge concurrent groups (politics markets trade
    simultaneously); this reports the mean fraction of each test group's lifespan that
    overlaps ≥1 training group — the honest residual-leakage channel (shared macro news),
    stated per brain/CODEX.md realism rule 3.
    """
    spans = {r.group_id: (r.t_first, r.t_last) for r in feat.itertuples()}
    rows = []
    for sp in splits:
        test = [order_to_group[g] for f in sp["test_groups_by_fold"].values() for g in f]
        train = [order_to_group[g] for g in sp["train_groups"]]
        fracs = []
        for tg in test:
            a0, a1 = spans[tg]
            cov = 0.0
            for tr in train:
                b0, b1 = spans[tr]
                cov = max(cov, max(0, min(a1, b1) - max(a0, b0)) / max(a1 - a0, 1))
            fracs.append(cov)
        rows.append({"split_id": sp["split_id"], "mean_test_overlap": float(np.mean(fracs)),
                     "n_test": len(test), "n_train": len(train)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Whole-lifecycle + τ-regime costed economics (arbitrary windows)
# ──────────────────────────────────────────────────────────────────────────────

# τ-regime edges (hours to resolution) — CONDITIONING dimension, pre-registered.
TAU_REGIMES = {
    "politics_negrisk": (("endgame", 0.0, 6.0), ("approach", 6.0, 48.0),
                         ("midlife", 48.0, float("inf"))),
    "esports": (("inplay", 0.0, 6.0), ("pre", 6.0, float("inf"))),
}


@dataclass(frozen=True)
class SpanCosted:
    """Costed economics of one run inside one arbitrary time window."""

    name: str
    n_fills: int
    filled_qty: float
    realized_usd: float
    carry_usd: float
    costed_usd: float
    costed_per_contract_c: float
    end_inventory: float


def costed_spans(fills: list[dict], quotes: list[dict],
                 spans: dict[str, tuple[int, int]]) -> dict[str, SpanCosted]:
    """Generalized windowed costed PnL: realized inside the span + Δ liquidation mark.

    Same accounting as ``protocol.windowed_costed`` (longs marked to best bid, shorts to
    best ask — the executable touch), for ANY set of windows: the full lifecycle and the
    τ-regime slices both come through here so the surface and the gate share one number.
    """
    fills = sorted(fills, key=lambda f: int(f["ts_exchange"]))
    ts, bb, ba = _touch_series(quotes)
    out: dict[str, SpanCosted] = {}
    for name, (a, b) in spans.items():
        sel = [f for f in fills if a <= int(f["ts_exchange"]) < b]
        realized = sum(float(f.get("realized_delta", 0.0)) for f in sel)
        qty = sum(float(f["qty"]) for f in sel)
        mark_a = _liq_mark(fills, ts, bb, ba, a)
        mark_b = _liq_mark(fills, ts, bb, ba, b)
        costed = realized + (mark_b - mark_a)
        end_inv = float(sel[-1]["position_after"]) if sel else 0.0
        out[name] = SpanCosted(
            name=name, n_fills=len(sel), filled_qty=qty, realized_usd=realized,
            carry_usd=mark_b - mark_a, costed_usd=costed,
            costed_per_contract_c=(costed / qty * CENTS) if qty > 0 else float("nan"),
            end_inventory=end_inv)
    return out


def regime_spans(end_ms: float, span: tuple[int, int], universe: str) -> dict[str, tuple[int, int]]:
    """τ-regime windows for one token: contiguous absolute-time slices from the τ edges."""
    t0, t1 = span
    out = {"full": (t0, t1)}
    if end_ms is None or not np.isfinite(end_ms):
        return out
    for name, lo_h, hi_h in TAU_REGIMES[universe]:
        a = int(end_ms - hi_h * 3.6e6) if np.isfinite(hi_h) else t0
        b = int(end_ms - lo_h * 3.6e6)
        a, b = max(a, t0), min(b, t1)
        if b > a:
            out[f"tau_{name}"] = (a, b)
    return out


def daily_series(fills: list[dict], quotes: list[dict], span: tuple[int, int]) -> np.ndarray:
    """Per-UTC-day costed PnL ($) over the WHOLE span (the audit's return unit)."""
    a, b = span
    fills = sorted(fills, key=lambda f: int(f["ts_exchange"]))
    ts, bb, ba = _touch_series(quotes)
    day_ms = 86_400_000
    out = []
    for d in range(int(a // day_ms), int((b - 1) // day_ms) + 1):
        lo, hi = max(a, d * day_ms), min(b, (d + 1) * day_ms)
        realized = sum(float(f.get("realized_delta", 0.0)) for f in fills
                       if lo <= int(f["ts_exchange"]) < hi)
        carry = _liq_mark(fills, ts, bb, ba, hi) - _liq_mark(fills, ts, bb, ba, lo)
        out.append(realized + carry)
    return np.asarray(out, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Nested selection / estimation over the (config × group) matrix
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NestedResult:
    """Honest outer-CV estimates for one rung (a set of candidate configs)."""

    rung: str
    per_split: pd.DataFrame        # split_id, selected_config, test pooled metric
    per_group: pd.DataFrame        # group_id, honest metric (mean over splits testing it)
    selection_counts: dict         # config -> #splits that inner-selected it
    modal_config: str              # most-frequently selected config (for bracketing/live)


def _pooled(matrix_usd: pd.DataFrame, matrix_qty: pd.DataFrame, cfg: str,
            groups: list[str]) -> float:
    """Volume-pooled ¢/contract of one config over a set of groups (Σ$ / Σqty)."""
    usd = matrix_usd.loc[groups, cfg].sum()
    qty = matrix_qty.loc[groups, cfg].sum()
    return float(usd / qty * CENTS) if qty > 0 else float("nan")


def nested_outer_estimates(matrix_usd: pd.DataFrame, matrix_qty: pd.DataFrame,
                           configs: list[str], splits: list[dict],
                           order_to_group: dict[int, str], rung: str) -> NestedResult:
    """Inner-select on outer-training groups, estimate once on held-out groups.

    ``matrix_usd`` / ``matrix_qty``: index = group_id, columns = configs (per-group costed
    $ and filled qty, pessimistic queue). Inner CV = leave-one-training-FOLD-out is
    implicit here: with per-group metrics precomputed, the inner argmax over training
    groups is the exact leave-fold-out selection limit — the selection never sees any
    held-out group. Scoring: volume-pooled ¢/contract on the held-out groups.
    """
    per_split_rows, sel_counts = [], {}
    per_group_vals: dict[str, list[float]] = {}
    per_group_cfg: dict[str, list[str]] = {}
    for sp in splits:
        train = [order_to_group[g] for g in sp["train_groups"]]
        test = [order_to_group[g] for f in sp["test_groups_by_fold"].values() for g in f]
        train = [g for g in train if g in matrix_usd.index]
        test = [g for g in test if g in matrix_usd.index]
        if not train or not test:
            continue
        scores = {c: _pooled(matrix_usd, matrix_qty, c, train) for c in configs}
        finite = {c: s for c, s in scores.items() if np.isfinite(s)}
        if not finite:
            continue
        sel = max(finite, key=finite.get)
        sel_counts[sel] = sel_counts.get(sel, 0) + 1
        oos = _pooled(matrix_usd, matrix_qty, sel, test)
        per_split_rows.append({"split_id": sp["split_id"], "selected": sel,
                               "train_pooled_c": finite[sel], "test_pooled_c": oos,
                               "n_train": len(train), "n_test": len(test)})
        for g in test:
            q = matrix_qty.loc[g, sel]
            v = (matrix_usd.loc[g, sel] / q * CENTS) if q > 0 else np.nan
            per_group_vals.setdefault(g, []).append(float(v))
            per_group_cfg.setdefault(g, []).append(sel)
    per_group = pd.DataFrame(
        [{"group_id": g, "honest_c": float(np.nanmean(v)) if len(v) else np.nan,
          "n_splits": len(v),
          "honest_usd": float(np.nanmean([matrix_usd.loc[g, c] for c in per_group_cfg[g]])),
          "honest_qty": float(np.nanmean([matrix_qty.loc[g, c] for c in per_group_cfg[g]]))}
         for g, v in per_group_vals.items()])
    modal = max(sel_counts, key=sel_counts.get) if sel_counts else ""
    return NestedResult(rung=rung, per_split=pd.DataFrame(per_split_rows),
                        per_group=per_group, selection_counts=sel_counts,
                        modal_config=modal)


def path_estimates(nested: NestedResult, splits: list[dict], paths: list[dict],
                   order_to_group: dict[int, str]) -> pd.DataFrame:
    """Pooled honest metric per complete OOS path (the CPCV path distribution)."""
    split_rows = {int(r.split_id): r for r in nested.per_split.itertuples()}
    rows = []
    for p in paths:
        vals = []
        ok = True
        for _fold, sid in p["split_assignments"]:
            r = split_rows.get(sid)
            if r is None:
                ok = False
                break
            vals.append(r.test_pooled_c)
        if ok and vals:
            rows.append({"path_id": p["path_id"],
                         "path_mean_c": float(np.nanmean(vals)),
                         "n_splits": len(vals)})
    return pd.DataFrame(rows)


def group_cluster_delta(per_group_a: pd.DataFrame, per_group_b: pd.DataFrame,
                        n_boot: int = 4000, seed: int = 0):
    """Keep-gate: paired per-group honest delta (A − B), group-bootstrap CI.

    Under whole-market CPCV the group IS the independent resample unit AND the
    observation unit, so this is a plain (seeded) bootstrap over paired group deltas.
    Returns a ``protocol.GroupDelta``-compatible object via that module's dataclass.
    """
    from mm_eval.protocol import GroupDelta
    m = per_group_a[["group_id", "honest_c"]].merge(
        per_group_b[["group_id", "honest_c"]], on="group_id", suffixes=("_a", "_b")).dropna()
    if m.empty:
        return GroupDelta("", float("nan"), float("nan"), float("nan"), 0, 0, False)
    d = (m["honest_c_a"] - m["honest_c_b"]).to_numpy(dtype=float)
    point = float(np.mean(d))
    if d.size < 2:
        return GroupDelta("", point, float("nan"), float("nan"), int(d.size), int(d.size), False)
    rng = np.random.default_rng(seed)
    draws = np.array([np.mean(rng.choice(d, d.size, replace=True)) for _ in range(n_boot)])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return GroupDelta("", point, float(lo), float(hi), int(d.size), int(d.size),
                      bool(np.isfinite(lo) and lo > 0))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Overfitting audit wiring (shared infrastructure/validation/overfitting_audit)
# ──────────────────────────────────────────────────────────────────────────────

def group_returns_matrix(df: pd.DataFrame, value_col: str = "full_costed_c",
                         group_col: str = "group_id", cfg_col: str = "config",
                         order: list[str] | None = None) -> pd.DataFrame:
    """(groups × configs) per-group return matrix — CSCV blocks = event groups."""
    mat = df.pivot_table(index=group_col, columns=cfg_col, values=value_col, aggfunc="mean")
    if order is not None:
        mat = mat.reindex([g for g in order if g in mat.index])
    return mat


def pbo_over_groups(mat: pd.DataFrame):
    """Real CSCV PBO (Bailey–Borwein–LdP–Zhu) with event-groups as the blocks.

    Rows = groups (ordered deterministically), one row per group; ``n_blocks`` chosen so
    each CSCV block is a small run of whole groups (never a within-group cut). Small-K
    honesty: at K groups the combination count is limited; the result is directional.
    """
    oa = _load_audit()
    m = mat.to_numpy(dtype=float)
    K = m.shape[0]
    if K < 4 or m.shape[1] < 2:
        return None
    n_blocks = min(10, K // 2 * 2)   # even, ≤10, and ≤ K/2 blocks of ≥2 rows... see below
    # pbo_cscv requires rows ≥ 2×n_blocks; blocks are contiguous row-runs of groups.
    n_blocks = max(2, min(n_blocks, K // 2))
    if n_blocks % 2:
        n_blocks -= 1
    if n_blocks < 2:
        return None
    return oa.pbo_cscv(np.nan_to_num(m, nan=0.0), n_blocks=n_blocks,
                       sensitivity_blocks=tuple(b for b in (4, 6, 8) if 2 <= b <= K // 2))


def dsr_outer(selected_daily: np.ndarray, trial_daily: dict[str, np.ndarray]):
    """DSR of the shipped config's daily costed PnL, deflated by EFFECTIVE trials."""
    oa = _load_audit()
    r = np.asarray(selected_daily, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5 or np.allclose(r.std(), 0):
        return {"dsr_p": float("nan"), "note": "underpowered (<5 daily obs or zero var)",
                "n_days": int(r.size)}
    L = max(len(v) for v in trial_daily.values())
    cols = {k: np.pad(np.nan_to_num(np.asarray(v, float), nan=0.0), (0, L - len(v)))
            for k, v in trial_daily.items()}
    tmat = np.column_stack(list(cols.values()))
    n_eff = oa.effective_n_trials(tmat)
    trial_srs = [float(c.mean() / c.std(ddof=1)) for c in tmat.T
                 if c.std(ddof=1) > 0 and len(c) >= 3]
    if len(trial_srs) < 2:
        return {"dsr_p": float("nan"), "note": "fewer than 2 finite trial Sharpes",
                "n_days": int(r.size)}
    res = oa.deflated_sharpe_ratio(r, n_trials=len(trial_daily), periods_per_year=365.0,
                                   trial_sharpes=trial_srs, n_eff=n_eff)
    return {"dsr_p": float(res.dsr_prob), "sharpe_ann": float(res.sr_ann),
            "sr_star_ann": float(res.sr_star_ann), "n_days": int(r.size),
            "n_trials": len(trial_daily), "n_eff": float(n_eff),
            "note": f"{r.size} daily obs; deflated by n_eff={n_eff:.1f} of {len(trial_daily)} trials"}


def whites_rc_daily(trial_daily: dict[str, np.ndarray], seed: int = 0):
    """White's Reality Check p-value over the config set's daily costed PnL."""
    oa = _load_audit()
    L = max(len(v) for v in trial_daily.values())
    cols = [np.pad(np.nan_to_num(np.asarray(v, float), nan=0.0), (0, L - len(v)))
            for v in trial_daily.values()]
    m = np.column_stack(cols)
    if m.shape[0] < 8:
        return {"p_value": float("nan"), "note": "too few daily obs for the bootstrap"}
    r = oa.whites_reality_check(m, n_boot=2000, seed=seed)
    return {"p_value": float(r.p_value), "p_value_raw": float(r.p_value_raw),
            "stat_obs": float(r.stat_obs), "n_boot": r.n_boot,
            "note": "H0: best config's true mean daily PnL <= 0 (data-snooping-adjusted)"}
