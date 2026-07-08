"""Task-5.1 gated ladder under whole-market nested CPCV — baseline → NSQ ablations → A-S → basket.

The redesigned Task-5 comparison arc, on the FULL R2 sample (17.9 d), with both fixes live:

* **Split = whole market (event group), nested CPCV** (`mm_eval.cpcv`): every config runs
  once per (token × queue); all selection/estimation is post-hoc arithmetic over the
  per-group matrices. For each of the C(6,2)=15 CPCV splits, knobs are inner-selected on
  the training groups only and the selected config is scored once on the held-out groups —
  the per-group "honest" series that every gate below uses. τ conditions (regime slices
  reported), never splits.
* **Controller = NeutralSpikeQuoter** (no calendar flatten): core (microprice skew + cap)
  → + two-lens toxicity gate (VPIN volume-clock + sweep weighting / AS z-score) →
  + asymmetric repricing → + OFI size-dampening — each rung ablates one component.
  The defensive rungs (lenses, damp) enter a category's GATED ladder only where the v0
  wiring rule (`mm_task5_1_v0_wiring.json`) found net-negative toxicity; elsewhere they
  run as UNSUPPORTED diagnostics (reported, never shipped).
* **Keep rule:** a rung is KEPT iff its per-group honest OOS delta vs the previous kept
  rung has bootstrap lower CI > 0 under the pessimistic queue (group = independent unit).
* **Audit:** real CSCV PBO (blocks = event groups), DSR deflated by effective trials,
  White's Reality Check — all via the shared `infrastructure/validation/overfitting_audit`.
* **Brackets:** each rung's modal (most-inner-selected) config re-runs under
  {Optimistic, Prob(0.5), RiskAverse}; every reported number is bracketed.

Usage (from polymarket/research/):
    PYTHONPATH=. uv run python scripts/mm_task5_1_ladder_run.py [--workers 7] [--quick]

Runs are cached per (token, config, queue) hash under the scratchpad → crash-resumable.
Deterministic: seeded bootstraps, deterministic engine, event-stream-only strategy state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mm_engine import (BACKTEST, ASQuoter, BasketCarryQuoter, ConstantLatency, FeeModel,
                       NeutralSpikeQuoter, OptimisticQueue, ProbQueue, RiskAverseQueue,
                       SymmetricQuoter, Telemetry, run_engine)
from mm_engine.feeds.replay_parquet import replay_parquet
from mm_engine.telemetry import JsonlSink

from mm_eval import cpcv
from mm_eval.metrics import CENTS
from mm_eval.tape import TradeTape, tape_feed

RESEARCH = Path(__file__).resolve().parents[1]
CSV_OUT = RESEARCH / "data/analysis/csv_outputs/market_making"
PLOT_OUT = RESEARCH / "data/analysis/plots/market_making"
SELECTION_JSON = RESEARCH / "data/markets/mm_task5_1_selection.json"
GROUPS_PARQUET = CSV_OUT / "mm_task5_1_groups.parquet"
WIRING_JSON = CSV_OUT / "mm_task5_1_v0_wiring.json"
SCRATCH = Path("/private/tmp/claude-501/-Users-justiniturregui-Desktop-github-epsilon-quant-research/"
               "b6ea1a3f-cca4-465b-b11c-5ab18e4a749c/scratchpad")
CACHE = SCRATCH / "mm_task5_1_cache"
RUNS = CACHE / "runs"
UNIVERSES = ("politics_negrisk", "esports")

QUEUES = {"Optimistic": OptimisticQueue, "Prob(0.5)": lambda: ProbQueue(0.5),
          "RiskAverse": RiskAverseQueue}
PESS = "RiskAverse"
STRATS = {"symmetric": SymmetricQuoter, "nsq": NeutralSpikeQuoter, "as": ASQuoter,
          "basket": BasketCarryQuoter}

# ── pre-registered grids (declared before any result was seen) ────────────────
K_GRID = (5e-6, 2e-5, 8e-5)
CAP_GRID = (200.0, 500.0)
VPIN_W_GRID = (20, 50)
DAMP_GRID = (0.3, 0.6)
AS_GAMMA_GRID = (1e-5, 1e-4, 1e-3)
K_TEST = 2
# Replay-feasibility cap (declared ex-ante, data-volume criterion — NOT a performance
# screen): tokens whose per-token L2 event count exceeds this are excluded from the eval
# universe (a 23M-event token needs ~10 GB RAM and ~10 CPU-hours per config sweep). The
# exclusion is printed and carried into the findings' assumption ledger; a group keeps its
# remaining tokens, so the split-unit count is unaffected.
MAX_TOKEN_EVENTS = 8_000_000
# purge_groups=0 BY DESIGN: labels (30 s markouts, per-group costed PnL) never span group
# boundaries (each group = separate engine runs), so the purge lives at the markout level;
# whole-market groups are calendar-concurrent, making an ordering purge symbolic. The
# residual concurrency channel is reported via the overlap diagnostic CSV instead.
PURGE_GROUPS = 0


def cfg_name(rung: str, **kw) -> str:
    parts = ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in sorted(kw.items()))
    return f"{rung}[{parts}]" if parts else rung


# The rung ladder: (rung, kind, [ (config_name, overrides) ]). Defensive rungs are gated
# by the v0 wiring rule per category (else diagnostics-only).
def build_rungs(wire_defensive: bool) -> list[dict]:
    core = [(cfg_name("core", k=k, cap=c), {"skew_k": k, "inv_cap": c})
            for k in K_GRID for c in CAP_GRID]
    lens = [(cfg_name("lens", k=k, cap=c, w=w),
             {"skew_k": k, "inv_cap": c, "nsq_lens1": True, "nsq_lens2": True,
              "nsq_vpin_window": w})
            for k in K_GRID for c in CAP_GRID for w in VPIN_W_GRID]
    asym = [(nm.replace("lens[", "asym["), {**ov, "nsq_asym": True}) for nm, ov in lens]
    # damp rung fixes the VPIN window at 20 (its knob is the damp coefficient; carrying the
    # w dimension again would double the rung's trial count for a knob already selected)
    damp = [(nm.replace("asym[", "damp[")[:-1] + f",d={d:g}]", {**ov, "nsq_damp_coeff": d})
            for nm, ov in asym if ov["nsq_vpin_window"] == 20 for d in DAMP_GRID]
    rung1 = [(cfg_name("rung1", g=g, cap=c),
              {"as_gamma": g, "inv_cap": c, "pull_hours": 0.0})
             for g in AS_GAMMA_GRID for c in CAP_GRID]
    rung2 = [(nm.replace("rung1[", "rung2["), {**ov, "as_use_spread": True})
             for nm, ov in rung1]
    rung3 = [(nm.replace("rung2[", "rung3["), {**ov, "tox_velocity": True})
             for nm, ov in rung2]
    basket = [(cfg_name("basket", k=k, cap=c), {"skew_k": k, "inv_cap": c})
              for k in K_GRID for c in CAP_GRID]
    rungs = [
        {"rung": "baseline", "kind": "symmetric", "configs": [("baseline", {})],
         "gated": True},
        {"rung": "nsq_core", "kind": "nsq", "configs": core, "gated": True},
        {"rung": "nsq_lens", "kind": "nsq", "configs": lens, "gated": wire_defensive},
        {"rung": "nsq_asym", "kind": "nsq", "configs": asym, "gated": wire_defensive},
        {"rung": "nsq_damp", "kind": "nsq", "configs": damp, "gated": wire_defensive},
        {"rung": "as_rung1", "kind": "as", "configs": rung1, "gated": True},
        {"rung": "as_rung2", "kind": "as", "configs": rung2, "gated": True},
        {"rung": "as_rung3", "kind": "as", "configs": rung3, "gated": wire_defensive},
        {"rung": "basket", "kind": "basket", "configs": basket, "gated": True},
    ]
    return rungs


# ──────────────────────────────────────────────────────────────────────────────
# Job runner (cached per token×config×queue; per-token event parse shared)
# ──────────────────────────────────────────────────────────────────────────────

def _hash(run_id: str, config: str, kind: str, overrides: str, queue: str) -> str:
    return hashlib.md5(f"{run_id}|{config}|{kind}|{overrides}|{queue}".encode()).hexdigest()[:16]


class SlimQuotesSink(JsonlSink):
    KEEP = ("ts_exchange", "token_id", "best_bid", "best_ask")

    def emit(self, rec: dict) -> None:  # noqa: D102
        if self.keep:
            self.records.append({k: rec.get(k) for k in self.KEEP})


def _read_cache(cache_f: Path) -> list[dict] | None:
    """Tolerant cache read: a torn/partial file (concurrent writer) → recompute, not crash."""
    if not cache_f.exists():
        return None
    try:
        return json.loads(cache_f.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(cache_f: Path, rows: list[dict]) -> None:
    """Atomic write (temp + rename) so a concurrent reader never sees a partial file."""
    tmp = cache_f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rows))
    tmp.replace(cache_f)


def run_cell(job: dict, events: list | None = None) -> list[dict]:
    """One (token|group) × config × queue cell → per-token rows with regime-sliced costed PnL."""
    cache_f = RUNS / f"{job['hash']}.json"
    cached = _read_cache(cache_f)
    if cached is not None:
        return cached
    tokens = job["tokens"]
    overrides = json.loads(job["overrides_json"])
    tape = TradeTape()
    params = {"size": 100.0, "tick": 0.001, "trade_tape": tape, **overrides}
    if job["kind"] == "basket":
        params["half_spread_by_token"] = {t["token_id"]: t["half_spread"] for t in tokens}
        params["half_spread"] = float(np.median([t["half_spread"] for t in tokens]))
    else:
        params["half_spread"] = tokens[0]["half_spread"]
        if tokens[0]["end_ms"] is not None and np.isfinite(tokens[0]["end_ms"]):
            params.setdefault("end_date_ms", tokens[0]["end_ms"])
    tele = Telemetry(fills=JsonlSink(keep=True), orders=JsonlSink(keep=False),
                     quotes=SlimQuotesSink(keep=True))
    feed = iter(events) if events is not None else replay_parquet(Path(job["replay_dir"]), gaps=[])
    result = run_engine(tape_feed(feed, tape), strategy=STRATS[job["kind"]](),
                        queue_model=QUEUES[job["queue"]](),
                        latency_model=ConstantLatency(0.0), mode=BACKTEST, params=params,
                        fee_model=FeeModel(), telemetry=tele)
    rows = []
    for t in tokens:
        tok = t["token_id"]
        fills = [f for f in result.fills if f["token_id"] == tok]
        quotes = (tele.quotes.records if len(tokens) == 1
                  else [q for q in tele.quotes.records if q.get("token_id") == tok])
        spans = cpcv.regime_spans(t["end_ms"] if t["end_ms"] is not None else float("nan"),
                                  tuple(job["span"]), job["universe"])
        sc = cpcv.costed_spans(fills, quotes, spans)
        daily = cpcv.daily_series(fills, quotes, tuple(job["span"]))
        max_abs_inv = max((abs(float(f["position_after"])) for f in fills), default=0.0)
        row = {"run_id": job["run_id"], "token_id": tok, "universe": job["universe"],
               "group_id": t["group_id"], "config": job["config"], "rung": job["rung"],
               "kind": job["kind"], "queue": job["queue"],
               "daily": list(map(float, daily)), "max_abs_inv": max_abs_inv,
               "end_inv": sc["full"].end_inventory if "full" in sc else 0.0}
        for name, s in sc.items():
            row[f"{name}_usd"] = s.costed_usd
            row[f"{name}_qty"] = s.filled_qty
            row[f"{name}_c"] = s.costed_per_contract_c
            row[f"{name}_fills"] = s.n_fills
        rows.append(row)
    RUNS.mkdir(parents=True, exist_ok=True)
    _write_cache(cache_f, rows)
    return rows


def run_token_batch(batch: list[dict]) -> list[dict]:
    """Worker: all of one (token|group)'s cells against a single event parse."""
    out, todo = [], []
    for j in batch:
        cached = _read_cache(RUNS / f"{j['hash']}.json")
        if cached is not None:
            out.extend(cached)
        else:
            todo.append(j)
    if todo:
        events = list(replay_parquet(Path(todo[0]["replay_dir"]), gaps=[]))
        for j in todo:
            out.extend(run_cell(j, events))
    return out


def run_all(jobs: list[dict], workers: int) -> list[dict]:
    if not jobs:
        return []
    by_dir: dict[str, list[dict]] = {}
    for j in jobs:
        by_dir.setdefault(j["replay_dir"], []).append(j)
    batches = sorted(by_dir.values(), key=len, reverse=True)
    out = []
    # max_tasks_per_child=1: a worker exits after each token batch, returning its heap to
    # the OS. Without this the pool's long-lived workers RATCHET memory (Python keeps the
    # freed event-list pages), which drove the machine into the compressor (observed: 17 GB
    # compressed, 50% sys CPU, workers stalled at ~20%). Fork cost is trivial per batch.
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_token_batch, b): b for b in batches}
        for i, fut in enumerate(as_completed(futs), 1):
            b = futs[fut]
            try:
                out.extend(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"  !! batch {b[0]['run_id'][:14]} ({len(b)} cells): {e}", flush=True)
            print(f"  … {i}/{len(batches)} batches ({b[0]['run_id'][:12]}, {len(b)} cells)",
                  flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Setup / job construction
# ──────────────────────────────────────────────────────────────────────────────

def token_payload(t: dict) -> dict:
    end_ms = None
    if t.get("end_date"):
        end_ms = datetime.fromisoformat(t["end_date"].replace("Z", "+00:00")).timestamp() * 1000.0
    return {"token_id": t["token_id"], "half_spread": t["half_spread"],
            "end_ms": end_ms, "group_id": t["group_id"], "market": t["market"]}


def estimate_k_arr_leadin(universe: str, tok: str, lead_a: int, lead_b: int) -> float:
    """A-S rung-2 arrival decay 1/mean|trade − mid|, from the LEAD-IN window only.

    Leakage-safe by construction (the same pre-registered window the cohort features use)
    — Task-5 estimated this on the calendar-IS window, which no longer exists.
    """
    import duckdb
    tdir = CACHE / universe / tok
    con = duckdb.connect()
    try:
        tr = con.execute(
            f"SELECT timestamp_ms, price FROM read_parquet('{tdir}/trades_x.parquet') "
            f"WHERE timestamp_ms BETWEEN ? AND ? ORDER BY timestamp_ms",
            [lead_a, lead_b]).df()
        bb = con.execute(
            f"SELECT timestamp_ms, best_bid, best_ask FROM read_parquet('{tdir}/bba_x.parquet') "
            f"WHERE timestamp_ms BETWEEN ? AND ? AND best_bid IS NOT NULL "
            f"AND best_ask IS NOT NULL ORDER BY timestamp_ms", [lead_a, lead_b]).df()
    finally:
        con.close()
    if len(tr) < 30 or len(bb) < 30:
        return 50.0
    m = pd.merge_asof(tr, bb, on="timestamp_ms", direction="backward").dropna()
    if m.empty:
        return 50.0
    mean_d = float((m["price"] - (m["best_bid"] + m["best_ask"]) / 2).abs().mean())
    if not np.isfinite(mean_d) or mean_d <= 0:
        return 50.0
    return float(np.clip(1.0 / mean_d, 5.0, 500.0))


def jobs_for(universe: str, tokens: list[dict], span, rung: str, config: str, kind: str,
             overrides: dict, queues: list[str], groups_map: dict | None = None,
             k_arr_by_token: dict | None = None) -> list[dict]:
    ov_json = json.dumps(overrides, sort_keys=True)
    out = []
    if kind == "basket":
        by_group: dict[str, list[dict]] = {}
        for t in tokens:
            by_group.setdefault(t["group_id"], []).append(t)
        for gid, toks in by_group.items():
            gdir = groups_map[gid]
            legs = {}
            for t in toks:
                cid = t["market"]
                siblings = [x for x in toks if x["market"] == cid]
                sign = 1 if siblings.index(t) == 0 else -1
                legs[t["token_id"]] = {"cond": cid, "sign": sign}
            ov = json.dumps({**overrides, "basket_legs": legs}, sort_keys=True)
            for q in queues:
                out.append({"run_id": f"group:{gid[:18]}", "universe": universe,
                            "replay_dir": str(gdir), "config": config, "rung": rung,
                            "kind": kind, "overrides_json": ov, "queue": q,
                            "span": list(span),
                            "tokens": [token_payload(t) for t in toks],
                            "hash": _hash(f"group:{gid[:18]}", config, kind, ov, q)})
        return out
    for t in tokens:
        oj = ov_json
        if kind == "as" and overrides.get("as_use_spread") and k_arr_by_token:
            oj = json.dumps({**overrides,
                             "as_k_arr": k_arr_by_token.get(t["token_id"], 50.0)},
                            sort_keys=True)
        for q in queues:
            out.append({"run_id": t["token_id"], "universe": universe,
                        "replay_dir": str(CACHE / universe / t["token_id"]),
                        "config": config, "rung": rung, "kind": kind,
                        "overrides_json": oj, "queue": q, "span": list(span),
                        "tokens": [token_payload(t)],
                        "hash": _hash(t["token_id"], config, kind, oj, q)})
    return out


def materialize_groups(universe: str, tokens: list[dict]) -> dict[str, Path]:
    import duckdb
    from mm_eval import markets as mk
    con = duckdb.connect()
    out = {}
    by_group: dict[str, list[str]] = {}
    for t in tokens:
        by_group.setdefault(t["group_id"], []).append(t["token_id"])
    for gid, toks in by_group.items():
        gdir = CACHE / "groups" / f"{universe}_{gid[:18]}"
        if not (gdir.exists() and all((gdir / f"{tb}_x.parquet").exists() for tb in mk.TABLES)):
            gdir.mkdir(parents=True, exist_ok=True)
            ph = ", ".join(["?"] * len(toks))
            for table, cols in mk.TABLE_COLS.items():
                cp = CACHE / f"_compact_{universe}_{table}.parquet"
                target = str(gdir / f"{table}_x.parquet").replace("'", "''")
                con.execute(
                    f"COPY (SELECT {cols} FROM read_parquet(?) WHERE asset_id IN ({ph}) "
                    f"ORDER BY timestamp_ms, received_ns) TO '{target}' (FORMAT parquet)",
                    [str(cp), *toks])
        out[gid] = gdir
    con.close()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation: nested CPCV per rung, gates, audit, surface
# ──────────────────────────────────────────────────────────────────────────────

def per_group_matrices(df: pd.DataFrame, configs: list[str],
                       value: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(groups × configs) costed-$ and qty matrices (pessimistic rows of ``df``)."""
    usd = df.pivot_table(index="group_id", columns="config", values=f"{value}_usd",
                         aggfunc="sum").reindex(columns=configs)
    qty = df.pivot_table(index="group_id", columns="config", values=f"{value}_qty",
                         aggfunc="sum").reindex(columns=configs)
    return usd.fillna(0.0), qty.fillna(0.0)


def aggregate_category(u: str, df: pd.DataFrame, rungs: list[dict], groups_df: pd.DataFrame,
                       seed: int = 7) -> dict:
    """Nested-CPCV honest estimates per rung → gated ladder + audit + surface for one category."""
    gsub = groups_df[groups_df.universe == u].sort_values(["fold", "order_idx"])
    n_groups = len(gsub)
    n_folds = int(gsub["n_folds"].iloc[0])
    # cohort-balanced folds are unequal-sized: pass the actual per-fold counts so the
    # generator's fold f == the assigned fold f (groups laid out fold-sorted).
    fold_sizes = [int((gsub["fold"] == f).sum()) for f in range(n_folds)]
    splits = cpcv.generate_group_cpcv_splits(n_groups, n_folds, K_TEST,
                                             purge_groups=PURGE_GROUPS,
                                             fold_sizes=fold_sizes)
    order_to_group = {i: g for i, g in enumerate(gsub["group_id"].tolist())}

    pess = df[(df.universe == u) & (df.queue == PESS)]
    out = {"universe": u, "rungs": {}, "ladder_rows": [], "trial_daily": {}, "splits": splits}
    out["overlap_diag"] = cpcv.fold_overlap_diagnostic(gsub, splits["splits"], order_to_group)

    prev_kept: cpcv.NestedResult | None = None
    prev_kept_rung = None
    for rung in rungs:
        rname = rung["rung"]
        cfgs = [c for c, _ in rung["configs"]]
        sub = pess[pess.config.isin(cfgs)]
        if sub.empty:
            continue
        usd, qty = per_group_matrices(sub, cfgs)
        nested = cpcv.nested_outer_estimates(usd, qty, cfgs, splits["splits"],
                                             order_to_group, rung=rname)
        paths = cpcv.path_estimates(nested, splits["splits"], splits["paths"], order_to_group)
        out["rungs"][rname] = {"nested": nested, "paths": paths}
        # trial daily series (every config is a trial for DSR/White's)
        for cfg in cfgs:
            rows = sub[sub.config == cfg]
            if rows.empty:
                continue
            arrs = [np.asarray(a, float) for a in rows.daily]
            L = max(len(a) for a in arrs)
            out["trial_daily"][cfg] = np.nansum(
                [np.pad(a, (0, L - len(a)), constant_values=0.0) for a in arrs], axis=0)

        row = {"universe": u, "rung": rname, "gated": rung["gated"],
               "modal_config": nested.modal_config,
               "n_configs": len(cfgs),
               "honest_mean_c": float(nested.per_group["honest_c"].mean()),
               "honest_pooled_c": (float(nested.per_group["honest_usd"].sum()
                                         / nested.per_group["honest_qty"].sum() * CENTS)
                                   if nested.per_group["honest_qty"].sum() > 0 else np.nan),
               "path_mean_c": float(paths["path_mean_c"].mean()) if len(paths) else np.nan,
               "path_p10_c": float(paths["path_mean_c"].quantile(0.1)) if len(paths) else np.nan,
               "path_p90_c": float(paths["path_mean_c"].quantile(0.9)) if len(paths) else np.nan,
               "n_groups": int(len(nested.per_group))}
        if prev_kept is None:
            row.update(delta_c=np.nan, delta_lo=np.nan, delta_hi=np.nan,
                       keep="KEPT (baseline)" if rung["gated"] else "DIAGNOSTIC")
            if rung["gated"]:
                prev_kept, prev_kept_rung = nested, rname
        else:
            d = cpcv.group_cluster_delta(nested.per_group, prev_kept.per_group, seed=seed)
            row.update(delta_c=d.point, delta_lo=d.lo, delta_hi=d.hi)
            if not rung["gated"]:
                row["keep"] = "DIAGNOSTIC (v0-unsupported — reported, not gated)"
            elif d.beats:
                row["keep"] = f"KEPT (beats {prev_kept_rung} on honest-OOS group CI)"
                prev_kept, prev_kept_rung = nested, rname
            elif d.improves:
                row["keep"] = "DROP — improves point, CI spans 0 (FRAGILE)"
            else:
                row["keep"] = "DROP — does not improve honest OOS"
        out["ladder_rows"].append(row)
    out["kept_rung"] = prev_kept_rung
    out["kept_nested"] = prev_kept
    return out


def surface_rows(u: str, df: pd.DataFrame, kept: cpcv.NestedResult, groups_df: pd.DataFrame,
                 rungs: list[dict]) -> list[dict]:
    """(cohort × τ-regime) surface of the kept rung's honest per-group configs."""
    gsub = groups_df[groups_df.universe == u]
    cohort = gsub.set_index("group_id")["cohort_aggr"]
    pess = df[(df.universe == u) & (df.queue == PESS)]
    # honest config per group = the modal inner-selected config across splits testing it
    rows = []
    regs = [f"tau_{n}" for n, *_ in cpcv.TAU_REGIMES[u]]
    for _, g in kept.per_group.iterrows():
        gid = g["group_id"]
        sub = pess[(pess.group_id == gid) & (pess.rung == kept.rung)]
        if sub.empty:
            continue
        # use the group's honest value for 'full'; regime slices from the modal config
        modal = kept.modal_config
        msub = sub[sub.config == modal]
        for reg in ["full"] + regs:
            usd = msub[f"{reg}_usd"].sum() if f"{reg}_usd" in msub else np.nan
            qty = msub[f"{reg}_qty"].sum() if f"{reg}_qty" in msub else np.nan
            rows.append({"universe": u, "group_id": gid,
                         "cohort_aggr": cohort.get(gid, "?"), "regime": reg,
                         "honest_c": g["honest_c"] if reg == "full" else np.nan,
                         "modal_usd": float(usd) if np.isfinite(usd) else np.nan,
                         "modal_qty": float(qty) if np.isfinite(qty) else np.nan})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="core+baseline rungs only")
    ap.add_argument("--prewarm", action="store_true",
                    help="run all pessimistic cells into the cache, skip aggregation "
                         "(the job set is wiring-independent; wiring only gates rungs)")
    ap.add_argument("--universe", choices=UNIVERSES, default=None,
                    help="prewarm only this universe (for overlapping a second process)")
    args = ap.parse_args()
    universes = (args.universe,) if args.universe else UNIVERSES
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    sel = json.loads(SELECTION_JSON.read_text())
    groups_df = pd.read_parquet(GROUPS_PARQUET)
    wiring = json.loads(WIRING_JSON.read_text()) if WIRING_JSON.exists() else None
    if wiring is None:
        if not args.prewarm:
            raise RuntimeError("v0 wiring rule missing — run mm_task5_1_v0_attribution.py first")
        wiring = {u: {"wire_defensive_knobs": True} for u in UNIVERSES}   # jobs identical

    # replay-feasibility cap (see MAX_TOKEN_EVENTS)
    import duckdb
    con = duckdb.connect()
    excluded: list[str] = []
    for u in UNIVERSES:
        kept = []
        for t in sel["universes"][u]["tokens"]:
            d = CACHE / u / t["token_id"]
            n = sum(con.execute(f"SELECT count(*) FROM read_parquet('{d}/{tb}_x.parquet')")
                    .fetchone()[0] for tb in ("book", "trades", "price_change", "bba"))
            if n > MAX_TOKEN_EVENTS:
                excluded.append(f"{u}:{t['token_id'][:14]}… ({n:,} events) {t.get('question')}")
            else:
                kept.append(t)
        sel["universes"][u]["tokens"] = kept
    con.close()
    if excluded:
        print(f"EXCLUDED by replay-feasibility cap (> {MAX_TOKEN_EVENTS:,} events):")
        for e in excluded:
            print(f"  {e}")

    all_rows: list[dict] = []
    rungs_by_u: dict[str, list[dict]] = {}
    for u in universes:
        tokens = sel["universes"][u]["tokens"]
        span = sel["universes"][u]["span"]
        wire = bool(wiring[u]["wire_defensive_knobs"])
        rungs = build_rungs(wire)
        if args.quick:
            rungs = [r for r in rungs if r["rung"] in ("baseline", "nsq_core")]
        rungs_by_u[u] = rungs
        print(f"[{u}] wire_defensive={wire}; rungs: "
              f"{[(r['rung'], len(r['configs']), 'gated' if r['gated'] else 'diag') for r in rungs]}")
        groups_map = materialize_groups(u, tokens)
        # leakage-safe A-S arrival calibration from each group's lead-in window
        lead = {r.group_id: (int(r.lead_in_a), int(r.lead_in_b))
                for r in groups_df[groups_df.universe == u].itertuples()
                if np.isfinite(r.lead_in_a)}
        k_arr = {}
        if any(r["rung"].startswith("as_rung2") or r["rung"].startswith("as_rung3")
               for r in rungs):
            for t in tokens:
                la, lb = lead.get(t["group_id"], (None, None))
                k_arr[t["token_id"]] = (estimate_k_arr_leadin(u, t["token_id"], la, lb)
                                        if la is not None else 50.0)
        jobs: list[dict] = []
        for r in rungs:
            for cfg_nm, ov in r["configs"]:
                jobs.append((r, cfg_nm, ov))
        # pessimistic grid for every config
        pess_jobs = []
        for r, cfg_nm, ov in jobs:
            pess_jobs += jobs_for(u, tokens, span, r["rung"], cfg_nm, r["kind"], ov,
                                  [PESS], groups_map, k_arr)
        # Memory guard: the politics tokens parse to ~3.5 GB of event objects per worker
        # (1.6–2.3M events each); 7 workers overrun 32 GB RAM and the compressor tax
        # (observed: 18 GB compressed, >50% sys CPU) costs more than the lost parallelism.
        u_workers = min(args.workers, 4) if u == "politics_negrisk" else args.workers
        print(f"[{u}] pessimistic cells: {len(pess_jobs)} (workers={u_workers})")
        all_rows += run_all(pess_jobs, u_workers)

    if args.prewarm:
        print(f"\nprewarm complete: {len(all_rows)} rows cached under {RUNS}")
        return

    df = pd.DataFrame(all_rows)
    df.to_json(SCRATCH / "mm_task5_1_all_rows.json", orient="records")

    # nested CPCV per category (pessimistic), gates, then bracket the modal configs
    results = {}
    ladder_rows = []
    for u in UNIVERSES:
        res = aggregate_category(u, df, rungs_by_u[u], groups_df)
        results[u] = res
        ladder_rows += res["ladder_rows"]
        print(f"\n[{u}] kept rung: {res['kept_rung']}")

    # bracket runs (Optimistic + Prob) for each rung's modal config
    bracket_rows: list[dict] = []
    for u in UNIVERSES:
        tokens = sel["universes"][u]["tokens"]
        span = sel["universes"][u]["span"]
        groups_map = materialize_groups(u, tokens)
        lead = {r.group_id: (int(r.lead_in_a), int(r.lead_in_b))
                for r in groups_df[groups_df.universe == u].itertuples()
                if np.isfinite(r.lead_in_a)}
        k_arr = {}
        for t in tokens:
            la, lb = lead.get(t["group_id"], (None, None))
            k_arr[t["token_id"]] = (estimate_k_arr_leadin(u, t["token_id"], la, lb)
                                    if la is not None else 50.0)
        bjobs = []
        for r in rungs_by_u[u]:
            rr = results[u]["rungs"].get(r["rung"])
            if rr is None or not rr["nested"].modal_config:
                continue
            modal = rr["nested"].modal_config
            ov = dict(next(o for c, o in r["configs"] if c == modal))
            bjobs += jobs_for(u, tokens, span, r["rung"], modal, r["kind"], ov,
                              ["Optimistic", "Prob(0.5)"], groups_map, k_arr)
        print(f"[{u}] bracket cells: {len(bjobs)}")
        bracket_rows += run_all(bjobs, args.workers)
    bdf = pd.DataFrame(bracket_rows) if bracket_rows else pd.DataFrame()
    full_df = pd.concat([df, bdf], ignore_index=True) if len(bdf) else df
    full_df.to_json(SCRATCH / "mm_task5_1_all_rows.json", orient="records")

    # ladder table with brackets
    lrows = []
    for row in ladder_rows:
        u, rname = row["universe"], row["rung"]
        modal = row["modal_config"]
        for q in ("Optimistic", "Prob(0.5)"):
            sub = full_df[(full_df.universe == u) & (full_df.config == modal)
                          & (full_df.queue == q)]
            qty = sub["full_qty"].sum()
            row[f"modal_pooled_c_{q}"] = (float(sub["full_usd"].sum() / qty * CENTS)
                                          if qty > 0 else np.nan)
        sub = full_df[(full_df.universe == u) & (full_df.config == modal)
                      & (full_df.queue == PESS)]
        qty = sub["full_qty"].sum()
        row["modal_pooled_c_RiskAverse"] = (float(sub["full_usd"].sum() / qty * CENTS)
                                            if qty > 0 else np.nan)
        lrows.append(row)
    ladder_df = pd.DataFrame(lrows)
    ladder_df.to_csv(CSV_OUT / "mm_task5_1_ladder_table.csv", index=False)

    # audit: PBO (groups), DSR (kept rung's honest daily), White's RC — per category
    audit_rows = []
    for u in UNIVERSES:
        res = results[u]
        pess = df[(df.universe == u) & (df.queue == PESS)].copy()
        pess["full_costed_c"] = pess["full_c"]
        mat = cpcv.group_returns_matrix(pess)
        pbo = cpcv.pbo_over_groups(mat)
        kept = res["kept_nested"]
        kept_cfg = kept.modal_config if kept is not None else "baseline"
        dsr = cpcv.dsr_outer(res["trial_daily"].get(kept_cfg, np.array([])),
                             res["trial_daily"])
        rc = cpcv.whites_rc_daily(res["trial_daily"])
        audit_rows.append({
            "universe": u, "kept_rung": res["kept_rung"], "kept_modal": kept_cfg,
            "pbo": float(pbo.pbo) if pbo else np.nan,
            "pbo_blocks": pbo.n_blocks if pbo else 0,
            "pbo_note": f"CSCV blocks=group-runs; sensitivity {pbo.sensitivity}" if pbo else "undefined",
            **{f"dsr_{k}": v for k, v in dsr.items()},
            **{f"rc_{k}": v for k, v in rc.items()},
        })
        res["overlap_diag"].to_parquet(CSV_OUT / f"mm_task5_1_overlap_{u}.parquet", index=False)
    pd.DataFrame(audit_rows).to_parquet(CSV_OUT / "mm_task5_1_audit.parquet", index=False)

    # per-split selection records + per-group honest estimates + paths (all rungs)
    sel_rows, grp_rows, path_rows = [], [], []
    for u in UNIVERSES:
        for rname, rr in results[u]["rungs"].items():
            n = rr["nested"]
            s = n.per_split.copy()
            s["universe"], s["rung"] = u, rname
            sel_rows.append(s)
            g = n.per_group.copy()
            g["universe"], g["rung"] = u, rname
            grp_rows.append(g)
            p = rr["paths"].copy()
            p["universe"], p["rung"] = u, rname
            path_rows.append(p)
    pd.concat(sel_rows, ignore_index=True).to_parquet(CSV_OUT / "mm_task5_1_splits.parquet", index=False)
    pd.concat(grp_rows, ignore_index=True).to_parquet(CSV_OUT / "mm_task5_1_groups_honest.parquet", index=False)
    pd.concat(path_rows, ignore_index=True).to_parquet(CSV_OUT / "mm_task5_1_paths.parquet", index=False)

    # (cohort × τ-regime) surface for the kept rung
    srows = []
    for u in UNIVERSES:
        kept = results[u]["kept_nested"]
        if kept is not None:
            srows += surface_rows(u, df, kept, groups_df, rungs_by_u[u])
    pd.DataFrame(srows).to_parquet(CSV_OUT / "mm_task5_1_surface.parquet", index=False)

    print("\n================ LADDER (nested-CPCV honest OOS) ================")
    show = [c for c in ("universe", "rung", "gated", "modal_config", "honest_pooled_c",
                        "honest_mean_c", "path_mean_c", "path_p10_c", "path_p90_c",
                        "modal_pooled_c_RiskAverse", "modal_pooled_c_Prob(0.5)",
                        "modal_pooled_c_Optimistic", "delta_c", "delta_lo", "delta_hi",
                        "keep") if c in ladder_df.columns]
    print(ladder_df[show].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\n================ AUDIT ================")
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print(f"\nCSVs -> {CSV_OUT}")


if __name__ == "__main__":
    main()
