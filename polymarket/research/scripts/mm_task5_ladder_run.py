"""Task-5 gated ladder — symmetric baseline → v1 InventoryAwareQuoter → A-S rungs → basket-carry.

Runs the full comparison arc of the Task-5 PRD on the ~11-day R2 capture, under the pinned
IS/OOS protocol (`mm_eval.protocol`):

* **Select on IS only** (2026-06-19 → 06-23, RiskAverse pessimistic queue), **report the OOS
  bracket** (06-24 → 06-30, {Optimistic, Prob(0.5), RiskAverse}), per category (politics vs
  esports — never pooled for the gate; cross-regime read = robustness only).
* **Costed PnL** (realized round-trips + liquidation-marked inventory carry at the
  executable touch) — never mark-to-mid.
* **Keep rule (pre-registered):** a later config is KEPT only if its paired per-token OOS
  costed-net delta vs the previous kept config has a group-cluster-bootstrap lower CI > 0
  under the pessimistic queue. Point-improves-but-CI-spans-zero → FRAGILE/underpowered
  (reported, not kept). Group = NegRisk event (the split unit).
* **v0 gate honored:** the near-expiry pull (`pull_hours`) and toxicity knobs are only in
  the grid for a universe whose v0 pre-registered check said the near-expiry regime is
  net-negative (else those knobs are held OFF and reported as unsupported).
* **Overfitting apparatus live:** group-CSCV PBO over every trial; DSR deflated by the
  total number of configs tried. Diagnostics (uncapped run, cap sweep) count toward the
  trial budget (harsher, honest).

Usage (from polymarket/research/):
    PYTHONPATH=. uv run python scripts/mm_task5_ladder_run.py [--quick] [--workers 6]

Every run is cached by (config, token/group, queue) hash under the scratchpad, so reruns
and crashes resume for free. Deterministic: seeded bootstraps, deterministic engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from mm_engine import (BACKTEST, ASQuoter, BasketCarryQuoter, ConstantLatency, FeeModel,
                       InventoryAwareQuoter, OptimisticQueue, ProbQueue, RiskAverseQueue,
                       SymmetricQuoter, Telemetry, run_engine)
from mm_engine.feeds.replay_parquet import replay_parquet
from mm_engine.telemetry import JsonlSink

from mm_eval import markets as mk
from mm_eval import protocol as pr
from mm_eval.metrics import CENTS

RESEARCH = Path(__file__).resolve().parents[1]
L2_ROOT = Path.home() / "epsilon_l2_full"
CSV_OUT = RESEARCH / "data/analysis/csv_outputs/market_making"
PLOT_OUT = RESEARCH / "data/analysis/plots/market_making"
META_JSON = RESEARCH / "data/markets/mm_task5_market_meta.json"
VERDICT_CSV = CSV_OUT / "mm_validation_verdict.csv"
SCRATCH = Path("/private/tmp/claude-501/-Users-justiniturregui-Desktop-github-epsilon-quant-research/"
               "bdab8b15-7f54-4662-bb7c-a5684c173098/scratchpad")
CACHE = SCRATCH / "mm_task5_cache"
RUNS = CACHE / "runs"
UNIVERSES = ("politics_negrisk", "esports")

QUEUES = {"Optimistic": OptimisticQueue, "Prob(0.5)": lambda: ProbQueue(0.5),
          "RiskAverse": RiskAverseQueue}
PESS = "RiskAverse"

STRATS = {"symmetric": SymmetricQuoter, "inventory": InventoryAwareQuoter,
          "as": ASQuoter, "basket": BasketCarryQuoter}

# ── the pre-registered grids (declared before any result was seen) ────────────
V1_K_GRID = (5e-6, 2e-5, 8e-5)
V1_CAP_GRID = (200.0, 500.0)
V1_PULL_GRID = (2.0, 6.0)
TOX_SUBSETS = {
    "vel": {"tox_velocity": True},
    "imb": {"tox_imbalance": True},
    "depth": {"tox_depth": True},
    "vel+imb": {"tox_velocity": True, "tox_imbalance": True},
    "all": {"tox_velocity": True, "tox_imbalance": True, "tox_depth": True},
}
AS_GAMMA_GRID = (1e-5, 1e-4, 1e-3)
CAP_SWEEP = (100.0, 200.0, 500.0, 1000.0, float("inf"))   # understanding-only diagnostics


class SlimQuotesSink(JsonlSink):
    """Keep only the quote fields the eval needs — memory-slim (drops the per-order snaps)."""

    KEEP = ("ts_exchange", "token_id", "mid", "best_bid", "best_ask")

    def emit(self, rec: dict) -> None:  # noqa: D102
        if self.keep:
            self.records.append({k: rec.get(k) for k in self.KEEP})


@dataclass(frozen=True)
class Job:
    """One engine run: a (token|group) × config × queue cell."""

    run_id: str            # token_id or "group:<gid>"
    universe: str
    replay_dir: str
    config: str            # display name, e.g. "v1[k=2e-5,cap=200,pull=6,tox=off]"
    kind: str              # symmetric | inventory | as | basket
    overrides_json: str    # params overrides (json — keeps Job hashable/picklable)
    queue: str
    span: tuple[int, int]
    tokens_json: str       # [{token_id, half_spread, end_ms, group_id, resolved_payoff}]


def _hash(job: Job) -> str:
    return hashlib.md5(
        f"{job.run_id}|{job.config}|{job.kind}|{job.overrides_json}|{job.queue}".encode()
    ).hexdigest()[:16]


def run_job(job: Job, events: list | None = None) -> list[dict]:
    """Run one cell (against pre-parsed ``events`` when given), return per-token rows (cached)."""
    cache_f = RUNS / f"{_hash(job)}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text())

    tokens = json.loads(job.tokens_json)
    overrides = json.loads(job.overrides_json)
    tele = Telemetry(fills=JsonlSink(keep=True), orders=JsonlSink(keep=False),
                     quotes=SlimQuotesSink(keep=True))
    params = {"size": 100.0, "tick": 0.001, **overrides}
    if job.kind == "basket":
        params["half_spread_by_token"] = {t["token_id"]: t["half_spread"] for t in tokens}
        params["half_spread"] = float(np.median([t["half_spread"] for t in tokens]))
    else:
        params["half_spread"] = tokens[0]["half_spread"]
        if tokens[0]["end_ms"] is not None and np.isfinite(tokens[0]["end_ms"]):
            params.setdefault("end_date_ms", tokens[0]["end_ms"])

    result = run_engine(
        events if events is not None else replay_parquet(Path(job.replay_dir), gaps=[]),
        strategy=STRATS[job.kind](), queue_model=QUEUES[job.queue](),
        latency_model=ConstantLatency(0.0), mode=BACKTEST, params=params,
        fee_model=FeeModel(), telemetry=tele,
    )

    rows = []
    for t in tokens:
        tok = t["token_id"]
        fills = [f for f in result.fills if f["token_id"] == tok]
        # per-token quote stream (group runs interleave several tokens' events; the mid
        # trajectory and liquidation marks must come from the token's OWN quotes only)
        quotes = (tele.quotes.records if len(tokens) == 1
                  else [q for q in tele.quotes.records if q.get("token_id") == tok])
        w = pr.windowed_costed(fills, quotes, span=job.span, n_boot=800)
        daily_oos = pr.daily_pnl_series(fills, quotes, span=job.span, window="OOS")
        daily_is = pr.daily_pnl_series(fills, quotes, span=job.span, window="IS")
        # settle diagnostic: terminal inventory at the ACTUAL resolution payoff (if resolved)
        settle_alt = None
        if t.get("resolved_payoff") is not None and fills:
            last = fills[-1]
            q_end = float(last["position_after"])
            basis = float(last.get("cost_basis_after", 0.0))
            realized_total = sum(float(f.get("realized_delta", 0.0)) for f in fills)
            settle_alt = realized_total + q_end * (float(t["resolved_payoff"]) - basis)
        rows.append({
            "run_id": job.run_id, "token_id": tok, "universe": job.universe,
            "group_id": t["group_id"], "config": job.config, "kind": job.kind,
            "queue": job.queue,
            **{f"IS_{k}": v for k, v in _flat(w["IS"]).items()},
            **{f"OOS_{k}": v for k, v in _flat(w["OOS"]).items()},
            "daily_oos": list(map(float, daily_oos)), "daily_is": list(map(float, daily_is)),
            "settled_alt_usd": settle_alt,
            "naive_net_ex_rebate": result.net_ex_rebate if len(tokens) == 1 else None,
        })
    RUNS.mkdir(parents=True, exist_ok=True)
    cache_f.write_text(json.dumps(rows))
    return rows


def _flat(w: pr.WindowCosted) -> dict:
    return {"n_fills": w.n_fills, "qty": w.filled_qty, "realized_usd": w.realized_usd,
            "carry_usd": w.carry_usd, "costed_usd": w.costed_usd,
            "costed_c": w.costed_per_contract_c,
            "edge_c": w.net_edge_cents.point, "edge_lo": w.net_edge_cents.lo,
            "edge_hi": w.net_edge_cents.hi, "end_inv": w.end_inventory}


# ──────────────────────────────────────────────────────────────────────────────
# Setup: specs, dirs, calibration inputs
# ──────────────────────────────────────────────────────────────────────────────

def setup(quick: bool) -> tuple[dict, dict, dict]:
    """Materialize per-token dirs; return (token_meta, spec_by_token, span_by_universe)."""
    tmeta = pr.load_token_meta(META_JSON, VERDICT_CSV)
    con = duckdb.connect()
    spec_by_token, spans = {}, {}
    top_k = 4 if quick else 12
    for u in UNIVERSES:
        spans[u] = mk.capture_span(L2_ROOT, u, con=con)
        specs = mk.select_markets(L2_ROOT, u, top_k=top_k, con=con)
        mk.build_compact(L2_ROOT, u, specs, CACHE, con=con)
        for s in specs:
            mk.materialize_token(s, CACHE, con=con)
            spec_by_token[s.token_id] = s
    con.close()
    missing = set(spec_by_token) - set(tmeta)
    if missing:
        raise RuntimeError(f"tokens without Gamma meta: {sorted(missing)[:3]}…")
    return tmeta, spec_by_token, spans


def token_payload(tok: str, tmeta: dict, spec_by_token: dict) -> dict:
    m = tmeta[tok]
    s = spec_by_token[tok]
    return {"token_id": tok, "half_spread": s.half_spread,
            "end_ms": (m.end_ms if np.isfinite(m.end_ms) else None),
            "group_id": m.group_id, "resolved_payoff": m.resolved_payoff}


def estimate_k_arr(tok: str, con: duckdb.DuckDBPyConnection) -> float:
    """Rung-2 arrival-decay 1/mean|trade − mid| on the IS window only (per token)."""
    tdir = None
    for u in UNIVERSES:
        cand = CACHE / u / tok
        if cand.exists():
            tdir = cand
            break
    if tdir is None:
        return 50.0
    tr = con.execute(f"SELECT timestamp_ms, price FROM read_parquet('{tdir}/trades_x.parquet') "
                     f"WHERE timestamp_ms < {pr.SPLIT_TS_MS} ORDER BY timestamp_ms").df()
    bb = con.execute(f"SELECT timestamp_ms, best_bid, best_ask FROM "
                     f"read_parquet('{tdir}/bba_x.parquet') "
                     f"WHERE timestamp_ms < {pr.SPLIT_TS_MS} ORDER BY timestamp_ms").df()
    if len(tr) < 30 or len(bb) < 30:
        return 50.0
    m = pd.merge_asof(tr, bb, on="timestamp_ms", direction="backward").dropna()
    dist = (m["price"] - (m["best_bid"] + m["best_ask"]) / 2).abs()
    mean_d = float(dist.mean())
    if not np.isfinite(mean_d) or mean_d <= 0:
        return 50.0
    return float(np.clip(1.0 / mean_d, 5.0, 500.0))


def materialize_group(gid: str, toks: list[str], con: duckdb.DuckDBPyConnection,
                      universe: str) -> Path:
    out = CACHE / "groups" / f"{universe}_{gid[:18]}"
    if out.exists() and all((out / f"{t}_x.parquet").exists() for t in mk.TABLES):
        return out
    out.mkdir(parents=True, exist_ok=True)
    ph = ", ".join(["?"] * len(toks))
    for table, cols in mk.TABLE_COLS.items():
        cp = CACHE / f"_compact_{universe}_{table}.parquet"
        target = str(out / f"{table}_x.parquet").replace("'", "''")
        con.execute(
            f"COPY (SELECT {cols} FROM read_parquet(?) WHERE asset_id IN ({ph}) "
            f"ORDER BY timestamp_ms, received_ns) TO '{target}' (FORMAT parquet)",
            [str(cp), *toks],
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Config builders
# ──────────────────────────────────────────────────────────────────────────────

def v1_name(k, cap, pull, tox="off") -> str:
    p = "off" if pull == 0 else f"{pull:g}h"
    c = "inf" if np.isinf(cap) else f"{cap:g}"
    return f"v1[k={k:g},cap={c},pull={p},tox={tox}]"


def v1_overrides(k, cap, pull, tox: dict | None = None) -> dict:
    o = {"skew_k": k, "inv_cap": cap, "pull_hours": pull}
    o.update(tox or {})
    return o


def load_v0_gate() -> dict[str, bool]:
    """v0's pre-registered near-expiry answer per universe (wire pull/tox or not)."""
    f = CSV_OUT / "mm_task5_v0_preregistered_check.csv"
    if not f.exists():
        raise RuntimeError("v0 gate output missing — run scripts/mm_task5_v0_attribution.py first")
    df = pd.read_csv(f)
    return {r.universe: bool(r.wire_decisions_2_4) for r in df.itertuples()}


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def pooled_costed(df: pd.DataFrame, window: str) -> float:
    """Category-pooled per-contract costed net (¢): Σ costed_usd / Σ qty."""
    qty = df[f"{window}_qty"].sum()
    return float(df[f"{window}_costed_usd"].sum() / qty * CENTS) if qty > 0 else float("nan")


def pooled_by_config(df: pd.DataFrame, window: str) -> pd.Series:
    return pd.Series({cfg: pooled_costed(g, window) for cfg, g in df.groupby("config")},
                     dtype=float)


def level_ci(df: pd.DataFrame, col: str, n_boot=4000, seed=0) -> pr.GroupDelta:
    """Group-cluster CI on the LEVEL of a per-token metric (delta vs zero)."""
    d = df[["group_id", col]].copy()
    d["_zero"] = 0.0
    return pr.group_delta(d, col, "_zero", n_boot=n_boot, seed=seed, metric=col)


def bracket_verdict(rows: pd.DataFrame) -> str:
    """VIABLE/FRAGILE/DEAD on the OOS costed-net level, bracketed across queues."""
    def clears(queue: str) -> bool:
        sub = rows[rows.queue == queue]
        if sub.empty:
            return False
        return level_ci(sub, "OOS_costed_c").beats
    if clears(PESS):
        return "VIABLE"
    if clears("Optimistic"):
        return "FRAGILE"
    return "DEAD"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    v0_gate = load_v0_gate()
    print(f"v0 gate (wire pull/tox?): {v0_gate}")
    tmeta, spec_by_token, spans = setup(args.quick)
    tokens_by_u = {u: [t for t in spec_by_token if tmeta[t].universe == u] for u in UNIVERSES}
    span_by_u = {u: (spans[u]["ts_min"], spans[u]["ts_max"]) for u in UNIVERSES}

    con = duckdb.connect()
    k_arr_by_token = {t: estimate_k_arr(t, con) for t in spec_by_token}

    def jobs_for(config: str, kind: str, overrides: dict, queues: list[str],
                 only_universe: str | None = None) -> list[Job]:
        out = []
        for u in UNIVERSES:
            if only_universe and u != only_universe:
                continue
            for tok in tokens_by_u[u]:
                ov = dict(overrides)
                if kind == "as":
                    ov.setdefault("as_k_arr", k_arr_by_token[tok])
                payload = [token_payload(tok, tmeta, spec_by_token)]
                for q in queues:
                    out.append(Job(tok, u, str(CACHE / u / tok), config, kind,
                                   json.dumps(ov, sort_keys=True), q, span_by_u[u],
                                   json.dumps(payload)))
        return out

    all_jobs: list[Job] = []
    trial_names: set[str] = set()

    # baseline (all queues — it's in the final bracket)
    all_jobs += jobs_for("baseline", "symmetric", {}, list(QUEUES))
    trial_names.add("baseline")

    # Phase A — v1 grid (pessimistic only, IS selection). pull/tox gated by v0 per universe.
    for u in UNIVERSES:
        pulls = V1_PULL_GRID if v0_gate.get(u, False) else (0.0,)
        for k in V1_K_GRID:
            for cap in V1_CAP_GRID:
                for pull in pulls:
                    nm = v1_name(k, cap, pull)
                    all_jobs += jobs_for(nm, "inventory", v1_overrides(k, cap, pull),
                                         [PESS], only_universe=u)
                    trial_names.add(nm)

    print(f"phase A jobs: {len(all_jobs)}")
    results = run_all(all_jobs, args.workers)
    df = pd.DataFrame(results)

    # IS selection per category (pessimistic)
    sel = {}
    for u in UNIVERSES:
        cand = df[(df.universe == u) & (df.queue == PESS) & (df.kind == "inventory")]
        pooled = pooled_by_config(cand, "IS")
        sel[u] = pooled.idxmax() if len(pooled) and pooled.notna().any() else None
        print(f"[{u}] v1 IS selection: {sel[u]}  (IS pooled ¢: "
              f"{pooled.max():+.3f} vs baseline "
              f"{pooled_costed(df[(df.universe == u) & (df.queue == PESS) & (df.config == 'baseline')], 'IS'):+.3f})")

    # Phase B — tox ablation + cap sweep at each category's selected (k, cap, pull)
    phase_b: list[Job] = []
    sel_parts = {}
    for u in UNIVERSES:
        if sel[u] is None:
            continue
        m = re.match(r"v1\[k=([^,]+),cap=([^,]+),pull=([^,]+),tox=off\]", sel[u])
        k = float(m.group(1)); cap = float(m.group(2))
        pull = 0.0 if m.group(3) == "off" else float(m.group(3).rstrip("h"))
        sel_parts[u] = (k, cap, pull)
        if v0_gate.get(u, False):
            for tox_name, tox in TOX_SUBSETS.items():
                nm = v1_name(k, cap, pull, tox_name)
                phase_b += jobs_for(nm, "inventory", v1_overrides(k, cap, pull, tox),
                                    [PESS], only_universe=u)
                trial_names.add(nm)
        for cs in CAP_SWEEP:   # understanding-only diagnostics (counted as trials anyway)
            nm = v1_name(k, cs, pull) + "#capsweep"
            phase_b += jobs_for(nm, "inventory", v1_overrides(k, cs, pull),
                                [PESS], only_universe=u)
            trial_names.add(nm)
    print(f"phase B jobs: {len(phase_b)}")
    df = pd.concat([df, pd.DataFrame(run_all(phase_b, args.workers))], ignore_index=True)

    # tox subset selection ON IS: smallest subset that improves IS pooled vs tox=off
    tox_sel = {}
    order = ["vel", "imb", "depth", "vel+imb", "all"]
    for u in UNIVERSES:
        tox_sel[u] = None
        if u not in sel_parts or not v0_gate.get(u, False):
            continue
        k, cap, pull = sel_parts[u]
        base_is = pooled_costed(df[(df.universe == u) & (df.queue == PESS)
                                   & (df.config == v1_name(k, cap, pull))], "IS")
        for tox_name in order:
            cand = df[(df.universe == u) & (df.queue == PESS)
                      & (df.config == v1_name(k, cap, pull, tox_name))]
            if len(cand) and pooled_costed(cand, "IS") > base_is:
                tox_sel[u] = tox_name
                break
        print(f"[{u}] tox IS selection: {tox_sel[u]}")

    # Phase C — A-S rung 1 (γ grid, pessimistic IS selection), per category
    phase_c: list[Job] = []
    for u in UNIVERSES:
        if u not in sel_parts:
            continue
        _, cap, pull = sel_parts[u]
        for g in AS_GAMMA_GRID:
            nm = f"rung1[g={g:g}]"
            phase_c += jobs_for(nm, "as", {"as_gamma": g, "inv_cap": cap, "pull_hours": pull},
                                [PESS], only_universe=u)
            trial_names.add(f"rung1[g={g:g}]")
    print(f"phase C jobs: {len(phase_c)}")
    df = pd.concat([df, pd.DataFrame(run_all(phase_c, args.workers))], ignore_index=True)

    gamma_sel = {}
    for u in UNIVERSES:
        cand = df[(df.universe == u) & (df.queue == PESS) & (df.config.str.startswith("rung1["))]
        if cand.empty:
            continue
        pooled = pooled_by_config(cand, "IS")
        if not pooled.notna().any():
            continue
        gamma_sel[u] = float(pooled.idxmax().split("g=")[1].rstrip("]"))
        print(f"[{u}] rung1 IS selection: γ={gamma_sel[u]:g}")

    # Phase D — final ladder configs under the FULL queue bracket
    phase_d: list[Job] = []
    ladder = {}
    for u in UNIVERSES:
        if u not in sel_parts:
            continue
        k, cap, pull = sel_parts[u]
        tox = TOX_SUBSETS.get(tox_sel[u], {}) if tox_sel[u] else {}
        tox_tag = tox_sel[u] or "off"
        g = gamma_sel.get(u, AS_GAMMA_GRID[1])
        ladder[u] = [
            ("baseline", "symmetric", {}),
            (v1_name(k, cap, pull, tox_tag), "inventory", v1_overrides(k, cap, pull, tox)),
            (f"rung1[g={g:g}]", "as", {"as_gamma": g, "inv_cap": cap, "pull_hours": pull}),
            (f"rung2[g={g:g}]", "as", {"as_gamma": g, "inv_cap": cap, "pull_hours": pull,
                                       "as_use_spread": True}),
        ]
        # rung 3 (toxicity overlay) only where the v0 pre-registered gate wired decision 4;
        # for a wire=False universe it is OMITTED (unsupported), not silently shipped.
        if v0_gate.get(u, False):
            r3_tox = TOX_SUBSETS.get(tox_sel[u] or "vel+imb")
            r3_tag = tox_sel[u] or "vel+imb(seed)"
            ladder[u].append(
                (f"rung3[g={g:g},tox={r3_tag}]", "as",
                 {"as_gamma": g, "inv_cap": cap, "pull_hours": pull, "as_use_spread": True,
                  **r3_tox}))
        for nm, kind, ov in ladder[u]:
            phase_d += jobs_for(nm, kind, ov, list(QUEUES), only_universe=u)
            trial_names.add(nm)
    print(f"phase D jobs: {len(phase_d)}")
    df = pd.concat([df, pd.DataFrame(run_all(phase_d, args.workers))], ignore_index=True)

    # Phase E — basket-balanced carry (v2 alternative): per event group, full bracket
    phase_e: list[Job] = []
    for u in UNIVERSES:
        if u not in sel_parts:
            continue
        k, cap, _ = sel_parts[u]
        by_group: dict[str, list[str]] = {}
        for tok in tokens_by_u[u]:
            by_group.setdefault(tmeta[tok].group_id, []).append(tok)
        for gid, toks in by_group.items():
            gdir = materialize_group(gid, toks, con, u)
            legs = {}
            for tok in toks:
                cid = tmeta[tok].market
                siblings = [x for x in toks if tmeta[x].market == cid]
                sign = 1 if siblings.index(tok) == 0 else -1
                legs[tok] = {"cond": cid, "sign": sign}
            ov = {"skew_k": k, "inv_cap": cap, "basket_legs": legs}
            payload = [token_payload(t, tmeta, spec_by_token) for t in toks]
            for q in QUEUES:
                phase_e.append(Job(f"group:{gid[:18]}", u, str(gdir), "basket_carry", "basket",
                                   json.dumps(ov, sort_keys=True), q, span_by_u[u],
                                   json.dumps(payload)))
        trial_names.add("basket_carry")
    print(f"phase E jobs: {len(phase_e)}")
    df = pd.concat([df, pd.DataFrame(run_all(phase_e, args.workers))], ignore_index=True)
    con.close()

    df.to_json(SCRATCH / "mm_task5_all_rows.json", orient="records")
    n_trials = len(trial_names)
    print(f"\nTOTAL trials (configs tried, feeds DSR deflation): {n_trials}")

    aggregate(df, ladder, n_trials, sel_parts, tox_sel, gamma_sel, v0_gate)


def run_token_batch(batch: list[Job]) -> list[dict]:
    """Worker: ALL of one (token|group)'s cells against a single parse of its event stream.

    Parsing the parquet into MarketEvents dominates a single run's wall-clock, and the ladder
    replays each token under ~dozens of configs×queues — so events are parsed once per token
    and reused (they are frozen dataclasses; every engine run gets fresh state).
    """
    out: list[dict] = []
    todo: list[Job] = []
    for j in batch:
        cache_f = RUNS / f"{_hash(j)}.json"
        if cache_f.exists():
            out.extend(json.loads(cache_f.read_text()))
        else:
            todo.append(j)
    if todo:
        events = list(replay_parquet(Path(todo[0].replay_dir), gaps=[]))
        for j in todo:
            out.extend(run_job(j, events))
    return out


def run_all(jobs: list[Job], workers: int) -> list[dict]:
    if not jobs:
        return []
    by_dir: dict[str, list[Job]] = {}
    for j in jobs:
        by_dir.setdefault(j.replay_dir, []).append(j)
    # largest batches first so the busiest token doesn't straggle at the end
    batches = sorted(by_dir.values(), key=len, reverse=True)
    out: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_token_batch, b): b for b in batches}
        for fut in as_completed(futs):
            b = futs[fut]
            try:
                out.extend(fut.result())
            except Exception as e:  # noqa: BLE001 — surface, don't hide, per-batch failures
                print(f"  !! batch {b[0].run_id[:14]} ({len(b)} cells): {e}", flush=True)
            done += 1
            print(f"  … batch {done}/{len(batches)} done ({b[0].run_id[:14]}, {len(b)} cells)",
                  flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Final aggregation → ladder table, PBO, DSR, cross-regime
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame, ladder: dict, n_trials: int, sel_parts: dict,
              tox_sel: dict, gamma_sel: dict, v0_gate: dict) -> None:
    # 1) IS-selection table (research: report ALL configs, pessimistic, both windows)
    rows = []
    for (u, cfg), g in df[df.queue == PESS].groupby(["universe", "config"]):
        rows.append({"universe": u, "config": cfg, "n_tokens": g.token_id.nunique(),
                     "IS_pooled_c": pooled_costed(g, "IS"),
                     "OOS_pooled_c": pooled_costed(g, "OOS"),
                     "IS_fills": int(g.IS_n_fills.sum()), "OOS_fills": int(g.OOS_n_fills.sum())})
    sel_df = pd.DataFrame(rows).sort_values(["universe", "IS_pooled_c"], ascending=[True, False])
    sel_df.to_csv(CSV_OUT / "mm_task5_is_selection.csv", index=False)

    # 2) the ladder table per category
    ladder_rows = []
    for u, configs in ladder.items():
        prev_name = None
        prev_kept_name = None
        for i, (nm, kind, ov) in enumerate(configs + [("basket_carry", "basket", {})]):
            sub = df[(df.universe == u) & (df.config == nm)]
            if sub.empty:
                continue
            pess = sub[sub.queue == PESS]
            row = {"universe": u, "config": nm, "kind": kind,
                   "knobs": json.dumps({k: v for k, v in ov.items() if k != "basket_legs"}),
                   "IS_pooled_c": pooled_costed(pess, "IS"),
                   "OOS_pooled_c_RiskAverse": pooled_costed(pess, "OOS"),
                   "OOS_pooled_c_Prob": pooled_costed(sub[sub.queue == "Prob(0.5)"], "OOS"),
                   "OOS_pooled_c_Optimistic": pooled_costed(sub[sub.queue == "Optimistic"], "OOS"),
                   "OOS_fills_RA": int(pess.OOS_n_fills.sum()),
                   "verdict_bracket": bracket_verdict(sub)}
            if prev_kept_name is None:
                row.update(delta_vs_prev_c=np.nan, delta_lo=np.nan, delta_hi=np.nan,
                           keep="KEPT (baseline)")
                prev_kept_name = nm
            else:
                prev = df[(df.universe == u) & (df.config == prev_kept_name) & (df.queue == PESS)]
                merged = pess[["token_id", "group_id", "OOS_costed_c"]].merge(
                    prev[["token_id", "OOS_costed_c"]], on="token_id",
                    suffixes=("_new", "_prev"))
                d = pr.group_delta(merged, "OOS_costed_c_new", "OOS_costed_c_prev",
                                   n_boot=4000, seed=7, metric=f"{nm} vs {prev_kept_name}")
                if d.beats:
                    keep = "KEPT (beats prev OOS, group-cluster CI)"
                    prev_kept_name = nm
                elif d.improves:
                    keep = "DROP — improves point, CI spans 0 (FRAGILE/underpowered)"
                else:
                    keep = "DROP — does not improve OOS"
                row.update(delta_vs_prev_c=d.point, delta_lo=d.lo, delta_hi=d.hi, keep=keep)
            prev_name = nm
            ladder_rows.append(row)
    ladder_df = pd.DataFrame(ladder_rows)

    # 3) PBO over event groups (per category; metric = whole-capture costed ¢, pessimistic)
    pbo_rows = []
    pess_all = df[df.queue == PESS].copy()
    pess_all["combined_c"] = ((pess_all.IS_costed_usd + pess_all.OOS_costed_usd)
                              / (pess_all.IS_qty + pess_all.OOS_qty).replace(0, np.nan) * CENTS)
    for u in df.universe.unique():
        sub = pess_all[pess_all.universe == u]
        mat = sub.pivot_table(index=["token_id", "group_id"], columns="config",
                              values="combined_c").reset_index().drop(columns="token_id")
        r = pr.group_cscv_pbo(mat)
        pbo_rows.append({"universe": u, "pbo": r.pbo, "n_splits": r.n_splits,
                         "n_groups": r.n_groups, "n_configs": r.n_configs, "note": r.note})
    pbo_df = pd.DataFrame(pbo_rows)

    # 4) DSR per category for the best OOS-surviving (shipped) config
    dsr_rows = []
    shipped = {}
    for u in ladder:
        lu = ladder_df[ladder_df.universe == u]
        kept = lu[lu.keep.str.startswith("KEPT")]
        ship = kept.iloc[-1].config if len(kept) else "baseline"
        shipped[u] = ship
        pess = df[(df.universe == u) & (df.queue == PESS)]
        def pooled_daily(cfg):
            sub = pess[pess.config == cfg]
            if sub.empty:
                return np.array([])
            arrs = [np.asarray(a, dtype=float) for a in sub.daily_oos]
            L = max(len(a) for a in arrs)
            return np.nansum([np.pad(a, (0, L - len(a)), constant_values=0.0) for a in arrs], axis=0)
        trials = {cfg: pooled_daily(cfg) for cfg in pess.config.unique()}
        d = pr.dsr_for_config(trials.get(ship, np.array([])), trials)
        dsr_rows.append({"universe": u, "shipped_config": ship, **d})
    dsr_df = pd.DataFrame(dsr_rows)

    # 5) cross-regime robustness: each category's selected v1 config evaluated on the OTHER
    xr = []
    for u, other in (("politics_negrisk", "esports"), ("esports", "politics_negrisk")):
        if u not in sel_parts:
            continue
        k, cap, pull = sel_parts[u]
        nm = v1_name(k, cap, pull)
        sub = df[(df.universe == other) & (df.config == nm) & (df.queue == PESS)]
        if len(sub):
            xr.append({"selected_on": u, "evaluated_on": other, "config": nm,
                       "OOS_pooled_c": pooled_costed(sub, "OOS")})
    xr_df = pd.DataFrame(xr)

    ladder_df.to_csv(CSV_OUT / "mm_task5_ladder_table.csv", index=False)
    pbo_df.to_csv(CSV_OUT / "mm_task5_pbo.csv", index=False)
    dsr_df.to_csv(CSV_OUT / "mm_task5_dsr.csv", index=False)
    if len(xr_df):
        xr_df.to_csv(CSV_OUT / "mm_task5_cross_regime.csv", index=False)
    # per-token A/B detail for the findings note
    detail = df[df.config.isin(set(ladder_df.config))][[
        "universe", "config", "queue", "token_id", "group_id",
        "IS_costed_c", "OOS_costed_c", "OOS_edge_c", "OOS_edge_lo", "OOS_edge_hi",
        "OOS_n_fills", "OOS_qty", "OOS_realized_usd", "OOS_carry_usd", "OOS_costed_usd",
        "OOS_end_inv", "settled_alt_usd"]]
    detail.to_csv(CSV_OUT / "mm_task5_ab_tokens.csv", index=False)

    print("\n================ LADDER TABLE ================")
    print(ladder_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\n================ PBO ================")
    print(pbo_df.to_string(index=False))
    print("\n================ DSR (shipped) ================")
    print(dsr_df.to_string(index=False))
    if len(xr_df):
        print("\n================ CROSS-REGIME ================")
        print(xr_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\nshipped: {shipped}")
    summary = {"shipped": shipped, "n_trials": n_trials, "sel_parts": {k: list(v) for k, v in sel_parts.items()},
               "tox_sel": tox_sel, "gamma_sel": gamma_sel, "v0_gate": v0_gate}
    (SCRATCH / "mm_task5_ladder_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"CSVs -> {CSV_OUT}")


if __name__ == "__main__":
    main()
