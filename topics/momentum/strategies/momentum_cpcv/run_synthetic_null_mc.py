"""
Synthetic-null Monte Carlo for the live 6-asset momentum book — the empirical
complement to the 2026-06-10 DSR/PBO/Reality-Check audit (run_overfitting_audit.py).

Pre-registered design (2026-06-10, BEFORE running):

- Question: could the search pipeline (best of 400 TPE-explored configs)
  manufacture the real data's max Sharpe on synthetic data that preserves
  drift, fat tails, and short-range vol clustering but has NO exploitable
  long-range trend structure?
- Candidate set: the SAME 400 configs per asset as the audit (deterministic
  TPE replay, seed 42; max trial Sharpe asserted equal to the audit pickle's).
- Null: stationary block bootstrap of bar-level relative bars, mean block
  10 bars, resampled JOINTLY across the 6 assets (same block indices ->
  cross-asset correlation preserved) on their common dates (2020-10 ->
  2026-04, ~2,030 daily bars). Drift-preserving primary; sensitivity:
  demeaned null (drift removed) and mean block 5 / 20.
- Statistic (same on both sides): max-of-400-configs full-period Sharpe per
  asset, computed with the verified fast pipeline; portfolio = equal-weight
  of the 6 per-asset best-config return series. Real stats computed on the
  same common dates with the same pipeline.
- GATE (pre-registered): real stat > 95th percentile of the null
  distribution, per asset and for the portfolio (primary null). 200 primary
  paths (gate resolution ~±1.5% MC error at p≈0.05); 100 per sensitivity.
- Fast-pipeline integrity: verify_fast_pipeline must pass (Sharpe identical
  to the notebook strategy + engine.backtest) for both variants, else abort.

Run:  ./.venv/bin/python topics/momentum/strategies/momentum_cpcv/run_synthetic_null_mc.py
Outputs: oos/synthetic_null_mc_2026-06-10.pkl, synthetic_null_mc_summary.csv,
         synthetic_null_mc_dist.png
"""

import math
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for sub in ("walkforward", "backtester", "validation"):
    sys.path.insert(0, str(ROOT / "infrastructure" / sub))
sys.path.insert(0, str(HERE))

from overfitting_audit import (  # noqa: E402
    make_null_ohlcv,
    reconstruct_ohlcv_from_cpcv,
    replay_search_trial_matrix,
    summarize_null_mc,
)
from fast_momentum import fast_net_returns, verify_fast_pipeline  # noqa: E402
import run_overfitting_audit as roa  # noqa: E402  (strategy fns, SCORE/REJECT, asset lists)

PPY = 365.0
AUDIT_DATE = "2026-06-10"
N_TRIALS = 400
SEED_REPLAY = 42
SEED_PATHS = 1000
N_PATHS_PRIMARY = 200
N_PATHS_SENS = 100
BLOCK_PRIMARY = 10.0

ASSETS = sorted(roa.SWING_ASSETS + roa.NO_VOL_ASSETS)
USE_VOL = {s: s in roa.SWING_ASSETS for s in ASSETS}

# Null variants: (label, mean_block_len, demean, n_paths)
VARIANTS = [
    ("primary_block10", BLOCK_PRIMARY, False, N_PATHS_PRIMARY),
    ("demeaned_block10", BLOCK_PRIMARY, True, N_PATHS_SENS),
    ("block5", 5.0, False, N_PATHS_SENS),
    ("block20", 20.0, False, N_PATHS_SENS),
]

_PAYLOAD = {}  # worker globals: {sym: (df_common, params_list, use_volume)}, 'cost'


def _eval_asset(df, params_list, use_volume, cost):
    """All configs on one OHLCV frame -> (max_sharpe, median_sharpe, best_net_returns)."""
    best_sr, best_net = -np.inf, None
    sharpes = np.empty(len(params_list))
    for i, p in enumerate(params_list):
        net = fast_net_returns(df, p, use_volume, cost)
        sd = net.std(ddof=1)
        sr = net.mean() / sd * math.sqrt(PPY) if sd > 0 else 0.0
        sharpes[i] = sr
        if sr > best_sr:
            best_sr, best_net = sr, net
    return float(best_sr), float(np.median(sharpes)), best_net


def _portfolio_sharpe(best_nets):
    port = np.mean(np.column_stack(best_nets), axis=1)
    sd = port.std(ddof=1)
    return float(port.mean() / sd * math.sqrt(PPY)) if sd > 0 else 0.0


def _init_worker(payload):
    _PAYLOAD.update(payload)


def _run_path(args):
    """One null path: joint resample -> per-asset max/median + portfolio Sharpe."""
    k, block_len, demean = args
    cost = _PAYLOAD["cost"]
    dfs = {s: _PAYLOAD[s][0] for s in _PAYLOAD["assets"]}
    synth = make_null_ohlcv(dfs, block_len, rng=np.random.default_rng(SEED_PATHS + k),
                            demean=demean)
    out, best_nets = {}, []
    for s in _PAYLOAD["assets"]:
        _, params_list, use_volume = _PAYLOAD[s]
        mx, med, net = _eval_asset(synth[s], params_list, use_volume, cost)
        out[s] = (mx, med)
        best_nets.append(net)
    return k, out, _portfolio_sharpe(best_nets)


def main():
    t0 = time.time()

    # ── 1. verify the fast pipeline before trusting half a million evals ──────
    print("verifying fast pipeline vs notebook strategy + engine.backtest ...")
    for sym in ("btcusdt", "adausdt"):  # one per variant
        res = pd.read_pickle(HERE / "oos" / f"{sym}_cpcv.pkl")
        df = reconstruct_ohlcv_from_cpcv(res)
        worst = verify_fast_pipeline(df, roa.make_strategy_fn(sym), USE_VOL[sym],
                                     res["param_defs"], res["fixed_params"],
                                     res["config"]["cost"], n_draws=15, seed=7)
        print(f"  {sym}: identical across 15 draws (worst |dSharpe| = {worst:.1e})")

    # ── 2. recover the audit's 400 candidate configs per asset (deterministic) ─
    audit = pd.read_pickle(HERE / "oos" / f"overfitting_audit_{AUDIT_DATE}.pkl")
    payload = {"assets": ASSETS}
    full_dfs, cost = {}, None
    for sym in ASSETS:
        res = pd.read_pickle(HERE / "oos" / f"{sym}_cpcv.pkl")
        cost = res["config"]["cost"]
        df = reconstruct_ohlcv_from_cpcv(res)
        full_dfs[sym] = df
        print(f"replaying {sym} search to recover candidate configs ...")
        mat, params_list = replay_search_trial_matrix(
            df, roa.make_strategy_fn(sym), res["param_defs"], res["fixed_params"],
            n_trials=N_TRIALS, cost=cost, score_fn=roa.make_score_fn(sym),
            reject_fn=roa.reject_fn, seed=SEED_REPLAY, return_params=True)
        # determinism check vs the audit run
        sds = mat.values.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            srs = np.where(sds > 0, mat.values.mean(axis=0) / sds, 0.0) * math.sqrt(PPY)
        audit_max = float(np.max(audit["per_asset"][sym]["trial_sharpes_ann"]))
        assert abs(float(srs.max()) - audit_max) < 1e-9, \
            f"{sym}: replay differs from audit ({srs.max()} vs {audit_max})"
        payload[sym] = (None, params_list, USE_VOL[sym])

    # ── 3. align to common dates, compute real stats with the same pipeline ───
    common = full_dfs[ASSETS[0]].index
    for sym in ASSETS[1:]:
        common = common.intersection(full_dfs[sym].index)
    print(f"\ncommon dates: {common[0].date()} -> {common[-1].date()} ({len(common)} bars)")
    real, real_best_nets = {}, []
    for sym in ASSETS:
        dfc = full_dfs[sym].loc[common]
        payload[sym] = (dfc, payload[sym][1], USE_VOL[sym])
        mx, med, net = _eval_asset(dfc, payload[sym][1], USE_VOL[sym], cost)
        real[sym] = {"max": mx, "median": med}
        real_best_nets.append(net)
        print(f"  {sym}: real max-of-400 = {mx:.2f} | median config = {med:.2f}")
    real_port = _portfolio_sharpe(real_best_nets)
    print(f"  PORTFOLIO (equal-weight best configs): real = {real_port:.2f}")
    payload["cost"] = cost

    # ── 4. null paths, all variants, parallel ──────────────────────────────────
    jobs = []
    offset = 0
    for label, block, demean, n_paths in VARIANTS:
        jobs += [(offset + k, block, demean) for k in range(n_paths)]
        offset += n_paths
    workers = max(2, (cpu_count() or 8) - 2)
    print(f"\nrunning {len(jobs)} null paths on {workers} workers ...")
    with Pool(workers, initializer=_init_worker, initargs=(payload,)) as pool:
        results = pool.map(_run_path, jobs, chunksize=4)
    results.sort(key=lambda r: r[0])

    # ── 5. split results back into variants, summarize ─────────────────────────
    summary, rows = {}, []
    offset = 0
    for label, block, demean, n_paths in VARIANTS:
        chunk = results[offset:offset + n_paths]
        offset += n_paths
        var = {"label": label, "mean_block_len": block, "demeaned": demean,
               "per_asset": {}, "portfolio": None}
        for sym in ASSETS:
            nm = summarize_null_mc(real[sym]["max"], [c[1][sym][0] for c in chunk],
                                   block, demean)
            nmed = summarize_null_mc(real[sym]["median"], [c[1][sym][1] for c in chunk],
                                     block, demean)
            var["per_asset"][sym] = {"max": nm, "median": nmed}
        var["portfolio"] = summarize_null_mc(real_port, [c[2] for c in chunk], block, demean)
        summary[label] = var

        p = var["portfolio"]
        print(f"\n[{label}] portfolio: real {p.real_stat:.2f} vs null q95 {p.q95:.2f} "
              f"(pctile {p.percentile:.3f}) -> {'PASS' if p.passes else 'FAIL'}")
        for sym in ASSETS:
            m = var["per_asset"][sym]["max"]
            print(f"  {sym}: real {m.real_stat:.2f} | null q95 {m.q95:.2f} | "
                  f"pctile {m.percentile:.3f} | {'PASS' if m.passes else 'FAIL'}")
            rows.append({"variant": label, "asset": sym.replace("usdt", "").upper(),
                         "real_max": round(m.real_stat, 3), "null_q95": round(m.q95, 3),
                         "null_median": round(float(np.median(m.null_stats)), 3),
                         "percentile": round(m.percentile, 4), "passes": m.passes})
        rows.append({"variant": label, "asset": "PORTFOLIO",
                     "real_max": round(p.real_stat, 3), "null_q95": round(p.q95, 3),
                     "null_median": round(float(np.median(p.null_stats)), 3),
                     "percentile": round(p.percentile, 4), "passes": p.passes})

    primary = summary["primary_block10"]
    gate = (primary["portfolio"].passes
            and all(primary["per_asset"][s]["max"].passes for s in ASSETS))
    print(f"\nPRE-REGISTERED GATE (primary null, real > q95 everywhere): "
          f"{'PASS' if gate else 'FAIL'} | runtime {time.time() - t0:.0f}s")

    out = {"audit_date": AUDIT_DATE, "design": __doc__, "assets": ASSETS,
           "real": real, "real_portfolio": real_port, "variants": summary,
           "gate_pass": gate, "n_paths_primary": N_PATHS_PRIMARY,
           "common_dates": (str(common[0].date()), str(common[-1].date()), len(common))}
    pd.to_pickle(out, HERE / "oos" / f"synthetic_null_mc_{AUDIT_DATE}.pkl")
    pd.DataFrame(rows).to_csv(HERE / "synthetic_null_mc_summary.csv", index=False)
    _plot(summary, real, real_port)
    print(f"saved: oos/synthetic_null_mc_{AUDIT_DATE}.pkl, synthetic_null_mc_summary.csv, "
          f"synthetic_null_mc_dist.png")
    return out


def _plot(summary, real, real_port):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prim = summary["primary_block10"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    p = prim["portfolio"]
    ax.hist(p.null_stats, bins=30, color="#9ecae1", edgecolor="white",
            label=f"null max-of-400 portfolio Sharpe ({p.n_paths} paths)")
    ax.axvline(p.q95, color="#d65f5f", ls="--", label=f"null 95th pct = {p.q95:.2f}")
    ax.axvline(p.real_stat, color="#2a7d2a", lw=2, label=f"real = {p.real_stat:.2f}")
    ax.set_title("Portfolio: real vs drift-preserving null (block 10)")
    ax.set_xlabel("Annualised Sharpe (equal-weight of per-asset best configs)")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = [s.replace("usdt", "").upper() for s in prim["per_asset"]] + ["PORT"]
    for j, (label, var) in enumerate(summary.items()):
        pct = [var["per_asset"][s]["max"].percentile for s in var["per_asset"]] + \
              [var["portfolio"].percentile]
        ax.plot(labels, pct, marker="o", ms=5, lw=1.2, label=label)
    ax.axhline(0.95, color="gray", ls="--", lw=0.8, label="gate (0.95)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Real stat's percentile in each null distribution")
    ax.set_ylabel("percentile of real max-of-400 Sharpe")
    ax.legend(fontsize=7)

    fig.suptitle(f"Synthetic-null Monte Carlo — momentum book {AUDIT_DATE}")
    fig.tight_layout()
    fig.savefig(HERE / "synthetic_null_mc_dist.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
