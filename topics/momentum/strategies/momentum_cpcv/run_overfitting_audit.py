"""
Overfitting audit of the live 6-asset momentum book — first application of
infrastructure/validation/overfitting_audit.py.

Pre-registered design (2026-06-10, BEFORE running; see
momentum_overfitting_audit_findings.md for the verdict):

- Universe: the live book (ADA, AVAX, BTC, ETH, SOL, XRP; BNB researched but
  not live). Artifacts: oos/{sym}usdt_cpcv.pkl (the set behind the 2.24
  headline) — OHLCV reconstructed FROM the artifacts so the sample is
  identical to the original run (a live Binance refetch would shift it).
- Per asset: replay one same-design 400-trial Optuna TPE study (TPESampler
  seed=42, the asset's exact param_defs/fixed_params from its pkl, the
  notebook's SCORE_FN/REJECT_FN) over the full sample, capturing every
  explored config's full-sample net returns -> T x ~400 trial matrix.
  The original per-split studies were discarded by the engine, so this is a
  same-design replay, not a bit-identical recovery.
- Per asset stats: DSR at n_eff (avg-pairwise-correlation effective trial
  count of one 400-trial selection event; raw 400 and total-search 11,200
  reported as sensitivity), PBO via CSCV (S=16; S=8/12 sensitivity), White's
  Reality Check (studentised, stationary bootstrap), stationary-bootstrap
  Sharpe CI of the median-Sharpe CPCV path, MinTRL at SR*.
- Portfolio: haircut aggregated in RETURN units — sleeve selection bias
  inflates each sleeve's mean by ~SR*_i * sigma_i; means average linearly
  while portfolio vol shrinks, so SR*_p = (1/6) sum(SR*_i sigma_i) / sigma_p
  (diversification does NOT diversify away selection bias). Portfolio DSR =
  PSR of the median-Sharpe portfolio path at benchmark SR*_p.
- GATE (pre-registered): portfolio deflated Sharpe > 0 AND mean per-asset
  PBO < 0.5 AND post-haircut lower-CI Sharpe > 0.25 (materiality bar, set
  before results were seen; engine-convention adjusted CI). Otherwise the
  live book is flagged for review. RC p < 0.05 is supporting, not gated.

Run:  ./.venv/bin/python topics/momentum/strategies/momentum_cpcv/run_overfitting_audit.py
Outputs: oos/overfitting_audit_2026-06-10.pkl, overfitting_audit_summary.csv,
         overfitting_audit_sharpe_haircut.png, markdown fragments on stdout.
"""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for sub in ("walkforward", "backtester", "validation"):
    sys.path.insert(0, str(ROOT / "infrastructure" / sub))

from overfitting_audit import (  # noqa: E402
    effective_n_trials,
    deflated_sharpe_ratio,
    expected_max_sharpe_null,
    min_track_record_length,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    reconstruct_ohlcv_from_cpcv,
    replay_search_trial_matrix,
    stationary_bootstrap_sharpe_ci,
    whites_reality_check,
    GATE_HAIRCUT_LCI_MIN,
)

PPY = 365.0
ANN = math.sqrt(PPY)
N_TRIALS = 400
SEED = 42
AUDIT_DATE = "2026-06-10"

# Live universe and variants (matches live_trading/dashboards/momentum
# STRATEGY_REGISTRY; verified identical to each asset's CPCV notebook cell).
SWING_ASSETS = ["adausdt", "ethusdt", "solusdt", "xrpusdt"]   # volume-MA filter
NO_VOL_ASSETS = ["avaxusdt", "btcusdt"]                       # no volume filter
# Notebook SCORE_FN CALMAR_MAX per asset (only difference between notebooks).
CALMAR_MAX = {"adausdt": 10.0, "btcusdt": 10.0, "ethusdt": 10.0, "xrpusdt": 10.0,
              "avaxusdt": 60.0, "solusdt": 60.0}


# ── strategy functions — verbatim from the CPCV notebooks (which produced the
#    audited artifacts). Two variants; only the Volume > Vol_MA entry clause
#    and its indicator bookkeeping differ. ─────────────────────────────────────

def _momentum_strategy(df_slice: pd.DataFrame, params: dict, use_volume: bool):
    df = df_slice.copy()

    df['EMA']          = df['Close'].ewm(span=params['ema_span'], adjust=False).mean()
    df['Swing_Hi_Cau'] = df['High'].rolling(params['swing_caution']).max()
    df['Swing_Lo_Cau'] = df['Low'].rolling(params['swing_caution']).min()
    df['Swing_Hi_Stp'] = df['High'].rolling(params['swing_stop']).max()

    def atr(period):
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift(1)).abs()
        lc = (df['Low']  - df['Close'].shift(1)).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=period, adjust=False).mean()

    df['ATR_Cau'] = atr(params['atr_caution'])
    df['ATR_Stp'] = atr(params['atr_stop'])
    df['ATR_Sz']  = atr(params['atr_size'])

    up    = df['High'].diff();  down = -df['Low'].diff()
    pdm   = up.where((up > down) & (up > 0), 0.0)
    ndm   = down.where((down > up) & (down > 0), 0.0)
    atr14 = atr(14)
    pdi   = 100 * pdm.ewm(span=14, adjust=False).mean() / atr14
    ndi   = 100 * ndm.ewm(span=14, adjust=False).mean() / atr14
    dx    = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
    df['ADX_14'] = dx.ewm(span=14, adjust=False).mean()

    if use_volume:
        df['Vol_MA'] = df['Volume'].rolling(params['vol_ma_period']).mean()
    direction     = df['Close'].diff().apply(lambda x: 1 if x > 0 else -1)
    df['OBV']     = (df['Volume'] * direction).cumsum()
    df['OBV_MA']  = df['OBV'].rolling(params['obv_ma_period']).mean()

    df['Caution_OBV']   = (df['Close'] > df['Close'].shift(params['obv_lookback'])) & (df['OBV'] < df['OBV_MA'])
    df['Caution_Long']  = ((df['Swing_Hi_Cau'] - df['Low']) > 1.5 * df['ATR_Cau']) | df['Caution_OBV']
    df['Caution_Short'] = ((df['High'] - df['Swing_Lo_Cau']) > 1.5 * df['ATR_Cau']) | (df['Close'] > df['EMA'])
    _valid = df['Swing_Hi_Stp'].notna() & df['ATR_Stp'].notna() & df['ATR_Sz'].notna() & df['OBV_MA'].notna()
    if use_volume:
        _valid &= df['Vol_MA'].notna()
    df['Entry_Long'] = (df['Close'] > df['EMA']) & (~df['Caution_Long'] | (df['ADX_14'] > params['adx_override'])) & _valid
    if use_volume:
        df['Entry_Long'] &= (df['Volume'] > df['Vol_MA'])
    df['position_size_raw'] = (params['risk_per_trade'] / (df['ATR_Sz'] / df['Close'])).clip(0.1, params['max_leverage'])

    n             = len(df)
    position      = [0]      * n
    position_size = [0.0]    * n
    stop_arr      = [np.nan] * n
    in_position   = 0
    stop_loss     = np.nan
    current_size  = 0.0

    for i in range(1, n):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if in_position == 1:
            if prev['Close'] < stop_loss:
                in_position  = 0
                current_size = 0.0
                stop_loss    = np.nan
            else:
                sm        = params['stop_mult_pos_caution'] if curr['Caution_Long'] else params['stop_mult_pos_normal']
                stop_loss = max(stop_loss, curr['Swing_Hi_Stp'] - curr['ATR_Stp'] * sm * params['stop_atr_scale'])

        if in_position == 0:
            if curr['Entry_Long']:
                in_position  = 1
                current_size = curr['position_size_raw']
                cl = curr['Caution_Long']; cs = curr['Caution_Short']
                if cl and cs: sm = params['stop_mult_ent_both']
                elif cl:      sm = params['stop_mult_ent_caution']
                else:         sm = params['stop_mult_ent_normal']
                stop_loss = curr['Swing_Hi_Stp'] - curr['ATR_Stp'] * sm * params['stop_atr_scale']
        position[i]      = in_position
        position_size[i] = current_size
        stop_arr[i]      = stop_loss

    df['position']      = position
    df['position_size'] = position_size
    df['stop_loss']     = stop_arr

    indicator_cols = ['EMA', 'ATR_Cau', 'ADX_14', 'Swing_Hi_Cau', 'OBV_MA']
    if use_volume:
        indicator_cols.insert(4, 'Vol_MA')
    df['position']      = df['position'].fillna(0).astype(int)
    df['position_size'] = df['position_size'].fillna(0.0)
    df['stop_loss']     = df['stop_loss'].fillna(0.0)

    return df, indicator_cols


def make_strategy_fn(sym):
    use_volume = sym in SWING_ASSETS
    return lambda df, params: _momentum_strategy(df, params, use_volume)


def make_score_fn(sym):
    calmar_max = CALMAR_MAX[sym]

    def score_fn(metrics):  # notebook SCORE_FN, CALMAR_MAX per asset
        calmar = (metrics['total_return'] / abs(metrics['max_drawdown'])
                  if metrics['max_drawdown'] != 0 else 0.0)
        s = np.clip(metrics['sharpe_ratio'] / 2.5, 0, 1)
        c = np.clip(calmar / calmar_max, 0, 1)
        r = np.clip(metrics['total_return'] / 15.0, 0, 1)
        return 0.50 * s + 0.30 * c + 0.20 * r
    return score_fn


def reject_fn(metrics):  # identical across all notebooks
    if metrics is None:                 return True
    if metrics['num_trades']    < 7:    return True
    if metrics['win_rate']      < 0.35: return True
    if metrics['max_drawdown']  < -0.80: return True
    if metrics['profit_factor'] < 0.8:  return True
    return False


def audit_asset(sym):
    t0 = time.time()
    results = pd.read_pickle(HERE / "oos" / f"{sym}_cpcv.pkl")
    cfg = results["config"]
    df = reconstruct_ohlcv_from_cpcv(results)
    print(f"\n=== {sym.upper()} | {len(df)} bars {df.index[0].date()} -> {df.index[-1].date()} "
          f"| {len(results['param_defs']) - len(results['fixed_params'])} free params ===")

    matrix = replay_search_trial_matrix(
        df, make_strategy_fn(sym), results["param_defs"], results["fixed_params"],
        n_trials=N_TRIALS, cost=cfg["cost"], score_fn=make_score_fn(sym),
        reject_fn=reject_fn, seed=SEED, verbose=False,
    )

    paths = [p for p in results["paths"] if p.get("sharpe") is not None]
    path_sharpes = np.array([p["sharpe"] for p in paths])
    med_path = sorted(paths, key=lambda p: p["sharpe"])[len(paths) // 2]
    selected = med_path["equity_curve"].pct_change().dropna()

    mat = matrix.values
    sds = mat.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        trial_sharpes = np.where(sds > 0, mat.mean(axis=0) / sds, 0.0)
    n_eff = effective_n_trials(mat)
    var_sr = float(np.asarray(trial_sharpes).var(ddof=1))

    dsr = deflated_sharpe_ratio(selected, N_TRIALS, PPY,
                                var_trial_sr=var_sr, n_eff=n_eff)
    # sensitivity: raw 400 (no correlation shrinkage) and the full 28x400 search
    sr_star_raw400 = expected_max_sharpe_null(N_TRIALS, var_sr) * ANN
    sr_star_total = expected_max_sharpe_null(28 * N_TRIALS, var_sr) * ANN

    pbo = pbo_cscv(mat, n_blocks=16)
    rc = whites_reality_check(mat, n_boot=2000, studentize=True, seed=SEED)
    lo, hi, _ = stationary_bootstrap_sharpe_ci(selected, PPY, n_boot=2000, seed=SEED)

    # cross-path summary with the engine's overlap-adjusted N_eff (= 7 for N=8,k=2)
    n_eff_paths = 7.0
    p_mean, p_std = path_sharpes.mean(), path_sharpes.std(ddof=1)
    t95 = 2.447  # t(df=6, 97.5%)
    adj = t95 * p_std / math.sqrt(n_eff_paths)

    out = {
        "sym": sym, "n_bars": len(df), "n_trials_matrix": mat.shape[1],
        "n_eff": n_eff, "var_trial_sr": var_sr,
        "path_sharpe_mean": float(p_mean), "path_sharpe_std": float(p_std),
        "path_sharpe_adj_ci": (float(p_mean - adj), float(p_mean + adj)),
        "selected_sharpe_ann": dsr.sr_ann, "sharpe_ci": (lo, hi),
        "sr_star_ann": dsr.sr_star_ann, "sr_star_raw400_ann": sr_star_raw400,
        "sr_star_total_ann": sr_star_total,
        "deflated_sr_ann": dsr.deflated_sr_ann, "dsr_prob": dsr.dsr_prob,
        "skew": dsr.skew, "kurt": dsr.kurt,
        "pbo": pbo.pbo, "pbo_sens": pbo.sensitivity, "p_oos_loss": pbo.p_oos_loss,
        "rc_p": rc.p_value, "rc_p_raw": rc.p_value_raw, "rc_mc_se": rc.mc_se,
        "min_trl_bars": min_track_record_length(selected, dsr.sr_star_ann / ANN),
        "sigma_ann": float(selected.std(ddof=1) * ANN),
        "selected_returns": selected, "trial_sharpes_ann": trial_sharpes * ANN,
        "runtime_s": time.time() - t0,
    }
    print(f"  path Sharpe {p_mean:.2f} (adjCI [{p_mean-adj:.2f}, {p_mean+adj:.2f}]) | "
          f"median-path {dsr.sr_ann:.2f} CI [{lo:.2f}, {hi:.2f}]")
    print(f"  N_eff {n_eff:.0f}/{mat.shape[1]} | SR* {dsr.sr_star_ann:.2f} "
          f"(raw400 {sr_star_raw400:.2f}, total11200 {sr_star_total:.2f}) | "
          f"deflated {dsr.deflated_sr_ann:.2f} | DSR prob {dsr.dsr_prob:.3f}")
    print(f"  PBO {pbo.pbo:.3f} (sens {pbo.sensitivity}) | P(OOS loss) {pbo.p_oos_loss:.3f} | "
          f"RC p {rc.p_value:.4f} (raw {rc.p_value_raw:.4f}) | {out['runtime_s']:.0f}s")
    return out


def main():
    assets = SWING_ASSETS + NO_VOL_ASSETS
    per_asset = {sym: audit_asset(sym) for sym in sorted(assets)}

    # ── portfolio layer ───────────────────────────────────────────────────────
    ci = pd.read_pickle(HERE / "oos" / "portfolio_cpcv_ci.pkl")
    port_paths = pd.read_pickle(HERE / "oos" / "portfolio_cpcv_paths.pkl")
    p_sh = ci["sharpe"]
    port_mean, port_adj_ci = float(p_sh["mean"]), tuple(map(float, p_sh["adjusted_ci"]))

    med_port = sorted(port_paths, key=lambda p: p["sharpe"])[len(port_paths) // 2]
    port_rets = med_port["portfolio_returns"].dropna()
    sigma_p = float(port_rets.std(ddof=1) * ANN)

    # selection-bias drag adds in RETURN units across sleeves; vol diversifies,
    # bias does not -> portfolio Sharpe haircut = mean(SR*_i * sigma_i) / sigma_p
    drags = [per_asset[s]["sr_star_ann"] * per_asset[s]["sigma_ann"] for s in per_asset]
    sr_star_p = float(np.mean(drags) / sigma_p)
    drags_raw = [per_asset[s]["sr_star_raw400_ann"] * per_asset[s]["sigma_ann"] for s in per_asset]
    sr_star_p_raw = float(np.mean(drags_raw) / sigma_p)
    drags_tot = [per_asset[s]["sr_star_total_ann"] * per_asset[s]["sigma_ann"] for s in per_asset]
    sr_star_p_tot = float(np.mean(drags_tot) / sigma_p)

    port_dsr_prob = probabilistic_sharpe_ratio(port_rets, sr_star_p / ANN)
    lo_b, hi_b, _ = stationary_bootstrap_sharpe_ci(port_rets, PPY, n_boot=2000, seed=SEED)

    mean_pbo = float(np.mean([per_asset[s]["pbo"] for s in per_asset]))
    post_haircut = port_mean - sr_star_p
    post_haircut_lci = port_adj_ci[0] - sr_star_p

    gates = {
        "deflated_sharpe_gt_0": post_haircut > 0,
        "mean_pbo_lt_0.5": mean_pbo < 0.5,
        "post_haircut_lci_gt_0.25": post_haircut_lci > GATE_HAIRCUT_LCI_MIN,
    }
    verdict = "PASS" if all(gates.values()) else "FLAG-FOR-REVIEW"

    print("\n" + "=" * 72)
    print("PORTFOLIO (live 6-asset equal-weight book)")
    print(f"  headline Sharpe {port_mean:.3f} | engine adjusted CI [{port_adj_ci[0]:.2f}, {port_adj_ci[1]:.2f}]")
    print(f"  median-port-path Sharpe {med_port['sharpe']:.2f} | bootstrap CI [{lo_b:.2f}, {hi_b:.2f}]")
    print(f"  portfolio SR* haircut {sr_star_p:.2f} (raw400 {sr_star_p_raw:.2f}, total {sr_star_p_tot:.2f})")
    print(f"  post-haircut Sharpe {post_haircut:.2f} | post-haircut adj-CI lower {post_haircut_lci:.2f}")
    print(f"  portfolio DSR prob {port_dsr_prob:.4f} | mean per-asset PBO {mean_pbo:.3f}")
    print(f"  gates: {gates}")
    print(f"  VERDICT: {verdict}")

    portfolio = {
        "headline_sharpe": port_mean, "adjusted_ci": port_adj_ci,
        "median_path_sharpe": float(med_port["sharpe"]),
        "bootstrap_ci": (lo_b, hi_b), "sigma_p_ann": sigma_p,
        "sr_star_p": sr_star_p, "sr_star_p_raw400": sr_star_p_raw,
        "sr_star_p_total": sr_star_p_tot,
        "post_haircut_sharpe": post_haircut, "post_haircut_lci": post_haircut_lci,
        "dsr_prob": port_dsr_prob, "mean_pbo": mean_pbo,
        "gates": gates, "verdict": verdict,
    }

    out = {"audit_date": AUDIT_DATE, "design": __doc__, "per_asset": per_asset,
           "portfolio": portfolio, "n_trials": N_TRIALS, "seed": SEED}
    pd.to_pickle(out, HERE / "oos" / f"overfitting_audit_{AUDIT_DATE}.pkl")

    rows = []
    for s, a in per_asset.items():
        rows.append({
            "asset": s.replace("usdt", "").upper(),
            "path_sharpe_mean": round(a["path_sharpe_mean"], 3),
            "selected_sharpe": round(a["selected_sharpe_ann"], 3),
            "sharpe_ci_lo": round(a["sharpe_ci"][0], 3),
            "sharpe_ci_hi": round(a["sharpe_ci"][1], 3),
            "n_eff": round(a["n_eff"], 1),
            "sr_star": round(a["sr_star_ann"], 3),
            "deflated_sharpe": round(a["deflated_sr_ann"], 3),
            "dsr_prob": round(a["dsr_prob"], 4),
            "pbo": round(a["pbo"], 4),
            "p_oos_loss": round(a["p_oos_loss"], 4),
            "rc_p": round(a["rc_p"], 4),
        })
    pd.DataFrame(rows).to_csv(HERE / "overfitting_audit_summary.csv", index=False)

    _plot(per_asset, portfolio)
    print(f"\nSaved: oos/overfitting_audit_{AUDIT_DATE}.pkl, overfitting_audit_summary.csv, "
          f"overfitting_audit_sharpe_haircut.png")
    return out


def _plot(per_asset, portfolio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    syms = sorted(per_asset)
    labels = [s.replace("usdt", "").upper() for s in syms] + ["PORTFOLIO"]
    observed = [per_asset[s]["selected_sharpe_ann"] for s in syms] + [portfolio["headline_sharpe"]]
    stars = [per_asset[s]["sr_star_ann"] for s in syms] + [portfolio["sr_star_p"]]
    deflated = [o - h for o, h in zip(observed, stars)]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.25, observed, 0.25, label="Observed OOS Sharpe", color="#4878d0")
    ax.bar(x, stars, 0.25, label="Selection haircut SR* (expected max under null)", color="#d65f5f")
    ax.bar(x + 0.25, deflated, 0.25, label="Deflated Sharpe (observed − SR*)", color="#6acc64")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(GATE_HAIRCUT_LCI_MIN, color="gray", lw=0.8, ls="--",
               label=f"materiality bar ({GATE_HAIRCUT_LCI_MIN})")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Annualised Sharpe")
    ax.set_title(f"Momentum book overfitting audit {AUDIT_DATE}: observed vs "
                 f"selection-haircut Sharpe (400-trial TPE replay, CSCV S=16)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "overfitting_audit_sharpe_haircut.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
