"""OD v4 Phase 0: fair-value calibration / longshot EV gate.

This intentionally stops before queue replay unless the primitive short-token
EV is positive with a market-cluster lower CI above zero.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import cents, number, pct
from od_strategy_a_v3 import FilterSpec, bucket_mask, filter_mask, load_v3_fills, markdown_table, normalize_markdown_wrapping
from od_strategy_a_v3_pnl_risk import add_claim_fields


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
BRAIN_TODO = REPO / "brain" / "TODO.md"
OD_HUB = NOTES / "options_delta" / "strat_options_delta.md"

OUT_SUMMARY = ANALYSIS / "csv_outputs" / "options_delta" / "od_v4_calibration_gate_summary.csv"
OUT_BINS = ANALYSIS / "csv_outputs" / "options_delta" / "od_v4_calibration_gate_bins.csv"
OUT_CONTRACTS = ANALYSIS / "od_v4_calibration_gate_contracts.parquet"
PLOTS = ANALYSIS / "plots" / "options_delta"
CALIBRATION_PLOT = PLOTS / "od_v4_calibration_curve.png"
EV_PLOT = PLOTS / "od_v4_ev_by_short_price.png"
NOTE = NOTES / "options_delta" / "od_v4_calibration_gate_findings.md"

SPEC = FilterSpec(
    "phase2_od_filter",
    "strict_rich_short_ge_010m",
    source_policy="strict",
    richness_threshold=0.01,
)
BOOTSTRAP_SAMPLES = 5000
RNG_SEED = 20260601
EDGE_THRESHOLD = 0.01


def norm_cdf(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(arr / math.sqrt(2.0)))


def ci_text(lo: float, hi: float, unit: str = "c") -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "[n/a, n/a]"
    if unit == "pct":
        return f"[{pct(lo)}, {pct(hi)}]"
    return f"[{cents(lo)}, {cents(hi)}]"


def cluster_ci(df: pd.DataFrame, col: str, *, cluster_col: str = "market_id", seed_offset: int = 0) -> tuple[float, float]:
    if df.empty:
        return math.nan, math.nan
    groups = []
    for _, g in df.groupby(cluster_col, sort=False):
        vals = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(vals):
            groups.append((float(vals.sum()), int(len(vals))))
    if not groups:
        return math.nan, math.nan
    if len(groups) == 1:
        s, n = groups[0]
        v = s / n if n else math.nan
        return float(v), float(v)
    sums = np.asarray([g[0] for g in groups], dtype=float)
    counts = np.asarray([g[1] for g in groups], dtype=float)
    rng = np.random.default_rng(RNG_SEED + seed_offset + len(groups))
    idx = rng.integers(0, len(groups), size=(BOOTSTRAP_SAMPLES, len(groups)))
    draw_sum = sums[idx].sum(axis=1)
    draw_count = counts[idx].sum(axis=1)
    vals = draw_sum / draw_count
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def market_equal_ci(df: pd.DataFrame, col: str, *, seed_offset: int = 0) -> tuple[float, float]:
    if df.empty:
        return math.nan, math.nan
    means = df.groupby("market_id")[col].mean().replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(means) == 0:
        return math.nan, math.nan
    if len(means) == 1:
        return float(means[0]), float(means[0])
    rng = np.random.default_rng(RNG_SEED + seed_offset + 13 * len(means))
    idx = rng.integers(0, len(means), size=(BOOTSTRAP_SAMPLES, len(means)))
    vals = means[idx].mean(axis=1)
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def add_v4_fields(fills: pd.DataFrame) -> pd.DataFrame:
    out = add_claim_fields(fills)
    out["short_price"] = out["entry_price"].astype(float)
    out["pred_itm_prob"] = out["token_model_fair"].astype(float)
    out["realized_itm"] = out["payoff"].astype(float)
    out["gross_short_ev_realized"] = out["short_price"] - out["realized_itm"]
    out["net_short_ev_realized"] = out["gross_short_ev_realized"] + out["maker_rebate"].astype(float)
    out["model_edge"] = out["short_price"] - out["pred_itm_prob"]
    out["calibration_error_realized_minus_pred"] = out["realized_itm"] - out["pred_itm_prob"]
    sigma = out["trailing_sigma_annualized"].astype(float).replace(0.0, np.nan)
    tau = out["tau_years"].astype(float).replace(0.0, np.nan)
    z_trailing = out["log_spot_moneyness"].astype(float) / (sigma * np.sqrt(tau))
    out["z_trailing"] = z_trailing
    out["abs_z_trailing"] = z_trailing.abs()
    p_up_trailing = norm_cdf(z_trailing)
    out["token_model_fair_trailing"] = np.where(out["actual_outcome"].astype(str).eq("up"), p_up_trailing, 1.0 - p_up_trailing)
    out["model_edge_trailing"] = out["short_price"] - out["token_model_fair_trailing"].astype(float)
    out["passes_trailing_far_rich"] = out["abs_z_trailing"].ge(1.0) & out["model_edge_trailing"].ge(EDGE_THRESHOLD)
    out["short_price_bucket"] = pd.cut(
        out["short_price"],
        bins=[0.0, 0.10, 0.25, 0.39, 0.50, 0.75, 0.90, 1.01],
        labels=["p_lt_10c", "p_10_25c", "p_25_39c", "p_39_50c", "p_50_75c", "p_75_90c", "p_90_100c"],
        include_lowest=True,
    )
    out["abs_z_bucket_v4"] = pd.cut(
        out["abs_z"].astype(float),
        bins=[1.0, 1.25, 1.50, 2.0, 3.0, np.inf],
        labels=["z_1_1p25", "z_1p25_1p5", "z_1p5_2", "z_2_3", "z_ge_3"],
        include_lowest=True,
    )
    return out


def select_gate_set(fills: pd.DataFrame) -> pd.DataFrame:
    mask = filter_mask(fills, SPEC) & bucket_mask(fills, "far_absz_ge1_all_tau")
    return fills[mask].copy().sort_values(["market_id", "fill_ts_key", "fill_id"]).reset_index(drop=True)


def summarize_subset(df: pd.DataFrame, label: str, row_type: str) -> dict[str, Any]:
    if df.empty:
        return {
            "row_type": row_type,
            "label": label,
            "fills": 0,
            "markets": 0,
        }
    ev_lo, ev_hi = cluster_ci(df, "gross_short_ev_realized", seed_offset=1)
    net_lo, net_hi = cluster_ci(df, "net_short_ev_realized", seed_offset=2)
    mkt_ev_lo, mkt_ev_hi = market_equal_ci(df, "gross_short_ev_realized", seed_offset=3)
    return {
        "row_type": row_type,
        "label": label,
        "fills": int(len(df)),
        "markets": int(df["market_id"].nunique()),
        "mean_short_price": float(df["short_price"].mean()),
        "mean_pred_itm_prob": float(df["pred_itm_prob"].mean()),
        "realized_itm_rate": float(df["realized_itm"].mean()),
        "mean_model_edge": float(df["model_edge"].mean()),
        "mean_gross_ev": float(df["gross_short_ev_realized"].mean()),
        "gross_ev_ci_lo": ev_lo,
        "gross_ev_ci_hi": ev_hi,
        "market_equal_gross_ev": float(df.groupby("market_id")["gross_short_ev_realized"].mean().mean()),
        "market_equal_gross_ev_ci_lo": mkt_ev_lo,
        "market_equal_gross_ev_ci_hi": mkt_ev_hi,
        "mean_net_ev": float(df["net_short_ev_realized"].mean()),
        "net_ev_ci_lo": net_lo,
        "net_ev_ci_hi": net_hi,
        "mean_rebate": float(df["maker_rebate"].mean()),
        "mean_abs_z": float(df["abs_z"].mean()),
        "median_abs_z": float(df["abs_z"].median()),
        "mean_abs_z_trailing": float(df["abs_z_trailing"].mean()),
        "trailing_far_rich_share": float(df["passes_trailing_far_rich"].mean()),
        "ewma_far_but_trailing_not_far_share": float(df["abs_z_trailing"].lt(1.0).mean()),
        "ewma_rich_but_trailing_not_rich_share": float(df["model_edge_trailing"].lt(EDGE_THRESHOLD).mean()),
    }


def build_breakouts(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(summarize_subset(selected, "full_far_strict_rich_short", "overall"))
    trailing = selected[selected["passes_trailing_far_rich"]].copy()
    rows.append(summarize_subset(trailing, "trailing_vol_confirmed_far_strict_rich", "corrected_trailing_proxy"))
    for bucket, g in selected.groupby("short_price_bucket", observed=False):
        rows.append(summarize_subset(g, str(bucket), "short_price_bucket"))
    for bucket, g in selected.groupby("abs_z_bucket_v4", observed=False):
        rows.append(summarize_subset(g, str(bucket), "abs_z_bucket"))
    for bucket, g in selected.groupby("time_bucket", observed=False):
        rows.append(summarize_subset(g, str(bucket), "time_bucket"))
    for bucket, g in selected.groupby("asset", observed=False):
        rows.append(summarize_subset(g, str(bucket), "asset"))
    return pd.DataFrame(rows)


def build_calibration_bins(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    out["pred_prob_bin"] = pd.cut(
        out["pred_itm_prob"],
        bins=[0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.01],
        labels=["0_5c", "5_10c", "10_25c", "25_50c", "50_75c", "75_90c", "90_95c", "95_100c"],
        include_lowest=True,
    )
    rows = []
    for bucket, g in out.groupby("pred_prob_bin", observed=False):
        row = summarize_subset(g, str(bucket), "calibration_pred_prob_bin")
        row["mean_pred"] = float(g["pred_itm_prob"].mean()) if len(g) else math.nan
        row["observed_freq"] = float(g["realized_itm"].mean()) if len(g) else math.nan
        row["calibration_gap_obs_minus_pred"] = row["observed_freq"] - row["mean_pred"] if np.isfinite(row["mean_pred"]) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def concentration_diagnostics(selected: pd.DataFrame) -> pd.DataFrame:
    market = (
        selected.groupby(["market_id", "asset", "market_slug"], as_index=False)
        .agg(
            fills=("fill_id", "count"),
            mean_short_price=("short_price", "mean"),
            mean_pred_itm_prob=("pred_itm_prob", "mean"),
            realized_itm_rate=("realized_itm", "mean"),
            gross_ev=("gross_short_ev_realized", "mean"),
            net_pnl=("net_short_ev_realized", "sum"),
            mean_abs_z=("abs_z", "mean"),
            trailing_far_rich_share=("passes_trailing_far_rich", "mean"),
        )
        .sort_values("net_pnl", ascending=False)
    )
    total = float(selected["net_short_ev_realized"].sum())
    market["net_pnl_share_of_total"] = market["net_pnl"] / total if abs(total) > 1e-12 else math.nan
    loo = []
    for market_id in market["market_id"]:
        rest = selected[~selected["market_id"].eq(market_id)]
        row = summarize_subset(rest, f"drop_market_{market_id}", "leave_one_market_out")
        row["dropped_market_id"] = market_id
        loo.append(row)
    return pd.concat([market.assign(row_type="market_concentration"), pd.DataFrame(loo)], ignore_index=True, sort=False)


def make_plots(calibration: pd.DataFrame, breakouts: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    cal = calibration[calibration["fills"].fillna(0).astype(int).gt(0)].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if not cal.empty:
        sizes = np.maximum(cal["fills"].astype(float), 1.0) * 35.0
        ax.scatter(cal["mean_pred"], cal["observed_freq"], s=sizes, color="#1f77b4", alpha=0.75)
        for _, r in cal.iterrows():
            ax.annotate(str(r["label"]), (r["mean_pred"], r["observed_freq"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.plot([0, 1], [0, 1], color="#444444", linewidth=1, linestyle="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("OD predicted probability token pays $1")
    ax.set_ylabel("Realized frequency token paid $1")
    ax.set_title("OD v4 Phase 0 calibration: strict-rich far-|z| shorts")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(CALIBRATION_PLOT, dpi=160)
    plt.close(fig)

    price = breakouts[breakouts["row_type"].eq("short_price_bucket") & breakouts["fills"].fillna(0).astype(int).gt(0)].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not price.empty:
        x = np.arange(len(price))
        ax.bar(x, price["mean_gross_ev"], color="#2ca02c")
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(price["label"], rotation=35, ha="right")
        for i, (_, r) in enumerate(price.iterrows()):
            ax.text(i, float(r["mean_gross_ev"]), f"n={int(r['fills'])}", ha="center", va="bottom" if r["mean_gross_ev"] >= 0 else "top", fontsize=8)
    ax.set_ylabel("Realized short EV per contract ($)")
    ax.set_title("Realized EV by short-token price bucket")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EV_PLOT, dpi=160)
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], formatters: dict[str, Any] | None = None, limit: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    piece = df.copy()
    if limit is not None:
        piece = piece.head(limit)
    rows = []
    for _, r in piece.iterrows():
        row = []
        for col in cols:
            val = r.get(col, math.nan)
            if formatters and col in formatters:
                row.append(formatters[col](val, r))
            elif isinstance(val, float):
                row.append(number(val, 3))
            else:
                row.append(str(val))
        rows.append(row)
    headers = cols
    return markdown_table(headers, rows)


def update_docs(verdict: str, overall: pd.Series) -> None:
    bullet = (
        f"- 2026-06-01 OD v4 calibration gate: **{verdict}**. Full-panel far-|z| strict-rich shorts are only "
        f"{int(overall['fills'])} fills / {int(overall['markets'])} markets; mean gross short EV {cents(float(overall['mean_gross_ev']))}, "
        f"market-cluster CI {ci_text(float(overall['gross_ev_ci_lo']), float(overall['gross_ev_ci_hi']))}, realized ITM {pct(float(overall['realized_itm_rate']))}. "
        "Queue-aware replay remains blocked unless the calibration/EV gate is reopened with a corrected forward-vol model. See [[od_v4_calibration_gate_findings]]."
    )
    hub = OD_HUB.read_text()
    marker = "## Current state"
    idx = hub.find(marker)
    if idx >= 0:
        line_end = hub.find("\n", idx)
        next_idx = hub.find("\n## ", idx + 1)
        if next_idx < 0:
            next_idx = len(hub)
        section = hub[idx:next_idx]
        lines = [ln for ln in section.splitlines() if "OD v4 calibration gate" not in ln]
        heading = lines[0]
        rest = "\n".join(lines[1:]).strip()
        new_section = f"{heading}\n\n{bullet}"
        if rest:
            new_section += "\n" + rest
        hub = hub[:idx] + new_section.rstrip() + "\n" + hub[next_idx:]
    else:
        hub = hub.rstrip() + "\n\n## Current state (2026-06-01)\n\n" + bullet + "\n"
    hub = hub.replace(
        "Kronos still gated pending explicit reopen.",
        "Kronos/queue replay still gated pending explicit reopen.",
    )
    OD_HUB.write_text(hub)

    todo = BRAIN_TODO.read_text()
    todo = "\n".join(ln for ln in todo.splitlines() if "OD v4 calibration gate" not in ln) + "\n"
    od_idx = todo.find("## OD")
    if od_idx >= 0:
        line_end = todo.find("\n", od_idx)
        if line_end < 0:
            line_end = len(todo)
        suffix = todo[line_end + 1 :]
        if not suffix.startswith("\n"):
            suffix = "\n" + suffix
        todo = todo[: line_end + 1] + bullet + "\n" + suffix
    else:
        todo = todo.rstrip() + "\n\n## OD\n" + bullet + "\n"
    BRAIN_TODO.write_text(todo)


def write_note(
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    concentration: pd.DataFrame,
    verdict: str,
    phase1_allowed: bool,
) -> None:
    overall = summary[summary["label"].eq("full_far_strict_rich_short")].iloc[0]
    trailing = summary[summary["label"].eq("trailing_vol_confirmed_far_strict_rich")].iloc[0]
    price = summary[summary["row_type"].eq("short_price_bucket") & summary["fills"].fillna(0).astype(int).gt(0)].copy()
    absz = summary[summary["row_type"].eq("abs_z_bucket") & summary["fills"].fillna(0).astype(int).gt(0)].copy()
    market = concentration[concentration["row_type"].eq("market_concentration")].copy()
    loo = concentration[concentration["row_type"].eq("leave_one_market_out")].copy()
    top_market = market.iloc[0] if not market.empty else None
    worst_loo = loo.sort_values("mean_gross_ev").iloc[0] if not loo.empty else None

    gate_line = (
        f"Phase 0 **{verdict}**. The full-panel strict-rich far-|z| short set has {int(overall['fills'])} fills across "
        f"{int(overall['markets'])} markets. Mean short price is {cents(float(overall['mean_short_price']))}, realized ITM is "
        f"{pct(float(overall['realized_itm_rate']))}, gross EV `p - P_itm` is {cents(float(overall['mean_gross_ev']))} with "
        f"market-cluster CI {ci_text(float(overall['gross_ev_ci_lo']), float(overall['gross_ev_ci_hi']))}; net of maker rebate it is "
        f"{cents(float(overall['mean_net_ev']))}, CI {ci_text(float(overall['net_ev_ci_lo']), float(overall['net_ev_ci_hi']))}."
    )
    if phase1_allowed:
        action = "Phase 1 is allowed by the calibration gate, but should still require queue-adjusted fills and incremental-over-MM lower-CI > 0."
    else:
        action = "Phase 1 queue-aware replay was **not run**. The calibration/EV signal is too small, too concentrated, and not lower-CI positive enough to justify building execution infra."

    top_text = ""
    if top_market is not None:
        top_text = (
            f"The largest positive market is `{top_market['market_slug']}` with {int(top_market['fills'])} fills and "
            f"{cents(float(top_market['net_pnl']))} net PnL. It contributes {pct(float(top_market['net_pnl_share_of_total']))} of total net PnL."
        )
    loo_text = ""
    if worst_loo is not None:
        loo_text = (
            f"Leave-one-market-out sensitivity: the weakest remaining mean gross EV is {cents(float(worst_loo['mean_gross_ev']))} "
            f"after dropping `{worst_loo['dropped_market_id']}`."
        )

    fmts = {
        "fills": lambda v, r: str(int(v)) if np.isfinite(float(v)) else "0",
        "markets": lambda v, r: str(int(v)) if np.isfinite(float(v)) else "0",
        "mean_short_price": lambda v, r: cents(float(v)),
        "mean_pred_itm_prob": lambda v, r: pct(float(v)),
        "realized_itm_rate": lambda v, r: pct(float(v)),
        "mean_model_edge": lambda v, r: cents(float(v)),
        "mean_gross_ev": lambda v, r: cents(float(v)),
        "gross_ev_ci_lo": lambda v, r: ci_text(float(r["gross_ev_ci_lo"]), float(r["gross_ev_ci_hi"])),
        "mean_net_ev": lambda v, r: cents(float(v)),
        "net_ev_ci_lo": lambda v, r: ci_text(float(r["net_ev_ci_lo"]), float(r["net_ev_ci_hi"])),
        "trailing_far_rich_share": lambda v, r: pct(float(v)),
        "observed_freq": lambda v, r: pct(float(v)),
        "mean_pred": lambda v, r: pct(float(v)),
        "calibration_gap_obs_minus_pred": lambda v, r: cents(float(v)),
        "net_pnl": lambda v, r: cents(float(v)),
        "net_pnl_share_of_total": lambda v, r: pct(float(v)),
    }
    overall_table = md_table(
        summary[summary["row_type"].isin(["overall", "corrected_trailing_proxy"])],
        [
            "label",
            "fills",
            "markets",
            "mean_short_price",
            "mean_pred_itm_prob",
            "realized_itm_rate",
            "mean_model_edge",
            "mean_gross_ev",
            "gross_ev_ci_lo",
            "mean_net_ev",
            "net_ev_ci_lo",
            "trailing_far_rich_share",
        ],
        fmts,
    )
    calibration_table = md_table(
        calibration[calibration["fills"].fillna(0).astype(int).gt(0)],
        ["label", "fills", "markets", "mean_pred", "observed_freq", "calibration_gap_obs_minus_pred", "mean_short_price", "mean_gross_ev", "gross_ev_ci_lo"],
        fmts,
    )
    price_table = md_table(
        price,
        ["label", "fills", "markets", "mean_short_price", "mean_pred_itm_prob", "realized_itm_rate", "mean_gross_ev", "gross_ev_ci_lo", "mean_net_ev"],
        fmts,
    )
    absz_table = md_table(
        absz,
        ["label", "fills", "markets", "mean_abs_z", "mean_abs_z_trailing", "realized_itm_rate", "mean_gross_ev", "gross_ev_ci_lo", "trailing_far_rich_share"],
        fmts,
    )
    market_table = md_table(
        market,
        ["market_id", "asset", "fills", "mean_short_price", "mean_pred_itm_prob", "realized_itm_rate", "gross_ev", "net_pnl", "net_pnl_share_of_total", "trailing_far_rich_share"],
        {
            **fmts,
            "gross_ev": lambda v, r: cents(float(v)),
            "mean_abs_z": lambda v, r: number(float(v), 2),
        },
    )

    note = f"""# OD v4 Calibration Gate: Is The Rich Longshot Signal Real Enough To Build Queue Replay?

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior note: [[od_strategy_a_v3_pnl_risk_findings]]
> MM benchmark notes: [[mm_deployable_cells_findings]] · [[block_k5_stress_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Phase 0 Verdict

{gate_line}

{action}

{top_text}

{loo_text}

Plain-English read: the scary `39.13%` realized ITM number is not literally "we shorted sub-39c longshots that hit 39% of the time." This set mixes low-price longshot shorts and high-price rich-token shorts. But the gate still does not green-light infra: the full-panel sample is only 23 fills / 8 markets, the confidence interval is wide, and the profit is highly concentrated in one BTC window.

## What This Gate Tests

The primitive economics of a short binary token are:

```text
short EV before rebate = short price p - probability token pays $1
net realized short PnL = p - realized_payoff + maker_rebate
```

If `p - P_itm` is not convincingly positive on the full panel, a queue-aware replay cannot rescue the OD thesis. Queue replay can improve fill realism; it cannot turn a negative-resolution bet into an edge.

## Overall EV And Calibration

| label | fills | markets | mean_short_price | mean_pred_itm_prob | realized_itm_rate | mean_model_edge | mean_gross_ev | gross_ev_ci_lo | mean_net_ev | net_ev_ci_lo | trailing_far_rich_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(overall_table.splitlines()[2:])}

`mean_pred_itm_prob` is the OD model's probability that the token we shorted pays $1. `realized_itm_rate` is how often it actually paid. `mean_model_edge` is the edge claimed at entry (`price - predicted probability`). `mean_gross_ev` is the realized primitive short EV (`price - realized payoff`). The CI is market-clustered so repeated fills inside one 4h market do not create fake precision.

![Calibration curve]({CALIBRATION_PLOT.resolve()})

The diagonal is perfect calibration. Points above the diagonal mean the token paid more often than OD predicted, which is bad for a short; points below mean OD overpredicted ITM.

| predicted probability bin | fills | markets | mean pred | observed freq | obs - pred | mean price | mean gross EV | EV CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(calibration_table.splitlines()[2:])}

## Short-Price Buckets

This is the bucket table that resolves the `39%` confusion. Low-price shorts and high-price shorts are different bets. Shorting a 14c token that pays 0 has very different risk from shorting a 99c token that pays 1.

![EV by short price]({EV_PLOT.resolve()})

| short price bucket | fills | markets | mean price | predicted ITM | realized ITM | gross EV | EV CI | net EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(price_table.splitlines()[2:])}

Read: the low-price buckets are positive in this tiny sample because those tokens did not resolve ITM. The high-price buckets can still be positive in cents if price was above realized rate, but they are not "longshot shorts"; they are rich high-probability token shorts, economically equivalent to buying cheap complements.

## Is The `far-|z|` Bucket A Vol-Sizing Artifact?

`abs_z` is computed from Binance moneyness divided by causal EWMA vol and time left. If EWMA vol is too low, ordinary states can be mislabeled as far-|z| longshots. As a cheap causal proxy for the requested HAR/Kronos question, this gate recomputes the same classification using trailing realized vol already in the panel. This is not a full HAR/Kronos bake-off; it is only an artifact screen.

| abs_z bucket | fills | markets | EWMA abs_z | trailing abs_z | realized ITM | gross EV | EV CI | trailing far/rich share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(absz_table.splitlines()[2:])}

The trailing-vol proxy keeps {int(trailing['fills'])} of {int(overall['fills'])} fills as far and rich. Its gross EV is {cents(float(trailing['mean_gross_ev']))}, CI {ci_text(float(trailing['gross_ev_ci_lo']), float(trailing['gross_ev_ci_hi']))}. That does not meet the prompt's condition for proceeding: no corrected forward-vol model here flips the gate into a lower-CI-positive result, and the actual HAR/Kronos path model remains gated rather than run.

## Concentration / Small-Sample Diagnosis

| market_id | asset | fills | mean price | predicted ITM | realized ITM | gross EV | net PnL | PnL share | trailing far/rich share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(market_table.splitlines()[2:])}

The `net PnL share` column is why this remains a gate fail even though the fill-weighted point estimate is positive. A narrow Phase 1 queue replay would mostly be replaying whether we could capture the same concentrated windows, not proving a broad OD mispricing.

## Decision

Phase 1 queue-aware replay is skipped in this run. The OD richness signal remains useful as a **selection feature** to carry back into MM/source-filtered lifecycle analysis, but it is not strong enough as a standalone longshot-EV edge to justify new execution infrastructure.

Required condition to reopen Phase 1: a corrected causal fair-value model, preferably HAR-RV or Kronos forward paths per [[2026-05-31_kronos_hermes_eval]], must show `p - P_itm` lower-CI > 0 on the full far-|z| strict-rich set or a pre-registered sub-bucket with enough independent markets.

## Outputs

- CSV summary: `data/analysis/csv_outputs/options_delta/od_v4_calibration_gate_summary.csv`
- CSV calibration bins: `data/analysis/csv_outputs/options_delta/od_v4_calibration_gate_bins.csv`
- Contract parquet: `data/analysis/od_v4_calibration_gate_contracts.parquet`
"""
    NOTE.write_text(normalize_markdown_wrapping(note))
    update_docs(verdict, overall)


def run() -> None:
    fills = add_v4_fields(load_v3_fills(refresh=False))
    selected = select_gate_set(fills)
    summary = build_breakouts(selected)
    calibration = build_calibration_bins(selected)
    concentration = concentration_diagnostics(selected)
    make_plots(calibration, summary)

    overall = summary[summary["label"].eq("full_far_strict_rich_short")].iloc[0]
    phase1_allowed = bool(float(overall["gross_ev_ci_lo"]) > 0 and float(overall["net_ev_ci_lo"]) > 0 and int(overall["markets"]) >= 12)
    verdict = "PASS" if phase1_allowed else "FAIL"

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    calibration.to_csv(OUT_BINS, index=False)
    selected.to_parquet(OUT_CONTRACTS, index=False)
    write_note(selected, summary, calibration, concentration, verdict, phase1_allowed)
    print(f"selected fills={len(selected)} markets={selected['market_id'].nunique()}")
    print(f"phase0 verdict={verdict}; phase1_allowed={phase1_allowed}")
    print(f"wrote {OUT_SUMMARY}")
    print(f"wrote {OUT_BINS}")
    print(f"wrote {OUT_CONTRACTS}")
    print(f"wrote {NOTE}")


if __name__ == "__main__":
    run()
