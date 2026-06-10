"""OD replacement-fair sensitivity on the existing v4/Strategy-A candidate set.

This consumes the already-built conditional/model-form PM fill artifacts. It
does not rebuild datasets; it asks which Strategy-A far-|z| strict-rich shorts
survive if the token fair changes from RV physical probability to later
conditional/jump-aware definitions.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import cents, markdown_table, number, pct
from od_strategy_a_v2_lifecycle import BOOTSTRAP_SAMPLES, RNG_SEED
from od_strategy_a_v3 import normalize_markdown_wrapping, resolve_token_rv_physical_prob_fair


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
CSV_OUT = ANALYSIS / "csv_outputs" / "options_delta"

PM_FILLS = ANALYSIS / "od_pricing_model_form_pm_fills.parquet"
OUT_SUMMARY = CSV_OUT / "od_replacement_fair_sensitivity_summary.csv"
OUT_FILLS = ANALYSIS / "od_replacement_fair_sensitivity_fills.parquet"
NOTE = NOTES / "options_delta" / "od_replacement_fair_sensitivity_findings.md"
OD_HUB = NOTES / "options_delta" / "strat_options_delta.md"
BRAIN_TODO = REPO / "brain" / "TODO.md"

NON_TOP3_AVAILABLE_SHARE = 0.05
BORROWED_STRUCTURAL_BASELINE_C = 0.0198
EDGE_THRESHOLDS = [0.01, 0.05]
FAIR_SOURCES: list[tuple[str, str, str]] = [
    ("rv_physical_prob", "RV physical probability", "rv_token_prob"),
    ("arm_b_empirical_conditional", "Arm B empirical conditional", "arm_b_token_prob"),
    ("merton", "Merton jump-aware", "arm_b_merton_token_prob"),
    ("kou", "Kou jump-aware", "arm_b_kou_token_prob"),
    ("edgeworth", "Edgeworth higher-moment", "arm_c_edgeworth_token_prob"),
]


def fmt_ci(lo: float, hi: float, unit: str = "c") -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "[n/a, n/a]"
    if unit == "pct":
        return f"[{pct(lo)}, {pct(hi)}]"
    return f"[{cents(lo)}, {cents(hi)}]"


def market_sum_ci(df: pd.DataFrame, col: str, seed_offset: int = 0) -> tuple[float, float]:
    if df.empty or col not in df:
        return math.nan, math.nan
    vals = df.groupby("market_id", sort=False)[col].sum().replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(RNG_SEED + seed_offset + 31 * len(vals))
    idx = rng.integers(0, len(vals), size=(BOOTSTRAP_SAMPLES, len(vals)))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def cluster_mean_ci(df: pd.DataFrame, col: str, seed_offset: int = 0) -> tuple[float, float]:
    if df.empty or col not in df:
        return math.nan, math.nan
    groups: list[tuple[float, int]] = []
    for _, g in df.groupby("market_id", sort=False):
        vals = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(vals):
            groups.append((float(vals.sum()), int(len(vals))))
    if not groups:
        return math.nan, math.nan
    if len(groups) == 1:
        total, n = groups[0]
        return float(total / n), float(total / n)
    rng = np.random.default_rng(RNG_SEED + seed_offset + 17 * len(groups))
    sums = np.asarray([g[0] for g in groups], dtype=float)
    counts = np.asarray([g[1] for g in groups], dtype=float)
    idx = rng.integers(0, len(groups), size=(BOOTSTRAP_SAMPLES, len(groups)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def load_pm() -> pd.DataFrame:
    if not PM_FILLS.exists():
        raise SystemExit(f"missing {PM_FILLS}; rerun od_pricing_model_form.py first")
    pm = pd.read_parquet(PM_FILLS)
    for col in ("fill_ts", "ts", "window_start", "window_end"):
        if col in pm:
            pm[col] = pd.to_datetime(pm[col], utc=True)
    pm["market_id"] = pm["market_id"].astype(str)
    pm["rv_token_prob"] = resolve_token_rv_physical_prob_fair(pm, context="od_replacement_fair_sensitivity.load_pm")
    if not pm["token_position"].astype(float).lt(0).all():
        raise ValueError("replacement-fair sensitivity expects the v4 strict-rich short set; found non-short token rows")
    missing = [col for _, _, col in FAIR_SOURCES if col not in pm.columns]
    if missing:
        raise ValueError(f"missing fair columns {missing}; rerun od_conditional_prob_calibration.py and od_pricing_model_form.py")
    pm["short_unit_net_pnl"] = pm["entry_price"].astype(float) - pm["payoff"].astype(float) + pm["maker_rebate"].astype(float)
    pm["short_unit_gross_pnl"] = pm["entry_price"].astype(float) - pm["payoff"].astype(float)
    return pm.sort_values(["fill_ts", "fill_id"]).reset_index(drop=True)


def expanded_fills(pm: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for source_id, source_label, prob_col in FAIR_SOURCES:
        sub = pm.copy()
        sub["fair_source"] = source_id
        sub["fair_source_label"] = source_label
        sub["token_replacement_fair"] = pd.to_numeric(sub[prob_col], errors="coerce").clip(0.0, 1.0)
        sub["replacement_short_edge"] = sub["entry_price"].astype(float) - sub["token_replacement_fair"]
        sub["survives_rich_ge_1c"] = sub["replacement_short_edge"].ge(0.01)
        sub["survives_rich_ge_5c"] = sub["replacement_short_edge"].ge(0.05)
        sub["replacement_edge_scaled_size"] = np.where(
            sub["survives_rich_ge_1c"],
            np.clip(sub["replacement_short_edge"].astype(float) / 0.05, 0.25, 3.0),
            0.0,
        )
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    rv_survivors = set(out[out["fair_source"].eq("rv_physical_prob") & out["survives_rich_ge_1c"]]["fill_id"].astype(int))
    out["also_survives_rv_ge_1c"] = out["fill_id"].astype(int).isin(rv_survivors)
    return out


def summarize_selection(df: pd.DataFrame, *, gate: str, threshold: float, policy: str, seed_offset: int) -> dict[str, Any]:
    selected = df[df["replacement_short_edge"].ge(threshold)].copy()
    if policy == "flat_1_contract":
        selected["size"] = 1.0
    elif policy == "replacement_edge_scaled":
        selected["size"] = np.clip(selected["replacement_short_edge"].astype(float) / 0.05, 0.25, 3.0)
    else:
        raise ValueError(f"unknown policy {policy}")
    selected["weighted_net_pnl"] = selected["short_unit_net_pnl"].astype(float) * selected["size"].astype(float)
    selected["weighted_edge"] = selected["replacement_short_edge"].astype(float) * selected["size"].astype(float)
    selected["weighted_after_top3_net"] = selected["weighted_net_pnl"] * NON_TOP3_AVAILABLE_SHARE

    edge_lo, edge_hi = cluster_mean_ci(selected, "replacement_short_edge", seed_offset=seed_offset + 1)
    fill_net_lo, fill_net_hi = cluster_mean_ci(selected, "short_unit_net_pnl", seed_offset=seed_offset + 2)
    market_net_lo, market_net_hi = market_sum_ci(selected, "weighted_net_pnl", seed_offset=seed_offset + 3)
    after_lo, after_hi = market_sum_ci(selected, "weighted_after_top3_net", seed_offset=seed_offset + 4)
    market_net = selected.groupby("market_id")["weighted_net_pnl"].sum()
    market_after = selected.groupby("market_id")["weighted_after_top3_net"].sum()
    return {
        "fair_source": str(df["fair_source"].iloc[0]),
        "fair_source_label": str(df["fair_source_label"].iloc[0]),
        "selection_gate": gate,
        "sizing_policy": policy,
        "candidate_fills": int(len(df)),
        "candidate_markets": int(df["market_id"].nunique()),
        "selected_fills": int(len(selected)),
        "selected_markets": int(selected["market_id"].nunique()),
        "survival_fill_share": float(len(selected) / len(df)) if len(df) else math.nan,
        "survival_market_share": float(selected["market_id"].nunique() / df["market_id"].nunique()) if df["market_id"].nunique() else math.nan,
        "rv_ge1c_overlap_fills": int(selected["also_survives_rv_ge_1c"].sum()) if not selected.empty else 0,
        "mean_short_price": float(selected["entry_price"].mean()) if not selected.empty else math.nan,
        "mean_token_fair": float(selected["token_replacement_fair"].mean()) if not selected.empty else math.nan,
        "mean_short_edge": float(selected["replacement_short_edge"].mean()) if not selected.empty else math.nan,
        "short_edge_ci_lo": edge_lo,
        "short_edge_ci_hi": edge_hi,
        "mean_unit_net_pnl": float(selected["short_unit_net_pnl"].mean()) if not selected.empty else math.nan,
        "unit_net_pnl_ci_lo": fill_net_lo,
        "unit_net_pnl_ci_hi": fill_net_hi,
        "mean_size": float(selected["size"].mean()) if not selected.empty else math.nan,
        "total_size": float(selected["size"].sum()) if not selected.empty else 0.0,
        "mean_market_weighted_net_pnl": float(market_net.mean()) if not market_net.empty else math.nan,
        "market_weighted_net_pnl_ci_lo": market_net_lo,
        "market_weighted_net_pnl_ci_hi": market_net_hi,
        "mean_market_after_top3_net": float(market_after.mean()) if not market_after.empty else math.nan,
        "market_after_top3_net_ci_lo": after_lo,
        "market_after_top3_net_ci_hi": after_hi,
        "incremental_vs_zero": float(market_after.mean()) if not market_after.empty else math.nan,
        "incremental_vs_zero_ci_lo": after_lo,
        "incremental_vs_zero_ci_hi": after_hi,
        "borrowed_structural_baseline_mean": BORROWED_STRUCTURAL_BASELINE_C,
        "incremental_vs_borrowed_structural": float(market_after.mean() - BORROWED_STRUCTURAL_BASELINE_C) if not market_after.empty else math.nan,
        "incremental_vs_borrowed_structural_ci_lo": after_lo - BORROWED_STRUCTURAL_BASELINE_C if np.isfinite(after_lo) else math.nan,
        "incremental_vs_borrowed_structural_ci_hi": after_hi - BORROWED_STRUCTURAL_BASELINE_C if np.isfinite(after_hi) else math.nan,
        "selected_fill_ids": ",".join(map(str, selected["fill_id"].astype(int).tolist())),
    }


def summarize(expanded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, (source, g) in enumerate(expanded.groupby("fair_source", sort=False)):
        rows.append(summarize_selection(g, gate="original_set_repriced", threshold=-math.inf, policy="flat_1_contract", seed_offset=100 * i))
        for threshold in EDGE_THRESHOLDS:
            gate = f"rich_ge_{int(threshold * 100):02d}c"
            rows.append(summarize_selection(g, gate=gate, threshold=threshold, policy="flat_1_contract", seed_offset=100 * i + int(threshold * 1000)))
            rows.append(summarize_selection(g, gate=gate, threshold=threshold, policy="replacement_edge_scaled", seed_offset=100 * i + int(threshold * 1000) + 50))
    return pd.DataFrame(rows)


def compact_survival_table(summary: pd.DataFrame) -> str:
    sub = summary[summary["selection_gate"].eq("rich_ge_01c") & summary["sizing_policy"].eq("replacement_edge_scaled")].copy()
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            [
                str(r["fair_source"]),
                f"{int(r['selected_fills'])}/{int(r['candidate_fills'])}",
                f"{int(r['selected_markets'])}/{int(r['candidate_markets'])}",
                cents(float(r["mean_short_edge"])),
                fmt_ci(float(r["short_edge_ci_lo"]), float(r["short_edge_ci_hi"])),
                number(float(r["mean_size"]), 2),
                number(float(r["total_size"]), 2),
                cents(float(r["mean_market_after_top3_net"])),
                fmt_ci(float(r["market_after_top3_net_ci_lo"]), float(r["market_after_top3_net_ci_hi"])),
                cents(float(r["incremental_vs_borrowed_structural_ci_lo"])),
            ]
        )
    return markdown_table(
        [
            "fair source",
            "fills survive",
            "markets survive",
            "mean edge",
            "edge CI",
            "mean size",
            "total size",
            "after-top3 net",
            "after-top3 CI",
            "borrowed-baseline CI lo",
        ],
        rows,
    )


def full_summary_table(summary: pd.DataFrame) -> str:
    rows = []
    for _, r in summary.iterrows():
        if int(r.get("selected_fills", 0)) == 0:
            rows.append(
                [
                    str(r["fair_source"]),
                    str(r["selection_gate"]),
                    str(r["sizing_policy"]),
                    "0",
                    "0",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                ]
            )
            continue
        rows.append(
            [
                str(r["fair_source"]),
                str(r["selection_gate"]),
                str(r["sizing_policy"]),
                str(int(r["selected_fills"])),
                str(int(r["selected_markets"])),
                cents(float(r["mean_short_edge"])),
                number(float(r["total_size"]), 2),
                cents(float(r["mean_market_weighted_net_pnl"])),
                fmt_ci(float(r["market_weighted_net_pnl_ci_lo"]), float(r["market_weighted_net_pnl_ci_hi"])),
            ]
        )
    return markdown_table(
        ["fair", "gate", "sizing", "fills", "markets", "edge", "total size", "market net", "market net CI"],
        rows,
    )


def write_note(pm: pd.DataFrame, expanded: pd.DataFrame, summary: pd.DataFrame) -> None:
    surv = summary[summary["selection_gate"].eq("rich_ge_01c") & summary["sizing_policy"].eq("replacement_edge_scaled")].copy()
    best = surv.sort_values("incremental_vs_borrowed_structural_ci_lo", ascending=False).iloc[0]
    rv = surv[surv["fair_source"].eq("rv_physical_prob")].iloc[0]
    non_rv = surv[~surv["fair_source"].eq("rv_physical_prob")]
    best_non_rv = non_rv.sort_values("incremental_vs_borrowed_structural_ci_lo", ascending=False).iloc[0]
    source_counts = expanded[expanded["survives_rich_ge_1c"]].groupby("fair_source")["fill_id"].nunique().to_dict()
    note = f"""# OD Replacement-Fair Sensitivity

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior notes: [[od_strategy_a_v3_findings]] · [[od_strategy_a_v3_pnl_risk_findings]] · [[od_conditional_prob_calibration_findings]] · [[od_pricing_model_form_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

Verdict: **does not improve OD**.

This is the no-remake sensitivity: it reuses the existing 23-fill v4 far-|z| strict-rich short artifact and asks which fills would still pass if Strategy A used a better fair definition than the original RV physical probability. The sources are RV physical probability, Arm B empirical conditional probability, Merton, Kou, and Edgeworth.

The answer is mostly a hygiene confirmation, not a rescue. RV keeps {int(rv['selected_fills'])}/{int(rv['candidate_fills'])} fills at the 1c rich-short gate. The best non-RV replacement by borrowed-baseline lower CI is `{best_non_rv['fair_source']}`, with {int(best_non_rv['selected_fills'])}/{int(best_non_rv['candidate_fills'])} fills surviving and borrowed-baseline after-top3 lower CI {cents(float(best_non_rv['incremental_vs_borrowed_structural_ci_lo']))}. The best row overall is `{best['fair_source']}`, and its borrowed-baseline lower CI is {cents(float(best['incremental_vs_borrowed_structural_ci_lo']))}. None clear the standalone reopen bar.

Plain-English read: changing fair definitions changes which fills are still "rich," especially for Merton/Kou. But using better fair definitions does not create a clean, capacity-adjusted residual edge over the structural MM baseline. The branch remains useful as a quote-selection/caution feature, not as standalone OD.

## Scope

- Input: `data/analysis/od_pricing_model_form_pm_fills.parquet`.
- No upstream dataset remake: this script consumes the conditional/model-form fill artifact already written by the prior runs.
- Candidate set: {len(pm):,} v4 far-|z| strict-source rich-short fills across {pm['market_id'].nunique():,} markets.
- Sizing: `flat_1_contract` and `replacement_edge_scaled`, where edge-scaled size is `clip(edge / 5c, 0.25, 3.0)` after the 1c/5c gate.
- PnL read: token-short resolution PnL, `entry_price - payoff + maker_rebate`, aggregated by market. The after-top3 lens multiplies by {pct(NON_TOP3_AVAILABLE_SHARE)} and then compares to either 0c or the borrowed {cents(BORROWED_STRUCTURAL_BASELINE_C)} v4 structural queue baseline.

## 1c Survival With Edge-Scaled Sizing

{compact_survival_table(summary)}

Surviving fill counts at the 1c gate: {", ".join(f"`{k}` {v}" for k, v in source_counts.items())}.

## Full Summary

{full_summary_table(summary)}

## Interpretation

Use this as the Strategy A fair-source guardrail:

- `rv_physical_prob` is the original RV physical-probability fair; it is a control, not option-market IV fair.
- `arm_b_empirical_conditional` is the most model-free replacement fair. If this does not reopen OD, escalating model form should be treated skeptically.
- `merton` and `kou` add jump mass and generally reduce apparent far-tail richness.
- `edgeworth` can preserve more RV-like richness, but it still fails the MM-integrated borrowed-baseline lens here.

## Outputs

- Summary CSV: `data/analysis/csv_outputs/options_delta/od_replacement_fair_sensitivity_summary.csv`
- Fill parquet: `data/analysis/od_replacement_fair_sensitivity_fills.parquet`
"""
    NOTE.write_text(normalize_markdown_wrapping(note), encoding="utf-8")


def update_docs(summary: pd.DataFrame) -> None:
    surv = summary[summary["selection_gate"].eq("rich_ge_01c") & summary["sizing_policy"].eq("replacement_edge_scaled")].copy()
    best_non_rv = surv[~surv["fair_source"].eq("rv_physical_prob")].sort_values("incremental_vs_borrowed_structural_ci_lo", ascending=False).iloc[0]
    bullet = (
        "- 2026-06-07 OD replacement-fair sensitivity: **CLOSE remains**. Existing 23-fill v4 far-|z| strict-rich short artifact was repriced with RV physical probability, Arm B empirical conditional, Merton, Kou, and Edgeworth fairs. "
        f"Best non-RV borrowed-baseline lower CI was {cents(float(best_non_rv['incremental_vs_borrowed_structural_ci_lo']))} "
        f"on `{best_non_rv['fair_source']}` with {int(best_non_rv['selected_fills'])}/23 fills surviving the 1c gate. "
        "No dataset remake; this is a selection/sizing sensitivity only. See [[od_replacement_fair_sensitivity_findings]]."
    )
    if OD_HUB.exists():
        hub = OD_HUB.read_text(encoding="utf-8")
        hub = "\n".join(ln for ln in hub.splitlines() if "OD replacement-fair sensitivity" not in ln) + "\n"
        idx = hub.find("## Status")
        if idx >= 0:
            line_end = hub.find("\n", idx)
            suffix = hub[line_end + 1 :]
            if not suffix.startswith("\n"):
                suffix = "\n" + suffix
            hub = hub[: line_end + 1] + bullet + "\n" + suffix
        else:
            hub = hub.rstrip() + "\n\n" + bullet + "\n"
        hub = hub.replace(
            "[[od_methodology_realism_audit_findings]], [[od_conditional_prob_calibration_findings]], [[od_strategy_a_realism_reaudit_findings]], [[od_pricing_model_form_findings]]",
            "[[od_methodology_realism_audit_findings]], [[od_replacement_fair_sensitivity_findings]], [[od_conditional_prob_calibration_findings]], [[od_strategy_a_realism_reaudit_findings]], [[od_pricing_model_form_findings]]",
        )
        OD_HUB.write_text(hub, encoding="utf-8")

    if BRAIN_TODO.exists():
        todo = BRAIN_TODO.read_text(encoding="utf-8")
        todo = "\n".join(ln for ln in todo.splitlines() if "OD replacement-fair sensitivity" not in ln) + "\n"
        od_idx = todo.find("## OD")
        if od_idx >= 0:
            line_end = todo.find("\n", od_idx)
            suffix = todo[line_end + 1 :]
            if not suffix.startswith("\n"):
                suffix = "\n" + suffix
            todo = todo[: line_end + 1] + bullet + "\n" + suffix
        else:
            todo = todo.rstrip() + "\n\n## OD\n" + bullet + "\n"
        BRAIN_TODO.write_text(todo, encoding="utf-8")


def main() -> None:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    pm = load_pm()
    expanded = expanded_fills(pm)
    summary = summarize(expanded)
    summary.to_csv(OUT_SUMMARY, index=False)
    expanded.to_parquet(OUT_FILLS, index=False)
    write_note(pm, expanded, summary)
    update_docs(summary)
    print(f"wrote {OUT_SUMMARY}")
    print(f"wrote {OUT_FILLS}")
    print(f"wrote {NOTE}")
    print(summary[summary["selection_gate"].eq("rich_ge_01c")][["fair_source", "sizing_policy", "selected_fills", "selected_markets", "mean_short_edge", "incremental_vs_borrowed_structural_ci_lo"]].to_string(index=False))


if __name__ == "__main__":
    main()
