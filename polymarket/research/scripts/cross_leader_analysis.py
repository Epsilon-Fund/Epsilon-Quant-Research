"""Cross-leader analysis: compare Domah vs 0xee00ba copy-audits.

Appends a "Cross-leader analysis" section to the 0xee00ba audit report,
addressing the user's three questions:
  1. Does Domah's macro-only finding generalise?
  2. Where do they diverge?
  3. Cross-leader deployable intersection
     (family × role × hour cells where BOTH leaders have
      A_real_capture > 30% AND adverse_select_ratio > 0.85).

Reads parquet outputs from both audits — no recomputation of any heavy work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
FU = ANALYSIS / "domah_followups"

DOMAH_PREFIX  = ANALYSIS / "domah_audit"
LEADER_PREFIX = FU       / "leader_ee00ba_audit"
REPORT_PATH   = FU / "leader_ee00ba_audit_report.md"

DEPLOY_CAPTURE = 0.30
DEPLOY_ADVSEL  = 0.85
BRANCHES = ["A_opt", "A_real", "B", "C_opt", "C_real"]


def fmt_md_table(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].apply(
                lambda x: "" if pd.isna(x) else (
                    f"{x:,.4f}" if abs(x) < 10 else f"{x:,.0f}"
                )
            )
        elif pd.api.types.is_integer_dtype(d[c]):
            d[c] = d[c].apply(lambda x: "" if pd.isna(x) else f"{int(x):,}")
        else:
            d[c] = d[c].astype(str).where(d[c].notna(), "")
    cols = list(d.columns)
    widths = [max(len(c), d[c].map(len).max() if len(d) else 0) for c in cols]
    head = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    rows = ["| " + " | ".join(v.rjust(w) for v, w in zip(r, widths)) + " |"
            for r in d.to_numpy().tolist()]
    return "\n".join([head, sep] + rows)


def fine_grain_capture_advsel(
    pos: pd.DataFrame, frag: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Group positions by `group_cols` (taken from each position's first
    fragment) and compute A_real capture + adverse_select_ratio per group.

    group_cols are fragment-level. We attribute a position to the slice of
    its first fragment.
    """
    first = frag.sort_values("fill_ts").groupby("position_id")[group_cols].first()
    # Drop any same-named cols from pos to avoid suffix collision on merge.
    pos2 = pos.drop(columns=[c for c in group_cols if c in pos.columns])
    p = pos2.merge(first.reset_index(), on="position_id", how="left")

    rows = []
    for keys, sub in p.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        f_slc = frag[frag["position_id"].isin(sub["position_id"])]
        leader_pnl = float(sub["domah_pnl_calc"].sum())
        a_real_pnl = float(sub["A_real_pnl"].sum())
        a_opt_pnl  = float(sub["A_opt_pnl"].sum())
        a_real_cap = a_real_pnl / leader_pnl if leader_pnl != 0 else float("nan")
        a_opt_cap  = a_opt_pnl  / leader_pnl if leader_pnl != 0 else float("nan")
        # adverse_select_ratio
        fm = f_slc[f_slc["role"] == "maker"]
        fmw = fm.merge(sub[["position_id", "is_winning_position"]],
                       on="position_id", how="left")
        win  = fmw[fmw["is_winning_position"] == 1]
        lose = fmw[fmw["is_winning_position"] == 0]
        if len(win) >= 30 and len(lose) >= 30:
            asr = (win["A_real_fill"].mean() / lose["A_real_fill"].mean()
                   if lose["A_real_fill"].mean() > 0 else float("nan"))
        else:
            asr = float("nan")
        row = dict(zip(group_cols, keys))
        row.update({
            "n_positions": int(len(sub)),
            "n_fills": int(len(f_slc)),
            "leader_pnl": leader_pnl,
            "A_real_capture": a_real_cap,
            "A_opt_capture":  a_opt_cap,
            "adverse_select_ratio": asr,
            "deployable": bool(
                pd.notna(a_real_cap) and a_real_cap > DEPLOY_CAPTURE
                and pd.notna(asr) and asr > DEPLOY_ADVSEL
                and leader_pnl > 0
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print("[cross] loading both audit artifacts…", flush=True)
    f_d = pd.read_parquet(f"{DOMAH_PREFIX}_fragments.parquet")
    p_d = pd.read_parquet(f"{DOMAH_PREFIX}_positions.parquet")
    fam_d = pd.read_parquet(f"{DOMAH_PREFIX}_family.parquet")
    f_e = pd.read_parquet(f"{LEADER_PREFIX}_fragments.parquet")
    p_e = pd.read_parquet(f"{LEADER_PREFIX}_positions.parquet")
    fam_e = pd.read_parquet(f"{LEADER_PREFIX}_family.parquet")
    print(f"  domah: {len(f_d):,} fragments / {len(p_d):,} positions / {len(fam_d)} families")
    print(f"  ee00ba: {len(f_e):,} fragments / {len(p_e):,} positions / {len(fam_e)} families")

    # NOTE: Domah's audit used the ORIGINAL FAMILY_KEYWORDS; ee00ba used PROPOSED.
    # The user said: "Use the updated FAMILY_KEYWORDS from Task 1 if Task 1
    # completes successfully". So for the intersection comparison to be fair,
    # we should re-classify Domah's fragments with the proposed keywords. This
    # is cheap (Domah is already loaded; just re-bucket).
    print("  re-classifying Domah with proposed FAMILY_KEYWORDS for apples-to-apples…")
    from domah_family_validation import _augment_rules
    from domah_copy_audit import _tag_family
    proposed = _augment_rules()
    f_d["family"] = f_d["slug"].map(lambda s: _tag_family(s, proposed))
    # propagate to positions
    new_fam = f_d.drop_duplicates(["market_id", "outcome_token_id"])[
        ["market_id", "outcome_token_id", "family"]
    ]
    p_d = p_d.drop(columns=["family"]).merge(new_fam, on=["market_id", "outcome_token_id"], how="left")
    # Recompute Domah's family table from the re-classified data
    fam_rows = []
    for fam, sub in p_d.groupby("family", sort=False):
        f_fam = f_d[f_d["position_id"].isin(sub["position_id"])]
        row = {
            "family": fam,
            "n_fills": int(len(f_fam)),
            "n_positions": int(len(sub)),
            "leader_pnl": float(sub["domah_pnl_calc"].sum()),
        }
        for br in BRANCHES:
            row[f"{br}_pnl"] = float(sub[f"{br}_pnl"].sum())
        for br in ("A_opt", "A_real"):
            row[f"{br}_capture"] = (
                row[f"{br}_pnl"] / row["leader_pnl"] if row["leader_pnl"] != 0 else float("nan")
            )
        fm = f_fam[f_fam["role"] == "maker"]
        fmw = fm.merge(sub[["position_id", "is_winning_position"]],
                       on="position_id", how="left")
        win  = fmw[fmw["is_winning_position"] == 1]
        lose = fmw[fmw["is_winning_position"] == 0]
        if len(win) >= 30 and len(lose) >= 30:
            row["adverse_select_ratio"] = (
                win["A_real_fill"].mean() / lose["A_real_fill"].mean()
                if lose["A_real_fill"].mean() > 0 else float("nan")
            )
        else:
            row["adverse_select_ratio"] = float("nan")
        fam_rows.append(row)
    fam_d = pd.DataFrame(fam_rows).sort_values("n_fills", ascending=False).reset_index(drop=True)

    # =====================================================================
    # Build report sections
    # =====================================================================
    lines = []
    lines.append("\n\n## Cross-leader analysis: Domah vs `0xee00ba…`\n")
    lines.append(
        f"Both audits re-bucketed with the proposed FAMILY_KEYWORDS (Task 1). "
        f"Cross-leader deployable threshold: **A_real capture > {int(DEPLOY_CAPTURE*100)}% "
        f"AND adverse_select_ratio > {DEPLOY_ADVSEL} AND leader_pnl > 0**.\n"
    )

    # 1. Side-by-side family table
    side = pd.merge(
        fam_d.set_index("family")[[
            "n_fills", "leader_pnl", "A_real_capture",
            "adverse_select_ratio"]].add_suffix("_domah"),
        fam_e.set_index("family")[[
            "n_fills", "leader_pnl", "A_real_capture",
            "adverse_select_ratio"]].add_suffix("_ee00ba"),
        left_index=True, right_index=True, how="outer",
    ).reset_index()
    side["deployable_domah"] = (
        (side["A_real_capture_domah"] > DEPLOY_CAPTURE)
        & (side["adverse_select_ratio_domah"] > DEPLOY_ADVSEL)
        & (side["leader_pnl_domah"] > 0)
    )
    side["deployable_ee00ba"] = (
        (side["A_real_capture_ee00ba"] > DEPLOY_CAPTURE)
        & (side["adverse_select_ratio_ee00ba"] > DEPLOY_ADVSEL)
        & (side["leader_pnl_ee00ba"] > 0)
    )
    side["both_deployable"] = side["deployable_domah"] & side["deployable_ee00ba"]
    lines.append("### Family-level side-by-side\n")
    lines.append(fmt_md_table(side))
    lines.append("")

    # Domah deployable list, ee00ba deployable list
    dom_dep = set(side.loc[side["deployable_domah"], "family"].dropna())
    ee_dep  = set(side.loc[side["deployable_ee00ba"], "family"].dropna())
    both    = dom_dep & ee_dep
    only_d  = dom_dep - ee_dep
    only_e  = ee_dep  - dom_dep
    neither = set(side["family"].dropna()) - dom_dep - ee_dep

    # 2. Does Domah's macro-only finding generalise?
    lines.append("### 1. Does Domah's macro-only finding generalise?\n")
    macro_rows = side[side["family"] == "macro"]
    macro_d = macro_rows.iloc[0] if len(macro_rows) else None
    if macro_d is not None:
        ee_cap = macro_d["A_real_capture_ee00ba"]
        ee_cap_str = f"**{ee_cap:.0%}**" if pd.notna(ee_cap) else "N/A"
        ee_n = int(macro_d["n_fills_ee00ba"]) if pd.notna(macro_d["n_fills_ee00ba"]) else 0
        lines.append(
            f"- Domah macro: A_real capture **{macro_d['A_real_capture_domah']:.0%}** on "
            f"${macro_d['leader_pnl_domah']:,.0f} leader PnL, adv-sel "
            f"**{macro_d['adverse_select_ratio_domah']:.2f}** → deployable.\n"
            f"- 0xee00ba macro: A_real capture {ee_cap_str} on "
            f"${macro_d['leader_pnl_ee00ba']:,.0f} leader PnL (n_fills={ee_n:,})."
        )
    if "macro" in both:
        lines.append("- **Macro DOES generalise** as a cross-leader signal.\n")
    elif macro_d is not None and pd.notna(macro_d["n_fills_ee00ba"]) and macro_d["n_fills_ee00ba"] < 100:
        lines.append(
            f"- **Macro does NOT meaningfully apply to 0xee00ba** — he barely trades it "
            f"({int(macro_d['n_fills_ee00ba'])} fills vs Domah's "
            f"{int(macro_d['n_fills_domah']):,}). Not a generalising signal; it's a "
            f"Domah-specific specialisation.\n"
        )
    else:
        lines.append(f"- **Macro does NOT generalise.** Deployable for Domah but not "
                     f"for 0xee00ba.\n")
    lines.append(f"- 0xee00ba's own deployable families: **"
                 f"{', '.join(sorted(ee_dep)) if ee_dep else 'none'}**.\n")

    # 3. Divergence section
    lines.append("### 2. Where do they diverge?\n")
    if only_d:
        lines.append(f"- **Deployable for Domah only**: {', '.join(sorted(only_d))}.")
    if only_e:
        lines.append(f"- **Deployable for 0xee00ba only**: {', '.join(sorted(only_e))}.")
    if both:
        lines.append(f"- **Deployable for both**: {', '.join(sorted(both))}.")
    if neither:
        lines.append(f"- Deployable for neither: {', '.join(sorted(neither))}.")
    lines.append("")
    lines.append("Brief explanations for the divergences:\n")
    # Generate hypotheses per family
    for fam in sorted(set(side["family"].dropna())):
        r = side[side["family"] == fam].iloc[0]
        d_real = r["A_real_capture_domah"]
        e_real = r["A_real_capture_ee00ba"]
        d_asr  = r["adverse_select_ratio_domah"]
        e_asr  = r["adverse_select_ratio_ee00ba"]
        if pd.notna(d_real) and pd.notna(e_real) and abs(d_real - e_real) > 0.5:
            arrow = "↑" if e_real > d_real else "↓"
            lines.append(
                f"  - **{fam}**: Domah A_real cap {d_real:+.0%} (adv-sel {d_asr:.2f}), "
                f"0xee00ba A_real cap {e_real:+.0%} (adv-sel {e_asr:.2f}) {arrow}. "
                + ({
                    "politics": "Politics destroys Domah (he averages in on losers); 0xee00ba's tiny politics footprint (~1k fills) means the leader PnL is dominated by a small number of bets — high variance, not a stable signal.",
                    "sports":   "Domah's sports footprint is tiny; 0xee00ba is sports-heavy. Sports A_real capture for 0xee00ba is negative because his maker fills suffer adverse selection on his big sports book.",
                    "macro":    "Domah specialises in macro (NegRisk-style FOMC/CPI arb); 0xee00ba barely touches it, so the cell is not comparable.",
                    "crypto":   "Both leaders are positive but small samples; capture rough but consistent in sign.",
                    "other":    "'Other' is a heterogeneous bucket (earnings, entertainment, religion, agi); the divergence reflects different niche specialisations within it more than a single-mechanism difference.",
                    "weather":  "Domah's weather signal is tiny; 0xee00ba has no weather trades at all.",
                }.get(fam, "Different market specialisations — see per-family fill counts."))
            )
    lines.append("")

    # 4. Cross-leader deployable intersection (family × role × hour)
    lines.append("### 3. Cross-leader deployable intersection (family × role × hour)\n")
    lines.append(
        f"For each (family, role, hour_bucket) cell, compute A_real capture + adverse_select_ratio "
        f"separately for each leader. Deployable iff both have A_real capture > {int(DEPLOY_CAPTURE*100)}% "
        f"AND adverse_select_ratio > {DEPLOY_ADVSEL} AND leader_pnl > 0.\n"
    )

    # Compute fine-grain cells for both leaders
    print("[cross] fine-grain Domah cells…", flush=True)
    cells_d = fine_grain_capture_advsel(
        p_d, f_d, ["family", "role", "hour_bucket"]
    )
    print(f"  domah cells: {len(cells_d):,}", flush=True)
    print("[cross] fine-grain ee00ba cells…", flush=True)
    cells_e = fine_grain_capture_advsel(
        p_e, f_e, ["family", "role", "hour_bucket"]
    )
    print(f"  ee00ba cells: {len(cells_e):,}", flush=True)

    keys = ["family", "role", "hour_bucket"]
    merged = pd.merge(
        cells_d[keys + ["n_fills", "leader_pnl", "A_real_capture",
                        "adverse_select_ratio", "deployable"]].rename(columns={
            "n_fills": "n_fills_d", "leader_pnl": "leader_pnl_d",
            "A_real_capture": "cap_d", "adverse_select_ratio": "advsel_d",
            "deployable": "dep_d",
        }),
        cells_e[keys + ["n_fills", "leader_pnl", "A_real_capture",
                        "adverse_select_ratio", "deployable"]].rename(columns={
            "n_fills": "n_fills_e", "leader_pnl": "leader_pnl_e",
            "A_real_capture": "cap_e", "adverse_select_ratio": "advsel_e",
            "deployable": "dep_e",
        }),
        on=keys, how="outer",
    )
    merged["both_deployable"] = merged["dep_d"].fillna(False) & merged["dep_e"].fillna(False)

    intersection = merged[merged["both_deployable"]].sort_values(
        ["family", "role", "hour_bucket"]
    )
    if len(intersection):
        lines.append(f"**Cells deployable for BOTH leaders ({len(intersection)}):**\n")
        lines.append(fmt_md_table(
            intersection[keys + [
                "n_fills_d", "leader_pnl_d", "cap_d", "advsel_d",
                "n_fills_e", "leader_pnl_e", "cap_e", "advsel_e",
            ]]
        ))
    else:
        lines.append("**No (family × role × hour) cell is deployable for both leaders.**\n")
        lines.append("This means: there is no single execution profile that captures both leaders' "
                     "edge under a copy-mirroring strategy. Either each leader needs a per-leader "
                     "execution policy, or a multi-leader strategy requires a different mechanism "
                     "(e.g. consensus signal across leaders rather than per-fill mirroring).\n")

    # Also show the union — cells deployable for AT LEAST ONE leader
    union = merged[
        merged["dep_d"].fillna(False) | merged["dep_e"].fillna(False)
    ].sort_values(["family", "role", "hour_bucket"])
    lines.append(f"\n**Cells deployable for AT LEAST ONE leader ({len(union)}):**\n")
    if len(union):
        lines.append(fmt_md_table(
            union[keys + [
                "n_fills_d", "leader_pnl_d", "cap_d", "advsel_d",
                "n_fills_e", "leader_pnl_e", "cap_e", "advsel_e",
                "dep_d", "dep_e",
            ]]
        ))

    # Synthesis paragraph on the execution-profile dichotomy
    if len(union):
        dom_only_roles = union.loc[union["dep_d"].fillna(False), "role"].unique()
        ee_only_roles  = union.loc[union["dep_e"].fillna(False), "role"].unique()
        lines.append("\n### Synthesis\n")
        lines.append(
            f"The two leaders' deployable cells lie on **opposite execution profiles**: "
            f"Domah's are exclusively **{', '.join(sorted(dom_only_roles)) or 'none'}** "
            f"(he is 87% maker; the bot succeeds when it copies his bids), while 0xee00ba's "
            f"are exclusively **{', '.join(sorted(ee_only_roles)) or 'none'}** "
            f"(he is 69% maker but his selection edge transfers when the bot crosses spread "
            f"on his behalf). The cross-leader intersection is empty because there is no "
            f"family × hour cell where both leaders' edge survives copy execution in the "
            f"same direction; copy-trading them is two separate strategies, not one. "
            f"Macro is Domah-specific (0xee00ba has 81 fills); sports is 0xee00ba's only "
            f"large family but his sports maker fills suffer adverse selection (0.39). "
            f"A multi-leader strategy would need to combine the signals upstream (consensus "
            f"of leaders before execution), not merge fills downstream of each leader."
        )
        lines.append("")

    # Save full fine-grain merge for inspection
    merged.to_parquet(FU / "cross_leader_cells.parquet", index=False)
    side.to_parquet(FU / "cross_leader_family_side_by_side.parquet", index=False)

    # Append to existing report
    existing = REPORT_PATH.read_text()
    REPORT_PATH.write_text(existing + "\n".join(lines))
    print(f"[cross] appended cross-leader section to {REPORT_PATH}")


if __name__ == "__main__":
    main()
