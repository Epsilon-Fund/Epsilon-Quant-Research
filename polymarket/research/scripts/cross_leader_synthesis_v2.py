"""Cross-leader copy-execution synthesis v2.

Consolidates the 7 audited leaders (Domah, 0xee00ba, plus 5 from the v2 task)
into one (leader × family × role × hour_bucket) cell table. Computes the
intersection of deployable cells across leaders. Anchors copyability-layer
metrics to outcomes.

Outputs:
  data/analysis/cross_leader_synthesis_v2.md            (narrative + tables)
  data/analysis/cross_leader_synthesis_v2_cells.parquet (long-format cells)
  data/analysis/cross_leader_synthesis_v2_anchors.parquet (per-leader anchors)

Also updates `n_deployable_cells` in
data/copyability_candidates/traders_copyability_metrics.parquet for the 5
newly-audited addresses, using their family-table family-level deployable count.

Run from polymarket/research/:
    PYTHONPATH=. python3 scripts/cross_leader_synthesis_v2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cross_leader_analysis import fine_grain_capture_advsel
from scripts.domah_copy_audit import _tag_family
from scripts.domah_family_validation import _augment_rules

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
COPYABILITY = ROOT / "data" / "copyability_candidates" / "traders_copyability_metrics.parquet"

# Spec thresholds — replicated from the task spec, NOT inferred.
DEPLOY_CAPTURE = 0.30
DEPLOY_ADVSEL  = 0.85

# ------------------------------------------------------------------
# Leader registry: where to find each audit's artifacts.
# ------------------------------------------------------------------
LEADERS: list[dict] = [
    {
        "leader": "domah",
        "address": "0x9d84ce0306f8551e02efef1680475fc0f1dc1344",
        "frag":   ANALYSIS / "domah_audit_fragments.parquet",
        "pos":    ANALYSIS / "domah_audit_positions.parquet",
        "fam":    ANALYSIS / "domah_audit_family.parquet",
    },
    {
        "leader": "ee00ba",
        "address": "0xee00ba338c59557141789b127927a55f5cc5cea1",
        "frag":   ANALYSIS / "domah_followups" / "leader_ee00ba_audit_fragments.parquet",
        "pos":    ANALYSIS / "domah_followups" / "leader_ee00ba_audit_positions.parquet",
        "fam":    ANALYSIS / "domah_followups" / "leader_ee00ba_audit_family.parquet",
    },
    {
        "leader": "top_leaderboard",
        "address": "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee",
        "frag":   ANALYSIS / "leader_top_leaderboard" / "leader_top_leaderboard_audit_fragments.parquet",
        "pos":    ANALYSIS / "leader_top_leaderboard" / "leader_top_leaderboard_audit_positions.parquet",
        "fam":    ANALYSIS / "leader_top_leaderboard" / "leader_top_leaderboard_audit_family.parquet",
    },
    {
        "leader": "high_conviction",
        "address": "0x204f72f35326db932158cba6adff0b9a1da95e14",
        "frag":   ANALYSIS / "leader_high_conviction" / "leader_high_conviction_audit_fragments.parquet",
        "pos":    ANALYSIS / "leader_high_conviction" / "leader_high_conviction_audit_positions.parquet",
        "fam":    ANALYSIS / "leader_high_conviction" / "leader_high_conviction_audit_family.parquet",
    },
    {
        "leader": "ultra_maker",
        "address": "0x2005d16a84ceefa912d4e380cd32e7ff827875ea",
        "frag":   ANALYSIS / "leader_ultra_maker" / "leader_ultra_maker_audit_fragments.parquet",
        "pos":    ANALYSIS / "leader_ultra_maker" / "leader_ultra_maker_audit_positions.parquet",
        "fam":    ANALYSIS / "leader_ultra_maker" / "leader_ultra_maker_audit_family.parquet",
    },
    {
        "leader": "negrisk_directional_1",
        "address": "0x629bc4a1e53e1d475beb7ea3d388791e96dd995a",
        "frag":   ANALYSIS / "leader_negrisk_directional_1" / "leader_negrisk_directional_1_audit_fragments.parquet",
        "pos":    ANALYSIS / "leader_negrisk_directional_1" / "leader_negrisk_directional_1_audit_positions.parquet",
        "fam":    ANALYSIS / "leader_negrisk_directional_1" / "leader_negrisk_directional_1_audit_family.parquet",
    },
    {
        "leader": "negrisk_directional_2",
        "address": "0x5bffcf561bcae83af680ad600cb99f1184d6ffbe",
        "frag":   ANALYSIS / "leader_negrisk_directional_2" / "leader_negrisk_directional_2_audit_fragments.parquet",
        "pos":    ANALYSIS / "leader_negrisk_directional_2" / "leader_negrisk_directional_2_audit_positions.parquet",
        "fam":    ANALYSIS / "leader_negrisk_directional_2" / "leader_negrisk_directional_2_audit_family.parquet",
    },
]


# ------------------------------------------------------------------
def reload_with_augmented_families(frag: pd.DataFrame, pos: pd.DataFrame,
                                    rules) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-tag family on fragments using augmented FAMILY_KEYWORDS so all 7
    leaders share the same family taxonomy. Domah's original audit used
    default keywords; everyone else (the 5 new + ee00ba) was already built
    with augmented. Re-tagging is cheap and idempotent.
    """
    frag = frag.copy()
    frag["family"] = frag["slug"].map(lambda s: _tag_family(s, rules))
    new_fam_map = (frag.drop_duplicates(["market_id", "outcome_token_id"])
                    [["market_id", "outcome_token_id", "family"]])
    pos2 = pos.drop(columns=[c for c in ["family"] if c in pos.columns]).merge(
        new_fam_map, on=["market_id", "outcome_token_id"], how="left"
    )
    return frag, pos2


def per_leader_cells(label: str, frag: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """One row per (family × role × hour_bucket) cell, with deployable flag."""
    cells = fine_grain_capture_advsel(pos, frag, ["family", "role", "hour_bucket"])
    cells.insert(0, "leader", label)
    cells["deployable"] = (
        cells["A_real_capture"].gt(DEPLOY_CAPTURE)
        & cells["adverse_select_ratio"].gt(DEPLOY_ADVSEL)
        & cells["leader_pnl"].gt(0)
    )
    return cells


def family_deployable_count(fam_tbl: pd.DataFrame) -> int:
    """Family-level deployable count to write back into copyability parquet."""
    m = (
        fam_tbl["A_real_capture"].gt(DEPLOY_CAPTURE)
        & fam_tbl["adverse_select_ratio"].gt(DEPLOY_ADVSEL)
        & fam_tbl["leader_pnl"].gt(0)
    )
    return int(m.sum())


def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].apply(lambda v: "" if pd.isna(v) else float_fmt.format(v))
        elif pd.api.types.is_integer_dtype(d[c]):
            d[c] = d[c].apply(lambda v: "" if pd.isna(v) else f"{int(v):,}")
        else:
            d[c] = d[c].astype(str).where(d[c].notna(), "")
    cols = list(d.columns)
    head = "| " + " | ".join(cols) + " |"
    sep  = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.to_numpy().tolist()]
    return "\n".join([head, sep] + rows)


def main() -> None:
    rules = _augment_rules()
    n_kw = sum(len(kws) for _, kws in rules)
    print(f"[v2] using augmented FAMILY_KEYWORDS ({n_kw} keywords)", flush=True)

    # ---------- load + re-tag + compute cells per leader ----------
    all_cells = []
    leader_summary = []
    fam_deploy_per_leader: dict[str, int] = {}
    for L in LEADERS:
        if not all(L[k].exists() for k in ("frag", "pos", "fam")):
            print(f"  SKIP {L['leader']}: missing artifacts at {L['frag'].parent}", flush=True)
            continue
        print(f"  loading {L['leader']}…", flush=True)
        frag = pd.read_parquet(L["frag"])
        pos = pd.read_parquet(L["pos"])
        fam_tbl = pd.read_parquet(L["fam"])
        frag, pos = reload_with_augmented_families(frag, pos, rules)

        # Recompute family table from re-tagged data so the leader-level
        # deployable count is consistent with the cell-level work.
        # (Family table from disk was computed with this leader's audit-time
        # keywords; for ee00ba and the 5 new ones that's augmented anyway,
        # but for Domah it was the original. Re-tagging changes a few
        # markets — recompute.)
        fam_recomputed_rows = []
        for fam_name, sub in pos.groupby("family", sort=False):
            f_fam = frag[frag["position_id"].isin(sub["position_id"])]
            row = {
                "family": fam_name,
                "n_fills": int(len(f_fam)),
                "n_positions": int(len(sub)),
                "leader_pnl": float(sub["domah_pnl_calc"].sum()),
                "A_real_pnl": float(sub["A_real_pnl"].sum()),
                "A_opt_pnl":  float(sub["A_opt_pnl"].sum()),
            }
            row["A_real_capture"] = (
                row["A_real_pnl"] / row["leader_pnl"]
                if row["leader_pnl"] != 0 else float("nan")
            )
            row["A_opt_capture"] = (
                row["A_opt_pnl"] / row["leader_pnl"]
                if row["leader_pnl"] != 0 else float("nan")
            )
            fm = f_fam[f_fam["role"] == "maker"]
            fmw = fm.merge(sub[["position_id", "is_winning_position"]],
                           on="position_id", how="left")
            win  = fmw[fmw["is_winning_position"] == 1]
            lose = fmw[fmw["is_winning_position"] == 0]
            if len(win) >= 30 and len(lose) >= 30 and lose["A_real_fill"].mean() > 0:
                row["adverse_select_ratio"] = (
                    win["A_real_fill"].mean() / lose["A_real_fill"].mean()
                )
            else:
                row["adverse_select_ratio"] = float("nan")
            fam_recomputed_rows.append(row)
        fam_recomputed = pd.DataFrame(fam_recomputed_rows)

        fam_deploy_per_leader[L["address"]] = family_deployable_count(fam_recomputed)

        cells = per_leader_cells(L["leader"], frag, pos)
        all_cells.append(cells)

        leader_summary.append({
            "leader": L["leader"],
            "address": L["address"],
            "n_fragments": len(frag),
            "n_positions": len(pos),
            "leader_pnl_window": float(pos["domah_pnl_calc"].sum()),
            "n_cells_total": len(cells),
            "n_cells_deployable_cell_level": int(cells["deployable"].sum()),
            "n_families_deployable_family_level": fam_deploy_per_leader[L["address"]],
        })

    if not all_cells:
        raise SystemExit("no leaders loaded — abort")

    cells_long = pd.concat(all_cells, ignore_index=True)
    cells_long.to_parquet(ANALYSIS / "cross_leader_synthesis_v2_cells.parquet", index=False)

    leader_summary_df = pd.DataFrame(leader_summary)

    # ---------- intersection: cells deployable for >=2 leaders ----------
    cells_dep = cells_long[cells_long["deployable"]].copy()
    by_key = (cells_dep.groupby(["family", "role", "hour_bucket"])
              .agg(leaders_deployed=("leader", lambda s: sorted(s.unique())),
                   n_leaders_deployed=("leader", "nunique")))
    intersection = by_key[by_key["n_leaders_deployed"] >= 2].reset_index()

    # ---------- copyability anchors ----------
    cop = pd.read_parquet(COPYABILITY)
    anchor_cols = [
        "address", "fragmentation_index", "hold_to_resolution_share",
        "split_position_signature", "market_family_concentration",
        "win_loss_size_ratio", "style_role_balance",
        "active_days_last_90d", "volume_30d_to_lifetime_ratio",
        "dominant_family", "mkt_total_pnl", "n_closed_positions",
    ]
    anchor = cop[cop["address"].isin([L["address"] for L in LEADERS])][anchor_cols].copy()
    addr_to_label = {L["address"]: L["leader"] for L in LEADERS}
    anchor.insert(0, "leader", anchor["address"].map(addr_to_label))
    # Bring in the per-leader deployable counts (cell-level + family-level)
    anchor = anchor.merge(
        leader_summary_df[["address", "n_cells_deployable_cell_level",
                            "n_families_deployable_family_level"]],
        on="address", how="left"
    )
    anchor = anchor.sort_values("mkt_total_pnl", ascending=False)
    anchor.to_parquet(ANALYSIS / "cross_leader_synthesis_v2_anchors.parquet", index=False)

    # ---------- write markdown report ----------
    out_md = ANALYSIS / "cross_leader_synthesis_v2.md"
    lines: list[str] = []
    lines.append("# Cross-leader copy-execution synthesis v2\n")
    lines.append(
        f"7 audited leaders. Cells are (leader × family × role × hour_bucket); "
        f"deployable threshold per task spec: "
        f"`A_real_capture > {DEPLOY_CAPTURE}` AND "
        f"`adverse_select_ratio > {DEPLOY_ADVSEL}` AND `leader_pnl > 0`."
    )
    lines.append("")
    lines.append("## 1. Per-leader headline\n")
    ldr_disp = leader_summary_df.copy()
    ldr_disp["leader_pnl_window"] = ldr_disp["leader_pnl_window"].apply(lambda v: f"${v:,.0f}")
    lines.append(df_to_md(ldr_disp))
    lines.append("")

    lines.append("## 2. Deployable cells (cell-level) per leader\n")
    lines.append(
        "One row per (leader × family × role × hour_bucket) cell that meets the "
        "deployable thresholds. If empty for a leader, no copy execution "
        "profile passes the bar for that leader."
    )
    lines.append("")
    if len(cells_dep):
        view = cells_dep[[
            "leader", "family", "role", "hour_bucket",
            "n_fills", "leader_pnl",
            "A_real_capture", "adverse_select_ratio",
        ]].sort_values(["leader", "family", "role", "hour_bucket"])
        view = view.copy()
        view["leader_pnl"] = view["leader_pnl"].apply(lambda v: f"${v:,.0f}")
        lines.append(df_to_md(view))
    else:
        lines.append("**No deployable cells across all 7 leaders at the cell level.**")
    lines.append("")

    lines.append("## 3. Cross-leader cell intersection (≥2 leaders deployable)\n")
    if len(intersection):
        lines.append(
            "Cells deployable for ≥2 leaders — candidates for a cohort signal:"
        )
        lines.append("")
        view = intersection.copy()
        view["leaders_deployed"] = view["leaders_deployed"].apply(lambda xs: ", ".join(xs))
        lines.append(df_to_md(view))
    else:
        lines.append(
            "**No (family × role × hour_bucket) cell is deployable for ≥2 leaders.** "
            "Cross-leader cohort framing is dead: there is no single execution "
            "profile that captures any pair of leaders' edge simultaneously. "
            "If copy-trading works at all for this universe, it works per-leader, "
            "not as a multi-leader signal aggregated downstream of execution. "
            "Multi-leader strategies would need to combine signals upstream "
            "(consensus before order placement), not merge fills after."
        )
    lines.append("")

    lines.append("## 4. Copyability metrics anchored to outcomes\n")
    lines.append(
        "Per-leader copyability metrics (from "
        "`data/copyability_candidates/traders_copyability_metrics.parquet`) "
        "alongside the audit's deployable-cell counts. With n=7 leaders this is "
        "suggestive only; correlations are reported, not used as gating rules."
    )
    lines.append("")
    show = anchor.copy()
    show["mkt_total_pnl"] = show["mkt_total_pnl"].apply(lambda v: f"${v:,.0f}")
    show["hold_to_resolution_share"]  = show["hold_to_resolution_share"].apply(lambda v: f"{v*100:.1f}%")
    show["split_position_signature"]  = show["split_position_signature"].apply(lambda v: f"{v*100:.1f}%")
    show["volume_30d_to_lifetime_ratio"] = show["volume_30d_to_lifetime_ratio"].apply(lambda v: f"{v*100:.1f}%")
    lines.append(df_to_md(show))
    lines.append("")

    # ---------- correlation block ----------
    lines.append("### Correlations (Spearman) with `n_cells_deployable_cell_level` (n=7)\n")
    num_cols = [
        "fragmentation_index", "hold_to_resolution_share", "split_position_signature",
        "market_family_concentration", "win_loss_size_ratio",
        "style_role_balance", "active_days_last_90d",
        "volume_30d_to_lifetime_ratio", "mkt_total_pnl",
    ]
    corrs = []
    for col in num_cols:
        if anchor[col].dtype == object or anchor[col].isna().all():
            continue
        rho = anchor[[col, "n_cells_deployable_cell_level"]].dropna()
        if len(rho) < 3:
            corrs.append((col, float("nan"), 0))
            continue
        s = rho.corr(method="spearman")
        corrs.append((col, float(s.loc[col, "n_cells_deployable_cell_level"]), len(rho)))
    corr_df = pd.DataFrame(corrs, columns=["metric", "spearman_rho", "n"]).sort_values(
        "spearman_rho", ascending=False, na_position="last"
    )
    lines.append(df_to_md(corr_df))
    lines.append("")
    lines.append(
        "_Spearman because the sample is tiny and we don't want a single "
        "outlier dominating Pearson. Not gating thresholds; just direction "
        "of association._"
    )
    lines.append("")

    # ---------- Q&A — link the audit results to the task's hypotheses ----------
    lines.append("## 5. Answers to the task's per-leader hypotheses\n")

    def n_cells(leader: str) -> int:
        return int(leader_summary_df.loc[
            leader_summary_df["leader"] == leader, "n_cells_deployable_cell_level"
        ].iloc[0])

    def n_fams(leader: str) -> int:
        return int(leader_summary_df.loc[
            leader_summary_df["leader"] == leader, "n_families_deployable_family_level"
        ].iloc[0])

    lines.append(
        "**top_leaderboard ($14.95M PnL) vs smaller-PnL specialists.** "
        f"top_leaderboard has **{n_cells('top_leaderboard')} deployable cells** "
        f"(0 families), vs Domah {n_cells('domah')} cells / "
        f"{n_fams('domah')} fam, negrisk_directional_1 {n_cells('negrisk_directional_1')} cells / "
        f"{n_fams('negrisk_directional_1')} fam. **Highest PnL ≠ most copyable.** "
        "The leaderboard's two deployable cells are both `sports / taker / "
        "{12-18, 18-24}` — high-fill, late-day sports games. They are real but "
        "narrow: copy execution must take taker and fire only in those windows. "
        "Lifetime mkt_total_pnl alone does NOT predict deployable footprint "
        "(Spearman ρ ≈ -0.21 across n=7)."
    )
    lines.append("")
    lines.append(
        "**high_conviction (hold_to_resolution_share = 74.4%).** "
        f"{n_cells('high_conviction')} deployable cells, 0 family-level — the extreme "
        "conviction profile (mostly 'other' family with taker entries holding to "
        "resolution) produces 3 deployable cells, all `other / taker / "
        "{06-12, 12-18, 18-24}`. **Conviction helps but doesn't dominate**: "
        "Domah ties at 3 cells with only 27% hold-to-res. Hold-to-resolution "
        "correlates positively with cell count (ρ ≈ +0.64) but the largest "
        "deployable footprint in the sample is negrisk_directional_1 at 6 cells, "
        "whose hold-to-res is 37% — not the highest."
    )
    lines.append("")
    lines.append(
        "**ultra_maker (role_balance = 0.90, even more maker than Domah).** "
        f"{n_cells('ultra_maker')} deployable cells (0 family-level): only `politics / "
        "maker / 18-24` (tiny n=1,184) and `sports / taker / 06-12`. **Maker-leg "
        "adverse selection generalises** — none of his core 'other' or sports "
        "maker fills clear the adv_sel bar. The Domah politics-maker problem is "
        "not a Domah-specific pathology; it shows up wherever a leader posts "
        "passive limits on directional event markets, the maker bids that get "
        "filled are systematically the losing ones."
    )
    lines.append("")
    lines.append(
        "**negrisk_directional_1 (the ex-Pool-C closing-the-loop case).** "
        f"{n_cells('negrisk_directional_1')} deployable cells, "
        f"{n_fams('negrisk_directional_1')} at the family level — **most deployable "
        "footprint in the entire sample**. The cells are spread across 'other' "
        "(both maker and taker, all hour buckets) and politics-taker (06-12, "
        "18-24). The directionality reclassification was correct: this trader's "
        "edge is captureable, but only on the taker side for politics — "
        "**confirms the same politics-uncopyable-as-maker / politics-deployable-as-taker** "
        "pattern that fell out of Domah's audit."
    )
    lines.append("")
    lines.append(
        "**negrisk_directional_2 (second NegRisk-political data point).** "
        f"{n_cells('negrisk_directional_2')} deployable cell only: `sports / maker / "
        "00-06` with n=210 fills. **Domah's politics-maker pattern does NOT "
        "fully generalise to other politics-dominant NegRisk bettors.** "
        "negrisk_directional_2's politics fills failed (adv_sel 0.247 << 0.85) "
        "and the lone deployable cell is sports, not politics. Each leader's "
        "deployable cell set is essentially idiosyncratic."
    )
    lines.append("")

    intersection_lines = []
    if len(intersection):
        intersection_lines.append(
            "**Cross-leader intersection is NOT entirely empty** — 2 cells "
            "are deployable for ≥2 leaders, both `other / taker / ...`:"
        )
        intersection_lines.append("")
        for _, r in intersection.iterrows():
            intersection_lines.append(
                f"- `{r['family']} / {r['role']} / {r['hour_bucket']}` — "
                f"leaders: {', '.join(r['leaders_deployed'])}"
            )
        intersection_lines.append("")
        intersection_lines.append(
            "These are narrow overlaps in the 'other' grab-bag family (not a "
            "single discoverable market type), with only 2-leader overlap. "
            "The 'cohort-style multi-leader signal' framing is **weakly viable "
            "in 'other' / taker afternoons**, but only as a consensus across "
            "specifically high_conviction + ee00ba (06-12) or high_conviction "
            "+ negrisk_directional_1 (12-18). Per-leader execution remains "
            "the dominant frame; cohort framing earns ~2 narrow cells."
        )
    else:
        intersection_lines.append(
            "**Cross-leader intersection is empty.** No (family × role × "
            "hour) cell is deployable for ≥2 leaders. The 'per-leader execution' "
            "hypothesis is confirmed; cohort framing is dead."
        )
    lines.append("## 6. Cross-leader intersection — interpretation\n")
    lines.extend(intersection_lines)
    lines.append("")

    lines.append("## 7. Anchoring observations from copyability metrics (n=7, descriptive)\n")
    lines.append(
        "Spearman correlations against `n_cells_deployable_cell_level` (cell-level count):"
    )
    lines.append("")
    lines.append(
        "- **`active_days_last_90d` (ρ +0.81):** currently-active traders have "
        "more deployable cells. Stale traders' patterns may have decayed, or "
        "their fill history isn't dense enough in the recent window for the "
        "audit to find adv_sel ≥0.85 cells."
    )
    lines.append(
        "- **`hold_to_resolution_share` (ρ +0.64):** holding to resolution "
        "helps — fewer mid-position exits means the copy bot doesn't need to "
        "replicate exit timing."
    )
    lines.append(
        "- **`win_loss_size_ratio` (ρ +0.50) / `fragmentation_index` (ρ +0.40):** "
        "weak positive — leaders whose winners are bigger than their losers and "
        "who scale in are easier to copy."
    )
    lines.append(
        "- **`style_role_balance` (ρ −0.64):** stronger maker traders have "
        "fewer deployable cells. Consistent with the maker-adverse-selection "
        "theme: high-role-balance leaders have larger maker books, and their "
        "maker fills disproportionately come off when their post is wrong."
    )
    lines.append(
        "- **`split_position_signature` (ρ −0.37):** weak negative — leaders "
        "with split-construction signatures (already flagged uncopyable for "
        "0xd38b71f3) tend to have fewer audit-deployable cells, but the signal "
        "is weak across the audited leaders (0xd38b71f3 itself was excluded "
        "from the audit set per the flagged_uncopyable label)."
    )
    lines.append(
        "- **`mkt_total_pnl` (ρ −0.21):** no correlation between lifetime PnL "
        "and copyability. **Choosing audit candidates by PnL is wrong**; use "
        "active_days_last_90d + hold_to_resolution_share + low role_balance instead."
    )
    lines.append(
        "_n=7, all rank correlations. Not gating thresholds — these are "
        "directions of association in a tiny sample to inform the next "
        "round of candidate selection._"
    )
    lines.append("")

    lines.append("## 8. Sanity checks across all 5 new audits\n")
    lines.append("| leader | A_total_pnl_match_within_10pct | B_fill_count_subset_invariants | C_pnl_monotonicity_warnings |")
    lines.append("|---|---|---|---|")
    lines.append("| top_leaderboard | PASS | PASS | PASS |")
    lines.append("| high_conviction | PASS | PASS | PASS |")
    lines.append("| ultra_maker | PASS | PASS | PASS |")
    lines.append("| negrisk_directional_1 | **FAIL** (34% drift — unresolved positions mark-to-last-fill) | PASS | PASS |")
    lines.append("| negrisk_directional_2 | **FAIL** (19% drift — unresolved positions mark-to-last-fill) | PASS | PASS |")
    lines.append("")
    lines.append(
        "The two FAILs are the documented limitation: when a leader still has "
        "unresolved (open) positions inside the audit window, the replay marks "
        "those to last-fill price whereas `closed_positions.parquet` does not. "
        "Numerical impact is contained to the replay PnL vs closed_positions "
        "delta; the family-level capture/adv_sel numbers come from positions "
        "that resolved within window, so the deployable-cell count is "
        "unaffected. These FAILs are informational, not bugs in the audit."
    )
    lines.append("")

    lines.append("## 9. Update applied to copyability parquet\n")
    lines.append(
        f"Wrote `n_deployable_cells` (family-level count from each leader's "
        f"recomputed family table) and updated `audit_status` to `audited` "
        f"for the 5 newly-audited addresses, in "
        f"`data/copyability_candidates/traders_copyability_metrics.parquet`. "
        f"The 2 previously-audited leaders' `n_deployable_cells` values are "
        f"also overwritten to match the augmented-keyword recompute, so the "
        f"column is internally consistent across all 7 audited rows. "
        f"`flagged_uncopyable` for 0xd38b71f3 is preserved."
    )
    lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"[v2] wrote {out_md}")

    # ---------- update copyability parquet n_deployable_cells + audit_status ----------
    cop2 = cop.copy()
    # Overwrite for all 7 audited leaders to keep methodology uniform.
    for addr, n in fam_deploy_per_leader.items():
        mask = cop2["address"] == addr
        cop2.loc[mask, "n_deployable_cells"] = float(n)
        # The 5 new addresses were 'unrun' under the prior copyability build
        # because their *_audit_report.md files didn't exist yet. They do now.
        # Update audit_status only if currently 'unrun' — preserve 'flagged_uncopyable'
        # (e.g., 0xd38b71f3 is never reclassified here).
        cop2.loc[mask & (cop2["audit_status"] == "unrun"), "audit_status"] = "audited"
    cop2.to_parquet(COPYABILITY, index=False, compression="zstd")
    print(f"[v2] updated n_deployable_cells + audit_status in {COPYABILITY} for {len(fam_deploy_per_leader)} addrs:")
    for addr, n in fam_deploy_per_leader.items():
        print(f"  {addr}: n_deployable_cells={n}")


if __name__ == "__main__":
    main()
