"""Copytrade attribution repartition — sub-task 1: active_order_leg style recompute.

Context: CTF Exchange `_matchOrders` emits the internal active leg of a
two-sided match with maker = takerOrder.maker and taker = address(this).
Any fill whose `taker` is one of the 4 exchange-internal-leg contracts is
therefore a row where the wallet in the `maker` column was the ACTIVE
(aggressor) order signer. PnL/position attribution is unaffected — this is
style framing only.

This script:
  1. Smoke-tests the new `active_order_leg` flag in sql/views.sql on a
     bounded (1-day) timestamp slice of trader_actions.
  2. Recomputes maker:taker style ratios for the audited copytrade leaders
     with internal-leg maker rows reclassified to the active/taker side,
     and reports old/new style categories (maker_heavy > 4, taker_heavy
     < 0.25, else mixed — matches scripts/copytrade_relayer_implications.py).
     Anchors (from the 2026-05-28 relayer-implications run, v1-only internal
     counts): Domah 7.89 -> 5.67, 0x6a72... 5.15 -> 3.46.
  3. Recomputes the Domah copy-audit role slice (maker vs taker sub-cell)
     from the cached audit fragments/positions parquets with the
     reclassified role, overall and within the macro family and the 18-24
     hour bucket (the smoke cell).

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/copytrade_attribution_repartition.py

Outputs:
    data/analysis/csv_outputs/copytrade/copytrade_style_reclass_leaders.csv
    data/analysis/csv_outputs/copytrade/copytrade_domah_role_slice_reclass.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from data_infra.operator_denylist import (
    EXCHANGE_INTERNAL_LEG,
    EXCHANGE_INTERNAL_LEG_V1,
)
from data_infra.views import load_views

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
OUT_DIR = ANALYSIS / "csv_outputs" / "copytrade"
OUT_STYLE = OUT_DIR / "copytrade_style_reclass_leaders.csv"
OUT_ROLE_SLICE = OUT_DIR / "copytrade_domah_role_slice_reclass.csv"

TRADERS_PATH = ROOT / "data" / "traders.parquet"
ESTIMATES_PATH = ANALYSIS / "copytrade_invisible_take_estimates.parquet"
FRAGMENTS_PATH = ANALYSIS / "domah_audit_fragments.parquet"
POSITIONS_PATH = ANALYSIS / "domah_audit_positions.parquet"

# Audited leaders: Domah + the audited-leader reports under data/analysis/
# + the relayer-implications Step 3 manual candidates.
LEADERS = {
    "0x9d84ce0306f8551e02efef1680475fc0f1dc1344": "domah",
    "0xee00ba338c59557141789b127927a55f5cc5cea1": "leader_ee00ba (domah_followups)",
    "0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029": "leader_dthreed8b71",
    "0x204f72f35326db932158cba6adff0b9a1da95e14": "leader_high_conviction",
    "0x629bc4a1e53e1d475beb7ea3d388791e96dd995a": "leader_negrisk_directional_1",
    "0x5bffcf561bcae83af680ad600cb99f1184d6ffbe": "leader_negrisk_directional_2",
    "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee": "leader_top_leaderboard",
    "0x2005d16a84ceefa912d4e380cd32e7ff827875ea": "leader_ultra_maker",
    "0x17db3fcd93ba12d38382a0cade24b200185c5f6d": "relayer_implications_candidate",
}

DOMAH = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"

# Style category thresholds — match copytrade_relayer_implications.py
# style_bucket_expr(): ratio > 4 maker_heavy, < 0.25 taker_heavy, else mixed.
MAKER_HEAVY_T = 4.0
TAKER_HEAVY_T = 0.25


def category(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "n/a"
    if ratio > MAKER_HEAVY_T:
        return "maker_heavy"
    if ratio < TAKER_HEAVY_T:
        return "taker_heavy"
    return "mixed"


def sql_list(addrs) -> str:
    return ", ".join(f"'{a}'" for a in sorted(addrs))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=6")
    con.execute("SET preserve_insertion_order=false")
    load_views(con)
    return con


# ---------------------------------------------------------------------------
# 1. Bounded smoke test of the new flag
# ---------------------------------------------------------------------------
def smoke_test_flag(con: duckdb.DuckDBPyConnection) -> None:
    log("smoke: active_order_leg counts on a bounded 1-day trader_actions slice")
    df = con.sql(
        """
        SELECT role, active_order_leg, count(*) AS n
        FROM trader_actions
        WHERE timestamp >= TIMESTAMP '2026-04-20 00:00:00'
          AND timestamp <  TIMESTAMP '2026-04-21 00:00:00'
        GROUP BY role, active_order_leg
        ORDER BY role, active_order_leg
        """
    ).fetchdf()
    print(df.to_string(index=False))
    # Consistency: maker-role active rows must equal taker-role artifact rows
    # (both count the same set of exchange-internal-leg fills).
    piv = df.set_index(["role", "active_order_leg"])["n"]
    maker_active = int(piv.get(("maker", True), 0))
    taker_artifact = int(piv.get(("taker", False), 0))
    assert maker_active == taker_artifact, (
        f"flag inconsistency: maker-role active={maker_active:,} "
        f"!= taker-role artifact={taker_artifact:,}"
    )
    # Artifact taker-role rows must all be exchange contracts.
    chk = con.sql(
        f"""
        SELECT count(*) AS n
        FROM trader_actions
        WHERE timestamp >= TIMESTAMP '2026-04-20 00:00:00'
          AND timestamp <  TIMESTAMP '2026-04-21 00:00:00'
          AND role = 'taker' AND NOT active_order_leg
          AND address NOT IN ({sql_list(EXCHANGE_INTERNAL_LEG)})
        """
    ).fetchone()[0]
    assert chk == 0, f"{chk} taker-role artifact rows are not exchange contracts"
    log(
        f"smoke OK: {maker_active:,} internal-leg fills on the slice; "
        "maker-role active == taker-role artifact; artifact rows are all exchange contracts"
    )


# ---------------------------------------------------------------------------
# 2. Leader style-ratio recompute
# ---------------------------------------------------------------------------
def leader_style_reclass(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    ls = sql_list(LEADERS)
    v1 = sql_list(EXCHANGE_INTERNAL_LEG_V1)

    log("fresh maker-side counts from joined_fills (streaming aggregate, 9 addresses)")
    maker_df = con.sql(
        f"""
        SELECT maker AS address,
               count(*) AS fresh_maker_n,
               sum(CASE WHEN exchange_internal_match THEN 1 ELSE 0 END) AS fresh_internal_n,
               sum(CASE WHEN taker IN ({v1}) THEN 1 ELSE 0 END) AS fresh_internal_v1_n
        FROM joined_fills
        WHERE maker IN ({ls})
        GROUP BY maker
        """
    ).fetchdf()

    log("fresh taker-side counts from joined_fills")
    taker_df = con.sql(
        f"""
        SELECT taker AS address, count(*) AS fresh_taker_n
        FROM joined_fills
        WHERE taker IN ({ls})
        GROUP BY taker
        """
    ).fetchdf()

    log("stored counts from traders.parquet + anchor internal counts from estimates parquet")
    stored = con.sql(
        f"""
        SELECT t.address,
               t.style_maker_fill_count AS stored_maker_n,
               t.style_taker_fill_count AS stored_taker_n,
               t.style_maker_taker_ratio AS stored_old_ratio,
               e.n_v1_internal_taker_legs_paired_with_addr_as_maker AS anchor_internal_v1_n
        FROM read_parquet('{TRADERS_PATH}') t
        LEFT JOIN read_parquet('{ESTIMATES_PATH}') e USING (address)
        WHERE t.address IN ({ls})
        """
    ).fetchdf()

    df = (
        stored.merge(maker_df, on="address", how="left")
        .merge(taker_df, on="address", how="left")
    )
    df["label"] = df["address"].map(LEADERS)

    # Anchor recompute (reproduces the 2026-05-28 relayer-implications numbers):
    # stored traders.parquet style counts + v1-only internal counts from that run.
    df["anchor_new_ratio"] = (df["stored_maker_n"] - df["anchor_internal_v1_n"]) / (
        df["stored_taker_n"] + df["anchor_internal_v1_n"]
    )
    # Fresh recompute, current shards, all 4 exchange-internal-leg contracts,
    # via the joined_fills exchange_internal_match flag (same population as the
    # style counts' source view).
    df["fresh_old_ratio"] = df["fresh_maker_n"] / df["fresh_taker_n"]
    df["fresh_new_ratio"] = (df["fresh_maker_n"] - df["fresh_internal_n"]) / (
        df["fresh_taker_n"] + df["fresh_internal_n"]
    )
    df["internal_share_of_maker_fills"] = df["fresh_internal_n"] / df["fresh_maker_n"]

    df["old_category"] = df["stored_old_ratio"].map(category)
    df["new_category_anchor"] = df["anchor_new_ratio"].map(category)
    df["new_category_fresh"] = df["fresh_new_ratio"].map(category)
    df["category_changed"] = df["old_category"] != df["new_category_fresh"]

    cols = [
        "label", "address",
        "stored_maker_n", "stored_taker_n", "stored_old_ratio", "old_category",
        "anchor_internal_v1_n", "anchor_new_ratio", "new_category_anchor",
        "fresh_maker_n", "fresh_taker_n", "fresh_internal_n", "fresh_internal_v1_n",
        "fresh_old_ratio", "fresh_new_ratio", "new_category_fresh",
        "internal_share_of_maker_fills", "category_changed",
    ]
    df = df[cols].sort_values("stored_old_ratio", ascending=False).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_STYLE, index=False)
    log(f"written: {OUT_STYLE.relative_to(ROOT)}")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.round(4).to_string(index=False))

    # Anchor verification
    domah = df[df["address"] == DOMAH].iloc[0]
    a6a72 = df[df["address"] == "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee"].iloc[0]
    print(
        f"\nanchor check: Domah {domah['stored_old_ratio']:.2f} -> "
        f"{domah['anchor_new_ratio']:.2f} (expected 7.89 -> 5.67); "
        f"0x6a72 {a6a72['stored_old_ratio']:.2f} -> {a6a72['anchor_new_ratio']:.2f} "
        f"(expected 5.15 -> 3.46)"
    )
    return df


# ---------------------------------------------------------------------------
# 3. Domah role-slice recompute (from cached audit parquets — no new sim)
# ---------------------------------------------------------------------------
def domah_role_slice_reclass() -> pd.DataFrame:
    """Replicates scripts/domah_copy_audit.py slice_table() role slice, with
    the role of each fragment reclassified by active_order_leg semantics:
    a maker-role fragment whose fill taker (cp_taker) is an exchange-internal-
    leg contract was actually Domah's ACTIVE order — it moves to the
    taker/active side of the slice. A position is attributed to the slice of
    its first fragment (same rule as the original audit)."""
    log("Domah role-slice recompute from cached audit fragments/positions")
    frag = pd.read_parquet(
        FRAGMENTS_PATH,
        columns=[
            "position_id", "fill_ts", "role", "cp_taker", "family",
            "hour_bucket", "usd_amount",
        ],
    )
    pos = pd.read_parquet(POSITIONS_PATH)

    frag["is_internal_leg"] = frag["role"].eq("maker") & frag["cp_taker"].isin(
        EXCHANGE_INTERNAL_LEG
    )
    frag["role_reclass"] = np.where(
        frag["role"].eq("taker") | frag["is_internal_leg"], "taker", "maker"
    )

    n_maker = int(frag["role"].eq("maker").sum())
    n_internal = int(frag["is_internal_leg"].sum())
    log(
        f"fragments: {len(frag):,}; maker-role {n_maker:,}; "
        f"internal-leg (reclassified) {n_internal:,} "
        f"({100.0 * n_internal / n_maker:.2f}% of maker-role fills, "
        f"${frag.loc[frag['is_internal_leg'], 'usd_amount'].sum():,.0f} notional)"
    )

    first = frag.sort_values("fill_ts").groupby("position_id").first()
    pos = pos.merge(
        first[["role", "role_reclass", "family", "hour_bucket"]]
        .rename(columns={
            "role": "first_role_old",
            "role_reclass": "first_role_new",
            "family": "first_family",
            "hour_bucket": "first_hour_bucket",
        }),
        left_on="position_id", right_index=True, how="left",
    )

    branches = ["A_opt", "A_real", "B", "C_opt", "C_real"]

    def slice_rows(p: pd.DataFrame, slice_col: str, scope: str, variant: str):
        rows = []
        for slc, g in p.groupby(slice_col, sort=False):
            row = {
                "scope": scope,
                "variant": variant,
                "slice": slc,
                "n_positions": int(len(g)),
                "leader_pnl": float(g["domah_pnl_calc"].sum()),
            }
            for br in branches:
                row[f"{br}_pnl"] = float(g[f"{br}_pnl"].sum())
            rows.append(row)
        return rows

    out_rows = []
    # Overall role slice — old (reproduces domah_audit_slice_role.parquet)
    # and new (active_order_leg reclassification).
    out_rows += slice_rows(pos, "first_role_old", "all_families", "old_role")
    out_rows += slice_rows(pos, "first_role_new", "all_families", "reclassified_role")
    # Macro family (the smoke-cell family)
    macro = pos[pos["first_family"] == "macro"]
    out_rows += slice_rows(macro, "first_role_old", "macro", "old_role")
    out_rows += slice_rows(macro, "first_role_new", "macro", "reclassified_role")
    # Macro + 18-24 hour bucket (the full smoke cell)
    macro_1824 = macro[macro["first_hour_bucket"] == "18-24"]
    out_rows += slice_rows(macro_1824, "first_role_old", "macro_18-24", "old_role")
    out_rows += slice_rows(macro_1824, "first_role_new", "macro_18-24", "reclassified_role")

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_ROLE_SLICE, index=False)
    log(f"written: {OUT_ROLE_SLICE.relative_to(ROOT)}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(0).to_string(index=False))

    # Per-fragment internal share inside the smoke cell, for the note.
    cell = frag[(frag["family"] == "macro") & (frag["hour_bucket"] == "18-24")]
    cell_maker = cell[cell["role"].eq("maker")]
    log(
        f"macro/18-24 fill-level: {len(cell):,} fills, "
        f"{len(cell_maker):,} maker-role, "
        f"{int(cell_maker['is_internal_leg'].sum()):,} internal-leg "
        f"({100.0 * cell_maker['is_internal_leg'].mean():.2f}% of maker-role)"
    )
    return out


def main() -> int:
    con = connect()
    smoke_test_flag(con)
    leader_style_reclass(con)
    domah_role_slice_reclass()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
