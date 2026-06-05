"""Equity index up/down as MM structural-maker scope.

This is not an OD pricing test. It mirrors the K5-STRESS structured/non-top3
maker cut on equity index up/down markets, using settled wallet-market rows only
so the result is no-mark-to-mid.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/mm_equities_updown_structural_scope.py
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
CSV_DIR = ANALYSIS / "csv_outputs" / "market_making"
WALLET_MARKET = ANALYSIS / "k5_stress_wallet_market_full.parquet"
TRADES_GLOB = str(DATA / "trades" / "*.parquet")

OD_SCOPE = ANALYSIS / "csv_outputs" / "options_delta" / "od_equities_index_pricing_scope_market_detail.csv"
DALI_CANDIDATES = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_equity_index_100_candidates.csv"
SPX_OPEN_SCOPE = ANALYSIS / "csv_outputs" / "options_delta" / "od_equities_spx_open_updown_scope_market_detail.csv"

OUT_MARKETS = CSV_DIR / "mm_equities_updown_structural_scope_markets.csv"
OUT_COVERAGE = CSV_DIR / "mm_equities_updown_structural_scope_coverage.csv"
OUT_GATE = CSV_DIR / "mm_equities_updown_structural_scope_gate.csv"
OUT_ADVERSE = CSV_DIR / "mm_equities_updown_structural_scope_adverse_selection_audit.csv"

RNG_SEED = 20260603
BOOTSTRAP_SAMPLES = 500
STRUCT_TWO_SIDED_MIN = 0.60
STRUCT_CARRY_MIN = 0.50
STRUCT_SPIKE_MAX = 0.02

# Power guardrail for calling the full structural gate decision-grade. These
# are deliberately modest; politics_negrisk had much more depth.
MIN_TARGET_MARKETS = 50
MIN_TARGET_WALLETS = 30
MIN_TARGET_GROSS_USD = 250_000.0


def sql_list(values: set[str] | list[str]) -> str:
    vals = sorted({str(v).lower().replace("'", "''") for v in values if str(v)})
    return ", ".join(f"'{v}'" for v in vals) if vals else "''"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    temp_dir = ANALYSIS / ".duckdb_tmp_mm_equities_updown"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    return con


def as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def discover_markets() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = [
        (OD_SCOPE, "od_equities_scope"),
        (DALI_CANDIDATES, "dali_tfi_candidates"),
        (SPX_OPEN_SCOPE, "spx_open_scope"),
    ]
    for path, source in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            slug = as_str(row.get("market_slug") or row.get("slug")).lower()
            question = as_str(row.get("market_question") or row.get("question"))
            market_id = as_str(row.get("market_id"))
            condition_id = as_str(row.get("condition_id")).lower()
            if not condition_id or condition_id == "nan":
                continue
            if re.search(r"^(spx|ndx)-up-or-down-on-", slug):
                underlying = slug.split("-")[0].upper()
                scope = "close_spx_ndx"
                preferred = True
            elif re.search(r"^(spy|qqq)-up-or-down-on-", slug):
                underlying = slug.split("-")[0].upper()
                scope = "close_spy_qqq_secondary"
                preferred = False
            elif re.search(r"^spx-opens-up-or-down-on-", slug):
                underlying = "SPX_OPEN"
                scope = "open_spx_secondary"
                preferred = False
            else:
                continue
            rows.append(
                {
                    "source": source,
                    "scope": scope,
                    "subscope": underlying,
                    "preferred_close_spx_ndx": preferred,
                    "market_id": market_id,
                    "condition_id": condition_id,
                    "slug": slug,
                    "question": question,
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(["preferred_close_spx_ndx", "scope", "subscope", "slug"], ascending=[False, True, True, True])
    return out.drop_duplicates(["condition_id"], keep="first").reset_index(drop=True)


def raw_coverage(con: duckdb.DuckDBPyConnection, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    con.register("candidate_markets", candidates)
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    return con.execute(
        f"""
        WITH scoped AS (
            SELECT
                c.scope,
                c.subscope,
                lower(rt.condition_id) AS condition_id,
                CAST(rt.market_id AS VARCHAR) AS market_id,
                lower(rt.maker) AS maker,
                rt.usd_amount,
                rt.timestamp
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN candidate_markets c ON lower(rt.condition_id) = c.condition_id
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
        ),
        maker_market AS (
            SELECT scope, subscope, market_id, maker, sum(usd_amount) AS maker_usd
            FROM scoped
            GROUP BY 1, 2, 3, 4
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (PARTITION BY scope, subscope, market_id ORDER BY maker_usd DESC) AS maker_rank,
                sum(maker_usd) OVER (PARTITION BY scope, subscope, market_id) AS market_maker_usd
            FROM maker_market
        ),
        rank_agg AS (
            SELECT
                scope,
                subscope,
                sum(maker_usd) AS raw_maker_usd,
                sum(maker_usd) FILTER (WHERE maker_rank <= 3) AS raw_top3_maker_usd,
                sum(maker_usd) FILTER (WHERE maker_rank > 3) AS raw_non_top3_maker_usd
            FROM ranked
            GROUP BY 1, 2
        )
        SELECT
            s.scope,
            s.subscope,
            count(*) AS raw_fills,
            sum(s.usd_amount) AS raw_usd,
            count(DISTINCT s.market_id) AS raw_markets,
            count(DISTINCT s.maker) AS raw_maker_wallets,
            min(s.timestamp) AS first_raw_fill_ts,
            max(s.timestamp) AS last_raw_fill_ts,
            r.raw_maker_usd,
            r.raw_top3_maker_usd,
            r.raw_non_top3_maker_usd,
            r.raw_top3_maker_usd / nullif(r.raw_maker_usd, 0) AS raw_top3_maker_share,
            r.raw_non_top3_maker_usd / nullif(r.raw_maker_usd, 0) AS raw_non_top3_maker_share
        FROM scoped s
        LEFT JOIN rank_agg r USING (scope, subscope)
        GROUP BY 1, 2, 9, 10, 11, 12, 13
        """
    ).df()


def wallet_market_scope(con: duckdb.DuckDBPyConnection, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    con.register("candidate_markets", candidates)
    return con.execute(
        f"""
        SELECT
            c.scope,
            c.subscope,
            c.slug,
            wm.*
        FROM read_parquet('{WALLET_MARKET}') wm
        JOIN candidate_markets c ON CAST(wm.market_id AS VARCHAR) = CAST(c.market_id AS VARCHAR)
        """
    ).df()


def bootstrap_market_ci(piece: pd.DataFrame, net_col: str = "net_pnl_usd") -> tuple[float, float]:
    if piece.empty:
        return math.nan, math.nan
    grouped = piece.groupby("market_id", as_index=False).agg(gross=("gross_usd_volume", "sum"), net=(net_col, "sum"))
    grouped = grouped[grouped["gross"] > 0].reset_index(drop=True)
    if grouped.empty:
        return math.nan, math.nan
    rng = np.random.default_rng(RNG_SEED + len(grouped))
    values: list[float] = []
    idx = np.arange(len(grouped))
    for _ in range(BOOTSTRAP_SAMPLES):
        take = rng.choice(idx, size=len(idx), replace=True)
        gross = float(grouped.loc[take, "gross"].sum())
        net = float(grouped.loc[take, "net"].sum())
        values.append(net / gross * 10_000.0 if gross > 0 else math.nan)
    return tuple(float(x) for x in np.nanpercentile(values, [2.5, 97.5]))


def structured_summary_for(piece: pd.DataFrame, scope: str, subscope: str) -> dict[str, Any]:
    if piece.empty:
        return {"scope": scope, "subscope": subscope, "target_rows": 0}
    settled = piece[piece["mark_source"].astype(str).str.contains("settlement", na=False)].copy()
    if settled.empty:
        return {
            "scope": scope,
            "subscope": subscope,
            "wallet_market_rows": len(piece),
            "settled_wallet_market_rows": 0,
            "target_rows": 0,
        }

    wallet = (
        settled.groupby("address", as_index=False)
        .agg(
            all_wallet_markets=("market_id", "nunique"),
            all_gross_usd_volume=("gross_usd_volume", "sum"),
            all_net_pnl_usd=("net_pnl_usd", "sum"),
            all_maker_rebate_usd=("maker_rebate_usd", "sum"),
            all_taker_fee_usd=("taker_fee_usd", "sum"),
            all_maker_fills=("maker_fills", "sum"),
            all_maker_usd=("maker_usd", "sum"),
            all_gross_token_volume=("gross_token_volume", "sum"),
            all_abs_final_token_position=("abs_final_token_position", "sum"),
            all_spike_zone_usd=("spike_zone_usd", "sum"),
        )
    )
    two_sided = (
        settled[settled["distinct_outcomes_made"] >= 2]
        .groupby("address")["maker_usd"]
        .sum()
        .rename("two_sided_maker_usd")
    )
    wallet = wallet.merge(two_sided, on="address", how="left")
    wallet["two_sided_maker_usd"] = wallet["two_sided_maker_usd"].fillna(0.0)
    wallet["two_sided_usd_share"] = wallet["two_sided_maker_usd"] / wallet["all_maker_usd"].replace(0, np.nan)
    wallet["carry_token_share"] = wallet["all_abs_final_token_position"] / wallet["all_gross_token_volume"].replace(0, np.nan)
    wallet["spike_zone_usd_share"] = wallet["all_spike_zone_usd"] / wallet["all_maker_usd"].replace(0, np.nan)
    wallet[["two_sided_usd_share", "carry_token_share", "spike_zone_usd_share"]] = wallet[
        ["two_sided_usd_share", "carry_token_share", "spike_zone_usd_share"]
    ].fillna(0.0)
    wallet["structured_playbook"] = (
        wallet["two_sided_usd_share"].ge(STRUCT_TWO_SIDED_MIN)
        & wallet["carry_token_share"].ge(STRUCT_CARRY_MIN)
        & wallet["spike_zone_usd_share"].le(STRUCT_SPIKE_MAX)
        & wallet["all_maker_fills"].gt(0)
    )

    tagged = settled.merge(wallet[["address", "structured_playbook"]], on="address", how="left")
    target = tagged[tagged["structured_playbook"].fillna(False) & ~tagged["is_global_top3_market_maker"].fillna(False)].copy()
    gross = float(target["gross_usd_volume"].sum()) if not target.empty else 0.0
    net = float(target["net_pnl_usd"].sum()) if not target.empty else 0.0
    rebate = float(target["maker_rebate_usd"].sum()) if not target.empty else 0.0
    lo, hi = bootstrap_market_ci(target)
    wallet_bps = pd.Series(dtype=float)
    if not target.empty:
        wallet_bps = (
            target.groupby("address")
            .agg(gross=("gross_usd_volume", "sum"), net=("net_pnl_usd", "sum"))
            .query("gross > 0")
            .assign(bps=lambda df: df["net"] / df["gross"] * 10_000.0)["bps"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
    thin_reasons: list[str] = []
    if target["market_id"].nunique() < MIN_TARGET_MARKETS:
        thin_reasons.append(f"target_markets<{MIN_TARGET_MARKETS}")
    if target["address"].nunique() < MIN_TARGET_WALLETS:
        thin_reasons.append(f"target_wallets<{MIN_TARGET_WALLETS}")
    if gross < MIN_TARGET_GROSS_USD:
        thin_reasons.append(f"target_gross_usd<{MIN_TARGET_GROSS_USD:,.0f}")
    decision_grade = not thin_reasons
    return {
        "scope": scope,
        "subscope": subscope,
        "wallet_market_rows": len(piece),
        "settled_wallet_market_rows": len(settled),
        "settled_markets": settled["market_id"].nunique(),
        "all_wallets": settled["address"].nunique(),
        "structured_wallets": int(wallet["structured_playbook"].sum()),
        "target_rows": len(target),
        "target_wallets": target["address"].nunique(),
        "target_markets": target["market_id"].nunique(),
        "target_maker_fills": int(target["maker_fills"].sum()) if not target.empty else 0,
        "gross_usd_volume": gross,
        "net_pnl_usd": net,
        "maker_rebate_usd": rebate,
        "net_pnl_bps": net / gross * 10_000.0 if gross > 0 else math.nan,
        "ci_lo_bps": lo,
        "ci_hi_bps": hi,
        "median_wallet_bps": float(wallet_bps.median()) if not wallet_bps.empty else math.nan,
        "net_without_rebate_bps": (net - rebate) / gross * 10_000.0 if gross > 0 else math.nan,
        "structured_two_sided_median": float(wallet.loc[wallet["structured_playbook"], "two_sided_usd_share"].median())
        if wallet["structured_playbook"].any()
        else math.nan,
        "structured_carry_median": float(wallet.loc[wallet["structured_playbook"], "carry_token_share"].median())
        if wallet["structured_playbook"].any()
        else math.nan,
        "structured_spike_median": float(wallet.loc[wallet["structured_playbook"], "spike_zone_usd_share"].median())
        if wallet["structured_playbook"].any()
        else math.nan,
        "decision_grade_sample": decision_grade,
        "thin_reasons": "; ".join(thin_reasons),
        "ci_positive": bool(np.isfinite(lo) and lo > 0),
        "median_positive": bool(np.isfinite(float(wallet_bps.median()) if not wallet_bps.empty else math.nan) and float(wallet_bps.median()) > 0)
        if not wallet_bps.empty
        else False,
        "ex_rebate_positive": bool(gross > 0 and (net - rebate) > 0),
    }


def wm_coverage(wm: pd.DataFrame) -> pd.DataFrame:
    if wm.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (scope, subscope), piece in wm.groupby(["scope", "subscope"], dropna=False):
        settled = piece[piece["mark_source"].astype(str).str.contains("settlement", na=False)]
        rows.append(
            {
                "scope": scope,
                "subscope": subscope,
                "wallet_market_rows": len(piece),
                "wm_markets": piece["market_id"].nunique(),
                "wm_wallets": piece["address"].nunique(),
                "maker_fills": int(piece["maker_fills"].sum()),
                "maker_usd": float(piece["maker_usd"].sum()),
                "gross_usd_volume": float(piece["gross_usd_volume"].sum()),
                "net_pnl_usd": float(piece["net_pnl_usd"].sum()),
                "settled_wallet_market_rows": len(settled),
                "settled_markets": settled["market_id"].nunique(),
                "settled_maker_fills": int(settled["maker_fills"].sum()),
                "settled_maker_usd": float(settled["maker_usd"].sum()),
            }
        )
    return pd.DataFrame(rows)


def combine_coverage(candidates: pd.DataFrame, wm: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    discovered = candidates.groupby(["scope", "subscope"], as_index=False).agg(discovered_markets=("condition_id", "nunique"))
    wm_cov = wm_coverage(wm)
    out = discovered.merge(wm_cov, on=["scope", "subscope"], how="left").merge(raw, on=["scope", "subscope"], how="left")
    return out.sort_values(["scope", "subscope"]).reset_index(drop=True)


def build_gate(wm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, piece in wm.groupby("scope", dropna=False):
        rows.append(structured_summary_for(piece, str(scope), "ALL"))
        for subscope, subpiece in piece.groupby("subscope", dropna=False):
            rows.append(structured_summary_for(subpiece, str(scope), str(subscope)))
    out = pd.DataFrame(rows)
    return out.sort_values(["scope", "subscope"]).reset_index(drop=True)


def adverse_selection_audit() -> pd.DataFrame:
    candidates = [
        {
            "ingredient": "PM settled maker fills / wallet-market PnL",
            "available": WALLET_MARKET.exists(),
            "usable_now": True,
            "detail": "K5-STRESS wallet-market cache is available; this pass uses settlement-marked rows only.",
        },
        {
            "ingredient": "PM historical CLOB books around equity-index fills",
            "available": False,
            "usable_now": False,
            "detail": "Owned local data has fills, not quote/cancel/queue history for SPX/NDX close-style up/down.",
        },
        {
            "ingredient": "ES/MES intraday futures at fill timestamps",
            "available": False,
            "usable_now": False,
            "detail": "No local ES/MES intraday parquet/CSV was found. Prior OD audit only produced a recent CME settlement probe, not fill-aligned futures states.",
        },
        {
            "ingredient": "SPX cash / realized-vol proxy from OD pricing pass",
            "available": (ANALYSIS / "od_equities_spx_nz_pricing_fills.parquet").exists(),
            "usable_now": False,
            "detail": "This is useful OD context but is not the requested continuous ES/MES adverse-selection reference.",
        },
    ]
    return pd.DataFrame(candidates)


def main() -> int:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    candidates = discover_markets()
    if candidates.empty:
        raise SystemExit("no equity up/down candidate markets discovered")
    con = connect()
    raw = raw_coverage(con, candidates)
    wm = wallet_market_scope(con, candidates)
    coverage = combine_coverage(candidates, wm, raw)
    gate = build_gate(wm)
    adverse = adverse_selection_audit()

    candidates.to_csv(OUT_MARKETS, index=False)
    coverage.to_csv(OUT_COVERAGE, index=False)
    gate.to_csv(OUT_GATE, index=False)
    adverse.to_csv(OUT_ADVERSE, index=False)

    print(f"wrote {OUT_MARKETS.relative_to(ROOT)} ({len(candidates)} rows)")
    print(f"wrote {OUT_COVERAGE.relative_to(ROOT)} ({len(coverage)} rows)")
    print(f"wrote {OUT_GATE.relative_to(ROOT)} ({len(gate)} rows)")
    print(f"wrote {OUT_ADVERSE.relative_to(ROOT)} ({len(adverse)} rows)")
    print("\ncoverage")
    print(
        coverage[
            [
                "scope",
                "subscope",
                "discovered_markets",
                "raw_fills",
                "raw_usd",
                "raw_markets",
                "raw_maker_wallets",
                "settled_markets",
                "settled_maker_fills",
                "settled_maker_usd",
            ]
        ].to_string(index=False)
    )
    print("\nstructured gate")
    print(
        gate[
            [
                "scope",
                "subscope",
                "target_wallets",
                "target_markets",
                "gross_usd_volume",
                "net_pnl_bps",
                "ci_lo_bps",
                "ci_hi_bps",
                "median_wallet_bps",
                "net_without_rebate_bps",
                "decision_grade_sample",
                "thin_reasons",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
