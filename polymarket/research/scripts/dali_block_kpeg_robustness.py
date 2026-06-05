"""Block K-PEG robustness / lookahead audit.

Re-uses the EXACT K-PEG candidate-pool builder and quote logic (imported from
dali_block_kpeg_chase_optimization) with the FROZEN winning params, and runs:

  V0  reproduce  : kpeg net_pnl_bps_30 (realized_spread + rebate - adverse_30 - inv_charge_30)
  ID  identity   : verify V0 == token_side*(future_mid_30 - entry)/denom*1e4 + rebate - inv_charge_30
  V1  ceiling    : mark-to-future-mid_30 + rebate, NO synthetic inv charge, NO exit cost
  V2  round-trip : V1's position closed as a TAKER at 60s (cross spread to touch + taker fee),
                   i.e. apples-to-apples with K2 which DID pay a taker exit. Not extra pessimism;
                   it just completes the round-trip K-PEG omits.
  OOS : frozen params, split discovery (a0,a0b) vs holdout (a0c,a0c_roll)
  PLACEBO : shuffle future_mid_30 within category -> edge should collapse if it is genuinely
            future-dependent (no constant-offset / non-lookahead artifact).

No re-optimization. optuna is stubbed because we never call optimize().
"""
from __future__ import annotations

import sys
import re
import types
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

# --- stub optuna so the module imports (we only use pool-build + simulate + helpers) ---
sys.modules.setdefault("optuna", types.ModuleType("optuna"))

SCRIPTS = Path(__file__).resolve().parent
KPEG_PATH = SCRIPTS / "dali_block_kpeg_chase_optimization.py"
spec = importlib.util.spec_from_file_location("kpeg", KPEG_PATH)
kpeg = importlib.util.module_from_spec(spec)
sys.modules["kpeg"] = kpeg  # required so dataclass introspection can resolve __module__
spec.loader.exec_module(kpeg)

ROOT = SCRIPTS.parent
# Slim, pre-projected panel (built once) so the per-market scan fits the sandbox time/memory budget.
# Falls back to the full panel if the slim one is absent.
_SLIM = Path("/tmp/kpeg_feat.parquet")
FEATURES = _SLIM if _SLIM.exists() else (ROOT / "data" / "analysis" / "block_a1_features.parquet")
kpeg.FEATURES = FEATURES            # redirect the imported builder's source
kpeg.CADENCE_OPTIONS = (1,)         # frozen winning cadence; cuts pool ~5x
OUT = ROOT / "data" / "analysis" / "kpeg_robustness.csv"
REPORT = ROOT / "data" / "analysis" / "kpeg_robustness_report.txt"
FILLS_OUT = ROOT / "data" / "analysis" / "kpeg_robustness_fills.parquet"
PHASE_OUT = ROOT / "data" / "analysis" / "kpeg_robustness_phase.csv"

WINDOW_SEC = 4 * 3600  # crypto-4h window length
PHASE_BINS = [0, 15, 30, 60, 120, 240, 1e9]  # minutes since window open
PHASE_LABELS = ["0-15m", "15-30m", "30-60m", "60-120m", "120-240m", "240m+"]


def window_open_epoch(slug: str) -> float:
    """Parse window-open unix ts from crypto-4h slugs like 'btc-updown-4h-1779912000'."""
    m = re.search(r"updown-4h-(\d{9,11})", str(slug))
    return float(m.group(1)) if m else float("nan")

BEST = kpeg.Params(peg_offset_ticks=0, chase_increment_ticks=2, chase_cap_c_ticks=7,
                   inventory_scaling=0.304, cadence_sec=1)
TICK = kpeg.TICK
HORIZ = kpeg.ADVERSE_HORIZONS  # (5,30,60)


def run_group_for_markets() -> dict:
    con = duckdb.connect()
    q = f"""
        SELECT market_id, string_agg(DISTINCT run_id, ',' ORDER BY run_id) runs
        FROM read_parquet('{FEATURES}')
        WHERE run_id IN ('a0','a0b','a0c','a0c_roll')
        GROUP BY market_id
    """
    df = con.execute(q).df()
    con.close()
    out = {}
    for r in df.itertuples(index=False):
        runs = set(str(r.runs).split(","))
        holdout = bool(runs & {"a0c", "a0c_roll"})
        discovery = bool(runs & {"a0", "a0b"})
        # classify by membership; markets that are purely holdout -> holdout, purely disc -> discovery
        if holdout and not discovery:
            out[str(r.market_id)] = "holdout"
        elif discovery and not holdout:
            out[str(r.market_id)] = "discovery"
        else:
            out[str(r.market_id)] = "mixed"
    return out


def simulate_rich(pool: pd.DataFrame, params: kpeg.Params) -> pd.DataFrame:
    """Faithful copy of kpeg.simulate inventory loop, storing extra fields for the round-trip."""
    sub = pool[pool["cadence_sec"].eq(params.cadence_sec)].copy()
    fills = []
    for _, asset_rows in sub.groupby(["market_id", "asset_id"], sort=False):
        open_lots = []
        for row in asset_rows.itertuples(index=False):
            now = int(row.fill_time_ns)
            open_lots = [lot for lot in open_lots if int(lot["exit_time_ns"]) > now]
            inventory = int(sum(int(lot["token_side"]) for lot in open_lots))
            bid, ask, bid_dist, ask_dist = kpeg.quote_prices(row, params, inventory)
            token_side = 0
            entry_price = np.nan
            side = str(row.trade_side_norm)
            trade_price = float(row.trade_price)
            if side == "SELL" and inventory < kpeg.INTERNAL_INVENTORY_CAP and trade_price <= bid + 1e-12:
                token_side, entry_price = 1, bid
            elif side == "BUY" and inventory > -kpeg.INTERNAL_INVENTORY_CAP and trade_price >= ask - 1e-12:
                token_side, entry_price = -1, ask
            if token_side == 0 or not np.isfinite(entry_price):
                continue
            denom = max(entry_price, 0.01)
            cur_mid = float(row.current_mid)
            cbid = float(row.current_best_bid)
            cask = float(row.current_best_ask)
            half_spread = max(0.0, (cask - cbid) / 2.0)
            rebate = kpeg.taker_rebate_bps(str(row.category), entry_price)
            fm = {h: float(getattr(row, f"future_mid_{h}s")) for h in HORIZ}
            rec = {
                "market_id": str(row.market_id), "asset_id": str(row.asset_id),
                "slug": str(row.slug), "category": str(row.category), "fee_segment": str(row.fee_segment),
                "fill_time_ns": now, "token_side": token_side, "entry_price": entry_price,
                "denom": denom, "current_mid": cur_mid, "micro_price": float(row.micro_price),
                "half_spread": half_spread, "book_spread_bps": (cask - cbid) / denom * 1e4,
                "abs_dist_from_50c": abs(entry_price - 0.5), "rebate_bps": rebate,
                "future_mid_5": fm[5], "future_mid_30": fm[30], "future_mid_60": fm[60],
            }
            # V0 components (mirror add_fill_economics @30s)
            adv30 = -token_side * (fm[30] - cur_mid) / denom * 1e4
            inv30 = 0.5 * abs(fm[30] - cur_mid) / denom * 1e4
            realized_spread = token_side * (cur_mid - entry_price) / denom * 1e4
            rec["realized_spread_bps"] = realized_spread
            rec["adverse_30_bps"] = adv30
            rec["inv_charge_30_bps"] = inv30
            rec["v0_net_30_bps"] = realized_spread + rebate - adv30 - inv30
            # V1 ceiling: mark-to-mid_30 + rebate
            rec["v1_marktomid_30_bps"] = (token_side * (fm[30] - entry_price) / denom * 1e4) + rebate
            # V2 round-trip: close as taker at 60s (cross spread, pay taker fee)
            fee_rate = kpeg.FEE_BY_CATEGORY.get(str(row.category), kpeg.FEE_BY_CATEGORY["Other"])["fee_rate"]
            pe = float(np.clip(fm[60], 0.001, 0.999))
            exit_fee_bps = fee_rate * pe * (1 - pe) / denom * 1e4
            half_spread_bps = half_spread / denom * 1e4
            gross_to_mid60 = token_side * (fm[60] - entry_price) / denom * 1e4
            rec["v2_roundtrip_60_bps"] = gross_to_mid60 + rebate - half_spread_bps - exit_fee_bps
            rec["exit_half_spread_bps"] = half_spread_bps
            rec["exit_fee_bps"] = exit_fee_bps
            fills.append(rec)
            open_lots.append({"exit_time_ns": now + kpeg.HOLD_SEC * 1_000_000_000, "token_side": token_side})
    return pd.DataFrame(fills)


def block_ci(df: pd.DataFrame, col: str, seed: int = 7) -> tuple:
    d = df[["market_id", "fill_time_ns", col]].dropna()
    d = d[np.isfinite(d[col])].reset_index(drop=True)
    if len(d) < 5:
        return (np.nan, np.nan)
    labels = []
    for mid, piece in d.groupby("market_id", sort=False):
        el = (piece["fill_time_ns"] - piece["fill_time_ns"].min()) / 1e9
        labels.extend([f"{mid}:{int(b)}" for b in (el // 300).astype(int)])
    d = d.assign(block=labels)
    blocks = [idx.to_numpy() for _, idx in d.groupby("block").groups.items()]
    if len(blocks) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = d[col].to_numpy(float)
    means = [np.nanmean(vals[np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])])
             for _ in range(500)]
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(df: pd.DataFrame, col: str, label: str) -> dict:
    n = len(df)
    mean = float(df[col].mean()) if n else np.nan
    lo, hi = block_ci(df, col) if n else (np.nan, np.nan)
    return {"view": label, "metric": col, "n": n, "mean_bps": round(mean, 1),
            "ci_lo": round(lo, 1) if np.isfinite(lo) else np.nan,
            "ci_hi": round(hi, 1) if np.isfinite(hi) else np.nan,
            "win_rate": round(float(df[col].gt(0).mean()), 3) if n else np.nan}


def main() -> None:
    print("building candidate pool (this is the slow part)...", flush=True)
    pool, meta = kpeg.build_candidate_pool()
    print(f"pool rows={len(pool):,} markets={pool['market_id'].nunique()}", flush=True)

    rg = run_group_for_markets()
    rich = simulate_rich(pool, BEST)
    rich["run_group"] = rich["market_id"].map(rg).fillna("mixed")
    print(f"frozen-param fills={len(rich):,}", flush=True)

    # cross-check against canonical kpeg.simulate net_30
    canon = kpeg.simulate(pool, BEST)
    canon_mean = float(canon["net_pnl_bps_30s"].mean())
    canon_crypto = float(canon[canon["category"].eq("Crypto")]["net_pnl_bps_30s"].mean())

    rows = []
    rows.append({"view": "CANON kpeg.simulate pooled", "metric": "net_pnl_bps_30s",
                 "n": len(canon), "mean_bps": round(canon_mean, 1), "ci_lo": np.nan, "ci_hi": np.nan,
                 "win_rate": np.nan})
    rows.append({"view": "CANON kpeg.simulate Crypto", "metric": "net_pnl_bps_30s",
                 "n": int(canon["category"].eq("Crypto").sum()), "mean_bps": round(canon_crypto, 1),
                 "ci_lo": np.nan, "ci_hi": np.nan, "win_rate": np.nan})

    # identity check: my v0 vs canonical (should match on Crypto + pooled)
    rows.append(summarize(rich, "v0_net_30_bps", "V0 reproduce pooled"))
    rows.append(summarize(rich[rich.category.eq("Crypto")], "v0_net_30_bps", "V0 reproduce Crypto"))

    # V1 ceiling (mark-to-mid, no inv charge, no exit)
    rows.append(summarize(rich, "v1_marktomid_30_bps", "V1 mark-to-mid pooled"))
    rows.append(summarize(rich[rich.category.eq("Crypto")], "v1_marktomid_30_bps", "V1 mark-to-mid Crypto"))

    # V2 round-trip (taker exit @60s, comparable to K2)
    rows.append(summarize(rich, "v2_roundtrip_60_bps", "V2 round-trip pooled"))
    rows.append(summarize(rich[rich.category.eq("Crypto")], "v2_roundtrip_60_bps", "V2 round-trip Crypto"))

    # OOS split (frozen params) on V0 and V2
    for grp in ("discovery", "holdout", "mixed"):
        sub = rich[rich.run_group.eq(grp)]
        if len(sub):
            rows.append(summarize(sub, "v0_net_30_bps", f"V0 [{grp}] pooled"))
            rows.append(summarize(sub[sub.category.eq("Crypto")], "v0_net_30_bps", f"V0 [{grp}] Crypto"))
            rows.append(summarize(sub, "v2_roundtrip_60_bps", f"V2 [{grp}] pooled"))
            rows.append(summarize(sub[sub.category.eq("Crypto")], "v2_roundtrip_60_bps", f"V2 [{grp}] Crypto"))

    # PLACEBO: shuffle future_mid_30 within category, recompute V1 -> should collapse toward ~0+rebate
    plc = rich.copy()
    rng = np.random.default_rng(99)
    plc["fm30_shuf"] = plc.groupby("category")["future_mid_30"].transform(
        lambda s: s.to_numpy()[rng.permutation(len(s))])
    plc["v1_placebo_bps"] = (plc["token_side"] * (plc["fm30_shuf"] - plc["entry_price"]) / plc["denom"] * 1e4) + plc["rebate_bps"]
    rows.append(summarize(plc, "v1_placebo_bps", "PLACEBO shuffled-mid pooled"))
    rows.append(summarize(plc[plc.category.eq("Crypto")], "v1_placebo_bps", "PLACEBO shuffled-mid Crypto"))

    # component means (Crypto) for the decomposition story
    cc = rich[rich.category.eq("Crypto")]
    comp = {
        "crypto_n": len(cc),
        "mean_entry_price": round(float(cc.entry_price.mean()), 4),
        "mean_dist_entry_to_micro_ticks": round(float(((cc.micro_price - cc.entry_price) * cc.token_side / TICK).mean()), 2),
        "mean_realized_spread_bps": round(float(cc.realized_spread_bps.mean()), 1),
        "mean_adverse_30_bps": round(float(cc.adverse_30_bps.mean()), 1),
        "mean_inv_charge_30_bps": round(float(cc.inv_charge_30_bps.mean()), 1),
        "mean_rebate_bps": round(float(cc.rebate_bps.mean()), 1),
        "mean_exit_half_spread_bps": round(float(cc.exit_half_spread_bps.mean()), 1),
        "mean_exit_fee_bps": round(float(cc.exit_fee_bps.mean()), 1),
    }

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)

    # ---- WINDOW-PHASE ANALYSIS (crypto-4h): does spread narrow over the window, and where does PnL come from? ----
    rich["window_open"] = rich["slug"].map(window_open_epoch)
    is4h = rich["window_open"].notna()
    phase_rows = []
    if is4h.any():
        w = rich[is4h].copy()
        w["elapsed_min"] = (w["fill_time_ns"] / 1e9 - w["window_open"]) / 60.0
        w["ttr_min"] = (w["window_open"] + WINDOW_SEC - w["fill_time_ns"] / 1e9) / 60.0
        w = w[(w["elapsed_min"] >= -1) & (w["elapsed_min"] <= 245)]
        w["phase"] = pd.cut(w["elapsed_min"].clip(lower=0), bins=PHASE_BINS, labels=PHASE_LABELS, right=False)
        for ph, sub in w.groupby("phase", observed=True):
            if not len(sub):
                continue
            phase_rows.append({
                "dim": "window_phase_4h", "bucket": str(ph), "n_fills": len(sub),
                "mean_book_spread_bps": round(float(sub.book_spread_bps.mean()), 1),
                "mean_entry_price": round(float(sub.entry_price.mean()), 3),
                "mean_realized_spread_bps": round(float(sub.realized_spread_bps.mean()), 1),
                "mean_adverse_30_bps": round(float(sub.adverse_30_bps.mean()), 1),
                "mean_v0_30_bps": round(float(sub.v0_net_30_bps.mean()), 1),
                "mean_v1_marktomid_bps": round(float(sub.v1_marktomid_30_bps.mean()), 1),
                "mean_v2_roundtrip_bps": round(float(sub.v2_roundtrip_60_bps.mean()), 1),
            })
        # unconditional book spread over phase across ALL candidate quote rows (cadence=1, crypto-4h), to
        # answer the spread-narrowing question without the fill-selection bias.
        cpool = pool[pool["cadence_sec"].eq(1)].copy()
        cpool["window_open"] = cpool["slug"].map(window_open_epoch)
        cpool = cpool[cpool["window_open"].notna()].copy()
        if len(cpool):
            cpool["spread_bps"] = (cpool["current_best_ask"] - cpool["current_best_bid"]) / cpool["current_mid"].clip(lower=0.01) * 1e4
            cpool["elapsed_min"] = (cpool["fill_time_ns"] / 1e9 - cpool["window_open"]) / 60.0
            cpool = cpool[(cpool["elapsed_min"] >= 0) & (cpool["elapsed_min"] <= 245)]
            cpool["phase"] = pd.cut(cpool["elapsed_min"], bins=PHASE_BINS, labels=PHASE_LABELS, right=False)
            for ph, sub in cpool.groupby("phase", observed=True):
                if not len(sub):
                    continue
                phase_rows.append({
                    "dim": "unconditional_spread_4h", "bucket": str(ph), "n_fills": len(sub),
                    "mean_book_spread_bps": round(float(sub.spread_bps.mean()), 1),
                    "mean_entry_price": round(float(sub.current_mid.mean()), 3),
                    "mean_realized_spread_bps": np.nan, "mean_adverse_30_bps": np.nan,
                    "mean_v0_30_bps": np.nan, "mean_v1_marktomid_bps": np.nan, "mean_v2_roundtrip_bps": np.nan,
                })

    # ---- ENTRY-PRICE / GAMMA-ZONE BUCKETS (crypto): the near-50c-late toxic cell prediction ----
    cc2 = rich[rich.category.eq("Crypto")].copy()
    cc2["dist50_bucket"] = pd.cut(cc2["abs_dist_from_50c"], bins=[0, 0.1, 0.25, 0.5], labels=["near_50c", "mid", "near_0_1"], right=False)
    for b, sub in cc2.groupby("dist50_bucket", observed=True):
        if not len(sub):
            continue
        phase_rows.append({
            "dim": "crypto_dist_from_50c", "bucket": str(b), "n_fills": len(sub),
            "mean_book_spread_bps": round(float(sub.book_spread_bps.mean()), 1),
            "mean_entry_price": round(float(sub.entry_price.mean()), 3),
            "mean_realized_spread_bps": round(float(sub.realized_spread_bps.mean()), 1),
            "mean_adverse_30_bps": round(float(sub.adverse_30_bps.mean()), 1),
            "mean_v0_30_bps": round(float(sub.v0_net_30_bps.mean()), 1),
            "mean_v1_marktomid_bps": round(float(sub.v1_marktomid_30_bps.mean()), 1),
            "mean_v2_roundtrip_bps": round(float(sub.v2_roundtrip_60_bps.mean()), 1),
        })

    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(PHASE_OUT, index=False)
    rich.drop(columns=[c for c in ("fm30_shuf",) if c in rich]).to_parquet(FILLS_OUT, index=False)

    lines = []
    lines.append("K-PEG ROBUSTNESS / LOOKAHEAD AUDIT  (frozen params: peg=0 inc=2 c=7 inv=0.304 cad=1s)")
    lines.append("=" * 90)
    lines.append(res.to_string(index=False))
    lines.append("")
    lines.append("Crypto component means (per fill):")
    for k, v in comp.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("WINDOW-PHASE / SPREAD-NARROWING / GAMMA-ZONE:")
    lines.append("=" * 90)
    lines.append(phase_df.to_string(index=False))
    report = "\n".join(lines)
    REPORT.write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"\nwrote {OUT}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
