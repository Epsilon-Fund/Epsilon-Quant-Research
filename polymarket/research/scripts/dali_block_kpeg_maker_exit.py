"""Block K-PEG maker-exit round-trip test.

Replaces the robustness audit's pessimistic V2 (force taker cross on 100% of exits)
with a SYMMETRIC exit: after a hold, post a passive maker exit on the opposite side;
if it does not fill within a deadline, fall back to a taker cross. Sweeps exit
aggressiveness (ticks away from mid) = "optimise maker vs taker vs fill". Uses the
entry fills already produced by dali_block_kpeg_robustness.py plus the trade-print
stream from the slim panel.

Timing uses the +30s / +60s mids we already carry:
  entry @ t (maker) -> post maker exit @ t+30s using mid(t+30) -> exit-fill window (t+30, t+60]
  unfilled -> taker fallback @ t+60s crossing to touch (+ taker fee).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
FILLS = ANALYSIS / "kpeg_robustness_fills.parquet"
SLIM = Path("/tmp/kpeg_feat.parquet")
SRC = SLIM if SLIM.exists() else (ANALYSIS / "block_a1_features.parquet")
OUT = ANALYSIS / "csv_outputs" / "market_making" / "kpeg_maker_exit.csv"
REPORT = ANALYSIS / "kpeg_maker_exit_report.txt"

TICK = 0.01
POST_DELAY_S = 30          # hold before posting the maker exit
EXIT_WINDOW_S = 30         # how long the maker exit rests before taker fallback (t+30 -> t+60)
OFFSET_TICKS = [0, 1, 2, 3, 5]   # how far past mid the maker exit is posted (spread captured)
FEE = {"Crypto": 0.07, "Sports": 0.03, "Finance": 0.04, "Politics": 0.04, "Economics": 0.05,
       "Culture": 0.05, "Weather": 0.05, "Tech": 0.04, "Other": 0.05, "Geopolitics": 0.0}
REBATE_PCT = {"Crypto": 0.20, "Geopolitics": 0.0}


def maker_rebate_bps(cat, price):
    p = float(np.clip(price, 0.001, 0.999)); denom = float(np.clip(price, 0.01, 0.99))
    return FEE.get(cat, 0.05) * p * (1 - p) * REBATE_PCT.get(cat, 0.25) / denom * 1e4


def taker_fee_bps(cat, price, denom):
    pe = float(np.clip(price, 0.001, 0.999))
    return FEE.get(cat, 0.05) * pe * (1 - pe) / denom * 1e4


def load_trades():
    con = duckdb.connect()
    q = f"""
        SELECT asset_id,
               CAST(epoch_ns(CAST(received_at AS TIMESTAMP)) AS BIGINT) AS t_ns,
               trade_price,
               upper(coalesce(trade_side, last_trade_side, '')) AS side
        FROM read_parquet('{SRC}')
        WHERE event_type = 'last_trade_price'
          AND trade_price IS NOT NULL AND trade_price BETWEEN 0 AND 1
          AND upper(coalesce(trade_side, last_trade_side, '')) IN ('BUY','SELL')
    """
    df = con.execute(q).df(); con.close()
    df["asset_id"] = df["asset_id"].astype(str)
    out = {}
    for aid, g in df.sort_values("t_ns").groupby("asset_id", sort=False):
        out[aid] = (g["t_ns"].to_numpy(np.int64), g["trade_price"].to_numpy(float), g["side"].to_numpy(object))
    return out


def block_ci(df, col, seed=7):
    d = df[["market_id", "fill_time_ns", col]].dropna()
    d = d[np.isfinite(d[col])].reset_index(drop=True)
    if len(d) < 5:
        return (np.nan, np.nan)
    labels = []
    for mid, p in d.groupby("market_id", sort=False):
        el = (p["fill_time_ns"] - p["fill_time_ns"].min()) / 1e9
        labels.extend([f"{mid}:{int(b)}" for b in (el // 300).astype(int)])
    d = d.assign(block=labels)
    blocks = [idx.to_numpy() for _, idx in d.groupby("block").groups.items()]
    if len(blocks) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed); vals = d[col].to_numpy(float)
    means = [np.nanmean(vals[np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])])
             for _ in range(500)]
    return tuple(round(x, 1) for x in np.quantile(means, [0.025, 0.975]))


def main():
    fills = pd.read_parquet(FILLS)
    trades = load_trades()
    print(f"entry fills={len(fills):,}  assets-with-trades={len(trades):,}", flush=True)

    # absolute-volume sanity (answers "look at absolute numbers / 24h timeframe")
    span_days = (fills["fill_time_ns"].max() - fills["fill_time_ns"].min()) / 1e9 / 86400.0
    cc = fills[fills.category.eq("Crypto")]
    vol = {"capture_span_days": round(span_days, 2),
           "crypto_fills": len(cc), "crypto_fills_per_day": round(len(cc) / max(span_days, 1e-9), 1),
           "pooled_fills": len(fills), "pooled_fills_per_day": round(len(fills) / max(span_days, 1e-9), 1)}

    rows = []
    for offset in OFFSET_TICKS:
        recs = []
        for r in fills.itertuples(index=False):
            cat = r.category; denom = float(r.denom); ts = int(r.token_side)
            entry = float(r.entry_price); ent_reb = float(r.rebate_bps)
            mid30 = float(r.future_mid_30); mid60 = float(r.future_mid_60)
            half = float(r.half_spread)
            post_t = int(r.fill_time_ns) + POST_DELAY_S * 1_000_000_000
            win_hi = post_t + EXIT_WINDOW_S * 1_000_000_000
            # maker exit quote on the opposite side, offset ticks past mid(t+30)
            if ts == 1:   # long -> sell: post ask above mid
                exit_q = min(mid30 + offset * TICK, 0.999)
            else:         # short -> buy: post bid below mid
                exit_q = max(mid30 - offset * TICK, 0.001)
            filled_maker = False
            arr = trades.get(str(r.asset_id))
            if arr is not None:
                t_ns, px, sd = arr
                lo = np.searchsorted(t_ns, post_t, "left"); hi = np.searchsorted(t_ns, win_hi, "right")
                if hi > lo:
                    w_px = px[lo:hi]; w_sd = sd[lo:hi]
                    if ts == 1:   # need a BUY lifting our ask
                        filled_maker = bool(((w_sd == "BUY") & (w_px >= exit_q - 1e-12)).any())
                    else:         # need a SELL hitting our bid
                        filled_maker = bool(((w_sd == "SELL") & (w_px <= exit_q + 1e-12)).any())
            if filled_maker:
                gross = ts * (exit_q - entry) / denom * 1e4
                pnl = gross + ent_reb + maker_rebate_bps(cat, exit_q)
                mode = "maker_exit"
            else:
                fb = mid60 - ts * half                       # taker cross to touch at t+60
                gross = ts * (fb - entry) / denom * 1e4
                pnl = gross + ent_reb - taker_fee_bps(cat, mid60, denom)
                mode = "taker_fallback"
            recs.append({"market_id": r.market_id, "fill_time_ns": r.fill_time_ns,
                         "category": cat, "pnl_bps": pnl, "maker_filled": filled_maker, "mode": mode})
        rdf = pd.DataFrame(recs)
        for scope, sub in [("pooled", rdf), ("Crypto", rdf[rdf.category.eq("Crypto")])]:
            lo, hi = block_ci(sub, "pnl_bps")
            rows.append({"offset_ticks": offset, "scope": scope, "n": len(sub),
                         "maker_exit_fill_rate": round(float(sub.maker_filled.mean()), 3),
                         "mean_pnl_bps": round(float(sub.pnl_bps.mean()), 1),
                         "ci_lo": lo, "ci_hi": hi,
                         "win_rate": round(float(sub.pnl_bps.gt(0).mean()), 3)})

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    lines = ["K-PEG MAKER-EXIT ROUND-TRIP (post maker exit @t+30, taker fallback @t+60)",
             "=" * 80, "Absolute volume:"]
    for k, v in vol.items():
        lines.append(f"  {k}: {v}")
    lines += ["", res.to_string(index=False)]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)); print(f"\nwrote {OUT}\nwrote {REPORT}")


if __name__ == "__main__":
    main()
