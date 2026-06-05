"""FTC take-profit backtest with 1/N sizing across active weather cities.

Strategy:
- Universe: highest-temperature-* and lowest-temperature-* slug families,
  collapsed to one entry per (city, end_ts) — the ~7 range buckets per city
  on a given day are mutually exclusive (exactly one resolves YES), so they
  represent ONE underlying view, not 7 independent bets.
- A city is "available" between (end_ts - window_hours) and resolution_ts.
- Entry: FTC at p_in. With one_trade_per_city=True (default), only the first
  market within each (city, resolution_date) to cross p_in is taken.
- Exit: TP at p_out if first_cross(p_out) > first_cross(p_in), else hold to
  resolution_ts.
- Sizing: position notional = 1/N where N = count of distinct active CITIES
  (not markets) at entry_ts. Each $1 of notional buys 1/entry_price shares.

PnL per share (taker model — pay ASK to enter long, receive BID on TP):
  - TP:        +(exit_bid - entry_ask)
  - hold-win:  +(1        - entry_ask)
  - hold-chop:  −entry_ask

Slippage source: next fill in (anchor_ts + min_seconds, anchor_ts + max_seconds]
in the same (market, token), by different counterparties. Fallbacks: ASK =
p_in/p_out + fallback_cents/100; BID = max(0, ASK − assumed_spread_cents/100).

Per-trade $ PnL on bankroll (size = 1/N of $1 bankroll):
  pnl_pct = notional_pct * (pnl_per_share / entry_ask)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def city_from_slug_family(sf: str) -> str:
    """Collapse derivative slug_families (e.g. '-84forhigher', '-neg-1corbelow') into the base city.

    'highest-temperature-in-miami-84forhigher' → 'highest-temperature-in-miami'.
    Used so 1/N counts independent cities, not the ~7 correlated range buckets per city.
    """
    return re.sub(r"-(?:neg-?)?[0-9].*$", "", sf)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_infra import weather_analysis as wa  # noqa: E402


def build_trades(
    p_wide: pd.DataFrame,
    inst: pd.DataFrame,
    p_in: float,
    p_out: float,
    min_seconds: int = wa.DEFAULT_MIN_SECONDS,
    max_seconds: int = wa.DEFAULT_MAX_SECONDS,
    fallback_cents: float = wa.DEFAULT_FALLBACK_CENTS,
    assumed_spread_cents: float = wa.DEFAULT_SPREAD_CENTS,
    trades_glob: str = wa.DEFAULT_TRADES_GLOB,
    pnl_model: str = "taker",
    exit_passive: str = "optimistic",
) -> tuple[pd.DataFrame, int]:
    """One row per trade with entry_ts / exit_ts / pnl_per_share.

    pnl_model:
      'taker'   (default) — cross-the-spread proxy. pnl_per_share uses next-fill
                            ASK on entry and next-fill BID on TP exit (with
                            fallback constants when no real next-fill exists).
                            Every cross becomes a trade. Original semantics.

                            PnL formulas:
                              TP:        exit_bid - entry_ask
                              hold_win:  1.0      - entry_ask
                              hold_chop: -entry_ask

      'passive' — WS-passive execution. ⚠️ OPTIMISTIC QUEUE MODEL: assumes
                  our bid at p_in fills whenever any aggressive seller arrives
                  in the 5-min window. In reality this only holds when we
                  have queue priority at the touch (= bid_nf_price ≤ p_in).
                  For the realistic queue answer, see
                  `wa.chase_bid_cutoff_sweep(audit, cutoffs_cents=[0])`.

                  Entry fills at EXACTLY p_in only when an aggressive seller
                  arrives (entry_next_opp_dir_source == 'next_fill'). Crosses
                  where the bid sits unhit are DROPPED. TP exit at exactly
                  p_out under `exit_passive`:
                    'optimistic' — TP fills if price reached p_out
                    'strict'     — also require a real aggressive-buy at exit

                  PnL formulas (per filled trade):
                    TP:        p_out - p_in
                    hold_win:  1.0   - p_in
                    hold_chop: -p_in

    Returns (trades_df, n_dropped_nan_resolution). NaN resolution rows are
    dropped (would otherwise be silently coerced to 0 = chop).
    """
    if pnl_model not in ("taker", "passive"):
        raise ValueError(f"pnl_model must be 'taker' or 'passive', got {pnl_model!r}")
    in_col  = f"fc_{int(round(p_in  * 100)):03d}"
    out_col = f"fc_{int(round(p_out * 100)):03d}"
    entered = p_wide[in_col].notna() & (p_wide["min_price"] < p_in)
    df = (p_wide[entered]
          .sort_values(in_col)
          .drop_duplicates("market_id", keep="first")
          .copy())

    n_before = len(df)
    df = df.dropna(subset=["resolution"]).copy()
    n_dropped_nan_resolution = n_before - len(df)

    res_ts = (inst[["market_id", "outcome_token_id", "resolution_ts"]]
              .drop_duplicates(["market_id", "outcome_token_id"]))
    df = df.merge(res_ts, on=["market_id", "outcome_token_id"], how="left")

    fc_in  = df[in_col]
    fc_out = df[out_col]
    tp_mask    = (fc_out.notna() & (fc_out > fc_in)).values
    resolution = df["resolution"].astype(float).values

    # Build audit log (entry/exit BID + ASK with fallbacks) via the shared
    # helper in weather_analysis. Resets index of df_for_audit so anchor_idx
    # lines up 1:1 with the trades rows we're about to build.
    df_for_audit = df.reset_index(drop=True)
    audit = wa._build_audit_log(
        df_for_audit, in_col, out_col, p_in, p_out, tp_mask,
        min_seconds=min_seconds, max_seconds=max_seconds,
        fallback_cents=fallback_cents,
        assumed_spread_cents=assumed_spread_cents,
        trades_glob=trades_glob,
    )

    if pnl_model == "taker":
        entry_ask = audit["entry_ask_price"].astype(float).values
        exit_bid  = audit["exit_bid_price"].astype(float).values
        pps = np.where(
            tp_mask,
            exit_bid - entry_ask,
            np.where(resolution == 1, 1.0 - entry_ask, -entry_ask),
        )
        df_for_audit["entry_ask_price"]  = entry_ask
        df_for_audit["entry_ask_source"] = audit["entry_ask_source"].values
        df_for_audit["exit_bid_price"]   = exit_bid
        df_for_audit["exit_bid_source"]  = audit["exit_bid_source"].values
        df_for_audit["entry_filled"]     = True   # taker always fills
        df_for_audit["pnl_per_share"]    = pps
        df_for_audit["bucket"] = np.where(
            tp_mask, "tp",
            np.where(resolution == 1, "hold_win", "hold_chop"),
        )

    else:  # pnl_model == "passive"
        aug_audit, _ = wa.passive_pnl_from_audit(audit, exit_passive=exit_passive)
        df_for_audit["entry_filled"] = aug_audit["entry_filled_passive"].values
        df_for_audit["exit_filled"]  = aug_audit["exit_filled_passive"].values
        df_for_audit["pnl_per_share"] = aug_audit["pnl_passive"].values
        df_for_audit["bucket"]        = aug_audit["outcome_passive"].values
        # Pricing under passive is the posted level itself; expose for share-conversion.
        df_for_audit["entry_ask_price"]  = p_in
        df_for_audit["entry_ask_source"] = "passive_at_p_in"
        df_for_audit["exit_bid_price"]   = np.where(
            aug_audit["exit_filled_passive"].values, p_out, np.nan)
        df_for_audit["exit_bid_source"]  = np.where(
            aug_audit["exit_filled_passive"].values, "passive_at_p_out", "n/a")

    df_for_audit["entry_ts"] = fc_in.values
    df_for_audit["exit_ts"]  = pd.to_datetime(
        np.where(tp_mask, fc_out.values, df_for_audit["resolution_ts"].values)
    )
    df_for_audit["p_in"], df_for_audit["p_out"] = p_in, p_out
    df_for_audit["pnl_model"] = pnl_model

    out = df_for_audit[[
        "market_id", "slug_family", "end_ts", "entry_ts", "exit_ts",
        "bucket", "entry_filled",
        "entry_ask_price", "entry_ask_source",
        "exit_bid_price",  "exit_bid_source",
        "pnl_per_share", "p_in", "p_out", "pnl_model",
    ]]
    if pnl_model == "passive":
        # Drop unfilled crosses — they have no risk, no PnL, no capital deployed.
        out = out[out["entry_filled"]].copy()

    return out, n_dropped_nan_resolution


def count_active_at(
    universe: pd.DataFrame,
    ts: pd.Series,
    window_hours: int = 24,
) -> np.ndarray:
    """For each ts, count distinct market_ids whose [end-window, end] contains ts.

    universe has columns market_id, end_ts.
    """
    win = pd.Timedelta(hours=window_hours)
    starts = (universe["end_ts"] - win).values
    ends   = universe["end_ts"].values
    t_arr  = ts.values

    s_sorted = np.sort(starts)
    e_sorted = np.sort(ends)
    started = np.searchsorted(s_sorted, t_arr, side="right")
    ended   = np.searchsorted(e_sorted, t_arr, side="left")
    return started - ended  # markets that started but haven't ended at t


def build_city_universe(p_wide: pd.DataFrame) -> pd.DataFrame:
    """One row per (city, end_ts) — collapses the ~7 range buckets per city per day to one."""
    uni = (p_wide[["market_id", "end_ts", "slug_family"]]
           .drop_duplicates("market_id"))
    uni = uni[uni["slug_family"].str.startswith(("highest-temperature",
                                                  "lowest-temperature"))].copy()
    uni["city"] = uni["slug_family"].map(city_from_slug_family)
    return uni.drop_duplicates(["city", "end_ts"])[["city", "end_ts"]]


def backtest(p_in: float = 0.60, p_out: float = 0.90,
             min_seconds: int = wa.DEFAULT_MIN_SECONDS,
             max_seconds: int = wa.DEFAULT_MAX_SECONDS,
             fallback_cents: float = wa.DEFAULT_FALLBACK_CENTS,
             assumed_spread_cents: float = wa.DEFAULT_SPREAD_CENTS,
             trades_glob: str = wa.DEFAULT_TRADES_GLOB,
             pnl_model: str = "taker",
             exit_passive: str = "optimistic",
             window_hours: int = 24, cap_N_at: int | None = None,
             one_trade_per_city: bool = True,
             max_notional_pct: float | None = 0.02):
    """N = active distinct cities at entry_ts; size = min(1/N, max_notional_pct).

    one_trade_per_city (default True): only the first market in each
    (city, resolution_date) to cross p_in is taken. Required for honest
    accounting — the ~7 buckets per city are mutually exclusive, so taking
    >1 of them is a forced-loss spread (pay >$1 for max $1 payout). Keyed
    on end_ts (resolution day), not entry_ts — entries that straddle
    midnight UTC still collapse to one trade per (city, resolution_date).

    max_notional_pct (default 0.02 = 2%): per-trade size cap. Prevents
    small-N blowups — e.g. when only 3 cities are active, 1/N = 33% would
    risk a -33% single-day loss on a chop. None = no cap.

    Equity is non-compounding (fixed-fractional on flat $1): equity =
    1 + cumsum(daily ret). `cagr_pct` is the arithmetic-annualized total
    return (not geometric CAGR). The annualization span is the full
    weather universe end_ts range, so Sharpe / CAGR are comparable across
    grid cells with different trade-active ranges.
    """
    res = wa.load_weather_results()
    INST = res["inst"]
    p_wide = wa.pivot_inst_to_wide(INST)
    uni = build_city_universe(p_wide)

    trades, n_dropped_nan_resolution = build_trades(
        p_wide, INST, p_in, p_out,
        min_seconds=min_seconds, max_seconds=max_seconds,
        fallback_cents=fallback_cents,
        assumed_spread_cents=assumed_spread_cents,
        trades_glob=trades_glob,
        pnl_model=pnl_model,
        exit_passive=exit_passive,
    )
    trades = trades.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)

    trades["city"] = trades["slug_family"].map(city_from_slug_family)
    if one_trade_per_city:
        trades["end_date"] = pd.to_datetime(trades["end_ts"]).dt.floor("D")
        trades = trades.drop_duplicates(["city", "end_date"], keep="first").reset_index(drop=True)

    trades["N_active"] = count_active_at(uni, trades["entry_ts"], window_hours)
    if cap_N_at is not None:
        trades["N_active"] = trades["N_active"].clip(lower=cap_N_at)
    trades["N_active"] = trades["N_active"].clip(lower=1)
    trades["notional_pct"] = 1.0 / trades["N_active"]
    if max_notional_pct is not None:
        trades["notional_pct"] = trades["notional_pct"].clip(upper=max_notional_pct)
    # Per-share PnL is in absolute $/share; convert to $-on-bankroll by
    # dividing by the actual entry price paid (not p_in — the cap on shares
    # bought with $1 of notional is 1/entry_ask).
    trades["pnl_pct"] = trades["notional_pct"] * (trades["pnl_per_share"] / trades["entry_ask_price"])

    # Daily aggregation by entry date.
    trades["entry_date"] = trades["entry_ts"].dt.floor("D")
    daily = (trades.groupby("entry_date")
                   .agg(n_trades=("pnl_pct", "size"),
                        gross_notional=("notional_pct", "sum"),
                        ret=("pnl_pct", "sum"))
                   .reset_index())

    # Global span from the universe (all weather markets' resolution days),
    # not from the trades — so Sharpe / annualized return are comparable
    # across (p_in, p_out) grid cells with different trade ranges.
    universe_start = pd.to_datetime(uni["end_ts"]).dt.floor("D").min()
    universe_end   = pd.to_datetime(uni["end_ts"]).dt.floor("D").max()
    full_idx = pd.date_range(universe_start, universe_end, freq="D")
    daily = daily.set_index("entry_date").reindex(full_idx, fill_value=0)
    daily.index.name = "date"

    equity = 1.0 + daily["ret"].cumsum()  # non-compounding fixed-fractional
    peak   = equity.cummax()
    dd     = equity / peak - 1.0

    mu_d   = daily["ret"].mean()
    sd_d   = daily["ret"].std(ddof=1)
    sharpe = (mu_d / sd_d) * np.sqrt(365) if sd_d > 0 else np.nan
    total_ret = equity.iloc[-1] - 1.0
    annualization_days = (daily.index[-1] - daily.index[0]).days or 1
    cagr = total_ret * (365 / annualization_days)  # arithmetic-annualized

    tp_rows = trades["bucket"] == "tp"
    summary = {
        "p_in": p_in, "p_out": p_out,
        "pnl_model": pnl_model,
        "exit_passive": exit_passive if pnl_model == "passive" else None,
        "min_seconds": min_seconds, "max_seconds": max_seconds,
        "fallback_cents": fallback_cents,
        "assumed_spread_cents": assumed_spread_cents,
        "n_trades": int(len(trades)),
        "n_dropped_nan_resolution": int(n_dropped_nan_resolution),
        "n_days": int(len(daily)),
        "annualization_days": int(annualization_days),
        "active_days": int((daily["n_trades"] > 0).sum()),
        "avg_N_active": float(trades["N_active"].mean()),
        "median_N_active": float(trades["N_active"].median()),
        "avg_notional_per_trade": float(trades["notional_pct"].mean()),
        "p_tp": float(tp_rows.mean()),
        "p_hold_win": float((trades["bucket"] == "hold_win").mean()),
        "p_hold_chop": float((trades["bucket"] == "hold_chop").mean()),
        "win_rate": float((trades["pnl_per_share"] > 0).mean()),
        "mean_entry_ask_cents_above_p_in": float((trades["entry_ask_price"] - p_in).mean() * 100),
        "mean_exit_bid_cents_below_p_out": float(((p_out - trades.loc[tp_rows, "exit_bid_price"]).mean() * 100) if tp_rows.any() else float("nan")),
        "entry_ask_fallback_rate": float((trades["entry_ask_source"] == "fallback").mean()),
        "exit_bid_fallback_rate":  float(((trades.loc[tp_rows, "exit_bid_source"] == "fallback").mean()) if tp_rows.any() else float("nan")),
        "avg_pnl_pct_per_trade_bps": float(trades["pnl_pct"].mean() * 10_000),
        "total_return_pct": float(total_ret * 100),
        "cagr_pct": float(cagr * 100),
        "daily_mean_bps": float(mu_d * 10_000),
        "daily_std_bps": float(sd_d * 10_000),
        "sharpe_ann": float(sharpe),
        "max_dd_pct": float(dd.min() * 100),
    }
    return summary, trades, daily, equity, dd


def grid_backtest(
    barriers: list[float] | None = None,
    min_seconds: int = wa.DEFAULT_MIN_SECONDS,
    max_seconds: int = wa.DEFAULT_MAX_SECONDS,
    fallback_cents: float = wa.DEFAULT_FALLBACK_CENTS,
    assumed_spread_cents: float = wa.DEFAULT_SPREAD_CENTS,
    trades_glob: str = wa.DEFAULT_TRADES_GLOB,
    window_hours: int = 24,
    one_trade_per_city: bool = True,
    max_notional_pct: float | None = 0.02,
) -> pd.DataFrame:
    """Sweep (p_in, p_out) and return one summary row per combination.

    Slippage is the same next-fill model for every cell — vary it explicitly
    via the (min_seconds, max_seconds, fallback_cents, assumed_spread_cents)
    parameters if you want a sensitivity grid.

    Heavy: each cell runs a fresh next-fill batch lookup over the 1B-row
    trades parquet. Expect several minutes for the full 21-pair sweep.
    """
    if barriers is None:
        barriers = wa.BARRIERS_EE
    res    = wa.load_weather_results()
    INST   = res["inst"]
    p_wide = wa.pivot_inst_to_wide(INST)
    uni    = build_city_universe(p_wide)

    universe_start = pd.to_datetime(uni["end_ts"]).dt.floor("D").min()
    universe_end   = pd.to_datetime(uni["end_ts"]).dt.floor("D").max()
    full_idx = pd.date_range(universe_start, universe_end, freq="D")
    annualization_days = (full_idx[-1] - full_idx[0]).days or 1

    rows = []
    for p_in in barriers:
        for p_out in barriers:
            if not (p_in < p_out):
                continue
            tr, n_dropped_nan_resolution = build_trades(
                p_wide, INST, p_in, p_out,
                min_seconds=min_seconds, max_seconds=max_seconds,
                fallback_cents=fallback_cents,
                assumed_spread_cents=assumed_spread_cents,
                trades_glob=trades_glob,
            )
            tr = tr.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)
            if len(tr) == 0:
                continue
            tr["city"] = tr["slug_family"].map(city_from_slug_family)
            if one_trade_per_city:
                tr["end_date"] = pd.to_datetime(tr["end_ts"]).dt.floor("D")
                tr = tr.drop_duplicates(["city", "end_date"], keep="first").reset_index(drop=True)
            tr["N_active"]     = count_active_at(uni, tr["entry_ts"], window_hours).clip(min=1)
            tr["notional_pct"] = 1.0 / tr["N_active"]
            if max_notional_pct is not None:
                tr["notional_pct"] = tr["notional_pct"].clip(upper=max_notional_pct)
            tr["pnl_pct"]      = tr["notional_pct"] * (tr["pnl_per_share"] / tr["entry_ask_price"])

            tr["entry_date"] = tr["entry_ts"].dt.floor("D")
            daily = (tr.groupby("entry_date")["pnl_pct"].sum())
            daily = daily.reindex(full_idx, fill_value=0)

            equity = 1.0 + daily.cumsum()
            dd     = equity / equity.cummax() - 1.0
            mu, sd = daily.mean(), daily.std(ddof=1)
            sharpe = (mu / sd) * np.sqrt(365) if sd > 0 else np.nan
            total_ret = equity.iloc[-1] - 1.0
            cagr = total_ret * (365 / annualization_days)

            tp_rows = tr["bucket"] == "tp"
            rows.append({
                "p_in": p_in, "p_out": p_out,
                "min_seconds": min_seconds, "max_seconds": max_seconds,
                "fallback_cents": fallback_cents,
                "assumed_spread_cents": assumed_spread_cents,
                "n_trades": int(len(tr)),
                "n_dropped_nan_resolution": int(n_dropped_nan_resolution),
                "annualization_days": int(annualization_days),
                "avg_N": float(tr["N_active"].mean()),
                "med_N": float(tr["N_active"].median()),
                "avg_notional_pct": float(tr["notional_pct"].mean() * 100),
                "p_tp": float(tp_rows.mean()),
                "p_hold_chop": float((tr["bucket"] == "hold_chop").mean()),
                "entry_ask_fallback_rate": float((tr["entry_ask_source"] == "fallback").mean()),
                "exit_bid_fallback_rate":  float(((tr.loc[tp_rows, "exit_bid_source"] == "fallback").mean()) if tp_rows.any() else float("nan")),
                "avg_pnl_bps": float(tr["pnl_pct"].mean() * 10_000),
                "total_ret_pct": float(total_ret * 100),
                "cagr_pct": float(cagr * 100),
                "sharpe_ann": float(sharpe),
                "max_dd_pct": float(dd.min() * 100),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Headline: one trade per (city, date), p_in=0.60, p_out=0.90, next-fill slippage ===")
    s, _, _, eq, _ = backtest(p_in=0.60, p_out=0.90)
    for k, v in s.items():
        print(f"  {k:32s} {v:>14.4f}" if isinstance(v, float) else f"  {k:32s} {v:>14}")
    print(f"  equity_final                     {eq.iloc[-1]:>14.4f}")

    print("\n=== Grid sweep (one trade per city, 1/N city sizing, next-fill slippage) ===")
    grid = grid_backtest()
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(grid.round({"avg_N": 1, "med_N": 1, "avg_notional_pct": 3,
                          "p_tp": 3, "p_hold_chop": 3,
                          "entry_ask_fallback_rate": 3, "exit_bid_fallback_rate": 3,
                          "avg_pnl_bps": 2, "total_ret_pct": 1, "cagr_pct": 1,
                          "sharpe_ann": 2, "max_dd_pct": 1}).to_string(index=False))

    print("\n=== Top 10 by Sharpe ===")
    print(grid.sort_values("sharpe_ann", ascending=False).head(10).round(2).to_string(index=False))
