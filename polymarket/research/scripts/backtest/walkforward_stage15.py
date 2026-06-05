"""Phase 5 Stage 1.5 walk-forward engine.

Changes vs Stage 1 (per phase5_design.md updates):

1. Cohort filter is percentile-based with absolute floors + active-trader gate
   (cohort_filters_stage15.py).
2. TopK=10 selection within qualified set per refresh, ranked by
   mkt_total_pnl × mkt_profit_factor.
3. Per-trade filter: market.volume (lifetime from Gamma snapshot) > $50,000.
4. Per-leader monthly signal cap: 10 trades / leader / month (insurance).
5. Sizing tightened:
   - Rule A (fixed_pct): 1% of strategy_capital (was 2%)
   - Rule B (leader_proportional): cap at 2.5% of strategy_capital (was 5%)
   - Max 5 concurrent positions per leader (unchanged)
   - Max 10% bankroll exposure per leader (was 20%)
   - Category cap not enforced (no category data; documented limitation).
6. Slippage model — NEXT-FILL via SQL (per design 3.5):
   - Find next OrderFilled in same market+outcome where leader is not the
     buyer-side (for buys) or seller-side (for sells)
   - Window: [leader_ts + 15s, leader_ts + 300s]
   - Fallback: 3¢ adverse slippage if no qualifying fill
   - Audit log records `slippage_source` ∈ {'next_fill', 'fallback'}
   - This is a model UPGRADE vs Stage 1's constant-cents formulation.

Audit log adds:
   n_qualified_at_refresh  INT
   selected_leader_rank    INT  (1..10 within cohort/refresh)
   slippage_source         VARCHAR  ('next_fill' or 'fallback')
"""
from __future__ import annotations

import heapq
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd
from dateutil.relativedelta import relativedelta

from scripts.backtest.cohort_filters_stage15 import select_top_k

ROOT = Path(__file__).resolve().parents[2]
CLOSED_POS = str(ROOT / "data" / "closed_positions.parquet")
TRADERS = str(ROOT / "data" / "traders.parquet")
BANKROLL = str(ROOT / "data" / "bankroll_timeseries.parquet")
BACKTESTS_DIR = ROOT / "data" / "backtests" / "stage15"
TRADES_GLOB = str(ROOT / "data" / "trades" / "trades_delta_shard*.parquet")
SEED = str(ROOT / "data" / "trades" / "trades_seed.parquet")
MARKETS_PATH = str(sorted((ROOT / "data" / "markets").glob("markets_*.parquet"))[-1])

BUCKET_DAYS = {"2d": 2, "7d": 7, "30d": 30, "60d": 60}

# Stage 1.5 caps
TOP_K = 10
PER_LEADER_MONTHLY_CAP = 10
MAX_OPEN_PER_LEADER = 5
MAX_EXPOSURE_PER_LEADER_PCT = 0.10   # was 0.20 in Stage 1
MAX_POSITION_SIZE_PCT_FIXED = 0.01   # was 0.02
MAX_POSITION_SIZE_PCT_PROP = 0.025   # was 0.05
MIN_MARKET_VOLUME_USD = 50_000.0     # was $10k

# Slippage model
SLIP_MIN_SECONDS = 15
SLIP_MAX_SECONDS = 300
FALLBACK_SLIPPAGE_CENTS = 3.0


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")
    # markets_tokens table for outcome_index lookup + resolution prices
    con.execute(f"""
        CREATE OR REPLACE TABLE markets_tokens AS
        SELECT CAST(m.id AS VARCHAR) AS market_id,
               m.neg_risk, m.closed,
               TRY_CAST(m.end_date AS TIMESTAMP) AS end_date,
               m.outcome_prices, m.clob_token_ids, m.volume AS market_volume,
               r.i AS outcome_index,
               m.clob_token_ids[r.i] AS outcome_token_id
        FROM read_parquet('{MARKETS_PATH}') m,
             range(1, len(m.clob_token_ids) + 1) AS r(i)
        WHERE len(m.clob_token_ids) > 0
    """)
    return con


def _query_signals_with_next_fill(
    con: duckdb.DuckDBPyConnection,
    addresses: list[str],
    T_start: datetime,
    T_end: datetime,
    bucket_days: int,
) -> pd.DataFrame:
    """Two-pass query for performance on dense data months (2026 had 280M
    fills/month). Pass 1: small — get leader signals (filter by qual_addrs).
    Pass 2: candidates filtered to leader-touched markets only.

    Direction-aware next-fill: when leader bought outcome tokens, find next
    fill where buyer-side ≠ leader; symmetric for sells. Fallback 3¢ if no
    qualifying fill within [15s, 300s].
    """
    if not addresses:
        return pd.DataFrame()

    con.execute("CREATE OR REPLACE TEMP TABLE qual_addrs (address VARCHAR)")
    con.executemany("INSERT INTO qual_addrs VALUES (?)", [(a,) for a in addresses])

    cand_end = T_end + pd.Timedelta(minutes=10)
    t_start_str = T_start.strftime("%Y-%m-%d %H:%M:%S")
    t_end_str = T_end.strftime("%Y-%m-%d %H:%M:%S")
    cand_end_str = cand_end.strftime("%Y-%m-%d %H:%M:%S")

    # --------- Pass 1: leader signals (small) ---------
    # Pre-filter raw_trades by leader address — narrows from ~280M to
    # likely tens of thousands of rows.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE leader_signals AS
        WITH leader_rows AS (
            SELECT * FROM read_parquet('{TRADES_GLOB}')
            WHERE timestamp >= TIMESTAMP '{t_start_str}'
              AND timestamp <  TIMESTAMP '{t_end_str}'
              AND maker IN (SELECT address FROM qual_addrs)
              AND taker IS NOT NULL AND market_id IS NOT NULL
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{SEED}')
            WHERE timestamp >= TIMESTAMP '{t_start_str}'
              AND timestamp <  TIMESTAMP '{t_end_str}'
              AND maker IN (SELECT address FROM qual_addrs)
              AND taker IS NOT NULL AND market_id IS NOT NULL
        ),
        with_outcome AS (
            SELECT *,
                CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END
                    AS outcome_token_id,
                CASE WHEN maker_asset_id = '0' THEN 'buy' ELSE 'sell' END
                    AS leader_direction
            FROM leader_rows
            WHERE maker <> taker
        )
        SELECT
            wo.timestamp AS leader_ts,
            wo.market_id, wo.outcome_token_id,
            mt.outcome_index, mt.neg_risk,
            wo.maker AS leader_address,
            wo.maker_asset_id, wo.leader_direction,
            wo.price AS leader_price,
            wo.usd_amount AS leader_trade_usd,
            wo.token_amount AS leader_token_amount,
            wo.transaction_hash,
            mt.end_date AS resolution_date,
            CAST(mt.outcome_prices[mt.outcome_index] AS DOUBLE) AS resolution_price,
            date_diff('day', wo.timestamp, mt.end_date) AS days_to_resolution,
            mt.market_volume
        FROM with_outcome wo
        JOIN markets_tokens mt
          ON mt.market_id = wo.market_id AND mt.outcome_token_id = wo.outcome_token_id
        WHERE mt.closed = TRUE
          AND mt.end_date IS NOT NULL
          AND mt.end_date > wo.timestamp
          AND date_diff('day', wo.timestamp, mt.end_date) < {bucket_days}
          AND mt.market_volume >= {MIN_MARKET_VOLUME_USD}
    """)

    n_leader = con.sql("SELECT count(*) FROM leader_signals").fetchone()[0]
    if n_leader == 0:
        return pd.DataFrame()

    # --------- Pass 2: candidates filtered to leader-touched (market_id, outcome_token_id) ---------
    # This is the load-bearing optimisation. Without the market filter, we'd
    # scan all ~200M trades in the OOS window; with it, we scan only the
    # subset that matters (typically a few thousand markets).
    con.execute("""
        CREATE OR REPLACE TEMP TABLE leader_markets AS
        SELECT DISTINCT market_id, outcome_token_id FROM leader_signals
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE candidates AS
        WITH rt AS (
            SELECT timestamp, market_id, maker, taker, maker_asset_id, taker_asset_id, price
            FROM read_parquet('{TRADES_GLOB}')
            WHERE timestamp >= TIMESTAMP '{t_start_str}'
              AND timestamp <  TIMESTAMP '{cand_end_str}'
              AND market_id IN (SELECT market_id FROM leader_markets)
              AND maker IS NOT NULL AND taker IS NOT NULL AND maker <> taker
            UNION ALL BY NAME
            SELECT timestamp, market_id, maker, taker, maker_asset_id, taker_asset_id, price
            FROM read_parquet('{SEED}')
            WHERE timestamp >= TIMESTAMP '{t_start_str}'
              AND timestamp <  TIMESTAMP '{cand_end_str}'
              AND market_id IN (SELECT market_id FROM leader_markets)
              AND maker IS NOT NULL AND taker IS NOT NULL AND maker <> taker
        )
        SELECT
            rt.timestamp, rt.market_id,
            CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END AS outcome_token_id,
            CASE WHEN rt.maker_asset_id = '0' THEN rt.maker ELSE rt.taker END AS buyer_side,
            CASE WHEN rt.maker_asset_id = '0' THEN rt.taker ELSE rt.maker END AS seller_side,
            rt.price
        FROM rt
        WHERE (rt.market_id, CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END)
              IN (SELECT market_id, outcome_token_id FROM leader_markets)
    """)

    # --------- Final JOIN with row_number to pick nearest match per signal ---------
    sql = f"""
    WITH next_fill_join AS (
        SELECT
            ls.*, c.timestamp AS nf_ts, c.price AS nf_price,
            row_number() OVER (
                PARTITION BY ls.leader_ts, ls.market_id,
                             ls.outcome_token_id, ls.leader_address
                ORDER BY c.timestamp
            ) AS rn
        FROM leader_signals ls
        LEFT JOIN candidates c
          ON c.market_id = ls.market_id
         AND c.outcome_token_id = ls.outcome_token_id
         AND c.timestamp >  ls.leader_ts + INTERVAL '{SLIP_MIN_SECONDS} seconds'
         AND c.timestamp <= ls.leader_ts + INTERVAL '{SLIP_MAX_SECONDS} seconds'
         AND (CASE WHEN ls.leader_direction = 'buy'
                   THEN c.buyer_side  <> ls.leader_address
                   ELSE c.seller_side <> ls.leader_address END)
    )
    SELECT
        leader_ts AS timestamp,
        market_id, outcome_token_id, outcome_index, neg_risk,
        leader_address, leader_direction, maker_asset_id,
        leader_price, leader_trade_usd, leader_token_amount,
        transaction_hash, resolution_date, resolution_price, days_to_resolution,
        market_volume, nf_price,
        CASE WHEN nf_price IS NOT NULL THEN 'next_fill' ELSE 'fallback' END AS slippage_source
    FROM next_fill_join
    WHERE rn = 1 OR rn IS NULL
    ORDER BY leader_ts
    """
    return con.sql(sql).fetchdf()


def _bankroll_snapshot(con, addresses, T):
    if not addresses:
        return {}
    con.execute("CREATE OR REPLACE TEMP TABLE bk_addrs (address VARCHAR)")
    con.executemany("INSERT INTO bk_addrs VALUES (?)", [(a,) for a in addresses])
    sql = f"""
    SELECT address, bankroll_30d_prior
    FROM (
        SELECT address, bankroll_30d_prior,
               row_number() OVER (PARTITION BY address ORDER BY date DESC) AS rn
        FROM read_parquet('{BANKROLL}')
        WHERE address IN (SELECT address FROM bk_addrs)
          AND date < DATE '{T.strftime("%Y-%m-%d")}'
    )
    WHERE rn = 1
    """
    df = con.sql(sql).fetchdf()
    return dict(zip(df["address"], df["bankroll_30d_prior"]))


@dataclass
class _State:
    open_per_leader: dict
    exposure_per_leader: dict
    closure_queue: list


def _close_resolved(state: _State, up_to_ts) -> None:
    while state.closure_queue and state.closure_queue[0][0] <= up_to_ts:
        _, leader, _, size = heapq.heappop(state.closure_queue)
        state.open_per_leader[leader] = max(0, state.open_per_leader[leader] - 1)
        state.exposure_per_leader[leader] = max(0.0, state.exposure_per_leader[leader] - size)


def run_backtest_stage15(
    cohort_name: str,
    cohort_fn,
    resolution_bucket: str,
    sizing_rule: str,
    test_window_start: date,
    test_window_end: date,
    window_label: str | None = None,
    strategy_capital_usd: float = 100_000.0,
) -> Path:
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

    if window_label is None:
        window_label = f"{test_window_start:%Y%m}_{test_window_end:%Y%m}"
    run_id = f"{cohort_name}__{resolution_bucket}__{sizing_rule}__{window_label}__stage15"
    print(f"\n=== {run_id} ===", flush=True)
    t_run = time.time()

    bucket_days = BUCKET_DAYS[resolution_bucket]
    con = _connect()

    refresh_dates = []
    d = datetime.combine(test_window_start, datetime.min.time())
    end_dt = datetime.combine(test_window_end, datetime.min.time())
    while d < end_dt:
        refresh_dates.append(d)
        d = d + relativedelta(months=1)

    state = _State(defaultdict(int), defaultdict(float), [])
    audit_rows: list[dict] = []
    skip_counts: dict = defaultdict(int)
    refresh_log: list = []

    for T in refresh_dates:
        t0 = time.time()
        qual_df = cohort_fn(con, T)
        topk_df = select_top_k(qual_df, TOP_K)
        rank_map = dict(zip(topk_df["address"], topk_df["rank"])) if not topk_df.empty else {}
        leader_addrs = topk_df["address"].tolist()
        bankroll_at_T = (_bankroll_snapshot(con, leader_addrs, T)
                         if sizing_rule == "leader_proportional" else {})

        T_oos_end = T + relativedelta(months=1)
        signals = _query_signals_with_next_fill(con, leader_addrs, T, T_oos_end, bucket_days)

        n_qualified = int(len(qual_df))
        refresh_log.append({
            "refresh_date": T.date().isoformat(),
            "n_qualified": n_qualified,
            "n_selected": len(leader_addrs),
            "n_signals_oos": int(len(signals)),
            "elapsed_s": round(time.time() - t0, 2),
        })

        if signals.empty:
            continue

        # Monthly cap counter (resets per refresh / month)
        monthly_signal_count: dict = defaultdict(int)

        for trade in signals.itertuples(index=False):
            ts = trade.timestamp
            _close_resolved(state, ts)

            leader = trade.leader_address
            # Per-leader monthly cap
            if monthly_signal_count[leader] >= PER_LEADER_MONTHLY_CAP:
                skip_counts["leader_monthly_cap"] += 1; continue
            # Concurrency caps
            if state.open_per_leader[leader] >= MAX_OPEN_PER_LEADER:
                skip_counts["leader_concurrent_cap"] += 1; continue
            if state.exposure_per_leader[leader] >= strategy_capital_usd * MAX_EXPOSURE_PER_LEADER_PCT:
                skip_counts["leader_exposure_cap"] += 1; continue

            # Sizing
            if sizing_rule == "fixed_pct":
                size = strategy_capital_usd * MAX_POSITION_SIZE_PCT_FIXED
            else:
                br = bankroll_at_T.get(leader)
                if not br or br <= 0:
                    skip_counts["no_bankroll"] += 1; continue
                leader_fraction = float(trade.leader_trade_usd) / br
                size = leader_fraction * strategy_capital_usd
                size = min(size, strategy_capital_usd * MAX_POSITION_SIZE_PCT_PROP)
            if size <= 0:
                skip_counts["zero_size"] += 1; continue

            # Next-fill or fallback price
            leader_direction = trade.leader_direction  # 'buy' | 'sell'
            if trade.nf_price is not None and not pd.isna(trade.nf_price):
                copy_price = float(trade.nf_price)
                slip_source = "next_fill"
            else:
                fallback = FALLBACK_SLIPPAGE_CENTS / 100.0
                copy_price = (float(trade.leader_price) + fallback
                              if leader_direction == "buy"
                              else float(trade.leader_price) - fallback)
                slip_source = "fallback"

            if copy_price <= 0 or copy_price >= 1:
                skip_counts["bad_copy_price"] += 1; continue

            slippage_cents = abs(copy_price - float(trade.leader_price)) * 100.0
            copy_token_amount = size / copy_price
            res_price = float(trade.resolution_price)
            if leader_direction == "buy":
                copy_pnl = (res_price - copy_price) * copy_token_amount
            else:
                copy_pnl = (copy_price - res_price) * copy_token_amount

            neg_risk_val = bool(trade.neg_risk) if trade.neg_risk is not None and not pd.isna(trade.neg_risk) else False

            audit_rows.append({
                "run_id": run_id,
                "cohort": cohort_name,
                "refresh_date": T.date(),
                "leader_address": leader,
                "selected_leader_rank": int(rank_map.get(leader, -1)),
                "n_qualified_at_refresh": n_qualified,
                "market_id": trade.market_id,
                "outcome_index": int(trade.outcome_index),
                "neg_risk": neg_risk_val,
                "market_volume": float(trade.market_volume) if trade.market_volume is not None else None,
                "trade_timestamp": ts,
                "resolution_date": trade.resolution_date,
                "days_to_resolution": int(trade.days_to_resolution),
                "resolution_bucket": resolution_bucket,
                "leader_direction": leader_direction,
                "leader_trade_usd": float(trade.leader_trade_usd),
                "leader_price": float(trade.leader_price),
                "copy_price": copy_price,
                "slippage_cents": slippage_cents,
                "slippage_source": slip_source,
                "copy_size_usd": size,
                "copy_token_amount": copy_token_amount,
                "position_resolution": res_price,
                "copy_pnl_usd": copy_pnl,
                "sizing_rule": sizing_rule,
            })
            heapq.heappush(state.closure_queue,
                           (trade.resolution_date, leader, trade.market_id, size))
            state.open_per_leader[leader] += 1
            state.exposure_per_leader[leader] += size
            monthly_signal_count[leader] += 1

    # Write audit
    audit_path = BACKTESTS_DIR / f"{run_id}.parquet"
    if audit_rows:
        pd.DataFrame(audit_rows).to_parquet(audit_path, compression="zstd", index=False)
    else:
        pd.DataFrame(columns=[
            "run_id","cohort","refresh_date","leader_address","selected_leader_rank",
            "n_qualified_at_refresh","market_id","outcome_index","neg_risk",
            "market_volume","trade_timestamp","resolution_date","days_to_resolution",
            "resolution_bucket","leader_direction","leader_trade_usd","leader_price",
            "copy_price","slippage_cents","slippage_source","copy_size_usd",
            "copy_token_amount","position_resolution","copy_pnl_usd","sizing_rule",
        ]).to_parquet(audit_path, compression="zstd", index=False)

    summary = _compute_summary(
        audit_rows, skip_counts, refresh_log,
        run_id, cohort_name, resolution_bucket, sizing_rule,
        test_window_start, test_window_end, strategy_capital_usd,
    )
    summary_path = BACKTESTS_DIR / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    elapsed = time.time() - t_run
    print(f"  → {len(audit_rows):,} signals, "
          f"PnL=${summary['total_pnl_usd']:,.0f}, "
          f"Sharpe={summary['sharpe_monthly']:.2f}, "
          f"deploy={summary['deployment_ratio']:.0%}, "
          f"fallback={summary['slippage_fallback_pct']:.0%} "
          f"({elapsed:.1f}s)", flush=True)
    return audit_path


def _compute_summary(audit_rows, skip_counts, refresh_log,
                     run_id, cohort_name, resolution_bucket, sizing_rule,
                     test_window_start, test_window_end, strategy_capital_usd):
    summary = {
        "run_id": run_id,
        "cohort": cohort_name,
        "resolution_bucket": resolution_bucket,
        "sizing_rule": sizing_rule,
        "test_window_start": str(test_window_start),
        "test_window_end": str(test_window_end),
        "strategy_capital_usd": strategy_capital_usd,
        "stage15_params": {
            "top_k": TOP_K,
            "per_leader_monthly_cap": PER_LEADER_MONTHLY_CAP,
            "max_open_per_leader": MAX_OPEN_PER_LEADER,
            "max_exposure_per_leader_pct": MAX_EXPOSURE_PER_LEADER_PCT,
            "fixed_pct_size": MAX_POSITION_SIZE_PCT_FIXED,
            "leader_prop_cap_pct": MAX_POSITION_SIZE_PCT_PROP,
            "min_market_volume_usd": MIN_MARKET_VOLUME_USD,
            "slippage_window_seconds": [SLIP_MIN_SECONDS, SLIP_MAX_SECONDS],
            "fallback_slippage_cents": FALLBACK_SLIPPAGE_CENTS,
        },
        "n_signals": len(audit_rows),
        "skip_counts": dict(skip_counts),
        "refresh_log": refresh_log,
    }
    if not audit_rows:
        summary.update({
            "total_pnl_usd": 0.0, "total_volume_usd": 0.0,
            "win_rate": None, "profit_factor": None,
            "sharpe_monthly": 0.0, "sortino_monthly": 0.0,
            "max_drawdown_usd": 0.0, "deployment_ratio": 0.0,
            "signals_per_week": 0.0,
            "slippage_fallback_pct": 0.0,
            "slippage_avg_cents": 0.0,
            "slippage_p50_cents": 0.0,
            "avg_signal_size_usd": 0.0,
            "top_10_leaders_pnl": [], "bottom_10_leaders_pnl": [],
        })
        return summary

    df = pd.DataFrame(audit_rows)
    df["trade_timestamp"] = pd.to_datetime(df["trade_timestamp"])
    df["resolution_date"] = pd.to_datetime(df["resolution_date"])
    df["month"] = df["trade_timestamp"].dt.to_period("M")

    total_pnl = float(df["copy_pnl_usd"].sum())
    total_vol = float(df["copy_size_usd"].sum())
    winners = int((df["copy_pnl_usd"] > 0).sum())
    losers = int((df["copy_pnl_usd"] < 0).sum())
    wins_sum = float(df.loc[df["copy_pnl_usd"] > 0, "copy_pnl_usd"].sum())
    loss_sum_abs = float(-df.loc[df["copy_pnl_usd"] < 0, "copy_pnl_usd"].sum())

    monthly = df.groupby("month")["copy_pnl_usd"].sum()
    mean_m, std_m = float(monthly.mean()), float(monthly.std(ddof=0))
    downside = monthly[monthly < 0]
    down_std = float((downside ** 2).mean() ** 0.5) if len(downside) else 0.0
    sharpe_m = (mean_m / std_m * (12 ** 0.5)) if std_m > 0 else 0.0
    sortino_m = (mean_m / down_std * (12 ** 0.5)) if down_std > 0 else 0.0
    cum = monthly.cumsum()
    peak = cum.cummax()
    drawdown_usd = float((peak - cum).max())

    window_start_dt = pd.Timestamp(test_window_start)
    window_end_dt = pd.Timestamp(test_window_end)
    window_days = max(1, (window_end_dt - window_start_dt).days)
    intervals = sorted(
        ((max(window_start_dt, t), min(window_end_dt, max(t, r)))
         for t, r in zip(df["trade_timestamp"], df["resolution_date"]))
    )
    deployed_days = 0.0
    cur_s, cur_e = None, None
    for s, e in intervals:
        if cur_s is None or s > cur_e:
            if cur_s is not None:
                deployed_days += (cur_e - cur_s).total_seconds() / 86400
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_s is not None:
        deployed_days += (cur_e - cur_s).total_seconds() / 86400
    deployment_ratio = min(1.0, deployed_days / window_days)

    n_weeks = window_days / 7
    sig_per_wk = len(df) / n_weeks if n_weeks > 0 else 0.0

    by_leader = (df.groupby("leader_address")["copy_pnl_usd"]
                 .sum().sort_values(ascending=False))
    top10 = [{"address": a, "pnl": float(p)} for a, p in by_leader.head(10).items()]
    bot10 = [{"address": a, "pnl": float(p)} for a, p in by_leader.tail(10).items()]

    summary.update({
        "total_pnl_usd": total_pnl,
        "total_volume_usd": total_vol,
        "win_rate": (winners / (winners + losers)) if (winners + losers) > 0 else None,
        "profit_factor": (wins_sum / loss_sum_abs) if loss_sum_abs > 0 else None,
        "sharpe_monthly": sharpe_m,
        "sortino_monthly": sortino_m,
        "max_drawdown_usd": drawdown_usd,
        "n_distinct_leaders": int(df["leader_address"].nunique()),
        "n_distinct_markets": int(df["market_id"].nunique()),
        "negrisk_signal_share": float(df["neg_risk"].mean()),
        "deployment_ratio": deployment_ratio,
        "signals_per_week": sig_per_wk,
        "slippage_fallback_pct": float((df["slippage_source"] == "fallback").mean()),
        "slippage_avg_cents": float(df["slippage_cents"].mean()),
        "slippage_p50_cents": float(df["slippage_cents"].median()),
        "avg_signal_size_usd": float(df["copy_size_usd"].mean()),
        "top_10_leaders_pnl": top10,
        "bottom_10_leaders_pnl": bot10,
    })
    return summary
