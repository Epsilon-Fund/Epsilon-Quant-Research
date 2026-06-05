"""Phase 5 Stage 1 — walk-forward backtest engine.

Single entry point: run_backtest(...).

Pipeline per run:
  1. Generate monthly refresh dates within test_window.
  2. For each refresh date T:
     a. Call cohort_fn(con, T) → list of qualified addresses (uses only
        data with resolution_ts < T — lookahead-free).
     b. Snapshot bankroll for those addresses AS-OF T (one bankroll value
        per leader per refresh, used for leader-proportional sizing).
     c. Query OOS signals: fills by qualified addresses in [T, T+1 month)
        meeting market_criteria, resolution_bucket, maker_only filters.
  3. Simulate copying in time order across the entire backtest:
     - Maintain a closure-queue heap to track open copy positions.
     - Pop resolved positions before each new signal.
     - Apply per-leader concurrency caps + exposure caps.
     - Apply sizing rule + slippage + copy-direction.
     - Compute PnL from resolution_price.
     - Append audit row.
  4. Write audit log parquet + summary JSON.

The state (open_per_leader, closure_queue, cumulative PnL) lives for the
entire backtest, not per refresh — a copy position opened in month M may
resolve in month M+3 and continues to count toward the leader's exposure
cap across refresh boundaries.
"""
from __future__ import annotations

import heapq
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parents[2]
CLOSED_POS = str(ROOT / "data" / "closed_positions.parquet")
TRADERS = str(ROOT / "data" / "traders.parquet")
BANKROLL = str(ROOT / "data" / "bankroll_timeseries.parquet")
BACKTESTS_DIR = ROOT / "data" / "backtests"
TRADES_GLOB = str(ROOT / "data" / "trades" / "trades_delta_shard*.parquet")
SEED = str(ROOT / "data" / "trades" / "trades_seed.parquet")
MARKETS = sorted((ROOT / "data" / "markets").glob("markets_*.parquet"))[-1]

BUCKET_DAYS = {"2d": 2, "7d": 7, "30d": 30, "60d": 60}

# Concurrency / exposure caps per design 3.4
MAX_OPEN_PER_LEADER = 5
MAX_EXPOSURE_PER_LEADER_PCT = 0.20
MAX_POSITION_SIZE_PCT_FIXED = 0.02  # Rule A: 2% per signal
MAX_POSITION_SIZE_PCT_PROP = 0.05   # Rule B cap: 5% of capital max


def slippage_cents(
    leader_role: str,        # 'maker' or 'taker'
    market_neg_risk: bool,
    base_slippage_cents: float,
    taker_penalty_cents: float,
    negrisk_penalty_cents: float,
) -> float:
    s = base_slippage_cents
    if leader_role == "taker":
        s += taker_penalty_cents
    if market_neg_risk:
        s += negrisk_penalty_cents
    return s


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")
    # Register the markets_tokens TABLE (needed for resolution prices)
    con.execute(f"""
        CREATE OR REPLACE TABLE markets_tokens AS
        SELECT CAST(m.id AS VARCHAR) AS market_id,
               m.neg_risk, m.closed,
               TRY_CAST(m.end_date AS TIMESTAMP) AS end_date,
               m.outcome_prices, m.clob_token_ids,
               r.i AS outcome_index,
               m.clob_token_ids[r.i] AS outcome_token_id
        FROM read_parquet('{MARKETS}') m,
             range(1, len(m.clob_token_ids) + 1) AS r(i)
        WHERE len(m.clob_token_ids) > 0
    """)
    return con


def _query_oos_signals(
    con: duckdb.DuckDBPyConnection,
    addresses: list[str],
    T_start: datetime,
    T_end: datetime,
    bucket_days: int,
    maker_only: bool,
) -> pd.DataFrame:
    """Return all qualifying signals in [T_start, T_end) from `addresses`."""
    # Stage addresses in a temp table for clean JOIN semantics
    con.execute("CREATE OR REPLACE TEMP TABLE qual_addrs (address VARCHAR)")
    if addresses:
        con.executemany("INSERT INTO qual_addrs VALUES (?)", [(a,) for a in addresses])
    else:
        return pd.DataFrame()

    # Build joined_fills inline (filtered to OOS window + addresses); skip
    # the orphan-handling complexity since we never copy unmarketed fills.
    # The CASE on (jf.maker IN ... vs jf.taker IN ...) determines leader role.
    maker_filter = "AND rt.maker IN (SELECT address FROM qual_addrs)" if maker_only else ""
    cond = "rt.maker IN (SELECT address FROM qual_addrs) OR rt.taker IN (SELECT address FROM qual_addrs)"

    sql = f"""
    WITH joined_window AS (
        SELECT rt.timestamp, rt.market_id, rt.condition_id,
               rt.maker, rt.taker, rt.maker_asset_id,
               rt.token_amount, rt.usd_amount, rt.price, rt.transaction_hash,
               CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id
                    ELSE rt.maker_asset_id END AS outcome_token_id
        FROM (
            SELECT * FROM read_parquet('{TRADES_GLOB}')
              WHERE timestamp >= TIMESTAMP '{T_start.strftime("%Y-%m-%d %H:%M:%S")}'
                AND timestamp <  TIMESTAMP '{T_end.strftime("%Y-%m-%d %H:%M:%S")}'
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{SEED}')
              WHERE timestamp >= TIMESTAMP '{T_start.strftime("%Y-%m-%d %H:%M:%S")}'
                AND timestamp <  TIMESTAMP '{T_end.strftime("%Y-%m-%d %H:%M:%S")}'
        ) rt
        WHERE rt.maker IS NOT NULL AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker AND rt.market_id IS NOT NULL
          AND ({cond})
          {maker_filter}
    )
    SELECT
        jw.timestamp,
        jw.market_id, jw.condition_id, mt.outcome_index, mt.neg_risk,
        jw.outcome_token_id,
        CASE WHEN jw.maker IN (SELECT address FROM qual_addrs) THEN jw.maker
             ELSE jw.taker END AS leader_address,
        CASE WHEN jw.maker IN (SELECT address FROM qual_addrs) THEN 'maker'
             ELSE 'taker' END AS leader_role,
        jw.maker_asset_id,
        jw.token_amount AS leader_token_amount,
        jw.usd_amount   AS leader_trade_usd,
        jw.price        AS leader_price,
        jw.transaction_hash,
        mt.end_date     AS resolution_date,
        CAST(mt.outcome_prices[mt.outcome_index] AS DOUBLE) AS resolution_price,
        date_diff('day', jw.timestamp, mt.end_date) AS days_to_resolution
    FROM joined_window jw
    JOIN markets_tokens mt
      ON mt.market_id = jw.market_id
     AND mt.outcome_token_id = jw.outcome_token_id
    WHERE mt.closed = TRUE
      AND mt.end_date IS NOT NULL
      AND mt.end_date >  jw.timestamp
      AND date_diff('day', jw.timestamp, mt.end_date) < {bucket_days}
    ORDER BY jw.timestamp
    """
    df = con.sql(sql).fetchdf()
    return df


def _bankroll_snapshot(
    con: duckdb.DuckDBPyConnection,
    addresses: list[str],
    T: datetime,
) -> dict[str, float]:
    """Latest bankroll_30d_prior < T per address, used for sizing."""
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
class BacktestState:
    open_per_leader: dict[str, int]
    exposure_per_leader: dict[str, float]
    closure_queue: list  # heap of (resolution_ts, leader, market_id, size, pnl_already_recorded)


def _close_resolved(state: BacktestState, up_to_ts) -> None:
    while state.closure_queue and state.closure_queue[0][0] <= up_to_ts:
        _, leader, _, size, _ = heapq.heappop(state.closure_queue)
        state.open_per_leader[leader] = max(0, state.open_per_leader[leader] - 1)
        state.exposure_per_leader[leader] = max(0.0, state.exposure_per_leader[leader] - size)


def run_backtest(
    cohort_name: str,
    cohort_fn,                                # callable(con, T) -> list[str]
    resolution_bucket: str,                   # '2d' | '7d' | '30d' | '60d'
    sizing_rule: str,                         # 'fixed_pct' | 'leader_proportional'
    test_window_start: date,
    test_window_end: date,
    base_slippage_cents: float = 2.0,
    taker_penalty_cents: float = 4.0,
    negrisk_penalty_cents: float = 2.0,
    maker_only: bool = True,
    category_denylist: list | None = None,    # unused in v1 — no category data
    strategy_capital_usd: float = 100_000.0,
    window_label: str | None = None,
) -> Path:
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

    if window_label is None:
        window_label = f"{test_window_start:%Y%m}_{test_window_end:%Y%m}"
    run_id = (f"{cohort_name}__{resolution_bucket}__{sizing_rule}__"
              f"{window_label}__b{int(base_slippage_cents)}t{int(taker_penalty_cents)}n{int(negrisk_penalty_cents)}")

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

    state = BacktestState(
        open_per_leader=defaultdict(int),
        exposure_per_leader=defaultdict(float),
        closure_queue=[],
    )

    audit_rows: list[dict] = []
    skip_counts: dict[str, int] = defaultdict(int)
    refresh_log: list[dict] = []

    for T in refresh_dates:
        t0 = time.time()
        addresses = cohort_fn(con, T)
        bankroll_at_T = (_bankroll_snapshot(con, addresses, T)
                         if sizing_rule == "leader_proportional" else {})
        T_oos_end = T + relativedelta(months=1)
        signals = _query_oos_signals(con, addresses, T, T_oos_end, bucket_days, maker_only)
        refresh_log.append({
            "refresh_date": T.date().isoformat(),
            "n_qualified_addrs": len(addresses),
            "n_signals_oos": int(len(signals)),
            "elapsed_s": round(time.time() - t0, 2),
        })

        if signals.empty:
            continue

        for trade in signals.itertuples(index=False):
            ts = trade.timestamp
            # Close any positions resolved before this trade
            _close_resolved(state, ts)

            leader = trade.leader_address
            leader_role = trade.leader_role

            # Caps
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

            # Slippage + copy direction
            neg_risk_val = bool(trade.neg_risk) if trade.neg_risk is not None and not pd.isna(trade.neg_risk) else False
            slip = slippage_cents(
                leader_role, neg_risk_val,
                base_slippage_cents, taker_penalty_cents, negrisk_penalty_cents,
            )
            slip_dollars = slip / 100.0
            # Did the leader receive outcome tokens (i.e. BUY outcome)?
            #   If leader=maker: maker received tokens iff maker_asset_id='0'.
            #   If leader=taker: taker received tokens iff maker_asset_id!='0'.
            maker_asset_is_usdc = (trade.maker_asset_id == "0")
            leader_bought = maker_asset_is_usdc if leader_role == "maker" else not maker_asset_is_usdc

            if leader_bought:
                copy_price = float(trade.leader_price) + slip_dollars
            else:
                copy_price = float(trade.leader_price) - slip_dollars
            if copy_price <= 0 or copy_price >= 1:
                skip_counts["bad_copy_price"] += 1; continue

            copy_token_amount = size / copy_price
            res_price = float(trade.resolution_price)
            if leader_bought:
                copy_pnl = (res_price - copy_price) * copy_token_amount
            else:
                copy_pnl = (copy_price - res_price) * copy_token_amount

            # Record
            audit_rows.append({
                "run_id": run_id,
                "refresh_date": T.date(),
                "leader_address": leader,
                "market_id": trade.market_id,
                "condition_id": trade.condition_id,
                "outcome_index": int(trade.outcome_index),
                "neg_risk": neg_risk_val,
                "category": None,
                "trade_timestamp": ts,
                "resolution_date": trade.resolution_date,
                "days_to_resolution": int(trade.days_to_resolution),
                "resolution_bucket": resolution_bucket,
                "leader_maker_side": leader_role,
                "leader_trade_usd": float(trade.leader_trade_usd),
                "leader_price": float(trade.leader_price),
                "copy_price": copy_price,
                "copy_size_usd": size,
                "copy_token_amount": copy_token_amount,
                "position_resolution": res_price,
                "copy_pnl_usd": copy_pnl,
                "sizing_rule": sizing_rule,
                "slippage_cents": slip,
            })
            heapq.heappush(state.closure_queue,
                           (trade.resolution_date, leader, trade.market_id, size, copy_pnl))
            state.open_per_leader[leader] += 1
            state.exposure_per_leader[leader] += size

    # Write audit log
    audit_path = BACKTESTS_DIR / f"{run_id}.parquet"
    if audit_rows:
        df = pd.DataFrame(audit_rows)
        df.to_parquet(audit_path, compression="zstd", index=False)
    else:
        # Empty schema-stable file
        pd.DataFrame(columns=[
            "run_id","refresh_date","leader_address","market_id","condition_id",
            "outcome_index","neg_risk","category","trade_timestamp","resolution_date",
            "days_to_resolution","resolution_bucket","leader_maker_side","leader_trade_usd",
            "leader_price","copy_price","copy_size_usd","copy_token_amount",
            "position_resolution","copy_pnl_usd","sizing_rule","slippage_cents",
        ]).to_parquet(audit_path, compression="zstd", index=False)

    summary = _compute_summary(
        audit_rows, skip_counts, refresh_log,
        run_id, cohort_name, resolution_bucket, sizing_rule,
        test_window_start, test_window_end, strategy_capital_usd,
        base_slippage_cents, taker_penalty_cents, negrisk_penalty_cents,
        maker_only,
    )
    summary_path = BACKTESTS_DIR / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    elapsed = time.time() - t_run
    print(f"  → {len(audit_rows):,} signals, "
          f"PnL=${summary['total_pnl_usd']:,.0f}, "
          f"Sharpe={summary['sharpe_monthly']:.2f}, "
          f"deploy_ratio={summary['deployment_ratio']:.1%} "
          f"({elapsed:.1f}s)", flush=True)
    return audit_path


def _compute_summary(
    audit_rows, skip_counts, refresh_log,
    run_id, cohort_name, resolution_bucket, sizing_rule,
    test_window_start, test_window_end, strategy_capital_usd,
    base_slippage_cents, taker_penalty_cents, negrisk_penalty_cents,
    maker_only,
):
    summary = {
        "run_id": run_id,
        "cohort": cohort_name,
        "resolution_bucket": resolution_bucket,
        "sizing_rule": sizing_rule,
        "test_window_start": str(test_window_start),
        "test_window_end": str(test_window_end),
        "strategy_capital_usd": strategy_capital_usd,
        "slippage": {
            "base_cents": base_slippage_cents,
            "taker_penalty_cents": taker_penalty_cents,
            "negrisk_penalty_cents": negrisk_penalty_cents,
        },
        "maker_only": maker_only,
        "stage1_approximations": [
            "style_role_balance uses lifetime traders.parquet value (slow-changing trait)",
            "leader bankroll uses most-recent bankroll_30d_prior < T (per-refresh snapshot, not per-day)",
            "no category data — category_denylist not enforced",
        ],
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
    mean_m = float(monthly.mean())
    std_m = float(monthly.std(ddof=0))
    downside = monthly[monthly < 0]
    down_std = float((downside ** 2).mean() ** 0.5) if len(downside) else 0.0
    sharpe_m = (mean_m / std_m * (12 ** 0.5)) if std_m > 0 else 0.0
    sortino_m = (mean_m / down_std * (12 ** 0.5)) if down_std > 0 else 0.0

    cum_pnl = monthly.cumsum()
    running_peak = cum_pnl.cummax()
    drawdown_usd = float((running_peak - cum_pnl).max())

    # Deployment ratio: union of [trade_ts, resolution_date) intervals, capped
    # to the test window, as fraction of test-window days.
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
    signals_per_week = len(df) / n_weeks if n_weeks > 0 else 0.0

    by_leader = (df.groupby("leader_address")["copy_pnl_usd"]
                 .sum().sort_values(ascending=False))
    top_10 = [{"address": a, "pnl": float(p)} for a, p in by_leader.head(10).items()]
    bot_10 = [{"address": a, "pnl": float(p)} for a, p in by_leader.tail(10).items()]

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
        "signals_per_week": signals_per_week,
        "top_10_leaders_pnl": top_10,
        "bottom_10_leaders_pnl": bot_10,
    })
    return summary
