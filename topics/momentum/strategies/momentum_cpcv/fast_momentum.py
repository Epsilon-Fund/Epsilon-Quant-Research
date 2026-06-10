"""
Fast (numpy state-machine) port of the momentum strategy + realized-sizing
equity, for the synthetic-null Monte Carlo where ~half a million backtests are
needed. The notebook strategy's per-bar `df.iloc` loop costs ~0.4s per call;
this port produces IDENTICAL output in ~10ms.

Correctness contract: `verify_fast_pipeline()` must pass (Sharpe identical to
the notebook strategy + engine.backtest pipeline across random param draws)
before any MC run uses this module — the runner asserts it.

Indicator math is the same vectorised pandas as the notebook; only the
sequential position/stop state machine and the realized-equity loop are ported
to numpy scalar indexing. The OBV direction `.apply(lambda)` is replaced with
`np.where` (identical output incl. the NaN-first-bar -> -1 case; covered by
verification).
"""

import math

import numpy as np
import pandas as pd


def fast_positions(df: pd.DataFrame, params: dict, use_volume: bool):
    """Indicators + entry/stop state machine. Returns (position, position_size) arrays."""
    close = df["Close"]
    high, low, vol = df["High"], df["Low"], df["Volume"]

    ema = close.ewm(span=params["ema_span"], adjust=False).mean()
    swing_hi_cau = high.rolling(params["swing_caution"]).max()
    swing_lo_cau = low.rolling(params["swing_caution"]).min()
    swing_hi_stp = high.rolling(params["swing_stop"]).max()

    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    def atr(period):
        return tr.ewm(span=period, adjust=False).mean()

    atr_cau = atr(params["atr_caution"])
    atr_stp = atr(params["atr_stop"])
    atr_sz = atr(params["atr_size"])

    up = high.diff()
    down = -low.diff()
    pdm = up.where((up > down) & (up > 0), 0.0)
    ndm = down.where((down > up) & (down > 0), 0.0)
    atr14 = atr(14)
    pdi = 100 * pdm.ewm(span=14, adjust=False).mean() / atr14
    ndi = 100 * ndm.ewm(span=14, adjust=False).mean() / atr14
    dx = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(span=14, adjust=False).mean()

    direction = np.where(close.diff().values > 0, 1, -1)  # NaN -> -1, same as the apply()
    obv = pd.Series((vol.values * direction), index=df.index).cumsum()
    obv_ma = obv.rolling(params["obv_ma_period"]).mean()

    caution_obv = (close > close.shift(params["obv_lookback"])) & (obv < obv_ma)
    caution_long = ((swing_hi_cau - low) > 1.5 * atr_cau) | caution_obv
    caution_short = ((high - swing_lo_cau) > 1.5 * atr_cau) | (close > ema)
    valid = swing_hi_stp.notna() & atr_stp.notna() & atr_sz.notna() & obv_ma.notna()
    if use_volume:
        vol_ma = vol.rolling(params["vol_ma_period"]).mean()
        valid &= vol_ma.notna()
    entry_long = (close > ema) & (~caution_long | (adx > params["adx_override"])) & valid
    if use_volume:
        entry_long &= vol > vol_ma
    pos_size_raw = (params["risk_per_trade"] / (atr_sz / close)).clip(0.1, params["max_leverage"])

    # ── state machine on raw arrays (the notebook loop, minus .iloc) ──────────
    c = close.values
    shs = swing_hi_stp.values
    ast = atr_stp.values
    cl_arr = caution_long.values
    cs_arr = caution_short.values
    el_arr = entry_long.values
    psr = pos_size_raw.values
    scale = params["stop_atr_scale"]
    sm_pos_cau, sm_pos_norm = params["stop_mult_pos_caution"], params["stop_mult_pos_normal"]
    sm_ent_both = params["stop_mult_ent_both"]
    sm_ent_cau, sm_ent_norm = params["stop_mult_ent_caution"], params["stop_mult_ent_normal"]

    n = len(c)
    position = np.zeros(n, dtype=np.int64)
    position_size = np.zeros(n)
    in_position = 0
    stop_loss = np.nan
    current_size = 0.0

    for i in range(1, n):
        if in_position == 1:
            if c[i - 1] < stop_loss:
                in_position = 0
                current_size = 0.0
                stop_loss = np.nan
            else:
                sm = sm_pos_cau if cl_arr[i] else sm_pos_norm
                cand = shs[i] - ast[i] * sm * scale
                if cand > stop_loss:  # exact Python max(stop_loss, cand) incl. NaN behaviour
                    stop_loss = cand
        if in_position == 0:
            if el_arr[i]:
                in_position = 1
                current_size = psr[i]
                if cl_arr[i] and cs_arr[i]:
                    sm = sm_ent_both
                elif cl_arr[i]:
                    sm = sm_ent_cau
                else:
                    sm = sm_ent_norm
                stop_loss = shs[i] - ast[i] * sm * scale
        position[i] = in_position
        position_size[i] = current_size

    return position, position_size


def fast_realized_equity(eff_pos, eff_size, raw, cost):
    """Numpy port of performance_metrics.build_realized_equity_curve (same math)."""
    n = len(eff_pos)
    realized = 1.0
    entry_notional = 0.0
    cum_mult = 1.0
    prev_pos = 0.0
    equity = np.empty(n)
    for i in range(n):
        curr_pos = eff_pos[i]
        if prev_pos != 0.0 and (curr_pos == 0.0 or curr_pos != prev_pos):
            realized += entry_notional * (cum_mult - 1.0)
            entry_notional = 0.0
            cum_mult = 1.0
        pos_chg = abs(curr_pos - prev_pos)
        if pos_chg > 0.0:
            portfolio = realized + entry_notional * (cum_mult - 1.0)
            realized -= pos_chg * cost * portfolio
        if curr_pos != 0.0 and (prev_pos == 0.0 or curr_pos != prev_pos):
            entry_notional = eff_size[i] * realized
            cum_mult = 1.0
        if curr_pos != 0.0:
            cum_mult *= 1.0 + np.sign(curr_pos) * raw[i]
        equity[i] = realized + entry_notional * (cum_mult - 1.0)
        prev_pos = curr_pos
    return equity


def fast_net_returns(df: pd.DataFrame, params: dict, use_volume: bool, cost: float) -> np.ndarray:
    """Per-bar net returns of one config — engine.backtest single-asset semantics."""
    position, position_size = fast_positions(df, params, use_volume)
    eff_pos = np.concatenate(([0.0], position[:-1])).astype(float)   # shift(1).fillna(0)
    eff_size = np.concatenate(([0.0], position_size[:-1]))
    c = df["Close"].values
    raw = np.concatenate(([0.0], c[1:] / c[:-1] - 1.0))               # pct_change, NaN->0
    equity = fast_realized_equity(eff_pos, eff_size, raw, cost)
    net = np.empty_like(equity)
    net[0] = 0.0
    net[1:] = equity[1:] / equity[:-1] - 1.0                          # pct_change().fillna(0)
    return net


def fast_sharpe(df: pd.DataFrame, params: dict, use_volume: bool, cost: float,
                periods_per_year: float = 365.0) -> float:
    net = fast_net_returns(df, params, use_volume, cost)
    sd = net.std(ddof=1)
    return float(net.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0


def verify_fast_pipeline(df, slow_strategy_fn, use_volume, param_defs, fixed_params,
                         cost, n_draws=20, seed=0, tol=1e-9):
    """
    Assert the fast pipeline's Sharpe matches the notebook strategy +
    engine.backtest pipeline across random param draws. Raises on mismatch.
    Requires infrastructure/backtester on sys.path (flat import, repo style).
    """
    from engine import backtest  # noqa

    rng = np.random.default_rng(seed)
    worst = 0.0
    for d in range(n_draws):
        params = dict(fixed_params)
        for name, (ptype, lo, hi) in param_defs.items():
            if name in fixed_params:
                continue
            params[name] = (int(rng.integers(int(lo), int(hi) + 1)) if ptype == "int"
                            else float(rng.uniform(lo, hi)))
        sdf, _ = slow_strategy_fn(df.copy(), params)
        slow = backtest(sdf, cost=cost, show_plot=False)["sharpe_ratio"]
        fast = fast_sharpe(df, params, use_volume, cost)
        worst = max(worst, abs(slow - fast))
        if abs(slow - fast) > tol:
            raise AssertionError(
                f"fast pipeline mismatch on draw {d}: slow={slow!r} fast={fast!r} params={params}")
    return worst
