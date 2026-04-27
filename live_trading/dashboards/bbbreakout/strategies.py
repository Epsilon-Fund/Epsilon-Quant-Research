"""
strategies.py — BB Breakout strategy callable.

Imported by dashboard.py and optimise.py.
No numerical config lives here — only function logic.
"""

import numpy as np
import pandas as pd


# ── Shared indicator helpers ───────────────────────────────────────────────────

def _atr(d: pd.DataFrame, period: int) -> pd.Series:
    hi, lo, cl = d["high"], d["low"], d["close"]
    prev_cl = cl.shift(1)
    tr = pd.concat(
        [(hi - lo), (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _sma(d: pd.DataFrame, period: int) -> pd.Series:
    return d["close"].rolling(period).mean()


def _bb_width(d: pd.DataFrame, period: int) -> pd.Series:
    mid = d["close"].rolling(period).mean()
    std = d["close"].rolling(period).std()
    return (std * 2) / mid.replace(0, np.nan)


def _ma_slope(d: pd.DataFrame, period: int) -> pd.Series:
    ma = d["close"].rolling(period).mean()
    return ma - ma.shift(1)


def _candle_range(d: pd.DataFrame) -> pd.Series:
    return d["high"] - d["low"]


def _adx(d: pd.DataFrame, period: int):
    hi, lo, cl = d["high"], d["low"], d["close"]
    prev_hi = hi.shift(1)
    prev_lo = lo.shift(1)
    prev_cl = cl.shift(1)
    tr = pd.concat(
        [(hi - lo), (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1
    ).max(axis=1)
    plus_dm  = (hi - prev_hi).clip(lower=0).where((hi - prev_hi) > (prev_lo - lo), 0.0)
    minus_dm = (prev_lo - lo).clip(lower=0).where((prev_lo - lo) > (hi - prev_hi), 0.0)
    alpha    = 1.0 / period
    atr_w    = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


# ── Strategy ──────────────────────────────────────────────────────────────────

def bb_breakout(df_slice: pd.DataFrame, params: dict) -> tuple:
    """
    Bollinger Band breakout strategy.

    Two-timeframe: 1H OHLCV resampled to 4H for setup detection, 1H for entry.

    Required params
    ---------------
    atr_period, bb_period, bb_exp_window, h4_ma_period, slope_epsilon,
    breakout_lookback, breakout_pct, h1_ma_period, trend_ma_period,
    adx_period, adx_strong,
    max_1h_bars, pullback_atr_mult, entry_zone_bps, overshoot_bps,
    trail_atr_mult,
    risk_per_trade, max_leverage

    Returns
    -------
    (df, indicator_cols)
      df            : input DataFrame with SMA / position / position_size / stop_loss appended
      indicator_cols: ['SMA']
    """
    df = df_slice.copy()
    h1 = df.rename(columns=str.lower)

    # 4H resample
    h4 = h1.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    # 4H indicators
    h4_range = _candle_range(h4)
    h4_bw    = _bb_width(h4, params["bb_period"])
    h4_slope = _ma_slope(h4, params["h4_ma_period"])
    h4_green = h4["close"] > h4["open"]
    h4_red   = h4["close"] < h4["open"]

    brk_threshold = h4_range.rolling(int(params["breakout_lookback"])).quantile(params["breakout_pct"])
    big           = h4_range > brk_threshold
    two_big_green = big & big.shift(1) & h4_green & h4_green.shift(1)
    two_big_red   = big & big.shift(1) & h4_red   & h4_red.shift(1)
    bb_exp        = h4_bw > h4_bw.rolling(params["bb_exp_window"]).mean()
    h4_slope_norm = h4_slope / h4["close"].replace(0, np.nan)
    slope_eps     = params["slope_epsilon"]
    h4_long  = two_big_green & bb_exp & (h4_slope_norm >= -slope_eps)
    h4_short = two_big_red   & bb_exp & (h4_slope_norm <=  slope_eps)

    h4_adx, h4_plus_di, h4_minus_di = _adx(h4, params["adx_period"])

    # 1H indicators
    h1_atr      = _atr(h1, params["atr_period"])
    h1_range    = _candle_range(h1)
    h1_sma      = _sma(h1, params["h1_ma_period"])
    h1_trend_ma = h1["close"].rolling(int(params["trend_ma_period"])).mean()
    h1_pos_size = (params["risk_per_trade"] / (h1_atr / h1["close"])).clip(0.1, params["max_leverage"])

    # Align 4H -> 1H (shift(1) = use previous closed 4H bar)
    h4_adx_1h      = h4_adx.shift(1).reindex(h1.index,      method="ffill").fillna(0.0)
    h4_plus_di_1h  = h4_plus_di.shift(1).reindex(h1.index,  method="ffill").fillna(0.0)
    h4_minus_di_1h = h4_minus_di.shift(1).reindex(h1.index, method="ffill").fillna(0.0)
    h4_long_1h  = (h4_long.shift(1).reindex(h1.index,  method="ffill") == True)  # noqa: E712
    h4_short_1h = (h4_short.shift(1).reindex(h1.index, method="ffill") == True)  # noqa: E712

    # Per-condition 4H series aligned to 1H — exposed on the result df so the
    # dashboard can display each setup condition individually. No effect on the
    # backtest, which only reads position / position_size / stop_loss.
    def _bool_align(s: pd.Series) -> pd.Series:
        # Reindex a 4H bool series onto the 1H index after shift(1), with
        # NaN->False, then cast back to bool.  Using .where(...) avoids
        # pandas' object-dtype downcasting FutureWarning from .fillna().
        aligned = s.shift(1).reindex(h1.index, method="ffill")
        return aligned.where(aligned.notna(), False).astype(bool)

    h4_two_big_green_1h = _bool_align(two_big_green)
    h4_two_big_red_1h   = _bool_align(two_big_red)
    h4_bb_exp_1h        = _bool_align(bb_exp)
    h4_slope_norm_1h    = h4_slope_norm.shift(1).reindex(h1.index, method="ffill").fillna(0.0)

    long_setup_fires  = h4_long_1h  & ~(h4_long_1h.shift(1)  == True)  # noqa: E712
    short_setup_fires = h4_short_1h & ~(h4_short_1h.shift(1) == True)  # noqa: E712

    # Extract numpy arrays
    close_arr    = h1["close"].to_numpy()
    sma_arr      = h1_sma.to_numpy()
    range_arr    = h1_range.to_numpy()
    atr_arr      = h1_atr.to_numpy()
    pos_size_arr = h1_pos_size.to_numpy()
    trend_ma_arr = h1_trend_ma.to_numpy()
    adx_arr      = h4_adx_1h.to_numpy()
    plus_di_arr  = h4_plus_di_1h.to_numpy()
    minus_di_arr = h4_minus_di_1h.to_numpy()
    long_fire    = long_setup_fires.to_numpy()
    short_fire   = short_setup_fires.to_numpy()

    max_1h_bars       = params["max_1h_bars"]
    pullback_atr_mult = params["pullback_atr_mult"]
    entry_zone_bps    = params["entry_zone_bps"]
    overshoot_bps     = params["overshoot_bps"]
    trail_mult        = params["trail_atr_mult"]

    # State machine
    n             = len(h1)
    position      = np.zeros(n, dtype=int)
    position_size = np.ones(n)
    stop_loss     = np.zeros(n)

    # Per-bar snapshots of the state machine — captured every iteration via
    # try/finally so that `continue` branches still record the post-bar state.
    # Used by the dashboard; ignored by the backtest.
    setup_active_arr = np.zeros(n, dtype=bool)
    setup_dir_arr    = np.zeros(n, dtype=np.int8)
    bars_since_arr   = np.zeros(n, dtype=np.int32)

    setup_active    = False; setup_direction = 0; bars_since = 0
    in_trade        = False; trade_direction = 0; trade_stop = 0.0
    trade_tp        = 0.0;   trade_size      = 1.0

    for i in range(1, n):
        try:
            # 1. Trade management
            if in_trade:
                close = close_arr[i]
                h1_at = atr_arr[i]
                if not np.isnan(h1_at):
                    if trade_direction == 1:
                        trade_stop = max(trade_stop, close - trail_mult * h1_at)
                    else:
                        trade_stop = min(trade_stop, close + trail_mult * h1_at)

                stop_hit = (
                    (trade_direction ==  1 and trade_stop > 0 and close <= trade_stop) or
                    (trade_direction == -1 and trade_stop > 0 and close >= trade_stop)
                )
                tp_hit = (
                    (trade_direction ==  1 and trade_tp > 0 and close >= trade_tp) or
                    (trade_direction == -1 and trade_tp > 0 and close <= trade_tp)
                )

                if stop_hit or tp_hit:
                    in_trade = False
                else:
                    position[i]      = trade_direction
                    position_size[i] = trade_size
                    # Expose the current ratcheted trail stop on every HOLD
                    # bar — backtest ignores this, but the dashboard reads
                    # it via compute_signals to display the live stop.
                    stop_loss[i]     = trade_stop
                continue

            # 2. Setup detection
            if not setup_active:
                if long_fire[i]:
                    setup_active    = True
                    setup_direction = 1
                    bars_since      = 0
                elif short_fire[i]:
                    trend_ma_i = trend_ma_arr[i]
                    adx_i      = adx_arr[i]
                    plus_di_i  = plus_di_arr[i]
                    minus_di_i = minus_di_arr[i]
                    above_ma   = not np.isnan(trend_ma_i) and close_arr[i] > trend_ma_i
                    bull_trend = adx_i > params["adx_strong"] and plus_di_i > minus_di_i
                    if not above_ma and not bull_trend:
                        setup_active    = True
                        setup_direction = -1
                        bars_since      = 0

            if not setup_active:
                continue

            # 3. Expiry checks
            bars_since += 1
            close  = close_arr[i]
            s_ma   = sma_arr[i]
            h1_rng = range_arr[i]
            h1_at  = atr_arr[i]

            if np.isnan(s_ma) or np.isnan(h1_at) or s_ma == 0:
                continue
            if bars_since > max_1h_bars:
                setup_active = False
                continue
            if h1_rng > pullback_atr_mult * h1_at:
                setup_active = False
                continue
            if setup_direction == 1:
                if close < s_ma - (s_ma * overshoot_bps / 10000):
                    setup_active = False
                    continue
            else:
                if close > s_ma + (s_ma * overshoot_bps / 10000):
                    setup_active = False
                    continue

            # 4. Entry
            bps_from_sma = abs(close - s_ma) / s_ma * 10000
            in_zone      = bps_from_sma <= entry_zone_bps
            momentum_ok  = (
                (setup_direction ==  1 and close > close_arr[i - 1]) or
                (setup_direction == -1 and close < close_arr[i - 1])
            )

            if in_zone and momentum_ok:
                sz    = pos_size_arr[i] if not np.isnan(pos_size_arr[i]) else 1.0
                h1_at = atr_arr[i]

                if not np.isnan(h1_at):
                    stop_dist = trail_mult * h1_at
                    sl_val    = (close - stop_dist) if setup_direction == 1 else (close + stop_dist)
                    ts_val    = sl_val
                    in_strong_bull = (
                        not np.isnan(trend_ma_arr[i])
                        and close > trend_ma_arr[i]
                        and adx_arr[i] > params["adx_strong"]
                        and plus_di_arr[i] > minus_di_arr[i]
                    )
                    if in_strong_bull or np.isnan(trend_ma_arr[i]):
                        tp_val = 0.0
                    else:
                        tp_val = (close + 6 * stop_dist) if setup_direction == 1 else (close - 6 * stop_dist)
                else:
                    sl_val = 0.0; ts_val = 0.0; tp_val = 0.0

                position[i]      = setup_direction
                position_size[i] = sz
                stop_loss[i]     = sl_val
                in_trade         = True
                trade_direction  = setup_direction
                trade_stop       = ts_val
                trade_tp         = tp_val
                trade_size       = sz
                setup_active     = False

        finally:
            setup_active_arr[i] = setup_active
            setup_dir_arr[i]    = setup_direction
            bars_since_arr[i]   = bars_since

    df["SMA"]           = h1_sma.to_numpy()
    df["position"]      = position
    df["position_size"] = position_size
    df["stop_loss"]     = stop_loss
    # Inspection columns for the dashboard — backtest doesn't read these.
    df["h4_two_big_green"]   = h4_two_big_green_1h.astype(bool).to_numpy()
    df["h4_two_big_red"]     = h4_two_big_red_1h.astype(bool).to_numpy()
    df["h4_bb_exp"]          = h4_bb_exp_1h.astype(bool).to_numpy()
    df["h4_slope_norm"]      = h4_slope_norm_1h.to_numpy()
    df["h4_long_setup"]      = h4_long_1h.astype(bool).to_numpy()
    df["h4_short_setup_raw"] = h4_short_1h.astype(bool).to_numpy()
    df["h4_adx"]             = h4_adx_1h.to_numpy()
    df["h4_plus_di"]         = h4_plus_di_1h.to_numpy()
    df["h4_minus_di"]        = h4_minus_di_1h.to_numpy()
    df["h1_atr"]             = h1_atr.to_numpy()
    df["h1_range"]           = h1_range.to_numpy()
    df["h1_trend_ma"]        = h1_trend_ma.to_numpy()
    df["setup_active"]       = setup_active_arr
    df["setup_direction"]    = setup_dir_arr
    df["bars_since"]         = bars_since_arr
    return df, ["SMA"]


# ── Registry ──────────────────────────────────────────────────────────────────
STRATEGY_REGISTRY = {
    "bb_breakout": bb_breakout,
}


# ── Asset config ──────────────────────────────────────────────────────────────
# lookback = 2151 days of hourly data
# param_defs: {name: (dtype, lo, hi)}  — optimised by optimise.py
# fixed_params: {name: value}          — held constant across all folds

ASSET_CONFIG = [
    # ── BTC ───────────────────────────────────────────────────────────────────
    {
        "symbol":   "BTCUSDT",
        "strategy": "bb_breakout",
        "lookback": 2151,
        "param_defs": {
            "bb_period":         ("int",   30,   40),
            "bb_exp_window":     ("int",    2,   20),
            "atr_period":        ("int",    5,   20),
            "breakout_pct":      ("float",  0.50, 0.85),
            "breakout_lookback": ("int",   20,  100),
            "h4_ma_period":      ("int",   10,   50),
            "slope_epsilon":     ("float",  0.0,  0.003),
            "h1_ma_period":      ("int",    5,   17),
            "entry_zone_bps":    ("int",   20,  100),
            "overshoot_bps":     ("int",    5,  150),
            "max_1h_bars":       ("int",   12,   48),
            "pullback_atr_mult": ("float",  1.0,  3.0),
            "trail_atr_mult":    ("float",  0.5,  4.0),
            "adx_period":        ("int",    7,   21),
            "adx_strong":        ("float", 20.0, 60.0),
            "trend_ma_period":   ("int",  150,  300),
        },
        "fixed_params": {
            "bb_period":         36,
            "bb_exp_window":     17,
            "breakout_lookback": 86,
            "trail_atr_mult":    3.1918,
            "trend_ma_period":   230,
            "risk_per_trade":    0.03,
            "max_leverage":      2.5,
        },
    },
    # ── ETH ───────────────────────────────────────────────────────────────────
    {
        "symbol":   "ETHUSDT",
        "strategy": "bb_breakout",
        "lookback": 2151,
        "param_defs": {
            "bb_period":         ("int",   10,   40),
            "bb_exp_window":     ("int",    8,   20),
            "atr_period":        ("int",    5,   20),
            "breakout_pct":      ("float",  0.70, 0.85),
            "breakout_lookback": ("int",   20,  100),
            "h4_ma_period":      ("int",   10,   50),
            "slope_epsilon":     ("float",  0.0,  0.003),
            "h1_ma_period":      ("int",    5,   17),
            "entry_zone_bps":    ("int",    5,  100),
            "overshoot_bps":     ("int",    5,  150),
            "max_1h_bars":       ("int",   12,   48),
            "pullback_atr_mult": ("float",  1.2,  3.0),
            "trail_atr_mult":    ("float",  0.5,  4.0),
            "adx_period":        ("int",    7,   21),
            "adx_strong":        ("float", 20.0, 60.0),
            "trend_ma_period":   ("int",  150,  300),
        },
        "fixed_params": {
            "bb_period":         38,
            "breakout_pct":      0.7953,
            "breakout_lookback": 50,
            "pullback_atr_mult": 1.9474,
            "trail_atr_mult":    3.3608,
            "risk_per_trade":    0.03,
            "max_leverage":      2.5,
        },
    },
    # ── AVAX ──────────────────────────────────────────────────────────────────
    {
        "symbol":   "AVAXUSDT",
        "strategy": "bb_breakout",
        "lookback": 2151,
        "param_defs": {
            "bb_period":         ("int",   20,   40),
            "bb_exp_window":     ("int",    1,   20),
            "atr_period":        ("int",    5,   20),
            "breakout_lookback": ("int",   20,  100),
            "h4_ma_period":      ("int",   10,   50),
            "slope_epsilon":     ("float",  0.001, 0.003),
            "overshoot_bps":     ("int",   60,  150),
            "max_1h_bars":       ("int",   12,   48),
            "pullback_atr_mult": ("float",  1.5,  3.0),
            "trail_atr_mult":    ("float",  0.5,  4.0),
            "adx_period":        ("int",    7,   21),
            "adx_strong":        ("float", 20.0, 60.0),
            "trend_ma_period":   ("int",  150,  300),
        },
        "fixed_params": {
            "breakout_pct":      0.6029,
            "breakout_lookback": 48,
            "h1_ma_period":      16,
            "entry_zone_bps":    71,
            "risk_per_trade":    0.03,
            "max_leverage":      2.5,
        },
    },
    # ── ADA ───────────────────────────────────────────────────────────────────
    {
        "symbol":   "ADAUSDT",
        "strategy": "bb_breakout",
        "lookback": 2151,
        "param_defs": {
            "bb_period":         ("int",   10,   40),
            "bb_exp_window":     ("int",    2,   20),
            "atr_period":        ("int",    5,   20),
            "breakout_pct":      ("float",  0.50, 0.85),
            "breakout_lookback": ("int",   20,  100),
            "h4_ma_period":      ("int",   10,   50),
            "slope_epsilon":     ("float",  0.0,  0.003),
            "entry_zone_bps":    ("int",    5,  100),
            "overshoot_bps":     ("int",    5,  110),
            "max_1h_bars":       ("int",   12,   48),
            "pullback_atr_mult": ("float",  1.0,  3.0),
            "adx_period":        ("int",    7,   21),
            "adx_strong":        ("float", 20.0, 60.0),
        },
        "fixed_params": {
            "breakout_pct":      0.5199,
            "h1_ma_period":      10,
            "trail_atr_mult":    3.7041,
            "trend_ma_period":   244,
            "risk_per_trade":    0.03,
            "max_leverage":      2.5,
        },
    },
    # ── NEAR ──────────────────────────────────────────────────────────────────
    {
        "symbol":   "NEARUSDT",
        "strategy": "bb_breakout",
        "lookback": 2151,
        "param_defs": {
            "bb_period":         ("int",   10,   40),
            "bb_exp_window":     ("int",    2,   20),
            "atr_period":        ("int",    5,   20),
            "breakout_pct":      ("float",  0.50, 0.85),
            "breakout_lookback": ("int",   20,  100),
            "h4_ma_period":      ("int",   10,   50),
            "slope_epsilon":     ("float",  0.0,  0.003),
            "h1_ma_period":      ("int",    5,   17),
            "entry_zone_bps":    ("int",    5,  100),
            "overshoot_bps":     ("int",    5,  150),
            "max_1h_bars":       ("int",   12,   48),
            "pullback_atr_mult": ("float",  1.0,  3.0),
            "trail_atr_mult":    ("float",  0.5,  4.0),
            "adx_period":        ("int",    7,   21),
            "adx_strong":        ("float", 20.0, 60.0),
            "trend_ma_period":   ("int",  150,  300),
        },
        "fixed_params": {
            "overshoot_bps":     99,
            "pullback_atr_mult": 2.717,
            "trail_atr_mult":    2.7646,
            "breakout_pct":      0.7113,
            "atr_period":        5,
            "max_1h_bars":       23,
            "trend_ma_period":   220,
            "risk_per_trade":    0.03,
            "max_leverage":      2.5,
        },
    },
]
