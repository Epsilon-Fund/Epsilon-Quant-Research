"""
xs_strategy.py — Cross-sectional momentum strategy primitives.

Consolidates code that was previously duplicated across the xs_momentum
notebooks (xs_3_i_r, xs_3_2_i_r, xs_cpcv ...).  Provides:

    wilder_adx(high, low, close, period)
        Wilder's ADX, no TA-lib dependency.

    compute_btc_regime_tilt_panel(btc_ohlc, ...)
        Continuous-intensity (EMA × ADX) BTC regime tilt Series.

    summarize_tilt_distribution(tilt_panel, ...)
        Print a tilt-bucket histogram for sanity checks.

    make_xs_strategy(panel, volume, meta, ...)
        Factory returning a strategy_fn compatible with both
        wf_engine.walk_forward() and cpcv_engine.run_cpcv().

The factory supports three optional layers on top of the residual-Sharpe base:
    1. Dynamic universe selection via universe_filter.get_universe()
    2. Lee-Swaminathan two-stage volume-change refinement (pool_multiplier)
    3. Inverse-volatility intra-leg weighting + BTC-regime asymmetric leg sizing

The strategy_fn can read regime parameters either from a pre-computed
`regime_tilt_panel` (fixed regime) OR from the per-trial `params` dict
(when `btc_ohlc` is captured in the closure) — the second mode is what
the CPCV / walk-forward engines need to optimise the regime EMA/ADX.
"""

import numpy as np
import pandas as pd

# universe_filter lives next to the xs_momentum notebooks, not in infrastructure.
# Resolve it lazily so this module can be imported without that package on path.
def _resolve_universe_filter():
    try:
        from universe_filter import get_universe, precompute_avg_volume
        return get_universe, precompute_avg_volume
    except ImportError:
        import os, sys
        here = os.path.dirname(os.path.abspath(__file__))
        uf_dir = os.path.abspath(os.path.join(
            here, '..', '..', 'topics', 'momentum', 'xs_momentum', 'universe'
        ))
        if uf_dir not in sys.path:
            sys.path.insert(0, uf_dir)
        from universe_filter import get_universe, precompute_avg_volume
        return get_universe, precompute_avg_volume


# ── ADX / regime helpers ─────────────────────────────────────────────────────

def wilder_adx(high, low, close, period=14):
    """Wilder's ADX.  Inline implementation to avoid a TA-lib dependency."""
    high_diff = high.diff()
    low_diff  = -low.diff()
    plus_dm   = high_diff.where((high_diff > low_diff)  & (high_diff > 0), 0.0)
    minus_dm  = low_diff .where((low_diff  > high_diff) & (low_diff  > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing = EMA with alpha = 1/period
    atr      = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/period,  adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def compute_btc_regime_tilt_panel(
    btc_ohlc,
    ema_period      = 19,
    adx_period      = 14,
    adx_scale       = 40,
    max_bull_tilt   = 0.2,
    max_bear_tilt   = 0.2,
):
    """
    Continuous-intensity BTC regime tilt.

    Direction comes from EMA position (binary: bull above EMA, bear below).
    Intensity comes from ADX magnitude, scaled linearly to [0, 1] saturating at
    `adx_scale`.  Tilt = direction × intensity × asymmetric directional MAX.

    Parameters
    ----------
    btc_ohlc      : DataFrame with High / Low / Close columns
    ema_period    : EMA span for the bull/bear discriminator
    adx_period    : ADX lookback (Wilder smoothing)
    adx_scale     : ADX value at which tilt saturates at MAX_*_TILT.
                    Below: linear ramp.  Above: capped at MAX.
    max_bull_tilt : cap on the bull-side tilt magnitude
    max_bear_tilt : cap on the bear-side tilt magnitude

    Returns
    -------
    tilt_panel : pd.Series indexed by btc_ohlc.index, values in [-max_bear_tilt, +max_bull_tilt]
    """
    btc_close = btc_ohlc['Close']
    btc_ema   = btc_close.ewm(span=ema_period, adjust=False).mean()
    btc_adx   = wilder_adx(btc_ohlc['High'], btc_ohlc['Low'], btc_close, period=adx_period)

    above_ema = btc_close > btc_ema
    intensity = (btc_adx / adx_scale).clip(upper=1.0)   # ∈ [0, 1]

    tilt = pd.Series(0.0, index=btc_close.index)
    tilt[ above_ema] = +intensity[ above_ema] * max_bull_tilt
    tilt[~above_ema] = -intensity[~above_ema] * max_bear_tilt
    return tilt


def compute_btc_regime_breadth_tilt_panel(
    btc_close,
    universe_panel,
    ma_period       = None,
    ma_type         = 'ema',
    ema_period      = None,
    max_bull_tilt   = 0.2,
    max_bear_tilt   = 0.2,
):
    """
    Layer 2 regime tilt: MA direction + UNIVERSE BREADTH intensity.

    Replaces the ADX-based intensity from `compute_btc_regime_tilt_panel`
    with breadth — the fraction of universe coins on the same side of their
    MA(period) as BTC is on its MA(period).

    Direction: BTC > MA(period) → bull, else bear (binary, same as Layer 1).
    Intensity: fraction of valid universe coins matching BTC's regime
               at each bar (∈ [0, 1]).
    Tilt    = direction × breadth × asymmetric directional MAX.

    Parameters
    ----------
    btc_close       : pd.Series  — BTC closing price series (daily)
    universe_panel  : pd.DataFrame — date × coin Close panel for breadth calc
    ma_period       : period for the regime discriminator MA. Used for both
                      BTC and each universe coin.
    ma_type         : 'ema' (default — exponential, recency-weighted, faster) or
                      'sma' (simple — equal-weight, smoother, academically
                      conventional for regime classification).
    ema_period      : legacy alias for `ma_period`. If provided AND `ma_period`
                      is None, used as the MA period. Kept for backward
                      compatibility with existing `_r2` notebooks.
    max_bull_tilt,
    max_bear_tilt   : asymmetric magnitude caps.

    Returns
    -------
    pd.Series indexed by `universe_panel.index`, values in [-max_bear_tilt, +max_bull_tilt].
    """
    # Resolve period (ma_period is preferred; ema_period is the legacy alias)
    period = ma_period if ma_period is not None else (
        ema_period if ema_period is not None else 100
    )

    # BTC and per-coin MAs (EMA or SMA, switchable)
    if ma_type == 'sma':
        btc_ma   = btc_close.rolling(window=period, min_periods=period).mean()
        coin_mas = universe_panel.rolling(window=period, min_periods=period).mean()
    elif ma_type == 'ema':
        btc_ma   = btc_close.ewm(span=period, adjust=False).mean()
        coin_mas = universe_panel.ewm(span=period, adjust=False).mean()
    else:
        raise ValueError(f"ma_type must be 'ema' or 'sma', got {ma_type!r}")

    btc_above  = (btc_close > btc_ma).reindex(universe_panel.index).ffill().fillna(False)
    coin_above = universe_panel > coin_mas       # boolean DataFrame

    # Breadth = fraction matching BTC's regime, weighted only by coins with
    # valid prices at each bar (NaN coins excluded from numerator and denominator).
    matching   = coin_above.eq(btc_above, axis=0)
    valid      = universe_panel.notna()
    matching_v = matching & valid
    n_valid    = valid.sum(axis=1).replace(0, np.nan)
    breadth    = (matching_v.sum(axis=1) / n_valid).fillna(0.0)

    tilt = pd.Series(0.0, index=universe_panel.index)
    tilt[ btc_above] = +breadth[ btc_above] * max_bull_tilt
    tilt[~btc_above] = -breadth[~btc_above] * max_bear_tilt
    return tilt


def compute_dispersion_confidence_panel(
    panel,
    rolling_window      = 252,
    min_periods         = 60,
    q_low               = 0.20,
    q_high              = 0.80,
    confidence_low      = 0.5,
    confidence_high     = 1.0,
):
    """
    Layer 3 — signal-quality gate via cross-sectional return dispersion.

    Computes a per-bar `confidence` multiplier (∈ [confidence_low, confidence_high])
    that the strategy uses to scale gross exposure based on whether the universe
    is "differentiated enough" for the ranking signal to have informational content.

    Construction:
      1. Per-bar cross-sectional dispersion = std(daily_returns) across coins.
      2. Point-in-time rolling quantiles (q_low, q_high) over `rolling_window`
         bars — strictly no lookahead.  Adapts to regime shifts in the dispersion
         distribution itself.
      3. Linear interpolation:
            dispersion ≤ p_low  → confidence_low   (signal weak, scale down)
            dispersion ≥ p_high → confidence_high  (signal strong, scale up)
            in between          → linear ramp
         Continuous (no whipsaw at threshold crossings).

    For a hedge-use-case strategy, default `[0.5, 1.0]` — defensive only, no
    leverage above baseline.  Switch to `[0.5, 1.5]` if you want the gate to
    also lever up in high-dispersion regimes.

    Returns
    -------
    confidence : pd.Series  — per-bar multiplier, indexed by panel.index
    dispersion : pd.Series  — per-bar raw dispersion (for diagnostics)
    """
    daily_rets = panel.pct_change()
    dispersion = daily_rets.std(axis=1)        # per-bar cross-sectional std

    # Point-in-time rolling quantiles — no lookahead
    rolling = dispersion.rolling(rolling_window, min_periods=min_periods)
    p_low   = rolling.quantile(q_low)
    p_high  = rolling.quantile(q_high)

    # Continuous interpolation, clamped outside [p_low, p_high]
    width = (p_high - p_low).replace(0, np.nan)
    raw   = (dispersion - p_low) / width
    confidence = confidence_low + (confidence_high - confidence_low) * raw.clip(0, 1)
    # Default to full confidence in the warmup region (rolling window not yet full)
    confidence = confidence.fillna(confidence_high)

    return confidence, dispersion


def summarize_tilt_distribution(tilt_panel, max_bull_tilt=0.2, max_bear_tilt=0.2,
                                 btc_adx=None):
    """Print a tilt-bucket histogram for the regime panel."""
    def _bucket(t):
        if t == 0:
            return 'neutral'
        if t > 0:
            if t >= 0.66 * max_bull_tilt: return 'bull (strong)'
            if t >= 0.33 * max_bull_tilt: return 'bull (mid)'
            return 'bull (mild)'
        if t <= -0.66 * max_bear_tilt: return 'bear (strong)'
        if t <= -0.33 * max_bear_tilt: return 'bear (mid)'
        return 'bear (mild)'

    bucket = tilt_panel.apply(_bucket)
    counts = bucket.value_counts()
    print('Tilt-intensity distribution over fetched window:')
    for s in ['bull (strong)', 'bull (mid)', 'bull (mild)', 'neutral',
              'bear (mild)', 'bear (mid)', 'bear (strong)']:
        n = int(counts.get(s, 0))
        pct = n / len(bucket) * 100
        print(f'  {s:<15} bars={n:>5}  ({pct:>5.1f}%)')

    if btc_adx is not None:
        print(f'\nADX stats: median={btc_adx.median():.1f}  '
              f'p25={btc_adx.quantile(0.25):.1f}  p75={btc_adx.quantile(0.75):.1f}')
    print(f'Tilt magnitude: mean abs={tilt_panel.abs().mean():.4f}  '
          f'max={tilt_panel.abs().max():.4f}')


# ── strategy factory ─────────────────────────────────────────────────────────

def make_xs_strategy(
    panel,
    volume            = None,
    meta              = None,
    uf_top_n          = None,
    uf_min_volume     = None,
    uf_min_age        = None,
    uf_volume_window  = None,
    pool_multiplier   = None,
    iv_vol_window     = 30,
    regime_tilt_panel = None,
    btc_ohlc          = None,
    btc_close         = None,
    universe_breadth_panel = None,
    regime_mode       = 'adx',
    max_bull_tilt     = 0.2,
    max_bear_tilt     = 0.2,
    signal_kind       = 'residual_sharpe',
    volume_filter_legs = 'both',
    breadth_ma_type   = 'ema',
    confidence_panel  = None,
):
    """
    Factory that captures the perp price panel and returns a strategy_fn
    compatible with wf_engine.walk_forward() and cpcv_engine.run_cpcv().

    Single perp panel for both signal and PnL — perp prices feed the ranking
    signal and the backtest execution (matches what would actually trade on
    Hyperliquid for the dollar-neutral L/S strategy).

    Parameters
    ----------
    panel             : DataFrame, DatetimeIndex × coin, perp Close prices.
                        Must include a 'BTC' column (used as the residual factor).
    volume, meta      : panels from universe_filter.load_cache() — when both
                        provided, the strategy applies the dynamic universe
                        filter at every rebalance.  Pass None to skip filtering
                        (use the full panel each rebalance).
    uf_*              : universe-filter parameters; passed through to
                        get_universe() at each rebalance.  Defaults: top_n =
                        len(panel.columns), min_volume = 50M, min_age = 180,
                        volume_window = 30.
    pool_multiplier   : Lee-Swaminathan two-stage refinement.
                          None  → single-stage (rank by composite, pick top/bottom).
                          float → Stage 1 picks top/bottom (mult × n) by composite,
                                  Stage 2 refines BOTH legs by HIGHEST 7d/30d
                                  volume-change.  Requires `volume`.
    iv_vol_window     : rolling stdev window for inverse-vol intra-leg weighting.
    regime_tilt_panel : pre-computed continuous-intensity tilt Series (fast path
                        for fixed regime params).  If provided, regime params in
                        `params` are ignored.
    btc_ohlc          : BTC OHLC DataFrame (with High/Low/Close) — enables the
                        per-trial regime path: strategy_fn computes the tilt from
                        params['regime_ema_period'], ['regime_adx_period'],
                        ['regime_adx_scale'] each call.  Required when
                        `regime_tilt_panel` is None and you want a non-zero tilt.
    max_bull_tilt,
    max_bear_tilt     : tilt magnitude caps used in the per-trial regime path
                        (ignored when `regime_tilt_panel` is provided).
    signal_kind       : ranking-signal formulation. One of:
                          'residual_sharpe' (default) — xs_3 family. Strip BTC
                              factor via rolling β, then take J-bar Sharpe of
                              residuals.  Warmup = 2J + 1.
                          'rolling_sharpe' — xs_1 family. Sharpe of raw returns
                              over J-bar window (vol-normalised momentum,
                              no β stripping).  Warmup = J + 1.

    Required Optuna params (read each call):
        J          int   formation / lookback window for the residual-Sharpe signal
        K          int   holding / rebalance period
        pct_long   float fraction of the eligible universe in the long leg
        pct_short  float fraction of the eligible universe in the short leg

    Optional Optuna params (read only when btc_ohlc is captured):
        regime_ema_period  int  BTC EMA span    (bull/bear discriminator)
        regime_adx_period  int  ADX lookback    (Wilder smoothing)
        regime_adx_scale   int  ADX scale at which tilt saturates

    Output columns (per-bar):
        strategy_returns, turnover, long_ret, short_ret, universe_size,
        long_turnover, short_turnover, n_long_held, n_short_held,
        long_weight, short_weight, regime_label

    Performance
    -----------
    Two-level optimisation for fast Optuna runs:
      1. Rolling avg-volume panel pre-computed ONCE (shared across all trials).
      2. Inner bar loop fully vectorised via numpy.
      3. Regime tilt panels cached by (ema, adx_period, adx_scale) when computed
         per-trial, so identical Optuna trials re-use the same panel.

    Turnover convention:
      long_turn  = |new_long  − old_long|  / n_long    ∈ [0, 1]
      short_turn = |new_short − old_short| / n_short   ∈ [0, 1]
      turnover   = long_turn + short_turn               ∈ [0, 2]
    """
    get_universe, precompute_avg_volume = _resolve_universe_filter()

    # ── Resolve filter defaults once ──────────────────────────────────────────
    _top_n      = uf_top_n         if uf_top_n         is not None else len(panel.columns)
    _min_vol    = uf_min_volume    if uf_min_volume    is not None else 50_000_000
    _min_age    = uf_min_age       if uf_min_age       is not None else 180
    _vol_window = uf_volume_window if uf_volume_window is not None else 30

    # ── Pre-compute rolling avg volume ONCE (shared across all Optuna trials) ─
    avg_vol_panel = (precompute_avg_volume(volume, volume_window=_vol_window)
                     if volume is not None else None)

    # ── Pre-compute coin → column-index mapping ──────────────────────────────
    coin_to_idx = {c: i for i, c in enumerate(panel.columns)}

    # ── Inverse-vol panel cache (per-trial when 'iv_vol_window' in params) ──
    # Keyed by integer window; the factory-default window is pre-warmed below.
    _default_iv_window = int(iv_vol_window)
    _vol_cache: dict   = {}
    _daily_rets        = panel.pct_change()

    def _get_vol_panel(window):
        if window not in _vol_cache:
            _vol_cache[window] = _daily_rets.rolling(window, min_periods=10).std()
        return _vol_cache[window]

    _get_vol_panel(_default_iv_window)   # warm cache for the default

    # ── Confidence panel cache (per-trial when dispersion params in params) ─
    # Keyed by (window, conf_low, conf_high); q_low/q_high held at (0.20, 0.80).
    _conf_cache: dict = {}

    def _resolve_confidence_panel(params):
        # No dispersion-related params in this trial → fall back to the
        # factory-level pre-computed panel (which may itself be None).
        if not any(k in params for k in
                   ('dispersion_window', 'confidence_low', 'confidence_high')):
            return confidence_panel
        win = int(params.get('dispersion_window', 252))
        cl  = float(params.get('confidence_low',  0.5))
        ch  = float(params.get('confidence_high', 1.5))
        key = (win, cl, ch)
        if key not in _conf_cache:
            cp, _ = compute_dispersion_confidence_panel(
                panel,
                rolling_window  = win,
                min_periods     = 60,
                q_low           = 0.20,
                q_high          = 0.80,
                confidence_low  = cl,
                confidence_high = ch,
            )
            _conf_cache[key] = cp
        return _conf_cache[key]

    # ── Tilt panel cache (only used in per-trial regime mode) ────────────────
    _tilt_cache: dict = {}

    def _resolve_tilt_panel(params):
        """
        Resolution order: pre-computed panel > per-trial computation > zero.
        Per-trial mode dispatches on `regime_mode`:
          'adx'     — Layer 1, ADX-scaled intensity (needs btc_ohlc)
          'breadth' — Layer 2, universe-breadth intensity (needs btc_close
                      and universe_breadth_panel)
        """
        if regime_tilt_panel is not None:
            return regime_tilt_panel

        if regime_mode == 'adx':
            if btc_ohlc is None:
                return None
            ema_p = int(params.get('regime_ema_period', 19))
            adx_p = int(params.get('regime_adx_period', 14))
            adx_s = int(params.get('regime_adx_scale',  40))
            key = ('adx', ema_p, adx_p, adx_s)
            if key not in _tilt_cache:
                _tilt_cache[key] = compute_btc_regime_tilt_panel(
                    btc_ohlc,
                    ema_period    = ema_p,
                    adx_period    = adx_p,
                    adx_scale     = adx_s,
                    max_bull_tilt = max_bull_tilt,
                    max_bear_tilt = max_bear_tilt,
                )
            return _tilt_cache[key]

        elif regime_mode == 'breadth':
            if btc_close is None or universe_breadth_panel is None:
                return None
            # Read MA period from params, supporting both the new generic
            # `breadth_ma_period` and the legacy `breadth_ema_period` names.
            ma_p = int(params.get('breadth_ma_period',
                                  params.get('breadth_ema_period', 100)))
            key = ('breadth', breadth_ma_type, ma_p)
            if key not in _tilt_cache:
                _tilt_cache[key] = compute_btc_regime_breadth_tilt_panel(
                    btc_close,
                    universe_breadth_panel,
                    ma_period     = ma_p,
                    ma_type       = breadth_ma_type,
                    max_bull_tilt = max_bull_tilt,
                    max_bear_tilt = max_bear_tilt,
                )
            return _tilt_cache[key]

        return None

    def strategy_fn(df_slice, params):
        J         = int(params['J'])
        # Asymmetric holding: K_long and K_short can rebalance on independent
        # schedules.  Legacy `K` (single value) is honoured as the default for
        # both legs so existing notebooks keep working unchanged.
        K_default = int(params.get('K', 0))
        K_long    = int(params.get('K_long',  K_default))
        K_short   = int(params.get('K_short', K_default))
        if K_long <= 0 or K_short <= 0:
            raise ValueError(
                "make_xs_strategy: provide either 'K' or both 'K_long' and 'K_short' in params"
            )
        pct_long  = float(params['pct_long'])
        pct_short = float(params['pct_short'])

        # ── Window panel to this fold's date range ────────────────────────────
        dates = df_slice.index
        win   = panel.reindex(dates)
        n     = len(dates)

        # ── Ranking signal (selectable formulation) ──────────────────────────
        daily_coin = win.pct_change()

        if signal_kind == 'residual_sharpe':
            daily_btc   = win['BTC'].pct_change()
            beta        = daily_coin.rolling(J).cov(daily_btc).divide(
                daily_btc.rolling(J).var(), axis=0
            )
            daily_resid = daily_coin.subtract(beta.multiply(daily_btc, axis=0))
            mom_signal  = (
                daily_resid.rolling(J).mean() / daily_resid.rolling(J).std()
            ).shift(1)
        elif signal_kind == 'rolling_sharpe':
            mom_signal = (
                daily_coin.rolling(J).mean() / daily_coin.rolling(J).std()
            ).shift(1)
        else:
            raise ValueError(
                f"signal_kind must be 'residual_sharpe' or 'rolling_sharpe', "
                f"got {signal_kind!r}"
            )

        # ── Lee-Swaminathan volume-change signal (Stage 2 refinement) ────────
        if pool_multiplier is not None and volume is not None:
            vol_short_w = int(params.get('vol_short_window', 7))
            vol_long_w  = int(params.get('vol_long_window',  30))
            if vol_short_w >= vol_long_w:
                vol_change = None
            else:
                vol_win    = volume.reindex(dates)
                vol_change = (vol_win.rolling(vol_short_w).mean() /
                              vol_win.rolling(vol_long_w).mean()).shift(1)
        else:
            vol_change = None

        rets_arr = win.pct_change().values            # numpy view for fast slicing
        # Inverse-vol window: per-trial when 'iv_vol_window' in params, otherwise
        # the factory default.  Cached by window value across trials.
        iv_w     = int(params.get('iv_vol_window', _default_iv_window))
        vol_arr  = _get_vol_panel(iv_w).reindex(dates).values   # shape (n, n_coins)

        # Bar-wise regime tilt and dispersion confidence — sampled at t-1 daily.
        # With asymmetric K the legs rebalance on different schedules, so there
        # is no single "rebalance moment" at which to sample.  Daily lookup is
        # the natural generalisation; for symmetric K the difference vs the old
        # K-held convention is negligible (BTC EMA / breadth move slowly).
        tilt_panel = _resolve_tilt_panel(params)
        if tilt_panel is not None:
            tilt_arr = tilt_panel.reindex(dates).fillna(0.0).values
        else:
            tilt_arr = np.zeros(n)

        cp = _resolve_confidence_panel(params)
        if cp is not None:
            conf_arr = cp.reindex(dates).fillna(1.0).values
        else:
            conf_arr = np.ones(n)

        # ── Per-leg rebalance pass ──────────────────────────────────────────
        # Each leg runs an independent loop with its own K_leg.  The two legs
        # share the same signal, vol_change, universe filter, and inverse-vol
        # weights, but pick coins and hold them on independent schedules.
        def _run_leg(side, K_leg, pct):
            leg_ret_arr  = np.full(n, np.nan)
            n_held_arr   = np.zeros(n)
            turn_arr     = np.zeros(n)
            uni_size_arr = np.zeros(n)
            side_uses_pool = (volume_filter_legs == 'both' or
                              volume_filter_legs == f'{side}_only')
            ascending = (side == 'short')   # short = bottom of ranked list

            prev_coins = []
            for r in range(J + 1, n, K_leg):
                signal = mom_signal.iloc[r]

                # Dynamic universe: eligible coins at dates[r-1] (formation)
                if volume is not None and meta is not None:
                    eligible = get_universe(
                        as_of_date     = dates[r - 1],
                        volume         = volume,
                        meta           = meta,
                        top_n          = _top_n,
                        min_avg_volume = _min_vol,
                        min_age_days   = _min_age,
                        volume_window  = _vol_window,
                        avg_vol_panel  = avg_vol_panel,
                    )
                    eligible = [c for c in eligible if c in coin_to_idx]
                    if eligible:
                        signal = signal[eligible]

                valid  = signal.dropna()
                n_pick = max(1, int(pct * len(valid)))

                # ── Stage 1: candidate pool (per-leg) ─────────────────────
                if pool_multiplier is not None and side_uses_pool:
                    half_universe = len(valid) // 2
                    n_pool = min(half_universe, int(round(n_pick * pool_multiplier)))
                    if n_pool < n_pick or len(valid) < n_pool:
                        continue
                    ranked = valid.sort_values(ascending=ascending)
                    pool   = ranked.index[:n_pool].tolist()

                    # ── Stage 2: vol_change refinement ────────────────────
                    if vol_change is not None:
                        vc_now = vol_change.iloc[r]
                        coins  = (vc_now[pool].sort_values(ascending=False)
                                              .head(n_pick).index.tolist())
                        if len(coins) < n_pick:
                            coins = pool[:n_pick]
                    else:
                        coins = pool[:n_pick]
                else:
                    if len(valid) < n_pick:
                        continue
                    ranked = valid.sort_values(ascending=ascending)
                    coins  = ranked.index[:n_pick].tolist()

                if not coins:
                    continue

                idx = [coin_to_idx[c] for c in coins]
                if not idx:
                    continue

                uni_size_arr[r] = len(valid)
                turn_arr[r] = (len(set(coins) - set(prev_coins)) / len(coins)
                               if prev_coins else 1.0)

                # ── Vectorised holding-period basket return ───────────────
                hold_end = min(r + K_leg, n)
                slc      = rets_arr[r:hold_end][:, idx]
                vols     = vol_arr[r - 1][idx]
                inv      = np.where(np.isnan(vols) | (vols == 0), 0.0, 1.0 / vols)
                w        = (inv / inv.sum() if inv.sum() > 0
                            else np.full_like(vols, 1.0 / len(vols)))

                mask  = ~np.isnan(slc)
                denom = (mask * w).sum(axis=1)
                denom = np.where(denom == 0, np.nan, denom)
                basket = np.nansum(slc * w, axis=1) / denom

                leg_ret_arr[r:hold_end] = basket
                n_held_arr [r:hold_end] = len(idx)
                prev_coins = coins

            return leg_ret_arr, n_held_arr, turn_arr, uni_size_arr

        long_ret_arr,  n_long_arr,  long_turn_arr,  uni_long  = _run_leg('long',  K_long,  pct_long)
        short_ret_arr, n_short_arr, short_turn_arr, uni_short = _run_leg('short', K_short, pct_short)

        # ── Bar-wise combine (tilt and confidence sampled daily at t-1) ─────
        def _lag1(a, fill):
            out     = np.empty_like(a)
            out[0]  = fill
            out[1:] = a[:-1]
            return out

        tilt_lag = _lag1(tilt_arr, 0.0)
        conf_lag = _lag1(conf_arr, 1.0)

        long_wo  = 0.5 + tilt_lag
        short_wo = 0.5 - tilt_lag

        long_filled  = np.where(np.isnan(long_ret_arr),  0.0, long_ret_arr)
        short_filled = np.where(np.isnan(short_ret_arr), 0.0, short_ret_arr)

        strategy_returns_arr = conf_lag * (long_wo  * long_filled
                                          - short_wo * short_filled)

        turnover_arr  = long_turn_arr + short_turn_arr
        universe_size = np.maximum(uni_long, uni_short)

        regime_label_arr = np.empty(n, dtype=object)
        regime_label_arr[:] = 'neutral'
        regime_label_arr[tilt_lag > 0] = 'bull'
        regime_label_arr[tilt_lag < 0] = 'bear'

        # ── Series-ify outputs ───────────────────────────────────────────────
        strategy_returns = pd.Series(strategy_returns_arr, index=dates).fillna(0.0)
        long_returns     = pd.Series(long_filled,    index=dates)
        short_returns    = pd.Series(short_filled,   index=dates)
        turnover         = pd.Series(turnover_arr,   index=dates)
        universe_size_s  = pd.Series(universe_size,  index=dates)
        long_turnover    = pd.Series(long_turn_arr,  index=dates)
        short_turnover   = pd.Series(short_turn_arr, index=dates)
        n_long_held      = pd.Series(n_long_arr,     index=dates)
        n_short_held     = pd.Series(n_short_arr,    index=dates)
        long_weight_s    = pd.Series(long_wo,        index=dates)
        short_weight_s   = pd.Series(short_wo,       index=dates)
        regime_label_s   = pd.Series(regime_label_arr, index=dates)
        confidence_s     = pd.Series(conf_lag,       index=dates)

        # ── Build output DataFrame ───────────────────────────────────────────
        sr_filled = strategy_returns.fillna(0.0)
        equity    = (1.0 + sr_filled).cumprod()

        result = pd.DataFrame({
            'Open':             equity.shift(1).bfill(),
            'High':             equity,
            'Low':              equity,
            'Close':            equity,
            'Volume':           1.0,
            'position':         1,
            'position_size':    1.0,
            'strategy_returns': strategy_returns,
            'turnover':         turnover,
            'long_ret':         long_returns,
            'short_ret':        short_returns,
            'universe_size':    universe_size_s,
            'long_turnover':    long_turnover,
            'short_turnover':   short_turnover,
            'n_long_held':      n_long_held,
            'n_short_held':     n_short_held,
            'long_weight':      long_weight_s,
            'short_weight':     short_weight_s,
            'regime_label':     regime_label_s,
            'confidence':       confidence_s,
        }, index=dates)

        return result, ['strategy_returns']

    return strategy_fn
