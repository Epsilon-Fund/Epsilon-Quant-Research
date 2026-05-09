import pandas as pd

from performance_metrics import calculate_all_metrics, build_realized_equity_curve
from visualizer import plot_results, plot_trades_on_price

""" 
Args:
    - 'Close': Price data (required)
    - 'position': Trading signals (required) 
                1 = long, 0 = flat, -1 = short
    - 'position_size': Fraction of capital (optional, default 1.0)
                    0.5 = 50% of capital, 1.0 = 100%
    
    cost: Trading cost as % per trade
    show_plot: Display chart (default True)
    save_html: Save as HTML file (optional)
    show_trades: Show trade markers on price chart (default False)
    benchmark_data: Buy & hold comparison (default: uses data['Close'])
    
Returns:
    dict: Complete backtest metrics
 """


def backtest(data, cost=0.0, show_plot=True, save_html=None, show_trades=False, benchmark_data=None):

    df = data.copy()

    if 'position' not in df.columns:
        raise ValueError("Data must have 'position' column (1=long, 0=flat, -1=short)")

    # decide which return stream to use
    use_precomputed = 'strategy_returns' in df.columns

    if not use_precomputed:
        if 'Close' not in df.columns:
            raise ValueError("Data must have 'Close' column OR a precomputed 'strategy_returns' column")

        if benchmark_data is None:
            benchmark_data = df[['Close']].copy()

        df['returns'] = df['Close'].pct_change()
        raw_strategy_returns = df['returns']
    else:
        # strategy_returns should already be the per-bar return of the strategy
        # eg for pairs: ret_y - b*ret_x
        raw_strategy_returns = df['strategy_returns'].copy()

        # if no benchmark supplied, skip benchmark by default
        if benchmark_data is None:
            benchmark_data = None

    # apply position sizing and 1-bar shift
    eff_pos  = df['position'].shift(1).fillna(0)
    eff_size = (df['position_size'].shift(1).fillna(0)
                if 'position_size' in df.columns
                else pd.Series(1.0, index=df.index))

    df['effective_position'] = eff_pos * eff_size

    # costs based on position changes (used for trade annotation / pairs cost)
    # If the strategy supplies an explicit 'turnover' column (fractional portfolio
    # turnover per bar, 0–2), use that instead of the binary position-change signal.
    # This lets portfolio strategies charge accurate partial-turnover costs without
    # baking them into strategy_returns.  Single-asset strategies that don't return
    # a 'turnover' column are unaffected.
    df['position_change'] = df['position'].diff().abs()
    if 'turnover' in df.columns:
        df['trade_cost'] = df['turnover'].fillna(0.0) * cost
    else:
        df['trade_cost'] = df['position_change'] * cost

    if not use_precomputed:
        # ── realized sizing (single-asset strategies) ─────────────────────────
        # Entry notional = position_size × realized_equity (closed-trade equity only).
        # Unrealized gains do not inflate the notional of subsequent entries.
        # raw_returns here are undirected (Close.pct_change); direction is applied
        # inside build_realized_equity_curve via sign(position).
        equity_curve = build_realized_equity_curve(
            position      = eff_pos,
            position_size = eff_size,
            raw_returns   = raw_strategy_returns,
            cost          = cost,
        )
        df['net_returns'] = equity_curve.pct_change().fillna(0.0)
    else:
        # ── pairs / precomputed returns ───────────────────────────────────────
        # raw_strategy_returns is already directional (direction baked in by the
        # strategy).  Apply position gating and cost via standard MTM compounding.
        df['strategy_returns'] = eff_pos * raw_strategy_returns
        df['net_returns']      = df['strategy_returns'] - df['trade_cost']
        equity_curve           = (1 + df['net_returns'].fillna(0)).cumprod()

    # make a synthetic Close if missing (needed for trade logging + plots)
    if 'Close' not in df.columns:
        df['Close'] = equity_curve

    metrics = calculate_all_metrics(
        data=df,
        net_returns=df['net_returns'],
        cost=cost,
        strategy_type = 'pairs' if use_precomputed else 'single_asset'
    )

    if show_plot or save_html:
        plot_results(
            metrics,
            benchmark_data=benchmark_data,
            show=show_plot,
            save_html=save_html
        )

    if show_trades and 'Close' in df.columns and len(metrics['trades']) > 0:
        trade_html = None
        if save_html:
            trade_html = save_html.replace('.html', '_trades.html')

        plot_trades_on_price(
            df,
            metrics['trades'],
            show=show_plot,
            save_html=trade_html
        )

    return metrics

def build_pair_df(price_df, y_col, x_col,
                  lookback=126, z_lookback=60,
                  entry=2.0, exit=0.5,
                  cost=0.0):

    df = price_df[[y_col, x_col]].dropna().copy()

    # log prices
    log_y = np.log(df[y_col])
    log_x = np.log(df[x_col])

    # rolling beta (hedge ratio)
    beta = log_y.rolling(lookback).cov(log_x) / log_x.rolling(lookback).var()
    beta = beta.shift(1)

    # spread
    spread = log_y - beta * log_x

    # z-score
    mean = spread.rolling(z_lookback).mean()
    std  = spread.rolling(z_lookback).std()
    z = (spread - mean) / std

    # position: +1 long spread, -1 short spread
    pos = pd.Series(0.0, index=df.index)
    pos[z > entry]  = -1.0
    pos[z < -entry] =  1.0
    pos[z.abs() < exit] = 0.0

    pos = pos.replace(0.0, np.nan).ffill().fillna(0.0)

    # returns
    ret_y = log_y.diff()
    ret_x = log_x.diff()

    pair_ret = ret_y - beta * ret_x

    # build output df for engine
    out = pd.DataFrame(index=df.index)
    out["position"] = pos
    out["strategy_returns"] = pair_ret

    # simple cost 
    if cost > 0:
        turnover = pos.diff().abs().fillna(0.0)
        out["strategy_returns"] = out["strategy_returns"] - cost * turnover

    return out.dropna()