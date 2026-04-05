import pandas as pd

from performance_metrics import calculate_all_metrics
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
    if 'position_size' in df.columns:
        df['effective_position'] = (df['position'] * df['position_size']).shift(1)
    else:
        df['effective_position'] = df['position'].shift(1)

    df['strategy_returns'] = df['effective_position'] * raw_strategy_returns

    # costs based on position changes
    df['position_change'] = df['position'].diff().abs()
    df['trade_cost'] = df['position_change'] * cost

    df['net_returns'] = df['strategy_returns'] - df['trade_cost']

    # make a synthetic Close if missing (needed for trade logging + plots)
    if 'Close' not in df.columns:
        df['Close'] = (1 + df['net_returns'].fillna(0)).cumprod()

    metrics = calculate_all_metrics(
        data=df,
        net_returns=df['net_returns'],
        cost=cost
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