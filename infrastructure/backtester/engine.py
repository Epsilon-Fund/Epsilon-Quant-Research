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


def backtest(data, cost=0.0, show_plot=True, save_html=None, show_trades=False, benchmark_data=None):     # Main backtesting function that calculates performance metrics and generates visualizations

    if 'Close' not in data.columns:
        raise ValueError("Data must have 'Close' column")
    if 'position' not in data.columns:
        raise ValueError("Data must have 'position' column (1=long, 0=flat, -1=short)")
    
    if benchmark_data is None:
        benchmark_data = data[['Close']]
    
    df = data.copy()
    
    df['returns'] = df['Close'].pct_change()
    
    if 'position_size' in df.columns:
        df['effective_position'] = (df['position'] * df['position_size']).shift(1)
    else:
        df['effective_position'] = df['position'].shift(1)
    
    df['strategy_returns'] = df['effective_position'] * df['returns']
    
    df['position_change'] = df['position'].diff().abs()
    df['trade_cost'] = df['position_change'] * cost
    
    df['net_returns'] = df['strategy_returns'] - df['trade_cost']
    
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
    
    if show_trades and len(metrics['trades']) > 0:
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