import pandas as pd
import numpy as np

# Helper functions 

def infer_frequency(index):

    if len(index) < 2:
        return 365
    
    time_diffs = index.to_series().diff().dropna()
    median_diff = time_diffs.median()
    
    hours = median_diff.total_seconds() / 3600
    
    if hours <= 1:
        return 8760 
    elif hours <= 4:
        return 2190  
    elif hours <= 24:
        return 365   
    elif hours <= 168:
        return 52   
    else:
        return 12  
    

# Metrics functions

def calculate_total_return(equity_curve):    # Calculate total return

    total_return = equity_curve.iloc[-1] - 1
    return total_return

def calculate_sharpe_ratio(returns, periods_per_year):   # Calculate Sharpe 

    if len(returns) < 2:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(equity_curve):   # Calculate max drawdown

    running_max = equity_curve.cummax()
    
    drawdown = (equity_curve - running_max) / running_max
    
    max_dd = drawdown.min()
    
    return max_dd


def identify_trades(data):   # Identify and log the trades

    trades = []
    
    change_points = data[data['position_change'] > 0].copy()
    
    if len(change_points) == 0:
        return pd.DataFrame()
    
    in_position = False
    entry_price = None
    entry_direction = None
    entry_time = None
    
    for idx, row in data.iterrows():
        current_position = row['position']
        
        # Entering a position (from 0 to 1 or -1)
        if not in_position and current_position != 0:
            in_position = True
            entry_price = row['Close']
            entry_direction = current_position
            entry_time = idx
        
        # Exiting a position (from 1/-1 to 0 or flipping)
        elif in_position and (current_position == 0 or (current_position != 0 and current_position != entry_direction)):
            exit_price = row['Close']
            exit_time = idx
            
            # Calculate P&L
            if entry_direction == 1:  # Long trade
                pnl = (exit_price - entry_price) / entry_price
            else:  # Short trade (entry_direction == -1)
                pnl = (entry_price - exit_price) / entry_price
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'Long' if entry_direction == 1 else 'Short',
                'pnl': pnl
            })
            
            # If flipping (not exiting to 0), immediately enter new position
            if current_position != 0:
                entry_price = row['Close']
                entry_direction = current_position
                entry_time = idx
            else:
                in_position = False
    
    return pd.DataFrame(trades)


def calculate_win_rate(trades_df):    # Calculate win rate

    if len(trades_df) == 0:
        return 0.0
    
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    total_trades = len(trades_df)
    
    win_rate = winning_trades / total_trades
    return win_rate


def calculate_num_trades(trades_df):   # Calculate number of trades

    return len(trades_df)


def calculate_avg_win_loss_ratio(trades_df):    # Calculate average win/loss ratio

    if len(trades_df) == 0:
        return 0.0
    
    winning_trades = trades_df[trades_df['pnl'] > 0]['pnl']
    losing_trades = trades_df[trades_df['pnl'] < 0]['pnl']
    
    if len(winning_trades) == 0 or len(losing_trades) == 0:
        return 0.0
    
    avg_win = winning_trades.mean()
    avg_loss = abs(losing_trades.mean())
    
    ratio = avg_win / avg_loss
    return ratio


def calculate_profit_factor(trades_df):    # Calculate profit factor

    if len(trades_df) == 0:
        return 0.0
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    profit_factor = gross_profit / gross_loss
    return profit_factor


def calculate_calmar_ratio(total_return, max_drawdown):    # Calculate Calmar ratio

    if max_drawdown == 0:
        return 0.0
    
    calmar = total_return / abs(max_drawdown)
    return calmar


def calculate_yearly_metrics(returns, equity_curve, periods_per_year):    # Calculate yearly returns, Sharpe, and max drawdown

    returns_by_year = returns.groupby(returns.index.year)
    equity_by_year = equity_curve.groupby(equity_curve.index.year)
    
    yearly_returns = {}
    yearly_sharpe = {}
    yearly_max_dd = {}
    
    for year in returns_by_year.groups.keys():
        year_returns = returns_by_year.get_group(year)
        year_equity = equity_by_year.get_group(year)

        start_value = year_equity.iloc[0]
        end_value = year_equity.iloc[-1]
        yearly_return = (end_value - start_value) / start_value
        yearly_returns[year] = yearly_return
        

        mean_ret = year_returns.mean()
        std_ret = year_returns.std()
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)
        else: 
            sharpe = 0.0
        yearly_sharpe[year] = sharpe
        

        running_max = year_equity.cummax()
        drawdown = (year_equity - running_max) / running_max
        yearly_max_dd[year] = drawdown.min()
    
    return {
        'yearly_returns': yearly_returns,
        'yearly_sharpe': yearly_sharpe,
        'yearly_max_drawdown': yearly_max_dd
    }

def calculate_all_metrics(data, net_returns, cost):    # Main function to calculate all performance metrics and compile results

    # Build equity curve
    equity_curve = (1 + net_returns).cumprod()
    equity_curve.fillna(1.0, inplace=True)
    
    # Infer data frequency
    periods_per_year = infer_frequency(data.index)
    
    # Identify trades
    trades_df = identify_trades(data)
    
    # Calculate individual metrics
    total_return = calculate_total_return(equity_curve)
    sharpe_ratio = calculate_sharpe_ratio(net_returns, periods_per_year)
    max_drawdown = calculate_max_drawdown(equity_curve)
    win_rate = calculate_win_rate(trades_df)
    num_trades = calculate_num_trades(trades_df)
    avg_win_loss = calculate_avg_win_loss_ratio(trades_df)
    profit_factor = calculate_profit_factor(trades_df)
    calmar_ratio = calculate_calmar_ratio(total_return, max_drawdown)
    
    # Calculate yearly metrics
    yearly_metrics = calculate_yearly_metrics(net_returns, equity_curve, periods_per_year)
    
    # Compile all metrics
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'avg_win_loss_ratio': avg_win_loss,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar_ratio,
        'yearly_returns': yearly_metrics['yearly_returns'],
        'yearly_sharpe': yearly_metrics['yearly_sharpe'],
        'yearly_max_drawdown': yearly_metrics['yearly_max_drawdown'],
        'cost_percent': cost,
        'equity_curve': equity_curve,
        'trades': trades_df
    }
    
    return metrics