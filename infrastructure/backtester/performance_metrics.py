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

def calculate_total_return(equity_curve):

    total_return = equity_curve.iloc[-1] - 1
    return total_return


def calculate_sharpe_ratio(returns, periods_per_year):

    returns = returns.dropna()

    if len(returns) < 2:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(equity_curve):

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    
    return max_dd


def identify_trades_pairs(data):
    """
    Trade identification for pairs strategies where df['Close'] is a synthetic
    equity curve (already direction-adjusted by the engine).  P&L is always
    (exit_equity - entry_equity) / entry_equity — the directional sign must NOT
    be flipped for shorts because the equity curve already accounts for it.
    """
    trades = []

    change_points = data[data['position_change'] > 0].copy()

    if len(change_points) == 0:
        return pd.DataFrame()

    first_pos       = data['position'].iloc[0]
    in_position     = first_pos != 0
    entry_price     = data['Close'].iloc[0] if in_position else None
    entry_direction = first_pos             if in_position else None
    entry_time      = data.index[0]         if in_position else None

    for idx, row in data.iterrows():
        current_position = row['position']

        if not in_position and current_position != 0:
            in_position     = True
            entry_price     = row['Close']
            entry_direction = current_position
            entry_time      = idx

        elif in_position and (current_position == 0 or current_position != entry_direction):
            exit_price = row['Close']
            exit_time  = idx

            # equity curve is already signed — always use (exit - entry) / entry
            pnl = (exit_price - entry_price) / entry_price

            trades.append({
                'entry_time':  entry_time,
                'exit_time':   exit_time,
                'entry_price': entry_price,
                'exit_price':  exit_price,
                'direction':   'Long' if entry_direction == 1 else 'Short',
                'pnl':         pnl,
            })

            if current_position != 0:
                entry_price     = row['Close']
                entry_direction = current_position
                entry_time      = idx
            else:
                in_position = False

    return pd.DataFrame(trades)


def identify_trades(data):

    pos    = data['position'].to_numpy()
    closes = data['Close'].to_numpy()
    index  = data.index

    n = len(pos)
    if n == 0:
        return pd.DataFrame()

    # prev_pos: shift right by 1; pad first element with 0 unless position already
    # open on bar 0 (burn-in carry-over), in which case treat bar 0 as an entry.
    prev_pos        = np.empty(n, dtype=pos.dtype)
    prev_pos[1:]    = pos[:-1]
    prev_pos[0]     = 0          # bar-0 handled as special entry below

    # entry: previous flat (or bar 0 already in position), current non-zero
    entry_mask = ((prev_pos == 0) & (pos != 0))
    # handle position already open on bar 0
    if pos[0] != 0:
        entry_mask[0] = True

    # exit: was in position, now flat OR flipped direction
    exit_mask  = (prev_pos != 0) & ((pos == 0) | (pos != prev_pos))

    entry_idx = np.where(entry_mask)[0]
    exit_idx  = np.where(exit_mask)[0]

    if len(entry_idx) == 0:
        return pd.DataFrame()

    # pair each entry with the next exit that comes after it
    trades = []
    ei = 0
    for e in entry_idx:
        # find first exit strictly after this entry
        candidates = exit_idx[exit_idx > e]
        if len(candidates) == 0:
            break
        x = candidates[0]

        direction   = int(pos[e])
        entry_price = float(closes[e])
        exit_price  = float(closes[x])
        pnl = ((exit_price - entry_price) / entry_price if direction == 1
               else (entry_price - exit_price) / entry_price)

        trades.append({
            'entry_time':  index[e],
            'exit_time':   index[x],
            'entry_price': entry_price,
            'exit_price':  exit_price,
            'direction':   'Long' if direction == 1 else 'Short',
            'pnl':         pnl,
        })

    return pd.DataFrame(trades)


def calculate_win_rate(trades_df):

    if len(trades_df) == 0:
        return 0.0
    
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    total_trades = len(trades_df)
    
    win_rate = winning_trades / total_trades
    return win_rate


def calculate_num_trades(trades_df):

    return len(trades_df)


def calculate_avg_win_loss_ratio(trades_df):

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


def calculate_profit_factor(trades_df):

    if len(trades_df) == 0:
        return 0.0
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    profit_factor = gross_profit / gross_loss
    return profit_factor


def calculate_calmar_ratio(total_return, max_drawdown, periods_per_year, n_periods):
    if max_drawdown == 0:
        return 0.0
    # annualise the total return
    n_years = n_periods / periods_per_year
    if n_years == 0:
        return 0.0
    annualised_return = (1 + total_return) ** (1 / n_years) - 1
    return annualised_return / abs(max_drawdown)


def build_realized_equity_curve(position, position_size, raw_returns, cost):
    """
    Build an equity curve using realized-sizing (matches portfolio_metrics.build_realized_equity).

    At each trade entry the notional is fixed as:
        entry_notional = position_size_at_entry × realized_equity

    where realized_equity is updated only when trades CLOSE.  Unrealized gains
    from open positions never inflate the notional of subsequent entries.

    Parameters
    ----------
    position      : pd.Series  effective position (already shift(1)-lagged by caller).
                               Represents the position held DURING each bar.
    position_size : pd.Series  effective size     (already shift(1)-lagged).
    raw_returns   : pd.Series  raw per-bar undirected instrument returns
                               (e.g. Close.pct_change()).  For precomputed strategy
                               returns that already carry direction, pass the signed
                               series and ensure position_size=1 to avoid re-scaling.
    cost          : float      cost fraction per unit of |position.diff()|.

    Returns
    -------
    pd.Series  equity curve starting near 1.0, indexed like raw_returns.
    """
    pos  = position.fillna(0).values.astype(float)
    size = position_size.fillna(0).values.astype(float)
    raw  = raw_returns.fillna(0).values.astype(float)
    idx  = raw_returns.index

    n              = len(pos)
    realized       = 1.0
    entry_notional = 0.0
    cum_mult       = 1.0      # signed cumulative mult since entry: accounts for direction
    prev_pos       = 0.0
    equity         = np.empty(n)

    for i in range(n):
        curr_pos = pos[i]   # position held DURING bar i (eff_pos, already shifted)
        sz       = size[i]  # position size held during bar i (already shifted)
        r        = raw[i]   # raw undirected bar return

        # ── 1. Exit: position closed or flipped ─────────────────────────────
        # Runs before earning so closing-bar return is NOT added to realized P&L.
        if prev_pos != 0.0 and (curr_pos == 0.0 or curr_pos != prev_pos):
            realized      += entry_notional * (cum_mult - 1.0)
            entry_notional = 0.0
            cum_mult       = 1.0

        # ── 2. Trade cost: applied at every position change ──────────────────
        pos_chg = abs(curr_pos - prev_pos)
        if pos_chg > 0.0:
            portfolio = realized + entry_notional * (cum_mult - 1.0)
            realized -= pos_chg * cost * portfolio

        # ── 3. Entry: size new notional against REALIZED equity only ─────────
        if curr_pos != 0.0 and (prev_pos == 0.0 or curr_pos != prev_pos):
            entry_notional = sz * realized
            cum_mult       = 1.0

        # ── 4. Earn bar i's return on the position held DURING this bar ──────
        # curr_pos is the effective (shifted) position, so curr_pos != 0 means
        # we ARE holding during bar i. np.sign applies direction: shorts gain
        # when price falls (cum_mult drops below 1 when price rises).
        if curr_pos != 0.0:
            cum_mult *= (1.0 + np.sign(curr_pos) * r)

        # ── 5. Record portfolio equity ────────────────────────────────────────
        equity[i] = realized + entry_notional * (cum_mult - 1.0)
        prev_pos  = curr_pos

    eq = pd.Series(equity, index=idx)
    return eq.ffill().fillna(1.0)


def build_equity_curve(returns, return_type="arithmetic"):
    """
    return_type:
        'arithmetic' -> equity = (1 + r).cumprod()
        'log'        -> equity = exp(cumsum(r))
    """
    returns = returns.fillna(0.0)

    if return_type == "log":
        equity_curve = np.exp(returns.cumsum())
    else:
        equity_curve = (1 + returns).cumprod()

    equity_curve = pd.Series(equity_curve, index=returns.index)
    equity_curve.fillna(1.0, inplace=True)
    return equity_curve


def to_arithmetic_returns(returns, return_type="arithmetic"):
    """
    Convert returns to arithmetic returns for reporting metrics.
    """
    returns = returns.copy()

    if return_type == "log":
        return np.exp(returns) - 1.0
    
    return returns


def calculate_yearly_metrics(returns, equity_curve, periods_per_year):

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

        if start_value != 0:
            yearly_return = (end_value - start_value) / start_value
        else:
            yearly_return = 0.0

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


def calculate_all_metrics(data, net_returns, cost, return_type="arithmetic", strategy_type="single_asset"):
    """
    Main function to calculate all performance metrics and compile results.

    return_type:
        'arithmetic' -> net_returns are simple returns
        'log'        -> net_returns are log returns
    """

    # Use arithmetic returns for reporting metrics like Sharpe
    arith_returns = to_arithmetic_returns(net_returns, return_type=return_type)

    # Build equity curve correctly depending on return type
    equity_curve = build_equity_curve(net_returns, return_type=return_type)
    
    # Infer data frequency
    periods_per_year = infer_frequency(data.index)
    
    # Identify trades
    if strategy_type == 'pairs':
       trades_df = identify_trades_pairs(data)
    else:
       trades_df = identify_trades(data)
    
    # Calculate individual metrics
    total_return = calculate_total_return(equity_curve)
    sharpe_ratio = calculate_sharpe_ratio(arith_returns, periods_per_year)
    max_drawdown = calculate_max_drawdown(equity_curve)
    win_rate = calculate_win_rate(trades_df)
    num_trades = calculate_num_trades(trades_df)
    avg_win_loss = calculate_avg_win_loss_ratio(trades_df)
    profit_factor = calculate_profit_factor(trades_df)
    calmar_ratio = calculate_calmar_ratio(
    total_return,
    max_drawdown,
    periods_per_year,
    n_periods = len(net_returns),
)
    
    # Calculate yearly metrics on arithmetic returns, using correct equity curve
    yearly_metrics = calculate_yearly_metrics(arith_returns, equity_curve, periods_per_year)
    
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