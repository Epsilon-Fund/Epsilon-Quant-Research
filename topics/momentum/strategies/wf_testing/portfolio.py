import os
import sys
import pandas as pd

# ── repo root — works on both Mac and Windows ────────────────────────────────
ROOT = os.path.expanduser('~/Desktop/epsilon/github/Epsilon-Quant-Research')
# ROOT = r'C:\Users\user\Documents\Epsilon Fund\Epsilon-Quant-Research'  # ← Windows path
# ─────────────────────────────────────────────────────────────────────────────
WF_DIR = os.path.join(ROOT, 'topics', 'momentum', 'strategies', 'wf_testing')

sys.path.append(os.path.join(ROOT, 'infrastructure', 'data'))
sys.path.append(os.path.join(ROOT, 'infrastructure', 'walkforward'))
sys.path.append(os.path.join(ROOT, 'infrastructure', 'backtester'))

from wf_visualizer import plot_portfolio_oos

# ── coins to include ──────────────────────────────────────────────────────────

coin_dfs = {}
for fname in os.listdir(WF_DIR):
    if fname.endswith('_oos.pkl'):
        coin = fname.replace('_oos.pkl', '').upper()
        coin_dfs[coin] = pd.read_pickle(os.path.join(WF_DIR, fname))

print(f'Loaded: {list(coin_dfs.keys())}')


# ── execution scenario ────────────────────────────────────────────────────────
# Controls what price the strategy executes at.
# Signals and trades are unchanged — only the fill price is adjusted.
#
# How the baseline works:
#   Engine shift(1): signal on day T → execute at Close[T] (midnight UTC).
#   Each scenario replaces Close[T] on signal days only, so that bar T+1's
#   pct_change uses the new execution price instead of midnight close.
#   Holding-period returns (no position change) are never touched.
#
# 'close'      : baseline — execute at Close[T], midnight UTC
# integer hour : execute at HH:00 UTC on T+1 (e.g. 10 = 10am next day)
#                requires a Binance API call — use execution_scenarios.ipynb
#                for faster iteration (fetches once, switch time in one cell)
# 'worst_long' : enter at High[T+1], exit at Low[T+1] — no API call needed

SCENARIO = 'close'   # ← 'close' | integer hour (0-23) | 'worst_long'

def apply_scenario(coin_dfs, scenario):
    """
    Replaces Close on signal days only with the chosen execution price.

    Engine logic: strategy_return[T] = position[T-1] * pct_change(Close)[T]
    Signal day S: position[S] changes, effective_position[S] = position[S-1] = 0
    so Close[S] only feeds into bar S+1's pct_change — the execution bar.

    Replacing Close[S] = exec_price makes the execution bar capture:
        entry: (Close[S+1] - exec_price) / exec_price
        exit:  (exec_price - Close[S-1]) / Close[S-1]
    """
    if scenario == 'close':
        return coin_dfs

    if isinstance(scenario, int):
        from binance_client import get_binance_client
        client = get_binance_client()

    adjusted = {}
    for coin, df in coin_dfs.items():
        d = df.copy()
        d.index = d.index.normalize()

        # signal days: any day position changes value
        signal_days = d['position'] != d['position'].shift(1).fillna(d['position'].iloc[0])

        if isinstance(scenario, int):
            # fetch 1h data and extract HH:00 UTC price for each day
            symbol = coin + 'USDT'
            klines = client.get_historical_klines(
                symbol, '1h', str(d.index[0].date()), str(d.index[-1].date())
            )
            h = pd.DataFrame(klines, columns=[
                'Time','Open','High','Low','Close','Volume',
                'Close_time','Quote_volume','Trades','Taker_base','Taker_quote','Ignore'
            ])
            h['Time']  = pd.to_datetime(h['Time'], unit='ms', utc=True)
            h['Close'] = h['Close'].astype(float)
            h = h.set_index('Time')

            prices_hh = h[h.index.hour == scenario]['Close'].resample('1D').last()
            prices_hh.index = prices_hh.index.tz_localize(None).normalize()

            # shift(-1): index T now holds HH:00 price of T+1 (execution day)
            next_exec = prices_hh.shift(-1).reindex(d.index).ffill()

            exec_close = d['Close'].copy()
            exec_close[signal_days] = next_exec[signal_days]
            d['Close'] = exec_close
            print(f'  {coin}: {scenario}h UTC prices applied on signal days')

        elif scenario == 'worst_long':
            # entry signal days → fill at High[T+1], exit signal days → fill at Low[T+1]
            # High and Low are already in the pkl — no API call needed
            entry_signal = signal_days & (d['position'] != 0)
            exit_signal  = signal_days & (d['position'] == 0)

            # shift(-1): index T now holds next day's High / Low
            next_high = d['High'].shift(-1)
            next_low  = d['Low'].shift(-1)

            exec_close = d['Close'].copy()
            exec_close[entry_signal] = next_high[entry_signal]
            exec_close[exit_signal]  = next_low[exit_signal]
            d['Close'] = exec_close

        adjusted[coin] = d

    return adjusted


coin_dfs_exec = apply_scenario(coin_dfs, SCENARIO)
label = f'{SCENARIO}h_UTC' if isinstance(SCENARIO, int) else SCENARIO
print(f'Execution scenario: {label}')


# ── portfolio configuration ───────────────────────────────────────────────────
# show_coins   : which coins to include in the portfolio this run.
#                Must match keys in coin_dfs.
#
# weights      : capital allocation per coin. Commented out = equal weight.
#                Uncomment and edit to use custom weights.
#                Any coins in show_coins but missing from weights get equal share
#                of the remainder. Weights are auto-normalised so they don't
#                need to sum to exactly 1.
#
# benchmark    : what the strategy equity curve is compared against.
#                Three options:
#                  None                          → equal-weight B&H of show_coins
#                  {'BTC': coin_dfs['BTC']}      → single coin B&H (e.g. BTC only)
#                  {'BTC': coin_dfs['BTC'],
#                   'ETH': coin_dfs['ETH']}      → multi-coin equal-weight B&H
#
# save_html    : path to save the chart as a standalone HTML file.
#                Set to None to skip saving.

metrics = plot_portfolio_oos(
    coin_dfs   = coin_dfs_exec,
   # weights  = {'BTC': 0.05, 'ETH': 0.3, 'XRP': 0.2, 'BNB':0.15, 'SOL': 0.3},  # uncomment for custom weights
    show_coins = # list(coin_dfs.keys()),
     ['ETH', 'XRP','AVAX','SOL','BNB'], # or list(coin_dfs.keys()) all loaded coins
     #list(coin_dfs.keys()),
    benchmark  = {'BTC': coin_dfs['BTC']},   # change to None for equal-weight B&H
    show       = True,
    save_html  = WF_DIR + '/portfolio.html',  # set to None to skip saving
)


#Todo: make this a notebook

# Coins rthat work best and most consistent in order : Eth/ Sol, XRP, BNB, BTC
# weighting in that order improves
