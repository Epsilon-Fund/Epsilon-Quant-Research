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
# 'close'  : default — execute at prior day close (original behaviour)
# 'high'   : worst case — execute at the day's high
# '10am'   : execute at the 10am UTC candle (requires Binance API call)
#
# To use '10am', uncomment the binance_client import below.

SCENARIO = 'close'   # ← change to 'high' or '10am'

def apply_scenario(coin_dfs, scenario):
    if scenario == 'close':
        return coin_dfs

    if scenario == '10am':
        from binance_client import get_binance_client
        client = get_binance_client()

    adjusted = {}
    for coin, df in coin_dfs.items():
        d = df.copy()

        if scenario == 'high':
            # swap Close for High — every execution fills at the worst intraday price
            d['Close'] = d['High']

        elif scenario == '10am':
            # fetch hourly data and extract the 10am UTC candle close for each day
            symbol = coin + 'USDT'
            start  = str(d.index[0].date())
            end    = str(d.index[-1].date())
            klines = client.get_historical_klines(symbol, '1h', start, end)
            h = pd.DataFrame(klines, columns=[
                'Time','Open','High','Low','Close','Volume',
                'Close_time','Quote_volume','Trades','Taker_base','Taker_quote','Ignore'
            ])
            h['Time']  = pd.to_datetime(h['Time'], unit='ms', utc=True)
            h['Close'] = h['Close'].astype(float)
            h = h.set_index('Time')

            # keep only the 10am UTC bar then resample to daily
            prices_10am = (
                h[h.index.hour == 10]['Close']
                .resample('1D').last()
            )
            prices_10am.index = prices_10am.index.tz_localize(None).normalize()
            d.index = d.index.normalize()
            d['Close'] = prices_10am.reindex(d.index).ffill()
            print(f'  {coin}: 10am prices applied ({prices_10am.notna().sum()} days)')

        adjusted[coin] = d

    return adjusted


coin_dfs_exec = apply_scenario(coin_dfs, SCENARIO)
print(f'Execution scenario: {SCENARIO}')


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
