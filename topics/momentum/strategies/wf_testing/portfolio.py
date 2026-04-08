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

SCENARIO = 'close'   # ← 'close' | integer hour e.g. 10 for 10am | 'worst_long'

def apply_scenario(coin_dfs, scenario):
    """
    Adjusts execution price by modifying Close on two specific bar types per trade.

    plot_portfolio_oos computes: strategy_ret[T] = position[T] * pct_change(Close)[T]
    (no shift — position[T] is already the active position on bar T)

    WHY previous approach was wrong:
      Replacing Close on signal days (where position is already non-zero) means
      the exec price appears in both the numerator of bar T and denominator of
      bar T+1. These always cancel: (exec/prev) * (close/exec) = close/prev.
      Results are identical to baseline, or broken for 1-bar trades.

    CORRECT approach — replace Close on bars where position = 0 on one side:

      PRE-ENTRY bar (position=0, next position!=0):
        position=0 so this bar contributes 0 to returns regardless of Close.
        But Close[pre_entry] is the denominator for the NEXT bar's pct_change.
        Setting it to exec_price makes the first holding bar capture:
          (Close[entry] - exec_price) / exec_price  ✓  no cancellation

      LAST-ACTIVE bar (position!=0, next position=0):
        The following bar has position=0, contributing 0, so nothing cancels.
        Setting Close[last_active] to exec_price makes that bar capture:
          (exec_price - Close[prev_holding]) / Close[prev_holding]  ✓
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
        pos = d['position']

        # last flat bar before entering a position
        pre_entry   = (pos == 0) & (pos.shift(-1).fillna(0) != 0)
        # last holding bar before going flat
        last_active = (pos != 0) & (pos.shift(-1).fillna(0) == 0)

        exec_close = d['Close'].copy()

        if isinstance(scenario, int):
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
            # shift(-1): value at index T is HH:00 price of day T+1 (execution day)
            next_exec = prices_hh.shift(-1).reindex(d.index).ffill()

            exec_close[pre_entry]   = next_exec[pre_entry]
            exec_close[last_active] = next_exec[last_active]
            print(f'  {coin}: {scenario}h UTC — {int(pre_entry.sum())} entries, {int(last_active.sum())} exits adjusted')

        elif scenario == 'worst_long':
            # High/Low already in the pkl — no API call needed
            # shift(-1): value at T is next day's High / Low
            next_high = d['High'].shift(-1)
            next_low  = d['Low'].shift(-1)
            exec_close[pre_entry]   = next_high[pre_entry]
            exec_close[last_active] = next_low[last_active]

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
