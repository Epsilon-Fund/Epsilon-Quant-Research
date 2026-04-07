import os
import sys
import pandas as pd

# ── repo root — works on both Mac and Windows ────────────────────────────────
ROOT = os.path.expanduser('~/Desktop/epsilon/github/Epsilon-Quant-Research')
# ROOT = r'C:\Users\user\Documents\Epsilon Fund\Epsilon-Quant-Research'  # ← Windows path
# ─────────────────────────────────────────────────────────────────────────────
WF_DIR = os.path.join(ROOT, 'topics', 'momentum', 'strategies', 'wf_testing')

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


# ── portfolio configuration ───────────────────────────────────────────────────
# show_coins   : which coins to include in the portfolio this run.
#                Subset of ACTIVE_COINS — e.g. ['BTC', 'ETH'] drops SOL.
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
    coin_dfs   = coin_dfs,
   # weights  = {'BTC': 0.1, 'ETH': 0.3, 'XRP': 0.3, 'BNB':0.3},  # uncomment for custom weights
    show_coins = # list(coin_dfs.keys()),
     ['ETH', 'XRP','BNB'], # or list(coin_dfs.keys()) all loaded coins
     #list(coin_dfs.keys()),
    benchmark  = {'BTC': coin_dfs['BTC']},   # change to None for equal-weight B&H
    show       = True,
    save_html  = WF_DIR + '/portfolio.html',  # set to None to skip saving
)