import os
import sys
import pandas as pd

ROOT    = r'/Users/justiniturregui/Desktop/epsilon/github/Epsilon-Quant-Research'
WF_DIR  = ROOT + '/topics/statistical-arbitrage/strategies/testing'

sys.path.append(ROOT + '/infrastructure/backtester')
sys.path.append(ROOT + '/infrastructure/walkforward')

from wf_visualizer import plot_portfolio_oos

# ── pairs to include ──────────────────────────────────────────────────────────
# Each notebook saves its OOS results as <pair_name>_oos.pkl in this directory.
# Run the notebooks first (through the walk-forward cell) to generate the files.
#
# Expected files:
#   trx_link_oos.pkl   → testing playground copy.ipynb   (TRXUSDT vs LINKUSDT)
#   bnb_ftm_oos.pkl    → testing playground copy 2.ipynb (BNBUSDT vs FTMUSDT)
#   shib_bonk_oos.pkl  → testing playground copy 3.ipynb (SHIBUSDT vs BONKUSDT)
#   matic_apt_oos.pkl  → testing playground copy 4.ipynb (MATICUSDT vs APTUSDT)

pair_dfs = {}
for fname in os.listdir(WF_DIR):
    if fname.endswith('_oos.pkl'):
        pair = fname.replace('_oos.pkl', '').upper()
        pair_dfs[pair] = pd.read_pickle(os.path.join(WF_DIR, fname))

print(f'Loaded: {list(pair_dfs.keys())}')


# ── portfolio configuration ───────────────────────────────────────────────────
# show_pairs   : which pairs to include in this portfolio run.
#                Must match keys in pair_dfs (e.g. 'TRX_LINK', 'BNB_FTM').
#                Use list(pair_dfs.keys()) to include all loaded pairs.
#
# weights      : capital allocation per pair. Commented out = equal weight.
#                Uncomment and edit to use custom weights.
#                Auto-normalised — weights don't need to sum to 1.
#
# benchmark    : equity curve the portfolio is compared against.
#                  None                              → equal-weight B&H of show_pairs
#                  {'TRX_LINK': pair_dfs['TRX_LINK']} → single pair B&H
#
# save_html    : path to save the chart as a standalone HTML file.
#                Set to None to skip saving.

metrics = plot_portfolio_oos(
    coin_dfs   = pair_dfs,
   # weights  = {'TRX_LINK': 0.25, 'BNB_FTM': 0.25, 'SHIB_BONK': 0.25, 'MATIC_APT': 0.25},
    show_coins = list(pair_dfs.keys()),
    benchmark  = None,   # None = equal-weight B&H of included pairs
    show       = True,
    save_html  = WF_DIR + '/portfolio.html',  # set to None to skip saving
)
