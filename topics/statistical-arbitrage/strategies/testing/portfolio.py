import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

ROOT   = str(Path(__file__).resolve().parents[4])
WF_DIR = str(Path(__file__).resolve().parent)

sys.path.append(ROOT + '/infrastructure/backtester')
from engine import backtest

# ── config ────────────────────────────────────────────────────────────────────
WEIGHT_SCHEME = 'inverse_vol'   # 'equal' | 'inverse_vol'
VOL_METHOD    = 'in_market'     # 'full' = all bars incl. flat days
                                # 'in_market' = only bars where position != 0
VOL_WINDOW    = None            # None = full OOS history; int = last N bars

# ── load pkl files ────────────────────────────────────────────────────────────
pair_dfs = {}
for fname in sorted(os.listdir(WF_DIR)):
    if fname.endswith('_oos.pkl'):
        pair = fname.replace('_oos.pkl', '').upper()
        pair_dfs[pair] = pd.read_pickle(os.path.join(WF_DIR, fname))

if not pair_dfs:
    print('No _oos.pkl files found. Run the save cell in each notebook first.')
    sys.exit(1)

print(f'Loaded: {list(pair_dfs.keys())}')

# ── weights ───────────────────────────────────────────────────────────────────
show_coins = list(pair_dfs.keys())

def compute_vol(k, method, window):
    ret = pair_dfs[k]['net_returns'].fillna(0)
    pos = pair_dfs[k]['position'].fillna(0)
    if window:
        ret = ret.iloc[-window:]
        pos = pos.iloc[-window:]
    if method == 'in_market':
        ret = ret[pos != 0]
    return ret.std()

if WEIGHT_SCHEME == 'inverse_vol':
    vols = {k: compute_vol(k, VOL_METHOD, VOL_WINDOW) for k in show_coins}
    inv_vol = {k: 1.0 / vols[k] if vols[k] > 0 else 0.0 for k in show_coins}
    total   = sum(inv_vol.values())
    w       = {k: inv_vol[k] / total for k in show_coins}
    print(f'Inverse-vol weights  (method={VOL_METHOD}):')
    for k in show_coins:
        print(f'  {k.replace("_","/"):<12}  vol={vols[k]*100:.3f}%  weight={w[k]*100:.1f}%')
else:
    w = {k: 1.0 / len(show_coins) for k in show_coins}
    print(f'Equal weights: {round(100/len(show_coins), 1)}% each')

# ── vol method comparison ─────────────────────────────────────────────────────
if WEIGHT_SCHEME == 'inverse_vol':
    vols_full = {k: compute_vol(k, 'full',      VOL_WINDOW) for k in show_coins}
    vols_mkt  = {k: compute_vol(k, 'in_market', VOL_WINDOW) for k in show_coins}
    iv_full = {k: 1/vols_full[k] for k in show_coins}; t_f = sum(iv_full.values())
    iv_mkt  = {k: 1/vols_mkt[k]  for k in show_coins}; t_m = sum(iv_mkt.values())
    print()
    print(f'  {"Pair":<12}  {"vol(full)":>10}  {"w(full)":>8}  {"vol(mkt)":>10}  {"w(mkt)":>8}')
    print(f'  {"-"*12}  {"-"*10}  {"-"*8}  {"-"*10}  {"-"*8}')
    for k in show_coins:
        mkt_pct = (pair_dfs[k]['position'].fillna(0) != 0).mean() * 100
        print(f'  {k.replace("_","/"):<12}  '
              f'{vols_full[k]*100:>9.3f}%  '
              f'{iv_full[k]/t_f*100:>7.1f}%  '
              f'{vols_mkt[k]*100:>9.3f}%  '
              f'{iv_mkt[k]/t_m*100:>7.1f}%  '
              f'(in-mkt {mkt_pct:.0f}%)')
    print()

# ── combined portfolio returns ────────────────────────────────────────────────
pair_equity = {}
for k in show_coins:
    net_ret = pair_dfs[k]['net_returns'].fillna(0)
    pair_equity[k] = (1 + net_ret).cumprod()

equity_df = pd.concat(pair_equity, axis=1).ffill().fillna(1.0)
ret_df    = equity_df.pct_change().fillna(0)
port_ret  = sum(ret_df[k] * w[k] for k in show_coins)

# ── portfolio backtest (metrics + optional chart) ─────────────────────────────
port_df = pd.DataFrame({
    'strategy_returns': port_ret,
    'position':         1,
}, index=port_ret.index)

m = backtest(port_df, cost=0.0, show_plot=False)

# ── chart ─────────────────────────────────────────────────────────────────────
eq     = m['equity_curve']
dd     = (eq / eq.cummax() - 1) * 100

pair_labels = ' · '.join(k.replace('_', '/') for k in show_coins)
colors      = ['#00bcd4', '#ff9800', '#4caf50', '#e040fb', '#f44336']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor('#1e1e1e')
for ax in (ax1, ax2):
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

# equity curves
for i, k in enumerate(show_coins):
    peq = pair_equity[k].reindex(eq.index).ffill().fillna(1.0)
    ax1.plot(eq.index, peq, color=colors[i % len(colors)],
             linewidth=1.2, alpha=0.7, label=k.replace('_', '/'))
ax1.plot(eq.index, eq, color='white', linewidth=2, label='PORTFOLIO')
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.1f}x'))
ax1.set_ylabel('Equity (1.0 = start)', color='#aaaaaa')
ax1.legend(facecolor='#2d2d2d', edgecolor='#444444', labelcolor='white', fontsize=9)
ax1.set_title(f'Portfolio OOS — Equal Weight ({pair_labels})',
              color='white', fontsize=11, pad=10)
ax1.grid(True, color='#333333', linewidth=0.5)

# drawdown
ax2.fill_between(eq.index, dd, 0, color='#f44336', alpha=0.4)
ax2.plot(eq.index, dd, color='#f44336', linewidth=0.8)
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0f}%'))
ax2.set_ylabel('Drawdown', color='#aaaaaa')
ax2.grid(True, color='#333333', linewidth=0.5)

plt.tight_layout()

# ── portfolio metrics ─────────────────────────────────────────────────────────
print()
print('═' * 52)
print('  PORTFOLIO METRICS  (OOS combined)')
print('═' * 52)
print(f'  Total Return     {m["total_return"]*100:>10.2f}%')
print(f'  Sharpe Ratio     {m["sharpe_ratio"]:>10.2f}')
print(f'  Max Drawdown     {m["max_drawdown"]*100:>10.2f}%')
print(f'  Calmar Ratio     {m["calmar_ratio"]:>10.2f}')
print(f'  Start            {str(m["equity_curve"].index[0].date()):>12}')
print(f'  End              {str(m["equity_curve"].index[-1].date()):>12}')
print('─' * 52)
print('  YEARLY BREAKDOWN')
print('─' * 52)
for yr in sorted(m['yearly_returns']):
    print(f'  {yr}    ret={m["yearly_returns"][yr]*100:>7.1f}%'
          f'  sharpe={m["yearly_sharpe"][yr]:>5.2f}'
          f'  dd={m["yearly_max_drawdown"][yr]*100:>6.1f}%')
print('─' * 52)
print('  PER-PAIR SUMMARY')
print('─' * 52)
for k in show_coins:
    # pass original oos_df so engine detects real position changes + trade stats
    pm = backtest(pair_dfs[k].copy(), cost=0.0, show_plot=False)
    print(f'  {k.replace("_","/"):<12}  '
          f'ret={pm["total_return"]*100:>7.1f}%  '
          f'sharpe={pm["sharpe_ratio"]:>5.2f}  '
          f'dd={pm["max_drawdown"]*100:>6.1f}%  '
          f'pf={pm["profit_factor"]:>4.2f}  '
          f'trades={pm["num_trades"]}')
print('═' * 52)

# ── correlation & overlap analysis ───────────────────────────────────────────
common_idx = pair_dfs[show_coins[0]].index
for k in show_coins[1:]:
    common_idx = common_idx.intersection(pair_dfs[k].index)

print()
print('═' * 52)
print('  PAIR CORRELATION ANALYSIS  (common OOS window)')
print('─' * 52)
print(f'  Common window: {common_idx[0].date()} → {common_idx[-1].date()}')
print(f'  Bars: {len(common_idx)}')
print('─' * 52)

# in-market stats
print('  In-market frequency:')
for k in show_coins:
    in_mkt = (pair_dfs[k]['position'].reindex(common_idx).fillna(0) != 0)
    print(f'    {k.replace("_","/"):<12}  {in_mkt.sum():>4} / {len(common_idx)} days '
          f'({in_mkt.mean()*100:>5.1f}%)')

pos_matrix = pd.DataFrame({
    k: (pair_dfs[k]['position'].reindex(common_idx).fillna(0) != 0)
    for k in show_coins
})
all_in  = pos_matrix.all(axis=1).sum()
any_two = (pos_matrix.sum(axis=1) >= 2).sum()
print(f'  All pairs in market same day:  {all_in:>4} ({all_in/len(common_idx)*100:.1f}%)')
print(f'  Any 2+ in market same day:     {any_two:>4} ({any_two/len(common_idx)*100:.1f}%)')
print('─' * 52)

# return correlations
ret_matrix = pd.DataFrame({
    k: pair_dfs[k]['net_returns'].reindex(common_idx).fillna(0)
    for k in show_coins
})
corr = ret_matrix.corr()
print('  Return correlations:')
keys = show_coins
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a = keys[i].replace('_', '/')
        b = keys[j].replace('_', '/')
        print(f'    {a} vs {b}:  {corr.loc[keys[i], keys[j]]:.3f}')
print('═' * 52)

# ── cost sensitivity sweep ────────────────────────────────────────────────────
COST_SWEEP = [0.0005, 0.001, 0.002, 0.003, 0.005]   # 5, 10, 20, 30, 50 bps

print()
print('═' * 67)
print('  COST SENSITIVITY  (per-leg cost applied to each pair)')
print('─' * 67)
print(f'  {"Cost":>6}   {"Return":>8}   {"Sharpe":>6}   {"MaxDD":>7}   {"Calmar":>6}')
print('─' * 67)

for cost_val in COST_SWEEP:
    # rebuild net_returns for each pair at this cost level
    swept_equity = {}
    for k in show_coins:
        df_k   = pair_dfs[k]
        pos    = df_k['position']
        sr     = df_k['strategy_returns'].fillna(0)
        to     = pos.diff().abs().fillna(0)
        nr     = pos.shift(1).fillna(0) * sr - cost_val * to
        swept_equity[k] = (1 + nr).cumprod()

    eq_df   = pd.concat(swept_equity, axis=1).ffill().fillna(1.0)
    r_df    = eq_df.pct_change().fillna(0)
    p_ret   = sum(r_df[k] * w[k] for k in show_coins)

    pf = pd.DataFrame({'strategy_returns': p_ret, 'position': 1}, index=p_ret.index)
    ms = backtest(pf, cost=0.0, show_plot=False)

    marker = ' <-- baseline' if cost_val == 0.001 else ''
    print(f'  {cost_val*100:>5.2f}%   '
          f'{ms["total_return"]*100:>7.1f}%   '
          f'{ms["sharpe_ratio"]:>6.2f}   '
          f'{ms["max_drawdown"]*100:>6.1f}%   '
          f'{ms["calmar_ratio"]:>6.2f}{marker}')

print('═' * 67)

plt.show() 
