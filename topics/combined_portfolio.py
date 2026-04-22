import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

ROOT     = str(Path(__file__).resolve().parent.parent)
STATARB  = os.path.join(ROOT, 'topics', 'statistical-arbitrage', 'strategies', 'testing')
MOM_ROOT = os.path.join(ROOT, 'topics', 'momentum', 'strategies')

sys.path.append(os.path.join(ROOT, 'infrastructure', 'backtester'))
from engine import backtest

# ── config ────────────────────────────────────────────────────────────────────
STRATEGY_WEIGHTS   = {'statarb': 0.30, 'momentum': 0.70}  # optimal Sharpe split

# stat arb inverse-vol method
STATARB_VOL_METHOD = 'in_market'   # 'full' | 'in_market'
STATARB_VOL_WINDOW = None          # None = full history | int = last N bars

# momentum — mirrors portfolio_master.ipynb SELECTION
MOMENTUM_STRATEGIES = {
    'wf2': 'wf_testing_2',
    'bb':  'bb_breakout_wf',
}
MOMENTUM_SELECTION = {
    'BTC_wf2':  ('wf2', 'BTC'),
    'ETH_wf2':  ('wf2', 'ETH'),
    'SOL_wf2':  ('wf2', 'SOL'),
    'XRP_wf2':  ('wf2', 'XRP'),
    'BTC_bb':   ('bb',  'BTC'),
    'ETH_bb':   ('bb',  'ETH'),
    'AVAX_bb':  ('bb',  'AVAX'),
    'LINK_bb':  ('bb',  'LINK'),
}
MOMENTUM_COST  = 0.001   # 10bps — matches portfolio_master
STATARB_COST   = 0.001   # already baked into pkl net_returns; kept for reference

# ── numpy compat loader (momentum pkls saved with older numpy) ─────────────────
class _Compat(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

def load_pkl(path):
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, 'rb') as f:
            return _Compat(f).load()

# ── load stat arb pkls ────────────────────────────────────────────────────────
sa_dfs = {}
for fname in sorted(os.listdir(STATARB)):
    if fname.endswith('_oos.pkl'):
        pair = fname.replace('_oos.pkl', '').upper()
        sa_dfs[pair] = load_pkl(os.path.join(STATARB, fname))

if not sa_dfs:
    print('No stat arb pkl files found.')
    sys.exit(1)
print(f'Stat arb loaded:  {list(sa_dfs.keys())}')

# ── load momentum pkls ────────────────────────────────────────────────────────
mom_raw = {}
for tag, folder in MOMENTUM_STRATEGIES.items():
    path = os.path.join(MOM_ROOT, folder)
    if not os.path.isdir(path):
        continue
    for fname in os.listdir(path):
        if fname.endswith('_oos.pkl'):
            coin = fname.replace('_oos.pkl', '').upper()
            mom_raw.setdefault(tag, {})[coin] = load_pkl(os.path.join(path, fname))

mom_dfs, missing = {}, []
for label, (tag, coin) in MOMENTUM_SELECTION.items():
    if tag not in mom_raw or coin not in mom_raw[tag]:
        missing.append(label)
    else:
        mom_dfs[label] = mom_raw[tag][coin]

if missing:
    print(f'  [missing momentum sleeves]: {missing}')
print(f'Momentum loaded:  {list(mom_dfs.keys())}')

# ── stat arb weights (inverse vol, in-market returns) ─────────────────────────
sa_pairs = list(sa_dfs.keys())

def _vol(df, method, window):
    ret = df['net_returns'].fillna(0)
    pos = df['position'].fillna(0)
    if window:
        ret, pos = ret.iloc[-window:], pos.iloc[-window:]
    return ret[pos != 0].std() if method == 'in_market' else ret.std()

sa_vols   = {k: _vol(sa_dfs[k], STATARB_VOL_METHOD, STATARB_VOL_WINDOW) for k in sa_pairs}
sa_inv    = {k: 1 / sa_vols[k] if sa_vols[k] > 0 else 0 for k in sa_pairs}
sa_total  = sum(sa_inv.values())
sa_w      = {k: sa_inv[k] / sa_total for k in sa_pairs}

print(f'\nStat arb weights  (method={STATARB_VOL_METHOD}):')
for k in sa_pairs:
    print(f'  {k.replace("_","/"):<12}  vol={sa_vols[k]*100:.3f}%  w={sa_w[k]*100:.1f}%')

# ── momentum weights (equal within strategy, equal across strategies) ──────────
mom_sleeves = list(mom_dfs.keys())
tags        = [MOMENTUM_SELECTION[s][0] for s in mom_sleeves]
tag_counts  = {t: tags.count(t) for t in set(tags)}
n_strats    = len(tag_counts)
mom_w       = {s: 1 / (n_strats * tag_counts[MOMENTUM_SELECTION[s][0]]) for s in mom_sleeves}

print(f'\nMomentum weights  (equal within strategy, equal across strategies):')
for s in mom_sleeves:
    print(f'  {s:<12}  w={mom_w[s]*100:.1f}%')

# ── stat arb portfolio returns ────────────────────────────────────────────────
sa_equity = {k: (1 + sa_dfs[k]['net_returns'].fillna(0)).cumprod() for k in sa_pairs}
sa_eq_df  = pd.concat(sa_equity, axis=1).ffill().fillna(1.0)
sa_ret    = sa_eq_df.pct_change().fillna(0)
sa_port   = sum(sa_ret[k] * sa_w[k] for k in sa_pairs)

# ── momentum portfolio returns ────────────────────────────────────────────────
def _mom_nr(df):
    pos  = df['position'].fillna(0)
    size = df['position_size'].shift(1).fillna(0) if 'position_size' in df.columns else pd.Series(1.0, index=df.index)
    ret  = df['Close'].pct_change().fillna(0)
    to   = pos.diff().abs().fillna(0)
    return ret * pos.shift(1).fillna(0) * size - MOMENTUM_COST * to

mom_equity = {s: (1 + _mom_nr(mom_dfs[s])).cumprod() for s in mom_sleeves}
mom_eq_df  = pd.concat(mom_equity, axis=1).ffill().fillna(1.0)
mom_ret    = mom_eq_df.pct_change().fillna(0)
mom_port   = sum(mom_ret[s] * mom_w[s] for s in mom_sleeves)

# ── combine strategies ────────────────────────────────────────────────────────
sw_total = sum(STRATEGY_WEIGHTS.values())
sw       = {k: v / sw_total for k, v in STRATEGY_WEIGHTS.items()}

all_idx  = sa_port.index.union(mom_port.index)
sa_r     = sa_port.reindex(all_idx).fillna(0)
mo_r     = mom_port.reindex(all_idx).fillna(0)
comb_ret = sa_r * sw['statarb'] + mo_r * sw['momentum']

# ── run backtests ─────────────────────────────────────────────────────────────
def _bt(ret):
    df = pd.DataFrame({'strategy_returns': ret, 'position': 1}, index=ret.index)
    return backtest(df, cost=0.0, show_plot=False)

m_sa   = _bt(sa_r)
m_mom  = _bt(mo_r)
m_comb = _bt(comb_ret)

# ── correlation & weight optimisation ────────────────────────────────────────
corr = sa_r.corr(mo_r)
print()
print('═' * 62)
print('  STRATEGY CORRELATION')
print('─' * 62)
print(f'  Stat Arb vs Momentum:  {corr:.4f}')
print('─' * 62)
print(f'  {"SA%":>5}  {"MOM%":>5}  {"Sharpe":>7}  {"Return":>8}  {"MaxDD":>7}  {"Calmar":>7}')
print(f'  {"-"*5}  {"-"*5}  {"-"*7}  {"-"*8}  {"-"*7}  {"-"*7}')
best_sharpe_split = (sw['statarb'], sw['momentum'])
for sa_pct in range(0, 101, 5):
    mo_pct = 100 - sa_pct
    m = _bt(sa_r * (sa_pct / 100) + mo_r * (mo_pct / 100))
    marker = ' <-- current' if (sa_pct / 100 == sw['statarb']) else ''
    print(f'  {sa_pct:>5}  {mo_pct:>5}  {m["sharpe_ratio"]:>7.2f}  '
          f'{m["total_return"]*100:>7.1f}%  {m["max_drawdown"]*100:>7.1f}%  '
          f'{m["calmar_ratio"]:>7.2f}{marker}')
print('═' * 62)

# ── print metrics ─────────────────────────────────────────────────────────────
print()
print('═' * 62)
print(f'  {"":20}  {"StatArb":>10}  {"Momentum":>10}  {"Combined":>10}')
print('─' * 62)
print(f'  {"Total Return":<20}  {m_sa["total_return"]*100:>9.1f}%  {m_mom["total_return"]*100:>9.1f}%  {m_comb["total_return"]*100:>9.1f}%')
print(f'  {"Sharpe Ratio":<20}  {m_sa["sharpe_ratio"]:>10.2f}  {m_mom["sharpe_ratio"]:>10.2f}  {m_comb["sharpe_ratio"]:>10.2f}')
print(f'  {"Max Drawdown":<20}  {m_sa["max_drawdown"]*100:>9.1f}%  {m_mom["max_drawdown"]*100:>9.1f}%  {m_comb["max_drawdown"]*100:>9.1f}%')
print(f'  {"Calmar Ratio":<20}  {m_sa["calmar_ratio"]:>10.2f}  {m_mom["calmar_ratio"]:>10.2f}  {m_comb["calmar_ratio"]:>10.2f}')
print(f'  {"Start":<20}  {str(m_sa["equity_curve"].index[0].date()):>10}  {str(m_mom["equity_curve"].index[0].date()):>10}  {str(m_comb["equity_curve"].index[0].date()):>10}')
print(f'  {"End":<20}  {str(m_sa["equity_curve"].index[-1].date()):>10}  {str(m_mom["equity_curve"].index[-1].date()):>10}  {str(m_comb["equity_curve"].index[-1].date()):>10}')
print('─' * 62)
print('  YEARLY  (combined)')
print('─' * 62)
for yr in sorted(m_comb['yearly_returns']):
    print(f'  {yr}    ret={m_comb["yearly_returns"][yr]*100:>7.1f}%'
          f'  sharpe={m_comb["yearly_sharpe"][yr]:>5.2f}'
          f'  dd={m_comb["yearly_max_drawdown"][yr]*100:>6.1f}%')
print('═' * 62)

# ── chart ─────────────────────────────────────────────────────────────────────
eq_sa   = m_sa['equity_curve'].reindex(all_idx).ffill().fillna(1.0)
eq_mom  = m_mom['equity_curve'].reindex(all_idx).ffill().fillna(1.0)
eq_comb = m_comb['equity_curve']
dd_comb = (eq_comb / eq_comb.cummax() - 1) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor('#1e1e1e')
for ax in (ax1, ax2):
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

ax1.plot(all_idx, eq_sa,   color='#00bcd4', linewidth=1.4, alpha=0.8, label='Stat Arb')
ax1.plot(all_idx, eq_mom,  color='#ff9800', linewidth=1.4, alpha=0.8, label='Momentum')
ax1.plot(all_idx, eq_comb, color='white',   linewidth=2.2,             label='Combined')
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.1f}x'))
ax1.set_ylabel('Equity (1.0 = start)', color='#aaaaaa')
ax1.set_title(f'Combined Portfolio OOS  —  {sw["statarb"]*100:.0f}% Stat Arb  /  {sw["momentum"]*100:.0f}% Momentum',
              color='white', fontsize=11, pad=10)
ax1.legend(facecolor='#2d2d2d', edgecolor='#444444', labelcolor='white', fontsize=9)
ax1.grid(True, color='#333333', linewidth=0.5)

ax2.fill_between(all_idx, dd_comb, 0, color='#f44336', alpha=0.4)
ax2.plot(all_idx, dd_comb, color='#f44336', linewidth=0.8)
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0f}%'))
ax2.set_ylabel('Drawdown', color='#aaaaaa')
ax2.grid(True, color='#333333', linewidth=0.5)

plt.tight_layout()
plt.show()
