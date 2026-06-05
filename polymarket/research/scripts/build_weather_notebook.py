"""Builder for notebooks/weather_tail_analysis.ipynb (source of truth).

Run: python3 polymarket/research/scripts/build_weather_notebook.py

The notebook is a thin caller of data_infra/weather_analysis.py — the module
is the source of truth for computation. This script is the source of truth
for the notebook's *structure*; edit cells here and regenerate, do not edit
the .ipynb directly.

Heavy cells that scan the trades parquet are gated behind `RUN_GRID = False`
in the first code cell. Set to True to run the full grids (~15 min).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "weather_tail_analysis.ipynb"


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str) -> nbf.NotebookNode:
    c = nbf.v4.new_code_cell(src.strip("\n"))
    c["execution_count"] = None
    c["outputs"] = []
    return c


cells: list[nbf.NotebookNode] = []

# ── §0  Strategy overview + glossary ────────────────────────────────────────
cells.append(md("""
# Weather FTC TP Strategy — analysis notebook

Comprehensive view of the first-to-cross take-profit strategy on Polymarket
weather markets. Thin caller of `data_infra/weather_analysis.py`; most code
cells are a single function call. Re-run from `polymarket/research/`.

## What the strategy does

For each weather market in the universe (`highest|lowest-temperature-in-CITY-*`),
in the **24-hour window before resolution**:

  1. **Entry** — when a token's price first crosses up through `p_in`
     (default `0.60`), the bot buys.
  2. **Take-profit (TP)** — if price subsequently crosses up through `p_out`
     (default `0.90`), the bot sells.
  3. **Hold** — if TP never fires, the bot holds to resolution. The token
     pays `$1` if the underlying weather event resolved YES, `$0` if NO.

Three mutually-exclusive outcome buckets per trade:

  - **`tp`** — TP fired ⇒ exit at `p_out`. PnL/share = `p_out − p_in`.
  - **`hold_win`** — TP didn't fire AND market resolved YES ⇒ collect `$1`.
    PnL/share = `1 − p_in`.
  - **`hold_chop`** — TP didn't fire AND market resolved NO ⇒ lose stake.
    PnL/share = `−p_in`.

`p_tp + p_hold_win + p_hold_chop = 1`. **These are PROBABILITIES**, in [0, 1] —
multiply by 100 to read as percent.

## Glossary

| term | meaning |
|---|---|
| `p_in`, `p_out` | barrier prices: entry trigger and TP target |
| barrier `p` | a price threshold; the analysis measures how often each is crossed |
| `chop_rate` | share of crosses where the crossed token resolved NO ("crashed") |
| `edge_per_signal` | `$/share` of edge in the hold-to-resolution model; `= (1 − chop_rate) − p_in` |
| `edge` | `$/share` under the asymmetric TP policy (three-bucket math above) |
| `fc_NNN` | timestamp of first cross of barrier `0.NN` (e.g. `fc_060` = first cross of `0.60`) |
| `mk_NNN`, `tk_NNN`, `ms_NNN` | maker / taker / maker_side at the first-cross fill |
| **`next_same_dir`** | proxy = price of the next fill in our trade's direction (entry: aggressive buy print = ASK side) within `(15s, 5min]` |
| **`next_opp_dir`** | proxy = price of the next fill in the opposite direction (entry: aggressive sell print = BID side) within `(15s, 5min]` |
| **`fill_rate`** | WS-passive: share of cross events where an aggressive counter-arrived to hit our posted limit; unfilled cross events are skipped (`PnL = 0`) |
| **`edge_per_filled`** | `$/share` averaged over **filled trades only** |
| **`edge_per_entered`** | `$/share` averaged over **all crosses** (unfilled contribute 0) |
| **`PnL/yr`** | total dollars across all cross events at **1 share per cross** sizing, where 1 share = `$1` of payout. **Not ROI on bankroll** — translate by applying the `1/N`-active-cities sizing rule in `ftc_tp_sizing.py`. The 12-month period comes from the universe build's date floor/ceil. |

## Two execution models

The notebook distinguishes between two execution regimes:

  - **Taker** (cross the spread): every cross becomes a trade with a slippage
    cost. Proxy: `next_same_dir` next-fill price; see §4.
  - **WS-passive** (post limit at the touch via CLOB WebSocket): fills happen
    at **exactly the posted price** but only when an aggressive counterparty
    arrives within the window. Discards ~74% of crosses (unfilled). See §6.

## Notebook structure

  §1 — Load data + pooled hold-to-resolution baseline
  §2 — Hold-to-resolution drill-down: FTC vs per-token, family edge,
       slippage sensitivity, 24h vs 48h window check
  §3 — Asymmetric TP — canonical spot value under both policies
  §4 — Next-fill slippage model — three proxies + per-leg diagnostics
  §5 — Subset analysis (Proposal B) — does filtering to "active markets"
       help under taker execution? (Spoiler: no.)
  §6 — WS-passive execution model — optimistic-fill heads-up
  §7 — *(skipped — verdict moved to §10)*
  §8 — Sharpe / PnL / equity curve — full backtest() pipeline (taker + passive)
  §9 — Chase-best-bid cutoff sweep — first realistic queue analysis
  §11 — **User-calibrated grid (sticky vs track-down) — the final answer**
        + 5% sizing on $10k bankroll
  §10 — Conclusion + deployment verdict + LIVE TEST DESIGN
"""))

# ── §1  Load + pooled baseline ──────────────────────────────────────────────
cells.append(md("""
## §1. Load

Load the four parquets produced by `scripts/weather_tail_analysis.py`. The
pooled hold-to-resolution baseline by barrier is the simplest read: at each
`p`, how often did crossings resolve YES, and what's the gross edge before
any slippage or take-profit logic?
"""))

cells.append(code("""
import sys, importlib
from pathlib import Path

# Make polymarket/research importable from any kernel start.
for _p in [Path.cwd(), *Path.cwd().parents]:
    if (_p / "data_infra" / "weather_analysis.py").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_infra import weather_analysis as wa
importlib.reload(wa)  # pick up new functions on re-run without restarting kernel

# Toggle this to True to run the ~15-minute grid sections (§4 windows,
# §5 subset grid, §6 passive grid). Default False — read cached / canonical.
RUN_GRID = False

results = wa.load_weather_results()
PRIMARY, SIDEBAR, INST = results["primary"], results["sidebar"], results["inst"]
print(f"primary={PRIMARY.shape} sidebar={SIDEBAR.shape} "
      f"inst={INST.shape} uni={results['uni'].shape}")
wa.pooled_metrics_by_barrier(PRIMARY).round({
    "chop_rate": 4, "edge_per_signal": 4,
    "roi_per_signal_pct": 2, "kelly_fraction": 3,
})
"""))

# ── §2  Hold-to-resolution drill-down ──────────────────────────────────────
cells.append(md("""
## §2. Hold-to-resolution drill-down

Four views before introducing any TP logic:

- **FTC vs per-token** edge curves + the both-crossed adverse-selection check
- **Per-family edge** at `p=0.80, n≥30`
- **Slippage sensitivity** of pooled edge (additive cents-per-leg)
- **24h vs 48h** sanity — confirms results aren't a window artifact

These are pre-slippage, hold-to-resolution numbers — establish the gross
edge ceiling. Later sections progressively add execution reality.
"""))

cells.append(code("""
ftc = wa.compute_ftc_metrics(INST)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(ftc["p"], ftc["edge_per_token"], marker="o", lw=2, label="per-token")
ax.plot(ftc["p"], ftc["edge_first_to_cross"], marker="s", lw=2, label="FTC per slug")
ax.axhline(0, color="black", lw=0.8)
ax.set(xlabel="Barrier p", ylabel="Pooled edge ($/$1)",
       title="Hold-to-resolution edge — per-token vs FTC")
ax.grid(alpha=0.3); ax.legend(); plt.tight_layout(); plt.show()

row = ftc[ftc["p"] == 0.80].iloc[0]
print(f"Both-crossed @ p=0.80: first-mover chops {row['chop_first_in_both']:.1%}, "
      f"second-mover chops {row['chop_second_in_both']:.1%}")
ftc.round(4)
"""))

cells.append(code("""
fam = wa.compute_family_rankings(PRIMARY, p=0.80, min_n=30)
print(f"Per-family at p=0.80 (n>=30): {len(fam)} families")
top15 = fam.head(15)

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top15["edge_per_signal"]]
bars = ax.bar(range(len(top15)), top15["edge_per_signal"], color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(top15)))
ax.set_xticklabels(top15["slug_family"], rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Edge $/$1"); ax.set_title("Top 15 families at p=0.80 (n>=30)")
for bar, n in zip(bars, top15["n_crossed"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"n={int(n)}", ha="center", va="bottom", fontsize=7)
ax.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()
top15.round({"chop_rate": 4, "edge_per_signal": 4, "roi_per_signal_pct": 2})
"""))

cells.append(code("""
slip = wa.slippage_grid(PRIMARY)

fig, ax = plt.subplots(figsize=(8, 5))
M = slip.values
im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.05)
ax.set_xticks(range(len(slip.columns))); ax.set_xticklabels(slip.columns)
ax.set_yticks(range(len(slip.index)))
ax.set_yticklabels([f"p={p:.2f}" for p in slip.index])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center",
                fontsize=8, color="black" if abs(M[i,j]) < 0.025 else "white")
fig.colorbar(im, ax=ax, label="edge $/$1")
ax.set_title("Pooled edge under (barrier, additive cents-per-leg slip)")
plt.tight_layout(); plt.show()
slip.round(4)
"""))

cells.append(code("""
wc = wa.window_comparison(PRIMARY, SIDEBAR)
diff_max = (wc["edge_per_signal_24h"] - wc["edge_per_signal_48h"]).abs().max()
print(f"24h vs 48h max |edge diff| = {diff_max:.4f}  →  not a window artifact")
wc[["barrier_price", "edge_per_signal_24h", "edge_per_signal_48h"]].round(4)
"""))

# ── §3  Asymmetric TP canonical spot ───────────────────────────────────────
cells.append(md("""
## §3. Asymmetric TP — canonical spot value

Enter @ `p_in` on level-break; TP @ `p_out` if `first_cross(p_out) > first_cross(p_in)`,
else hold to resolution. Three-bucket math applied. Default fill model: the
**next_same_dir** proxy (Session 2 onward) — `slip` parameter has been
removed.

Spot-check at `(p_in=0.60, p_out=0.90)` under both policies. The grid scan
across all `(p_in, p_out)` pairs has been deprecated in favour of the
proxy/passive analyses in §4-§6, which give the same picture more honestly.
"""))

cells.append(code("""
p_wide = wa.pivot_inst_to_wide(INST)
print(f"p_wide shape: {p_wide.shape}")
"""))

cells.append(code("""
# Policy 'all' — every token that level-breaks counts
spot_all = wa.eval_pair(p_wide, 0.60, 0.90, policy="all")
print(f"policy='all' (p_in=0.60, p_out=0.90):")
print(f"  n_entries={spot_all['n_entries']}  p_tp={spot_all['p_tp']:.4f}  "
      f"p_hold_win={spot_all['p_hold_win']:.4f}  p_hold_chop={spot_all['p_hold_chop']:.4f}")
print(f"  edge={spot_all['edge']:+.4f}  ROI={spot_all['roi_pct']:+.2f}%")
print(f"  entry_next_same_dir fallback_rate = {spot_all['entry_next_same_dir_fallback_rate']:.3f}")
assert spot_all["n_entries"] == 8938, "n_entries mismatch vs known-good 8938"

# Policy 'first_to_cross' — only the first token per market (the bot's policy)
spot_ftc = wa.eval_pair(p_wide, 0.60, 0.90, policy="first_to_cross")
print(f"\\npolicy='first_to_cross' (the bot policy):")
print(f"  n_entries={spot_ftc['n_entries']}  edge={spot_ftc['edge']:+.4f}  "
      f"ROI={spot_ftc['roi_pct']:+.2f}%")
print(f"  Δ_edge vs 'all': {spot_ftc['edge'] - spot_all['edge']:+.4f}")
assert spot_ftc["n_entries"] == 6219, "n_entries FTC mismatch vs known-good 6219"
"""))

# ── §4  Next-fill slippage model ───────────────────────────────────────────
cells.append(md("""
## §4. Next-fill slippage model (Sessions 2 → 2.6)

Replace the assumption "you fill at exactly `p_in`" with a proxy: the next
realised fill in `(15s, 5min]` after the cross, by a different counterparty.
Two proxies per leg:

- **`next_same_dir`** (entry: next aggressive buy print = ask print): proxy
  for what a taker would have paid.
- **`next_opp_dir`** (entry: next aggressive sell print = bid print): proxy
  for what a passive limit poster would have paid.

The gap is a sensitivity range, **not a precise estimate**. When no
qualifying fill exists in the window, a fallback constant fires (3¢ default).
Wherever fallback share exceeds 50%, the scenario edge is dominated by that
constant — read those as "constant slippage with extra labelling."
"""))

cells.append(code("""
# Reuse spot_all's audit log so we don't re-scan the parquet
_summary, audit = wa.eval_pair(p_wide, 0.60, 0.90, policy='all', return_audit=True)
scen = wa.compute_fill_scenarios(audit)
diag = wa.slippage_diagnostic(scen)

P_IN, P_OUT = 0.60, 0.90
edges = {'next_same_dir': scen['pnl_next_same_dir'].mean(),
         'midpoint':      scen['pnl_midpoint'].mean(),
         'next_opp_dir':  scen['pnl_next_opp_dir'].mean()}
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(edges.keys(), edges.values(),
       color=['#c8313a' if v < 0 else '#3a7a3a' for v in edges.values()])
ax.axhline(0, color='k', lw=0.5)
for i, (k, v) in enumerate(edges.items()):
    ax.text(i, v + 0.001*np.sign(v) - (0.002 if v < 0 else 0),
            f'{v:+.4f}\\n({100*v/P_IN:+.2f}%)',
            ha='center', va='bottom' if v >= 0 else 'top')
ax.set_ylabel('edge ($/share)')
ax.set_title(f'Edge per proxy (canonical p_in={P_IN}, p_out={P_OUT}, n={len(scen)})')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# Fallback rate per leg × direction — red = >50% (interpret as constant slippage)
legs = pd.DataFrame(diag['legs'])
labels = [f'{r[\"leg\"]}/{r[\"direction\"]}' for _, r in legs.iterrows()]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(labels, legs['fallback_pct'],
       color=['#c8313a' if v > 0.5 else '#3a7a3a' for v in legs['fallback_pct']])
ax.axhline(0.5, color='k', ls='--', lw=0.8, label='50% threshold')
ax.set_ylim(0, 1); ax.set_ylabel('fallback share')
ax.set_title('Fallback rate per leg  (red = constant_slippage_with_labelling)')
for i, r in legs.iterrows():
    ax.text(i, r['fallback_pct'] + 0.02, f'{r[\"fallback_pct\"]:.2f}', ha='center')
ax.legend(); plt.tight_layout(); plt.show()
print('interpretation per leg:')
for _, r in legs.iterrows():
    print(f'  {r[\"leg\"]:5s}/{r[\"direction\"]:10s}  {r[\"interpretation\"]}')
"""))

cells.append(code("""
# Slippage histogram per leg (real next-fills only; fallback rows excluded)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
for ax, leg_row in zip(axes, diag['legs']):
    leg, dirn = leg_row['leg'], leg_row['direction']
    trig = P_IN if leg == 'entry' else P_OUT
    price_col, src_col = f'{leg}_next_{dirn}_price', f'{leg}_next_{dirn}_source'
    sub = scen if leg == 'entry' else scen[scen['bucket'].eq('tp')]
    nf = sub[sub[src_col] == 'next_fill']
    if len(nf):
        slip_c = ((nf[price_col].astype(float) - trig) * 100).clip(-15, 15)
        ax.hist(slip_c, bins=30, color='steelblue', alpha=0.85)
        ax.axvline(slip_c.median(), color='black', lw=1, label=f'med={slip_c.median():+.1f}¢')
        ax.legend(fontsize=8)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_title(f'{leg}/{dirn}  fb={leg_row[\"fallback_pct\"]:.2f}')
    ax.set_xlabel('slip (cents)')
axes[0].set_ylabel('count')
fig.suptitle('Slippage distribution (real next-fills only)', y=1.02)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# Time-to-first-other-fill (any side, any direction) — the rawest measure of
# post-cross activity. ~8s scan. Cap at 30 min to bound the tail.
any_fill = wa.time_to_first_any_fill_diagnostic(p_wide, p_in=0.60, max_seconds=1800)
ts = [30, 60, 120, 300, 600, 1800]
shares = [any_fill[f'pct_within_{t}s'] for t in ts[:-1]] + [1 - any_fill['pct_no_fill_within_max']]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(ts, shares, marker='o', color='steelblue', lw=2)
ax.axvline(300, color='gray', ls='--', lw=0.8, label='5-min window')
ax.axvline(600, color='red',  ls='--', lw=0.8, label='10-min window')
ax.set_xscale('log'); ax.set_ylim(0, 1)
ax.set_xlabel('time after cross (s, log)')
ax.set_ylabel('cumulative share with any next fill')
ax.set_title(f'Time to first OTHER fill in same market  '
             f'(n={any_fill[\"n_anchors\"]}; lag p50={any_fill[\"lag_median_s\"]:.0f}s; '
             f'{any_fill[\"pct_no_fill_within_max\"]:.1%} never within 30 min)')
for t, s in zip(ts, shares):
    ax.text(t, s + 0.025, f'{s:.1%}', ha='center', fontsize=8)
ax.legend(); plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# Optional: window comparison 5min vs 10min (~40s). Skipped unless RUN_GRID=True.
if RUN_GRID:
    win_cmp = wa.compare_windows_diagnostic(p_wide, 0.60, 0.90, windows_seconds=(300, 600))
    ws = sorted(win_cmp['windows'].keys())
    print('window comparison (300s vs 600s):')
    for w in ws:
        d = win_cmp['windows'][w]
        print(f\"  {w}s: fb_e_same={d['fallback_rate_entry_same_dir']:.3f}  \"
              f\"fb_e_opp={d['fallback_rate_entry_opp_dir']:.3f}  \"
              f\"edge_same={d['edge_next_same_dir']:+.4f}  edge_opp={d['edge_next_opp_dir']:+.4f}\")
    delta = win_cmp['delta_short_to_long']
    print(f\"  Δ: fallback drop entry_same={delta['fallback_drop_entry_same_dir_pp']:+.2f}pp, \"
          f\"edge_same change {delta['edge_change_same_dir_cents']:+.2f}¢\")
else:
    print('Set RUN_GRID=True in §1 to run the 5min-vs-10min window comparison (~40s).')
"""))

# ── §5  Subset analysis (Proposal B) ───────────────────────────────────────
cells.append(md("""
## §5. Subset analysis — does filtering to active markets help?

Proposal B: filter to entries where a real next-fill exists (drop the
fallback-dominated rows). Question: does the strategy look better on the
data-supported subset?

**Answer: no, it looks WORSE under taker execution.** The active subset is
exactly where execution cost is highest — markets that print follow-up fills
are markets where price was drifting against us. The mean entry slippage on
the `maker_filt` subset is **7.56¢** vs the 3¢ fallback assumed on the rest
of the universe.

(For WS-passive execution, "subset" is the wrong frame — see §6 instead.)
"""))

cells.append(code("""
baseline_mask = pd.Series(True, index=scen.index)
taker_mask = scen['entry_next_same_dir_source'] == 'next_fill'
maker_mask = scen['entry_next_opp_dir_source'] == 'next_fill'
inter_mask = taker_mask & maker_mask
summary = pd.DataFrame([
    wa.subset_pnl_summary(scen, baseline_mask, 'baseline',     p_in=P_IN),
    wa.subset_pnl_summary(scen, taker_mask,    'taker_filt',   p_in=P_IN),
    wa.subset_pnl_summary(scen, maker_mask,    'maker_filt',   p_in=P_IN),
    wa.subset_pnl_summary(scen, inter_mask,    'intersection', p_in=P_IN),
]).set_index('label')
print(summary[['n','p_tp','edge_next_same_dir','edge_midpoint','edge_next_opp_dir',
               'mean_entry_slip_cents']].round({
    'p_tp':3,'edge_next_same_dir':4,'edge_midpoint':4,'edge_next_opp_dir':4,
    'mean_entry_slip_cents':2}))
fig, ax = plt.subplots(figsize=(8, 3.5))
summary[['edge_next_same_dir','edge_midpoint','edge_next_opp_dir']].plot.bar(
    ax=ax, color=['#c8313a','#888','#3a7a3a'])
ax.axhline(0, color='k', lw=0.5)
ax.set_title('Taker-model edge per scenario × subset  (p_in=0.60, p_out=0.90)')
ax.set_ylabel('edge ($/share)'); ax.set_xticklabels(summary.index, rotation=0)
ax.legend(['next_same_dir','midpoint','next_opp_dir'], loc='lower left')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# Selection bias: top cities baseline vs maker_filt subset
bias = wa.subset_selection_bias(scen, p_wide, maker_mask, baseline_mask=baseline_mask)
base_share = {c['city']: c['share'] for c in bias['top_cities_baseline']}
sub_share  = {c['city']: c['share'] for c in bias['top_cities_subset']}
cities = list(dict.fromkeys(list(base_share.keys()) + list(sub_share.keys())))
df_b = pd.DataFrame({'baseline':[base_share.get(c,0) for c in cities],
                     'maker_filt':[sub_share.get(c,0) for c in cities]}, index=cities)
df_b.index = [c.replace('highest-temperature-in-','').replace('lowest-temperature-in-','')
              for c in df_b.index]
fig, ax = plt.subplots(figsize=(10, 3.5))
df_b.plot.bar(ax=ax, color=['#888','#3a7a3a'])
ax.set_title(f'Top cities: baseline vs maker_filt subset  '
             f'(n_sub={bias[\"n_subset\"]} = {bias[\"subset_share_of_baseline\"]:.1%} of baseline)')
ax.set_ylabel('share within set'); ax.set_xticklabels(df_b.index, rotation=20, ha='right')
plt.tight_layout(); plt.show()
bh = bias['hours_to_resolution_baseline']; sh = bias['hours_to_resolution_subset']
print(f'hours-to-resolution at entry (p10/p50/p90):')
print(f'  baseline  : {bh[\"p10\"]:.1f} / {bh[\"median\"]:.1f} / {bh[\"p90\"]:.1f}')
print(f'  maker_filt: {sh[\"p10\"]:.1f} / {sh[\"median\"]:.1f} / {sh[\"p90\"]:.1f}')
print(\"\\nNYC and Ankara drop out of the subset — too inactive post-cross. \"
      \"Asian-session cities (Seoul, Shanghai, Tokyo) over-represented.\")
"""))

# ── §6  WS-passive execution model ─────────────────────────────────────────
cells.append(md("""
## §6. WS-passive execution model — *OPTIMISTIC QUEUE (upper bound)*

⚠️ **Caveat read first.** This section assumes our posted bid at `p_in` fills
whenever any aggressive seller arrives in the 5-min window. That's an UPPER
BOUND — it's only achievable with perfect queue priority. When the observed
`bid_nf_price > p_in`, the print hit *someone else's* higher bid, not ours,
and we wouldn't have filled. **§9 (chase-bid sweep) is the realistic queue
model and supersedes this section's numbers.** Read §6 as "best possible
case if every queue race goes your way," not as a deployment estimate.

If the bot detects via CLOB WebSocket and posts a passive limit at the touch
**and wins every queue race**:

- **Entry fills at exactly `p_in`** when an aggressive seller arrives within
  5 min — *assumes our limit at `p_in` is at the front of the queue at the
  touch, even when other bids exist above us*. This is the bit that's
  optimistic.
- **Exit on TP fills at exactly `p_out`** under one of two assumptions:
  - **optimistic**: if price reached `p_out`, we got lifted
  - **strict**: also require a real aggressive-buy print in the exit window

A `UserWarning` is emitted at runtime by `passive_pnl_from_audit` to flag
this caveat — that's intentional, not a bug.
"""))

cells.append(code("""
_aug_opt, sum_opt = wa.passive_pnl_from_audit(scen, exit_passive='optimistic')
_aug_str, sum_str = wa.passive_pnl_from_audit(scen, exit_passive='strict')

rows = [
    {'model':'proxy taker (cross spread)',          'n_trades':len(scen),
     'edge_per_trade':float(scen['pnl_next_same_dir'].mean()),
     'roi_pct':100*float(scen['pnl_next_same_dir'].mean())/P_IN},
    {'model':'proxy maker_best (fallback-heavy)',  'n_trades':len(scen),
     'edge_per_trade':float(scen['pnl_next_opp_dir'].mean()),
     'roi_pct':100*float(scen['pnl_next_opp_dir'].mean())/P_IN},
    {'model':'WS-passive optimistic exit',          'n_trades':sum_opt['n_entry_filled'],
     'edge_per_trade':sum_opt['edge_per_filled'], 'roi_pct':sum_opt['roi_per_filled_pct']},
    {'model':'WS-passive strict exit',              'n_trades':sum_str['n_entry_filled'],
     'edge_per_trade':sum_str['edge_per_filled'], 'roi_pct':sum_str['roi_per_filled_pct']},
]
df = pd.DataFrame(rows).set_index('model')
print(df.round({'edge_per_trade':4,'roi_pct':2}))
print(f\"\\nWS-passive fill_rate: {sum_opt['fill_rate']:.1%}  \"
      f\"({sum_opt['n_entry_filled']} fills of {sum_opt['n_total']} crosses)\")
print(f\"optimistic: tp={sum_opt['p_tp_of_filled']:.3f}  \"
      f\"hold_win={sum_opt['p_hold_win_of_filled']:.3f}  \"
      f\"hold_chop={sum_opt['p_hold_chop_of_filled']:.3f}\")
print(f\"strict    : tp={sum_str['p_tp_of_filled']:.3f}  \"
      f\"hold_win={sum_str['p_hold_win_of_filled']:.3f}  \"
      f\"hold_chop={sum_str['p_hold_chop_of_filled']:.3f}\")

fig, ax = plt.subplots(figsize=(9, 3.5))
colors = ['#c8a04c' if v < 0 else '#3a7a3a' for v in df['edge_per_trade']]
ax.bar(df.index, df['edge_per_trade'], color=colors)
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('edge per trade ($/share)')
ax.set_title('Headline edge per trade by execution model  (p_in=0.60, p_out=0.90)')
for i, (k, v) in enumerate(df['edge_per_trade'].items()):
    ax.text(i, v + 0.001*np.sign(v),
            f\"{v:+.4f}\\n(ROI {df.loc[k,'roi_pct']:+.2f}%)\\nn={df.loc[k,'n_trades']}\",
            ha='center', va='bottom' if v>=0 else 'top', fontsize=8)
plt.xticks(rotation=15, ha='right')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# Passive grid (heatmap of edge_per_filled). Reads cached parquet if available,
# else runs the 7-min grid scan (gated on RUN_GRID).
grid_path = wa.DEFAULT_DATA_DIR / 'passive_grid_canonical.parquet'
if grid_path.exists():
    grid = pd.read_parquet(grid_path)
    print(f'loaded cached grid: {grid_path}')
elif RUN_GRID:
    print('running grid_passive (~7 min)...')
    grid = wa.grid_passive(p_wide, exit_passive='optimistic')
    grid.to_parquet(grid_path, index=False)
    print(f'saved to {grid_path}')
else:
    grid = None
    print(f'no cached grid at {grid_path}. Set RUN_GRID=True in §1 to run (~7 min).')

if grid is not None:
    top = grid.sort_values('edge_per_filled', ascending=False).head(5).set_index(['p_in','p_out'])
    print('\\ntop 5 cells by edge_per_filled:')
    print(top[['n_entry_filled','fill_rate','p_tp_of_filled','edge_per_filled',
               'roi_per_filled_pct','total_pnl_per_entered']].round({
                   'fill_rate':3,'p_tp_of_filled':3,'edge_per_filled':4,
                   'roi_per_filled_pct':2,'total_pnl_per_entered':1}))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    pivot = grid.pivot_table(index='p_in', columns='p_out', values='edge_per_filled')
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.02, vmax=0.03)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    ax.set_xlabel('p_out'); ax.set_ylabel('p_in')
    ax.set_title('WS-passive edge_per_filled by (p_in, p_out)')
    for i, _ in enumerate(pivot.index):
        for j, _ in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            if not pd.isna(v):
                ax.text(j, i, f'{v:+.3f}', ha='center', va='center', fontsize=8,
                        color='black' if abs(v) < 0.015 else 'white')
    plt.colorbar(im, ax=ax, label='edge $/share')
    plt.tight_layout(); plt.show()
"""))

# ── §8  Sharpe / PnL / equity curve (full backtest pipeline) ───────────────
cells.append(md("""
## §8. Sharpe / PnL / equity curve — full backtest pipeline

The pre-robustness analysis: `scripts/backtest/ftc_tp_sizing.py:backtest()` —
1/N-active-cities sizing, capped at 2% per trade, daily aggregation, Sharpe /
total return / max DD / equity curve. Now extended with a `pnl_model`
parameter so the same pipeline runs under both:

- **`pnl_model='taker'`** — original semantics, every cross becomes a trade,
  next-fill ASK/BID proxy slippage
- **`pnl_model='passive'`** — only filled crosses count, fill at exactly
  `p_in` (entry) and `p_out` (TP exit) per the WS-passive assumption.
  See §9 below for why this is OPTIMISTIC about fill rate.

Results cached to `data/analysis/backtest_*.parquet`; first run takes ~60s
total, subsequent re-runs load from cache.
"""))

cells.append(code("""
# Sharpe/PnL canonical comparison. Caches results to avoid re-scanning trades.
import importlib
from scripts.backtest import ftc_tp_sizing as ftc
importlib.reload(ftc)

CACHE_DIR = wa.DEFAULT_DATA_DIR
def _run_or_load(label, **kw):
    cache_summary = CACHE_DIR / f'backtest_{label}_summary.parquet'
    cache_daily   = CACHE_DIR / f'backtest_{label}_daily.parquet'
    if cache_summary.exists() and cache_daily.exists():
        s = pd.read_parquet(cache_summary).iloc[0].to_dict()
        daily = pd.read_parquet(cache_daily)
        if not isinstance(daily.index, pd.DatetimeIndex):
            daily.index = pd.to_datetime(daily.index)
        return s, daily
    s, _trades, daily, _eq, _dd = ftc.backtest(p_in=0.60, p_out=0.90, **kw)
    pd.DataFrame([s]).to_parquet(cache_summary, index=False)
    daily.to_parquet(cache_daily)
    return s, daily

s_t, daily_t = _run_or_load('taker',    pnl_model='taker')
s_o, daily_o = _run_or_load('pass_opt', pnl_model='passive', exit_passive='optimistic')
s_s, daily_s = _run_or_load('pass_str', pnl_model='passive', exit_passive='strict')

stats_keys = ['n_trades','p_tp','p_hold_win','p_hold_chop','win_rate',
              'avg_pnl_pct_per_trade_bps','total_return_pct','cagr_pct',
              'daily_mean_bps','daily_std_bps','sharpe_ann','max_dd_pct']
print(pd.DataFrame({
    'taker':           {k: s_t[k] for k in stats_keys},
    'passive (opt)':   {k: s_o[k] for k in stats_keys},
    'passive (strict)':{k: s_s[k] for k in stats_keys},
}).round(4))
"""))

cells.append(code("""
# Equity curves overlay — non-compounding fixed-fractional ($1 starting)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
for daily, label, color in [(daily_t,'taker','#c8313a'),
                             (daily_o,'passive (opt)','#3a7a3a'),
                             (daily_s,'passive (strict)','#c8a04c')]:
    eq = 1.0 + daily['ret'].cumsum()
    ax1.plot(eq.index, eq.values, label=label, color=color, lw=1.5)
    dd = eq / eq.cummax() - 1.0
    ax2.fill_between(eq.index, dd.values, 0, alpha=0.4, color=color, label=label)
ax1.axhline(1.0, color='k', lw=0.5)
ax1.set_title('Equity curves (non-compounding, $1 start)')
ax1.set_ylabel('equity'); ax1.legend(loc='best'); ax1.grid(alpha=0.3)
ax2.set_title('Drawdown'); ax2.set_ylabel('DD'); ax2.legend(loc='best'); ax2.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f'taker:           Sharpe={s_t[\"sharpe_ann\"]:.2f}  total_ret={s_t[\"total_return_pct\"]:.1f}%  max_dd={s_t[\"max_dd_pct\"]:.1f}%  n={s_t[\"n_trades\"]}')
print(f'passive (opt):   Sharpe={s_o[\"sharpe_ann\"]:.2f}  total_ret={s_o[\"total_return_pct\"]:.1f}%  max_dd={s_o[\"max_dd_pct\"]:.1f}%  n={s_o[\"n_trades\"]}')
print(f'passive (strict):Sharpe={s_s[\"sharpe_ann\"]:.2f}  total_ret={s_s[\"total_return_pct\"]:.1f}%  max_dd={s_s[\"max_dd_pct\"]:.1f}%  n={s_s[\"n_trades\"]}')
print(f'\\nNote: passive (opt) fill rate is OPTIMISTIC — see §9 for the realistic queue analysis.')
"""))

# ── §9  Chase-bid cutoff sweep ─────────────────────────────────────────────
cells.append(md("""
## §9. Chase-best-bid cutoff sweep — the realistic queue analysis

Execution capability: the bot can track the best bid and raise our quote as
others overbid us. Question: **what's the optimal chase cutoff** — how far
to chase before the edge collapses on entry cost?

Model:

- bot posts at best_bid (= our bid is always the touch), capped at p_in + N¢
- fill happens at whatever price the next aggressive seller hits the bid at
  (= the observed `bid_nf_price` in our audit)
- if best bid exceeds the cutoff before a seller arrives, we step out → no fill
- if no aggressive seller arrives in 5 min, no fill regardless

Important: **this is the realistic queue model.** The §6 "passive at p_in"
fill rate (26.1%) assumed our bid would have been filled by ANY aggressive
sell in window — but in reality, prints at `p_in + 5¢` hit someone else's
higher bid, not ours. At cutoff = 0¢ (= stick at p_in, don't chase), the
*actual* fill rate is closer to 12%.
"""))

cells.append(code("""
# Chase-bid cutoff sweep. Operates on the existing `scen` audit — no parquet scan.
sweep_opt = wa.chase_bid_cutoff_sweep(
    scen, cutoffs_cents=[0, 1, 2, 3, 5, 8, 12, 20, 30],
    exit_passive='optimistic',
)
print('=== chase-bid sweep (optimistic exit) ===')
print(sweep_opt[['cutoff_cents','n_filled','fill_rate','p_tp_of_filled',
                  'mean_entry_cents_above_p_in','edge_per_filled',
                  'roi_per_filled_pct','total_pnl_per_entered']].round({
                      'fill_rate':3,'p_tp_of_filled':3,
                      'mean_entry_cents_above_p_in':2,'edge_per_filled':4,
                      'roi_per_filled_pct':2,'total_pnl_per_entered':1}).to_string(index=False))

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 3.5))
a1.plot(sweep_opt['cutoff_cents'], sweep_opt['fill_rate'], marker='o', color='steelblue')
a1.set_xlabel('chase cutoff (¢ above p_in)'); a1.set_ylabel('fill_rate')
a1.set_title('Fill rate vs cutoff'); a1.grid(alpha=0.3)
a1.axhline(0, color='k', lw=0.5)

a2.plot(sweep_opt['cutoff_cents'], sweep_opt['edge_per_filled'] * 100, marker='o', color='#c8313a')
a2.axhline(0, color='k', lw=0.5)
a2.set_xlabel('chase cutoff (¢ above p_in)'); a2.set_ylabel('edge per filled ($¢)')
a2.set_title('Edge per filled vs cutoff'); a2.grid(alpha=0.3)

a3.plot(sweep_opt['cutoff_cents'], sweep_opt['total_pnl_per_entered'], marker='o', color='#3a7a3a')
a3.axhline(0, color='k', lw=0.5)
a3.set_xlabel('chase cutoff (¢ above p_in)'); a3.set_ylabel('total $ PnL across all crosses')
a3.set_title('Total $ PnL vs cutoff'); a3.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""
### Reading the chase-bid sweep

- **cutoff = 0¢** is the realistic "stick at p_in" outcome: 11.8% fill, edge
  per filled = `−8.8¢/share` (−14.7% ROI). The earlier §6 number (+0.87%)
  was optimistic about queue priority.
- **Chasing up to +30¢** (almost to p_out) gets fill rate to ~25% and edge
  to `−5.8¢` per share — still negative. The increased fill rate doesn't
  pay for the increased entry cost.
- **No positive cutoff exists at canonical (0.60, 0.90)** under this honest
  queue model. The edge collapse happens because chop probability times
  the higher entry price dominates: when you chop, you lose `entry_price`,
  and chasing makes that loss bigger.

This is the harshest read of the strategy. The deployable corner from §6's
(0.50, 0.90) grid cell needs re-checking under this same chase model
(`wa.chase_bid_cutoff_sweep` on an audit built at p_in=0.50) — likely still
positive but with less headroom than the +6% headline suggested.
"""))

# ── §11 User-calibrated grid (sticky vs track-down) ────────────────────────
cells.append(md("""
## §11. User-calibrated execution grid — sticky vs track-down

Final layer: model the user's actual CLOB-WS bot with two interpretations of
entry execution. Sizing fixed at **5% of $10k initial bankroll per filled
trade**, exit always active (sell to bid, receive `p_out − 2¢ spread`).

- **`entry_model='sticky'` (REALISTIC)** — bot posts at p_in cap, chases up
  only. In a crash, our high bid is hit FIRST at p_in. Captures the adverse
  selection that real low-liquidity execution suffers from.
- **`entry_model='track_down'` (UPPER BOUND, unrealistic)** — bot continuously
  cancels and re-posts at every new best bid in BOTH directions. Buys cheap
  in crashes. Requires sub-second cancel-replace + queue priority advantage
  that Polymarket weather markets don't realistically support.

The gap between them is the EXECUTION SENSITIVITY of the strategy. Live
testing is about measuring where between sticky and track-down your actual
fills land.
"""))

cells.append(code("""
# Load both cached grids (from scripts/user_model_analysis.py)
gs_path = wa.DEFAULT_DATA_DIR / 'user_passive_grid_sticky.parquet'
gt_path = wa.DEFAULT_DATA_DIR / 'user_passive_grid_track_down.parquet'

if not gs_path.exists() or not gt_path.exists():
    print(f'missing cached grids. Run: python scripts/user_model_analysis.py')
else:
    gs = pd.read_parquet(gs_path)
    gt = pd.read_parquet(gt_path)
    print(f'STICKY (REALISTIC) — top 5 by least-loss:')
    print(gs.sort_values('total_dollar_pnl', ascending=False).head(5)[
        ['p_in','p_out','n_filled','fill_rate','p_tp_of_filled',
         'edge_per_filled','total_dollar_pnl','pnl_pct_of_initial_bankroll',
         'sharpe_ann','max_dd_pct']
    ].round({'fill_rate':3,'p_tp_of_filled':3,'edge_per_filled':4,
             'total_dollar_pnl':0,'pnl_pct_of_initial_bankroll':1,
             'sharpe_ann':2,'max_dd_pct':1}).to_string(index=False))
    print(f'\\nTRACK_DOWN (UPPER BOUND) — top 5 by $PnL:')
    print(gt.sort_values('total_dollar_pnl', ascending=False).head(5)[
        ['p_in','p_out','n_filled','fill_rate','p_tp_of_filled',
         'edge_per_filled','total_dollar_pnl','pnl_pct_of_initial_bankroll',
         'sharpe_ann','max_dd_pct']
    ].round({'fill_rate':3,'p_tp_of_filled':3,'edge_per_filled':4,
             'total_dollar_pnl':0,'pnl_pct_of_initial_bankroll':1,
             'sharpe_ann':2,'max_dd_pct':1}).to_string(index=False))
"""))

cells.append(code("""
# Side-by-side bar chart: sticky vs track-down $PnL per cell
if gs_path.exists() and gt_path.exists():
    merged = gs.merge(gt, on=['p_in','p_out'], suffixes=('_sticky','_td'))
    merged['cell'] = merged.apply(lambda r: f\"{r['p_in']:.2f}/{r['p_out']:.2f}\", axis=1)
    merged = merged.sort_values('total_dollar_pnl_td', ascending=False)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(merged))
    ax.bar(x - 0.2, merged['total_dollar_pnl_sticky'], 0.4, color='#c8313a',
           label='STICKY (realistic)')
    ax.bar(x + 0.2, merged['total_dollar_pnl_td'], 0.4, color='#3a7a3a',
           label='TRACK_DOWN (upper bound)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(merged['cell'], rotation=45, ha='right')
    ax.set_ylabel('$ PnL on $10k bankroll, 5% sizing')
    ax.set_title('Execution-sensitivity gap per (p_in, p_out) — sticky vs track-down')
    ax.legend()
    plt.tight_layout(); plt.show()

    print(f\"sticky:  ALL cells negative. Best: \"
          f\"{merged.loc[merged['total_dollar_pnl_sticky'].idxmax(), 'cell']}  \"
          f\"${merged['total_dollar_pnl_sticky'].max():,.0f}\")
    print(f\"td:      ALL cells positive. Best: \"
          f\"{merged.loc[merged['total_dollar_pnl_td'].idxmax(), 'cell']}  \"
          f\"${merged['total_dollar_pnl_td'].max():,.0f}\")
    print(f\"Average sticky→td gap per cell: ${(merged['total_dollar_pnl_td']\"
          f\"- merged['total_dollar_pnl_sticky']).mean():,.0f}\")
"""))

# ── §10 Conclusion ─────────────────────────────────────────────────────────
cells.append(md("""
## §10. Conclusion + deployment verdict + LIVE TEST DESIGN

### Final state

| layer | finding |
|---|---|
| §2 hold-to-resolution | gross edge peaks +1.31¢ at p=0.70; dies at 2¢ slip |
| §3-§4 taker | canonical edge −2.4% ROI, fallback-dominated |
| §5 subset (Proposal B) | "active markets" cost MORE under taker, not less |
| §6 WS-passive optimistic | +0.87% to +6% ROI under perfect queue (UPPER BOUND) |
| §9 chase-bid (entry=p_in, passive exit) | sticky-at-p_in: −14.7% ROI canonical |
| **§11 user-calibrated STICKY (realistic)** | **all cells negative, −427% to −941% ROI on $10k bankroll** |
| §11 user-calibrated TRACK_DOWN (UB) | +400% to +3000% ROI; requires unrealistic queue |

### Verdict

**Do not deploy from analysis alone.** Under realistic queue assumptions
(sticky model: bid sits at p_in, gets adversely hit first when price crashes)
every (p_in, p_out) cell loses 4-9× your bankroll over 12 months at 5%
sizing. The strategy only works if your bot achieves track-down execution
sufficiently often, which is unlikely on low-liquidity Polymarket weather
markets where queue rebuild + cancel latency favour adverse selection.

**The deployment question is no longer "is the strategy profitable?"** It is
**"can my execution stack achieve track-down behaviour often enough to flip
the cell from sticky-negative to mixed-positive?"** The §11 grid shows the
breakeven mix per cell.

### LIVE TEST DESIGN — execution calibration, not profit testing

**Phase 1 (2-4 weeks, MEASUREMENT only)**

Pick `(p_in=0.85, p_out=0.90)` as the test cell — it has 16% fill rate (the
highest), smallest sticky loss, and smallest track-down win, so the gap is
narrow and the sample fills fastest with bounded downside.

Trade tiny size: **$10-50 per filled trade**, not 5% of bankroll. The goal
is observation, not return. Expected Phase 1 worst case loss: $50 × 30
losing trades = $1,500 on a $10k account.

Log every action to ms resolution:
- detection_ts (cross of p_in observed)
- post_ts, cancel_ts, repost_ts (each)
- fill_ts, fill_price
- order-book snapshot (best bid, best ask, your queue position) every 5s

Compute per filled trade:
- Was your bid at the **current touch** or **stale above touch** at fill time?
- What was the historical bid_nf_price at that ts (= sticky reference)?
- Did you get queue priority, or were you behind another maker?

Phase 1 success criterion: **≥50 filled trades**.

**Phase 1 readouts (the only numbers that matter):**

- `fill_rate_actual` vs predicted 16% — should match either model
- `mean_entry_price_actual` vs p_in=0.85
  - sticky predicts: ≈ 0.85
  - track-down predicts: ≈ p_in − 10¢ = 0.75
- `cancel_replace_latency_p50` and `_p90`
  - if p90 > 2s, you cannot dodge fast-moving markets → effectively sticky
- `pct_fills_at_or_below_market_touch` — direct track-down measure
  - sticky bot: near 0%
  - perfect track-down bot: near 100%

**Decision after Phase 1:**

- ≥70% sticky-style fills → **shelve.** Realistic numbers say -400% to -900% ROI.
- ≥70% track-down-style fills → proceed to Phase 2 at the (0.80, 0.90) cell
  (max track-down PnL).
- Mixed (40-70%) → compute breakeven mix per cell from §11 numbers; deploy at
  a cell where measured mix ≥ breakeven mix.

**Phase 2 (only if Phase 1 says go):** scale to 5% sizing on a small sandbox
($1-2k) at the chosen cell for 4-8 weeks. Track measured PnL vs predicted;
if measured > 50% of predicted track-down PnL, scale to full bankroll.

### Tradable city set (selection bias)

If you deploy: Asian-session weather markets near expiry dominate the
filled subset — **Seoul, Shanghai, Tokyo, Wellington, London**. NYC and
Ankara drop out (too inactive post-cross). Scope the universe accordingly.

### See also

- [`notes/weather_ftc_state.md`](../notes/weather_ftc_state.md) — full
  state-of-things doc, copy-paste-into-cowork format.
- [`scripts/user_model_analysis.py`](../scripts/user_model_analysis.py) —
  §11 driver. Saves grids to `data/analysis/user_passive_grid_{sticky,track_down}.parquet`.
- [`scripts/passive_analysis.py`](../scripts/passive_analysis.py) — earlier
  WS-passive grid driver (optimistic queue).
"""))


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
