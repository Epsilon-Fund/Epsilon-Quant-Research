"""Builds notebooks/epsilon_data_examples.ipynb — five worked tasks through the PUBLIC loader
API only, each ending in a plot. Per repo convention, edit this builder and re-run; do not edit
the .ipynb by hand. Execute to verify: PYTHONPATH=. python scripts/build_examples_notebook.py
then nbconvert --execute (the build step below does both)."""
from __future__ import annotations
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []

def md(t): C.append(nbf.v4.new_markdown_cell(t))
def code(t): C.append(nbf.v4.new_code_cell(t))

md("# epsilon_data — worked examples\n"
   "Five real tasks, each a few lines through the **public** loader, each ending in a plot. "
   "If any of these is awkward to write, the API is wrong (fix the API, not the notebook).")

code("# make epsilon_data importable no matter where this notebook is opened from\n"
     "import sys, pathlib\n"
     "here = pathlib.Path.cwd()\n"
     "root = next((p for p in [here, *here.parents] if (p / 'epsilon_data').is_dir()), here.parent)\n"
     "sys.path.insert(0, str(root))\n"
     "import matplotlib.pyplot as plt   # inline backend under Jupyter captures the figures\n"
     "import epsilon_data as ed\n"
     "print('epsilon_data', ed.__version__, '| root', root)")

md("## 1 — Find the busiest politics markets and open one")
code("cat = ed.catalog(universe='politics_negrisk', min_trades=1)\n"
     "top = cat.sort_values('n_trades', ascending=False).head(10)\n"
     "print(top[['question','outcome_label','n_trades','median_spread_cents','path']].to_string(index=False))\n"
     "ref = top.iloc[0]['path']          # open the busiest\n"
     "print('\\nopening:', ref)")

md("## 2 — Plot a market's mid and spread across its life")
code("l1 = ed.load_l1(ref)\n"
     "fig, (a0, a1) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)\n"
     "a0.plot(l1['ts'], l1['mid']*100, lw=0.8); a0.set_ylabel('mid (¢)')\n"
     "a1.plot(l1['ts'], l1['spread_c'], lw=0.6, color='purple'); a1.set_ylabel('spread (¢)')\n"
     "a0.set_title(f'{ref}  ({len(l1):,} L1 events)'); a1.set_xlabel('UTC'); plt.tight_layout(); plt.show()")

md("## 3 — Plot both sides together and confirm they mirror\n"
   "`load_pair` returns both outcomes' mids on one aligned index; a healthy market sums to ~1.")
code("cid = top.iloc[0]['condition_id']\n"
     "pair = ed.load_pair(cid)\n"
     "labels = pair.attrs['labels']\n"
     "fig, ax = plt.subplots(figsize=(11, 4))\n"
     "for idx in (0, 1):\n"
     "    ax.plot(pair.index, pair[idx]*100, lw=0.8, label=f'{labels[idx]}')\n"
     "ax.plot(pair.index, (pair[0]+pair[1])*100, lw=0.6, color='grey', ls='--', label='sum')\n"
     "ax.axhline(100, color='k', lw=0.5); ax.set_ylabel('mid (¢)'); ax.legend(); plt.tight_layout(); plt.show()\n"
     "print('median sum:', round(float((pair[0]+pair[1]).median()), 4))")

md("## 4 — Plot every candidate in a NegRisk event and look at the YES sum\n"
   "Across a politics NegRisk event, the YES mids of all its markets should sum to ~1.")
code("evs = ed.events(universe='politics_negrisk')\n"
     "evs = evs[(evs['neg_risk'] == True) & (evs['n_markets'] >= 4)].sort_values('total_trades', ascending=False)\n"
     "ev_slug = evs.iloc[0]['event_slug']\n"
     "wide = ed.load_event(ev_slug)\n"
     "meta = wide.attrs['tokens']\n"
     "yes_cols = [a for a in wide.columns if meta[a]['outcome'] == 'YES']\n"
     "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
     "for a in yes_cols:\n"
     "    ax.plot(wide.index, wide[a]*100, lw=0.6)\n"
     "ax.plot(wide.index, wide[yes_cols].sum(axis=1)*100, color='black', lw=1.2, label='YES sum')\n"
     "ax.axhline(100, color='red', ls='--', lw=0.7); ax.set_title(ev_slug); ax.set_ylabel('mid (¢)'); ax.legend(); plt.tight_layout(); plt.show()\n"
     "print('event:', ev_slug, '| markets:', len(yes_cols), '| median YES-sum:', round(float(wide[yes_cols].sum(axis=1).median()), 3))")

md("## 5 — Take a resolved market and plot the path into settlement\n"
   "For a market that converged, the winner walks to 100¢ and the loser to 0¢.")
code("res = ed.catalog(universe='politics_negrisk', resolved=True)\n"
     "res = res[res['check4_status'] == 'converged_correct'].sort_values('n_l1_events', ascending=False)\n"
     "rcid = res.iloc[0]['condition_id']\n"
     "pair = ed.load_pair(rcid); labels = pair.attrs['labels']\n"
     "won = {i: (res[(res.condition_id==rcid) & (res.outcome_index==i)]['resolved_outcome'].iloc[0] == labels[i]) for i in (0,1)}\n"
     "fig, ax = plt.subplots(figsize=(11, 4))\n"
     "for i in (0, 1):\n"
     "    ax.plot(pair.index, pair[i]*100, lw=0.9, color=('green' if won[i] else 'red'),\n"
     "            label=f\"{labels[i]} ({'won' if won[i] else 'lost'})\")\n"
     "ax.set_ylim(-5, 105); ax.set_ylabel('mid (¢)'); ax.legend(); ax.set_title(res.iloc[0]['question'][:70]); plt.tight_layout(); plt.show()")

nb["cells"] = C
out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "epsilon_data_examples.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", os.path.abspath(out))
