"""Builds notebooks/cookbook.ipynb — short recipes for the questions people actually ask, each a
few lines through the public epsilon_data API. Edit this builder and re-run; do not edit the
.ipynb by hand."""
from __future__ import annotations
import os
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def code(t): C.append(nbf.v4.new_code_cell(t))

md("# epsilon_data cookbook\nShort recipes, a few lines each, all through the public API. "
   "Pair with `epsilon_data/README.md` and the worked-examples notebook.")
code("import sys, pathlib\n"
     "root = next((p for p in pathlib.Path.cwd().parents if (p/'epsilon_data').is_dir()), pathlib.Path.cwd().parent)\n"
     "sys.path.insert(0, str(root))\n"
     "import pandas as pd, epsilon_data as ed\n"
     "print('epsilon_data', ed.__version__)")

md("### 1 · Filter markets by liquidity")
code("ed.catalog(universe='politics_negrisk', min_trades=200).sort_values('n_trades', ascending=False)"
    ".head(10)[['question','outcome_label','n_trades','median_spread_cents']]")

md("### 2 · Find the busiest hour (UTC) to trade a universe")
code("a = ed.activity_by_time('esports')\n"
     "a.groupby('hour_of_day')['n_trades'].sum().sort_values(ascending=False).head(5)")

md("### 3 · Pull every market in an event")
code("ev = ed.events(universe='politics_negrisk').iloc[0]['event_slug']\n"
     "ed.catalog(event=ev)[['market_slug','outcome_label','n_trades','median_mid']]")

md("### 4 · Compare two markets' spreads")
code("top = ed.catalog(universe='politics_negrisk', min_trades=100).sort_values('n_trades', ascending=False)\n"
     "top.head(2).set_index('question')[['median_spread_cents','n_trades','median_mid']]")

md("### 5 · Build a returns series from a token's mid")
code("aid = ed.catalog(min_trades=100).sort_values('n_trades', ascending=False).iloc[0]['asset_id']\n"
     "l1 = ed.load_l1(aid).set_index('ts')\n"
     "ret = l1['mid'].resample('5min').last().ffill().pct_change().dropna()\n"
     "ret.describe()")

md("### 6 · Slice a token's tape by time window")
code("import pandas as pd\n"
     "end = l1.index.max(); start = end - pd.Timedelta('6h')\n"
     "ed.load_l1(aid, start=start, end=end)[['ts','best_bid','best_ask','mid','spread_c']].head()")

md("### 7 · The YES/NO mirror in one call")
code("cid = ed.catalog(min_trades=100).sort_values('n_trades', ascending=False).iloc[0]['condition_id']\n"
     "pair = ed.load_pair(cid); (pair[0] + pair[1]).describe()   # ~1 for a healthy market")

md("### 8 · Adverse-selection markout for a token")
code("mo = ed.markout(aid)\n"
     "{h: round(mo[f'markout_{h}'].mean()*100, 3) for h in (10,30,60)}   # ¢, maker view; negative = toxic")

md("### 9 · NegRisk YES-sum (instantaneous, not a sum of medians)")
code("negev = ed.events(universe='politics_negrisk')\n"
     "negev = negev[(negev.neg_risk==True) & (negev.n_markets>=4)].iloc[0]['event_slug']\n"
     "ns = ed.negrisk_sum(negev); ns['yes_sum'].describe()")

md("### 10 · Audit a market (report + recommendation; never writes)")
code("r = ed.audit_market(cid)\n"
     "print(r.verdict, '—', r.reason)\n"
     "pd.DataFrame([{'check':c.name,'level':c.level,'detail':c.detail} for c in r.checks])")

md("### 11 · Coverage — true gaps vs quiet hours")
code("cov = ed.coverage('esports'); cov[cov.n_missing>0][['date','missing_hours']]   # no-book = true gap")

nb['cells'] = C
out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "cookbook.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", os.path.abspath(out))
