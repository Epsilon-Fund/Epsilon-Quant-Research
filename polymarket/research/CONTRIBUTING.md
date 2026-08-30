# Contributing to the Epsilon Polymarket research library

Two rules, no exceptions:

1. **Everything goes through `epsilon_data`.** No panel, notebook, or script reads a raw parquet
   path. If you need data the loader doesn't expose, add a loader function (below).
2. **Nothing is ever excluded automatically.** `exclusions.csv` is a human instrument, applied at
   load. Verification results are columns to look at, never filters that fire on their own.

## Add a dashboard panel — one file, zero edits elsewhere

Panels live in `dashboard/panels/`. The registry auto-discovers any file that doesn't start with
`_`, so **adding a panel is adding a file.** You never edit `app.py` or `__init__.py`.

Worked example — a complete panel that plots a token's mid as a histogram. Save it as
`dashboard/panels/mid_hist.py`:

```python
"""Explore · Mid histogram — distribution of a token's mid over the window."""
import plotly.graph_objects as go
import streamlit as st

from ._base import panel, Ctx      # the contract
from . import _data as D           # shared cached loaders + colours + chart helpers

@panel("Mid histogram", section="explore", needs="market", order=40)
def render(ctx: Ctx):
    l1 = D.l1_df(ctx.a_row.asset_id)                 # loader, cached
    l1 = l1[(l1.ts >= ctx.start) & (l1.ts <= ctx.end)]
    if l1.empty:
        st.info("No L1 in this window.")            # empty states say why
        return
    fig = go.Figure(go.Histogram(x=l1.mid * 100, nbinsx=50, marker_color=D.C_A))
    fig.update_layout(title="mid (¢)")
    st.plotly_chart(D.style(fig, 320), width="stretch", theme=None)
```

That's it — restart streamlit and a new "Mid histogram" tab appears in Explore.

The contract (`panels/_base.py`):
- `@panel(name, section, needs, order)` — `section` is `"explore"` (needs a market) or `"audit"`
  (dataset-wide); `needs` is `"market"` / `"event"` / `"none"`; `order` sorts within the section.
- `render(ctx)` draws with streamlit. `ctx` (a `Ctx`) carries `cid`, `market`, `a_row`,
  `universe`, `neg_risk`, `closed`, `event_slug`, and the window (`start`, `end`, `window`,
  `scale`, `rv_win`). Use only what you declared you need.
- Get data from `panels._data` (`D.l1_df`, `D.trades_df`, `D.pair_df`, `D.event_wide`,
  `D.markout_df`, `D.negrisk_df`, `D.cat_all`, …) — all cached, all `epsilon_data` underneath.
- Colours: `D.C_A` (first/YES), `D.C_B` (second/NO), `D.C_WON`/`D.C_LOST` (won/lost, buy/sell).
  Colour carries meaning only.

## Add a loader function — and when not to

Add one to `epsilon_data` **only when a panel needs data the API can't currently express** — a
new aggregation over the tables, a new per-token metric. Put pure-tree/identity helpers in
`catalog.py`, time-series helpers in `tape.py`, and document the function (type, unit, meaning)
in `README.md`. Keep ids as strings; expose a UTC `ts` where relevant; read per token, never the
whole table.

**Don't** add a loader function for something you can compose from existing calls in the panel
(e.g. filtering a `catalog()` result, resampling a `load_l1()` frame). Keep the API small.

## Tests

```
PYTHONPATH=. python -m pytest tests/ -q
```

`tests/test_loader.py::test_anti_drift_l1` proves the loader returns exactly what a raw parquet
read returns. **It must keep passing** — if the loader ever silently diverges from the data,
everything built on it is wrong and nothing else would catch it. Also kept green:
`test_audit_never_writes` (the audit tool must never touch `exclusions.csv`).
