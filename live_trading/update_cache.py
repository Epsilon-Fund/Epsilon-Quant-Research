#!/usr/bin/env python3
"""
live_trading/update_cache.py
============================
Incremental cache updater — fetches only bars missing since the last run.

The dashboard (app.py) calls this automatically on every `streamlit run`
via @st.cache_resource, so you normally never need to run this manually.

Run manually only if you want to pre-warm the cache outside of Streamlit:
  python3 live_trading/update_cache.py
"""

import os
import sys
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

_REPO_ROOT  = os.path.dirname(_HERE)
_INFRA_DATA = os.path.join(_REPO_ROOT, 'infrastructure', 'data')
if _INFRA_DATA not in sys.path:
    sys.path.insert(0, _INFRA_DATA)

from shared.cache_manager import update_all_caches, is_cache_fresh
from dashboards.momentum.config import ACTIVE_ASSETS

if __name__ == '__main__':
    started = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f'\n[{started}] update_cache.py starting')
    print(f'Symbols: {ACTIVE_ASSETS}')

    update_all_caches(ACTIVE_ASSETS)

    # Quick freshness summary
    print('\nFreshness check:')
    for sym in ACTIVE_ASSETS:
        d_fresh = is_cache_fresh(sym, 'daily')
        h_fresh = is_cache_fresh(sym, 'hourly')
        print(f'  {sym:12s}  daily={d_fresh}  hourly={h_fresh}')

    finished = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f'\n[{finished}] Cache update complete.')
