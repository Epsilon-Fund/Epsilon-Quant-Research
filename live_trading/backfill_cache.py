#!/usr/bin/env python3
"""
live_trading/backfill_cache.py
==============================
One-time script to populate the local OHLCV cache from a given start date.

Run this once before using the dashboard for the first time, or after adding
a new symbol to ACTIVE_ASSETS.

Usage
-----
  python3 live_trading/backfill_cache.py
  python3 live_trading/backfill_cache.py --from 2026-01-01

Options
-------
  --from DATE   Start date (YYYY-MM-DD). Default: 2026-01-01.
                Fetches daily and hourly bars from this date to today.

Output
------
  ETHUSDT:  847 daily bars,  20328 hourly bars cached
  XRPUSDT:  847 daily bars,  20328 hourly bars cached
  ...
  Cache directory: /path/to/live_trading/cache
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

_REPO_ROOT  = os.path.dirname(_HERE)
_INFRA_DATA = os.path.join(_REPO_ROOT, 'infrastructure', 'data')
if _INFRA_DATA not in sys.path:
    sys.path.insert(0, _INFRA_DATA)

from shared.cache_manager import (
    _daily_path,
    _hourly_path,
    _fetch_daily_binance,
    _fetch_hourly_binance,
    _write_parquet,
    _merge_and_save,
    _read_parquet,
    _ensure_dirs,
    _CACHE_ROOT,
)
from dashboards.momentum.config import ACTIVE_ASSETS as _MOM_ASSETS
from dashboards.statarb.optimise import ASSET_CONFIG as _SA_CONFIG

# Aggregate all unique symbols across strategies
_SA_SYMBOLS = sorted({sym for a in _SA_CONFIG for sym in (a['symbol_y'], a['symbol_x'])})
ACTIVE_ASSETS = sorted(set(_MOM_ASSETS) | set(_SA_SYMBOLS))


def backfill_symbol(symbol: str, start_date: str, end_date: str) -> tuple[int, int]:
    """
    Fetch and cache daily + hourly OHLCV for one symbol over the given range.
    Merges with any existing cache rather than replacing it.

    Returns (n_daily_bars_in_cache, n_hourly_bars_in_cache).
    """
    _ensure_dirs()

    # ── Daily ─────────────────────────────────────────────────────────────────
    d_path = _daily_path(symbol)
    print(f'  {symbol} daily:  fetching {start_date} -> {end_date}…', end=' ', flush=True)
    try:
        new_daily = _fetch_daily_binance(symbol, start_date, end_date)
        if not new_daily.empty:
            if d_path.exists():
                existing = _read_parquet(d_path)
                if existing is not None and not existing.empty:
                    result_daily = _merge_and_save(existing, new_daily, d_path)
                else:
                    _write_parquet(new_daily, d_path)
                    result_daily = new_daily
            else:
                _write_parquet(new_daily, d_path)
                result_daily = new_daily
            n_daily = len(result_daily)
            print(f'{n_daily} bars total')
        else:
            n_daily = 0
            print('no bars returned')
    except Exception as e:
        print(f'FAILED: {e}')
        n_daily = 0

    # ── Hourly ────────────────────────────────────────────────────────────────
    h_path = _hourly_path(symbol)
    # Add a 2-day tail buffer so T+1 execution-hour lookups always succeed
    hourly_end = (
        datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=2)
    ).strftime('%Y-%m-%d')
    print(f'  {symbol} hourly: fetching {start_date} -> {hourly_end}…', end=' ', flush=True)
    try:
        new_hourly = _fetch_hourly_binance(symbol, start_date, hourly_end)
        if not new_hourly.empty:
            if h_path.exists():
                existing = _read_parquet(h_path)
                if existing is not None and not existing.empty:
                    result_hourly = _merge_and_save(existing, new_hourly, h_path)
                else:
                    _write_parquet(new_hourly, h_path)
                    result_hourly = new_hourly
            else:
                _write_parquet(new_hourly, h_path)
                result_hourly = new_hourly
            n_hourly = len(result_hourly)
            print(f'{n_hourly} bars total')
        else:
            n_hourly = 0
            print('no bars returned')
    except Exception as e:
        print(f'FAILED: {e}')
        n_hourly = 0

    return n_daily, n_hourly


def main():
    parser = argparse.ArgumentParser(
        description='Backfill local OHLCV cache from a given start date.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--from', dest='start_date',
        default='2026-01-01',
        help='Start date for backfill (YYYY-MM-DD). Default: 2026-01-01',
    )
    args = parser.parse_args()

    start_date = args.start_date
    end_date   = datetime.utcnow().strftime('%Y-%m-%d')

    # Validate date format
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
    except ValueError:
        print(f'ERROR: --from date must be YYYY-MM-DD, got: {start_date}')
        sys.exit(1)

    print(f'\n── Backfilling OHLCV cache ──────────────────────────────────────────')
    print(f'   Range  : {start_date} -> {end_date}')
    print(f'   Symbols: {ACTIVE_ASSETS}')
    print(f'   Output : {_CACHE_ROOT}')
    print()

    total_daily  = 0
    total_hourly = 0

    for symbol in ACTIVE_ASSETS:
        print(f'\n[{symbol}]')
        n_d, n_h = backfill_symbol(symbol, start_date, end_date)
        total_daily  += n_d
        total_hourly += n_h

    print(f'\n── Backfill complete ────────────────────────────────────────────────')
    print(f'   {total_daily} daily bars  |  {total_hourly} hourly bars')
    print(f'   Cache: {_CACHE_ROOT}')
    print()
    print('Next step: streamlit run live_trading/app.py')
    print('  The dashboard will read from parquet on every load.')
    print('  The cache is kept fresh automatically on each app startup.')


if __name__ == '__main__':
    main()
