"""
ingest_futures.py
=================
Backfill and incrementally update Binance Futures data for the ML project.

Produces three parquet files per symbol:
  topics/ml-prediction/data/raw/
    funding/   {SYMBOL}_funding.parquet    columns: funding_sum_1d
    oi/        {SYMBOL}_oi.parquet         columns: oi_usd
    longshort/ {SYMBOL}_ls.parquet         columns: ls_ratio

All indexed by UTC date (tz-naive, daily).

Usage:
  python ingest_futures.py                   # update all symbols
  python ingest_futures.py --symbol BTCUSDT  # single symbol
  python ingest_futures.py --force           # re-fetch full history
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_HERE = Path(os.path.abspath(__file__)).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from binance_client import get_binance_client

# ── Storage paths ─────────────────────────────────────────────────────────────
_ML_RAW      = Path(__file__).resolve().parents[2] / 'topics' / 'ml-prediction' / 'data' / 'raw'
_FUNDING_DIR = _ML_RAW / 'funding'
_OI_DIR      = _ML_RAW / 'oi'
_LS_DIR      = _ML_RAW / 'longshort'

UNIVERSE = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'LINKUSDT', 'AVAXUSDT']

_RATE_SLEEP  = 0.3   # seconds between API calls
_LIMIT       = 1000  # max rows per Binance request (funding)
_LIMIT_HIST  = 500   # max rows per request (OI, L/S — Binance caps at 500)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _save(df: pd.DataFrame, path: Path) -> None:
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.to_parquet(path, engine='pyarrow')


def _start_from(existing: pd.DataFrame, default: datetime) -> int:
    """Return ms timestamp to start fetching from — day after last existing row."""
    if existing.empty:
        return _ms(default)
    last = existing.index[-1]
    next_day = last + pd.Timedelta(days=1)
    return _ms(next_day.to_pydatetime().replace(tzinfo=timezone.utc))


# ── Funding rates ─────────────────────────────────────────────────────────────

def fetch_funding(client, symbol: str, start_ms: int) -> pd.DataFrame:
    """
    Fetch 8h funding payments from start_ms to now, aggregate to daily sum.
    Funding started 2019-09-13 for BTCUSDT.
    """
    now_ms  = _ms(datetime.now(timezone.utc))
    records = []

    while start_ms < now_ms:
        batch = client.futures_funding_rate(
            symbol=symbol, startTime=start_ms, limit=_LIMIT
        )
        if not batch:
            break
        records.extend(batch)
        if len(batch) < _LIMIT:
            break
        start_ms = batch[-1]['fundingTime'] + 1
        time.sleep(_RATE_SLEEP)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['date']        = pd.to_datetime(df['fundingTime'], unit='ms').dt.normalize()
    df['fundingRate'] = df['fundingRate'].astype(float)
    daily = (
        df.groupby('date')['fundingRate']
        .sum()
        .rename('funding_sum_1d')
        .to_frame()
    )
    daily.index = pd.to_datetime(daily.index)
    return daily


# ── Open interest ─────────────────────────────────────────────────────────────

def fetch_oi(client, symbol: str, start_ms: int) -> pd.DataFrame:
    """
    Fetch daily open interest in USD going back to start_ms.
    Paginates backwards with endTime to avoid Binance's startTime validation
    constraint on historical data endpoints.
    """
    end_ms  = _ms(datetime.now(timezone.utc))
    records = []

    while end_ms > start_ms:
        batch = client.futures_open_interest_hist(
            symbol=symbol, period='1d', endTime=end_ms, limit=_LIMIT_HIST
        )
        if not batch:
            break
        records.extend(batch)
        if batch[0]['timestamp'] <= start_ms:
            break
        if len(batch) < _LIMIT_HIST:
            break
        end_ms = batch[0]['timestamp'] - 1
        time.sleep(_RATE_SLEEP)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['date']   = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
    df['oi_usd'] = df['sumOpenInterestValue'].astype(float)
    df = df.set_index('date')[['oi_usd']]
    df.index = pd.to_datetime(df.index)
    return df


# ── Long/short ratio ──────────────────────────────────────────────────────────

def fetch_ls(client, symbol: str, start_ms: int) -> pd.DataFrame:
    """
    Fetch global long/short account ratio (daily) going back to start_ms.
    Paginates backwards with endTime for the same reason as fetch_oi.
    """
    end_ms  = _ms(datetime.now(timezone.utc))
    records = []

    while end_ms > start_ms:
        batch = client.futures_global_longshort_ratio(
            symbol=symbol, period='1d', endTime=end_ms, limit=_LIMIT_HIST
        )
        if not batch:
            break
        records.extend(batch)
        if batch[0]['timestamp'] <= start_ms:
            break
        if len(batch) < _LIMIT_HIST:
            break
        end_ms = batch[0]['timestamp'] - 1
        time.sleep(_RATE_SLEEP)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['date']     = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
    df['ls_ratio'] = df['longShortRatio'].astype(float)
    df = df.set_index('date')[['ls_ratio']]
    df.index = pd.to_datetime(df.index)
    return df


# ── Per-symbol update ─────────────────────────────────────────────────────────

_FUNDING_START = datetime(2019, 9, 1, tzinfo=timezone.utc)
_FUTURES_START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def update_symbol(client, symbol: str, force: bool = False) -> None:
    print(f'\n{symbol}')

    # Funding
    path     = _FUNDING_DIR / f'{symbol}_funding.parquet'
    existing = pd.DataFrame() if force else _load(path)
    start_ms = _ms(_FUNDING_START) if force else _start_from(existing, _FUNDING_START)
    fresh    = fetch_funding(client, symbol, start_ms)
    if not fresh.empty:
        combined = pd.concat([existing, fresh])
        _save(combined, path)
        combined = _load(path)
        print(f'  funding   {combined.index[0].date()} -> {combined.index[-1].date()}  ({len(combined)} rows)')
    else:
        print(f'  funding   already up to date')

    # OI
    path     = _OI_DIR / f'{symbol}_oi.parquet'
    existing = pd.DataFrame() if force else _load(path)
    start_ms = _ms(_FUTURES_START) if force else _start_from(existing, _FUTURES_START)
    fresh    = fetch_oi(client, symbol, start_ms)
    if not fresh.empty:
        combined = pd.concat([existing, fresh])
        _save(combined, path)
        combined = _load(path)
        print(f'  oi        {combined.index[0].date()} -> {combined.index[-1].date()}  ({len(combined)} rows)')
    else:
        print(f'  oi        already up to date')

    # L/S ratio
    path     = _LS_DIR / f'{symbol}_ls.parquet'
    existing = pd.DataFrame() if force else _load(path)
    start_ms = _ms(_FUTURES_START) if force else _start_from(existing, _FUTURES_START)
    fresh    = fetch_ls(client, symbol, start_ms)
    if not fresh.empty:
        combined = pd.concat([existing, fresh])
        _save(combined, path)
        combined = _load(path)
        print(f'  ls_ratio  {combined.index[0].date()} -> {combined.index[-1].date()}  ({len(combined)} rows)')
    else:
        print(f'  ls_ratio  already up to date')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default=None, help='Single symbol, e.g. BTCUSDT')
    parser.add_argument('--force',  action='store_true', help='Re-fetch full history')
    args = parser.parse_args()

    for d in (_FUNDING_DIR, _OI_DIR, _LS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    client  = get_binance_client()
    symbols = [args.symbol] if args.symbol else UNIVERSE

    for sym in symbols:
        update_symbol(client, sym, force=args.force)

    print('\nDone.')


if __name__ == '__main__':
    main()
