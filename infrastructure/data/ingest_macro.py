"""
ingest_macro.py
===============
Fetch macro daily closes via yfinance (no API key required).

Produces:
  topics/ml-prediction/data/raw/macro/macro_daily.parquet
  columns: dxy_close, spx_close, gold_close, vix_close, yield10y_close

IMPORTANT: US markets close ~21:00 UTC, which is after the Binance daily bar
closes at 00:00 UTC. Always apply a 1-day lag to these columns before joining
to crypto features — otherwise you are using data your model couldn't have seen.

Usage:
  python ingest_macro.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

_ML_RAW    = Path(__file__).resolve().parents[2] / 'topics' / 'ml-prediction' / 'data' / 'raw'
_MACRO_DIR = _ML_RAW / 'macro'
_OUT_PATH  = _MACRO_DIR / 'macro_daily.parquet'

_TICKERS = {
    'dxy_close':      'DX-Y.NYB',
    'spx_close':      '^GSPC',
    'gold_close':     'GC=F',
    'vix_close':      '^VIX',
    'yield10y_close': '^TNX',
}

_START = '2019-01-01'


def fetch_macro(start: str) -> pd.DataFrame:
    frames = {}
    for col, ticker in _TICKERS.items():
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty:
            print(f'  WARNING: no data for {ticker}')
            continue
        frames[col] = df['Close'].squeeze()

    if not frames:
        return pd.DataFrame()

    out = pd.DataFrame(frames)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = 'date'
    return out.dropna(how='all').sort_index()


def main() -> None:
    _MACRO_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing data — never overwrite historical rows
    existing = pd.read_parquet(_OUT_PATH) if _OUT_PATH.exists() else pd.DataFrame()
    if not existing.empty:
        last_date  = existing.index[-1]
        fetch_from = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        print(f'Existing data through {last_date.date()}, fetching from {fetch_from}...')
    else:
        fetch_from = _START
        print(f'No existing file, fetching from {fetch_from}...')

    new_data = fetch_macro(fetch_from)

    if new_data.empty:
        print('  No new data available — file unchanged.')
        return

    missing = set(_TICKERS) - set(new_data.columns)
    if missing:
        print(f'  WARNING: {missing} failed to download — appending available columns only.')

    # Merge: existing rows are never changed, new rows are appended
    if not existing.empty:
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep='first')].sort_index()
    else:
        combined = new_data.sort_index()

    combined.to_parquet(_OUT_PATH, engine='pyarrow')

    added = len(combined) - len(existing)
    print(f'  Added {added} new rows. Total: {combined.index[0].date()} -> {combined.index[-1].date()}')
    for col in combined.columns:
        print(f'  {col:16s}  {combined[col].notna().sum()} rows, {combined[col].isna().sum()} nulls')
    print('Done.')


if __name__ == '__main__':
    main()
