"""
ingest_binance_vision.py
========================
Download historical OI and long/short ratio from the Binance Vision data portal.
Bypasses the 30-day API limit on openInterestHist / globalLongShortAccountRatio.

Source: https://data.binance.vision/data/futures/um/
  daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{YYYY-MM-DD}.zip
  (monthly metrics files do not exist on Binance Vision)

Each zip contains a CSV with one row per 5-minute bar. We aggregate to daily:
  oi_usd    = last value of the day (end-of-day OI in USD)
  ls_ratio  = mean value of the day (average global long/short ratio)

Produces (same paths as ingest_futures.py):
  topics/ml-prediction/data/raw/
    oi/        {SYMBOL}_oi.parquet         columns: oi_usd
    longshort/ {SYMBOL}_ls.parquet         columns: ls_ratio

Usage:
  python ingest_binance_vision.py                   # all symbols
  python ingest_binance_vision.py --symbol BTCUSDT  # single symbol
  python ingest_binance_vision.py --force           # re-download everything
"""

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ── Storage ───────────────────────────────────────────────────────────────────
_ML_RAW  = Path(__file__).resolve().parents[2] / 'topics' / 'ml-prediction' / 'data' / 'raw'
_OI_DIR  = _ML_RAW / 'oi'
_LS_DIR  = _ML_RAW / 'longshort'

UNIVERSE = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'LINKUSDT', 'AVAXUSDT']

_BASE    = 'https://data.binance.vision/data/futures/um'
_TIMEOUT = 60

# Columns in the metrics CSV that we care about
_COL_TIME = 'create_time'
_COL_OI   = 'sum_open_interest_value'
_COL_LS   = 'count_long_short_ratio'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _save(df: pd.DataFrame, path: Path) -> None:
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.to_parquet(path, engine='pyarrow')


def _download_zip(url: str) -> bytes | None:
    """Download a zip from Binance Vision. Returns None if file doesn't exist yet."""
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content
    except requests.RequestException:
        return None


def _parse_metrics_zip(content: bytes) -> pd.DataFrame:
    """
    Parse a Binance Vision metrics zip into a DataFrame with daily OI and L/S.
    The zip contains a single CSV with 5-minute rows.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    df['create_time'] = pd.to_datetime(df['create_time'])
    df['date']        = df['create_time'].dt.normalize()
    df[_COL_OI]       = pd.to_numeric(df[_COL_OI], errors='coerce')
    df[_COL_LS]       = pd.to_numeric(df[_COL_LS], errors='coerce')

    # Aggregate 5-min bars to daily
    daily = df.groupby('date').agg(
        oi_usd   = (_COL_OI, 'last'),   # end-of-day OI
        ls_ratio = (_COL_LS, 'mean'),   # average intraday L/S
    )
    daily.index = pd.to_datetime(daily.index)
    return daily


# ── Download loop ─────────────────────────────────────────────────────────────

def _day_url(symbol: str, d: date) -> str:
    return f'{_BASE}/daily/metrics/{symbol}/{symbol}-metrics-{d}.zip'


def _fetch_day(symbol: str, d: date):
    """Download and parse one daily metrics zip. Returns (date, DataFrame) or (date, None)."""
    content = _download_zip(_day_url(symbol, d))
    if content is None:
        return d, None
    try:
        return d, _parse_metrics_zip(content)
    except Exception:
        return d, None


def fetch_all(symbol: str, force: bool = False) -> pd.DataFrame:
    """
    Download full history for symbol using daily files, fetched concurrently.
    Binance Vision only has daily metrics files (monthly do not exist).
    """
    if force:
        existing = pd.DataFrame()
    else:
        oi_ex = _load(_OI_DIR / f'{symbol}_oi.parquet')
        ls_ex = _load(_LS_DIR / f'{symbol}_ls.parquet')
        if not oi_ex.empty and not ls_ex.empty:
            existing = oi_ex.join(ls_ex, how='outer')
        elif not oi_ex.empty:
            existing = oi_ex
        else:
            existing = pd.DataFrame()

    start_date = date(2020, 1, 1) if existing.empty else (
        existing.index[-1].date() + timedelta(days=1)
    )
    today = date.today()

    if start_date >= today:
        return existing

    all_dates = [start_date + timedelta(days=i)
                 for i in range((today - start_date).days)]

    frames = [] if existing.empty else [existing]
    done   = 0

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_day, symbol, d): d for d in all_dates}
        for fut in as_completed(futures):
            _, df = fut.result()
            done += 1
            if df is not None:
                frames.append(df)
            if done % 100 == 0:
                print(f'    {done}/{len(all_dates)} days downloaded...')

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    return combined


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default=None)
    parser.add_argument('--force',  action='store_true')
    args = parser.parse_args()

    _OI_DIR.mkdir(parents=True, exist_ok=True)
    _LS_DIR.mkdir(parents=True, exist_ok=True)

    symbols = [args.symbol] if args.symbol else UNIVERSE

    for sym in symbols:
        print(f'\n{sym}')
        df = fetch_all(sym, force=args.force)

        if df.empty:
            print(f'  no data returned')
            continue

        oi_path = _OI_DIR / f'{sym}_oi.parquet'
        ls_path = _LS_DIR / f'{sym}_ls.parquet'

        _save(df[['oi_usd']],   oi_path)
        _save(df[['ls_ratio']], ls_path)

        print(f'  oi       {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} rows)')
        print(f'  ls_ratio {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} rows)')

    print('\nDone.')


if __name__ == '__main__':
    main()
