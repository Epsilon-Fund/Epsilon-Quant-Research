"""
ingest_deribit.py
=================
Fetch Deribit volatility index (DVOL) for BTC and ETH.

Produces:
  topics/ml-prediction/data/raw/deribit/
    BTC_dvol.parquet    columns: dvol
    ETH_dvol.parquet    columns: dvol

DVOL is Deribit's VIX-equivalent — a 30-day forward-looking IV derived from the
options market. Available from ~2021 for both BTC and ETH.

Note: 25-delta skew computation is deferred — it requires filtering the live
options chain by delta, which is ~10x more complex. DVOL alone is the primary
volatility signal; add skew in phase 2 if feature importance warrants it.

Usage:
  python ingest_deribit.py
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

_ML_RAW      = Path(__file__).resolve().parents[2] / 'topics' / 'ml-prediction' / 'data' / 'raw'
_DERIBIT_DIR = _ML_RAW / 'deribit'

_BASE        = 'https://www.deribit.com/api/v2'
_RESOLUTION  = 86400   # 1 day in seconds
_COINS       = ['BTC', 'ETH']
_RATE_SLEEP  = 0.5


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_dvol(currency: str) -> pd.DataFrame:
    """
    Fetch DVOL history via paginated forward pass.
    Response data shape: [[timestamp_ms, open, high, low, close], ...]
    """
    start_ms = _ms(datetime(2021, 1, 1, tzinfo=timezone.utc))
    end_ms   = _ms(datetime.now(timezone.utc))
    records  = []

    while start_ms < end_ms:
        resp = requests.get(
            f'{_BASE}/public/get_volatility_index_data',
            params={
                'currency':        currency,
                'start_timestamp': start_ms,
                'end_timestamp':   end_ms,
                'resolution':      _RESOLUTION,
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json().get('result', {})
        data   = result.get('data', [])

        if not data:
            break

        records.extend(data)

        # Deribit returns [ts, open, high, low, close] sorted ascending.
        # Advance start past last received bar.
        last_ts  = data[-1][0]
        start_ms = last_ts + _RESOLUTION * 1000

        # If we got fewer bars than a full page we're done.
        if len(data) < 700:
            break

        time.sleep(_RATE_SLEEP)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records, columns=['ts_ms', 'open', 'high', 'low', 'close'])
    df['date'] = pd.to_datetime(df['ts_ms'], unit='ms').dt.normalize()
    df = df.set_index('date')[['close']].rename(columns={'close': 'dvol'})
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def main() -> None:
    _DERIBIT_DIR.mkdir(parents=True, exist_ok=True)

    for coin in _COINS:
        print(f'Fetching DVOL for {coin}...')
        df = fetch_dvol(coin)
        if df.empty:
            print(f'  WARNING: no data returned for {coin}')
            continue
        path = _DERIBIT_DIR / f'{coin}_dvol.parquet'
        df.to_parquet(path, engine='pyarrow')
        print(f'  {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} rows)')

    print('Done.')


if __name__ == '__main__':
    main()
