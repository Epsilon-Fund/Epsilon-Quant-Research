"""
ingest_onchain.py
=================
Fetch on-chain / market-structure data from DefiLlama (no API key required).

Produces:
  topics/ml-prediction/data/raw/onchain/onchain_daily.parquet
  columns: defi_tvl_usd, stablecoin_mcap_usd

Both are market-wide signals broadcast across all coins at feature-join time.
  - defi_tvl_usd:        total value locked in DeFi protocols
  - stablecoin_mcap_usd: total USDT+USDC+others circulating supply

Usage:
  python ingest_onchain.py
"""

from pathlib import Path

import pandas as pd
import requests

_ML_RAW      = Path(__file__).resolve().parents[2] / 'topics' / 'ml-prediction' / 'data' / 'raw'
_ONCHAIN_DIR = _ML_RAW / 'onchain'
_OUT_PATH    = _ONCHAIN_DIR / 'onchain_daily.parquet'

_TVL_URL     = 'https://api.llama.fi/v2/historicalChainTvl'
_STABLES_URL = 'https://stablecoins.llama.fi/stablecoincharts/all'
_TIMEOUT     = 30


def fetch_tvl() -> pd.Series:
    resp = requests.get(_TVL_URL, timeout=_TIMEOUT)
    resp.raise_for_status()

    df = pd.DataFrame(resp.json())
    df['date'] = pd.to_datetime(df['date'], unit='s').dt.normalize()
    tvl_col = [c for c in df.columns if c != 'date'][0]
    df = df.set_index('date')[tvl_col].rename('defi_tvl_usd')
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def fetch_stablecoin_mcap() -> pd.Series:
    resp = requests.get(_STABLES_URL, timeout=_TIMEOUT)
    resp.raise_for_status()

    records = []
    for row in resp.json():
        date = pd.to_datetime(row.get('date', 0), unit='s').normalize()
        circ = row.get('totalCirculatingUSD', {})
        mcap = circ.get('peggedUSD', 0) if isinstance(circ, dict) else 0
        records.append({'date': date, 'stablecoin_mcap_usd': float(mcap)})

    df = pd.DataFrame(records).set_index('date')['stablecoin_mcap_usd']
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def main() -> None:
    _ONCHAIN_DIR.mkdir(parents=True, exist_ok=True)

    print('Fetching DefiLlama TVL...')
    tvl = fetch_tvl()
    print(f'  {tvl.index[0].date()} -> {tvl.index[-1].date()}  ({len(tvl)} rows)')

    print('Fetching stablecoin market cap...')
    stables = fetch_stablecoin_mcap()
    print(f'  {stables.index[0].date()} -> {stables.index[-1].date()}  ({len(stables)} rows)')

    df = pd.concat([tvl, stables], axis=1).sort_index()
    df.to_parquet(_OUT_PATH, engine='pyarrow')
    print(f'\nSaved {len(df)} rows -> {_OUT_PATH}')
    print('Done.')


if __name__ == '__main__':
    main()
