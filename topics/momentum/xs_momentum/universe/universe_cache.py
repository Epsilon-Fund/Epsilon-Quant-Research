"""
universe_cache.py — USDT perpetual futures OHLCV cache for XS momentum.

Run this script once to build the full cache, then daily to keep it current:
    python universe_cache.py

First run: pulls full history for all qualifying USDT perpetual futures (~5–10 min).
Subsequent runs: incremental — only fetches new candles since the last cached
date per symbol, making daily updates fast (1–3 min total for all symbols).

All data comes from Binance USDT perpetual futures (futures_historical_klines).
Perp prices are used for ranking signals, universe filtering (by perp volume),
AND backtest PnL — they match what would be traded on Hyperliquid (long top-N,
short bottom-N).  The spot/perp price difference is negligible for the large-cap
liquid assets the volume filter selects.

Cache files written to topics/momentum/xs_momentum/cache/:
    close.parquet    — wide DatetimeIndex × coin (perp Close prices)
    volume.parquet   — wide DatetimeIndex × coin (perp daily USDT quote volume)
    meta.parquet     — per-coin metadata: listing_date

Column conventions match get_data() in infrastructure/data/binance_client.py:
    - DatetimeIndex named 'Time', float dtypes, daily '1d' interval
    - Quote_volume = USDT notional traded (index 7 in the raw klines response)
    - Last incomplete candle (current day) is always dropped
"""

import os
import sys
import warnings
import pandas as pd

_HERE  = os.path.dirname(os.path.abspath(__file__))
_INFRA = os.path.abspath(os.path.join(_HERE, '..', '..', '..', 'infrastructure', 'data'))
if _INFRA not in sys.path:
    sys.path.insert(0, _INFRA)

from binance_client import get_binance_client

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR  = os.path.join(_HERE, 'cache')
CLOSE_PATH = os.path.join(CACHE_DIR, 'close.parquet')
VOL_PATH   = os.path.join(CACHE_DIR, 'volume.parquet')
META_PATH  = os.path.join(CACHE_DIR, 'meta.parquet')

INTERVAL         = '1d'
DEFAULT_LOOKBACK = 2500   # ~6.8 years; first-time pull length per symbol
                          # NOTE: increasing this AFTER the cache is built does NOT
                          # automatically backfill older data — incremental updates
                          # only fetch forward from the last cached date.  If you
                          # need a longer lookback, delete cache/ and re-run.

# ── exclusions (applied at fetch time so we don't waste API calls) ─────────────
STABLECOINS = {
    'USDT', 'USDC', 'DAI',  'BUSD', 'TUSD', 'FDUSD', 'USDD', 'PYUSD',
    'USDP', 'USDX', 'GUSD', 'USDN', 'SUSD', 'LUSD',  'FRAX', 'MIM',
    'DOLA', 'CRVUSD', 'EURS', 'EURT', 'EURC',
}
WRAPPED_STAKED = {
    'WBTC', 'WETH', 'WBETH', 'STETH', 'CBETH', 'WBNB',  'WMATIC', 'WAVAX',
    'RETH', 'STSOL', 'MSOL',  'LSETH', 'SWETH', 'FRXETH', 'SFRXETH',
    'ANKRETH', 'BETH',
}


# ── exchange info ─────────────────────────────────────────────────────────────

def _get_perp_symbols(client):
    """
    Returns dict[full_symbol → {'base': str, 'onboard_date': pd.Timestamp}]
    for all active USDT perpetual futures, excluding stablecoins and wrapped tokens.
    onboard_date is the exchange-reported listing timestamp (timezone-naive UTC).
    """
    info   = client.futures_exchange_info()
    result = {}
    for s in info['symbols']:
        if s.get('contractType') != 'PERPETUAL':
            continue
        if s.get('quoteAsset') != 'USDT':
            continue
        if s.get('status') != 'TRADING':
            continue
        base = s['baseAsset'].upper()
        if base in STABLECOINS or base in WRAPPED_STAKED:
            continue
        onboard_ms = s.get('onboardDate')
        onboard    = pd.Timestamp(onboard_ms, unit='ms') if onboard_ms else pd.NaT
        result[s['symbol']] = {'base': base, 'onboard_date': onboard}
    return result


# ── kline fetcher ─────────────────────────────────────────────────────────────

def _fetch_perp(client, symbol, start_str):
    """
    Pull daily perp klines using client.futures_historical_klines.

    Returns DataFrame(index=DatetimeIndex('Time'), columns=['Close','Quote_volume'])
    or None on failure.  Drops the last bar (incomplete current-day candle).
    """
    try:
        klines = client.futures_historical_klines(symbol, INTERVAL, start_str)
    except Exception as exc:
        warnings.warn(f'perp {symbol}: {exc}')
        return None
    if not klines:
        return None
    df = pd.DataFrame(klines, columns=[
        'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_volume', 'Trades', 'Taker_base', 'Taker_quote', 'Ignore',
    ])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    df = df.set_index('Time')
    df = df[['Close', 'Quote_volume']].astype(float)
    return df.iloc[:-1] if len(df) > 1 else None


# ── incremental helpers ───────────────────────────────────────────────────────

def _start_str_from(last_date, lookback=DEFAULT_LOOKBACK):
    """
    Return the Binance start argument given the last cached date.

    For incremental updates we return an integer millisecond timestamp, which
    Binance's API accepts directly and which bypasses python-binance's
    internal dateparser.  The dateparser raises a DeprecationWarning on
    Python 3.14+ for absolute date strings, even ones with explicit years.

    First-time pulls keep the relative `'N days ago UTC'` string — that
    phrase is parsed cleanly without triggering the warning.
    """
    if last_date is None:
        return f'{lookback} days ago UTC'
    next_day = last_date + pd.Timedelta(days=1)
    # pd.Timestamp.value is nanoseconds since the Unix epoch (UTC).
    return int(next_day.value // 1_000_000)


def _append_column(panel, base, new_series):
    """
    Append new rows (or a new column) for `base` into a wide Date × coin panel.
    Returns (updated_panel, n_new_rows).
    """
    if len(new_series) == 0:
        return panel, 0

    new_series = new_series.rename(base)

    if base in panel.columns:
        last     = panel[base].last_valid_index()
        new_rows = new_series[new_series.index > last] if last is not None else new_series
        if len(new_rows) == 0:
            return panel, 0
        panel = pd.concat([panel, new_rows.to_frame()], axis=0)
        return panel, len(new_rows)
    else:
        panel = pd.concat([panel, new_series.to_frame()], axis=1)
        return panel, len(new_series)


def _dedup(df):
    return df[~df.index.duplicated(keep='last')].sort_index()


# ── main entry point ──────────────────────────────────────────────────────────

def update_cache(client=None, lookback=DEFAULT_LOOKBACK, verbose=True):
    """
    Build or incrementally update the perp cache.

    Parameters
    ----------
    client   : Binance Client; created automatically if None
    lookback : days of history for first-time symbol pulls
    verbose  : print progress

    Returns
    -------
    dict: {'updated_symbols': int, 'new_candles': int, 'new_pairs': int}
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if client is None:
        client = get_binance_client()

    close = pd.read_parquet(CLOSE_PATH) if os.path.exists(CLOSE_PATH) else pd.DataFrame()
    vol   = pd.read_parquet(VOL_PATH)   if os.path.exists(VOL_PATH)   else pd.DataFrame()
    meta  = pd.read_parquet(META_PATH)  if os.path.exists(META_PATH)  else pd.DataFrame()

    if verbose:
        print('Fetching perp exchange info...')
    perp_symbols = _get_perp_symbols(client)
    n_perp = len(perp_symbols)
    if verbose:
        print(f'  Active USDT perp pairs: {n_perp}')

    stats       = {'updated_symbols': 0, 'new_candles': 0, 'new_pairs': 0}
    meta_update = {}

    if verbose:
        print(f'\nUpdating perps ({n_perp} symbols)...')

    for i, (sym, info) in enumerate(sorted(perp_symbols.items()), 1):
        base   = info['base']
        is_new = base not in close.columns

        # Use the earliest last date across close and volume to avoid gaps
        last_close = close[base].last_valid_index() if base in close.columns else None
        last_vol   = vol[base].last_valid_index()   if base in vol.columns   else None
        last_any   = min((d for d in [last_close, last_vol] if d is not None), default=None)
        start      = _start_str_from(last_any, lookback)

        df = _fetch_perp(client, sym, start)
        if df is None:
            continue

        close, n_close = _append_column(close, base, df['Close'])
        vol,   _       = _append_column(vol,   base, df['Quote_volume'])

        if n_close > 0:
            stats['new_candles']     += n_close
            stats['updated_symbols'] += 1
        if is_new:
            stats['new_pairs'] += 1

        meta_update[base] = {'listing_date': info['onboard_date']}

        if verbose and i % 50 == 0:
            print(f'  {i}/{n_perp}  ({stats["new_candles"]} new candles so far)')

    # ── deduplicate and save ──────────────────────────────────────────────────
    close = _dedup(close)
    vol   = _dedup(vol)
    close.index.name = 'Time'
    vol.index.name   = 'Time'

    close.to_parquet(CLOSE_PATH)
    vol.to_parquet(VOL_PATH)

    new_meta = pd.DataFrame(meta_update).T
    new_meta.index.name = 'base'
    if len(meta) > 0:
        meta = new_meta.combine_first(meta)
    else:
        meta = new_meta
    meta.to_parquet(META_PATH)

    if verbose:
        print(f'\n✓ Cache update complete')
        print(f'  Updated symbols : {stats["updated_symbols"]}')
        print(f'  New candles     : {stats["new_candles"]}')
        print(f'  New pairs added : {stats["new_pairs"]}')
        print(f'  close           : {close.shape}  ({len(close.columns)} coins × {len(close)} dates)')
        print(f'  volume          : {vol.shape}')
        print(f'  meta            : {meta.shape}')

    return stats


if __name__ == '__main__':
    update_cache()
