"""
shared/cache_manager.py
=======================
Local OHLCV parquet cache — eliminates Binance API calls from Streamlit page loads.

Cache directory layout
----------------------
  live_trading/cache/
    daily/    {symbol}_daily.parquet   one file per symbol
    hourly/   {symbol}_hourly.parquet  one file per symbol

All timestamps stored as tz-naive UTC (matching binance_client.get_data output).

Public API
----------
  get_daily_ohlcv(symbol, warmup_bars=100)               -> pd.DataFrame
  get_daily_ohlcv_range(symbol, start_date, end_date)    -> pd.DataFrame
  get_hourly_ohlcv(symbol, start_date, end_date)         -> pd.DataFrame
  is_cache_fresh(symbol, interval="daily")               -> bool
  update_all_caches(symbols)                             -> None
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE      = Path(os.path.abspath(__file__)).parent        # live_trading/shared/
_LT_ROOT   = _HERE.parent                                   # live_trading/
_REPO_ROOT = _LT_ROOT.parent                                # repo root
_INFRA_DATA = _REPO_ROOT / 'infrastructure' / 'data'

if str(_INFRA_DATA) not in sys.path:
    sys.path.insert(0, str(_INFRA_DATA))

# ── Cache directories ─────────────────────────────────────────────────────────
_CACHE_ROOT = _LT_ROOT / 'cache'
_DAILY_DIR  = _CACHE_ROOT / 'daily'
_HOURLY_DIR = _CACHE_ROOT / 'hourly'

_DAILY_LOOKBACK = 2150   # ~6 years; covers all indicator warmups + history
_PARQUET_ENGINE = 'pyarrow'

_EMPTY_OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


# ── Directory helpers ─────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    _DAILY_DIR.mkdir(parents=True, exist_ok=True)
    _HOURLY_DIR.mkdir(parents=True, exist_ok=True)


# ── Date utilities ────────────────────────────────────────────────────────────

def _to_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    return datetime.strptime(str(v)[:10], '%Y-%m-%d').date()


def _yesterday() -> date:
    return date.today() - timedelta(days=1)


def _last_bar_date(df: pd.DataFrame) -> date | None:
    if df is None or df.empty:
        return None
    ts = df.index[-1]
    return ts.date() if hasattr(ts, 'date') else _to_date(str(ts))


def _first_bar_date(df: pd.DataFrame) -> date | None:
    if df is None or df.empty:
        return None
    ts = df.index[0]
    return ts.date() if hasattr(ts, 'date') else _to_date(str(ts))


# ── Binance fetch helpers (tz-naive output) ───────────────────────────────────

def _klines_to_df(klines: list) -> pd.DataFrame:
    """Convert raw Binance klines list to a tz-naive UTC-indexed OHLCV DataFrame."""
    if not klines:
        return pd.DataFrame(columns=_EMPTY_OHLCV)
    df = pd.DataFrame(klines, columns=[
        'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_volume', 'Trades', 'Taker_base', 'Taker_quote', 'Ignore',
    ])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')   # tz-naive UTC
    df = df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df = df.set_index('Time').astype(float)
    return df


def _fetch_daily_binance(symbol: str, start_str: str, end_str: str = None) -> pd.DataFrame:
    """Fetch daily OHLCV from Binance (tz-naive output)."""
    from binance_client import get_binance_client
    client = get_binance_client()
    klines = client.get_historical_klines(symbol, '1d', start_str, end_str)
    return _klines_to_df(klines)


def _fetch_hourly_binance(symbol: str, start_str: str, end_str: str = None) -> pd.DataFrame:
    """Fetch hourly OHLCV from Binance (tz-naive output)."""
    from binance_client import get_binance_client
    client = get_binance_client()
    klines = client.get_historical_klines(symbol, '1h', start_str, end_str)
    return _klines_to_df(klines)


# ── Parquet I/O ───────────────────────────────────────────────────────────────

def _daily_path(symbol: str) -> Path:
    return _DAILY_DIR / f'{symbol}_daily.parquet'


def _hourly_path(symbol: str) -> Path:
    return _HOURLY_DIR / f'{symbol}_hourly.parquet'


def _read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path, engine=_PARQUET_ENGINE)
    except Exception as e:
        print(f'  cache_manager: read error {path.name}: {e}')
        return None


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine=_PARQUET_ENGINE)


def _merge_and_save(existing: pd.DataFrame,
                    new_bars: pd.DataFrame,
                    path: Path) -> pd.DataFrame:
    """Concatenate, deduplicate on timestamp (keep last), sort, and save."""
    merged = pd.concat([existing, new_bars])
    merged = merged[~merged.index.duplicated(keep='last')]
    merged.sort_index(inplace=True)
    _write_parquet(merged, path)
    return merged


# ── Core: full daily cache ─────────────────────────────────────────────────────

def _get_full_daily_cache(symbol: str) -> pd.DataFrame | None:
    """
    Return the complete daily OHLCV DataFrame for `symbol`, refreshing if stale.

    On first call: fetches _DAILY_LOOKBACK (~6 years) from Binance and saves.
    On subsequent calls: fetches only the missing delta since last cached bar.
    Returns None on unrecoverable error; returns stale data rather than empty
    when an incremental fetch fails.
    """
    _ensure_dirs()
    path = _daily_path(symbol)

    # Use UTC date so the freshness check is correct regardless of the local
    # system timezone.  The cache is only considered fresh when it already
    # contains today's (UTC) bar, which guarantees that yesterday's bar was
    # written *after* it fully closed at 23:59:59 UTC and therefore carries
    # its correct final close price, not a stale mid-day value.
    _today_utc = datetime.utcnow().date()

    if path.exists():
        cached = _read_parquet(path)
        if cached is not None and not cached.empty:
            last_date = _last_bar_date(cached)
            if last_date and last_date >= _today_utc:
                return cached   # has today's bar -> yesterday is fully closed

            # Incremental update: fetch only the missing window
            days_needed = (_today_utc - last_date).days + 2
            print(f'  {symbol} daily: stale by {days_needed - 2}d — fetching delta…')
            try:
                new_bars = _fetch_daily_binance(symbol, f'{days_needed} days ago UTC')
                if not new_bars.empty:
                    merged = _merge_and_save(cached, new_bars, path)
                    print(f'  {symbol} daily: +{len(new_bars)} bars -> {len(merged)} total')
                    return merged
                return cached
            except Exception as e:
                print(f'  {symbol} daily incremental fetch failed: {e} — using stale cache')
                return cached   # stale data > empty

    # No cache — full history fetch
    print(f'  {symbol} daily: no cache — fetching {_DAILY_LOOKBACK} days from Binance…')
    try:
        df = _fetch_daily_binance(symbol, f'{_DAILY_LOOKBACK} days ago UTC')
        if not df.empty:
            _write_parquet(df, path)
            print(f'  {symbol} daily: {len(df)} bars saved -> {path.name}')
        return df if not df.empty else None
    except Exception as e:
        print(f'  {symbol} daily full fetch failed: {e}')
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def is_cache_fresh(symbol: str, interval: str = 'daily') -> bool:
    """
    Return True if the local cache is up to date.

    Daily  : fresh when last bar date >= yesterday's completed close.
    Hourly : fresh when last bar timestamp is within the past 2 hours.
    """
    _ensure_dirs()
    path = _daily_path(symbol) if interval == 'daily' else _hourly_path(symbol)
    if not path.exists():
        return False
    df = _read_parquet(path)
    if df is None or df.empty:
        return False

    if interval == 'daily':
        last = _last_bar_date(df)
        return last is not None and last >= _yesterday()

    # Hourly
    last_ts = df.index[-1]
    if hasattr(last_ts, 'tzinfo') and last_ts.tzinfo is not None:
        last_ts = last_ts.tz_convert('UTC').tz_localize(None)
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=2)
    return bool(last_ts >= cutoff)


def get_daily_ohlcv(symbol: str, warmup_bars: int = 100) -> pd.DataFrame:
    """
    Return the last `warmup_bars` rows of daily OHLCV for `symbol`.

    First call: fetches ~6 years of history from Binance, saves to parquet.
    Subsequent calls: reads parquet, fetches only missing delta if stale.
    Returns an empty DataFrame on unrecoverable error.
    """
    df = _get_full_daily_cache(symbol)
    if df is None or df.empty:
        return pd.DataFrame(columns=_EMPTY_OHLCV)
    return df.iloc[-warmup_bars:]


def get_daily_ohlcv_range(
    symbol: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Return daily OHLCV for `symbol` filtered to [start_date, end_date] inclusive.

    Uses the local cache, refreshing stale data before slicing.
    """
    df = _get_full_daily_cache(symbol)
    if df is None or df.empty:
        return pd.DataFrame(columns=_EMPTY_OHLCV)

    s = pd.Timestamp(_to_date(start_date))
    e = pd.Timestamp(_to_date(end_date)) + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)]


def get_hourly_ohlcv(
    symbol: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Return hourly OHLCV for `symbol` covering [start_date − 1d, end_date + 2d].

    The generous buffer guarantees T+1 execution-hour bar lookups always succeed
    (signal fires on close of day T, execution at HH:00 UTC on day T+1).

    Cache strategy:
    1. Cache exists and fully covers the requested range -> slice and return.
    2. Cache exists but has a gap -> fetch only the missing portion and merge.
    3. No cache -> fetch the full requested range from Binance and save.
    """
    _ensure_dirs()
    path = _hourly_path(symbol)

    req_start = _to_date(start_date)
    req_end   = _to_date(end_date)
    # Buffer for T+1 execution-hour lookups
    fetch_start = req_start - timedelta(days=1)
    fetch_end   = req_end   + timedelta(days=2)

    if path.exists():
        cached = _read_parquet(path)
        if cached is not None and not cached.empty:
            c_start = _first_bar_date(cached)
            c_end   = _last_bar_date(cached)

            needs_head = c_start is not None and c_start > fetch_start
            needs_tail = c_end   is not None and c_end   < fetch_end

            if not needs_head and not needs_tail:
                # Full coverage — just slice
                return _slice_hourly(cached, fetch_start, fetch_end)

            # Fetch missing tail (most common: cache is current but the requested
            # range extends to dates after the last cached bar)
            if needs_tail:
                days_needed = (fetch_end - c_end).days + 2
                try:
                    tail = _fetch_hourly_binance(symbol, f'{days_needed} days ago UTC')
                    if not tail.empty:
                        cached = _merge_and_save(cached, tail, path)
                        print(f'  {symbol} hourly: +{len(tail)} tail bars -> {len(cached)} total')
                except Exception as e:
                    print(f'  {symbol} hourly tail fetch failed: {e}')

            # Fetch missing head (cache predates the request's start)
            if needs_head:
                try:
                    head = _fetch_hourly_binance(
                        symbol,
                        start_str=fetch_start.strftime('%Y-%m-%d'),
                        end_str=(c_start + timedelta(days=1)).strftime('%Y-%m-%d'),
                    )
                    if not head.empty:
                        cached = _merge_and_save(head, cached, path)
                        print(f'  {symbol} hourly: +{len(head)} head bars -> {len(cached)} total')
                except Exception as e:
                    print(f'  {symbol} hourly head fetch failed: {e}')

            return _slice_hourly(cached, fetch_start, fetch_end)

    # No cache — fetch full requested range from scratch
    print(f'  {symbol} hourly: no cache — fetching {fetch_start} -> {fetch_end} from Binance…')
    try:
        df = _fetch_hourly_binance(
            symbol,
            start_str=fetch_start.strftime('%Y-%m-%d'),
            end_str=(fetch_end + timedelta(days=1)).strftime('%Y-%m-%d'),
        )
        if not df.empty:
            _write_parquet(df, path)
            print(f'  {symbol} hourly: {len(df)} bars saved -> {path.name}')
        return _slice_hourly(df, fetch_start, fetch_end)
    except Exception as e:
        print(f'  {symbol} hourly fetch failed: {e}')
        return pd.DataFrame(columns=_EMPTY_OHLCV)


def _slice_hourly(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Filter a time-indexed DataFrame to [start, end + 1 day]."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_EMPTY_OHLCV)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index <= e)]


def update_all_caches(symbols: list) -> None:
    """
    Ensure daily and hourly caches are up to date for all symbols.

    Fetches only the missing delta — completes in seconds when caches are fresh.
    Designed to be called from a daily cron job (see update_cache.py).
    """
    print(f'\n── Cache update: {len(symbols)} symbol(s) ───────────────────────────')
    total_new_daily  = 0
    total_new_hourly = 0

    for symbol in symbols:
        print(f'\n  [{symbol}]')

        # ── Daily ─────────────────────────────────────────────────────────────
        d_path = _daily_path(symbol)
        before = len(_read_parquet(d_path)) if d_path.exists() else 0

        if is_cache_fresh(symbol, 'daily'):
            df = _read_parquet(d_path)
            n  = len(df) if df is not None else 0
            print(f'    daily  — already fresh ({n} bars, last: {_last_bar_date(df)})')
        else:
            _get_full_daily_cache(symbol)   # triggers incremental update
            df    = _read_parquet(d_path)
            after = len(df) if df is not None else 0
            added = after - before
            total_new_daily += max(added, 0)
            print(f'    daily  — updated (+{added} bars -> {after} total, last: {_last_bar_date(df)})')

        # ── Hourly ────────────────────────────────────────────────────────────
        h_path = _hourly_path(symbol)
        if not h_path.exists():
            print(f'    hourly — no cache (run backfill_cache.py)')
            continue

        cached = _read_parquet(h_path)
        if cached is None or cached.empty:
            print(f'    hourly — cache empty (run backfill_cache.py)')
            continue

        if is_cache_fresh(symbol, 'hourly'):
            print(f'    hourly — already fresh ({len(cached)} bars, last: {_last_bar_date(cached)})')
        else:
            c_end       = _last_bar_date(cached)
            days_needed = (date.today() - c_end).days + 2
            before_h    = len(cached)
            try:
                new_bars = _fetch_hourly_binance(symbol, f'{days_needed} days ago UTC')
                if not new_bars.empty:
                    merged = _merge_and_save(cached, new_bars, h_path)
                    added  = len(merged) - before_h
                    total_new_hourly += max(added, 0)
                    print(f'    hourly — updated (+{added} bars -> {len(merged)} total)')
                else:
                    print(f'    hourly — no new bars from Binance')
            except Exception as e:
                print(f'    hourly — update failed: {e}')

    print(f'\n── Done: +{total_new_daily} daily bars, +{total_new_hourly} hourly bars added ──')
