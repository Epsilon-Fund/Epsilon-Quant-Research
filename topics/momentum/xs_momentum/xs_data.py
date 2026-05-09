"""
xs_data.py — Multi-asset data loader for cross-sectional momentum research.

Pulls daily OHLCV from Binance for a configurable coin list and returns
data in the same format used throughout the rest of this repo:
  - Per-coin DataFrames: DatetimeIndex named 'Time', float columns Open/High/Low/Close/Volume
  - Close panel: wide DataFrame indexed by date, one column per coin (symbol without USDT)
  - Returns panel: daily pct_change of the close panel, for cross-sectional ranking

Usage (from a notebook next to this file):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from xs_data import load_xs_data, build_close_panel, build_returns_panel

    data    = load_xs_data()
    closes  = build_close_panel(data)
    returns = build_returns_panel(data)
"""

import os
import sys

# ── resolve infrastructure/data so we can reuse the existing client helpers ──
_HERE  = os.path.dirname(os.path.abspath(__file__))
_INFRA = os.path.abspath(os.path.join(_HERE, '..', '..', '..', 'infrastructure', 'data'))
if _INFRA not in sys.path:
    sys.path.insert(0, _INFRA)

import pandas as pd
from binance_client import get_binance_client, get_data


# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_COINS = [
    'BTC', 'ETH', 'SOL', 'BNB', 'XRP',
    'DOGE', 'ADA', 'AVAX', 'LINK', 'MATIC', 'TRX',
    'SUI', 'TON', #'HYPE',
    'SHIB', 'XLM',
]

DEFAULT_QUOTE    = 'USDT'
DEFAULT_INTERVAL = '1d'
DEFAULT_LOOKBACK = 1500   # days; covers ~4 years of daily bars for walk-forward


# ── public API ────────────────────────────────────────────────────────────────

def load_xs_data(
    coins    = None,
    quote    = DEFAULT_QUOTE,
    interval = DEFAULT_INTERVAL,
    lookback = DEFAULT_LOOKBACK,
    client   = None,
    verbose  = True,
):
    """
    Pull historical OHLCV from Binance for every coin in `coins`.

    Parameters
    ----------
    coins    : list of base-asset strings, e.g. ['BTC', 'ETH'].
               Defaults to DEFAULT_COINS.
    quote    : quote currency appended to each coin, e.g. 'USDT'.
    interval : Binance kline interval string — '1d', '4h', '1h', etc.
    lookback : history in days passed to get_historical_klines.
               Must be >= train_bars + test_bars for walk-forward.
    client   : existing Binance Client instance; one is created if None.
    verbose  : print a status line per coin.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are bare coin names (e.g. 'BTC', 'ETH').
        Each DataFrame has a DatetimeIndex named 'Time' and float columns
        Open, High, Low, Close, Volume — identical to get_data() output.
        Any coin that fails to fetch is skipped with a warning.
    """
    if coins is None:
        coins = DEFAULT_COINS

    if client is None:
        client = get_binance_client()

    data = {}
    for coin in coins:
        symbol = coin + quote
        try:
            df = get_data(client, symbol, interval, lookback)
            data[coin] = df
            if verbose:
                print(f'  {symbol:<12} {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} bars)')
        except Exception as exc:
            print(f'  WARNING: could not fetch {symbol} — {exc}')

    return data


def build_close_panel(data):
    """
    Pivot per-coin DataFrames into a wide Close-price DataFrame.

    Parameters
    ----------
    data : dict[str, pd.DataFrame] as returned by load_xs_data()

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (union of all coins, forward-filled so no gaps),
        one column per coin (bare name, e.g. 'BTC').
        NaN where a coin had no bar on that date.
    """
    close_series = {
        coin: df['Close'].rename(coin)
        for coin, df in data.items()
    }
    panel = pd.concat(close_series, axis=1).sort_index()
    return panel


def build_returns_panel(data, periods=1):
    """
    Build a daily (or N-period) return panel for cross-sectional ranking.

    Parameters
    ----------
    data    : dict[str, pd.DataFrame] as returned by load_xs_data()
    periods : lookback for pct_change (1 = 1-bar return, 20 = monthly, etc.)

    Returns
    -------
    pd.DataFrame
        Same shape as build_close_panel(); values are fractional returns.
    """
    closes = build_close_panel(data)
    return closes.pct_change(periods=periods)


def build_momentum_panel(data, lookback_bars, skip_bars=1):
    """
    Compute the cross-sectional momentum signal for each bar:
      momentum = return over [t - lookback_bars - skip_bars : t - skip_bars]

    The `skip_bars` lag (default 1) avoids short-term reversal contamination
    that is common in daily crypto data.

    Parameters
    ----------
    data          : dict[str, pd.DataFrame] as returned by load_xs_data()
    lookback_bars : formation window length in bars (e.g. 20 = ~1 month daily)
    skip_bars     : recent bars to exclude from the formation window (default 1)

    Returns
    -------
    pd.DataFrame
        Same index / columns as close panel.
        Values are fractional returns over the formation window.
    """
    closes = build_close_panel(data)

    lagged  = closes.shift(skip_bars)
    formed  = closes.shift(skip_bars + lookback_bars)
    momentum = lagged / formed - 1

    return momentum


def align_xs_data(data, dropna_threshold=0.5):
    """
    Align all coin DataFrames to a common DatetimeIndex.

    Coins missing more than `dropna_threshold` fraction of bars across the
    common date range are dropped entirely with a warning.

    Parameters
    ----------
    data               : dict[str, pd.DataFrame]
    dropna_threshold   : fraction of NaN rows above which a coin is dropped

    Returns
    -------
    dict[str, pd.DataFrame]
        Aligned DataFrames, all sharing the same DatetimeIndex.
    """
    panel = build_close_panel(data)
    common_index = panel.index

    nan_fracs = panel.isna().mean()
    keep = nan_fracs[nan_fracs <= dropna_threshold].index.tolist()
    drop = nan_fracs[nan_fracs >  dropna_threshold].index.tolist()

    if drop:
        print(f'  align_xs_data: dropping {drop} (>{dropna_threshold*100:.0f}% NaN)')

    aligned = {}
    for coin in keep:
        df = data[coin].reindex(common_index)
        aligned[coin] = df

    return aligned


# ── quick self-test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading cross-sectional data...')
    data    = load_xs_data(lookback=365)

    closes  = build_close_panel(data)
    rets    = build_returns_panel(data)
    mom     = build_momentum_panel(data, lookback_bars=20, skip_bars=1)

    print(f'\nClose panel:   {closes.shape}  ({closes.index[0].date()} → {closes.index[-1].date()})')
    print(f'Returns panel: {rets.shape}')
    print(f'Momentum panel (20d, skip 1): {mom.shape}')
    print(f'\nCoins loaded: {list(data.keys())}')
    print(f'\nLast row of momentum panel:\n{mom.iloc[-1].sort_values(ascending=False).round(4)}')
