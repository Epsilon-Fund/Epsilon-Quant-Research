"""
universe_filter.py — Lightweight universe filter that reads from the local cache.

No API calls.  Reads the three parquet files written by universe_cache.py.

All data is from Binance USDT perpetual futures — used for ranking signal,
universe filtering, AND backtest PnL.

Typical usage (from a notebook):
    from universe_filter import load_cache, get_universe, precompute_avg_volume

    close, volume, meta = load_cache()

    # At each rebalance bar (strictly no lookahead):
    coins = get_universe(
        as_of_date = dates[r - 1],   # signal formation date, not execution date
        volume     = volume,
        meta       = meta,
    )
    # Returns e.g. ['BTC', 'ETH', 'SOL', 'BNB', ...]

    # Use close[coins].pct_change(J).shift(1) for signal
    # Use close[coins].pct_change()           for execution / PnL
"""

import os
import warnings
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR  = os.path.join(_HERE, 'cache')
CLOSE_PATH = os.path.join(CACHE_DIR, 'close.parquet')
VOL_PATH   = os.path.join(CACHE_DIR, 'volume.parquet')
META_PATH  = os.path.join(CACHE_DIR, 'meta.parquet')

# ── configurable defaults ─────────────────────────────────────────────────────
DEFAULT_MIN_AVG_VOLUME   = 50_000_000    # $50 M 30-day avg daily USDT perp volume
DEFAULT_MIN_AGE_DAYS     = 180           # listed ≥ 6 months before query date
DEFAULT_TOP_N            = 15
DEFAULT_VOLUME_WINDOW    = 30            # days for rolling avg volume
DEFAULT_EXCLUDE_STABLE   = frozenset({
    'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'FDUSD', 'USDD', 'PYUSD',
})
DEFAULT_EXCLUDE_WRAPPED  = frozenset({
    'WBTC', 'WETH', 'WBETH', 'STETH', 'CBETH',
})


# ── cache loader ──────────────────────────────────────────────────────────────

def load_cache():
    """
    Load all three cache files into memory.

    Returns
    -------
    close  : pd.DataFrame — DatetimeIndex × coin, perp Close prices
    volume : pd.DataFrame — DatetimeIndex × coin, perp daily USDT quote volume
    meta   : pd.DataFrame — index=base_asset, cols: listing_date

    Raises
    ------
    FileNotFoundError if any cache file is missing.  Run universe_cache.py first.
    """
    missing = [p for p in (CLOSE_PATH, VOL_PATH, META_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f'Cache file(s) not found:\n' +
            '\n'.join(f'  {p}' for p in missing) +
            '\n\nRun  python universe_cache.py  to build the cache first.'
        )
    close  = pd.read_parquet(CLOSE_PATH)
    volume = pd.read_parquet(VOL_PATH)
    meta   = pd.read_parquet(META_PATH)
    return close, volume, meta


# ── pre-computation helper ────────────────────────────────────────────────────

def precompute_avg_volume(volume, volume_window=DEFAULT_VOLUME_WINDOW):
    """
    Pre-compute the rolling N-day avg daily USDT perp volume panel.

    Use this when calling get_universe() many times across the same `volume`
    panel — typically inside a walk-forward backtest where get_universe() runs
    at every rebalance bar across hundreds of Optuna trials.  Pass the result
    as get_universe(..., avg_vol_panel=...) to skip the per-call rolling mean
    (~10× speedup on dynamic-universe walk-forward runs).

    Returns
    -------
    pd.DataFrame
        Same shape as `volume`; each cell is the rolling-window avg daily
        USDT volume ending at that date (`min_periods=1` so early bars aren't NaN).
    """
    return volume.rolling(volume_window, min_periods=1).mean()


# ── universe query ────────────────────────────────────────────────────────────

def get_universe(
    as_of_date,
    volume,
    meta,
    min_avg_volume      = DEFAULT_MIN_AVG_VOLUME,
    min_age_days        = DEFAULT_MIN_AGE_DAYS,
    top_n               = DEFAULT_TOP_N,
    volume_window       = DEFAULT_VOLUME_WINDOW,
    exclude_stablecoins = DEFAULT_EXCLUDE_STABLE,
    exclude_wrapped     = DEFAULT_EXCLUDE_WRAPPED,
    avg_vol_panel       = None,
):
    """
    Return the eligible universe at `as_of_date` using only data available up
    to and including that date — strictly no lookahead bias.

    Filters applied (in order):
      1. listing_date ≤ as_of_date − min_age_days
      2. Not in exclude_stablecoins or exclude_wrapped
      3. volume_window-day avg daily USDT perp volume ≥ min_avg_volume
      4. Ranked by avg volume; return top_n

    Performance: pass `avg_vol_panel = precompute_avg_volume(volume)` to skip
    the per-call rolling mean — ~10× speedup for backtest loops.

    Parameters
    ----------
    as_of_date     : date-like — signal formation date (data up to here used)
    volume         : pd.DataFrame — from load_cache(); ignored when avg_vol_panel given
    meta           : pd.DataFrame — from load_cache(); index = base asset
    avg_vol_panel  : pd.DataFrame or None — pre-computed rolling avg volume.
                     When provided, `volume` and `volume_window` are ignored.

    Returns
    -------
    list[str]
        Base asset names sorted by descending avg volume, e.g. ['BTC', 'ETH', 'SOL']
        Returns [] if no coins pass all filters (with a warning).
    """
    as_of_date = pd.Timestamp(as_of_date)
    cutoff     = as_of_date - pd.Timedelta(days=min_age_days)
    exclude    = set(exclude_stablecoins) | set(exclude_wrapped)

    # ── vectorised listing-age + exclusion filter (no .copy) ──────────────────
    listing = meta['listing_date']
    mask = (~meta.index.isin(exclude)) & listing.notna() & (listing <= cutoff)
    eligible_idx = meta.index[mask]

    if len(eligible_idx) == 0:
        warnings.warn(
            f'get_universe: no coins pass listing-age filter at {as_of_date.date()}. '
            f'min_age_days={min_age_days}.'
        )
        return []

    # ── volume average (fast path if pre-computed) ────────────────────────────
    if avg_vol_panel is not None:
        idx = avg_vol_panel.index
        if as_of_date in idx:
            avg_vol = avg_vol_panel.loc[as_of_date]
        else:
            valid = idx[idx <= as_of_date]
            if len(valid) == 0:
                warnings.warn(f'get_universe: no avg-vol data up to {as_of_date.date()}.')
                return []
            avg_vol = avg_vol_panel.loc[valid[-1]]
    else:
        hist = volume.loc[volume.index <= as_of_date]
        if len(hist) == 0:
            warnings.warn(f'get_universe: no volume data up to {as_of_date.date()}.')
            return []
        if len(hist) < volume_window:
            warnings.warn(
                f'get_universe: only {len(hist)} bars of volume up to {as_of_date.date()}, '
                f'fewer than volume_window={volume_window}. Average computed over available bars.'
            )
        avg_vol = hist.tail(volume_window).mean()

    # ── volume filter + ranking ───────────────────────────────────────────────
    avg_vol    = avg_vol.dropna()
    candidates = eligible_idx.intersection(avg_vol.index)
    if len(candidates) == 0:
        return []

    avg_in_universe = avg_vol[candidates]
    qualified       = avg_in_universe[avg_in_universe >= min_avg_volume]

    if len(qualified) == 0:
        warnings.warn(
            f'get_universe: no coins pass volume filter at {as_of_date.date()}. '
            f'min_avg_volume=${min_avg_volume:,.0f}.'
        )
        return []

    return qualified.sort_values(ascending=False).head(top_n).index.tolist()


# ── convenience: print snapshot ──────────────────────────────────────────────

def universe_summary(as_of_date, volume, meta, **kwargs):
    """Print a readable universe snapshot. Accepts same kwargs as get_universe()."""
    coins    = get_universe(as_of_date, volume, meta, **kwargs)
    as_of_ts = pd.Timestamp(as_of_date)
    vol_win  = kwargs.get('volume_window', DEFAULT_VOLUME_WINDOW)
    hist     = volume.loc[volume.index <= as_of_ts]
    avg_vol  = hist.tail(vol_win).mean()

    print(f'\nUniverse at {as_of_ts.date()}  (top {len(coins)} by {vol_win}-day avg USDT perp volume)')
    print(f"{'Coin':<10} {'Avg daily vol (USDT M)':>22}")
    print(f"{'─'*10} {'─'*22}")
    for c in coins:
        vol_m = avg_vol.get(c, float('nan')) / 1e6
        print(f'{c:<10} {vol_m:>21.1f}M')


# ── quick self-test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    close, volume, meta = load_cache()
    print(f'Cache loaded:')
    print(f'  close  : {close.shape}')
    print(f'  volume : {volume.shape}')
    print(f'  meta   : {meta.shape}')

    today = pd.Timestamp.utcnow().normalize()
    universe_summary(today, volume, meta)
