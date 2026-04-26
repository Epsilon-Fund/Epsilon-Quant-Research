"""
Shared Binance client helpers.

Single source of truth for live-price fetching across all dashboards.
The batched fetcher uses Binance's `/api/v3/ticker/price` (no symbol param)
to retrieve every ticker in one REST call, instead of one call per symbol.
"""

import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_live_prices() -> dict:
    """
    Return a {symbol: price} dict containing every Binance spot ticker.

    One REST call covers all coins across momentum, statarb, and bbbreakout
    dashboards.  Cached 60 s in the Streamlit process so repeated calls
    within a render cycle (and across tabs) hit the in-memory cache.

    Returns an empty dict on any error so callers can fall back gracefully.
    """
    try:
        from binance_client import get_binance_client
        client  = get_binance_client()
        tickers = client.get_all_tickers()
        return {t['symbol']: float(t['price']) for t in tickers}
    except Exception as e:
        print(f"  fetch_all_live_prices failed: {e}")
        return {}


def get_live_prices(symbols) -> dict:
    """
    Convenience wrapper: fetch all tickers once, return only the symbols asked for.

    Calling this from multiple dashboards in the same render still results in
    exactly one batched fetch thanks to the cache on fetch_all_live_prices.
    """
    all_prices = fetch_all_live_prices()
    return {sym: all_prices.get(sym) for sym in symbols}
