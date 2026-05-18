"""Thin price-fetch helper for Polymarket CLOB.

Used by mirror_engine to determine the submission price for a
candidate order. Will eventually be replaced by polymarket-apis's
get_order_book_midpoint when the project moves to Python 3.12.
For now, inline implementation matches midas's transport style.
"""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

_TIMEOUT_S: float = 5.0


def get_best_price(clob_url: str, asset_id: str, side: str) -> float | None:
    """Return current best price for an asset from Polymarket CLOB.

    side="BUY"  → best ask (the price we'd pay to buy now)
    side="SELL" → best bid (the price we'd receive to sell now)

    Returns None on any failure (network error, empty book, malformed
    response). Never raises.
    """
    base = clob_url.rstrip("/")
    qs = urlencode({"token_id": asset_id})
    url = f"{base}/book?{qs}"
    try:
        with urlopen(url, timeout=_TIMEOUT_S) as resp:
            body = resp.read()
        data = json.loads(body)
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError, ValueError):
        return None

    try:
        if side == "BUY":
            level = data["asks"][0]
        elif side == "SELL":
            level = data["bids"][0]
        else:
            return None
        return float(level["price"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
