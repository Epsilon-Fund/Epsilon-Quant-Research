"""Tests for mirror/orderbook.py."""
from __future__ import annotations

import io
import json
from urllib.error import URLError

import pytest

from polymarket.execution.mirror import orderbook


def _fake_response(payload: dict) -> io.BytesIO:
    body = json.dumps(payload).encode("utf-8")
    return io.BytesIO(body)


def test_get_best_price_returns_best_ask_for_buy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "bids": [{"price": "0.42", "size": "100"},
                 {"price": "0.41", "size": "50"}],
        "asks": [{"price": "0.43", "size": "200"},
                 {"price": "0.44", "size": "75"}],
    }
    monkeypatch.setattr(
        orderbook, "urlopen",
        lambda *a, **k: _fake_response(payload),
    )
    assert orderbook.get_best_price("https://clob", "asset-1", "BUY") == 0.43
    assert orderbook.get_best_price("https://clob", "asset-1", "SELL") == 0.42


def test_get_best_price_returns_none_on_url_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*a, **k):
        raise URLError("connection refused")
    monkeypatch.setattr(orderbook, "urlopen", boom)
    assert orderbook.get_best_price("https://clob", "asset-1", "BUY") is None


def test_get_best_price_returns_none_on_empty_book(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orderbook, "urlopen",
        lambda *a, **k: _fake_response({"bids": [], "asks": []}),
    )
    assert orderbook.get_best_price("https://clob", "asset-1", "BUY") is None
    assert orderbook.get_best_price("https://clob", "asset-1", "SELL") is None
