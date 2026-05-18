"""Tests for mirror/clob_http_client.py — substitute Polymarket CLOB HTTP client."""
from __future__ import annotations

import io
import json
import socket
from typing import Any
from urllib.error import URLError

import pytest

from polymarket.execution._kernel.polymarket_adapter import PolymarketOrderRequest
from polymarket.execution.mirror.clob_http_client import (
    ClobHttpClient,
    ClobHttpClientConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**overrides: Any) -> ClobHttpClientConfig:
    base = dict(
        api_url="https://clob.polymarket.com",
        api_key="apikey",
        api_secret="apisecret",
        passphrase="passphrase",
        private_key="0x" + "a" * 64,
        quantity_scale=10_000,
        default_tick_size=0.01,
    )
    base.update(overrides)
    return ClobHttpClientConfig(**base)


class _FakeResponse:
    def __init__(self, *, status: int = 200, body: str = "{}") -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body.encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _capture_signer():
    """Returns (signer_callable, captured_calls_list)."""
    calls: list[dict[str, Any]] = []

    def signer(unsigned: dict[str, object], private_key: str | None) -> dict[str, object]:
        calls.append({"unsigned": dict(unsigned), "private_key": private_key})
        return {"order": dict(unsigned), "signature": "0xfake"}

    return signer, calls


def _request(*, size: str, price: str, token_id: str = "T1") -> PolymarketOrderRequest:
    return PolymarketOrderRequest(
        market_id="M1",
        token_id=token_id,
        side="BUY",
        size=size,
        price=price,
        tif="IOC",
        client_order_id="coid-1",
        expiration_ts=1_700_000_000,
    )


# ---------------------------------------------------------------------------
# create_signed_order — encoding correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoded,scale,expected", [
    ("50000", 10_000, "5"),
    ("111100", 10_000, "11.11"),
    ("129600", 10_000, "12.96"),
    ("500", 10_000, "0.05"),
    ("111110", 10_000, "11.111"),
])
def test_quantity_decode_to_fractional_shares(encoded, scale, expected) -> None:
    signer, calls = _capture_signer()
    client = ClobHttpClient(_config(quantity_scale=scale), signer=signer)
    client.create_signed_order(_request(size=encoded, price="42"))
    assert calls[0]["unsigned"]["size"] == expected


@pytest.mark.parametrize("ticks,tick_size,expected", [
    ("42", 0.01, "0.42"),
    ("99", 0.01, "0.99"),
    ("13", 0.001, "0.013"),
    ("100", 0.01, "1"),
    ("1", 0.01, "0.01"),
])
def test_price_decode_to_dollars(ticks, tick_size, expected) -> None:
    signer, calls = _capture_signer()
    client = ClobHttpClient(_config(), signer=signer)
    client.set_tick_size("T1", tick_size)
    client.create_signed_order(_request(size="50000", price=ticks))
    assert calls[0]["unsigned"]["price"] == expected


def test_default_tick_size_when_unset() -> None:
    signer, calls = _capture_signer()
    client = ClobHttpClient(_config(default_tick_size=0.01), signer=signer)
    # No set_tick_size call; should fall back to default 0.01.
    client.create_signed_order(_request(size="50000", price="42", token_id="UNSEEN"))
    assert calls[0]["unsigned"]["price"] == "0.42"


def test_signed_order_includes_expiration_and_coid() -> None:
    signer, calls = _capture_signer()
    client = ClobHttpClient(_config(), signer=signer)
    client.set_tick_size("T1", 0.01)
    client.create_signed_order(_request(size="50000", price="42"))
    unsigned = calls[0]["unsigned"]
    assert unsigned["client_order_id"] == "coid-1"
    assert unsigned["expiration_ts"] == 1_700_000_000
    assert unsigned["tif"] == "IOC"
    assert unsigned["side"] == "BUY"
    assert unsigned["market_id"] == "M1"
    assert unsigned["token_id"] == "T1"


def test_no_signer_returns_unsigned_envelope() -> None:
    client = ClobHttpClient(_config(), signer=None)
    client.set_tick_size("T1", 0.01)
    out = client.create_signed_order(_request(size="50000", price="42"))
    assert "order" in out
    assert out["order"]["size"] == "5"
    assert out["order"]["price"] == "0.42"


# ---------------------------------------------------------------------------
# HTTP transport: headers, status, errors
# ---------------------------------------------------------------------------


def test_headers_include_api_credentials() -> None:
    captured = {}

    def fake_urlopen(request, timeout):  # noqa: ARG001
        # urllib normalises header names to title-case; keep the dict
        # straight from request.headers (already title-cased).
        captured["headers"] = dict(request.headers)
        return _FakeResponse(status=200, body='{"status":"ACCEPTED","order_id":"v1"}')

    client = ClobHttpClient(_config(api_key="K", api_secret="S", passphrase="P"),
                            urlopen_fn=fake_urlopen)
    client.submit_order({"order": {}}, timeout_ms=500)
    headers = captured["headers"]
    # urllib title-cases header names; X-API-KEY becomes X-Api-Key.
    assert headers.get("X-api-key") == "K"
    assert headers.get("X-api-secret") == "S"
    assert headers.get("X-api-passphrase") == "P"


def test_submit_order_http_200() -> None:
    body = json.dumps({"status": "ACCEPTED", "order_id": "venue-1"})
    client = ClobHttpClient(_config(), urlopen_fn=lambda *a, **k: _FakeResponse(status=200, body=body))
    response = client.submit_order({"order": {}}, timeout_ms=500)
    assert response["status"] == "ACCEPTED"
    assert response["order_id"] == "venue-1"
    assert response["http_status"] == 200


def test_submit_order_http_400_carries_status_marker() -> None:
    body = json.dumps({"reason": "invalid order"})
    client = ClobHttpClient(_config(), urlopen_fn=lambda *a, **k: _FakeResponse(status=400, body=body))
    response = client.submit_order({"order": {}}, timeout_ms=500)
    # No "status" key in payload → augmented to HTTP_400.
    assert response["status"] == "HTTP_400"
    assert response["http_status"] == 400
    assert response["reason"] == "invalid order"


def test_submit_order_socket_timeout_raises_timeout_error() -> None:
    def boom(*a, **k):
        raise socket.timeout("read timeout")

    client = ClobHttpClient(_config(), urlopen_fn=boom)
    with pytest.raises(TimeoutError):
        client.submit_order({"order": {}}, timeout_ms=500)


def test_submit_order_url_error_raises_os_error() -> None:
    def boom(*a, **k):
        raise URLError("connection refused")

    client = ClobHttpClient(_config(), urlopen_fn=boom)
    with pytest.raises(OSError):
        client.submit_order({"order": {}}, timeout_ms=500)


# ---------------------------------------------------------------------------
# cancel + open orders + updates
# ---------------------------------------------------------------------------


def test_cancel_order_posts_correct_body() -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout):  # noqa: ARG001
        captured["url"] = request.full_url
        captured["body"] = request.data.decode("utf-8") if request.data else ""
        return _FakeResponse(status=200, body='{"status":"CANCELED"}')

    client = ClobHttpClient(_config(), urlopen_fn=fake_urlopen)
    response = client.cancel_order(client_order_id="coid-x", venue_order_id="v-x", timeout_ms=500)
    assert response["status"] == "CANCELED"
    body_decoded = json.loads(captured["body"])
    assert body_decoded == {"client_order_id": "coid-x", "order_id": "v-x"}


def test_get_open_orders_parses_list() -> None:
    body = json.dumps([
        {"client_order_id": "a", "status": "OPEN"},
        {"client_order_id": "b", "status": "OPEN"},
    ])
    client = ClobHttpClient(_config(), urlopen_fn=lambda *a, **k: _FakeResponse(status=200, body=body))
    items = client.get_open_orders(timeout_ms=500)
    assert len(items) == 2
    assert items[0]["client_order_id"] == "a"


def test_get_open_orders_4xx_raises() -> None:
    client = ClobHttpClient(_config(), urlopen_fn=lambda *a, **k: _FakeResponse(status=403, body="{}"))
    with pytest.raises(OSError):
        client.get_open_orders(timeout_ms=500)


def test_get_order_updates_parses_envelope() -> None:
    body = json.dumps({"updates": [{"client_order_id": "a", "status": "FILLED"}]})
    client = ClobHttpClient(_config(), urlopen_fn=lambda *a, **k: _FakeResponse(status=200, body=body))
    items = client.get_order_updates(since_sequence=42, limit=10, timeout_ms=500)
    assert len(items) == 1
    assert items[0]["status"] == "FILLED"


def test_get_order_updates_returns_empty_after_max_retries() -> None:
    def always_timeout(*a, **k):
        raise socket.timeout("timeout")

    cfg = _config(updates_max_retries=2, updates_retry_base_ms=0)
    client = ClobHttpClient(cfg, urlopen_fn=always_timeout)
    items = client.get_order_updates(since_sequence=None, limit=5, timeout_ms=100)
    assert items == tuple()


# ---------------------------------------------------------------------------
# Round-trip preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shares,scale,expected", [
    (5.0, 10_000, "5"),
    (11.11, 10_000, "11.11"),
    (12.96, 10_000, "12.96"),
    (0.05, 10_000, "0.05"),
    (11.111, 10_000, "11.111"),
])
def test_quantity_round_trip_preserves_decimals(shares, scale, expected) -> None:
    """Encode shares as int(shares*scale), pass through, decode, verify."""
    encoded = int(round(shares * scale))
    signer, calls = _capture_signer()
    client = ClobHttpClient(_config(quantity_scale=scale), signer=signer)
    client.create_signed_order(_request(size=str(encoded), price="42"))
    assert calls[0]["unsigned"]["size"] == expected


def test_validation_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="api_url"):
        ClobHttpClient(_config(api_url=""))
    with pytest.raises(ValueError, match="quantity_scale"):
        ClobHttpClient(_config(quantity_scale=0))
    with pytest.raises(ValueError, match="default_tick_size"):
        ClobHttpClient(_config(default_tick_size=0))


def test_set_tick_size_rejects_non_positive() -> None:
    client = ClobHttpClient(_config())
    with pytest.raises(ValueError, match="tick_size"):
        client.set_tick_size("T1", 0)
