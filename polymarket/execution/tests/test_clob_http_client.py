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
        api_secret="YXBpc2VjcmV0",
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


def test_submit_uses_l2_headers_and_polymarket_envelope() -> None:
    """Regression for the live "missing address header" rejection: submits must carry
    POLY_* L2 headers (not the legacy X-API-*) and the py-clob wire envelope
    {"order": …, "owner": api_key, "orderType": …, "postOnly": false}, with the POSTed
    bytes exactly the compact JSON the HMAC was computed over."""
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout):  # noqa: ARG001
        captured["headers"] = dict(request.headers)
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["data"] = request.data
        return _FakeResponse(status=200, body='{"success":true,"status":"live","orderID":"0xv1"}')

    client = ClobHttpClient(_config(), urlopen_fn=fake_urlopen)
    signed = {"maker": "0xabc", "signature": "0xfake",
              "_order_type": "FAK", "_client_order_id": "coid-1"}
    out = client.submit_order(signed, timeout_ms=500)

    headers = captured["headers"]
    assert "Poly_address" in headers and "Poly_signature" in headers
    assert headers["Poly_api_key"] == "apikey"
    assert "X-api-key" not in headers
    assert captured["url"].endswith("/order")
    assert captured["method"] == "POST"
    body = json.loads(captured["data"].decode("utf-8"))
    assert body == {"order": {"maker": "0xabc", "signature": "0xfake"},
                    "owner": "apikey", "orderType": "FAK", "postOnly": False}
    # reserved side-channel keys never reach the wire
    assert "_order_type" not in json.dumps(body) and "_client_order_id" not in json.dumps(body)
    # bytes are the exact compact serialization the HMAC covers
    assert captured["data"] == json.dumps(body, separators=(",", ":"),
                                          ensure_ascii=False).encode("utf-8")
    # venue response remap: orderID → order_id (kernel reads order_id/id)
    assert out["order_id"] == "0xv1"
    assert out["status"] == "live"


def test_create_signed_order_attaches_order_type_and_coid() -> None:
    signer, _calls = _capture_signer()
    client = ClobHttpClient(_config(), signer=signer)
    client.set_tick_size("T1", 0.01)
    out = client.create_signed_order(_request(size="50000", price="42"))  # tif=IOC
    assert out["_order_type"] == "FAK"     # IOC is not a Polymarket wire type
    assert out["_client_order_id"] == "coid-1"


def test_submit_error_response_remaps_error_msg() -> None:
    body = json.dumps({"success": False, "errorMsg": "not enough balance / allowance"})
    client = ClobHttpClient(_config(),
                            urlopen_fn=lambda *a, **k: _FakeResponse(status=400, body=body))
    out = client.submit_order({"order": {}}, timeout_ms=500)
    assert out["error"] == "not enough balance / allowance"
    assert out["http_status"] == 400


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


def test_cancel_order_deletes_order_with_order_id_body() -> None:
    """Polymarket cancels via DELETE /order {"orderID": …} with L2 headers — the old
    POST /cancel {"order_id": …} was never a real venue endpoint."""
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout):  # noqa: ARG001
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.headers)
        captured["body"] = request.data.decode("utf-8") if request.data else ""
        return _FakeResponse(status=200, body='{"canceled":["v-x"],"not_canceled":{}}')

    client = ClobHttpClient(_config(), urlopen_fn=fake_urlopen)
    response = client.cancel_order(client_order_id="coid-x", venue_order_id="v-x", timeout_ms=500)
    assert captured["method"] == "DELETE"
    assert captured["url"].endswith("/order")
    assert "Poly_address" in captured["headers"]
    assert json.loads(captured["body"]) == {"orderID": "v-x"}
    assert response["status"] == "CANCELED"


def test_cancel_not_canceled_maps_to_not_found() -> None:
    body = '{"canceled":[],"not_canceled":{"v-x":"order not found"}}'
    client = ClobHttpClient(_config(),
                            urlopen_fn=lambda *a, **k: _FakeResponse(status=200, body=body))
    response = client.cancel_order(venue_order_id="v-x", timeout_ms=500)
    assert response["status"] == "NOT_FOUND"


def test_cancel_without_venue_id_falls_back_to_cancel_all() -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout):  # noqa: ARG001
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        return _FakeResponse(status=200, body='{"canceled":[]}')

    client = ClobHttpClient(_config(), urlopen_fn=fake_urlopen)
    response = client.cancel_order(client_order_id="coid-only", timeout_ms=500)
    assert captured["method"] == "DELETE"
    assert captured["url"].endswith("/cancel-all")
    assert response["fallback"] == "cancel_all"


def test_open_orders_annotates_client_id_from_submit_ack() -> None:
    """Polymarket /data/orders items carry no client ids; the kernel's reconcile can
    only attribute items via client_order_id, learned from our own submit acks."""
    responses = iter([
        _FakeResponse(status=200, body='{"success":true,"status":"live","orderID":"0xv9"}'),
        _FakeResponse(status=200,
                      body='{"data":[{"id":"0xv9","status":"LIVE"}],"next_cursor":"LTE="}'),
    ])
    client = ClobHttpClient(_config(), urlopen_fn=lambda *a, **k: next(responses))
    client.submit_order({"order": {}, "_client_order_id": "coid-9"}, timeout_ms=500)
    items = client.get_open_orders(timeout_ms=500)
    assert items[0]["client_order_id"] == "coid-9"


class _FakeGateway:
    """Captures gateway calls; scripted responses per op (V2 order path)."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses = responses or {}

    def _resp(self, op: str, default: dict[str, Any]) -> dict[str, Any]:
        return self.responses.get(op, default)

    def create_limit_order(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("create_limit_order", kw))
        return self._resp("create_limit_order",
                          {"ok": True, "handle": "h1", "order": {"maker": "0xm"}})

    def post_order(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("post_order", kw))
        return self._resp("post_order",
                          {"ok": True, "order_id": "0xv2", "status": "live"})

    def cancel_order(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("cancel_order", kw))
        return self._resp("cancel_order",
                          {"ok": True, "canceled": [kw.get("order_id")], "not_canceled": {}})

    def cancel_all(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("cancel_all", kw))
        return self._resp("cancel_all", {"ok": True, "canceled": [], "not_canceled": {}})


def test_gateway_sign_submit_maps_v2_flow() -> None:
    """V2 path: sign via SDK gateway (decoded decimal strings in), post via handle,
    kernel-shaped ack out (order_id/status), venue-id→coid correlation recorded."""
    gw = _FakeGateway()
    client = ClobHttpClient(_config(), gateway=gw)
    client.set_tick_size("T1", 0.001)
    req = PolymarketOrderRequest(
        market_id="M1", token_id="T1", side="BUY", size="50000", price="232",
        tif="GTC", client_order_id="coid-7", expiration_ts=None)
    signed = client.create_signed_order(req)
    assert signed["_handle"] == "h1" and signed["_client_order_id"] == "coid-7"
    op, kw = gw.calls[0]
    assert op == "create_limit_order"
    assert kw == {"token_id": "T1", "price": "0.232", "size": "5", "side": "BUY"}

    out = client.submit_order(signed, timeout_ms=1500)
    assert out["order_id"] == "0xv2" and out["status"] == "live"
    # correlation available for open-orders annotation
    assert client._coid_by_venue_id["0xv2"] == "coid-7"  # noqa: SLF001


def test_gateway_non_gtc_fails_closed() -> None:
    client = ClobHttpClient(_config(), gateway=_FakeGateway())
    client.set_tick_size("T1", 0.001)
    with pytest.raises(ValueError):
        client.create_signed_order(_request(size="50000", price="42"))  # tif=IOC


def test_gateway_submit_timeout_and_transport_map_to_ambiguity() -> None:
    for kind, exc in (("timeout", TimeoutError), ("transport", OSError)):
        gw = _FakeGateway({"post_order": {"ok": False, "kind": kind, "error": "x"}})
        client = ClobHttpClient(_config(), gateway=gw)
        with pytest.raises(exc):
            client.submit_order({"order": {}, "_handle": "h1", "_client_order_id": "c"},
                                timeout_ms=1500)


def test_gateway_submit_rejection_is_clean_nack() -> None:
    gw = _FakeGateway({"post_order": {"ok": False, "kind": "rejected",
                                      "error": "not enough balance"}})
    client = ClobHttpClient(_config(), gateway=gw)
    out = client.submit_order({"order": {}, "_handle": "h1"}, timeout_ms=1500)
    assert out["status"] == "REJECTED"
    assert out["error"] == "not enough balance"


def test_gateway_cancel_maps_not_found_and_fallback() -> None:
    gw = _FakeGateway({"cancel_order": {"ok": True, "canceled": [],
                                        "not_canceled": {"0xv2": "order not found"}}})
    client = ClobHttpClient(_config(), gateway=gw)
    out = client.cancel_order(venue_order_id="0xv2", timeout_ms=500)
    assert out["status"] == "NOT_FOUND"

    gw2 = _FakeGateway()
    client2 = ClobHttpClient(_config(), gateway=gw2)
    out2 = client2.cancel_order(client_order_id="coid-only", timeout_ms=500)
    assert gw2.calls[0][0] == "cancel_all"
    assert out2["fallback"] == "cancel_all"


def test_open_orders_pagination_follows_next_cursor() -> None:
    responses = iter([
        _FakeResponse(status=200,
                      body='{"data":[{"id":"a","client_order_id":"c-a"}],"next_cursor":"NX=="}'),
        _FakeResponse(status=200,
                      body='{"data":[{"id":"b","client_order_id":"c-b"}],"next_cursor":"LTE="}'),
    ])
    client = ClobHttpClient(_config(), urlopen_fn=lambda *a, **k: next(responses))
    items = client.get_open_orders(timeout_ms=500)
    assert [i["id"] for i in items] == ["a", "b"]


def test_get_open_orders_parses_list() -> None:
    body = json.dumps([
        {"client_order_id": "a", "status": "OPEN"},
        {"client_order_id": "b", "status": "OPEN"},
    ])
    client = ClobHttpClient(_config(), urlopen_fn=lambda *a, **k: _FakeResponse(status=200, body=body))
    items = client.get_open_orders(timeout_ms=500)
    assert len(items) == 2
    assert items[0]["client_order_id"] == "a"


def test_get_open_orders_uses_data_orders_endpoint_and_l2_headers() -> None:
    captured: dict[str, Any] = {}
    body = json.dumps({"data": [], "next_cursor": "LTE="})

    def fake_urlopen(request, timeout):  # noqa: ARG001
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        return _FakeResponse(status=200, body=body)

    client = ClobHttpClient(_config(), urlopen_fn=fake_urlopen)
    assert client.get_open_orders(timeout_ms=500) == tuple()
    assert captured["url"] == (
        "https://clob.polymarket.com/data/orders?next_cursor=MA%3D%3D"
    )
    headers = captured["headers"]
    assert "Poly_address" in headers
    assert "Poly_signature" in headers
    assert "Poly_timestamp" in headers
    assert headers["Poly_api_key"] == "apikey"
    assert headers["Poly_passphrase"] == "passphrase"
    assert "X-api-key" not in headers


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


# ---------------------------------------------------------------------------
# NegRisk side channel
# ---------------------------------------------------------------------------


def test_set_neg_risk_stores_per_asset_flag() -> None:
    client = ClobHttpClient(_config())
    client.set_neg_risk("asset-A", True)
    client.set_neg_risk("asset-B", False)
    assert client._neg_risk_by_token["asset-A"] is True
    assert client._neg_risk_by_token["asset-B"] is False


def test_create_signed_order_passes_neg_risk_true_through_unsigned() -> None:
    seen: dict[str, object] = {}

    def fake_signer(unsigned, pk):
        seen.update(unsigned)
        return {"order": dict(unsigned)}

    client = ClobHttpClient(_config(), signer=fake_signer)
    client.set_tick_size("asset-NR", 0.01)
    client.set_neg_risk("asset-NR", True)
    request = PolymarketOrderRequest(
        market_id="cond-NR", token_id="asset-NR", side="BUY",
        size="50000", price="42", tif="IOC",
        client_order_id="coid-x", expiration_ts=None,
    )
    client.create_signed_order(request)
    assert seen["_neg_risk"] is True
    assert seen["_tick_size"] == 0.01


def test_create_signed_order_passes_neg_risk_false_through_unsigned() -> None:
    seen: dict[str, object] = {}

    def fake_signer(unsigned, pk):
        seen.update(unsigned)
        return {"order": dict(unsigned)}

    client = ClobHttpClient(_config(), signer=fake_signer)
    client.set_tick_size("asset-B", 0.01)
    client.set_neg_risk("asset-B", False)
    request = PolymarketOrderRequest(
        market_id="cond-B", token_id="asset-B", side="SELL",
        size="100000", price="42", tif="IOC",
        client_order_id="coid-y", expiration_ts=None,
    )
    client.create_signed_order(request)
    assert seen["_neg_risk"] is False


def test_create_signed_order_warns_if_neg_risk_unset(capsys) -> None:
    seen: dict[str, object] = {}

    def fake_signer(unsigned, pk):
        seen.update(unsigned)
        return {"order": dict(unsigned)}

    client = ClobHttpClient(_config(), signer=fake_signer)
    # NB: deliberately NOT calling set_neg_risk for this token_id.
    client.set_tick_size("asset-unset", 0.01)
    request = PolymarketOrderRequest(
        market_id="cond-unset", token_id="asset-unset", side="BUY",
        size="50000", price="42", tif="IOC",
        client_order_id="coid-z", expiration_ts=None,
    )
    client.create_signed_order(request)
    # Defaults to False.
    assert seen["_neg_risk"] is False
    # Stderr warning surfaced.
    captured = capsys.readouterr()
    assert "set_neg_risk not called" in captured.err
