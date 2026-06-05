from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac as _hmac
import json
import random
import socket
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .polymarket_adapter import PolymarketCLOBClient, PolymarketOrderRequest


OrderSigner = Callable[[dict[str, object], str | None], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class PolymarketCLOBHttpClientConfig:
    api_url: str
    private_key: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    passphrase: str | None = None
    funder: str | None = None   # proxy wallet address; positions are held here, not at the signing key
    request_timeout_ms: int = 10_000
    updates_max_retries: int = 3
    updates_retry_base_ms: int = 100
    submit_path: str = "/order"
    cancel_path: str = "/order"
    updates_path: str = "/data/orders"
    open_orders_path: str = "/data/orders"
    positions_path: str = "/data/positions"   # set to "" to disable position reconciliation
    data_api_url: str = "https://data-api.polymarket.com"  # Data API base URL for positions
    user_agent: str = "polyexecutor/1.0"


class SyncHttpTransport(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        timeout_ms: int,
        headers: Mapping[str, str],
        params: Mapping[str, object] | None = None,
        json_body: Mapping[str, object] | None = None,
    ) -> tuple[int, str]: ...


class UrllibHttpTransport:
    def request(
        self,
        *,
        method: str,
        url: str,
        timeout_ms: int,
        headers: Mapping[str, str],
        params: Mapping[str, object] | None = None,
        json_body: Mapping[str, object] | None = None,
    ) -> tuple[int, str]:
        final_url = _append_query(url, params)
        payload_bytes: bytes | None = None
        if json_body is not None:
            payload_bytes = json.dumps(dict(json_body), separators=(",", ":")).encode("utf-8")

        request = Request(final_url, data=payload_bytes, method=method.upper())
        for name, value in headers.items():
            request.add_header(name, value)

        timeout_seconds = max(0.1, timeout_ms / 1000.0)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
                status = int(getattr(response, "status", 200))
                return status, body
        except HTTPError as exc:
            body_bytes = exc.read() if hasattr(exc, "read") else b""
            body = body_bytes.decode("utf-8", errors="replace") if isinstance(body_bytes, (bytes, bytearray)) else str(body_bytes)
            return int(exc.code), body
        except socket.timeout as exc:
            raise TimeoutError("http request timed out") from exc
        except URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise TimeoutError("http request timed out") from exc
            text = str(reason or exc)
            if "timed out" in text.lower():
                raise TimeoutError("http request timed out") from exc
            raise OSError(text) from exc
        except OSError:
            raise
        except Exception as exc:
            raise OSError(str(exc)) from exc


class PolymarketCLOBHttpClient(PolymarketCLOBClient):
    __slots__ = ("_config", "_transport", "_signer")

    def __init__(
        self,
        config: PolymarketCLOBHttpClientConfig,
        *,
        transport: SyncHttpTransport | None = None,
        signer: OrderSigner | None = None,
    ) -> None:
        self._validate_config(config)
        self._config = config
        self._transport = transport or UrllibHttpTransport()
        self._signer = signer

    def create_signed_order(self, request: PolymarketOrderRequest) -> Mapping[str, object]:
        unsigned: dict[str, object] = {
            "market_id": request.market_id,
            "token_id": request.token_id,
            "side": request.side,
            "size": request.size,
            "price": request.price,
            "tif": request.tif,
            "client_order_id": request.client_order_id,
        }
        if request.expiration_ts is not None:
            unsigned["expiration_ts"] = int(request.expiration_ts)

        if self._signer is None:
            return {"order": unsigned}

        signed = self._signer(unsigned, self._config.private_key)
        return _as_mapping(signed, fallback={"order": unsigned})

    def submit_order(self, signed_order: Mapping[str, object], timeout_ms: int) -> Mapping[str, object]:
        raw = _as_mapping(signed_order, fallback={})
        # If create_signed_order returned raw EIP-712 fields, wrap them per Polymarket spec.
        # If it returned an already-wrapped dict ({"order": ..., "owner": ...}), use as-is.
        if "order" in raw and isinstance(raw.get("order"), Mapping):
            body = raw
        else:
            body = {
                "order": raw,
                "owner": self._config.api_key or "",
                "orderType": "GTC",
                "postOnly": False,
                "deferExec": False,
            }
        status, payload = self._request_json(
            method="POST",
            path=self._config.submit_path,
            timeout_ms=timeout_ms,
            json_body=body,
        )
        response = _as_mapping(payload, fallback={"raw": payload})
        if "status" not in response and "state" not in response:
            response["status"] = f"HTTP_{status}"
        response.setdefault("http_status", status)
        return response

    def cancel_order(
        self,
        *,
        client_order_id: str | None = None,
        venue_order_id: str | None = None,
        timeout_ms: int,
    ) -> Mapping[str, object]:
        # Polymarket cancel: DELETE /order with {"orderID": <venue_order_id>}
        order_id = venue_order_id or client_order_id
        body: dict[str, object] = {"orderID": order_id} if order_id else {}

        status, payload = self._request_json(
            method="DELETE",
            path=self._config.cancel_path,
            timeout_ms=timeout_ms,
            json_body=body,
        )
        response = _as_mapping(payload, fallback={"raw": payload})
        if "status" not in response and "state" not in response:
            response["status"] = f"HTTP_{status}"
        response.setdefault("http_status", status)
        return response

    def get_order_updates(
        self,
        *,
        since_sequence: int | None,
        limit: int,
        timeout_ms: int,
    ) -> Sequence[Mapping[str, object]]:
        # No status filter — fetches LIVE (open) and MATCHED (filled) orders so
        # fills are captured even when the WebSocket is temporarily disconnected.
        params: dict[str, object] = {"limit": max(1, int(limit))}

        attempts = max(1, self._config.updates_max_retries)
        for attempt in range(1, attempts + 1):
            try:
                status, payload = self._request_json(
                    method="GET",
                    path=self._config.updates_path,
                    timeout_ms=timeout_ms,
                    params=params,
                    json_body=None,
                )
                if status >= 400:
                    return tuple()
                return tuple(_extract_items(payload))
            except (TimeoutError, OSError):
                if attempt >= attempts:
                    return tuple()
                delay_ms = self._config.updates_retry_base_ms * (2 ** (attempt - 1))
                jitter_ms = random.uniform(0.0, self._config.updates_retry_base_ms)
                time.sleep((delay_ms + jitter_ms) / 1000.0)
        return tuple()

    def get_open_orders(self, *, timeout_ms: int) -> Sequence[Mapping[str, object]]:
        http_status, payload = self._request_json(
            method="GET",
            path=self._config.open_orders_path,
            timeout_ms=timeout_ms,
            params=None,
            json_body=None,
        )
        if http_status >= 400:
            raise OSError(f"open orders request failed with http_status={http_status}")
        return tuple(_extract_items(payload))

    def get_positions(self, *, timeout_ms: int) -> Sequence[Mapping[str, object]]:
        """Fetch positions from the Polymarket Data API.

        Uses https://data-api.polymarket.com/positions (not the CLOB API, which has
        no positions endpoint). Returns raw dicts with conditionId, outcome, size,
        avgPrice fields for the adapter to map to VenuePosition.
        """
        if not self._config.data_api_url:
            return tuple()

        # Proxy wallet accounts: positions are held by the funder address, not the signing key.
        address = self._config.funder or ""
        if not address and self._config.private_key:
            try:
                from eth_account import Account
                address = Account.from_key(self._config.private_key).address
            except Exception:
                pass
        if not address:
            return tuple()

        url = _join_url(self._config.data_api_url, "/positions")
        final_url = _append_query(url, {"user": address})
        status, body = self._transport.request(
            method="GET",
            url=final_url,
            timeout_ms=max(1, int(timeout_ms)),
            headers={"Accept": "application/json", "User-Agent": self._config.user_agent},
        )
        if status >= 400:
            raise OSError(f"get_positions (data api) failed with http_status={status}")
        payload = _decode_json(body)
        if isinstance(payload, list):
            return tuple(item for item in payload if isinstance(item, Mapping))
        return tuple(_extract_items(payload))

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        timeout_ms: int,
        params: Mapping[str, object] | None = None,
        json_body: Mapping[str, object] | None = None,
    ) -> tuple[int, object]:
        body_str = json.dumps(dict(json_body), separators=(",", ":")) if json_body else ""
        status, body = self._transport.request(
            method=method,
            url=_join_url(self._config.api_url, path),
            timeout_ms=max(1, int(timeout_ms)),
            headers=self._l2_headers(method, path, body_str),
            params=params,
            json_body=json_body,
        )
        return status, _decode_json(body)

    def _l2_headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self._config.user_agent,
        }
        if not (self._config.api_key and self._config.api_secret and self._config.passphrase):
            return headers

        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path
        if body:
            message += body.replace("'", '"')

        secret = self._config.api_secret
        padding = (4 - len(secret) % 4) % 4
        raw_key = base64.urlsafe_b64decode(secret + "=" * padding)
        sig = base64.urlsafe_b64encode(
            _hmac.new(raw_key, message.encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")

        address = ""
        if self._config.private_key:
            try:
                from eth_account import Account
                address = Account.from_key(self._config.private_key).address
            except Exception:
                pass

        headers["POLY_ADDRESS"] = address
        headers["POLY_SIGNATURE"] = sig
        headers["POLY_TIMESTAMP"] = timestamp
        headers["POLY_API_KEY"] = self._config.api_key
        headers["POLY_PASSPHRASE"] = self._config.passphrase
        if self._config.funder:
            headers["POLY_FUNDER"] = self._config.funder
        return headers

    @staticmethod
    def _validate_config(config: PolymarketCLOBHttpClientConfig) -> None:
        if not config.api_url:
            raise ValueError("api_url must be non-empty")
        if config.request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be > 0")
        if config.updates_max_retries <= 0:
            raise ValueError("updates_max_retries must be > 0")
        if config.updates_retry_base_ms < 0:
            raise ValueError("updates_retry_base_ms must be >= 0")


def _decode_json(body: str) -> object:
    text = body.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_body": body}


def _extract_items(payload: object) -> list[Mapping[str, object]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]

    if isinstance(payload, Mapping):
        for key in ("updates", "events", "items", "orders", "open_orders", "openOrders"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                extracted = [item for item in candidate if isinstance(item, Mapping)]
                if extracted:
                    return extracted

        for key in ("data", "payload", "result"):
            nested = payload.get(key)
            extracted = _extract_items(nested)
            if extracted:
                return extracted

        if any(key in payload for key in ("client_order_id", "clientOrderId", "order_id", "id", "status", "state")):
            return [dict(payload)]

    return []


def _join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def _append_query(url: str, params: Mapping[str, object] | None) -> str:
    if not params:
        return url
    query = urlencode({key: value for key, value in params.items() if value is not None})
    if not query:
        return url
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}{query}"


def _as_mapping(value: object, fallback: dict[str, object]) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return dict(fallback)
