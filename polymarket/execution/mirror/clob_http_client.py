"""Substitute Polymarket CLOB HTTP client.

Replaces ``_kernel/polymarket_clob_client.py``. Implements the
``PolymarketCLOBClient`` Protocol from
``_kernel/polymarket_adapter.py`` so the kernel adapter can use it
without modification.

Why this exists: the vendored kernel's ``_build_request`` stringifies
``VenueOrderIntent.quantity`` and ``limit_price_ticks`` directly as
the wire ``size`` and ``price`` fields. py-clob-client (and Polymarket
itself) expects those as decimal strings — fractional shares like
``"5.18"`` and dollar prices in ``[0, 1]`` like ``"0.42"``. The
kernel as vendored sends ``"5"`` and ``"42"`` respectively, the
latter being off by 100×.

This substitute keeps the kernel adapter intact (its idempotency,
ambiguous-submit handling, and event normalization are correct) but
fixes the wire encoding by:

  * Decoding ``quantity`` (an int) back to fractional shares via
    ``shares = int(request.size) / quantity_scale``.
  * Decoding ``limit_price_ticks`` (an int) back to a dollar price
    via ``price = int(request.price) * tick_size`` where tick_size
    is provided per-token via :meth:`set_tick_size` (the wrapper
    populates this from the orderbook before each submit).

Encoding scale is fixed at construction. Wrapper code must use the
same scale when building intents — recommended ``quantity_scale =
10_000`` (4 decimal places, sufficient for every fill we've observed
on RTDS).

The Protocol surface is exactly the kernel's; the only public method
beyond it is :meth:`set_tick_size`, which the wrapper uses as a
side-channel before each ``submit_order``.
"""
from __future__ import annotations

import json
import random
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from polymarket.execution._kernel.polymarket_adapter import (
    PolymarketCLOBClient,
    PolymarketOrderRequest,
)

_OrderSigner = Callable[[dict[str, object], str | None], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class ClobHttpClientConfig:
    api_url: str
    api_key: str
    api_secret: str
    passphrase: str
    private_key: str
    quantity_scale: int = 10_000
    default_tick_size: float = 0.01
    request_timeout_ms: int = 1_500
    updates_max_retries: int = 3
    updates_retry_base_ms: int = 100
    submit_path: str = "/order"
    cancel_path: str = "/cancel"
    updates_path: str = "/orders/updates"
    open_orders_path: str = "/orders/open"
    user_agent: str = "polyexecutor/1.0-substitute"


class ClobHttpClient(PolymarketCLOBClient):
    """Wire-format-correct substitute for the kernel's HTTP client."""

    __slots__ = ("_config", "_signer", "_tick_size_by_token", "_urlopen")

    def __init__(
        self,
        config: ClobHttpClientConfig,
        *,
        signer: _OrderSigner | None = None,
        urlopen_fn: Callable[..., Any] | None = None,
    ) -> None:
        if not config.api_url:
            raise ValueError("api_url must be non-empty")
        if config.quantity_scale <= 0:
            raise ValueError("quantity_scale must be > 0")
        if config.default_tick_size <= 0:
            raise ValueError("default_tick_size must be > 0")
        if config.request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be > 0")
        if config.updates_max_retries <= 0:
            raise ValueError("updates_max_retries must be > 0")
        self._config: ClobHttpClientConfig = config
        self._signer: _OrderSigner | None = signer
        self._tick_size_by_token: dict[str, float] = {}
        # Injectable for tests; defaults to stdlib urlopen.
        self._urlopen = urlopen_fn if urlopen_fn is not None else urlopen

    # ------------------------------------------------------------------
    # Side-channel for the wrapper to seed tick sizes before submit.
    # ------------------------------------------------------------------

    def set_tick_size(self, token_id: str, tick_size: float) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        self._tick_size_by_token[token_id] = float(tick_size)

    # ------------------------------------------------------------------
    # Protocol methods.
    # ------------------------------------------------------------------

    def create_signed_order(
        self, request: PolymarketOrderRequest
    ) -> Mapping[str, object]:
        # Decode the kernel's int-stringified fields back to wire-correct
        # decimal strings.
        try:
            quantity_int = int(request.size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"size must be a stringified int, got {request.size!r}") from exc
        try:
            ticks_int = int(request.price)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"price must be a stringified int, got {request.price!r}") from exc

        shares = quantity_int / self._config.quantity_scale
        tick_size = self._tick_size_by_token.get(
            request.token_id, self._config.default_tick_size
        )
        price_dollars = ticks_int * tick_size

        # Format with enough precision for both fields. 4 decimal places
        # matches quantity_scale=10_000; tick_size handles its own
        # precision via the multiplication. Trim trailing zeros so
        # py-clob-client doesn't trip on padded strings.
        size_str = _format_decimal(shares, max_decimals=6)
        price_str = _format_decimal(price_dollars, max_decimals=6)

        unsigned: dict[str, object] = {
            "market_id": request.market_id,
            "token_id": request.token_id,
            "side": request.side,
            "size": size_str,
            "price": price_str,
            "tif": request.tif,
            "client_order_id": request.client_order_id,
        }
        if request.expiration_ts is not None:
            unsigned["expiration_ts"] = int(request.expiration_ts)

        if self._signer is None:
            return {"order": unsigned}

        signed = self._signer(unsigned, self._config.private_key)
        if isinstance(signed, Mapping):
            return signed
        return {"order": unsigned}

    def submit_order(
        self, signed_order: Mapping[str, object], timeout_ms: int
    ) -> Mapping[str, object]:
        body = signed_order if isinstance(signed_order, Mapping) else {}
        status, payload = self._request_json(
            method="POST",
            path=self._config.submit_path,
            timeout_ms=timeout_ms,
            json_body=dict(body),
        )
        return _augment_response(payload, status)

    def cancel_order(
        self,
        *,
        client_order_id: str | None = None,
        venue_order_id: str | None = None,
        timeout_ms: int,
    ) -> Mapping[str, object]:
        body: dict[str, object] = {}
        if client_order_id is not None:
            body["client_order_id"] = client_order_id
        if venue_order_id is not None:
            body["order_id"] = venue_order_id
        status, payload = self._request_json(
            method="POST",
            path=self._config.cancel_path,
            timeout_ms=timeout_ms,
            json_body=body,
        )
        return _augment_response(payload, status)

    def get_order_updates(
        self,
        *,
        since_sequence: int | None,
        limit: int,
        timeout_ms: int,
    ) -> Sequence[Mapping[str, object]]:
        params: dict[str, object] = {"limit": max(1, int(limit))}
        if since_sequence is not None:
            params["since_sequence"] = int(since_sequence)

        attempts = max(1, self._config.updates_max_retries)
        for attempt in range(1, attempts + 1):
            try:
                status, payload = self._request_json(
                    method="GET",
                    path=self._config.updates_path,
                    timeout_ms=timeout_ms,
                    params=params,
                )
            except (TimeoutError, OSError):
                if attempt >= attempts:
                    return tuple()
                delay_ms = self._config.updates_retry_base_ms * (2 ** (attempt - 1))
                jitter_ms = random.uniform(0.0, float(self._config.updates_retry_base_ms))
                time.sleep((delay_ms + jitter_ms) / 1000.0)
                continue
            if status >= 400:
                return tuple()
            return tuple(_extract_items(payload))
        return tuple()

    def get_open_orders(self, *, timeout_ms: int) -> Sequence[Mapping[str, object]]:
        status, payload = self._request_json(
            method="GET",
            path=self._config.open_orders_path,
            timeout_ms=timeout_ms,
        )
        if status >= 400:
            raise OSError(f"open orders request failed with http_status={status}")
        return tuple(_extract_items(payload))

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        timeout_ms: int,
        params: Mapping[str, object] | None = None,
        json_body: Mapping[str, object] | None = None,
    ) -> tuple[int, object]:
        url = _join_url(self._config.api_url, path)
        url = _append_query(url, params)

        payload_bytes: bytes | None = None
        if json_body is not None:
            payload_bytes = json.dumps(dict(json_body), separators=(",", ":")).encode("utf-8")

        request = Request(url, data=payload_bytes, method=method.upper())
        for name, value in self._headers().items():
            request.add_header(name, value)

        timeout_seconds = max(0.1, max(1, int(timeout_ms)) / 1000.0)
        try:
            with self._urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
                status = int(getattr(response, "status", 200))
                return status, _decode_json(body)
        except HTTPError as exc:
            body_bytes = exc.read() if hasattr(exc, "read") else b""
            body = body_bytes.decode("utf-8", errors="replace") if isinstance(
                body_bytes, (bytes, bytearray)
            ) else str(body_bytes)
            return int(exc.code), _decode_json(body)
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutError(f"http request timed out: {url}") from exc
        except URLError as exc:
            raise OSError(f"http transport error: {exc!r}") from exc

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self._config.user_agent,
        }
        if self._config.api_key:
            headers["X-API-KEY"] = self._config.api_key
        if self._config.api_secret:
            headers["X-API-SECRET"] = self._config.api_secret
        if self._config.passphrase:
            headers["X-API-PASSPHRASE"] = self._config.passphrase
        return headers


def _format_decimal(value: float, *, max_decimals: int) -> str:
    """Format a float as a decimal string with up to `max_decimals` precision,
    trimming trailing zeros and the decimal point if integral."""
    formatted = f"{value:.{max_decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


def _augment_response(payload: object, status: int) -> dict[str, object]:
    if isinstance(payload, Mapping):
        out = dict(payload)
    else:
        out = {"raw": payload}
    if "status" not in out and "state" not in out:
        out["status"] = f"HTTP_{status}"
    out.setdefault("http_status", status)
    return out


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
        if any(
            key in payload
            for key in ("client_order_id", "clientOrderId", "order_id", "id", "status", "state")
        ):
            return [dict(payload)]
    return []


def _join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def _append_query(url: str, params: Mapping[str, object] | None) -> str:
    if not params:
        return url
    cleaned = {key: value for key, value in params.items() if value is not None}
    if not cleaned:
        return url
    query = urlencode(cleaned)
    if not query:
        return url
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}{query}"
