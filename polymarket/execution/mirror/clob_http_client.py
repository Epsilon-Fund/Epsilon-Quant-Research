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

# Reserved keys attached to the signed-order mapping by create_signed_order and
# stripped by submit_order before anything goes on the wire. They ride the mapping
# because the kernel treats it as opaque (sign → submit), and the wire body needs
# the order TYPE and the venue-id→client-id correlation needs the COID — neither of
# which appears in the EIP-712 signed order itself.
_RESERVED_ORDER_TYPE = "_order_type"
_RESERVED_CLIENT_ORDER_ID = "_client_order_id"
_RESERVED_HANDLE = "_handle"          # gateway-stashed SignedOrder handle (V2 path)

# The gateway child processes requests SERIALLY and its SDK httpx read-timeout is
# ~10s. If the parent's per-request budget (the kernel's 500ms submit/cancel timeout)
# is SHORTER than that, the parent pre-empts a slow post with a TimeoutError, the
# kernel marks it ambiguous and fires the rollback cancel — but that cancel is queued
# behind the still-running post in the serial child, so a real order can rest
# unattended until the child drains (~10s). Waiting LONGER than the child's own
# timeout inverts the hierarchy: the child always returns a clean classified response
# first (and is then free to service the cancel), so no cancel is ever head-of-line
# blocked by an in-flight post. (Adversarial-review HIGH finding, 2026-07-09.)
_GATEWAY_MIN_TIMEOUT_S = 20.0


def _gateway_timeout_s(timeout_ms: int) -> float:
    return max(max(1, int(timeout_ms)) / 1000.0, _GATEWAY_MIN_TIMEOUT_S)

# Polymarket wire order types. The kernel's TimeInForce only knows GTC/IOC; the
# venue's IOC-equivalent is FAK (fill-and-kill). "IOC" itself is NOT a valid wire
# value and is rejected.
_ORDER_TYPE_BY_TIF = {"GTC": "GTC", "IOC": "FAK", "FOK": "FOK", "GTD": "GTD"}

# py-clob-client's pagination sentinel for GET /data/orders.
_END_CURSOR = "LTE="


@dataclass(frozen=True, slots=True)
class ClobHttpClientConfig:
    api_url: str
    api_key: str
    api_secret: str
    passphrase: str
    private_key: str
    chain_id: int = 137
    quantity_scale: int = 10_000
    default_tick_size: float = 0.01
    request_timeout_ms: int = 1_500
    updates_max_retries: int = 3
    updates_retry_base_ms: int = 100
    submit_path: str = "/order"
    cancel_path: str = "/order"          # Polymarket cancels via DELETE /order {"orderID": …}
    cancel_all_path: str = "/cancel-all"  # coid-only fallback (no venue id known)
    updates_path: str = "/orders/updates"
    open_orders_path: str = "/data/orders"
    user_agent: str = "polyexecutor/1.0-substitute"


class ClobHttpClient(PolymarketCLOBClient):
    """Wire-format-correct substitute for the kernel's HTTP client."""

    __slots__ = (
        "_config", "_signer", "_tick_size_by_token",
        "_neg_risk_by_token", "_urlopen", "_coid_by_venue_id", "_gateway",
    )

    def __init__(
        self,
        config: ClobHttpClientConfig,
        *,
        signer: _OrderSigner | None = None,
        urlopen_fn: Callable[..., Any] | None = None,
        gateway: Any | None = None,
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
        # NegRisk flag per token. Populated by the wrapper before each
        # submit via set_neg_risk(). Default (missing) is False —
        # treat as binary. The ClobSigner reads this off the unsigned
        # dict on the reserved `_neg_risk` key.
        self._neg_risk_by_token: dict[str, bool] = {}
        # Injectable for tests; defaults to stdlib urlopen.
        self._urlopen = urlopen_fn if urlopen_fn is not None else urlopen
        # venue order id -> our client_order_id, learned from submit acks. Used to
        # annotate GET /data/orders items (Polymarket does not store client ids), so
        # the kernel's reconcile can attribute venue-open orders. Restart limitation:
        # the map is per-process — orders resting across a restart cannot be
        # attributed and reconcile will not see them (the runbook's end-of-session
        # "UI shows zero open orders" check covers this).
        self._coid_by_venue_id: dict[str, str] = {}
        # V2 order path: Polymarket archived py-clob-client and rejects V1 signed
        # orders ("invalid order version"). When a PySdkOrderGateway is wired in,
        # sign/submit/cancel delegate to the successor SDK in its isolated child
        # process; the legacy V1 wire path below remains ONLY for gateway-less
        # construction (unit tests / fake flows) and cannot place real V2 orders.
        self._gateway = gateway

    # ------------------------------------------------------------------
    # Side-channel for the wrapper to seed tick sizes before submit.
    # ------------------------------------------------------------------

    def set_tick_size(self, token_id: str, tick_size: float) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        self._tick_size_by_token[token_id] = float(tick_size)

    def set_neg_risk(self, token_id: str, is_neg_risk: bool) -> None:
        """Stores per-asset NegRisk flag.

        Must be called by the wrapper before each ``submit_order``.
        Mirrors :meth:`set_tick_size`. The ClobSigner reads this off
        the unsigned dict on a reserved key and passes it to
        py-clob-client's :class:`PartialCreateOrderOptions` so the
        EIP-712 ``verifyingContract`` resolves to the NegRisk CTF
        Exchange.

        Missing entry defaults to ``False`` (binary). Calling
        ``set_neg_risk(token_id, False)`` is a no-op semantically
        but is still recommended for clarity.
        """
        self._neg_risk_by_token[token_id] = bool(is_neg_risk)

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

        # Per-token side-channel reads. The wrapper is responsible for
        # populating both via set_tick_size() and set_neg_risk() before
        # each submit; missing entries surface a stderr warning and
        # fall back to safe defaults (binary + default tick).
        neg_risk = self._neg_risk_by_token.get(request.token_id)
        if neg_risk is None:
            print(
                f"[clob_http_client] set_neg_risk not called for "
                f"token_id={request.token_id}; defaulting to False (binary). "
                "NegRisk orders signed under this default will be rejected "
                "as invalid signature.",
                file=__import__("sys").stderr, flush=True,
            )
            neg_risk = False
        tick_size_for_signing = self._tick_size_by_token.get(
            request.token_id, self._config.default_tick_size
        )

        unsigned: dict[str, object] = {
            "market_id": request.market_id,
            "token_id": request.token_id,
            "side": request.side,
            "size": size_str,
            "price": price_str,
            "tif": request.tif,
            "client_order_id": request.client_order_id,
            # Reserved keys read by ClobSigner; the kernel's old signer
            # ignores unknown keys, so this is safe to populate
            # unconditionally even when a different signer is wired in.
            "_neg_risk": neg_risk,
            "_tick_size": tick_size_for_signing,
        }
        if request.expiration_ts is not None:
            unsigned["expiration_ts"] = int(request.expiration_ts)

        # Reserved side-channel keys (stripped from the wire body by submit_order):
        # the wire "orderType" and the venue-id→client-id correlation. tif "IOC" maps
        # to Polymarket's FAK; "IOC" itself is not a valid wire order type.
        reserved = {
            _RESERVED_ORDER_TYPE: _ORDER_TYPE_BY_TIF.get(
                str(request.tif or "GTC").upper(), "GTC"),
            _RESERVED_CLIENT_ORDER_ID: request.client_order_id,
        }

        if self._gateway is not None:
            # V2 path: the SDK signs (and resolves tick/neg-risk metadata itself).
            # Only GTC resting orders are supported here — fail closed on anything
            # else rather than mistranslate a time-in-force.
            tif = str(request.tif or "GTC").upper()
            if tif != "GTC":
                raise ValueError(f"py-sdk gateway path supports GTC only, got {tif!r}")
            resp = self._gateway.create_limit_order(
                token_id=request.token_id, price=price_str, size=size_str,
                side=str(request.side).upper(),
            )
            if not resp.get("ok"):
                kind = resp.get("kind")
                msg = f"gateway create_limit_order failed: {resp.get('error')}"
                if kind == "timeout":
                    raise TimeoutError(msg)
                if kind == "transport":
                    raise OSError(msg)
                raise ValueError(msg)
            return {
                "order": resp.get("order", {}),
                _RESERVED_HANDLE: resp.get("handle"),
                **reserved,
            }

        if self._signer is None:
            return {"order": unsigned, **reserved}

        signed = self._signer(unsigned, self._config.private_key)
        if isinstance(signed, Mapping):
            return {**signed, **reserved}
        return {"order": unsigned, **reserved}

    def submit_order(
        self, signed_order: Mapping[str, object], timeout_ms: int
    ) -> Mapping[str, object]:
        """POST the signed order in Polymarket's wire envelope with L2 auth.

        Wire shape (py-clob-client ``post_order``/``order_to_json``):
        ``{"order": <SignedOrder.dict()>, "owner": <api_key>, "orderType": <type>,
        "postOnly": false}`` — POSTed to ``/order`` with POLY_* L2 headers whose HMAC
        covers the exact serialized body. The previous implementation sent the bare
        signed dict with legacy X-API headers → the venue's "missing address header".
        """
        raw = dict(signed_order) if isinstance(signed_order, Mapping) else {}
        order_type = str(raw.pop(_RESERVED_ORDER_TYPE, "GTC") or "GTC")
        coid = raw.pop(_RESERVED_CLIENT_ORDER_ID, None)
        handle = raw.pop(_RESERVED_HANDLE, None)

        if self._gateway is not None and handle:
            # V2 path: post the SDK-signed order from the gateway's stash. A
            # child-reported timeout/transport failure maps onto the kernel's
            # ambiguous-submit semantics (TimeoutError/OSError); a clean venue
            # rejection returns as a normal NACK mapping.
            resp = self._gateway.post_order(
                handle=str(handle), timeout_s=_gateway_timeout_s(timeout_ms))
            if not resp.get("ok"):
                kind = resp.get("kind")
                msg = str(resp.get("error") or resp.get("message") or "rejected")
                if kind == "timeout":
                    raise TimeoutError(f"gateway post_order timeout: {msg}")
                if kind == "transport":
                    raise OSError(f"gateway post_order transport error: {msg}")
                return {"status": "REJECTED", "error": msg,
                        "code": resp.get("code"), "http_status": 400}
            out = {k: v for k, v in resp.items() if k not in ("id", "ok")}
            out.setdefault("status", "LIVE")
            out.setdefault("http_status", 200)
            venue_id = out.get("order_id")
            if coid and isinstance(venue_id, str) and venue_id:
                self._coid_by_venue_id[venue_id] = str(coid)
            return out

        body: dict[str, object] = {
            "order": raw,
            "owner": self._config.api_key,
            "orderType": order_type,
            "postOnly": False,
        }
        status, payload = self._request_json(
            method="POST",
            path=self._config.submit_path,
            timeout_ms=timeout_ms,
            json_body=body,
            use_l2_auth=True,
        )
        out = _augment_response(payload, status)
        venue_id = out.get("order_id")
        if coid and isinstance(venue_id, str) and venue_id:
            self._coid_by_venue_id[venue_id] = str(coid)
        return out

    def cancel_order(
        self,
        *,
        client_order_id: str | None = None,
        venue_order_id: str | None = None,
        timeout_ms: int,
    ) -> Mapping[str, object]:
        """Cancel via Polymarket's DELETE /order {"orderID": …} with L2 auth.

        Without a venue order id (ambiguous-submit rollback, ack never parsed) there
        is nothing to address a single-order cancel at, so fall back to cancel-all —
        cancels are reduce-only and this account runs only this bot, so the blast
        radius is exactly our own resting quotes.
        """
        if self._gateway is not None:
            try:
                resp = (self._gateway.cancel_order(
                            order_id=venue_order_id,
                            timeout_s=_gateway_timeout_s(timeout_ms))
                        if venue_order_id else
                        self._gateway.cancel_all(
                            timeout_s=_gateway_timeout_s(timeout_ms)))
            except (TimeoutError, OSError):
                raise
            if not resp.get("ok"):
                kind = resp.get("kind")
                msg = str(resp.get("error") or "cancel failed")
                if kind == "timeout":
                    raise TimeoutError(f"gateway cancel timeout: {msg}")
                if kind == "transport":
                    raise OSError(f"gateway cancel transport error: {msg}")
                return {"status": "REJECTED", "error": msg, "http_status": 400}
            out = {k: v for k, v in resp.items() if k not in ("id", "ok")}
            canceled = out.get("canceled")
            not_canceled = out.get("not_canceled")
            if venue_order_id and isinstance(canceled, list) and venue_order_id in canceled:
                out["status"] = "CANCELED"
            elif venue_order_id and isinstance(not_canceled, Mapping) \
                    and venue_order_id in not_canceled:
                out["status"] = "NOT_FOUND"
                out.setdefault("reason", str(not_canceled[venue_order_id]))
            else:
                out.setdefault("status", "CANCELED")
            if not venue_order_id:
                out.setdefault("fallback", "cancel_all")
            out.setdefault("http_status", 200)
            return out

        if venue_order_id:
            status, payload = self._request_json(
                method="DELETE",
                path=self._config.cancel_path,
                timeout_ms=timeout_ms,
                json_body={"orderID": venue_order_id},
                use_l2_auth=True,
            )
            out = _augment_response(payload, status)
            # Polymarket answers {"canceled": [...ids], "not_canceled": {id: reason}}.
            canceled = out.get("canceled")
            not_canceled = out.get("not_canceled")
            if isinstance(canceled, list) and venue_order_id in canceled:
                out.setdefault("status", "CANCELED")
                out["status"] = "CANCELED"
            elif isinstance(not_canceled, Mapping) and venue_order_id in not_canceled:
                out["status"] = "NOT_FOUND"
                out.setdefault("reason", str(not_canceled[venue_order_id]))
            return out
        status, payload = self._request_json(
            method="DELETE",
            path=self._config.cancel_all_path,
            timeout_ms=timeout_ms,
            use_l2_auth=True,
        )
        out = _augment_response(payload, status)
        out.setdefault("fallback", "cancel_all")
        return out

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
        """GET /data/orders (L2), following pagination, annotating client ids.

        Polymarket pages with ``next_cursor`` until the ``LTE=`` sentinel and does
        NOT store client order ids on orders — the kernel's reconcile attributes an
        open order only via ``client_order_id``, so each item is annotated from the
        submit-ack correlation map (see ``_coid_by_venue_id``; restart limitation
        documented there).
        """
        items: list[Mapping[str, object]] = []
        cursor = "MA=="
        for _page in range(64):   # hard bound; a session's open orders fit in page 1
            status, payload = self._request_json(
                method="GET",
                path=self._config.open_orders_path,
                timeout_ms=timeout_ms,
                params={"next_cursor": cursor},
                use_l2_auth=True,
            )
            if status >= 400:
                raise OSError(f"open orders request failed with http_status={status}")
            items.extend(_extract_items(payload))
            next_cursor = (payload.get("next_cursor")
                           if isinstance(payload, Mapping) else None)
            if not isinstance(next_cursor, str) or not next_cursor \
                    or next_cursor in (_END_CURSOR, cursor):
                break
            cursor = next_cursor
        out: list[Mapping[str, object]] = []
        for item in items:
            if item.get("client_order_id") or item.get("clientOrderId"):
                out.append(item)
                continue
            venue_id = str(item.get("id") or item.get("order_id") or "")
            coid = self._coid_by_venue_id.get(venue_id)
            out.append({**item, "client_order_id": coid} if coid else item)
        return tuple(out)

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
        use_l2_auth: bool = False,
    ) -> tuple[int, object]:
        url = _join_url(self._config.api_url, path)
        url = _append_query(url, params)

        payload_bytes: bytes | None = None
        serialized_body: str | None = None
        if json_body is not None:
            # Byte-identical to py-clob-client's serialization: the L2 HMAC covers this
            # exact string, and the venue verifies it against the raw received body.
            serialized_body = json.dumps(dict(json_body), separators=(",", ":"),
                                         ensure_ascii=False)
            payload_bytes = serialized_body.encode("utf-8")

        request = Request(url, data=payload_bytes, method=method.upper())
        headers = (
            self._l2_headers(method=method.upper(), path=path, body=serialized_body)
            if use_l2_auth else self._headers()
        )
        for name, value in headers.items():
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

    def _l2_headers(self, *, method: str, path: str, body: str | None) -> dict[str, str]:
        try:
            from py_clob_client.clob_types import ApiCreds, RequestArgs
            from py_clob_client.headers.headers import create_level_2_headers
            from py_clob_client.signer import Signer
        except ImportError as exc:
            raise RuntimeError("py-clob-client is required for CLOB L2 auth") from exc

        signer = Signer(self._config.private_key, self._config.chain_id)
        creds = ApiCreds(
            api_key=self._config.api_key,
            api_secret=self._config.api_secret,
            api_passphrase=self._config.passphrase,
        )
        request_args = RequestArgs(
            method=method,
            request_path=path,
            body=body,
            serialized_body=body,
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self._config.user_agent,
        }
        headers.update(create_level_2_headers(signer, creds, request_args))
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
    # Polymarket's field names → the keys the frozen kernel normalizer reads.
    # Success: {"success": true, "status": "live"|"matched", "orderID": "0x…"} —
    # the kernel reads order_id/id, so a missing remap loses the venue id (breaking
    # cancel-by-id and reconcile). Errors arrive as errorMsg; kernel reads
    # reason/error/message.
    if "order_id" not in out and "id" not in out and out.get("orderID"):
        out["order_id"] = out["orderID"]
    if "error" not in out and "reason" not in out and "message" not in out \
            and out.get("errorMsg"):
        out["error"] = out["errorMsg"]
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
