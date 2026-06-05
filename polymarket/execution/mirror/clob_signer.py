"""NegRisk-aware order signer that calls py-clob-client directly.

Replaces ``_kernel/polymarket_sdk_signer.PyClobClientOrderSigner`` in
the real-venue path. The kernel signer ignores
``PartialCreateOrderOptions`` and therefore can't tell py-clob-client
to use the NegRisk CTF Exchange for the EIP-712 ``verifyingContract``.
NegRisk orders signed by the kernel signer are rejected with
``invalid signature``.

Interface is intentionally compatible with the kernel signer
(``signer(unsigned: dict, private_key: str) -> Mapping``) so the
substitute HTTP client can swap implementations without further
plumbing. The ``neg_risk`` flag and ``tick_size`` flow through the
``unsigned`` dict on reserved keys ``_neg_risk`` and ``_tick_size``,
populated by the substitute HTTP client from its per-asset caches
before each submit.

This module owns the py-clob-client ``ClobClient`` instance lifecycle
(one client cached per ``private_key`` — same pattern as the kernel
signer, since constructing a client is non-trivial).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ClobSignerConfig:
    api_url: str
    chain_id: int = 137
    signature_type: int = 1
    funder: str | None = None
    maker: str | None = None
    default_tick_size: float = 0.01


class ClobSigner:
    """Callable wrapper around ``py_clob_client.ClobClient.create_order``.

    Construction is lazy — py-clob-client is imported the first time
    ``__call__`` runs, matching the kernel signer's pattern so tests
    can run without py-clob-client installed (they mock the signer
    rather than the SDK).
    """

    __slots__ = (
        "_config",
        "_clients_by_private_key",
        "_clob_client_cls",
        "_order_args_cls",
        "_options_cls",
    )

    def __init__(self, config: ClobSignerConfig) -> None:
        if not config.api_url:
            raise ValueError("api_url must be non-empty")
        if config.chain_id <= 0:
            raise ValueError("chain_id must be > 0")
        self._config = config
        self._clients_by_private_key: dict[str, Any] = {}
        self._clob_client_cls: Any = None
        self._order_args_cls: Any = None
        self._options_cls: Any = None

    def __call__(
        self, unsigned: dict[str, object], private_key: str | None
    ) -> Mapping[str, object]:
        if not private_key:
            raise ValueError("private_key is required for SDK signing")
        self._lazy_load_bindings()

        # Reserved keys populated by ClobHttpClient before the call.
        # Defaults are paranoid: treat as binary + default tick if
        # the caller forgot to seed them (warning surfaces in the
        # HTTP client; this is just a safety net).
        neg_risk = bool(unsigned.get("_neg_risk", False))
        tick_size = float(
            unsigned.get("_tick_size", self._config.default_tick_size)
        )

        order_args = self._build_order_args(unsigned)
        options = self._options_cls(neg_risk=neg_risk, tick_size=tick_size)

        client = self._client_for_private_key(private_key)
        signed = client.create_order(order_args, options)
        return _to_mapping(signed)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lazy_load_bindings(self) -> None:
        if self._clob_client_cls is not None:
            return
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import (
                OrderArgs,
                PartialCreateOrderOptions,
            )
        except ImportError as exc:
            raise RuntimeError(
                "py-clob-client is required for ClobSigner. "
                "Install with: pip install py-clob-client"
            ) from exc
        self._clob_client_cls = ClobClient
        self._order_args_cls = OrderArgs
        self._options_cls = PartialCreateOrderOptions

    def _client_for_private_key(self, private_key: str) -> Any:
        existing = self._clients_by_private_key.get(private_key)
        if existing is not None:
            return existing
        kwargs: dict[str, object] = {
            "host": self._config.api_url,
            "key": private_key,
            "chain_id": self._config.chain_id,
        }
        if self._config.signature_type is not None:
            kwargs["signature_type"] = self._config.signature_type
        if self._config.funder:
            kwargs["funder"] = self._config.funder
        client = self._clob_client_cls(**kwargs)
        self._clients_by_private_key[private_key] = client
        return client

    def _build_order_args(self, unsigned: Mapping[str, object]) -> Any:
        # py-clob-client's OrderArgs fields:
        #   token_id, price, size, side, fee_rate_bps, nonce, expiration, taker
        # The ClobHttpClient passes us pre-decoded `size` (decimal shares
        # as a string) and `price` (dollar price as a string). We forward
        # them as-is — py-clob-client accepts string-decimal forms and
        # converts internally.
        payload: dict[str, object] = {
            "token_id": unsigned.get("token_id"),
            "price": unsigned.get("price"),
            "size": unsigned.get("size"),
            "side": unsigned.get("side"),
        }
        expiration = unsigned.get("expiration_ts")
        if expiration is not None:
            payload["expiration"] = int(expiration)
        # Drop None values so OrderArgs uses its defaults for unspecified
        # fields (fee_rate_bps, nonce, taker).
        return self._order_args_cls(
            **{k: v for k, v in payload.items() if v is not None}
        )


def _to_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    # py-clob-client's signed orders are often pydantic-y / dataclass-y;
    # try the common shapes the kernel signer also handles.
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            mapped = dict_method()
            if isinstance(mapped, Mapping):
                return dict(mapped)
        except Exception:
            pass
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            mapped = model_dump()
            if isinstance(mapped, Mapping):
                return dict(mapped)
        except Exception:
            pass
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, Mapping):
        return {
            str(k): v
            for k, v in value_dict.items()
            if not str(k).startswith("_")
        }
    raise RuntimeError(
        f"ClobSigner: cannot convert signed order to mapping (type={type(value).__name__})"
    )
