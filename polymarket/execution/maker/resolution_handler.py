"""Resolution detection and best-effort NegRisk self-redemption."""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Protocol

from polymarket.execution.journal import (
    JsonlWriter,
    MarketResolved,
    PositionRedeemable,
    PositionRedeemed,
    RedemptionFailed,
)

NEG_RISK_ADAPTER_ADDRESS = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
NEG_RISK_REDEEM_ABI: list[dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "_conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "_amounts", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]
_STOP_JOIN_TIMEOUT_S = 5.0
_HTTP_USER_AGENT = "curl/8.0"


def _http_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": _HTTP_USER_AGENT})


class Redeemer(Protocol):
    def redeem(self, row: dict[str, Any]) -> str: ...


@dataclass(frozen=True, slots=True)
class DirectWeb3NegRiskRedeemer:
    """Small direct web3.py wrapper around NegRiskAdapter.redeemPositions."""

    rpc_url: str
    private_key: str
    chain_id: int = 137
    adapter_address: str = NEG_RISK_ADAPTER_ADDRESS

    def redeem(self, row: dict[str, Any]) -> str:
        try:
            from web3 import Web3
        except ImportError as exc:
            raise RuntimeError("web3.py is required for self-redemption") from exc

        condition_id = _condition_id(row)
        amounts = _redeem_amounts(row)
        web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        account = web3.eth.account.from_key(self.private_key)
        contract = web3.eth.contract(
            address=Web3.to_checksum_address(self.adapter_address),
            abi=NEG_RISK_REDEEM_ABI,
        )
        tx = contract.functions.redeemPositions(condition_id, amounts).build_transaction({
            "from": account.address,
            "nonce": web3.eth.get_transaction_count(account.address),
            "chainId": self.chain_id,
            "gas": 250_000,
            "gasPrice": web3.eth.gas_price,
        })
        signed = web3.eth.account.sign_transaction(tx, private_key=self.private_key)
        raw = getattr(signed, "rawTransaction", None) or getattr(
            signed, "raw_transaction"
        )
        tx_hash = web3.eth.send_raw_transaction(raw)
        return tx_hash.hex()


class ResolutionHandler:
    """Polls Gamma for closure and Data API for redeemable positions."""

    def __init__(
        self,
        *,
        condition_ids_provider: Callable[[], set[str]],
        funder: str,
        gamma_url: str,
        data_url: str,
        journal: JsonlWriter,
        poll_interval_seconds: float = 60.0,
        redeemer: Redeemer | None = None,
        private_key: str | None = None,
        rpc_url: str | None = None,
        chain_id: int = 137,
        urlopen_fn: Callable[..., Any] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        self._condition_ids_provider = condition_ids_provider
        self._funder = funder.lower()
        self._gamma_url = gamma_url.rstrip("/")
        self._data_url = data_url.rstrip("/")
        self._journal = journal
        self._poll_interval = poll_interval_seconds
        self._urlopen = urlopen_fn if urlopen_fn is not None else urllib.request.urlopen
        if redeemer is not None:
            self._redeemer = redeemer
        else:
            resolved_rpc = rpc_url or os.environ.get("POLYMARKET_RPC_URL")
            if resolved_rpc and private_key:
                self._redeemer = DirectWeb3NegRiskRedeemer(
                    rpc_url=resolved_rpc,
                    private_key=private_key,
                    chain_id=chain_id,
                )
            else:
                self._redeemer = None
        self._resolved_conditions: set[str] = set()
        self._seen_redeemable: set[str] = set()
        self._attempted_redemptions: set[str] = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def is_resolved(self, condition_id: str) -> bool:
        return condition_id.lower() in self._resolved_conditions

    def poll_once(self) -> None:
        self._poll_gamma()
        self._poll_redeemable_positions()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="resolution_handler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_JOIN_TIMEOUT_S)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:
                pass
            if self._stop_event.wait(self._poll_interval):
                break

    def _poll_gamma(self) -> None:
        condition_ids = sorted(cid.lower() for cid in self._condition_ids_provider() if cid)
        if not condition_ids:
            return
        payload = self._fetch_json(
            self._gamma_url,
            "/markets",
            {"condition_ids": ",".join(condition_ids)},
        )
        for row in _extract_rows(payload):
            condition_id = _condition_id(row)
            closed = bool(row.get("closed", False))
            active = bool(row.get("active", True))
            if (closed or not active) and condition_id not in self._resolved_conditions:
                self._resolved_conditions.add(condition_id)
                self._journal.write(MarketResolved(
                    ts_utc=datetime.now(timezone.utc),
                    condition_id=condition_id,
                    closed=closed,
                    active=active,
                    source="gamma",
                ))

    def _poll_redeemable_positions(self) -> None:
        payload = self._fetch_json(
            self._data_url,
            "/positions",
            {"user": self._funder, "redeemable": "true"},
        )
        for row in _extract_rows(payload):
            if not bool(row.get("redeemable", False)):
                continue
            condition_id = _condition_id(row)
            asset_id = str(row.get("asset") or row.get("assetId") or "")
            outcome_index = _int(row.get("outcomeIndex"), default=0)
            size = _float(row.get("size"), default=0.0)
            negative_risk = bool(row.get("negativeRisk", True))
            key = f"{condition_id}|{asset_id}|{outcome_index}"
            if key not in self._seen_redeemable:
                self._seen_redeemable.add(key)
                self._journal.write(PositionRedeemable(
                    ts_utc=datetime.now(timezone.utc),
                    condition_id=condition_id,
                    asset_id=asset_id,
                    outcome_index=outcome_index,
                    size=size,
                    negative_risk=negative_risk,
                ))
            if not negative_risk or key in self._attempted_redemptions:
                continue
            self._attempted_redemptions.add(key)
            self._attempt_redemption(row, key=key)

    def _attempt_redemption(self, row: dict[str, Any], *, key: str) -> None:
        condition_id = _condition_id(row)
        asset_id = str(row.get("asset") or row.get("assetId") or "")
        outcome_index = _int(row.get("outcomeIndex"), default=0)
        size = _float(row.get("size"), default=0.0)
        if self._redeemer is None:
            self._journal.write(RedemptionFailed(
                ts_utc=datetime.now(timezone.utc),
                condition_id=condition_id,
                asset_id=asset_id,
                reason="redeemer_unconfigured",
                detail=f"no web3 redeemer configured for {key}",
            ))
            return
        try:
            tx_hash = self._redeemer.redeem(row)
        except Exception as exc:  # noqa: BLE001
            self._journal.write(RedemptionFailed(
                ts_utc=datetime.now(timezone.utc),
                condition_id=condition_id,
                asset_id=asset_id,
                reason="redeem_positions_failed",
                detail=f"{type(exc).__name__}: {exc}",
            ))
            return
        self._journal.write(PositionRedeemed(
            ts_utc=datetime.now(timezone.utc),
            condition_id=condition_id,
            asset_id=asset_id,
            outcome_index=outcome_index,
            size=size,
            tx_hash=tx_hash,
        ))

    def _fetch_json(
        self, base_url: str, path: str, params: dict[str, str]
    ) -> Any:
        query = urllib.parse.urlencode(params)
        url = f"{base_url}{path}?{query}"
        try:
            with self._urlopen(_http_request(url), timeout=5.0) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            TimeoutError,
            OSError,
        ):
            return []


def _redeem_amounts(row: dict[str, Any]) -> list[int]:
    outcome_index = _int(row.get("outcomeIndex"), default=0)
    if outcome_index < 0:
        raise ValueError(f"outcomeIndex must be >= 0 (got {outcome_index})")
    size = Decimal(str(row.get("size", "0")))
    raw = int((size * Decimal("1000000")).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    ))
    if raw <= 0:
        raise ValueError("redeem size must be positive")
    # NegRisk markets can have 3+ outcomes; size the amounts vector to cover
    # this outcome's index (the adapter accepts a sparse/zero-padded vector).
    amounts = [0] * (outcome_index + 1)
    amounts[outcome_index] = raw
    return amounts


def _condition_id(row: dict[str, Any]) -> str:
    value = row.get("conditionId") or row.get("condition_id")
    if not isinstance(value, str) or not value:
        raise ValueError("row missing conditionId")
    return value.lower()


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "positions", "markets", "items", "results"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
