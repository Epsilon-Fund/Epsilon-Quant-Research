"""Composite NegRisk basket inventory tracker.

RTDS emits on-book trades but not split/merge/redeem/convert actions.
This poller watches the Data API activity endpoint and keeps a small
in-memory basket exposure ledger per condition_id.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polymarket.execution.journal import BasketInventoryUpdated, JsonlWriter

_ACTIVITY_TYPES = ("SPLIT", "MERGE", "REDEEM", "CONVERSION")
_STOP_JOIN_TIMEOUT_S = 5.0
_HTTP_USER_AGENT = "curl/8.0"


def _http_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": _HTTP_USER_AGENT})


class NegRiskInventoryTracker:
    """Polls Data API activity and tracks net basket exposure."""

    def __init__(
        self,
        *,
        funder: str,
        data_url: str,
        state_path: Path,
        journal: JsonlWriter | None = None,
        poll_interval_seconds: float = 30.0,
        urlopen_fn: Callable[..., Any] | None = None,
    ) -> None:
        if not funder:
            raise ValueError("funder must be non-empty")
        if not data_url:
            raise ValueError("data_url must be non-empty")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        self._funder = funder.lower()
        self._data_url = data_url.rstrip("/")
        self._state_path = Path(state_path)
        self._journal = journal
        self._poll_interval = poll_interval_seconds
        self._urlopen = urlopen_fn if urlopen_fn is not None else urllib.request.urlopen
        self._basket_inventory: dict[str, float] = {}
        self._processed_activity_ids: set[str] = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._replay_state()

    def get_basket_exposure(self, condition_id: str) -> float:
        return self._basket_inventory.get(condition_id.lower(), 0.0)

    def poll_once(self) -> int:
        rows = self._fetch_activity()
        applied = 0
        for row in rows:
            parsed = _parse_activity_row(row)
            if parsed is None:
                continue
            activity_id, condition_id, activity_type, delta, tx_hash = parsed
            if activity_id in self._processed_activity_ids:
                continue
            self._processed_activity_ids.add(activity_id)
            new_exposure = self.get_basket_exposure(condition_id) + delta
            self._basket_inventory[condition_id] = new_exposure
            self._persist_update(
                activity_id=activity_id,
                condition_id=condition_id,
                activity_type=activity_type,
                delta=delta,
                exposure=new_exposure,
                transaction_hash=tx_hash,
            )
            if self._journal is not None:
                self._journal.write(BasketInventoryUpdated(
                    ts_utc=datetime.now(timezone.utc),
                    condition_id=condition_id,
                    activity_type=activity_type,
                    delta=delta,
                    exposure=new_exposure,
                    transaction_hash=tx_hash,
                ))
            applied += 1
        return applied

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="negrisk_inventory_poller",
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
                # Inventory freshness is important, but a transient poll
                # failure must not kill the process.
                pass
            if self._stop_event.wait(self._poll_interval):
                break

    def _fetch_activity(self) -> list[dict[str, Any]]:
        qs = urllib.parse.urlencode({
            "user": self._funder,
            "type": ",".join(_ACTIVITY_TYPES),
        })
        url = f"{self._data_url}/activity?{qs}"
        try:
            with self._urlopen(_http_request(url), timeout=5.0) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            TimeoutError,
            OSError,
        ):
            return []
        return _extract_rows(payload)

    def _persist_update(
        self,
        *,
        activity_id: str,
        condition_id: str,
        activity_type: str,
        delta: float,
        exposure: float,
        transaction_hash: str,
    ) -> None:
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "activity_id": activity_id,
            "condition_id": condition_id,
            "activity_type": activity_type,
            "delta": delta,
            "exposure": exposure,
            "transaction_hash": transaction_hash,
        }
        with open(self._state_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _replay_state(self) -> None:
        if not self._state_path.exists():
            return
        with open(self._state_path, "r", encoding="utf-8") as f:
            for raw in f:
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                activity_id = record.get("activity_id")
                condition_id = record.get("condition_id")
                if not isinstance(activity_id, str) or not isinstance(condition_id, str):
                    continue
                self._processed_activity_ids.add(activity_id)
                try:
                    exposure = float(record.get("exposure"))
                except (TypeError, ValueError):
                    continue
                self._basket_inventory[condition_id.lower()] = exposure


def _parse_activity_row(
    row: dict[str, Any],
) -> tuple[str, str, str, float, str] | None:
    activity_type = str(row.get("type", "")).upper()
    if activity_type not in _ACTIVITY_TYPES:
        return None
    condition_id = row.get("conditionId") or row.get("condition_id")
    if not isinstance(condition_id, str) or not condition_id:
        return None
    condition_id = condition_id.lower()
    qty = _quantity(row)
    if qty <= 0:
        return None
    sign = 1.0 if activity_type == "SPLIT" else -1.0
    if activity_type == "CONVERSION":
        sign = _conversion_sign(row)
    delta = sign * qty
    tx_hash = str(
        row.get("transactionHash")
        or row.get("transaction_hash")
        or row.get("txHash")
        or ""
    )
    activity_id = "|".join((
        tx_hash,
        condition_id,
        activity_type,
        str(row.get("timestamp", "")),
        str(row.get("asset", "")),
    ))
    return activity_id, condition_id, activity_type, delta, tx_hash


def _quantity(row: dict[str, Any]) -> float:
    for key in ("size", "amount", "quantity", "shares"):
        raw = row.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    amounts = row.get("amounts")
    if isinstance(amounts, list):
        total = 0.0
        for raw in amounts:
            try:
                total += float(raw)
            except (TypeError, ValueError):
                continue
        return total
    return 0.0


def _conversion_sign(row: dict[str, Any]) -> float:
    raw = row.get("delta")
    if raw is not None:
        try:
            return 1.0 if float(raw) >= 0 else -1.0
        except (TypeError, ValueError):
            pass
    return -1.0


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "activity", "items", "results"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []
