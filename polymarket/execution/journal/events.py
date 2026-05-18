"""Event vocabulary for the execution journal.

Original controlled vocabulary (per skeleton):
    SignalReceived, OrderSubmitted, FillRecorded, RiskHalt,
    KillSwitchTripped.

Added per PLAN.md:
    WatcherReconnected   (per CLAUDE.md "every reconnection — log to JSONL")
    AmbiguousSubmit      (per PLAN.md decision 9)
    LeaderFillDropped    (per PLAN.md decision 10)

Naming reconciliation between original vocabulary and PLAN.md drafts:
    "OrderFilled" mentioned in PLAN.md decision 10 == FillRecorded.
        We use FillRecorded — it was named first in the skeleton.
    "RiskHalted" mentioned in PLAN.md decision 9 == RiskHalt.
        We use RiskHalt — it was named first in the skeleton.

Datetime convention:
    All ts_utc fields in this codebase use datetime.now(timezone.utc),
    never datetime.utcnow(). Naive datetimes (tzinfo is None) and any
    non-UTC timezone are rejected at construction time.

Drop-reason vocabulary:
    LeaderFillDropped.reason is validated against
    LEADER_FILL_DROP_REASONS = {"no_position", "duplicate"}.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

LEADER_FILL_DROP_REASONS: frozenset[str] = frozenset({
    "no_position", "duplicate", "leader_no_position",
})
WATCHER_MALFORMED_REASONS: frozenset[str] = frozenset({
    "json_parse",
    "missing_field",
    "wrong_type",
    "wrong_topic",
    "payload_parse",
})


def _validate_ts_utc(ts: datetime) -> None:
    if ts.tzinfo is None:
        raise ValueError("ts_utc must be tz-aware (got naive datetime)")
    offset = ts.utcoffset()
    if offset != timedelta(0):
        raise ValueError(f"ts_utc must be UTC (got offset {offset})")


@dataclass(frozen=True, slots=True)
class _BaseEvent:
    ts_utc: datetime

    event_type: ClassVar[str] = "BASE_EVENT"

    def __post_init__(self) -> None:
        _validate_ts_utc(self.ts_utc)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "event_type": type(self).event_type,
            "ts_utc": self.ts_utc.isoformat(),
        }
        for f in fields(self):
            if f.name == "ts_utc":
                continue
            value = getattr(self, f.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, Path):
                value = str(value)
            out[f.name] = value
        return out


@dataclass(frozen=True, slots=True)
class SignalReceived(_BaseEvent):
    signal_kind: str
    condition_id: str
    asset_id: str

    event_type: ClassVar[str] = "SIGNAL_RECEIVED"


@dataclass(frozen=True, slots=True)
class OrderSubmitted(_BaseEvent):
    client_order_id: str
    condition_id: str
    asset_id: str
    side: str
    size: float
    price: float
    order_type: str

    event_type: ClassVar[str] = "ORDER_SUBMITTED"


@dataclass(frozen=True, slots=True)
class OrderRejected(_BaseEvent):
    client_order_id: str
    reason: str
    detail: str

    event_type: ClassVar[str] = "ORDER_REJECTED"


@dataclass(frozen=True, slots=True)
class OrderAcknowledged(_BaseEvent):
    client_order_id: str
    venue_order_id: str

    event_type: ClassVar[str] = "ORDER_ACKNOWLEDGED"


@dataclass(frozen=True, slots=True)
class FillRecorded(_BaseEvent):
    transaction_hash: str
    condition_id: str
    asset_id: str
    side: str
    size: float
    price: float
    proxy_wallet: str

    event_type: ClassVar[str] = "FILL_RECORDED"


@dataclass(frozen=True, slots=True)
class RiskHalt(_BaseEvent):
    reason: str
    detail: str

    event_type: ClassVar[str] = "RISK_HALT"


@dataclass(frozen=True, slots=True)
class KillSwitchTripped(_BaseEvent):
    path: str

    event_type: ClassVar[str] = "KILL_SWITCH_TRIPPED"


@dataclass(frozen=True, slots=True)
class WatcherReconnected(_BaseEvent):
    gap_seconds: float

    event_type: ClassVar[str] = "WATCHER_RECONNECTED"


@dataclass(frozen=True, slots=True)
class WatcherStarted(_BaseEvent):
    ws_url: str

    event_type: ClassVar[str] = "WATCHER_STARTED"


@dataclass(frozen=True, slots=True)
class WatcherConnectFailed(_BaseEvent):
    ws_url: str
    attempt_number: int
    error: str

    event_type: ClassVar[str] = "WATCHER_CONNECT_FAILED"


@dataclass(frozen=True, slots=True)
class WatcherStopped(_BaseEvent):
    event_type: ClassVar[str] = "WATCHER_STOPPED"


@dataclass(frozen=True, slots=True)
class WatcherMalformedMessage(_BaseEvent):
    raw: str
    reason: str

    event_type: ClassVar[str] = "WATCHER_MALFORMED_MESSAGE"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reason not in WATCHER_MALFORMED_REASONS:
            raise ValueError(
                "WatcherMalformedMessage.reason must be one of "
                f"{sorted(WATCHER_MALFORMED_REASONS)} "
                f"(got {self.reason!r}); see WATCHER_MALFORMED_REASONS"
            )


@dataclass(frozen=True, slots=True)
class LeaderFillObserved(_BaseEvent):
    transaction_hash: str
    condition_id: str
    asset_id: str
    side: str
    size: float
    price: float
    proxy_wallet: str
    observed_at_utc: datetime

    event_type: ClassVar[str] = "LEADER_FILL_OBSERVED"

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_ts_utc(self.observed_at_utc)


@dataclass(frozen=True, slots=True)
class AmbiguousSubmit(_BaseEvent):
    client_order_id: str
    detail: str

    event_type: ClassVar[str] = "AMBIGUOUS_SUBMIT"


@dataclass(frozen=True, slots=True)
class MirrorSignalEmitted(_BaseEvent):
    signal_id: str
    kind: str
    condition_id: str
    asset_id: str
    side: str
    target_size_shares: float
    leader_fill_price: float

    event_type: ClassVar[str] = "MIRROR_SIGNAL_EMITTED"


@dataclass(frozen=True, slots=True)
class LeaderFillDropped(_BaseEvent):
    transaction_hash: str
    reason: str

    event_type: ClassVar[str] = "LEADER_FILL_DROPPED"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reason not in LEADER_FILL_DROP_REASONS:
            raise ValueError(
                "LeaderFillDropped.reason must be one of "
                f"{sorted(LEADER_FILL_DROP_REASONS)} "
                f"(got {self.reason!r}); see LEADER_FILL_DROP_REASONS"
            )
