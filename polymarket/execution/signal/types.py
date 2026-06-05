"""FillEvent, MirrorSignal, SignalKind dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SignalKind(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


@dataclass(frozen=True, slots=True)
class Position:
    condition_id: str
    asset_id: str
    side: str
    shares: float
    avg_entry_price: float
    total_entry_usd: float


@dataclass(frozen=True, slots=True)
class MirrorSignal:
    signal_id: str
    kind: SignalKind
    condition_id: str
    asset_id: str
    side: str
    target_size_shares: float
    leader_fill_price: float
    source_transaction_hash: str


def position_key(condition_id: str, asset_id: str) -> tuple[str, str]:
    return (condition_id.lower(), asset_id)
