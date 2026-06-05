"""Classify a fill as ENTRY, EXIT, or SCALE based on leader's prior positions.

Reads today's + yesterday's journal at construction time to rebuild
both bot and leader position ledgers. Subsequent process_fill calls
mutate leader state (bot state is updated only via FillRecorded
events that mirror_engine writes, not here).

Per PLAN.md decision 10, EXIT signals where the bot holds no position
are dropped with reason "no_position". Per the leader-side analogue,
SELLs where the leader has no recorded position are dropped with
reason "leader_no_position" — the bot can't compute an exit fraction
without a denominator.
"""
from __future__ import annotations

import queue
import uuid
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Any

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import (
    JsonlWriter,
    LeaderFillDropped,
    MirrorSignalEmitted,
)

from .dedup import Deduplicator
from .types import MirrorSignal, Position, SignalKind, position_key


class Classifier:
    def __init__(
        self,
        config: ExecutionConfig,
        journal: JsonlWriter,
        dedup: Deduplicator,
        signal_queue: "queue.Queue[MirrorSignal]",
        today_utc: date | None = None,
    ) -> None:
        self._config: ExecutionConfig = config
        self._journal: JsonlWriter = journal
        self._dedup: Deduplicator = dedup
        self._signals: queue.Queue[MirrorSignal] = signal_queue
        self._bot_positions: dict[tuple[str, str], Position] = {}
        self._leader_positions: dict[tuple[str, str], Position] = {}
        self._rebuild_state_from_journal(today_utc)

    def _rebuild_state_from_journal(self, today_utc: date | None) -> None:
        today = today_utc if today_utc is not None else datetime.now(timezone.utc).date()
        # Apply yesterday's events first, then today's, in chronological order.
        for day in (today - timedelta(days=1), today):
            for event in self._journal.read_today(today_utc=day):
                self._absorb(event)

    def _absorb(self, event: dict[str, Any]) -> None:
        et = event.get("event_type")
        if et == "FILL_RECORDED":
            self._apply(self._bot_positions, event)
        elif et == "LEADER_FILL_OBSERVED":
            self._apply(self._leader_positions, event)

    @staticmethod
    def _apply(
        positions: dict[tuple[str, str], Position],
        event: dict[str, Any],
    ) -> None:
        side = event.get("side")
        try:
            shares = float(event.get("size", 0.0))
            price = float(event.get("price", 0.0))
        except (TypeError, ValueError):
            return
        condition_id = event.get("condition_id")
        asset_id = event.get("asset_id")
        if not isinstance(condition_id, str) or not isinstance(asset_id, str):
            return
        if side == "BUY":
            Classifier._apply_buy(positions, condition_id, asset_id, side, shares, price)
        elif side == "SELL":
            Classifier._apply_sell(positions, condition_id, asset_id, shares)

    @staticmethod
    def _apply_buy(
        positions: dict[tuple[str, str], Position],
        condition_id: str,
        asset_id: str,
        side: str,
        shares: float,
        price: float,
    ) -> None:
        key = position_key(condition_id, asset_id)
        existing = positions.get(key)
        if existing is None:
            positions[key] = Position(
                condition_id=condition_id.lower(),
                asset_id=asset_id,
                side=side,
                shares=shares,
                avg_entry_price=price,
                total_entry_usd=shares * price,
            )
            return
        new_shares = existing.shares + shares
        new_total = existing.total_entry_usd + (shares * price)
        new_avg = new_total / new_shares if new_shares > 0 else 0.0
        positions[key] = replace(
            existing,
            shares=new_shares,
            avg_entry_price=new_avg,
            total_entry_usd=new_total,
        )

    @staticmethod
    def _apply_sell(
        positions: dict[tuple[str, str], Position],
        condition_id: str,
        asset_id: str,
        shares: float,
    ) -> None:
        key = position_key(condition_id, asset_id)
        existing = positions.get(key)
        if existing is None:
            return
        new_shares = max(0.0, existing.shares - shares)
        if new_shares == 0.0:
            del positions[key]
            return
        new_total = existing.total_entry_usd * (new_shares / existing.shares)
        positions[key] = replace(
            existing, shares=new_shares, total_entry_usd=new_total
        )

    def process_fill(self, fill: Any) -> None:
        """Consume a LeaderFillObserved event from the watcher queue."""
        # 1. Dedup
        if self._dedup.is_duplicate(fill.transaction_hash):
            self._journal.write(LeaderFillDropped(
                ts_utc=datetime.now(timezone.utc),
                transaction_hash=fill.transaction_hash,
                reason="duplicate",
            ))
            return

        key = position_key(fill.condition_id, fill.asset_id)

        # 2. Classify
        signal: MirrorSignal | None = None
        if fill.side == "BUY":
            target = self._config.sizing_usd / fill.price
            signal = MirrorSignal(
                signal_id=uuid.uuid4().hex,
                kind=SignalKind.ENTRY,
                condition_id=fill.condition_id,
                asset_id=fill.asset_id,
                side=fill.side,
                target_size_shares=target,
                leader_fill_price=fill.price,
                source_transaction_hash=fill.transaction_hash,
            )
        elif fill.side == "SELL":
            leader_pos = self._leader_positions.get(key)
            if leader_pos is None or leader_pos.shares <= 0:
                self._journal.write(LeaderFillDropped(
                    ts_utc=datetime.now(timezone.utc),
                    transaction_hash=fill.transaction_hash,
                    reason="leader_no_position",
                ))
                # apply_sell is a no-op when no position exists; still call defensively.
                self._apply_sell(
                    self._leader_positions, fill.condition_id, fill.asset_id, fill.size
                )
                self._dedup.mark_seen(fill.transaction_hash)
                return
            bot_pos = self._bot_positions.get(key)
            if bot_pos is None or bot_pos.shares <= 0:
                self._journal.write(LeaderFillDropped(
                    ts_utc=datetime.now(timezone.utc),
                    transaction_hash=fill.transaction_hash,
                    reason="no_position",
                ))
                self._apply_sell(
                    self._leader_positions, fill.condition_id, fill.asset_id, fill.size
                )
                self._dedup.mark_seen(fill.transaction_hash)
                return
            exit_fraction = fill.size / leader_pos.shares
            bot_exit_shares = exit_fraction * bot_pos.shares
            signal = MirrorSignal(
                signal_id=uuid.uuid4().hex,
                kind=SignalKind.EXIT,
                condition_id=fill.condition_id,
                asset_id=fill.asset_id,
                side=fill.side,
                target_size_shares=bot_exit_shares,
                leader_fill_price=fill.price,
                source_transaction_hash=fill.transaction_hash,
            )

        # 3. Update leader state
        if fill.side == "BUY":
            self._apply_buy(
                self._leader_positions, fill.condition_id, fill.asset_id,
                fill.side, fill.size, fill.price,
            )
        elif fill.side == "SELL":
            self._apply_sell(
                self._leader_positions, fill.condition_id, fill.asset_id, fill.size
            )

        # 4. Mark dedup
        self._dedup.mark_seen(fill.transaction_hash)

        # 5. + 6. Journal and push (only if a signal was constructed).
        if signal is not None:
            self._journal.write(MirrorSignalEmitted(
                ts_utc=datetime.now(timezone.utc),
                signal_id=signal.signal_id,
                kind=signal.kind.value,
                condition_id=signal.condition_id,
                asset_id=signal.asset_id,
                side=signal.side,
                target_size_shares=signal.target_size_shares,
                leader_fill_price=signal.leader_fill_price,
            ))
            self._signals.put(signal, block=True)
