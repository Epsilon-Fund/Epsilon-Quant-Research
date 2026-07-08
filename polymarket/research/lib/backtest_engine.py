"""Minimal latency-aware paper engine for Dali live CLOB captures."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lib.clob_book import ClobBook, OfiContribution


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_received_at(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, utc=True)


def parse_levels(raw: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        price = as_float(item.get("price"))
        size = as_float(item.get("size"))
        if price is not None and size is not None:
            out.append((price, size))
    return out


@dataclass(frozen=True)
class BacktestConfig:
    latency_median_ms: float = 300.0
    latency_p99_ms: float = 900.0
    latency_seed: int = 7
    fixed_latency_ms: float | None = None
    order_size: float = 5.0
    tick_size: float = 0.001
    fee_rate: float = 0.0
    min_fill_size: float = 1.0
    skip_incomplete_book: bool = True
    rule_mode: str = "every_last_trade_price"
    allowed_sides: tuple[str, ...] = ("BUY", "SELL")
    take_profit_ticks: float = 2.0
    stop_loss_ticks: float = 2.0
    time_stop_seconds: float = 180.0
    ofi_reversal_threshold: float = 0.0
    force_close_at_end: bool = True


@dataclass
class PendingOrder:
    asset_id: str
    side: str
    size: float
    signal_time: pd.Timestamp
    target_time: pd.Timestamp
    latency_ms: float


@dataclass
class Position:
    asset_id: str
    side: str
    size: float
    entry_time: pd.Timestamp
    entry_price: float
    entry_notional: float
    entry_fee: float
    latency_ms: float
    partial_fill: bool
    implied_entry_spread_cost: float | None


def load_config(path: Path) -> BacktestConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    latency = raw.get("latency") or {}
    execution = raw.get("execution") or {}
    rule = raw.get("rule") or {}
    return BacktestConfig(
        latency_median_ms=float(latency.get("median_ms", 300)),
        latency_p99_ms=float(latency.get("p99_ms", 900)),
        latency_seed=int(latency.get("seed", 7)),
        fixed_latency_ms=(
            float(latency["fixed_ms"])
            if latency.get("fixed_ms") is not None
            else None
        ),
        order_size=float(execution.get("order_size", 5)),
        tick_size=float(execution.get("tick_size", 0.001)),
        fee_rate=float(execution.get("fee_rate", 0.0)),
        min_fill_size=float(execution.get("min_fill_size", 1)),
        skip_incomplete_book=bool(execution.get("skip_incomplete_book", True)),
        rule_mode=str(rule.get("mode", "every_last_trade_price")),
        allowed_sides=tuple(str(x).upper() for x in rule.get("allowed_sides", ["BUY", "SELL"])),
        take_profit_ticks=float(rule.get("take_profit_ticks", 2)),
        stop_loss_ticks=float(rule.get("stop_loss_ticks", 2)),
        time_stop_seconds=float(rule.get("time_stop_seconds", 180)),
        ofi_reversal_threshold=float(rule.get("ofi_reversal_threshold", 0)),
        force_close_at_end=bool(rule.get("force_close_at_end", True)),
    )


class LatencyModel:
    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.latency_seed)
        if config.latency_median_ms > 0 and config.latency_p99_ms > config.latency_median_ms:
            self._mu = math.log(config.latency_median_ms)
            self._sigma = (
                math.log(config.latency_p99_ms) - math.log(config.latency_median_ms)
            ) / 2.3263478740408408
        else:
            self._mu = math.log(max(config.latency_median_ms, 1e-9))
            self._sigma = 0.0

    def sample_ms(self) -> float:
        if self._config.fixed_latency_ms is not None:
            return self._config.fixed_latency_ms
        return float(self._rng.lognormal(self._mu, self._sigma))


class BacktestEngine:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.latency = LatencyModel(config)
        self.books: dict[str, ClobBook] = {}
        self.pending: list[PendingOrder] = []
        self.positions: list[Position] = []
        self.journal: list[dict[str, Any]] = []
        self.skipped_orders: list[dict[str, Any]] = []
        self.last_event_time: pd.Timestamp | None = None

    def book(self, asset_id: str) -> ClobBook:
        return self.books.setdefault(asset_id, ClobBook())

    def run_jsonl(self, path: Path) -> pd.DataFrame:
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                received_at = parse_received_at(rec.get("received_at"))
                if received_at is None:
                    continue
                self.last_event_time = received_at
                self._execute_due_orders(received_at)
                ofi_by_asset = self._apply_market_event(rec)
                self._check_exits(received_at, ofi_by_asset)
                self._maybe_signal(rec, received_at)
        if self.config.force_close_at_end and self.last_event_time is not None:
            self._force_close(self.last_event_time)
        return pd.DataFrame(self.journal)

    def _apply_market_event(self, rec: dict[str, Any]) -> dict[str, OfiContribution]:
        msg = rec.get("message")
        if not isinstance(msg, dict):
            return {}
        event_type = str(rec.get("event_type") or msg.get("event_type") or "")
        out: dict[str, OfiContribution] = {}
        if event_type == "book":
            asset_id = str(msg.get("asset_id") or "")
            if asset_id:
                out[asset_id] = self.book(asset_id).replace(
                    parse_levels(msg.get("bids")),
                    parse_levels(msg.get("asks")),
                )
        elif event_type == "price_change":
            for change in msg.get("price_changes") or []:
                if not isinstance(change, dict):
                    continue
                asset_id = str(change.get("asset_id") or "")
                price = as_float(change.get("price"))
                size = as_float(change.get("size"))
                side = str(change.get("side") or "").upper()
                if asset_id and price is not None and size is not None and side in {"BUY", "SELL"}:
                    out[asset_id] = self.book(asset_id).update_level(side, price, size)
        return out

    def _maybe_signal(self, rec: dict[str, Any], received_at: pd.Timestamp) -> None:
        if self.config.rule_mode != "every_last_trade_price":
            return
        msg = rec.get("message")
        if not isinstance(msg, dict):
            return
        event_type = str(rec.get("event_type") or msg.get("event_type") or "")
        if event_type != "last_trade_price":
            return
        asset_id = str(msg.get("asset_id") or "")
        side = str(msg.get("side") or "").upper()
        if not asset_id or side not in self.config.allowed_sides:
            return
        latency_ms = self.latency.sample_ms()
        self.pending.append(
            PendingOrder(
                asset_id=asset_id,
                side=side,
                size=self.config.order_size,
                signal_time=received_at,
                target_time=received_at + pd.Timedelta(milliseconds=latency_ms),
                latency_ms=latency_ms,
            )
        )

    def _execute_due_orders(self, now: pd.Timestamp) -> None:
        due = [order for order in self.pending if order.target_time <= now]
        self.pending = [order for order in self.pending if order.target_time > now]
        for order in due:
            book = self.book(order.asset_id)
            if self.config.skip_incomplete_book and not book.is_complete:
                self.skipped_orders.append({"reason": "incomplete_book", **order.__dict__})
                continue
            position = self._execute_entry(order, book)
            if position is not None:
                self.positions.append(position)

    def _execute_entry(self, order: PendingOrder, book: ClobBook) -> Position | None:
        price, filled, notional = book.walk(order.side, order.size)
        if price is None or filled < self.config.min_fill_size:
            self.skipped_orders.append({"reason": "insufficient_depth", **order.__dict__})
            return None
        top = book.top()
        mid = book.mid()
        implied_spread_cost = None
        if mid is not None:
            implied_spread_cost = (
                price - mid if order.side == "BUY" else mid - price
            ) * filled
        return Position(
            asset_id=order.asset_id,
            side=order.side,
            size=filled,
            entry_time=order.target_time,
            entry_price=price,
            entry_notional=notional,
            entry_fee=notional * self.config.fee_rate,
            latency_ms=order.latency_ms,
            partial_fill=filled < order.size,
            implied_entry_spread_cost=implied_spread_cost,
        )

    def _check_exits(self, now: pd.Timestamp, ofi_by_asset: dict[str, OfiContribution]) -> None:
        keep: list[Position] = []
        for pos in self.positions:
            reason = self._exit_reason(pos, now, ofi_by_asset.get(pos.asset_id))
            if reason:
                if not self._close_position(pos, now, reason):
                    keep.append(pos)
            else:
                keep.append(pos)
        self.positions = keep

    def _exit_reason(
        self,
        pos: Position,
        now: pd.Timestamp,
        ofi: OfiContribution | None,
    ) -> str | None:
        book = self.book(pos.asset_id)
        exit_side = "SELL" if pos.side == "BUY" else "BUY"
        exit_price, filled, _ = book.walk(exit_side, pos.size)
        if exit_price is None or filled < self.config.min_fill_size:
            return None
        pnl_per_share = (
            exit_price - pos.entry_price
            if pos.side == "BUY"
            else pos.entry_price - exit_price
        )
        if pnl_per_share >= self.config.take_profit_ticks * self.config.tick_size:
            return "take_profit"
        if pnl_per_share <= -self.config.stop_loss_ticks * self.config.tick_size:
            return "stop_loss"
        age = (now - pos.entry_time).total_seconds()
        if age >= self.config.time_stop_seconds:
            return "time_stop"
        if ofi is not None and self.config.ofi_reversal_threshold > 0:
            if pos.side == "BUY" and ofi.combined <= -self.config.ofi_reversal_threshold:
                return "ofi_reversal"
            if pos.side == "SELL" and ofi.combined >= self.config.ofi_reversal_threshold:
                return "ofi_reversal"
        return None

    def _force_close(self, now: pd.Timestamp) -> None:
        keep = []
        for pos in self.positions:
            if not self._close_position(pos, now, "force_close_end"):
                keep.append(pos)
        self.positions = keep

    def _close_position(self, pos: Position, now: pd.Timestamp, reason: str) -> bool:
        book = self.book(pos.asset_id)
        exit_side = "SELL" if pos.side == "BUY" else "BUY"
        exit_price, filled, notional = book.walk(exit_side, pos.size)
        if exit_price is None or filled < self.config.min_fill_size:
            return False
        exit_fee = notional * self.config.fee_rate
        gross = (
            (exit_price - pos.entry_price) * filled
            if pos.side == "BUY"
            else (pos.entry_price - exit_price) * filled
        )
        net = gross - pos.entry_fee - exit_fee
        mid = book.mid()
        implied_exit_spread_cost = None
        if mid is not None:
            implied_exit_spread_cost = (
                mid - exit_price if exit_side == "SELL" else exit_price - mid
            ) * filled
        self.journal.append(
            {
                "asset_id": pos.asset_id,
                "side": pos.side,
                "entry_time": pos.entry_time,
                "entry_price": pos.entry_price,
                "exit_time": now,
                "exit_price": exit_price,
                "exit_reason": reason,
                "size": filled,
                "partial_fill": pos.partial_fill or filled < pos.size,
                "latency_ms": pos.latency_ms,
                "gross_pnl": gross,
                "fees": pos.entry_fee + exit_fee,
                "net_pnl": net,
                "entry_notional": pos.entry_notional,
                "exit_notional": notional,
                "implied_entry_spread_cost": pos.implied_entry_spread_cost,
                "implied_exit_spread_cost": implied_exit_spread_cost,
            }
        )
        return True
