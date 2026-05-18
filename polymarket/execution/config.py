"""Typed config dataclass loaded from environment.

Datetime convention: this codebase uses datetime.now(timezone.utc),
never datetime.utcnow(). Naive datetimes are rejected at the
event-construction boundary (see journal/events.py).
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_ADDRESS_RE = re.compile(r"^0x[0-9a-f]{40}$")
_VALID_ORDER_TYPES = frozenset({"FOK", "GTC", "IOC"})
_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_PRICING_MODES = frozenset({"leader_fill", "current_book"})

_REQUIRED_FIELDS: tuple[str, ...] = (
    "POLYMARKET_LEADER_ADDRESS",
    "POLYMARKET_PRIVATE_KEY",
    "POLYMARKET_API_KEY",
    "POLYMARKET_API_SECRET",
    "POLYMARKET_PASSPHRASE",
    "POLYMARKET_FUNDER",
)

_DEFAULTS: Mapping[str, str] = {
    "POLYMARKET_CHAIN_ID": "137",
    "POLYMARKET_SIGNATURE_TYPE": "1",
    "POLYMARKET_CLOB_URL": "https://clob.polymarket.com",
    "POLYMARKET_GAMMA_URL": "https://gamma-api.polymarket.com",
    "POLYMARKET_DATA_URL": "https://data-api.polymarket.com",
    "POLYMARKET_WS_URL": "wss://ws-live-data.polymarket.com",
    "POLYMARKET_MAX_CAPITAL_USD": "100",
    "POLYMARKET_PER_TRADE_CAP_USD": "50",
    "POLYMARKET_PER_MARKET_CAP_USD": "50",
    "POLYMARKET_SIZING_USD": "50",
    "POLYMARKET_MAX_OPEN_POSITIONS": "3",
    "POLYMARKET_DEFAULT_ORDER_TYPE": "FOK",
    "POLYMARKET_PRICING_MODE": "leader_fill",
    "POLYMARKET_PRICE_DEVIATION_PCT": "2.0",
    "POLYMARKET_DAILY_LOSS_HALT_USD": "200",
    "POLYMARKET_KILLSWITCH_PATH": "/tmp/polymarket_killswitch",
    "POLYMARKET_JOURNAL_DIR": "./journal_logs",
    "POLYMARKET_LOG_LEVEL": "INFO",
    "POLYMARKET_MAX_REAL_ORDERS": "5",
    "POLYMARKET_REQUIRE_OPERATOR_CONFIRM": "false",
}


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    leader_address: str
    private_key: str
    api_key: str
    api_secret: str
    passphrase: str
    funder: str
    chain_id: int
    signature_type: int
    clob_url: str
    gamma_url: str
    data_url: str
    ws_url: str
    max_capital_usd: float
    per_trade_cap_usd: float
    per_market_cap_usd: float
    sizing_usd: float
    max_open_positions: int
    default_order_type: str
    pricing_mode: str
    price_deviation_pct: float
    daily_loss_halt_usd: float
    killswitch_path: Path
    journal_dir: Path
    log_level: str
    max_real_orders: int
    require_operator_confirm: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ExecutionConfig:
        source: Mapping[str, str] = os.environ if env is None else env

        def _required(name: str) -> str:
            value = source.get(name)
            if value is None or value.strip() == "":
                raise ValueError(f"{name} is required and must be non-empty")
            return value

        def _optional(name: str) -> str:
            value = source.get(name)
            if value is None or value == "":
                return _DEFAULTS[name]
            return value

        def _addr(name: str) -> str:
            raw = _required(name).strip().lower()
            if not _ADDRESS_RE.match(raw):
                raise ValueError(
                    f"{name} must match ^0x[0-9a-f]{{40}}$ (got {raw!r})"
                )
            return raw

        def _int(name: str) -> int:
            raw = _optional(name)
            try:
                return int(raw)
            except ValueError:
                raise ValueError(f"{name} must be an int (got {raw!r})") from None

        def _positive_float(name: str) -> float:
            raw = _optional(name)
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"{name} must be a float (got {raw!r})") from None
            if value <= 0:
                raise ValueError(f"{name} must be > 0 (got {value})")
            return value

        def _nonempty_str(name: str) -> str:
            value = _optional(name)
            if not value:
                raise ValueError(f"{name} must be non-empty")
            return value

        order_type = _optional("POLYMARKET_DEFAULT_ORDER_TYPE")
        if order_type not in _VALID_ORDER_TYPES:
            raise ValueError(
                "POLYMARKET_DEFAULT_ORDER_TYPE must be one of "
                f"{sorted(_VALID_ORDER_TYPES)} (got {order_type!r})"
            )

        pricing_mode = _optional("POLYMARKET_PRICING_MODE")
        if pricing_mode not in _VALID_PRICING_MODES:
            raise ValueError(
                "POLYMARKET_PRICING_MODE must be one of "
                f"{sorted(_VALID_PRICING_MODES)} (got {pricing_mode!r})"
            )

        log_level = _optional("POLYMARKET_LOG_LEVEL")
        if log_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                "POLYMARKET_LOG_LEVEL must be one of "
                f"{sorted(_VALID_LOG_LEVELS)} (got {log_level!r})"
            )

        # Safety harness for real venue (only consulted when POLYMARKET_VENUE=real).
        max_real_orders_raw = _optional("POLYMARKET_MAX_REAL_ORDERS")
        try:
            max_real_orders_value = int(max_real_orders_raw)
        except ValueError:
            raise ValueError(
                "POLYMARKET_MAX_REAL_ORDERS must be an int "
                f"(got {max_real_orders_raw!r})"
            ) from None
        if max_real_orders_value < 1:
            raise ValueError(
                "POLYMARKET_MAX_REAL_ORDERS must be >= 1 "
                f"(got {max_real_orders_value})"
            )

        require_confirm_raw = _optional("POLYMARKET_REQUIRE_OPERATOR_CONFIRM").strip().lower()
        if require_confirm_raw == "true":
            require_confirm_value = True
        elif require_confirm_raw == "false":
            require_confirm_value = False
        else:
            raise ValueError(
                "POLYMARKET_REQUIRE_OPERATOR_CONFIRM must be 'true' or 'false' "
                f"(got {require_confirm_raw!r})"
            )

        return cls(
            leader_address=_addr("POLYMARKET_LEADER_ADDRESS"),
            private_key=_required("POLYMARKET_PRIVATE_KEY"),
            api_key=_required("POLYMARKET_API_KEY"),
            api_secret=_required("POLYMARKET_API_SECRET"),
            passphrase=_required("POLYMARKET_PASSPHRASE"),
            funder=_addr("POLYMARKET_FUNDER"),
            chain_id=_int("POLYMARKET_CHAIN_ID"),
            signature_type=_int("POLYMARKET_SIGNATURE_TYPE"),
            clob_url=_nonempty_str("POLYMARKET_CLOB_URL"),
            gamma_url=_nonempty_str("POLYMARKET_GAMMA_URL"),
            data_url=_nonempty_str("POLYMARKET_DATA_URL"),
            ws_url=_nonempty_str("POLYMARKET_WS_URL"),
            max_capital_usd=_positive_float("POLYMARKET_MAX_CAPITAL_USD"),
            per_trade_cap_usd=_positive_float("POLYMARKET_PER_TRADE_CAP_USD"),
            per_market_cap_usd=_positive_float("POLYMARKET_PER_MARKET_CAP_USD"),
            sizing_usd=_positive_float("POLYMARKET_SIZING_USD"),
            max_open_positions=_int("POLYMARKET_MAX_OPEN_POSITIONS"),
            default_order_type=order_type,
            pricing_mode=pricing_mode,
            price_deviation_pct=_positive_float("POLYMARKET_PRICE_DEVIATION_PCT"),
            daily_loss_halt_usd=_positive_float("POLYMARKET_DAILY_LOSS_HALT_USD"),
            killswitch_path=Path(_optional("POLYMARKET_KILLSWITCH_PATH")),
            journal_dir=Path(_optional("POLYMARKET_JOURNAL_DIR")),
            log_level=log_level,
            max_real_orders=max_real_orders_value,
            require_operator_confirm=require_confirm_value,
        )
