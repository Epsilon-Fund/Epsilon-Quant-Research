"""Tests for polymarket/execution/config.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from polymarket.execution.config import ExecutionConfig

_VALID_LEADER = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"
_VALID_FUNDER = "0x" + "ab" * 20

_REQUIRED = {
    "POLYMARKET_LEADER_ADDRESS": _VALID_LEADER,
    "POLYMARKET_PRIVATE_KEY": "0xprivkey",
    "POLYMARKET_API_KEY": "k",
    "POLYMARKET_API_SECRET": "s",
    "POLYMARKET_PASSPHRASE": "p",
    "POLYMARKET_FUNDER": _VALID_FUNDER,
}

_FULL_ENV = {
    **_REQUIRED,
    "POLYMARKET_CHAIN_ID": "137",
    "POLYMARKET_SIGNATURE_TYPE": "1",
    "POLYMARKET_CLOB_URL": "https://clob.polymarket.com",
    "POLYMARKET_GAMMA_URL": "https://gamma-api.polymarket.com",
    "POLYMARKET_DATA_URL": "https://data-api.polymarket.com",
    "POLYMARKET_WS_URL": "wss://ws-live-data.polymarket.com",
    "POLYMARKET_MAX_CAPITAL_USD": "100",
    "POLYMARKET_PER_TRADE_CAP_USD": "20",
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


def test_happy_path_full_env() -> None:
    cfg = ExecutionConfig.from_env(env=_FULL_ENV)
    assert cfg.leader_address == _VALID_LEADER
    assert cfg.funder == _VALID_FUNDER
    assert cfg.chain_id == 137
    assert cfg.signature_type == 1
    assert cfg.max_capital_usd == 100.0
    assert cfg.sizing_usd == 50.0
    assert cfg.max_open_positions == 3
    assert cfg.default_order_type == "FOK"
    assert cfg.pricing_mode == "leader_fill"
    assert cfg.log_level == "INFO"
    assert isinstance(cfg.killswitch_path, Path)
    assert isinstance(cfg.journal_dir, Path)
    assert cfg.killswitch_path == Path("/tmp/polymarket_killswitch")
    assert cfg.journal_dir == Path("./journal_logs")


@pytest.mark.parametrize("missing", list(_REQUIRED.keys()))
def test_missing_required_field_raises(missing: str) -> None:
    env = dict(_FULL_ENV)
    env[missing] = ""
    with pytest.raises(ValueError, match=missing):
        ExecutionConfig.from_env(env=env)


def test_leader_address_uppercase_is_lowercased() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_LEADER_ADDRESS"] = _VALID_LEADER.upper().replace("0X", "0x")
    cfg = ExecutionConfig.from_env(env=env)
    assert cfg.leader_address == _VALID_LEADER


def test_leader_address_bad_format_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_LEADER_ADDRESS"] = "not_an_address"
    with pytest.raises(ValueError, match="POLYMARKET_LEADER_ADDRESS"):
        ExecutionConfig.from_env(env=env)


def test_funder_bad_format_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_FUNDER"] = "0xZZ"
    with pytest.raises(ValueError, match="POLYMARKET_FUNDER"):
        ExecutionConfig.from_env(env=env)


def test_default_order_type_invalid_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_DEFAULT_ORDER_TYPE"] = "BAD"
    with pytest.raises(ValueError, match="POLYMARKET_DEFAULT_ORDER_TYPE"):
        ExecutionConfig.from_env(env=env)


def test_pricing_mode_default_is_leader_fill() -> None:
    cfg = ExecutionConfig.from_env(env=dict(_REQUIRED))
    assert cfg.pricing_mode == "leader_fill"


def test_pricing_mode_current_book_accepted() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_PRICING_MODE"] = "current_book"
    cfg = ExecutionConfig.from_env(env=env)
    assert cfg.pricing_mode == "current_book"


def test_max_real_orders_default_is_5() -> None:
    cfg = ExecutionConfig.from_env(env=dict(_REQUIRED))
    assert cfg.max_real_orders == 5


def test_max_real_orders_zero_is_observe_only() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_MAX_REAL_ORDERS"] = "0"
    cfg = ExecutionConfig.from_env(env=env)
    assert cfg.max_real_orders == 0


def test_max_real_orders_negative_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_MAX_REAL_ORDERS"] = "-1"
    with pytest.raises(ValueError, match="POLYMARKET_MAX_REAL_ORDERS"):
        ExecutionConfig.from_env(env=env)


def test_max_real_orders_non_int_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_MAX_REAL_ORDERS"] = "abc"
    with pytest.raises(ValueError, match="POLYMARKET_MAX_REAL_ORDERS"):
        ExecutionConfig.from_env(env=env)


def test_require_operator_confirm_default_false() -> None:
    cfg = ExecutionConfig.from_env(env=dict(_REQUIRED))
    assert cfg.require_operator_confirm is False


def test_require_operator_confirm_true_accepted() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_REQUIRE_OPERATOR_CONFIRM"] = "TRUE"
    cfg = ExecutionConfig.from_env(env=env)
    assert cfg.require_operator_confirm is True


def test_require_operator_confirm_invalid_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_REQUIRE_OPERATOR_CONFIRM"] = "garbage"
    with pytest.raises(ValueError, match="POLYMARKET_REQUIRE_OPERATOR_CONFIRM"):
        ExecutionConfig.from_env(env=env)


def test_pricing_mode_invalid_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_PRICING_MODE"] = "garbage"
    with pytest.raises(ValueError, match="POLYMARKET_PRICING_MODE"):
        ExecutionConfig.from_env(env=env)


def test_log_level_invalid_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_LOG_LEVEL"] = "verbose"
    with pytest.raises(ValueError, match="POLYMARKET_LOG_LEVEL"):
        ExecutionConfig.from_env(env=env)


def test_negative_sizing_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_SIZING_USD"] = "-5"
    with pytest.raises(ValueError, match="POLYMARKET_SIZING_USD"):
        ExecutionConfig.from_env(env=env)


def test_zero_sizing_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_SIZING_USD"] = "0"
    with pytest.raises(ValueError, match="POLYMARKET_SIZING_USD"):
        ExecutionConfig.from_env(env=env)


def test_optional_defaults_match_env_example() -> None:
    cfg = ExecutionConfig.from_env(env=dict(_REQUIRED))
    assert cfg.chain_id == 137
    assert cfg.signature_type == 1
    assert cfg.clob_url == "https://clob.polymarket.com"
    assert cfg.gamma_url == "https://gamma-api.polymarket.com"
    assert cfg.data_url == "https://data-api.polymarket.com"
    assert cfg.ws_url == "wss://ws-live-data.polymarket.com"
    assert cfg.max_capital_usd == 100.0
    assert cfg.per_trade_cap_usd == 50.0
    assert cfg.per_market_cap_usd == 50.0
    assert cfg.sizing_usd == 50.0
    assert cfg.max_open_positions == 3
    assert cfg.default_order_type == "FOK"
    assert cfg.pricing_mode == "leader_fill"
    assert cfg.price_deviation_pct == 2.0
    assert cfg.daily_loss_halt_usd == 200.0
    assert cfg.killswitch_path == Path("/tmp/polymarket_killswitch")
    assert cfg.journal_dir == Path("./journal_logs")
    assert cfg.log_level == "INFO"


def test_int_field_invalid_raises() -> None:
    env = dict(_FULL_ENV)
    env["POLYMARKET_CHAIN_ID"] = "not_an_int"
    with pytest.raises(ValueError, match="POLYMARKET_CHAIN_ID"):
        ExecutionConfig.from_env(env=env)


def test_journal_dir_is_not_created_by_config() -> None:
    env = dict(_REQUIRED)
    env["POLYMARKET_JOURNAL_DIR"] = "/nonexistent/path/to/journal"
    cfg = ExecutionConfig.from_env(env=env)
    assert cfg.journal_dir == Path("/nonexistent/path/to/journal")
    assert not cfg.journal_dir.exists()
