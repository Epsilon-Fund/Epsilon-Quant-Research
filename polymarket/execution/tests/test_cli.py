"""Tests for cli.py — venue dispatch + early-exit code paths.

The main loop itself is not unit-tested (it's threaded glue);
end-to-end behaviour is covered by manual smoke runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from polymarket.execution import cli
from polymarket.execution.config import ExecutionConfig

_LEADER = "0x" + "a" * 40
_FUNDER = "0x" + "b" * 40


def _config(**overrides: object) -> ExecutionConfig:
    base: dict[str, object] = {
        "leader_address": _LEADER,
        "private_key": "0xpk",
        "api_key": "k",
        "api_secret": "s",
        "passphrase": "p",
        "funder": _FUNDER,
        "chain_id": 137,
        "signature_type": 1,
        "clob_url": "https://clob.polymarket.com",
        "gamma_url": "https://gamma-api.polymarket.com",
        "data_url": "https://data-api.polymarket.com",
        "ws_url": "wss://ws-live-data.polymarket.com",
        "max_capital_usd": 100.0,
        "per_trade_cap_usd": 20.0,
        "per_market_cap_usd": 50.0,
        "sizing_usd": 50.0,
        "max_open_positions": 3,
        "default_order_type": "FOK",
        "pricing_mode": "leader_fill",
        "price_deviation_pct": 2.0,
        "daily_loss_halt_usd": 200.0,
        "killswitch_path": Path("/tmp/polymarket_killswitch_unused"),
        "journal_dir": Path("./journal_logs"),
        "log_level": "INFO",
    "max_real_orders": 5,
    "require_operator_confirm": False,
    }
    base.update(overrides)
    return ExecutionConfig(**base)  # type: ignore[arg-type]


def _full_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env: dict[str, str] = {
        "POLYMARKET_LEADER_ADDRESS": _LEADER,
        "POLYMARKET_PRIVATE_KEY": "0x" + "0" * 64,
        "POLYMARKET_API_KEY": "k",
        "POLYMARKET_API_SECRET": "s",
        "POLYMARKET_PASSPHRASE": "p",
        "POLYMARKET_FUNDER": _FUNDER,
        "POLYMARKET_JOURNAL_DIR": str(tmp_path / "journal"),
        "POLYMARKET_KILLSWITCH_PATH": str(tmp_path / "killswitch"),
    }
    env.update(overrides)
    return env


# --- build_venue_adapter ----------------------------------------

def test_build_venue_adapter_fake_returns_print_adapter() -> None:
    adapter = cli.build_venue_adapter("fake", _config())
    assert hasattr(adapter, "submit_order")
    assert hasattr(adapter, "cancel_order")
    # Sanity check the print adapter actually returns a SubmitResult-shaped object.
    result = adapter.submit_order(
        client_order_id="coid", condition_id="0xcond", asset_id="42",
        side="BUY", size_shares=10.0, price=0.50, order_type="FOK",
    )
    assert result.accepted is True
    assert result.fill_size_shares == 10.0
    assert adapter.cancel_order(client_order_id="coid") == {"ambiguous": False}


def test_build_venue_adapter_real_raises_on_placeholder_credentials() -> None:
    # _config() sets all-zero private_key via "0xpk"; that's not a
    # placeholder by detection (mixed letters) but PASSPHRASE="p" and
    # API_KEY="k" are 1-char strings. Raise on the all-zeros fallback
    # by overriding private_key.
    cfg = _config(private_key="0x" + "0" * 64)
    with pytest.raises(ValueError, match="POLYMARKET_PRIVATE_KEY"):
        cli.build_venue_adapter("real", cfg, journal=None)


def test_build_venue_adapter_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError, match="POLYMARKET_VENUE"):
        cli.build_venue_adapter("garbage", _config())


# --- _looks_like_placeholder + _validate_real_credentials -----------

@pytest.mark.parametrize("value,expected", [
    ("0x" + "0" * 64, True),
    ("0x" + "0" * 40, True),
    ("dummy", True),
    ("DUMMY", True),
    ("", True),
    ("placeholder", True),
    ("0x9d84ce0306f8551e02efef1680475fc0f1dc1344", False),
    ("k", False),  # 1-char keys aren't great but aren't a placeholder
    ("real-secret-12345", False),
])
def test_looks_like_placeholder(value, expected) -> None:
    assert cli._looks_like_placeholder(value) is expected


def test_validate_real_credentials_passes_with_real_values() -> None:
    cfg = _config(
        private_key="0x" + "9d84ce" * 10 + "abcd",
        api_key="real-key-abcdef",
        api_secret="real-secret-abc123",
        passphrase="real-passphrase",
        funder="0x9d84ce0306f8551e02efef1680475fc0f1dc1344",
    )
    cli._validate_real_credentials(cfg)


def test_validate_real_credentials_rejects_zero_private_key() -> None:
    cfg = _config(private_key="0x" + "0" * 64)
    with pytest.raises(ValueError, match="POLYMARKET_PRIVATE_KEY"):
        cli._validate_real_credentials(cfg)


def test_validate_real_credentials_rejects_dummy_api_key() -> None:
    cfg = _config(
        private_key="0x" + "9d84ce" * 10 + "abcd",
        api_key="dummy",
        api_secret="real",
        passphrase="real",
    )
    with pytest.raises(ValueError, match="POLYMARKET_API_KEY"):
        cli._validate_real_credentials(cfg)


def test_validate_real_credentials_rejects_empty_passphrase() -> None:
    cfg = _config(
        private_key="0x" + "9d84ce" * 10 + "abcd",
        api_key="real",
        api_secret="real",
        passphrase="",
    )
    with pytest.raises(ValueError, match="POLYMARKET_PASSPHRASE"):
        cli._validate_real_credentials(cfg)


def test_validate_real_credentials_rejects_zero_funder() -> None:
    cfg = _config(
        private_key="0x" + "9d84ce" * 10 + "abcd",
        api_key="real",
        api_secret="real",
        passphrase="real",
        funder="0x" + "0" * 40,
    )
    with pytest.raises(ValueError, match="POLYMARKET_FUNDER"):
        cli._validate_real_credentials(cfg)


# --- main() early-exit codes ------------------------------------

def test_main_returns_2_on_config_error() -> None:
    rc = cli.main(env={})  # missing required fields
    assert rc == 2


def test_main_returns_4_on_real_venue_with_placeholder_credentials(tmp_path: Path) -> None:
    # _full_env uses an all-zeros private_key — a placeholder.
    # Real-venue path runs _validate_real_credentials, which raises
    # ValueError, which main() maps to exit code 4.
    rc = cli.main(env=_full_env(tmp_path, POLYMARKET_VENUE="real"))
    assert rc == 4


def test_main_returns_4_on_unknown_venue(tmp_path: Path) -> None:
    rc = cli.main(env=_full_env(tmp_path, POLYMARKET_VENUE="garbage"))
    assert rc == 4
