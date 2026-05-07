from __future__ import annotations

import pytest

from harvester.config import HarvesterConfig

pytestmark = [pytest.mark.unit, pytest.mark.harvester]

_REQUIRED = {
    "PM_API_KEY": "key-test",
    "PM_API_SECRET": "secret-test",
    "PM_PASSPHRASE": "pass-test",
    "PM_SLUGS": "london-temp-april-30",
}


def _set_required(monkeypatch) -> None:
    for k, v in _REQUIRED.items():
        monkeypatch.setenv(k, v)


def _load(monkeypatch) -> HarvesterConfig:
    """Load config without reading any .env file."""
    return HarvesterConfig.from_env(dotenv_path=None)


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


def test_loads_with_all_required_fields(monkeypatch) -> None:
    _set_required(monkeypatch)
    config = _load(monkeypatch)
    assert config.api_key == "key-test"
    assert config.api_secret == "secret-test"
    assert config.passphrase == "pass-test"
    assert config.slugs == ("london-temp-april-30",)


def test_missing_api_key_raises(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.delenv("PM_API_KEY")
    with pytest.raises(ValueError, match="PM_API_KEY"):
        _load(monkeypatch)


def test_missing_api_secret_raises(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.delenv("PM_API_SECRET")
    with pytest.raises(ValueError, match="PM_API_SECRET"):
        _load(monkeypatch)


def test_missing_passphrase_raises(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.delenv("PM_PASSPHRASE")
    with pytest.raises(ValueError, match="PM_PASSPHRASE"):
        _load(monkeypatch)


def test_missing_slugs_auto_generates_weather_cities(monkeypatch) -> None:
    # PM_SLUGS absent → auto-generate all 59 weather city slugs for today
    _set_required(monkeypatch)
    monkeypatch.delenv("PM_SLUGS")
    config = _load(monkeypatch)
    assert len(config.slugs) == 59
    assert all("-temperature-in-" in s for s in config.slugs)


def test_empty_slugs_raises(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("PM_SLUGS", "  ,  ")
    with pytest.raises(ValueError, match="PM_SLUGS"):
        _load(monkeypatch)


def test_multiple_slugs_parsed(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("PM_SLUGS", "slug-a, slug-b , slug-c")
    config = _load(monkeypatch)
    assert config.slugs == ("slug-a", "slug-b", "slug-c")


# ---------------------------------------------------------------------------
# Optional private key
# ---------------------------------------------------------------------------


def test_private_key_absent_is_none(monkeypatch) -> None:
    _set_required(monkeypatch)
    config = _load(monkeypatch)
    assert config.private_key is None


def test_private_key_set(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("PM_PRIVATE_KEY", "0xdeadbeef")
    config = _load(monkeypatch)
    assert config.private_key == "0xdeadbeef"


# ---------------------------------------------------------------------------
# Credential masking in repr
# ---------------------------------------------------------------------------


def test_credentials_not_in_repr(monkeypatch) -> None:
    _set_required(monkeypatch)
    config = _load(monkeypatch)
    r = repr(config)
    assert "key-test" not in r
    assert "secret-test" not in r
    assert "pass-test" not in r


# ---------------------------------------------------------------------------
# Strategy tuning defaults
# ---------------------------------------------------------------------------


def test_default_bid_threshold(monkeypatch) -> None:
    _set_required(monkeypatch)
    config = _load(monkeypatch)
    assert config.bid_threshold == 0.90


def test_custom_bid_threshold(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("STRATEGY_BID_THRESHOLD", "0.95")
    config = _load(monkeypatch)
    assert config.bid_threshold == 0.95


def test_bid_threshold_out_of_range_raises(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("STRATEGY_BID_THRESHOLD", "1.5")
    with pytest.raises(ValueError, match="STRATEGY_BID_THRESHOLD"):
        _load(monkeypatch)


def test_bid_threshold_zero_raises(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("STRATEGY_BID_THRESHOLD", "0.0")
    with pytest.raises(ValueError, match="STRATEGY_BID_THRESHOLD"):
        _load(monkeypatch)


def test_invalid_float_raises(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("STRATEGY_BID_THRESHOLD", "not-a-number")
    with pytest.raises(ValueError, match="STRATEGY_BID_THRESHOLD"):
        _load(monkeypatch)


# ---------------------------------------------------------------------------
# OMS defaults
# ---------------------------------------------------------------------------


def test_default_oms_order_qty(monkeypatch) -> None:
    _set_required(monkeypatch)
    config = _load(monkeypatch)
    assert config.oms_order_qty == 10


def test_oms_order_qty_zero_raises(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("OMS_ORDER_QTY", "0")
    with pytest.raises(ValueError, match="OMS_ORDER_QTY"):
        _load(monkeypatch)


def test_invalid_int_raises(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("OMS_ORDER_QTY", "ten")
    with pytest.raises(ValueError, match="OMS_ORDER_QTY"):
        _load(monkeypatch)


# ---------------------------------------------------------------------------
# Risk sub-config
# ---------------------------------------------------------------------------


def test_default_risk_config(monkeypatch) -> None:
    _set_required(monkeypatch)
    config = _load(monkeypatch)
    assert config.risk.daily_loss_cap_usdc == 200.0
    assert config.risk.max_notional_per_event_usdc == 20.0
    assert config.risk.enable_auto_kill_switch is True


def test_custom_risk_config(monkeypatch) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("RISK_DAILY_LOSS_CAP_USDC", "500.0")
    monkeypatch.setenv("RISK_MAX_NOTIONAL_PER_EVENT_USDC", "50.0")
    monkeypatch.setenv("RISK_ENABLE_AUTO_KILL", "false")
    config = _load(monkeypatch)
    assert config.risk.daily_loss_cap_usdc == 500.0
    assert config.risk.max_notional_per_event_usdc == 50.0
    assert config.risk.enable_auto_kill_switch is False
