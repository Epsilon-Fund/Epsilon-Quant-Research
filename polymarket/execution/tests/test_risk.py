"""Tests for risk/ — every breaker fires when it should, no false positives."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.risk import (
    VETO_REASONS,
    CandidateOrder,
    RiskState,
    Veto,
    check_daily_loss,
    check_deployed_cap,
    check_kill_switch,
    check_market_cap,
    check_max_open_positions,
    check_price_deviation,
    check_size_cap,
    run_all_checks,
)

_LEADER = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"
_FUNDER = "0x" + "ab" * 20

_BASE_CONFIG_FIELDS: dict[str, Any] = {
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
    "killswitch_path": Path("/tmp/polymarket_killswitch"),
    "journal_dir": Path("./journal_logs"),
    "log_level": "INFO",
    "max_real_orders": 5,
    "require_operator_confirm": False,
}

_BASE_STATE_FIELDS: dict[str, Any] = {
    "current_market_price": 0.50,
    "deployed_usd": 0.0,
    "deployed_in_market_usd": 0.0,
    "open_positions_count": 0,
    "realised_pnl_today_usd": 0.0,
    "killswitch_present": False,
}

_BASE_ORDER_FIELDS: dict[str, Any] = {
    "client_order_id": "coid-1",
    "condition_id": "0xcond",
    "asset_id": "42",
    "side": "BUY",
    "size_usd": 10.0,
    "leader_fill_price": 0.50,
}


def _config(**overrides: Any) -> ExecutionConfig:
    return ExecutionConfig(**{**_BASE_CONFIG_FIELDS, **overrides})


def _state(**overrides: Any) -> RiskState:
    return RiskState(**{**_BASE_STATE_FIELDS, **overrides})


def _order(**overrides: Any) -> CandidateOrder:
    return CandidateOrder(**{**_BASE_ORDER_FIELDS, **overrides})


# --- size_cap ---------------------------------------------------------

def test_size_cap_passes_under() -> None:
    assert check_size_cap(_config(), _state(), _order(size_usd=20.0)) is None


def test_size_cap_vetoes_over() -> None:
    veto = check_size_cap(_config(), _state(), _order(size_usd=20.01))
    assert veto is not None
    assert veto.reason == "size_cap"


# --- market_cap -------------------------------------------------------

def test_market_cap_passes_when_sum_under() -> None:
    state = _state(deployed_in_market_usd=30.0)
    assert check_market_cap(_config(), state, _order(size_usd=20.0)) is None


def test_market_cap_vetoes_when_sum_over() -> None:
    state = _state(deployed_in_market_usd=40.0)
    veto = check_market_cap(_config(), state, _order(size_usd=10.01))
    assert veto is not None
    assert veto.reason == "market_cap"


# --- deployed_cap -----------------------------------------------------

def test_deployed_cap_passes_when_sum_under() -> None:
    state = _state(deployed_usd=80.0)
    assert check_deployed_cap(_config(), state, _order(size_usd=20.0)) is None


def test_deployed_cap_vetoes_when_sum_over() -> None:
    state = _state(deployed_usd=90.0)
    veto = check_deployed_cap(_config(), state, _order(size_usd=10.01))
    assert veto is not None
    assert veto.reason == "deployed_cap"


# --- price_deviation --------------------------------------------------

def test_price_deviation_passes_within_tolerance() -> None:
    # 0.505 vs 0.500 = 1% diff, tolerance is 2%.
    state = _state(current_market_price=0.505)
    assert check_price_deviation(_config(), state, _order(leader_fill_price=0.50)) is None


def test_price_deviation_vetoes_outside_tolerance() -> None:
    # 0.520 vs 0.500 = 4% diff, tolerance is 2%.
    state = _state(current_market_price=0.520)
    veto = check_price_deviation(_config(), state, _order(leader_fill_price=0.50))
    assert veto is not None
    assert veto.reason == "price_deviation"


def test_price_deviation_zero_leader_price_raises() -> None:
    with pytest.raises(ValueError, match="leader_fill_price"):
        check_price_deviation(_config(), _state(), _order(leader_fill_price=0.0))


def test_price_deviation_negative_leader_price_raises() -> None:
    with pytest.raises(ValueError, match="leader_fill_price"):
        check_price_deviation(_config(), _state(), _order(leader_fill_price=-0.1))


# --- daily_loss -------------------------------------------------------

def test_daily_loss_passes_above_threshold() -> None:
    state = _state(realised_pnl_today_usd=-199.0)
    assert check_daily_loss(_config(), state, _order()) is None


def test_daily_loss_vetoes_at_threshold() -> None:
    state = _state(realised_pnl_today_usd=-200.0)
    veto = check_daily_loss(_config(), state, _order())
    assert veto is not None
    assert veto.reason == "daily_loss"


def test_daily_loss_vetoes_below_threshold() -> None:
    state = _state(realised_pnl_today_usd=-250.0)
    veto = check_daily_loss(_config(), state, _order())
    assert veto is not None
    assert veto.reason == "daily_loss"


# --- max_open_positions ----------------------------------------------

def test_max_open_positions_passes_under_cap() -> None:
    state = _state(open_positions_count=2)
    assert check_max_open_positions(_config(), state, _order()) is None


def test_max_open_positions_vetoes_at_cap() -> None:
    state = _state(open_positions_count=3)
    veto = check_max_open_positions(_config(), state, _order())
    assert veto is not None
    assert veto.reason == "max_open_positions"


def test_max_open_positions_vetoes_over_cap() -> None:
    state = _state(open_positions_count=4)
    veto = check_max_open_positions(_config(), state, _order())
    assert veto is not None
    assert veto.reason == "max_open_positions"


# --- kill_switch ------------------------------------------------------

def test_kill_switch_passes_when_absent() -> None:
    assert check_kill_switch(_config(), _state(killswitch_present=False), _order()) is None


def test_kill_switch_vetoes_when_present() -> None:
    veto = check_kill_switch(_config(), _state(killswitch_present=True), _order())
    assert veto is not None
    assert veto.reason == "kill_switch"
    assert "/tmp/polymarket_killswitch" in veto.detail


# --- run_all_checks ---------------------------------------------------

def test_run_all_checks_returns_none_when_all_pass() -> None:
    assert run_all_checks(_config(), _state(), _order()) is None


def test_run_all_checks_returns_first_veto_in_order() -> None:
    # size_cap (3rd) and deployed_cap (5th) would both veto.
    # Expect size_cap because it runs earlier.
    state = _state(deployed_usd=200.0)  # already over max_capital_usd of 100
    order = _order(size_usd=50.0)  # also over per_trade_cap_usd of 20
    veto = run_all_checks(_config(), state, order)
    assert veto is not None
    assert veto.reason == "size_cap"


def test_run_all_checks_kill_switch_fails_fast() -> None:
    # Even if everything else would also veto, kill_switch is first.
    state = _state(killswitch_present=True, deployed_usd=999.0,
                   open_positions_count=99)
    veto = run_all_checks(_config(), state, _order(size_usd=999.0))
    assert veto is not None
    assert veto.reason == "kill_switch"


# --- Veto validation --------------------------------------------------

def test_veto_invalid_reason_raises() -> None:
    with pytest.raises(ValueError, match="VETO_REASONS"):
        Veto(reason="bogus", detail="should fail")


def test_veto_reasons_constant() -> None:
    assert VETO_REASONS == frozenset({
        "size_cap", "market_cap", "deployed_cap", "price_deviation",
        "daily_loss", "kill_switch", "max_open_positions",
        "max_real_orders", "operator_aborted",
    })


def test_veto_real_venue_harness_reasons_accepted() -> None:
    # Mirror_engine writes Veto-shaped events via RiskHalt; the reasons
    # must round-trip through Veto's __post_init__ validation.
    assert Veto(reason="max_real_orders", detail="x").reason == "max_real_orders"
    assert Veto(reason="operator_aborted", detail="x").reason == "operator_aborted"
