"""
tests/helpers/factories.py

Test helpers for the executor infrastructure tests.
Only contains helpers for journal, venue adapter, and risk manager tests.
No planner or state machine helpers — those modules have been removed.
"""
from __future__ import annotations

from executor.risk import RiskManagerConfig


def make_default_risk_config() -> RiskManagerConfig:
    """Standard test risk config with conservative limits."""
    return RiskManagerConfig(
        daily_loss_cap_usdc=200.0,
        max_notional_per_event_usdc=20.0,
        enable_auto_kill_switch=True,
    )
