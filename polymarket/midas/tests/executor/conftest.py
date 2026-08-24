"""
tests/executor/conftest.py

Shared pytest fixtures for executor infrastructure tests.
"""
from __future__ import annotations

import pytest

from executor.risk import ExecutionRiskManager, RiskManagerConfig
from tests.helpers.factories import make_default_risk_config


@pytest.fixture
def now_ns() -> int:
    return 1_700_000_000_000_000_000


@pytest.fixture
def risk_config() -> RiskManagerConfig:
    return make_default_risk_config()


@pytest.fixture
def risk_manager(risk_config: RiskManagerConfig) -> ExecutionRiskManager:
    return ExecutionRiskManager(risk_config)
