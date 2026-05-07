"""
executor/risk.py

Minimal risk manager for the Tail Harvester strategy.

Provides three controls:
  1. Manual kill switch  — operator halts all trading immediately
  2. Auto kill switch    — triggered automatically when a threshold is breached
  3. Daily loss cap      — halts trading if realised losses exceed a configured limit

This is intentionally simpler than the full multi-leg risk engine from the
structural arb executor. The harvester has one order per event at a time,
so multi-leg package tracking, exposure reservations, and leg-timeout logic
are not needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .venue import MetricsSink, NullLogger, NullMetrics, StructuredLogger


class RiskReasonCode(str, Enum):
    MANUAL_KILL_SWITCH     = "MANUAL_KILL_SWITCH"
    AUTO_KILL_SWITCH       = "AUTO_KILL_SWITCH"
    DAILY_LOSS_CAP         = "DAILY_LOSS_CAP"
    MAX_NOTIONAL_PER_EVENT = "MAX_NOTIONAL_PER_EVENT"


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    code: RiskReasonCode | None
    reason: str
    ts_ns: int
    should_halt_trading: bool = False

    @staticmethod
    def allow(ts_ns: int, reason: str = "allowed") -> RiskDecision:
        return RiskDecision(allowed=True, code=None, reason=reason, ts_ns=ts_ns)

    @staticmethod
    def deny(
        *,
        code: RiskReasonCode,
        reason: str,
        ts_ns: int,
        should_halt_trading: bool = False,
    ) -> RiskDecision:
        return RiskDecision(
            allowed=False,
            code=code,
            reason=reason,
            ts_ns=ts_ns,
            should_halt_trading=should_halt_trading,
        )


@dataclass(frozen=True, slots=True)
class RiskManagerConfig:
    """
    Configuration for all risk controls.

    daily_loss_cap_usdc:
        Maximum realised loss (in USDC) before trading halts for the session.
        Default $200. Set to 0 to disable.

    max_notional_per_event_usdc:
        Maximum total USDC that can be filled on a single event.
        This is the same as HARVESTER_MAX_NOTIONAL_PER_EVENT in .env.
        Stored here so the risk manager can enforce it independently.

    enable_auto_kill_switch:
        If False, automatic halts from loss cap are suppressed.
        Manual kill switch always works regardless.
    """
    daily_loss_cap_usdc: float = 200.0
    max_notional_per_event_usdc: float = 20.0
    enable_auto_kill_switch: bool = True


class ExecutionRiskManager:
    """
    Single-threaded risk manager for the Tail Harvester.

    Called on the hot path — all methods are O(1) with no I/O.

    Usage pattern:
        # On startup
        risk = ExecutionRiskManager(RiskManagerConfig())

        # Before placing any order
        decision = risk.check_order_allowed(event_slug, fill_usdc_so_far, ts_ns)
        if not decision.allowed:
            # respect the decision

        # After a fill is confirmed
        risk.record_fill(usdc_amount, ts_ns)

        # Operator halt
        risk.activate_manual_kill_switch("reason", ts_ns)
    """

    __slots__ = (
        "_config",
        "_logger",
        "_metrics",
        "_manual_kill_active",
        "_manual_kill_reason",
        "_auto_kill_active",
        "_auto_kill_reason",
        "_daily_loss_usdc",
    )

    def __init__(
        self,
        config: RiskManagerConfig,
        *,
        logger: StructuredLogger | None = None,
        metrics: MetricsSink | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or NullLogger()
        self._metrics = metrics or NullMetrics()
        self._manual_kill_active = False
        self._manual_kill_reason: str | None = None
        self._auto_kill_active = False
        self._auto_kill_reason: str | None = None
        self._daily_loss_usdc: float = 0.0

    @property
    def is_trading_halted(self) -> bool:
        """True if either kill switch is active."""
        return self._manual_kill_active or self._auto_kill_active

    def check_order_allowed(
        self,
        *,
        event_slug: str,
        filled_usdc: float,
        ts_ns: int,
    ) -> RiskDecision:
        """
        Pre-order check. Call before placing or adjusting any resting bid.

        Checks in order:
          1. Manual kill switch
          2. Auto kill switch (daily loss cap)
          3. Per-event notional cap
        """
        if self._manual_kill_active:
            return RiskDecision.deny(
                code=RiskReasonCode.MANUAL_KILL_SWITCH,
                reason=self._manual_kill_reason or "manual kill switch active",
                ts_ns=ts_ns,
                should_halt_trading=True,
            )

        if self._auto_kill_active:
            return RiskDecision.deny(
                code=RiskReasonCode.AUTO_KILL_SWITCH,
                reason=self._auto_kill_reason or "auto kill switch active",
                ts_ns=ts_ns,
                should_halt_trading=True,
            )

        if filled_usdc >= self._config.max_notional_per_event_usdc:
            return RiskDecision.deny(
                code=RiskReasonCode.MAX_NOTIONAL_PER_EVENT,
                reason=(
                    f"event {event_slug} filled {filled_usdc:.2f} USDC "
                    f">= cap {self._config.max_notional_per_event_usdc:.2f}"
                ),
                ts_ns=ts_ns,
            )

        return RiskDecision.allow(ts_ns)

    def record_loss(self, loss_usdc: float, ts_ns: int) -> RiskDecision | None:
        """
        Record a realised loss (positive value = money lost).
        Returns a RiskDecision if the daily loss cap is now breached, else None.

        Call this when a market resolves against a position.
        """
        if loss_usdc <= 0:
            return None

        self._daily_loss_usdc += loss_usdc
        self._metrics.increment("risk.daily_loss_usdc")

        if (
            self._config.enable_auto_kill_switch
            and self._config.daily_loss_cap_usdc > 0
            and self._daily_loss_usdc >= self._config.daily_loss_cap_usdc
        ):
            decision = self._trigger_auto_kill(
                code=RiskReasonCode.DAILY_LOSS_CAP,
                reason=(
                    f"daily loss {self._daily_loss_usdc:.2f} USDC "
                    f">= cap {self._config.daily_loss_cap_usdc:.2f} USDC"
                ),
                ts_ns=ts_ns,
            )
            return decision

        return None

    def activate_manual_kill_switch(self, reason: str, ts_ns: int) -> RiskDecision:
        """Operator-triggered halt. Blocks all new orders immediately."""
        self._manual_kill_active = True
        self._manual_kill_reason = reason
        self._logger.warning("manual kill switch activated", reason=reason)
        self._metrics.increment("risk.kill_switch.manual")
        return RiskDecision.deny(
            code=RiskReasonCode.MANUAL_KILL_SWITCH,
            reason=reason,
            ts_ns=ts_ns,
            should_halt_trading=True,
        )

    def clear_manual_kill_switch(self, ts_ns: int) -> RiskDecision:
        """Clear the manual halt. Does not clear the auto kill switch."""
        self._manual_kill_active = False
        self._manual_kill_reason = None
        self._logger.info("manual kill switch cleared", ts_ns=ts_ns)
        return RiskDecision.allow(ts_ns, reason="manual kill switch cleared")

    def _trigger_auto_kill(
        self, *, code: RiskReasonCode, reason: str, ts_ns: int
    ) -> RiskDecision:
        self._auto_kill_active = True
        self._auto_kill_reason = reason
        self._logger.error("auto kill switch activated", code=code.value, reason=reason)
        self._metrics.increment("risk.kill_switch.auto")
        return RiskDecision.deny(
            code=code,
            reason=reason,
            ts_ns=ts_ns,
            should_halt_trading=True,
        )
