"""Shared real-venue order safety gate — the single source of truth for the
``MAX_REAL_ORDERS`` + ``REQUIRE_OPERATOR_CONFIRM`` harness.

Both the politics NegRisk maker loop (:class:`~polymarket.execution.maker.maker_engine.MakerEngine`)
and the MM-engine live bridge
(:class:`~polymarket.execution.maker.mm_engine_bridge.MMEngineBridge`) place real orders
through the same venue adapter, so they must gate real submits identically. This module
holds that gate once instead of copying it into each engine.

Semantics (unchanged from the original inline maker gate):

* The gate is a **no-op on fake venues** — it only fires when ``venue.is_real_venue()`` is
  truthy (duck-typed; absence treats the venue as fake). Dry-run / smoke runs therefore
  never touch it.
* ``max_real_orders`` bounds the number of real submit *attempts* (a rejection still
  consumes the budget). ``0`` is valid and means observe-only (real venue can be wired for
  reads, but no submit is ever permitted). On the limit it writes a ``RiskHalt`` with reason
  ``"max_real_orders"`` and returns that reason so the caller can skip/halt as it sees fit.
* ``require_operator_confirm`` blocks on stdin per order; declining writes a ``RiskHalt``
  reason ``"operator_aborted"`` and returns it, but does NOT itself halt — the operator can
  decline individual orders without killing the loop.

The gate journals only the ``RiskHalt``; the caller journals whatever skip/telemetry event
fits its own vocabulary (``MakerQuoteSkipped`` for the maker, etc.). ``label`` lets each
caller name itself in the ``RiskHalt`` detail without changing the machine-readable reason.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import JsonlWriter, RiskHalt


def is_real_venue(venue: Any) -> bool:
    """True iff the venue self-identifies as real (duck-typed; missing ⇒ fake)."""
    fn = getattr(venue, "is_real_venue", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 — a broken probe must not be treated as real
        return False


@dataclass
class RealOrderGate:
    """Per-process real-order safety gate: ``max_real_orders`` + ``operator_confirm``.

    Construct one per engine/session and call :meth:`check` immediately before every real
    submit. The attempt counter is per-instance (per-process), matching the original maker
    semantics; a restart resets it.
    """

    config: ExecutionConfig
    journal: JsonlWriter
    label: str = "maker"
    # injectable so tests can drive the operator prompt without real stdin
    prompt_fn: Callable[[str], bool] | None = None
    _real_attempts: int = field(default=0, init=False)

    @property
    def real_attempts(self) -> int:
        return self._real_attempts

    def check(
        self,
        *,
        venue: Any,
        condition_id: str,
        asset_id: str,
        side: str,
        size: float,
        price: float,
    ) -> tuple[str, str] | None:
        """Return ``(reason, detail)`` to block the submit, or ``None`` to allow it.

        A ``None`` return has consumed one unit of the real-order budget (the counter is
        incremented) — mirroring the original maker gate, where reaching the venue counts as
        an attempt.
        """
        if not is_real_venue(venue):
            return None
        if self._real_attempts >= self.config.max_real_orders:
            detail = (
                f"{self.label} reached limit of "
                f"{self.config.max_real_orders} real submits"
            )
            self.journal.write(RiskHalt(
                ts_utc=datetime.now(timezone.utc),
                reason="max_real_orders",
                detail=detail,
            ))
            return "max_real_orders", detail
        if self.config.require_operator_confirm:
            if not self._confirm(
                condition_id=condition_id,
                asset_id=asset_id,
                side=side,
                size=size,
                price=price,
            ):
                detail = f"operator declined {self.label} order via stdin"
                self.journal.write(RiskHalt(
                    ts_utc=datetime.now(timezone.utc),
                    reason="operator_aborted",
                    detail=detail,
                ))
                return "operator_aborted", detail
        self._real_attempts += 1
        return None

    def _confirm(
        self,
        *,
        condition_id: str,
        asset_id: str,
        side: str,
        size: float,
        price: float,
    ) -> bool:
        prompt = (
            f"\n[operator confirm] {self.label} order:\n"
            f"  condition: {condition_id}\n"
            f"  asset    : {asset_id}\n"
            f"  side     : {side}\n"
            f"  size     : {size}\n"
            f"  price    : ${price:.4f}\n"
            f"Type 'yes' to proceed: "
        )
        if self.prompt_fn is not None:
            return bool(self.prompt_fn(prompt))
        print(prompt, end="", flush=True)
        try:
            return sys.stdin.readline().strip().lower() == "yes"
        except (EOFError, KeyboardInterrupt):
            return False
