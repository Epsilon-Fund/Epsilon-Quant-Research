"""Latency-model STUB. Alvaro owns the real model; this is the placeholder swap point.

Phase 0 ships only :class:`ConstantLatency` — a single fixed round-trip. The realism
ladder (feed-derived latency, ``IntpOrderLatency`` from measured round-trips collected by
submitting unexecutable orders) is Alvaro's to build. Latency is a minor refinement for
slow politics markets but essential for crypto/in-play sports — see
[[mm_backtesting_methodology_explainer]] §2.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConstantLatency:
    """Constant round-trip latency in milliseconds, independent of time."""

    round_trip: float = 0.0

    def round_trip_ms(self, ts_exchange: int) -> float:
        return float(self.round_trip)
