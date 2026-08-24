"""Latency models — round-trip (submit→live) latency used to gate fills.

The engine's fill simulator (``fills.py``) only lets a resting order fill a trade once the
order could have *landed*: ``placement_ts + round_trip_ms <= trade.ts_exchange``. A
mis-set round-trip fabricates (or denies) fills — hftbacktest reports the *same* strategy
swinging Sharpe −0.20 → +1.54 → −0.38 purely by changing the latency model. See
[[mm_backtesting_methodology_explainer]] §2.

Two implementations behind the frozen :class:`~mm_engine.interfaces.LatencyModel` protocol
(``round_trip_ms(ts_exchange) -> float``):

* :class:`ConstantLatency` — one fixed round-trip. The default is a **realistic ~200ms**
  retail/cloud round-trip (not 0ms — instant fills are the over-optimism the explainer
  warns about). Override per market once measured.
* :class:`SampledLatency` — a **Phase-2 placeholder** that draws each round-trip from a
  fitted ``(mean, std)``, so a backtest can reflect latency *dispersion*, not just its
  mean. Seeded, so replay stays deterministic. The ``(mean, std)`` come from the live
  measurement loop specced in [[mm_latency_measurement_spec]] (unexecutable 0.001/0.999
  orders, timed submit→ack, fit, market-stratified). The real fitted model (hftbacktest's
  ``IntpOrderLatency`` from interpolated measured round-trips) is the eventual successor.

Latency is market-stratified: a minor refinement for slow politics, **essential** for
crypto-4h / in-play sports where a stale quote is sniped (the K2-family adverse selection).
"""
from __future__ import annotations

import random
from dataclasses import dataclass


# Realistic default round-trip (ms). A retail/cloud submit→live round-trip to the Polymarket
# CLOB is on the order of ~200ms; 0ms (instant fills) flatters every backtest. Measure and
# override per market via the Phase-2 loop ([[mm_latency_measurement_spec]]).
DEFAULT_ROUND_TRIP_MS = 200.0


@dataclass
class ConstantLatency:
    """Constant round-trip latency in milliseconds, independent of time.

    Defaults to a realistic :data:`DEFAULT_ROUND_TRIP_MS`. Pass ``round_trip=0.0`` for a
    zero-latency idealization (e.g. isolating queue/fill logic from the latency gate).
    """

    round_trip: float = DEFAULT_ROUND_TRIP_MS

    def round_trip_ms(self, ts_exchange: int) -> float:
        return float(self.round_trip)


@dataclass
class SampledLatency:
    """Phase-2 placeholder: draw each round-trip from a fitted ``(mean, std)`` (ms).

    A constant round-trip understates a real engine's *variance* — some quotes land late
    and miss fills they'd otherwise get. This stub samples ``Normal(mean, std)`` (clamped to
    ``>= floor_ms``, since latency is non-negative) so the fill gate sees that dispersion.

    :meth:`round_trip_ms` is a **pure function of** ``ts_exchange``: the draw is keyed on
    ``(seed, int(ts_exchange))``, so the same submit time always returns the same round-trip.
    This matters because the fill simulator (``fills.py``) calls ``round_trip_ms(placement_ts)``
    **once per (order, trade-check)** — a single resting order is probed against every trade
    that might hit it — and an order's submit→live latency must be **fixed once at placement**,
    not re-rolled on each probe. Keying on the timestamp (rather than drawing from a sequential
    RNG) also makes the result independent of call count and order, so replay stays
    deterministic across processes (repo invariant). Different submit times still disperse.

    The ``(mean, std)`` are NOT guessed here — they are the output of the live measurement
    loop in [[mm_latency_measurement_spec]] (timed submit→ack on unexecutable orders, fit,
    **market-stratified**: a slow-politics fit and a fast-crypto/sports fit differ). Use
    :meth:`from_samples` / :meth:`calibrate` to load measured round-trips. The eventual
    successor is hftbacktest's ``IntpOrderLatency`` (interpolated measured round-trips),
    which this placeholder stands in for until that data exists.
    """

    mean: float = DEFAULT_ROUND_TRIP_MS
    std: float = 0.0
    floor_ms: float = 0.0
    seed: int = 0

    def round_trip_ms(self, ts_exchange: int) -> float:
        if self.std <= 0.0:
            return max(self.floor_ms, float(self.mean))     # degenerate -> the mean (clamped)
        # Key the draw on (seed, ts_exchange) so repeated probes of the same order (same
        # placement_ts) return an IDENTICAL round-trip, while different submit times disperse.
        # A string key is used because random.Random rejects tuple seeds; str/bytes seeds are
        # folded in via SHA-512 (independent of PYTHONHASHSEED), so the draw is stable across
        # processes and replay stays deterministic.
        rng = random.Random(f"{self.seed}:{int(ts_exchange)}")
        return max(self.floor_ms, float(rng.gauss(self.mean, self.std)))

    @classmethod
    def from_samples(
        cls,
        samples,
        *,
        floor_ms: float = 0.0,
        seed: int = 0,
    ) -> "SampledLatency":
        """Build from measured submit→ack round-trips (ms).

        Phase-2 fit stub: population mean/std of ``samples``. The real fit — outlier
        handling, market stratification (politics vs crypto/sports), and a richer
        interpolated model — lives in the measurement loop, not here.
        """
        vals = [float(s) for s in samples]
        n = len(vals)
        if n == 0:
            return cls(floor_ms=floor_ms, seed=seed)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n if n > 1 else 0.0
        return cls(mean=mean, std=var ** 0.5, floor_ms=floor_ms, seed=seed)

    def calibrate(self, samples) -> None:
        """Refit ``(mean, std)`` in place from measured round-trips.

        Mirrors ``QueueModel.calibrate`` ergonomics — a no-op-until-data Phase-2 hook. With
        no samples, leaves the current fit untouched. There is no RNG state to reset: each
        draw is keyed afresh on ``(seed, ts_exchange)`` (see :meth:`round_trip_ms`), so the
        refit takes effect immediately and deterministically.
        """
        vals = [float(s) for s in samples]
        if not vals:
            return None
        fit = SampledLatency.from_samples(vals, floor_ms=self.floor_ms, seed=self.seed)
        self.mean = fit.mean
        self.std = fit.std
        return None
