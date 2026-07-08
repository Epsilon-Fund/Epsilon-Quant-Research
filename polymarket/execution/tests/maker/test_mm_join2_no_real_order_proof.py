"""The Join-2 no-real-order PROOF (v1 success criterion, PRD § SUCCESS).

Demonstrates, for BOTH order-originating paths of the Join-2 machinery — the 2c bridge
quote path and the 2b latency-probe path — that a submit reaching the venue requires the
CONJUNCTION of all three conditions:

    (real venue)  ∧  (MAX_REAL_ORDERS budget remaining)  ∧  (operator confirms the order)

Each test knocks out exactly one conjunct and asserts the venue never sees a submit; the
final test enables all three and asserts the submit flows (bounded by the budget) — proving
the earlier blocks came from the gate, not from an accident of wiring. On a fake venue the
gate is a no-op by design, and no real order is possible because nothing in the process
holds real credentials or a network client (the fake adapter prints/records only).
"""
from __future__ import annotations

from polymarket.execution.maker.mm_engine_bridge import VenueOrderRouter
from polymarket.execution.maker.mm_latency_harness import (
    LatencyProbeConfig,
    LatencyProbeHarness,
    TimingVenue,
)
from polymarket.execution.maker.order_safety import RealOrderGate
from polymarket.execution.risk import RiskState

from .test_mm_engine_bridge import _book_event, _bridge, _CapturingJournal, _config, _MockVenue


def _probe_harness(venue: _MockVenue, journal: _CapturingJournal, config, prompt_fn=None,
                   samples_target: int = 4) -> LatencyProbeHarness:
    timing = TimingVenue(venue)
    gate = RealOrderGate(config=config, journal=journal, label="mm_latency", prompt_fn=prompt_fn)
    router = VenueOrderRouter(venue=timing, config=config, journal=journal, gate=gate,
                              order_type="GTC", coid_prefix="lat-")
    state = RiskState(current_market_price=0.5, deployed_usd=0.0, deployed_in_market_usd=0.0,
                      open_positions_count=0, realised_pnl_today_usd=0.0, killswitch_present=False)
    return LatencyProbeHarness(
        router=router, timing=timing,
        cfg=LatencyProbeConfig(condition_id="0xcond", asset_id="t1",
                               samples_target=samples_target, cadence_s=0.0),
        journal=journal, book_fn=lambda: (0.47, 0.49), risk_state_fn=lambda _a, _p: state,
    )


# --------------------------------------------------------------------------------------
# conjunct 1 knocked out: NOT a real venue → the gate never even arms
# --------------------------------------------------------------------------------------

def test_fake_venue_never_consumes_real_order_budget_bridge_and_probe() -> None:
    venue, journal = _MockVenue(real=False), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal,
                     config=_config(max_real_orders=0, require_operator_confirm=True))
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    # DRY-RUN placements flow to the FAKE venue, but the real-order budget is untouched
    # and the operator was never prompted — the gate is a no-op off real venues.
    assert len(venue.submit_calls) == 2
    assert bridge.router.gate.real_attempts == 0
    assert journal.of_type("RISK_HALT") == []

    venue2, journal2 = _MockVenue(real=False), _CapturingJournal()
    harness = _probe_harness(venue2, journal2,
                             _config(max_real_orders=0, require_operator_confirm=True))
    harness.probe_once()
    assert len(venue2.submit_calls) == 1
    assert harness.router.gate.real_attempts == 0


# --------------------------------------------------------------------------------------
# conjunct 2 knocked out: real venue, ZERO budget → nothing reaches the venue
# --------------------------------------------------------------------------------------

def test_real_venue_zero_budget_blocks_bridge_and_probe() -> None:
    venue, journal = _MockVenue(real=True), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal, config=_config(max_real_orders=0))
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    assert venue.submit_calls == []

    venue2, journal2 = _MockVenue(real=True), _CapturingJournal()
    harness = _probe_harness(venue2, journal2, _config(max_real_orders=0))
    report = harness.run()
    assert venue2.submit_calls == []
    assert report["samples"] == 0


# --------------------------------------------------------------------------------------
# conjunct 3 knocked out: real venue + budget, operator DECLINES → nothing reaches the venue
# --------------------------------------------------------------------------------------

def test_real_venue_operator_decline_blocks_bridge_and_probe() -> None:
    cfg = _config(max_real_orders=5, require_operator_confirm=True)

    venue, journal = _MockVenue(real=True), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal, config=cfg, prompt_fn=lambda _p: False)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    assert venue.submit_calls == []

    venue2, journal2 = _MockVenue(real=True), _CapturingJournal()
    harness = _probe_harness(venue2, journal2, cfg, prompt_fn=lambda _p: False)
    report = harness.run()
    assert venue2.submit_calls == []
    assert report["samples"] == 0


# --------------------------------------------------------------------------------------
# all three conjuncts TRUE → the submit flows, and ONLY within the budget
# --------------------------------------------------------------------------------------

def test_all_three_conditions_together_allow_bounded_submits() -> None:
    cfg = _config(max_real_orders=2, require_operator_confirm=True)

    venue, journal = _MockVenue(real=True), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal, config=cfg, prompt_fn=lambda _p: True)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    # two-sided quote = 2 submits = the entire budget; nothing further can flow
    assert len(venue.submit_calls) == 2
    bridge.on_market_event(_book_event(1100, 0.40, 0.44))     # mid moves → replace → blocked
    assert len(venue.submit_calls) == 2
    assert any(e.reason == "max_real_orders" for e in journal.of_type("RISK_HALT"))

    venue2, journal2 = _MockVenue(real=True), _CapturingJournal()
    harness = _probe_harness(venue2, journal2, cfg, prompt_fn=lambda _p: True,
                             samples_target=10)
    report = harness.run()
    assert len(venue2.submit_calls) == 2      # budget bounds the probes too
    assert report["samples"] == 2
