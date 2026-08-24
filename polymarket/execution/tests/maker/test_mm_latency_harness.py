"""Join-2b latency-harness tests — DRY-RUN/mock only, no network, no real orders.

Covers: the probe-side safety guard (never probe near 0/1), the timing wrapper (measures
the venue call only, delegates ``is_real_venue`` so wrapping can't fake a real venue), the
raw+trimmed fit, the probe→time→cancel cycle through the REUSED safety path (kill-switch +
caps + ``RealOrderGate``), and that a real-ish venue with an exhausted budget / declining
operator never sees a submit.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from polymarket.execution.maker.mm_engine_bridge import VenueOrderRouter
from polymarket.execution.maker.mm_latency_harness import (
    PROBE_BUY_PRICE,
    PROBE_SELL_PRICE,
    LatencyProbeConfig,
    LatencyProbeHarness,
    TimingVenue,
    choose_probe_side,
    fit_latency,
    render_fit,
)
from polymarket.execution.maker.order_safety import RealOrderGate
from polymarket.execution.risk import RiskState

from .test_mm_engine_bridge import _CapturingJournal, _config, _MockVenue


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _risk_state(killswitch: bool = False) -> RiskState:
    return RiskState(
        current_market_price=0.5,
        deployed_usd=0.0,
        deployed_in_market_usd=0.0,
        open_positions_count=0,
        realised_pnl_today_usd=0.0,
        killswitch_present=killswitch,
    )


def _harness(
    *,
    venue: _MockVenue,
    journal: _CapturingJournal,
    config=None,
    book=(0.47, 0.49),
    samples_target: int = 3,
    prompt_fn=None,
    killswitch: bool = False,
    out_dir: Path | None = None,
) -> tuple[LatencyProbeHarness, TimingVenue]:
    cfg = config or _config()
    timing = TimingVenue(venue)
    gate = RealOrderGate(config=cfg, journal=journal, label="mm_latency", prompt_fn=prompt_fn)
    router = VenueOrderRouter(
        venue=timing, config=cfg, journal=journal, gate=gate,
        order_type="GTC", coid_prefix="lat-",
    )
    probe_cfg = LatencyProbeConfig(
        condition_id="0xcond", asset_id="t1",
        samples_target=samples_target, cadence_s=0.0, out_dir=out_dir,
    )
    harness = LatencyProbeHarness(
        router=router, timing=timing, cfg=probe_cfg, journal=journal,
        book_fn=lambda: book,
        risk_state_fn=lambda _a, _p: _risk_state(killswitch),
    )
    return harness, timing


# --------------------------------------------------------------------------------------
# probe-side selection — the spec's safety guard
# --------------------------------------------------------------------------------------

def test_choose_side_prefers_side_further_from_touch() -> None:
    # book high in the range → the BUY probe (0.001) is much further from the touch
    assert choose_probe_side(0.90, 0.95) == "BUY"
    # book low in the range → the SELL probe (0.999) is further
    assert choose_probe_side(0.05, 0.10) == "SELL"


def test_choose_side_refuses_near_resolved_books() -> None:
    # bid within a tick of 0 → BUY probe unsafe; ask within a tick of 1 → SELL unsafe
    assert choose_probe_side(0.001, 0.999) is None
    assert choose_probe_side(0.002, 0.999) is None      # bid not strictly above 0.001+tick
    assert choose_probe_side(0.001, 0.05) == "SELL"     # only SELL is safe
    assert choose_probe_side(0.95, 0.999) == "BUY"      # only BUY is safe
    assert choose_probe_side(None, 0.5) is None         # one-sided book → no probe


def test_choose_side_respects_explicit_preference() -> None:
    assert choose_probe_side(0.47, 0.49, prefer="buy") == "BUY"
    assert choose_probe_side(0.47, 0.49, prefer="sell") == "SELL"
    assert choose_probe_side(0.002, 0.49, prefer="buy") is None   # preferred side unsafe → None


# --------------------------------------------------------------------------------------
# timing wrapper
# --------------------------------------------------------------------------------------

def test_timing_venue_measures_submit_and_cancel_round_trip() -> None:
    venue = _MockVenue()
    ticks = iter([0, 5_000_000, 10_000_000, 12_000_000])  # ns: submit 5ms, cancel 2ms
    timing = TimingVenue(venue, clock_ns=lambda: next(ticks))
    timing.submit_order(client_order_id="c1", condition_id="0xcond", asset_id="t1",
                        side="BUY", size_shares=1.0, price=0.001, order_type="GTC")
    assert timing.last_submit_ms == pytest.approx(5.0)
    timing.cancel_order(client_order_id="c1")
    assert timing.last_cancel_ms == pytest.approx(2.0)


def test_timing_venue_delegates_is_real_venue_and_attrs() -> None:
    fake, real = _MockVenue(real=False), _MockVenue(real=True)
    assert TimingVenue(fake).is_real_venue() is False
    assert TimingVenue(real).is_real_venue() is True        # wrapping can't fake realness
    # arbitrary attr passthrough (e.g. reconcile_open_orders for the venue reconcile)
    assert TimingVenue(real).reconcile_open_orders(set()) is not None


# --------------------------------------------------------------------------------------
# the fit — raw AND trimmed (spec §3)
# --------------------------------------------------------------------------------------

def test_fit_latency_reports_raw_and_trimmed_with_outlier() -> None:
    samples = [200.0] * 59 + [5000.0]     # one multi-second stall in 60 samples
    fit = fit_latency(samples, trim_frac=0.02)
    assert fit["raw"]["n"] == 60
    assert fit["raw"]["mean"] > fit["trimmed"]["mean"]      # stall inflates raw only
    assert fit["trimmed"]["mean"] == pytest.approx(200.0)   # winsorized to the bulk
    assert fit["raw"]["p99"] > 200.0                        # the honest tail stays visible
    assert fit["constant_ms"] == fit["trimmed"]["mean"]
    assert fit["sampled"]["std"] == fit["trimmed"]["std"]


def test_fit_latency_empty_is_nan_not_crash() -> None:
    fit = fit_latency([])
    assert fit["raw"]["n"] == 0


# --------------------------------------------------------------------------------------
# probe cycle — place through the reused safety path, time, cancel immediately
# --------------------------------------------------------------------------------------

def test_probe_places_unexecutable_order_and_cancels_immediately() -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    harness, _ = _harness(venue=venue, journal=journal, book=(0.47, 0.49))
    rec = harness.probe_once()

    assert rec.accepted is True
    assert rec.submit_ms is not None and rec.cancel_ms is not None
    assert len(venue.submit_calls) == 1
    call = venue.submit_calls[0]
    # unexecutable by construction: the extreme tick, 1 contract
    assert call["price"] in (PROBE_BUY_PRICE, PROBE_SELL_PRICE)
    assert call["size_shares"] == 1.0
    # cancelled immediately after the ack
    assert len(venue.cancel_calls) == 1
    assert venue._open == {}    # nothing left resting at the venue
    canceled = journal.of_type("MAKER_QUOTE_CANCELED")
    assert canceled and canceled[0].reason == "latency_probe"


def test_probe_skips_unsafe_book_and_run_stops_after_persistent_skips() -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    harness, _ = _harness(venue=venue, journal=journal, book=(0.001, 0.999), samples_target=5)
    report = harness.run()

    assert venue.submit_calls == []                       # nothing ever submitted
    assert report["samples"] == 0
    assert report["probes_attempted"] == 3                # 3 consecutive skips → stop, no spin
    skips = [e for e in journal.of_type("MAKER_QUOTE_SKIPPED")
             if e.reason == "latency_probe_unsafe_book"]
    assert len(skips) == 3


def test_run_collects_target_samples_and_fits(tmp_path: Path) -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    harness, _ = _harness(venue=venue, journal=journal, samples_target=5, out_dir=tmp_path)
    report = harness.run()

    assert report["samples"] == 5
    assert len(venue.submit_calls) == 5 and len(venue.cancel_calls) == 5
    assert report["submit_ack_fit"]["raw"]["n"] == 5
    # artifacts are written (append-only samples JSONL + fit JSON)
    assert Path(report["samples_path"]).exists()
    assert Path(report["fit_path"]).exists()
    # render never crashes and carries the model hand-off line
    assert "POLYMARKET_MM_BRIDGE_LATENCY_MS" in render_fit(report)


# --------------------------------------------------------------------------------------
# safety: the probe path is gated EXACTLY like a quote
# --------------------------------------------------------------------------------------

def test_real_venue_with_zero_budget_never_submits_a_probe() -> None:
    venue, journal = _MockVenue(real=True), _CapturingJournal()
    harness, _ = _harness(venue=venue, journal=journal, config=_config(max_real_orders=0))
    report = harness.run()

    assert venue.submit_calls == []                       # the venue never saw a request
    assert report["samples"] == 0
    assert any(e.reason == "max_real_orders" for e in journal.of_type("RISK_HALT"))


def test_real_venue_operator_decline_blocks_every_probe() -> None:
    venue, journal = _MockVenue(real=True), _CapturingJournal()
    harness, _ = _harness(
        venue=venue, journal=journal,
        config=_config(require_operator_confirm=True, max_real_orders=5),
        prompt_fn=lambda _p: False,
    )
    report = harness.run()

    assert venue.submit_calls == []
    assert report["samples"] == 0
    assert [e.reason for e in journal.of_type("RISK_HALT")].count("operator_aborted") >= 1


def test_real_venue_budget_bounds_accepted_probes() -> None:
    """With confirm accepted and MAX_REAL_ORDERS=2, exactly 2 probes reach the venue."""
    venue, journal = _MockVenue(real=True), _CapturingJournal()
    harness, _ = _harness(
        venue=venue, journal=journal,
        config=_config(require_operator_confirm=True, max_real_orders=2),
        prompt_fn=lambda _p: True,
        samples_target=10,
    )
    report = harness.run()

    assert len(venue.submit_calls) == 2                   # budget consumed, then blocked
    assert report["samples"] == 2


def test_ambiguous_probe_submit_fires_best_effort_cancel() -> None:
    """Adversarial-review regression (HIGH): an ambiguous submit may have LANDED the probe
    (timeout after resting) — a GTC at the extreme tick must never rest unattended, so the
    harness fires a best-effort cancel-by-coid before reporting the block."""
    from polymarket.execution.mirror.mirror_engine import SubmitResult

    class _AmbiguousVenue(_MockVenue):
        def __init__(self) -> None:
            super().__init__(real=True)

        def submit_order(self, **kwargs):
            self.submit_calls.append(kwargs)
            coid = kwargs["client_order_id"]
            self._open[coid] = f"venue-{coid}"      # it DID land...
            return SubmitResult(accepted=False, ambiguous=True, message="timeout")

    venue, journal = _AmbiguousVenue(), _CapturingJournal()
    harness, _ = _harness(venue=venue, journal=journal,
                          config=_config(max_real_orders=5), samples_target=1)
    rec = harness.probe_once()

    assert rec.accepted is False and rec.reason == "ambiguous_submit"
    assert len(venue.cancel_calls) == 1                       # best-effort cancel fired
    assert venue._open == {}                                  # nothing rests at the venue
    assert any(e.reason == "latency_probe_ambiguous_rollback"
               for e in journal.of_type("MAKER_QUOTE_CANCELED"))
    assert harness.router.halted is True                      # ambiguous still halts the run


def test_kill_switch_blocks_probes_and_halts(tmp_path: Path) -> None:
    ks = tmp_path / "killswitch"
    ks.write_text("halt", encoding="utf-8")
    venue, journal = _MockVenue(), _CapturingJournal()
    harness, _ = _harness(
        venue=venue, journal=journal,
        config=_config(killswitch_path=ks),
        killswitch=True,
    )
    report = harness.run()

    assert venue.submit_calls == []
    assert harness.router.halted is True
    assert report["samples"] == 0
    assert any(e.reason == "kill_switch" for e in journal.of_type("RISK_HALT"))


# --------------------------------------------------------------------------------------
# CLI config
# --------------------------------------------------------------------------------------

def test_probe_config_from_env_and_missing_condition_rejected() -> None:
    cfg = LatencyProbeConfig.from_env({
        "POLYMARKET_MAKER_CONDITION_ID": "0xCOND",
        "POLYMARKET_MM_BRIDGE_ASSET_ID": "t1",
        "POLYMARKET_MM_LATENCY_SAMPLES": "7",
        "POLYMARKET_MM_LATENCY_CADENCE_S": "0",
        "POLYMARKET_MM_LATENCY_SIDE": "SELL",
    })
    assert cfg.condition_id == "0xcond"
    assert cfg.samples_target == 7
    assert cfg.prefer == "sell"
    assert cfg.size_contracts == 1.0        # default preserved: historical 1-contract probe
    with pytest.raises(ValueError):
        LatencyProbeConfig.from_env({})


def test_probe_size_env_override_reaches_the_order() -> None:
    """Venue-minimum regression: ``MAKER_SIZE_CONTRACTS`` sizes the probe order itself.

    Markets with ``minimum_order_size=5`` reject 1-share probes, so the same env knob the
    bridge uses must flow from ``from_env`` into the submitted order (``size_shares``).
    """
    cfg = LatencyProbeConfig.from_env({
        "POLYMARKET_MAKER_CONDITION_ID": "0xCOND",
        "POLYMARKET_MM_BRIDGE_ASSET_ID": "t1",
        "MAKER_SIZE_CONTRACTS": "5",
    })
    assert cfg.size_contracts == 5.0

    venue = _MockVenue()
    journal = _CapturingJournal()
    harness, _ = _harness(venue=venue, journal=journal)
    harness.cfg.size_contracts = 5.0
    rec = harness.probe_once()
    assert rec.accepted
    assert venue.submit_calls, "probe order must reach the venue"
    assert float(venue.submit_calls[-1]["size_shares"]) == 5.0


@pytest.mark.parametrize("bad", ["nan", "-5", "0", "inf", "-inf", "abc", ""])
def test_probe_size_fails_closed_on_malformed_env(bad: str) -> None:
    """A malformed MAKER_SIZE_CONTRACTS must raise (fail closed), never reach an order.

    NaN/negative sizes otherwise slip past the ``>``-based USD risk caps (fail open);
    from_env rejects them so main() exits 2 (adversarial-review MEDIUM finding)."""
    with pytest.raises(ValueError):
        LatencyProbeConfig.from_env({
            "POLYMARKET_MAKER_CONDITION_ID": "0xCOND",
            "POLYMARKET_MM_BRIDGE_ASSET_ID": "t1",
            "MAKER_SIZE_CONTRACTS": bad,
        })


def test_main_returns_2_on_missing_condition_id() -> None:
    from polymarket.execution.maker.mm_latency_harness import main

    env = {
        "POLYMARKET_LEADER_ADDRESS": "0x" + "1" * 40,
        "POLYMARKET_PRIVATE_KEY": "key",
        "POLYMARKET_API_KEY": "api",
        "POLYMARKET_API_SECRET": "secret",
        "POLYMARKET_PASSPHRASE": "pass",
        "POLYMARKET_FUNDER": "0x" + "2" * 40,
        # POLYMARKET_MAKER_CONDITION_ID deliberately missing
    }
    assert main(env) == 2
