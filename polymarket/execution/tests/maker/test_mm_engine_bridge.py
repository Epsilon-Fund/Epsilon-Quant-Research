"""Join-2a bridge tests: prove the MM-engine → maker bridge in DRY-RUN / mock only.

No test here places a network order. The MOCK venue records submits/cancels and reports
fills synchronously. The suite asserts:

* the SAME mm_engine ``Strategy`` / ``QueueModel`` / ``OrderManager`` / ``BookTracker`` drive
  the bridge (same-code-path: on a fills-free feed the bridge's orders+quotes telemetry is
  byte-identical to ``run_engine``'s);
* the ONLY thing that swaps is the fill path (fills come from the venue, booked by the same
  ``_apply_fill`` accounting; the fill-record schema is a superset of the backtest's);
* the reused safety harness — ``MAX_REAL_ORDERS`` + ``REQUIRE_OPERATOR_CONFIRM`` (shared
  ``RealOrderGate``) + kill-switch + per-trade/market/deployed/daily caps — sits in the order
  path;
* the venv boundary is clean (mm_engine imports no signing / no ``polymarket.execution``; the
  bridge imports no ``_kernel``);
* runs are deterministic.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.mirror.mirror_engine import SubmitResult

from polymarket.execution.maker.mm_engine_bridge import (
    LIVE_BRIDGE,
    BridgeConfig,
    DataApiFillSource,
    MMEngineBridge,
    VenueFill,
    VenueOrderRouter,
    ensure_mm_engine_importable,
)
from polymarket.execution.maker.mm_bridge_cli import build_bridge
from polymarket.execution.maker.order_safety import RealOrderGate

# the bridge import already put polymarket/research on sys.path
ensure_mm_engine_importable()
from mm_engine.engine import BACKTEST, EngineResult, run_engine  # noqa: E402
from mm_engine.interfaces import MarketEvent, Order  # noqa: E402
from mm_engine.latency_models import ConstantLatency  # noqa: E402
from mm_engine.queue_models import OptimisticQueue  # noqa: E402
from mm_engine.strategies import SymmetricQuoter, mid  # noqa: E402
from mm_engine.telemetry import Telemetry  # noqa: E402


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _config(**overrides: Any) -> ExecutionConfig:
    base = ExecutionConfig(
        leader_address="0x" + "1" * 40,
        private_key="key",
        api_key="api",
        api_secret="secret",
        passphrase="pass",
        funder="0x" + "2" * 40,
        chain_id=137,
        signature_type=1,
        clob_url="https://clob.polymarket.com",
        gamma_url="https://gamma-api.polymarket.com",
        data_url="https://data-api.polymarket.com",
        ws_url="wss://ws-live-data.polymarket.com",
        max_capital_usd=100,
        per_trade_cap_usd=50,
        per_market_cap_usd=50,
        sizing_usd=50,
        max_open_positions=3,
        default_order_type="FOK",
        pricing_mode="leader_fill",
        price_deviation_pct=2.0,
        daily_loss_halt_usd=200,
        killswitch_path=Path("/tmp/mm_bridge_no_such_killswitch"),
        journal_dir=Path("./journal_logs"),
        log_level="INFO",
        max_real_orders=5,
        require_operator_confirm=False,
    )
    for key, value in overrides.items():
        base = replace(base, **{key: value})
    return base


class _CapturingJournal:
    """Duck-typed JsonlWriter: records events in memory (no file IO)."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def write(self, event: Any) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def of_type(self, event_type: str) -> list[Any]:
        return [e for e in self.events if type(e).event_type == event_type]


class _MockVenue:
    """Mock CLOB venue: records submits/cancels, tracks its own open orders, reports
    scheduled fills. No network. Mimics ``RealVenueAdapter.reconcile_open_orders`` so the
    bridge's venue-state reconcile can be exercised on the ``is_real_venue()==True`` path.

    ``reject_first`` makes the first N submits return ``accepted=False`` (simulate a venue
    NACK / safety skip); ``external_cancel`` / ``add_orphan`` simulate the venue's open set
    drifting from the engine's (external cancel, missed ack, orphan).
    """

    def __init__(self, *, real: bool = False, reject_first: int = 0) -> None:
        self.real = real
        self.reject_first = reject_first
        self.submit_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []
        self._open: dict[str, str] = {}          # coid -> venue_order_id (what the venue holds)
        self._pending_fills: list[VenueFill] = []

    def is_real_venue(self) -> bool:
        return self.real

    def submit_order(self, **kwargs: Any) -> SubmitResult:
        self.submit_calls.append(kwargs)
        coid = kwargs["client_order_id"]
        if self.reject_first > 0:
            self.reject_first -= 1
            return SubmitResult(accepted=False, ambiguous=False, message="mock reject")
        voi = f"venue-{coid}"
        self._open[coid] = voi
        return SubmitResult(accepted=True, ambiguous=False, venue_order_id=voi)

    def cancel_order(self, *, client_order_id: str | None = None,
                     venue_order_id: str | None = None) -> dict[str, Any]:
        self.cancel_calls.append({"client_order_id": client_order_id,
                                  "venue_order_id": venue_order_id})
        self._open.pop(client_order_id, None)
        return {"ambiguous": False}

    def reconcile_open_orders(self, expected: Any = None) -> Any:  # noqa: ARG002
        from types import SimpleNamespace
        return SimpleNamespace(venue_open_client_order_ids=tuple(sorted(self._open)))

    def external_cancel(self, coid: str) -> None:
        """Simulate a venue-side / external cancel the engine wasn't told about."""
        self._open.pop(coid, None)

    def add_orphan(self, coid: str) -> None:
        """Simulate a venue order the engine has no record of."""
        self._open[coid] = f"venue-{coid}"

    # acts as its own FillSource
    def schedule_fill(self, fill: VenueFill) -> None:
        self._pending_fills.append(fill)

    def poll(self, active_orders: list) -> list[VenueFill]:  # noqa: ARG002
        out = self._pending_fills
        self._pending_fills = []
        return out


def _engine_coids(bridge: MMEngineBridge) -> set[str]:
    """The venue coids the bridge's OrderManager currently believes are resting."""
    return {bridge.router.venue_coid(ao.client_id) for ao in bridge.om.active_orders()}


def _book_event(ts: int, bid: float, ask: float, *, bid_sz: float = 100.0,
                ask_sz: float = 100.0, token: str = "t1") -> MarketEvent:
    return MarketEvent(
        type="book", token_id=token, ts_exchange=ts, ts_local_iso="", ts_monotonic_ns=0,
        payload={
            "asset_id": token, "market": "0xcond",
            "bids": [{"price": bid, "size": bid_sz}],
            "asks": [{"price": ask, "size": ask_sz}],
        },
    )


def _trade_event(ts: int, price: float, side: str, size: float,
                 token: str = "t1") -> MarketEvent:
    return MarketEvent(
        type="last_trade", token_id=token, ts_exchange=ts, ts_local_iso="", ts_monotonic_ns=0,
        payload={"asset_id": token, "market": "0xcond", "price": price, "side": side, "size": size},
    )


def _bridge(
    *,
    venue: _MockVenue,
    journal: _CapturingJournal,
    config: ExecutionConfig | None = None,
    bridge_config: BridgeConfig | None = None,
    prompt_fn=None,
    strategy: SymmetricQuoter | None = None,
    queue_model: OptimisticQueue | None = None,
    telemetry: Telemetry | None = None,
) -> MMEngineBridge:
    cfg = config or _config()
    bcfg = bridge_config or BridgeConfig(
        condition_id="0xcond", asset_id="t1", half_spread=0.01, size_contracts=10.0, tick=0.001,
    )
    gate = RealOrderGate(config=cfg, journal=journal, label="mm_bridge", prompt_fn=prompt_fn)
    router = VenueOrderRouter(
        venue=venue, config=cfg, journal=journal, gate=gate, order_type=bcfg.order_type,
        coid_prefix=bcfg.coid_prefix,
    )
    return MMEngineBridge(
        strategy=strategy or SymmetricQuoter(),
        queue_model=queue_model or OptimisticQueue(),
        latency_model=ConstantLatency(0.0),
        router=router,
        bridge_config=bcfg,
        journal=journal,
        telemetry=telemetry,
    )


# --------------------------------------------------------------------------------------
# same-code-path / same models
# --------------------------------------------------------------------------------------

def test_bridge_uses_the_same_strategy_and_queue_objects() -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    strat, qm = SymmetricQuoter(), OptimisticQueue()
    bridge = _bridge(venue=venue, journal=journal, strategy=strat, queue_model=qm)
    assert bridge.strategy is strat
    assert bridge.queue_model is qm


def test_book_event_places_two_sided_quote_through_venue() -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))

    assert len(venue.submit_calls) == 2
    sides = {c["side"]: c for c in venue.submit_calls}
    assert sides["BUY"]["price"] == 0.47
    assert sides["SELL"]["price"] == 0.49
    assert {c["order_type"] for c in venue.submit_calls} == {"GTC"}
    # journal continuity: MAKER_QUOTE_PLACED for both sides
    assert len(journal.of_type("MAKER_QUOTE_PLACED")) == 2


def test_only_the_fill_path_swaps_orders_and_quotes_match_backtest() -> None:
    """On a fills-free feed, the bridge's orders+quotes telemetry is byte-identical to
    ``run_engine``'s — proving the quoting code path is untouched; only fills differ."""
    params = {"half_spread": 0.01, "size": 10.0, "tick": 0.001}
    feed = [
        _book_event(1000, 0.47, 0.49),
        _book_event(1100, 0.46, 0.50, bid_sz=80),   # book moves → replace
        _book_event(1200, 0.46, 0.50),              # unchanged → idempotent no-op
    ]

    bt: EngineResult = run_engine(
        list(feed),
        strategy=SymmetricQuoter(),
        queue_model=OptimisticQueue(),
        latency_model=ConstantLatency(0.0),
        mode=BACKTEST,
        params=params,
        fee_model=__import__("mm_engine.fees", fromlist=["FeeModel"]).FeeModel.fee_free_model(),
    )

    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(
        venue=venue, journal=journal,
        bridge_config=BridgeConfig(condition_id="0xcond", asset_id="t1",
                                   half_spread=0.01, size_contracts=10.0, tick=0.001),
    )
    res = bridge.run(list(feed), fill_source=venue)

    assert res.fill_count == 0 and bt.fill_count == 0
    assert res.orders == bt.orders          # identical place/cancel/replace ops
    assert res.quotes == bt.quotes          # identical per-event quote snapshots
    assert res.mode == LIVE_BRIDGE


def test_venue_fill_books_position_and_pnl_via_same_accounting() -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    buy_coid = next(c["client_order_id"] for c in venue.submit_calls if c["side"] == "BUY")

    bridge.ingest_fill(VenueFill(
        token_id="t1", side="BUY", price=0.47, qty=50.0, ts_ms=2000,
        client_order_id=buy_coid, venue_order_id=f"venue-{buy_coid}", transaction_hash="0xtx",
    ))
    res = bridge.result()
    assert res.fill_count == 1
    assert res.filled_qty == 50.0
    assert res.position["t1"] == 50.0
    # gross_cash = -dq*price = -(+50)*0.47
    assert res.gross_cash == pytest.approx(-50 * 0.47)
    assert journal.of_type("MAKER_FILL_TELEMETRY")


def test_unmatched_funder_fill_is_still_booked_with_rank_none() -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    # a taker fill at a price that matches no resting quote and no coid
    bridge.ingest_fill(VenueFill(
        token_id="t1", side="SELL", price=0.55, qty=7.0, ts_ms=2000, transaction_hash="0xtx2",
    ))
    res = bridge.result()
    assert res.fill_count == 1
    rec = res.fills[0]
    assert rec["matched"] is False
    assert rec["queue_ahead"] is None
    assert rec["client_id"] == ""


# --------------------------------------------------------------------------------------
# telemetry parity vs the backtest
# --------------------------------------------------------------------------------------

def test_fill_record_schema_is_superset_of_backtest() -> None:
    """Join-2d must consume live + backtest fill logs identically → the bridge fill record
    carries every core key the backtest emits (plus live-only measurement fields)."""
    from mm_engine.fees import FeeModel

    # backtest that produces a fill: 2c-spread touch quote, then a SELL trade-through
    params = {"half_spread": 0.01, "size": 100.0, "tick": 0.001}
    bt = run_engine(
        [
            _book_event(1000, 0.47, 0.49, bid_sz=10, ask_sz=10),
            _trade_event(1100, 0.47, "SELL", 200.0),   # hits our resting BUY at 0.47
        ],
        strategy=SymmetricQuoter(),
        queue_model=OptimisticQueue(),
        latency_model=ConstantLatency(0.0),
        mode=BACKTEST,
        params=params,
        fee_model=FeeModel.fee_free_model(),
    )
    assert bt.fill_count >= 1
    backtest_keys = set(bt.fills[0].keys())

    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    buy_coid = next(c["client_order_id"] for c in venue.submit_calls if c["side"] == "BUY")
    bridge.ingest_fill(VenueFill(
        token_id="t1", side="BUY", price=0.47, qty=50.0, ts_ms=1100, client_order_id=buy_coid,
    ))
    bridge_keys = set(bridge.result().fills[0].keys())

    missing = backtest_keys - bridge_keys
    assert not missing, f"bridge fill record missing backtest keys: {missing}"
    # and the live-only measurement fields the task enumerates are present
    assert {"own_round_trip_ms", "news_proximate", "fill_share_this_market", "source"} <= bridge_keys


def test_result_is_engine_result_and_settles_like_backtest() -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    buy_coid = next(c["client_order_id"] for c in venue.submit_calls if c["side"] == "BUY")
    bridge.ingest_fill(VenueFill(token_id="t1", side="BUY", price=0.47, qty=50.0, ts_ms=1100,
                                 client_order_id=buy_coid))
    res = bridge.result()
    assert isinstance(res, EngineResult)
    settlement = res.settle({"t1": 1.0})   # YES resolves true
    # 50 contracts bought at 0.47, settle at 1.0 → +0.53 * 50
    assert settlement.settled_pnl == pytest.approx(50 * (1.0 - 0.47))


# --------------------------------------------------------------------------------------
# safety harness in the order path
# --------------------------------------------------------------------------------------

def test_real_venue_max_real_orders_zero_blocks_submits() -> None:
    venue = _MockVenue(real=True)
    journal = _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal, config=_config(max_real_orders=0))
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))

    assert venue.submit_calls == []
    assert any(e.reason == "max_real_orders" for e in journal.of_type("RISK_HALT"))
    skips = [e for e in journal.of_type("MAKER_QUOTE_SKIPPED") if e.reason == "max_real_orders"]
    assert len(skips) == 2   # BUY + SELL both blocked


def test_real_venue_operator_confirm_declined_blocks_submits() -> None:
    venue = _MockVenue(real=True)
    journal = _CapturingJournal()
    bridge = _bridge(
        venue=venue, journal=journal,
        config=_config(require_operator_confirm=True),
        prompt_fn=lambda _prompt: False,     # operator declines every order
    )
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))

    assert venue.submit_calls == []
    assert [e.reason for e in journal.of_type("RISK_HALT")] == ["operator_aborted", "operator_aborted"]


def test_real_venue_operator_confirm_accepted_allows_submit() -> None:
    venue = _MockVenue(real=True)
    journal = _CapturingJournal()
    bridge = _bridge(
        venue=venue, journal=journal,
        config=_config(require_operator_confirm=True, max_real_orders=5),
        prompt_fn=lambda _prompt: True,
    )
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    assert len(venue.submit_calls) == 2


def test_kill_switch_blocks_submit_and_halts(tmp_path: Path) -> None:
    ks = tmp_path / "killswitch"
    ks.write_text("halt", encoding="utf-8")
    venue = _MockVenue()          # even a fake/DRY-RUN venue respects the kill-switch
    journal = _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal, config=_config(killswitch_path=ks))

    # run over a 2-event feed → halts, second event never processed
    bridge.run([_book_event(1000, 0.47, 0.49), _book_event(1100, 0.47, 0.49)], fill_source=venue)

    assert venue.submit_calls == []
    assert any(e.reason == "kill_switch" for e in journal.of_type("RISK_HALT"))
    assert bridge.router.halted is True
    assert bridge.event_count == 1    # loop broke after the halting event


def test_per_trade_cap_blocks_oversized_order() -> None:
    venue = _MockVenue()
    journal = _CapturingJournal()
    # size 10 @ ~0.47 → notional ~4.7 > per_trade_cap 0.1 → size_cap veto
    bridge = _bridge(venue=venue, journal=journal, config=_config(per_trade_cap_usd=0.1))
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))

    assert venue.submit_calls == []
    skips = [e for e in journal.of_type("MAKER_QUOTE_SKIPPED") if e.reason == "size_cap"]
    assert len(skips) == 2


def test_gap_cancels_resting_orders_at_venue() -> None:
    from mm_engine.events import GapMarker

    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    assert len(venue.submit_calls) == 2
    bridge.on_gap(GapMarker(reason="disconnect"))
    assert len(venue.cancel_calls) == 2


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------

def test_deterministic_orders_and_quotes() -> None:
    feed = [_book_event(1000, 0.47, 0.49), _book_event(1100, 0.46, 0.50)]

    def run() -> tuple[list, list]:
        venue, journal = _MockVenue(), _CapturingJournal()
        bridge = _bridge(venue=venue, journal=journal)
        res = bridge.run(list(feed), fill_source=venue)
        return res.orders, res.quotes

    o1, q1 = run()
    o2, q2 = run()
    assert o1 == o2
    assert q1 == q2


# --------------------------------------------------------------------------------------
# item 1a — phantom resting orders: rejected/skipped places roll back + re-propose
# --------------------------------------------------------------------------------------

def test_rejected_place_rolls_back_and_is_reproposed_next_event() -> None:
    """A place the venue REJECTS must not linger as a phantom in the OrderManager — it is
    rolled back so the next reconcile re-proposes it, and it rests once the venue accepts."""
    venue = _MockVenue(real=True, reject_first=2)   # NACK the first BUY+SELL, accept after
    journal = _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)

    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    # both rejected → rolled back → NOT resting, and the venue holds nothing
    assert bridge.om.active_orders() == []
    assert _engine_coids(bridge) == set()
    assert len(venue.submit_calls) == 2

    bridge.on_market_event(_book_event(1000, 0.47, 0.49))   # same book → re-propose
    assert len(venue.submit_calls) == 4                     # 2 re-proposed
    assert len(bridge.om.active_orders()) == 2              # now resting
    assert _engine_coids(bridge) == set(venue._open)        # engine view == venue reality


def test_gate_skipped_place_is_reproposed_each_event_not_silenced() -> None:
    """A gate-skipped place must be re-proposed every event (not left as an idempotent
    no-op). Two events over the same book → two skips PER event, nothing resting."""
    venue = _MockVenue(real=True)
    journal = _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal, config=_config(max_real_orders=0))

    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))

    assert venue.submit_calls == []
    skips = [e for e in journal.of_type("MAKER_QUOTE_SKIPPED") if e.reason == "max_real_orders"]
    assert len(skips) == 4          # 2 sides × 2 events — proves re-proposal, not a phantom
    assert bridge.om.active_orders() == []


# --------------------------------------------------------------------------------------
# item 1b — periodic venue-state reconcile: resting set must equal the venue's open orders
# --------------------------------------------------------------------------------------

def test_reconcile_drops_order_the_venue_no_longer_has() -> None:
    venue = _MockVenue(real=True)
    journal = _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    assert len(bridge.om.active_orders()) == 2
    gone = sorted(venue._open)[0]                 # pick one coid the venue holds
    venue.external_cancel(gone)                   # external/venue-side cancel

    summary = bridge.reconcile_with_venue()

    assert summary["dropped"] == 1 and summary["reconciled"] is True
    assert gone not in _engine_coids(bridge)
    assert _engine_coids(bridge) == set(venue._open)      # invariant: equal after reconcile
    missing = [e for e in journal.of_type("MAKER_QUOTE_CANCELED")
               if e.reason == "venue_reconcile_missing"]
    assert len(missing) == 1 and missing[0].client_order_id == gone


def test_reconcile_cancels_orphan_venue_order() -> None:
    venue = _MockVenue(real=True)
    journal = _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    venue.add_orphan("orphan-coid")               # venue has an order the engine never tracked

    summary = bridge.reconcile_with_venue()

    assert summary["orphans_cancelled"] == 1
    assert any(c["client_order_id"] == "orphan-coid" for c in venue.cancel_calls)
    assert "orphan-coid" not in venue._open       # cancelled at the venue
    assert _engine_coids(bridge) == set(venue._open)      # invariant: equal after reconcile
    assert any(e.reason == "venue_reconcile_orphan"
               for e in journal.of_type("MAKER_QUOTE_CANCELED"))


def test_resting_set_equals_venue_after_reconcile_both_directions() -> None:
    venue = _MockVenue(real=True)
    journal = _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    # drift in BOTH directions at once: venue loses one of ours, gains an orphan
    venue.external_cancel(sorted(venue._open)[0])
    venue.add_orphan("orphan-coid")

    bridge.reconcile_with_venue()

    assert _engine_coids(bridge) == set(venue._open)   # the two resting sets are equal


def test_run_invokes_periodic_reconcile_at_interval() -> None:
    venue = _MockVenue(real=True)
    journal = _CapturingJournal()
    bridge = _bridge(
        venue=venue, journal=journal,
        bridge_config=BridgeConfig(condition_id="0xcond", asset_id="t1", half_spread=0.01,
                                   size_contracts=10.0, tick=0.001, reconcile_interval_events=1),
    )
    calls: list[int] = []
    orig = bridge.reconcile_with_venue
    bridge.reconcile_with_venue = lambda: (calls.append(1), orig())[1]  # type: ignore[method-assign]

    bridge.run([_book_event(1000, 0.47, 0.49), _book_event(1100, 0.46, 0.50)], fill_source=venue)
    assert len(calls) == 2      # reconcile ran once per market event (interval=1)


# --------------------------------------------------------------------------------------
# item 3 — fill_share_this_market = our fills / total market trades (backtest AND bridge)
# --------------------------------------------------------------------------------------

def test_fill_share_backtest_equals_our_over_total_on_synthetic_tape() -> None:
    from mm_engine.fees import FeeModel

    params = {"half_spread": 0.01, "size": 100.0, "tick": 0.001}
    tape = [
        _book_event(1000, 0.47, 0.49, bid_sz=10, ask_sz=10),  # rest BUY 0.47 / SELL 0.49
        _trade_event(1100, 0.48, "SELL", 5.0),                # market trade #1, no fill
        _trade_event(1200, 0.47, "SELL", 200.0),              # #2, fills BUY → 1/2
        _trade_event(1300, 0.49, "BUY", 200.0),               # #3, fills SELL → 2/3
    ]
    res = run_engine(
        tape, strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
        latency_model=ConstantLatency(0.0), mode=BACKTEST, params=params,
        fee_model=FeeModel.fee_free_model(),
    )
    assert res.fill_count == 2
    assert res.fills[0]["fill_share_this_market"] == pytest.approx(1 / 2)
    assert res.fills[1]["fill_share_this_market"] == pytest.approx(2 / 3)


def test_fill_share_bridge_equals_our_over_total_on_synthetic_tape() -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    buy_coid = next(c["client_order_id"] for c in venue.submit_calls if c["side"] == "BUY")
    sell_coid = next(c["client_order_id"] for c in venue.submit_calls if c["side"] == "SELL")
    # two market trades on the tape (the denominator)
    bridge.on_market_event(_trade_event(1100, 0.48, "SELL", 5.0))
    bridge.on_market_event(_trade_event(1200, 0.48, "BUY", 5.0))
    # our two fills reported by the venue
    bridge.ingest_fill(VenueFill(token_id="t1", side="BUY", price=0.47, qty=10.0, ts_ms=1200,
                                 client_order_id=buy_coid))
    bridge.ingest_fill(VenueFill(token_id="t1", side="SELL", price=0.49, qty=10.0, ts_ms=1250,
                                 client_order_id=sell_coid))
    fills = bridge.result().fills
    assert fills[0]["fill_share_this_market"] == pytest.approx(1 / 2)   # 1 ours / 2 market
    assert fills[1]["fill_share_this_market"] == pytest.approx(2 / 2)   # 2 ours / 2 market


# --------------------------------------------------------------------------------------
# venv boundary + non-modification of frozen code
# --------------------------------------------------------------------------------------

def test_mm_engine_imports_no_signing_or_execution_code() -> None:
    """The research mm_engine must not import the execution stack or any signer — the bridge
    is a one-way path dependency (execution imports mm_engine, never the reverse)."""
    research_root = Path(ensure_mm_engine_importable())
    forbidden = ("py_clob_client", "polymarket.execution", "clob_signer", "ClobSigner")
    for py in (research_root / "mm_engine").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{py.name} references {token!r} — venv boundary broken"


def test_bridge_module_does_not_import_kernel() -> None:
    """Real orders route through the venue adapter (build_venue_adapter), never by the bridge
    constructing kernel objects directly."""
    bridge_src = Path(__file__).resolve().parents[2] / "maker" / "mm_engine_bridge.py"
    text = bridge_src.read_text(encoding="utf-8")
    assert "_kernel" not in text


def test_build_bridge_wires_join1_models() -> None:
    """The operator CLI seam wires the SAME JOIN-1 models (SymmetricQuoter + OptimisticQueue
    + ConstantLatency)."""
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = build_bridge(
        config=_config(),
        bridge_config=BridgeConfig(condition_id="0xcond", asset_id="t1"),
        venue=venue,
        journal=journal,
    )
    assert isinstance(bridge.strategy, SymmetricQuoter)
    assert isinstance(bridge.queue_model, OptimisticQueue)
    assert isinstance(bridge.latency_model, ConstantLatency)


def test_bridge_config_from_env() -> None:
    env = {
        "POLYMARKET_MAKER_CONDITION_ID": "0xCOND",
        "POLYMARKET_MM_BRIDGE_ASSET_ID": "tokenYES",
        "POLYMARKET_MM_BRIDGE_HALF_SPREAD": "0.02",
        "MAKER_SIZE_CONTRACTS": "1",
    }
    cfg = BridgeConfig.from_env(env)
    assert cfg.condition_id == "0xcond"      # lowercased
    assert cfg.asset_id == "tokenYES"
    assert cfg.half_spread == 0.02
    assert cfg.size_contracts == 1.0
    assert cfg.params == {"half_spread": 0.02, "size": 1.0, "tick": 0.001}


# --------------------------------------------------------------------------------------
# DataApiFillSource (live fill path glue, reuses the maker DataApiTradeClient shape)
# --------------------------------------------------------------------------------------

class _FakeDataClient:
    def __init__(self, trades: list[dict]) -> None:
        self._trades = trades

    def get_trades(self, condition_id: str) -> list[dict]:  # noqa: ARG002
        return list(self._trades)


_SESSION_START = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
_TS_BEFORE = int(_SESSION_START.timestamp()) - 3600   # 1h before session
_TS_AFTER = int(_SESSION_START.timestamp()) + 3600    # 1h after session


def test_data_api_fill_source_matches_funder_fills_to_resting_quotes() -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))

    funder = _config().funder
    trades = [
        {"transactionHash": "0xaaa", "asset": "t1", "proxyWallet": funder,
         "side": "BUY", "price": 0.47, "size": 3.0, "timestamp": _TS_AFTER},
        {"transactionHash": "0xbbb", "asset": "t1", "proxyWallet": "0x" + "9" * 40,
         "side": "BUY", "price": 0.47, "size": 5.0, "timestamp": _TS_AFTER},  # not us → ignored
    ]
    src = DataApiFillSource(
        condition_id="0xcond", asset_id="t1", funder=funder,
        session_start=_SESSION_START,
        data_client=_FakeDataClient(trades), tick=0.001,
    )
    fills = src.poll(bridge.om.active_orders())
    assert len(fills) == 1
    assert fills[0].qty == 3.0
    # matched to the resting BUY quote (client id assigned by the OrderManager)
    assert fills[0].client_order_id is not None
    # idempotent: re-poll yields nothing (the fill was already seen)
    assert src.poll(bridge.om.active_orders()) == []


def test_data_api_fill_source_skips_pre_session_trades() -> None:
    """The funder's trade HISTORY (pre-session) must NOT be booked as live fills — the same
    guard maker_engine.process_fills_once applies. Regression for the confirmed HIGH bug."""
    funder = _config().funder
    trades = [
        {"transactionHash": "0xold", "asset": "t1", "proxyWallet": funder,
         "side": "BUY", "price": 0.47, "size": 100.0, "timestamp": _TS_BEFORE},  # history
        {"transactionHash": "0xnew", "asset": "t1", "proxyWallet": funder,
         "side": "BUY", "price": 0.47, "size": 4.0, "timestamp": _TS_AFTER},     # live
    ]
    src = DataApiFillSource(
        condition_id="0xcond", asset_id="t1", funder=funder,
        session_start=_SESSION_START, data_client=_FakeDataClient(trades), tick=0.001,
    )
    fills = src.poll([])
    assert len(fills) == 1
    assert fills[0].transaction_hash == "0xnew"
    assert fills[0].qty == 4.0


def test_data_api_fill_source_skips_rows_with_no_timestamp() -> None:
    """Adversarial-review regression (LOW): a funder row whose timestamp cannot be parsed
    must be treated as PRE-SESSION (skipped), not booked as a live fill with ts_ms=0 —
    a genuinely-live data-api fill always carries a timestamp."""
    funder = _config().funder
    trades = [
        {"transactionHash": "0xnots", "asset": "t1", "proxyWallet": funder,
         "side": "BUY", "price": 0.47, "size": 9.0},                          # no timestamp
        {"transactionHash": "0xgood", "asset": "t1", "proxyWallet": funder,
         "side": "BUY", "price": 0.47, "size": 4.0, "timestamp": _TS_AFTER},  # live
    ]
    src = DataApiFillSource(
        condition_id="0xcond", asset_id="t1", funder=funder,
        session_start=_SESSION_START, data_client=_FakeDataClient(trades), tick=0.001,
    )
    fills = src.poll([])
    assert len(fills) == 1
    assert fills[0].transaction_hash == "0xgood"


class _AmbiguousVenue(_MockVenue):
    """Real-ish venue whose submits time out AFTER landing: SubmitResult says ambiguous,
    but the venue actually holds the order — the divergence case the rollback must cancel."""

    def __init__(self) -> None:
        super().__init__(real=True)

    def submit_order(self, **kwargs: Any) -> SubmitResult:
        self.submit_calls.append(kwargs)
        coid = kwargs["client_order_id"]
        self._open[coid] = f"venue-{coid}"          # it DID land...
        return SubmitResult(accepted=False, ambiguous=True, message="timeout")  # ...ambiguously


def test_ambiguous_bridge_submit_fires_best_effort_cancel_and_final_reconcile() -> None:
    """Adversarial-review regression (MEDIUM): an ambiguous submit rolls the intent out of
    the engine while the venue may hold the order, and the halt breaks the loop before the
    next periodic reconcile — nothing may rest unattended. The rollback now cancels by coid
    and run() does one final venue reconcile on halt."""
    venue, journal = _AmbiguousVenue(), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)

    bridge.run([_book_event(1000, 0.47, 0.49)], fill_source=venue)

    assert bridge.router.halted is True                       # ambiguous submit halts
    assert bridge.om.active_orders() == []                    # intent rolled back
    assert venue._open == {}                                  # NOTHING rests at the venue
    assert any(e.reason == "ambiguous_rollback"
               for e in journal.of_type("MAKER_QUOTE_CANCELED"))


def test_data_api_fill_source_emits_all_fills_of_multi_fill_tx() -> None:
    """One transaction can fill several of our resting levels; the dedup key must not collapse
    them to a single fill. Regression for the confirmed multi-fill-tx bug."""
    funder = _config().funder
    trades = [
        {"transactionHash": "0xsweep", "asset": "t1", "proxyWallet": funder,
         "side": "BUY", "price": 0.47, "size": 50.0, "timestamp": _TS_AFTER},
        {"transactionHash": "0xsweep", "asset": "t1", "proxyWallet": funder,
         "side": "BUY", "price": 0.46, "size": 30.0, "timestamp": _TS_AFTER},  # same tx, 2nd leg
    ]
    src = DataApiFillSource(
        condition_id="0xcond", asset_id="t1", funder=funder,
        session_start=_SESSION_START, data_client=_FakeDataClient(trades), tick=0.001,
    )
    fills = src.poll([])
    assert len(fills) == 2
    assert {round(f.qty, 6) for f in fills} == {50.0, 30.0}
    # idempotent across polls
    assert src.poll([]) == []


    # --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Join-2c wiring: hard inventory cap (contracts) + measured latency constant
# --------------------------------------------------------------------------------------

def test_inventory_cap_withholds_buy_side_at_positive_cap() -> None:
    """At +cap the BUY side is withheld (reduce-only): no new BUY reaches the venue, the
    SELL still quotes, and the skip is journaled as ``inventory_cap``."""
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(
        venue=venue, journal=journal,
        bridge_config=BridgeConfig(condition_id="0xcond", asset_id="t1", half_spread=0.01,
                                   size_contracts=10.0, tick=0.001,
                                   max_inventory_contracts=5.0),
    )
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    buy = next(c for c in venue.submit_calls if c["side"] == "BUY")
    assert buy["size_shares"] == pytest.approx(5.0)   # clamped to the band (cap 5, clip 10)
    bridge.ingest_fill(VenueFill(token_id="t1", side="BUY", price=0.47, qty=5.0, ts_ms=1050,
                                 client_order_id=buy["client_order_id"]))
    assert bridge.positions["t1"].qty == 5.0          # at the cap — NEVER beyond it

    submits_before = len(venue.submit_calls)
    bridge.on_market_event(_book_event(1100, 0.47, 0.49))
    new_submits = venue.submit_calls[submits_before:]
    assert all(c["side"] != "BUY" for c in new_submits)          # no BUY reaches the venue
    assert {ao.order.side for ao in bridge.om.active_orders()} == {"SELL"}
    skips = [e for e in journal.of_type("MAKER_QUOTE_SKIPPED") if e.reason == "inventory_cap"]
    assert skips and skips[0].side == "BUY"


def test_inventory_cap_clamps_clip_so_a_full_fill_cannot_overshoot() -> None:
    """Adversarial-review regression (HIGH): with cap 5 and clip 10, side-withholding alone
    would let one full-clip fill land position 10 — 2× the 'hard' cap. The cap must clamp
    the ORDER SIZE to the remaining band so even a full fill stops exactly at ±cap."""
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(
        venue=venue, journal=journal,
        bridge_config=BridgeConfig(condition_id="0xcond", asset_id="t1", half_spread=0.01,
                                   size_contracts=10.0, tick=0.001,
                                   max_inventory_contracts=5.0),
    )
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    sizes = {c["side"]: c["size_shares"] for c in venue.submit_calls}
    assert sizes == {"BUY": pytest.approx(5.0), "SELL": pytest.approx(5.0)}

    # a short position widens the BUY band and narrows the SELL band asymmetrically:
    # inventory −3 → BUY may rest 8 (cross zero up to +5), SELL only 2 (down to −5)
    bridge.ingest_fill(VenueFill(token_id="t1", side="SELL", price=0.55, qty=3.0, ts_ms=1050))
    assert bridge.positions["t1"].qty == pytest.approx(-3.0)
    before = len(venue.submit_calls)
    bridge.on_market_event(_book_event(1100, 0.46, 0.50))
    sizes2 = {c["side"]: c["size_shares"] for c in venue.submit_calls[before:]}
    assert sizes2["BUY"] == pytest.approx(8.0)
    assert sizes2["SELL"] == pytest.approx(2.0)
    # worst case after FULL fills of both: +5 or −5 — never beyond the cap


def test_inventory_cap_withholds_sell_side_at_negative_cap() -> None:
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(
        venue=venue, journal=journal,
        bridge_config=BridgeConfig(condition_id="0xcond", asset_id="t1", half_spread=0.01,
                                   size_contracts=10.0, tick=0.001,
                                   max_inventory_contracts=5.0),
    )
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    sell_coid = next(c["client_order_id"] for c in venue.submit_calls if c["side"] == "SELL")
    bridge.ingest_fill(VenueFill(token_id="t1", side="SELL", price=0.49, qty=5.0, ts_ms=1050,
                                 client_order_id=sell_coid))
    assert bridge.positions["t1"].qty == -5.0

    bridge.on_market_event(_book_event(1100, 0.47, 0.49))
    assert {ao.order.side for ao in bridge.om.active_orders()} == {"BUY"}
    skips = [e for e in journal.of_type("MAKER_QUOTE_SKIPPED") if e.reason == "inventory_cap"]
    assert skips and skips[0].side == "SELL"


def test_inventory_cap_releases_when_position_reduces() -> None:
    """Reduce-only quoting must re-open the withheld side once inventory re-enters the band."""
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(
        venue=venue, journal=journal,
        bridge_config=BridgeConfig(condition_id="0xcond", asset_id="t1", half_spread=0.01,
                                   size_contracts=10.0, tick=0.001,
                                   max_inventory_contracts=5.0),
    )
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    buy_coid = next(c["client_order_id"] for c in venue.submit_calls if c["side"] == "BUY")
    bridge.ingest_fill(VenueFill(token_id="t1", side="BUY", price=0.47, qty=5.0, ts_ms=1050,
                                 client_order_id=buy_coid))
    bridge.on_market_event(_book_event(1100, 0.47, 0.49))
    assert {ao.order.side for ao in bridge.om.active_orders()} == {"SELL"}

    # the resting SELL reduces the position → the BUY side must come back
    sell_coid = next(ao.client_id for ao in bridge.om.active_orders())
    bridge.ingest_fill(VenueFill(token_id="t1", side="SELL", price=0.49, qty=4.0, ts_ms=1150,
                                 client_order_id=sell_coid))
    assert bridge.positions["t1"].qty == pytest.approx(1.0)
    bridge.on_market_event(_book_event(1200, 0.47, 0.49))
    assert {ao.order.side for ao in bridge.om.active_orders()} == {"BUY", "SELL"}


def test_inventory_cap_disabled_or_slack_keeps_backtest_parity() -> None:
    """cap=0 (disabled) and a slack cap must not perturb the byte-identical orders/quotes
    parity with the backtest on a fills-free feed."""
    from mm_engine.fees import FeeModel

    params = {"half_spread": 0.01, "size": 10.0, "tick": 0.001}
    feed = [_book_event(1000, 0.47, 0.49), _book_event(1100, 0.46, 0.50, bid_sz=80)]
    bt = run_engine(
        list(feed), strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
        latency_model=ConstantLatency(0.0), mode=BACKTEST, params=params,
        fee_model=FeeModel.fee_free_model(),
    )
    for cap in (0.0, 1000.0):
        venue, journal = _MockVenue(), _CapturingJournal()
        bridge = _bridge(
            venue=venue, journal=journal,
            bridge_config=BridgeConfig(condition_id="0xcond", asset_id="t1", half_spread=0.01,
                                       size_contracts=10.0, tick=0.001,
                                       max_inventory_contracts=cap),
        )
        res = bridge.run(list(feed), fill_source=venue)
        assert res.orders == bt.orders, f"cap={cap} broke orders parity"
        assert res.quotes == bt.quotes, f"cap={cap} broke quotes parity"


def test_bridge_config_from_env_parses_join2c_knobs() -> None:
    cfg = BridgeConfig.from_env({
        "POLYMARKET_MAKER_CONDITION_ID": "0xcond",
        "POLYMARKET_MM_BRIDGE_LATENCY_MS": "203",
        "POLYMARKET_MM_BRIDGE_MAX_INVENTORY": "5",
        "POLYMARKET_MM_BRIDGE_RECONCILE_EVERY": "100",
    })
    assert cfg.latency_ms == 203.0
    assert cfg.max_inventory_contracts == 5.0
    assert cfg.reconcile_interval_events == 100


def test_build_bridge_applies_measured_latency_constant() -> None:
    """The 2b fit lands in the bridge: POLYMARKET_MM_BRIDGE_LATENCY_MS → ConstantLatency."""
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = build_bridge(
        config=_config(),
        bridge_config=BridgeConfig(condition_id="0xcond", asset_id="t1", latency_ms=203.0),
        venue=venue,
        journal=journal,
    )
    assert isinstance(bridge.latency_model, ConstantLatency)
    assert bridge.latency_model.round_trip == 203.0


def test_bridge_fill_record_trade_size_is_none_on_live_path() -> None:
    """trade_size (backtest = aggressor total print size) is unknown from a venue fill, so the
    live path emits None rather than falsely equating it to our fill qty. Regression for the
    confirmed telemetry-parity bug."""
    venue, journal = _MockVenue(), _CapturingJournal()
    bridge = _bridge(venue=venue, journal=journal)
    bridge.on_market_event(_book_event(1000, 0.47, 0.49))
    buy_coid = next(c["client_order_id"] for c in venue.submit_calls if c["side"] == "BUY")
    bridge.ingest_fill(VenueFill(token_id="t1", side="BUY", price=0.47, qty=50.0, ts_ms=1100,
                                 client_order_id=buy_coid))
    rec = bridge.result().fills[0]
    assert rec["trade_size"] is None      # not 50.0 → never a comparable aggressor-size column
    assert rec["qty"] == 50.0             # our own fill size still carried here
