"""Tests for mirror/real_venue_adapter.py — kwargs ↔ kernel intent translation."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from polymarket.execution._kernel.polymarket_adapter import (
    VenueTimeoutError,
    VenueTransportError,
)
from polymarket.execution._kernel.state_machine import (
    Side,
    TimeInForce,
    VenueFillEvent,
    VenueOrderAck,
    VenueRejectEvent,
)
from polymarket.execution._kernel.venue import (
    SubmitOrderResult,
    SubmitOrderStatus,
    VenueOrderIntent,
)
from polymarket.execution.journal import (
    FillRecorded,
    JsonlWriter,
    OrderAcknowledged,
    OrderRejected,
    OrderSubmitted,
)
from polymarket.execution.mirror.real_venue_adapter import RealVenueAdapter


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockHttpClient:
    """Minimal stand-in: records set_tick_size and set_neg_risk calls
    so tests can assert ordering / values."""
    def __init__(self) -> None:
        self.tick_calls: list[tuple[str, float]] = []
        self.neg_risk_calls: list[tuple[str, bool]] = []
        self.all_setter_calls: list[tuple[str, str, Any]] = []  # (kind, token, value)

    def set_tick_size(self, token_id: str, tick_size: float) -> None:
        self.tick_calls.append((token_id, tick_size))
        self.all_setter_calls.append(("tick", token_id, tick_size))

    def set_neg_risk(self, token_id: str, is_neg_risk: bool) -> None:
        self.neg_risk_calls.append((token_id, is_neg_risk))
        self.all_setter_calls.append(("neg_risk", token_id, is_neg_risk))


class _MockKernel:
    """Configurable kernel-shaped mock for the wrapper."""

    def __init__(self) -> None:
        self.submit_calls: list[VenueOrderIntent] = []
        self.cancel_calls: list[dict[str, Any]] = []
        self.next_submit_result: SubmitOrderResult | None = None
        self.submit_exception: BaseException | None = None
        self.poll_queue: list[Any] = []
        self.poll_exception: BaseException | None = None

    def submit_order(self, intent: VenueOrderIntent) -> SubmitOrderResult:
        self.submit_calls.append(intent)
        if self.submit_exception is not None:
            raise self.submit_exception
        if self.next_submit_result is not None:
            return self.next_submit_result
        return SubmitOrderResult(
            status=SubmitOrderStatus.ACKNOWLEDGED,
            client_order_id=intent.client_order_id or "coid",
            venue_order_id="venue-1",
            events=tuple(),
            ambiguous=False,
        )

    def cancel_order(self, **kwargs: Any) -> Any:
        self.cancel_calls.append(kwargs)
        # Minimal cancel result; we don't exercise this much.
        from polymarket.execution._kernel.venue import (
            CancelOrderResult,
            CancelOrderStatus,
        )
        return CancelOrderResult(
            status=CancelOrderStatus.CANCELED,
            client_order_id=kwargs.get("client_order_id"),
            venue_order_id=kwargs.get("venue_order_id"),
            events=tuple(),
            ambiguous=False,
        )

    def poll_or_process_order_updates(self, raw=None) -> tuple[Any, ...]:  # noqa: ARG002
        if self.poll_exception is not None:
            exc = self.poll_exception
            self.poll_exception = None  # one-shot
            raise exc
        events = tuple(self.poll_queue)
        self.poll_queue = []
        return events

    def reconcile_open_orders(self, expected: set[str]) -> Any:  # noqa: ARG002
        from polymarket.execution._kernel.venue import ReconciliationResult
        return ReconciliationResult(tuple(), tuple(), tuple(), tuple())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockMarketMetadataCache:
    """Configurable mock for MarketMetadataCache.

    Default: returns is_neg_risk=False, tick_size=0.01 for any asset.
    Tests can override by populating ``self.responses[asset_id]``.
    Setting ``self.return_none = True`` simulates Gamma fetch failure.
    """

    def __init__(self) -> None:
        self.responses: dict[str, Any] = {}
        self.return_none: bool = False
        self.calls: list[tuple[str, str]] = []

    def get_by_asset(self, asset_id: str, condition_id: str):
        self.calls.append((asset_id, condition_id))
        if self.return_none:
            return None
        if asset_id in self.responses:
            return self.responses[asset_id]
        from datetime import datetime, timezone
        from polymarket.execution.mirror.market_metadata import MarketMetadata
        return MarketMetadata(
            condition_id=condition_id,
            is_neg_risk=False,
            tick_size=0.01,
            fetched_at_utc=datetime.now(timezone.utc),
        )


def _wrapper(
    tmp_path: Path,
    *,
    seed_events: list[Any] | None = None,
    today: Any = None,
    metadata: "_MockMarketMetadataCache | None" = None,
) -> tuple[RealVenueAdapter, _MockKernel, _MockHttpClient, JsonlWriter]:
    journal = JsonlWriter(tmp_path, "rva-test")
    if seed_events:
        for ev in seed_events:
            journal.write(ev)
    kernel = _MockKernel()
    http = _MockHttpClient()
    meta = metadata or _MockMarketMetadataCache()
    wrapper = RealVenueAdapter(
        kernel_adapter=kernel,  # type: ignore[arg-type]
        http_client=http,  # type: ignore[arg-type]
        journal=journal,
        clob_url="https://clob.polymarket.com",
        bot_proxy_wallet="0x" + "b" * 40,
        market_metadata=meta,  # type: ignore[arg-type]
        polling_interval_seconds=0.05,
        today_utc=today,
    )
    # Pre-seed metadata for the two asset_ids that tests use, so the
    # mock returns predictable values for the "penny" sub-cent market.
    from datetime import datetime, timezone
    from polymarket.execution.mirror.market_metadata import MarketMetadata
    if "asset-1" not in meta.responses:
        meta.responses["asset-1"] = MarketMetadata(
            condition_id="cond-1", is_neg_risk=False, tick_size=0.01,
            fetched_at_utc=datetime.now(timezone.utc),
        )
    if "penny" not in meta.responses:
        meta.responses["penny"] = MarketMetadata(
            condition_id="cond-penny", is_neg_risk=False, tick_size=0.001,
            fetched_at_utc=datetime.now(timezone.utc),
        )
    return wrapper, kernel, http, journal


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        client_order_id="coid-1",
        condition_id="cond-1",
        asset_id="asset-1",
        side="BUY",
        size_shares=5.0,
        price=0.42,
        order_type="FOK",
    )
    base.update(overrides)
    return base


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Translation: kwargs → VenueOrderIntent
# ---------------------------------------------------------------------------


def test_submit_order_calls_set_tick_size_first(tmp_path: Path) -> None:
    wrapper, _kernel, http, _journal = _wrapper(tmp_path)
    wrapper.submit_order(**_kwargs())
    # _wrapper pre-caches asset-1 → 0.01; submit_order should
    # propagate that to the http client BEFORE the kernel call.
    assert ("asset-1", 0.01) in http.tick_calls


def test_submit_order_constructs_intent_correctly(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    wrapper.submit_order(**_kwargs(
        client_order_id="coid-x", condition_id="cond-x",
        asset_id="asset-1", side="BUY",
        size_shares=11.11, price=0.42, order_type="FOK",
    ))
    intent = kernel.submit_calls[0]
    assert intent.market_id == "cond-x"
    assert intent.token_id == "asset-1"
    assert intent.side == Side.BUY
    assert intent.tif == TimeInForce.IOC
    # 11.11 × 10_000 = 111100
    assert intent.quantity == 111100
    # 0.42 / 0.01 = 42
    assert intent.limit_price_ticks == 42
    assert intent.client_order_id == "coid-x"
    assert intent.package_id == "coid-x"
    assert intent.leg_id == "leg-0"


def test_submit_order_side_sell_translates(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    wrapper.submit_order(**_kwargs(side="SELL"))
    assert kernel.submit_calls[0].side == Side.SELL


def test_submit_order_order_type_gtc_translates(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    wrapper.submit_order(**_kwargs(order_type="GTC"))
    assert kernel.submit_calls[0].tif == TimeInForce.GTC


def test_submit_order_quantity_below_resolution_rejected(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    # 0.00001 × 10_000 = 0 (rounded)
    result = wrapper.submit_order(**_kwargs(size_shares=0.00001))
    assert result.accepted is False
    assert "below resolution" in (result.message or "")
    assert kernel.submit_calls == []  # never reached the kernel


def test_submit_order_quantity_at_resolution_boundary_passes(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    # 0.0001 × 10_000 = 1, just above the floor
    result = wrapper.submit_order(**_kwargs(size_shares=0.0001))
    assert result.accepted is True
    assert kernel.submit_calls[0].quantity == 1


# ---------------------------------------------------------------------------
# Result mapping
# ---------------------------------------------------------------------------


def test_submit_acknowledged_to_accepted(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    kernel.next_submit_result = SubmitOrderResult(
        status=SubmitOrderStatus.ACKNOWLEDGED, client_order_id="c",
        venue_order_id="v-1", events=tuple(), ambiguous=False,
    )
    result = wrapper.submit_order(**_kwargs())
    assert result.accepted is True
    assert result.ambiguous is False
    assert result.venue_order_id == "v-1"
    # State map populated.
    assert wrapper._venue_to_coid["v-1"] == "coid-1"


def test_submit_rejected_to_not_accepted(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    kernel.next_submit_result = SubmitOrderResult(
        status=SubmitOrderStatus.REJECTED, client_order_id="c",
        venue_order_id=None, events=tuple(), ambiguous=False,
        message="invalid order",
    )
    result = wrapper.submit_order(**_kwargs())
    assert result.accepted is False
    assert result.ambiguous is False
    assert result.message == "invalid order"


def test_submit_ambiguous_to_ambiguous(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    kernel.next_submit_result = SubmitOrderResult(
        status=SubmitOrderStatus.AMBIGUOUS, client_order_id="c",
        venue_order_id=None, events=tuple(), ambiguous=True,
    )
    result = wrapper.submit_order(**_kwargs())
    assert result.accepted is True
    assert result.ambiguous is True


def test_submit_already_submitted_treated_as_ambiguous(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    kernel.next_submit_result = SubmitOrderResult(
        status=SubmitOrderStatus.ALREADY_SUBMITTED, client_order_id="c",
        venue_order_id="v-prev", events=tuple(), ambiguous=False,
    )
    result = wrapper.submit_order(**_kwargs())
    assert result.accepted is True
    assert result.ambiguous is True
    assert result.venue_order_id == "v-prev"


def test_submit_failed_retryable_treated_as_ambiguous(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    kernel.next_submit_result = SubmitOrderResult(
        status=SubmitOrderStatus.FAILED_RETRYABLE, client_order_id="c",
        venue_order_id=None, events=tuple(), ambiguous=True,
    )
    result = wrapper.submit_order(**_kwargs())
    assert result.accepted is True
    assert result.ambiguous is True


def test_submit_kernel_timeout_to_ambiguous(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    kernel.submit_exception = VenueTimeoutError("submit timeout")
    result = wrapper.submit_order(**_kwargs())
    assert result.accepted is True
    assert result.ambiguous is True
    assert "VenueTimeoutError" in (result.message or "")


def test_submit_kernel_transport_to_ambiguous(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    kernel.submit_exception = VenueTransportError("network bad")
    result = wrapper.submit_order(**_kwargs())
    assert result.accepted is True
    assert result.ambiguous is True


# ---------------------------------------------------------------------------
# State maps
# ---------------------------------------------------------------------------


def test_coid_to_fields_set_before_kernel_call(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    # Even if the kernel raises a non-ambiguous exception, we need
    # the mapping to be present for late fills. Use the timeout path.
    kernel.submit_exception = VenueTimeoutError("submit timeout")
    wrapper.submit_order(**_kwargs(client_order_id="coid-late"))
    assert wrapper._coid_to_fields["coid-late"] == ("cond-1", "asset-1", "BUY")


def test_rejected_does_not_populate_venue_to_coid(tmp_path: Path) -> None:
    wrapper, kernel, _http, _journal = _wrapper(tmp_path)
    kernel.next_submit_result = SubmitOrderResult(
        status=SubmitOrderStatus.REJECTED, client_order_id="c",
        venue_order_id=None, events=tuple(), ambiguous=False,
    )
    wrapper.submit_order(**_kwargs())
    assert wrapper._venue_to_coid == {}


# ---------------------------------------------------------------------------
# Polling thread
# ---------------------------------------------------------------------------


def test_polling_journals_fill(tmp_path: Path) -> None:
    wrapper, kernel, _http, journal = _wrapper(tmp_path)
    # Pre-register the order so the fill correlates.
    wrapper._coid_to_fields["coid-1"] = ("cond-1", "asset-1", "BUY")

    fill = VenueFillEvent(
        package_id="coid-1", leg_id="leg-0", client_order_id="coid-1",
        fill_qty=51850,            # 5.185 shares × 10_000
        fill_price_ticks=42,        # $0.42 at tick=0.01
        ts_ns=12345,
    )
    kernel.poll_queue.append(fill)

    wrapper.start()
    # Wait briefly for the polling thread to consume the event.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        events = list(journal.read_today())
        if any(e["event_type"] == "FILL_RECORDED" for e in events):
            break
        time.sleep(0.02)
    wrapper.stop()
    journal.close()

    fills = [e for e in journal.read_today() if e["event_type"] == "FILL_RECORDED"]
    assert len(fills) == 1
    fill_ev = fills[0]
    assert fill_ev["condition_id"] == "cond-1"
    assert fill_ev["asset_id"] == "asset-1"
    assert fill_ev["side"] == "BUY"
    assert fill_ev["size"] == pytest.approx(5.185, abs=1e-9)
    assert fill_ev["price"] == pytest.approx(0.42, abs=1e-9)
    assert fill_ev["proxy_wallet"] == "0x" + "b" * 40
    assert "coid-1:fill:" in fill_ev["transaction_hash"]


def test_polling_unknown_coid_does_not_crash(tmp_path: Path) -> None:
    wrapper, kernel, _http, journal = _wrapper(tmp_path)
    fill = VenueFillEvent(
        package_id="?", leg_id="?", client_order_id="never-seen",
        fill_qty=100, fill_price_ticks=42, ts_ns=1,
    )
    kernel.poll_queue.append(fill)
    wrapper.start()
    time.sleep(0.2)
    wrapper.stop()
    journal.close()
    fills = [e for e in journal.read_today() if e["event_type"] == "FILL_RECORDED"]
    assert fills == []


def test_polling_loop_survives_kernel_exception(tmp_path: Path) -> None:
    wrapper, kernel, _http, journal = _wrapper(tmp_path)
    kernel.poll_exception = RuntimeError("kernel boom")
    # Also queue a real fill to verify the loop kept going.
    wrapper._coid_to_fields["coid-2"] = ("cond-1", "asset-1", "SELL")

    wrapper.start()
    time.sleep(0.15)  # let the exception fire and the loop continue
    kernel.poll_queue.append(VenueFillEvent(
        package_id="coid-2", leg_id="leg-0", client_order_id="coid-2",
        fill_qty=10000, fill_price_ticks=50, ts_ns=42,
    ))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        events = list(journal.read_today())
        if any(e["event_type"] == "FILL_RECORDED" for e in events):
            break
        time.sleep(0.02)
    wrapper.stop()
    journal.close()
    fills = [e for e in journal.read_today() if e["event_type"] == "FILL_RECORDED"]
    assert len(fills) == 1


def test_polling_post_ack_reject_logs_only(tmp_path: Path) -> None:
    wrapper, kernel, _http, journal = _wrapper(tmp_path)
    kernel.poll_queue.append(VenueRejectEvent(
        package_id="x", leg_id="y", client_order_id="coid-r",
        reason="late_reject", ts_ns=1,
    ))
    wrapper.start()
    time.sleep(0.15)
    wrapper.stop()
    journal.close()
    fills = [e for e in journal.read_today() if e["event_type"] == "FILL_RECORDED"]
    assert fills == []  # no journal write for post-ack reject


# ---------------------------------------------------------------------------
# State rebuild from journal
# ---------------------------------------------------------------------------


def test_rebuild_recovers_open_order(tmp_path: Path) -> None:
    today = datetime.now(timezone.utc).date()
    seed = [
        OrderSubmitted(
            ts_utc=_now(), client_order_id="coid-A",
            condition_id="cond-A", asset_id="asset-A", side="BUY",
            size=10.0, price=0.5, order_type="FOK",
        ),
        OrderAcknowledged(
            ts_utc=_now(), client_order_id="coid-A", venue_order_id="v-A",
        ),
    ]
    wrapper, _kernel, _http, _journal = _wrapper(tmp_path, seed_events=seed, today=today)
    assert wrapper._coid_to_fields["coid-A"] == ("cond-A", "asset-A", "BUY")
    assert wrapper._venue_to_coid["v-A"] == "coid-A"


def test_rebuild_drops_filled_order(tmp_path: Path) -> None:
    today = datetime.now(timezone.utc).date()
    seed = [
        OrderSubmitted(
            ts_utc=_now(), client_order_id="coid-B",
            condition_id="cond-B", asset_id="asset-B", side="BUY",
            size=10.0, price=0.5, order_type="FOK",
        ),
        OrderAcknowledged(
            ts_utc=_now(), client_order_id="coid-B", venue_order_id="v-B",
        ),
        FillRecorded(
            ts_utc=_now(), transaction_hash="coid-B:fill:99",
            condition_id="cond-B", asset_id="asset-B", side="BUY",
            size=10.0, price=0.5, proxy_wallet="0xbot",
        ),
    ]
    wrapper, _kernel, _http, _journal = _wrapper(tmp_path, seed_events=seed, today=today)
    assert "coid-B" not in wrapper._coid_to_fields
    assert "v-B" not in wrapper._venue_to_coid


def test_rebuild_drops_rejected_order(tmp_path: Path) -> None:
    today = datetime.now(timezone.utc).date()
    seed = [
        OrderSubmitted(
            ts_utc=_now(), client_order_id="coid-C",
            condition_id="cond-C", asset_id="asset-C", side="BUY",
            size=10.0, price=0.5, order_type="FOK",
        ),
        OrderRejected(
            ts_utc=_now(), client_order_id="coid-C",
            reason="venue_rejected", detail="insufficient funds",
        ),
    ]
    wrapper, _kernel, _http, _journal = _wrapper(tmp_path, seed_events=seed, today=today)
    assert "coid-C" not in wrapper._coid_to_fields
    assert wrapper._venue_to_coid == {}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_is_real_venue_returns_true(tmp_path: Path) -> None:
    wrapper, _kernel, _http, _journal = _wrapper(tmp_path)
    assert wrapper.is_real_venue() is True


def test_start_is_idempotent(tmp_path: Path) -> None:
    wrapper, _kernel, _http, _journal = _wrapper(tmp_path)
    wrapper.start()
    first_thread = wrapper._polling_thread
    wrapper.start()
    assert wrapper._polling_thread is first_thread
    wrapper.stop()


def test_stop_joins_polling_thread(tmp_path: Path) -> None:
    wrapper, _kernel, _http, _journal = _wrapper(tmp_path)
    wrapper.start()
    assert wrapper._polling_thread is not None
    assert wrapper._polling_thread.is_alive()
    wrapper.stop()
    assert not wrapper._polling_thread.is_alive()


# ---------------------------------------------------------------------------
# NegRisk wiring
# ---------------------------------------------------------------------------


def test_submit_calls_both_setters_before_kernel(tmp_path: Path) -> None:
    """Both set_tick_size and set_neg_risk must be on the http client
    BEFORE the kernel.submit_order call, in either order."""
    wrapper, kernel, http, _journal = _wrapper(tmp_path)
    wrapper.submit_order(**_kwargs())
    kinds_in_order = [c[0] for c in http.all_setter_calls]
    assert "tick" in kinds_in_order
    assert "neg_risk" in kinds_in_order
    # Both setters must precede the kernel call. Recorded by call-order
    # on the mocks: setter calls happen inside submit_order, the kernel
    # call happens after; if we got here with submit_calls populated,
    # both setters fired first.
    assert len(kernel.submit_calls) == 1


def test_negrisk_market_sets_neg_risk_true(tmp_path: Path) -> None:
    meta = _MockMarketMetadataCache()
    from polymarket.execution.mirror.market_metadata import MarketMetadata
    meta.responses["asset-nr"] = MarketMetadata(
        condition_id="cond-nr",
        is_neg_risk=True,
        tick_size=0.01,
        fetched_at_utc=_now(),
    )
    wrapper, _kernel, http, _journal = _wrapper(tmp_path, metadata=meta)
    wrapper.submit_order(**_kwargs(asset_id="asset-nr", condition_id="cond-nr"))
    assert ("asset-nr", True) in http.neg_risk_calls
    assert ("asset-nr", 0.01) in http.tick_calls


def test_binary_market_sets_neg_risk_false(tmp_path: Path) -> None:
    wrapper, _kernel, http, _journal = _wrapper(tmp_path)
    wrapper.submit_order(**_kwargs())  # asset-1 default → binary
    assert ("asset-1", False) in http.neg_risk_calls


def test_condition_negrisk_cache_is_sticky(tmp_path: Path) -> None:
    meta = _MockMarketMetadataCache()
    from polymarket.execution.mirror.market_metadata import MarketMetadata
    meta.responses["asset-nr-1"] = MarketMetadata(
        condition_id="cond-nr",
        is_neg_risk=True,
        tick_size=0.01,
        fetched_at_utc=_now(),
    )
    meta.responses["asset-nr-2"] = MarketMetadata(
        condition_id="cond-nr",
        is_neg_risk=False,
        tick_size=0.01,
        fetched_at_utc=_now(),
    )
    wrapper, _kernel, http, _journal = _wrapper(tmp_path, metadata=meta)

    wrapper.submit_order(**_kwargs(asset_id="asset-nr-1", condition_id="cond-nr"))
    wrapper.submit_order(**_kwargs(asset_id="asset-nr-2", condition_id="cond-nr"))

    assert ("asset-nr-1", True) in http.neg_risk_calls
    assert ("asset-nr-2", True) in http.neg_risk_calls


def test_gamma_fetch_failure_refuses_to_submit(tmp_path: Path) -> None:
    meta = _MockMarketMetadataCache()
    meta.return_none = True
    wrapper, kernel, http, journal = _wrapper(tmp_path, metadata=meta)
    result = wrapper.submit_order(**_kwargs())
    assert result.accepted is False
    assert "cannot_classify_market" in (result.message or "")
    # No http client setter calls.
    assert http.tick_calls == []
    assert http.neg_risk_calls == []
    # No kernel call.
    assert kernel.submit_calls == []
    # OrderRejected event journaled with the right reason.
    journal.close()
    events = list(journal.read_today())
    rejects = [e for e in events if e["event_type"] == "ORDER_REJECTED"]
    assert len(rejects) == 1
    assert rejects[0]["reason"] == "cannot_classify_market"
