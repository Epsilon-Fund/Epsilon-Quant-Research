from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import JsonlWriter
from polymarket.execution.maker.event_calendar import EventCalendar, ScheduledEvent
from polymarket.execution.maker.maker_engine import (
    ClobBookClient,
    GammaMarketLookup,
    MakerEngine,
    MakerEngineConfig,
    MakerMarket,
    TopOfBook,
)
from polymarket.execution.mirror.mirror_engine import SubmitResult


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
        killswitch_path=Path("/tmp/killswitch"),
        journal_dir=Path("./journal_logs"),
        log_level="INFO",
        max_real_orders=5,
        require_operator_confirm=False,
    )
    for key, value in overrides.items():
        base = replace(base, **{key: value})
    return base


class _Venue:
    def __init__(self, *, real: bool = False) -> None:
        self.real = real
        self.submit_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []

    def is_real_venue(self) -> bool:
        return self.real

    def submit_order(self, **kwargs: Any) -> SubmitResult:
        self.submit_calls.append(kwargs)
        return SubmitResult(
            accepted=True,
            ambiguous=False,
            venue_order_id=f"venue-{kwargs['client_order_id']}",
        )

    def cancel_order(self, **kwargs: Any) -> dict[str, Any]:
        self.cancel_calls.append(kwargs)
        return {"ambiguous": False}


class _MarketLookup:
    def __init__(self, market: MakerMarket | None = None) -> None:
        self.market = market or MakerMarket("0xabc", "asset-yes", 0.01)

    def get_market(self, condition_id: str) -> MakerMarket | None:  # noqa: ARG002
        return self.market


class _BookClient:
    def __init__(self, books: list[TopOfBook]) -> None:
        self.books = books
        self.calls = 0

    def get_top_of_book(self, asset_id: str) -> TopOfBook | None:  # noqa: ARG002
        book = self.books[min(self.calls, len(self.books) - 1)]
        self.calls += 1
        return book


class _Inventory:
    def __init__(self, exposure: float = 0.0) -> None:
        self.exposure = exposure

    def get_basket_exposure(self, condition_id: str) -> float:  # noqa: ARG002
        return self.exposure


class _Resolution:
    def __init__(self, resolved: bool = False) -> None:
        self.resolved = resolved

    def is_resolved(self, condition_id: str) -> bool:  # noqa: ARG002
        return self.resolved


class _DataClient:
    def __init__(self, trades: list[dict[str, Any]] | None = None) -> None:
        self.trades = trades or []
        self.future_price: float | None = None

    def get_trades(self, condition_id: str) -> list[dict[str, Any]]:  # noqa: ARG002
        return list(self.trades)

    def price_after_60s(
        self, condition_id: str, asset_id: str, fill_ts: datetime
    ) -> float | None:  # noqa: ARG002
        return self.future_price


class _Response:
    def __init__(self, payload: Any) -> None:
        import json

        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _engine(
    tmp_path: Path,
    *,
    execution_config: ExecutionConfig | None = None,
    maker_config: MakerEngineConfig | None = None,
    venue: _Venue | None = None,
    market: MakerMarket | None = None,
    book: _BookClient | None = None,
    inventory: _Inventory | None = None,
    resolution: _Resolution | None = None,
    data_client: _DataClient | None = None,
    calendar: EventCalendar | None = None,
) -> tuple[MakerEngine, JsonlWriter, _Venue]:
    journal = JsonlWriter(tmp_path, "maker")
    v = venue or _Venue()
    engine = MakerEngine(
        execution_config=execution_config or _config(),
        maker_config=maker_config or MakerEngineConfig(condition_id="0xabc"),
        journal=journal,
        venue=v,
        inventory=inventory or _Inventory(),
        market_lookup=_MarketLookup(market),
        book_client=book or _BookClient([TopOfBook(0.40, 0.60)]),
        data_client=data_client or _DataClient(),
        event_calendar=calendar,
        resolution_state=resolution,
    )
    return engine, journal, v


def _events(journal: JsonlWriter) -> list[dict[str, Any]]:
    return list(journal.read_today(today_utc=datetime.now(timezone.utc).date()))


def test_gamma_lookup_uses_user_agent_request() -> None:
    seen: list[Any] = []

    def urlopen(request: Any, **_kwargs: Any) -> _Response:
        seen.append(request)
        return _Response([{
            "conditionId": "0xabc",
            "clobTokenIds": "[\"asset-yes\", \"asset-no\"]",
            "tick_size": "0.01",
        }])

    lookup = GammaMarketLookup(
        "https://gamma-api.polymarket.com",
        urlopen_fn=urlopen,
    )

    assert lookup.get_market("0xabc") == MakerMarket("0xabc", "asset-yes", 0.01)
    assert seen
    assert seen[0].headers["User-agent"] == "curl/8.0"


def test_clob_book_client_uses_best_prices_not_array_order() -> None:
    def urlopen(_request: Any, **_kwargs: Any) -> _Response:
        return _Response({
            "bids": [
                {"price": "0.01", "size": "10"},
                {"price": "0.73", "size": "1"},
            ],
            "asks": [
                {"price": "0.99", "size": "10"},
                {"price": "0.74", "size": "1"},
            ],
        })

    book = ClobBookClient(
        "https://clob.polymarket.com",
        urlopen_fn=urlopen,
    ).get_top_of_book("asset-yes")

    assert book == TopOfBook(best_bid=0.73, best_ask=0.74)


def test_places_two_sided_passive_quotes(tmp_path: Path) -> None:
    engine, journal, venue = _engine(tmp_path)
    engine.run_once()

    assert len(venue.submit_calls) == 2
    assert venue.submit_calls[0]["side"] == "BUY"
    assert venue.submit_calls[0]["price"] == 0.40
    assert venue.submit_calls[1]["side"] == "SELL"
    assert venue.submit_calls[1]["price"] == 0.60
    placed = [e for e in _events(journal) if e["event_type"] == "MAKER_QUOTE_PLACED"]
    assert len(placed) == 2
    assert {e["order_type"] for e in placed} == {"GTC"}


def test_real_observe_only_max_real_orders_zero_skips_without_submit(
    tmp_path: Path,
) -> None:
    engine, journal, venue = _engine(
        tmp_path,
        execution_config=_config(max_real_orders=0),
        venue=_Venue(real=True),
    )
    engine.run_once()
    engine.run_once()

    assert venue.submit_calls == []
    events = _events(journal)
    assert any(
        e["event_type"] == "RISK_HALT" and e["reason"] == "max_real_orders"
        for e in events
    )
    skips = [
        e for e in events
        if e["event_type"] == "MAKER_QUOTE_SKIPPED"
        and e["reason"] == "max_real_orders"
    ]
    assert len(skips) >= 2


def test_disallowed_tick_size_skips_both_sides_without_submit(
    tmp_path: Path,
) -> None:
    engine, journal, venue = _engine(
        tmp_path,
        market=MakerMarket("0xabc", "asset-yes", 0.001),
    )
    engine.run_once()

    assert venue.submit_calls == []
    skips = [
        e for e in _events(journal)
        if e["event_type"] == "MAKER_QUOTE_SKIPPED"
    ]
    tick_skips = [e for e in skips if e["reason"] == "tick_size_not_allowed"]
    assert len(tick_skips) == 2
    assert {e["side"] for e in tick_skips} == {"BUY", "SELL"}
    assert all("tick_size=0.001" in e["detail"] for e in tick_skips)


def test_refresh_cancels_and_replaces_when_book_moves_more_than_tick(
    tmp_path: Path,
) -> None:
    book = _BookClient([TopOfBook(0.40, 0.60), TopOfBook(0.42, 0.62)])
    engine, journal, venue = _engine(tmp_path, book=book)

    engine.run_once()
    engine.run_once()

    assert len(venue.cancel_calls) == 2
    assert len(venue.submit_calls) == 4
    cancels = [
        e for e in _events(journal)
        if e["event_type"] == "MAKER_QUOTE_CANCELED"
    ]
    assert len(cancels) == 2
    assert {e["reason"] for e in cancels} == {"price_moved"}


def test_resolution_cancels_open_quotes(tmp_path: Path) -> None:
    resolution = _Resolution(False)
    engine, _journal, venue = _engine(tmp_path, resolution=resolution)

    engine.run_once()
    resolution.resolved = True
    engine.run_once()

    assert len(venue.cancel_calls) == 2
    assert len(venue.submit_calls) == 2


def test_basket_exposure_cap_blocks_new_bid_but_leaves_ask(
    tmp_path: Path,
) -> None:
    engine, journal, venue = _engine(tmp_path, inventory=_Inventory(exposure=10.0))
    engine.run_once()

    assert len(venue.submit_calls) == 1
    assert venue.submit_calls[0]["side"] == "SELL"
    skips = [e for e in _events(journal) if e["event_type"] == "MAKER_QUOTE_SKIPPED"]
    assert skips[0]["reason"] == "basket_exposure_cap"


def test_fill_telemetry_logs_required_fields(tmp_path: Path) -> None:
    fill_ts = datetime.now(timezone.utc).replace(microsecond=0)
    # Start with no funder fill in the trade history so run_once places quotes
    # cleanly; the funder fill arrives afterwards and is matched by price.
    data = _DataClient([
        {
            "transactionHash": "0xother",
            "conditionId": "0xabc",
            "asset": "asset-yes",
            "proxyWallet": "0x" + "9" * 40,
            "price": "0.40",
            "timestamp": int(fill_ts.timestamp()),
        },
        {
            "transactionHash": "0xfuture",
            "conditionId": "0xabc",
            "asset": "asset-yes",
            "proxyWallet": "0x" + "8" * 40,
            "price": "0.43",
            "timestamp": int((fill_ts + timedelta(seconds=60)).timestamp()),
        },
    ])
    data.future_price = 0.43
    calendar = EventCalendar([
        ScheduledEvent(name="event", timestamp_utc=fill_ts, category="court")
    ])
    engine, journal, _venue = _engine(
        tmp_path, data_client=data, calendar=calendar,
    )
    engine._session_start_utc = fill_ts - timedelta(seconds=1)
    engine.run_once()

    # Our fill now prints in the Data API at the resting BUY price (0.40).
    data.trades.append({
        "transactionHash": "0xours",
        "conditionId": "0xabc",
        "asset": "asset-yes",
        "proxyWallet": "0x" + "2" * 40,
        "side": "BUY",
        "size": "1.0",
        "price": "0.40",
        "timestamp": int((fill_ts + timedelta(seconds=1)).timestamp()),
    })
    market = MakerMarket("0xabc", "asset-yes", 0.01)
    engine.process_fills_once(market)

    telemetry = [
        e for e in _events(journal)
        if e["event_type"] == "MAKER_FILL_TELEMETRY"
    ]
    assert len(telemetry) == 1
    row = telemetry[0]
    assert row["top_maker_rank_at_fill"] == 2
    assert row["post_fill_price_drift_60s"] == pytest.approx(0.03)
    assert row["news_proximate"] is True
    assert row["fill_share_this_market"] == pytest.approx(1 / 3)
    # The matched BUY quote is removed; the SELL quote remains live.
    assert {q.side for q in engine._quotes.values()} == {"SELL"}


def test_unmatched_funder_fill_logged_as_taker_with_no_rank(tmp_path: Path) -> None:
    fill_ts = datetime.now(timezone.utc).replace(microsecond=0)
    data = _DataClient([])
    engine, journal, _venue = _engine(tmp_path, data_client=data)
    engine._session_start_utc = fill_ts - timedelta(seconds=1)
    engine.run_once()  # places BUY @ 0.40, SELL @ 0.60

    # A funder fill at a price that matches no live quote (e.g. a taker fill on
    # a resting order that already left our quote state) is still ours.
    data.trades.append({
        "transactionHash": "0xtaker",
        "conditionId": "0xabc",
        "asset": "asset-yes",
        "proxyWallet": "0x" + "2" * 40,
        "side": "BUY",
        "size": "1.0",
        "price": "0.50",
        "timestamp": int((fill_ts + timedelta(seconds=1)).timestamp()),
    })
    engine.process_fills_once(MakerMarket("0xabc", "asset-yes", 0.01))

    telemetry = [
        e for e in _events(journal)
        if e["event_type"] == "MAKER_FILL_TELEMETRY"
    ]
    assert len(telemetry) == 1
    assert telemetry[0]["top_maker_rank_at_fill"] is None
    # No quote matched the price, so both quotes remain live.
    assert {q.side for q in engine._quotes.values()} == {"BUY", "SELL"}


def test_missed_fill_logged_when_market_trades_at_our_price(
    tmp_path: Path,
) -> None:
    trade_ts = datetime.now(timezone.utc).replace(microsecond=0)
    data = _DataClient([
        {
            "transactionHash": "0xmiss",
            "conditionId": "0xabc",
            "asset": "asset-yes",
            "proxyWallet": "0x" + "9" * 40,
            "price": "0.40",
            "timestamp": int(trade_ts.timestamp()),
        }
    ])
    engine, journal, _venue = _engine(tmp_path, data_client=data)
    engine._session_start_utc = trade_ts - timedelta(seconds=1)
    engine.run_once()
    engine.detect_missed_fills_once(MakerMarket("0xabc", "asset-yes", 0.01))

    missed = [
        e for e in _events(journal)
        if e["event_type"] == "MAKER_MISSED_FILL"
    ]
    assert len(missed) == 1
    assert missed[0]["reason"] == "market_traded_at_our_price_without_fill"
