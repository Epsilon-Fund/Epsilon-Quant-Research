"""Minimal passive maker measurement scaffold for one NegRisk market."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import (
    JsonlWriter,
    MakerFillTelemetry,
    MakerMissedFill,
    MakerQuoteCanceled,
    MakerQuotePlaced,
    MakerQuoteSkipped,
    OrderAcknowledged,
    OrderRejected,
    OrderSubmitted,
    RiskHalt,
)
from polymarket.execution.mirror.mirror_engine import SubmitResult

from .event_calendar import EventCalendar

_STOP_JOIN_TIMEOUT_S = 5.0
_HTTP_USER_AGENT = "curl/8.0"


class MakerVenue(Protocol):
    def submit_order(
        self,
        *,
        client_order_id: str,
        condition_id: str,
        asset_id: str,
        side: str,
        size_shares: float,
        price: float,
        order_type: str,
    ) -> SubmitResult: ...

    def cancel_order(
        self,
        *,
        client_order_id: str | None = None,
        venue_order_id: str | None = None,
    ) -> dict[str, Any]: ...


class BasketInventory(Protocol):
    def get_basket_exposure(self, condition_id: str) -> float: ...


class ResolutionState(Protocol):
    def is_resolved(self, condition_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class MakerEngineConfig:
    condition_id: str
    size_contracts: float = 1.0
    refresh_interval_seconds: float = 30.0
    order_type: str = "GTC"
    basket_exposure_cap: float = 10.0
    allowed_tick_sizes: frozenset[float] = frozenset((0.01,))

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> MakerEngineConfig:
        source = os.environ if env is None else env
        condition_id = source.get("POLYMARKET_MAKER_CONDITION_ID", "").strip()
        if not condition_id:
            raise ValueError("POLYMARKET_MAKER_CONDITION_ID is required")
        return cls(
            condition_id=condition_id.lower(),
            size_contracts=float(source.get("MAKER_SIZE_CONTRACTS", "1")),
            refresh_interval_seconds=float(
                source.get("POLYMARKET_MAKER_REFRESH_SECONDS", "30")
            ),
            order_type=source.get("POLYMARKET_MAKER_ORDER_TYPE", "GTC").upper(),
            basket_exposure_cap=float(
                source.get("POLYMARKET_MAKER_BASKET_EXPOSURE_CAP", "10")
            ),
            allowed_tick_sizes=_parse_allowed_tick_sizes(
                source.get("POLYMARKET_MAKER_ALLOWED_TICKS", "0.01")
            ),
        )


@dataclass(frozen=True, slots=True)
class MakerMarket:
    condition_id: str
    asset_id: str
    tick_size: float


@dataclass(frozen=True, slots=True)
class TopOfBook:
    best_bid: float
    best_ask: float


@dataclass(frozen=True, slots=True)
class QuoteState:
    client_order_id: str
    venue_order_id: str | None
    condition_id: str
    asset_id: str
    side: str
    price: float
    size: float


class GammaMarketLookup:
    """Fetches the YES token and tick size for a condition_id."""

    def __init__(
        self,
        gamma_url: str,
        *,
        urlopen_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._gamma_url = gamma_url.rstrip("/")
        self._urlopen = urlopen_fn if urlopen_fn is not None else urllib.request.urlopen
        self._cache: dict[str, MakerMarket] = {}

    def get_market(self, condition_id: str) -> MakerMarket | None:
        condition_id = condition_id.lower()
        cached = self._cache.get(condition_id)
        if cached is not None:
            return cached
        qs = urllib.parse.urlencode({"condition_ids": condition_id})
        url = f"{self._gamma_url}/markets?{qs}"
        try:
            with self._urlopen(_http_request(url), timeout=5.0) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            TimeoutError,
            OSError,
        ):
            return None
        rows = payload if isinstance(payload, list) else []
        if not rows or not isinstance(rows[0], dict):
            return None
        token_ids = _parse_clob_token_ids(rows[0].get("clobTokenIds"))
        if not token_ids:
            return None
        tick_size = _parse_tick_size(rows[0])
        market = MakerMarket(
            condition_id=condition_id,
            asset_id=str(token_ids[0]),
            tick_size=tick_size,
        )
        self._cache[condition_id] = market
        return market


class ClobBookClient:
    def __init__(
        self,
        clob_url: str,
        *,
        urlopen_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._clob_url = clob_url.rstrip("/")
        self._urlopen = urlopen_fn if urlopen_fn is not None else urllib.request.urlopen

    def get_top_of_book(self, asset_id: str) -> TopOfBook | None:
        qs = urllib.parse.urlencode({"token_id": asset_id})
        url = f"{self._clob_url}/book?{qs}"
        try:
            with self._urlopen(_http_request(url), timeout=5.0) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            TimeoutError,
            OSError,
        ):
            return None
        bids = _book_prices(payload.get("bids"))
        asks = _book_prices(payload.get("asks"))
        if not bids or not asks:
            return None
        return TopOfBook(best_bid=max(bids), best_ask=min(asks))


class DataApiTradeClient:
    def __init__(
        self,
        data_url: str,
        *,
        urlopen_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._data_url = data_url.rstrip("/")
        self._urlopen = urlopen_fn if urlopen_fn is not None else urllib.request.urlopen

    def get_trades(self, condition_id: str) -> list[dict[str, Any]]:
        qs = urllib.parse.urlencode({"condition_id": condition_id})
        url = f"{self._data_url}/trades?{qs}"
        try:
            with self._urlopen(_http_request(url), timeout=5.0) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            TimeoutError,
            OSError,
        ):
            return []
        return _extract_rows(payload)

    def price_after_60s(
        self, condition_id: str, asset_id: str, fill_ts: datetime
    ) -> float | None:
        """Price of the first trade at or after ``fill_ts + 60s``.

        Politics NegRisk markets are illiquid, so a symmetric ±10s window
        around the 60s mark produced too many ``None`` values. We instead
        take the earliest trade at or after the 60s mark with no upper bound,
        so a sparse market still yields a post-fill drift reference.
        """
        target = fill_ts + timedelta(seconds=60)
        best: tuple[datetime, float] | None = None
        for row in self.get_trades(condition_id):
            if str(row.get("asset") or row.get("assetId") or "") != asset_id:
                continue
            ts = _row_ts(row)
            if ts is None or ts < target:
                continue
            price = _float(row.get("price"), default=-1.0)
            if price < 0:
                continue
            if best is None or ts < best[0]:
                best = (ts, price)
        return best[1] if best is not None else None


class MakerEngine:
    """One-market passive quote loop for Phase-2 measurement."""

    def __init__(
        self,
        *,
        execution_config: ExecutionConfig,
        maker_config: MakerEngineConfig,
        journal: JsonlWriter,
        venue: MakerVenue,
        inventory: BasketInventory,
        market_lookup: GammaMarketLookup | None = None,
        book_client: ClobBookClient | None = None,
        data_client: DataApiTradeClient | None = None,
        event_calendar: EventCalendar | None = None,
        resolution_state: ResolutionState | None = None,
        today_utc: date | None = None,
    ) -> None:
        self._execution_config = execution_config
        self._config = maker_config
        self._journal = journal
        self._venue = venue
        self._inventory = inventory
        self._market_lookup = market_lookup or GammaMarketLookup(
            execution_config.gamma_url
        )
        self._book_client = book_client or ClobBookClient(execution_config.clob_url)
        self._data_client = data_client or DataApiTradeClient(execution_config.data_url)
        self._event_calendar = event_calendar
        self._resolution_state = resolution_state
        self._quotes: dict[str, QuoteState] = {}
        self._seen_fill_txs: set[str] = set()
        self._seen_missed_txs: set[str] = set()
        self._our_fill_count = 0
        self._session_start_utc = datetime.now(timezone.utc)
        self._today_utc = today_utc
        self._real_attempts = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def open_condition_ids(self) -> set[str]:
        if self._quotes:
            return {quote.condition_id for quote in self._quotes.values()}
        return {self._config.condition_id.lower()}

    def run_once(self) -> None:
        condition_id = self._config.condition_id.lower()
        if self._is_resolved(condition_id):
            self.cancel_all(reason="market_resolved")
            return

        market = self._market_lookup.get_market(condition_id)
        if market is None:
            self._skip(condition_id, "", "BUY", "market_lookup_failed")
            self._skip(condition_id, "", "SELL", "market_lookup_failed")
            return
        if not self._tick_size_allowed(market.tick_size):
            self.cancel_all(reason="tick_size_not_allowed")
            detail = (
                f"tick_size={market.tick_size:g} "
                f"allowed={sorted(self._config.allowed_tick_sizes)}"
            )
            self._skip(
                condition_id,
                market.asset_id,
                "BUY",
                "tick_size_not_allowed",
                detail=detail,
            )
            self._skip(
                condition_id,
                market.asset_id,
                "SELL",
                "tick_size_not_allowed",
                detail=detail,
            )
            return

        self.process_fills_once(market)
        self.detect_missed_fills_once(market)

        book = self._book_client.get_top_of_book(market.asset_id)
        if book is None:
            self._skip(condition_id, market.asset_id, "BUY", "no_book")
            self._skip(condition_id, market.asset_id, "SELL", "no_book")
            return

        exposure = self._inventory.get_basket_exposure(condition_id)
        desired: list[tuple[str, float]] = []
        if exposure < self._config.basket_exposure_cap:
            desired.append(("BUY", book.best_bid))
        else:
            self._skip(
                condition_id,
                market.asset_id,
                "BUY",
                "basket_exposure_cap",
                detail=(
                    f"exposure={exposure} cap={self._config.basket_exposure_cap}"
                ),
            )
        desired.append(("SELL", book.best_ask))

        for side, price in desired:
            self._refresh_side(market, side=side, price=price)

    def process_fills_once(self, market: MakerMarket) -> None:
        """Detect our own fills by polling the Data API trades endpoint.

        ``FillRecorded`` journal events carry no ``client_order_id`` (see
        journal/events.py), so they cannot be matched back to a live quote.
        Instead we poll ``data-api/trades`` for the market and pick out rows
        whose ``proxyWallet`` is our funder and whose timestamp is at or after
        the session start. A fill is matched to a resting quote when its price
        is within half a tick of that quote's price; matched quotes are popped
        from ``_quotes``. An unmatched funder fill (e.g. a taker fill on a
        resting order that already left our quote state) is still ours, so it
        is logged with ``top_maker_rank_at_fill=None`` and no quote to pop.
        """
        funder = self._execution_config.funder.lower()
        half_tick = market.tick_size / 2.0
        for row in self._data_client.get_trades(market.condition_id):
            tx = str(row.get("transactionHash") or row.get("transaction_hash") or "")
            if not tx or tx in self._seen_fill_txs:
                continue
            asset_id = str(row.get("asset") or row.get("assetId") or "")
            if asset_id != market.asset_id:
                continue
            wallet = str(row.get("proxyWallet") or row.get("proxy_wallet") or "")
            if wallet.lower() != funder:
                continue
            ts = _row_ts(row)
            if ts is not None and ts < self._session_start_utc:
                continue
            fill_price = _float(row.get("price"), default=-1.0)
            if fill_price < 0:
                continue
            matched_side: str | None = None
            for side, quote in self._quotes.items():
                if abs(quote.price - fill_price) <= half_tick:
                    matched_side = side
                    break
            self._seen_fill_txs.add(tx)
            if matched_side is not None:
                coid = self._quotes[matched_side].client_order_id
                self._log_fill_telemetry(row, coid=coid, matched=True)
                self._quotes.pop(matched_side, None)
            else:
                self._log_fill_telemetry(row, coid="", matched=False)

    def detect_missed_fills_once(self, market: MakerMarket) -> None:
        if not self._quotes:
            return
        trades = self._data_client.get_trades(market.condition_id)
        for row in trades:
            tx = str(row.get("transactionHash") or row.get("transaction_hash") or "")
            if not tx or tx in self._seen_missed_txs:
                continue
            ts = _row_ts(row)
            if ts is not None and ts < self._session_start_utc:
                continue
            price = _float(row.get("price"), default=-1.0)
            asset_id = str(row.get("asset") or row.get("assetId") or "")
            if asset_id != market.asset_id or price < 0:
                continue
            for quote in self._quotes.values():
                if abs(quote.price - price) > market.tick_size / 2:
                    continue
                wallet = str(row.get("proxyWallet") or row.get("proxy_wallet") or "")
                if wallet.lower() == self._execution_config.funder.lower():
                    continue
                self._seen_missed_txs.add(tx)
                news = self._news_proximate(ts or datetime.now(timezone.utc))
                self._journal.write(MakerMissedFill(
                    ts_utc=datetime.now(timezone.utc),
                    condition_id=market.condition_id,
                    asset_id=market.asset_id,
                    side=quote.side,
                    price=quote.price,
                    size=quote.size,
                    transaction_hash=tx,
                    news_proximate=news,
                    reason="market_traded_at_our_price_without_fill",
                ))
                break

    def cancel_all(self, *, reason: str) -> None:
        for side in list(self._quotes):
            self._cancel_side(side, reason=reason)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="maker_engine",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_JOIN_TIMEOUT_S)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            if self._stop_event.wait(self._config.refresh_interval_seconds):
                break

    def _refresh_side(self, market: MakerMarket, *, side: str, price: float) -> None:
        existing = self._quotes.get(side)
        if existing is not None and abs(existing.price - price) <= market.tick_size:
            return
        if existing is not None:
            self._cancel_side(side, reason="price_moved")
        self._place_quote(market, side=side, price=price)

    def _place_quote(self, market: MakerMarket, *, side: str, price: float) -> None:
        safety_block = self._safety_block_reason(market, side=side, price=price)
        if safety_block is not None:
            reason, detail = safety_block
            self._skip(
                market.condition_id,
                market.asset_id,
                side,
                reason,
                detail=detail,
            )
            return
        coid = _make_client_order_id(market.condition_id, side)
        size = float(self._config.size_contracts)
        self._journal.write(OrderSubmitted(
            ts_utc=datetime.now(timezone.utc),
            client_order_id=coid,
            condition_id=market.condition_id,
            asset_id=market.asset_id,
            side=side,
            size=size,
            price=price,
            order_type=self._config.order_type,
        ))
        try:
            result = self._venue.submit_order(
                client_order_id=coid,
                condition_id=market.condition_id,
                asset_id=market.asset_id,
                side=side,
                size_shares=size,
                price=price,
                order_type=self._config.order_type,
            )
        except Exception as exc:  # noqa: BLE001
            self._journal.write(OrderRejected(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                reason="maker_submit_exception",
                detail=f"{type(exc).__name__}: {exc}",
            ))
            return
        if result.ambiguous:
            self._journal.write(RiskHalt(
                ts_utc=datetime.now(timezone.utc),
                reason="ambiguous_submit",
                detail=result.message or "maker quote submit ambiguous",
            ))
            return
        if not result.accepted:
            self._journal.write(OrderRejected(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                reason="maker_venue_rejected",
                detail=result.message or "venue rejected maker quote",
            ))
            return
        if result.venue_order_id:
            self._journal.write(OrderAcknowledged(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                venue_order_id=result.venue_order_id,
            ))
        self._quotes[side] = QuoteState(
            client_order_id=coid,
            venue_order_id=result.venue_order_id,
            condition_id=market.condition_id,
            asset_id=market.asset_id,
            side=side,
            price=price,
            size=size,
        )
        self._journal.write(MakerQuotePlaced(
            ts_utc=datetime.now(timezone.utc),
            client_order_id=coid,
            condition_id=market.condition_id,
            asset_id=market.asset_id,
            side=side,
            size=size,
            price=price,
            order_type=self._config.order_type,
            venue_order_id=result.venue_order_id,
        ))

    def _cancel_side(self, side: str, *, reason: str) -> None:
        quote = self._quotes.pop(side, None)
        if quote is None:
            return
        ambiguous = False
        try:
            result = self._venue.cancel_order(
                client_order_id=quote.client_order_id,
                venue_order_id=quote.venue_order_id,
            )
            ambiguous = bool(result.get("ambiguous", False))
        except Exception as exc:  # noqa: BLE001
            ambiguous = True
            self._journal.write(OrderRejected(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=quote.client_order_id,
                reason="maker_cancel_exception",
                detail=f"{type(exc).__name__}: {exc}",
            ))
        self._journal.write(MakerQuoteCanceled(
            ts_utc=datetime.now(timezone.utc),
            client_order_id=quote.client_order_id,
            condition_id=quote.condition_id,
            asset_id=quote.asset_id,
            side=quote.side,
            price=quote.price,
            reason=reason,
            ambiguous=ambiguous,
        ))

    def _safety_block_reason(
        self, market: MakerMarket, *, side: str, price: float
    ) -> tuple[str, str] | None:
        if not _is_real_venue(self._venue):
            return None
        if self._real_attempts >= self._execution_config.max_real_orders:
            detail = (
                f"maker reached limit of "
                f"{self._execution_config.max_real_orders} real submits"
            )
            self._journal.write(RiskHalt(
                ts_utc=datetime.now(timezone.utc),
                reason="max_real_orders",
                detail=detail,
            ))
            return "max_real_orders", detail
        if self._execution_config.require_operator_confirm:
            if not self._prompt_operator(market, side=side, price=price):
                detail = "operator declined maker quote via stdin"
                self._journal.write(RiskHalt(
                    ts_utc=datetime.now(timezone.utc),
                    reason="operator_aborted",
                    detail=detail,
                ))
                return "operator_aborted", detail
        self._real_attempts += 1
        return None

    def _tick_size_allowed(self, tick_size: float) -> bool:
        return any(
            abs(tick_size - allowed) < 1e-12
            for allowed in self._config.allowed_tick_sizes
        )

    def _prompt_operator(self, market: MakerMarket, *, side: str, price: float) -> bool:
        print(
            f"\n[operator confirm] Maker quote:\n"
            f"  condition: {market.condition_id}\n"
            f"  asset    : {market.asset_id}\n"
            f"  side     : {side}\n"
            f"  size     : {self._config.size_contracts}\n"
            f"  price    : ${price:.4f}\n"
            f"Type 'yes' to proceed: ",
            end="", flush=True,
        )
        try:
            return sys.stdin.readline().strip().lower() == "yes"
        except (EOFError, KeyboardInterrupt):
            return False

    def _log_fill_telemetry(
        self, row: dict[str, Any], *, coid: str, matched: bool
    ) -> None:
        condition_id = str(
            row.get("conditionId")
            or row.get("condition_id")
            or self._config.condition_id
        ).lower()
        asset_id = str(row.get("asset") or row.get("assetId") or "")
        side = str(row.get("side", ""))
        price = _float(row.get("price"), default=0.0)
        size = _float(row.get("size"), default=0.0)
        fill_ts = _row_ts(row) or datetime.now(timezone.utc)
        rank = (
            self._top_maker_rank_at_fill(
                condition_id=condition_id,
                asset_id=asset_id,
                price=price,
                fill_ts=fill_ts,
            )
            if matched
            else None
        )
        future_price = self._data_client.price_after_60s(condition_id, asset_id, fill_ts)
        drift = None if future_price is None else future_price - price
        self._our_fill_count += 1
        total_fills = len([
            row for row in self._data_client.get_trades(condition_id)
            if _row_ts(row) is None or _row_ts(row) >= self._session_start_utc
        ])
        fill_share = (
            self._our_fill_count / total_fills
            if total_fills > 0 else None
        )
        self._journal.write(MakerFillTelemetry(
            ts_utc=datetime.now(timezone.utc),
            client_order_id=coid,
            condition_id=condition_id,
            asset_id=asset_id,
            side=side,
            size=size,
            price=price,
            top_maker_rank_at_fill=rank,
            post_fill_price_drift_60s=drift,
            news_proximate=self._news_proximate(fill_ts),
            fill_share_this_market=fill_share,
        ))

    def _top_maker_rank_at_fill(
        self,
        *,
        condition_id: str,
        asset_id: str,
        price: float,
        fill_ts: datetime,
    ) -> int | None:
        """Proxy for queue position at fill time — NOT a true rank.

        We order the funder's and others' fills that printed at the same
        price within a ±5s window around ``fill_ts`` and return the funder's
        1-indexed position in that ordering. Fill *order* in a 5s window is
        only a coarse stand-in for actual on-book queue position; treat the
        value as a directional proxy, not an exact maker rank.
        """
        rows = []
        for row in self._data_client.get_trades(condition_id):
            if str(row.get("asset") or row.get("assetId") or "") != asset_id:
                continue
            if abs(_float(row.get("price"), default=-1.0) - price) > 1e-9:
                continue
            ts = _row_ts(row)
            if ts is None or abs((ts - fill_ts).total_seconds()) > 5:
                continue
            rows.append((ts, row))
        rows.sort(key=lambda item: item[0])
        if not rows:
            return None
        funder = self._execution_config.funder.lower()
        for idx, (_, row) in enumerate(rows, start=1):
            wallet = str(row.get("proxyWallet") or row.get("proxy_wallet") or "")
            if wallet.lower() == funder:
                return idx
        return None

    def _news_proximate(self, ts: datetime) -> bool | None:
        if self._event_calendar is None:
            return None
        return self._event_calendar.is_event_proximate(ts, window_minutes=5)

    def _is_resolved(self, condition_id: str) -> bool:
        return (
            self._resolution_state is not None
            and self._resolution_state.is_resolved(condition_id)
        )

    def _skip(
        self,
        condition_id: str,
        asset_id: str,
        side: str,
        reason: str,
        *,
        detail: str = "",
    ) -> None:
        self._journal.write(MakerQuoteSkipped(
            ts_utc=datetime.now(timezone.utc),
            condition_id=condition_id,
            asset_id=asset_id,
            side=side,
            reason=reason,
            detail=detail or reason,
        ))


def _make_client_order_id(condition_id: str, side: str) -> str:
    raw = f"mm:{condition_id}:{side}:{time.time_ns()}".encode("utf-8")
    return f"mm-{hashlib.blake2b(raw, digest_size=8).hexdigest()}"


def _is_real_venue(venue: Any) -> bool:
    fn = getattr(venue, "is_real_venue", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _parse_clob_token_ids(raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [raw]
    else:
        parsed = raw
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
        return [str(item) for item in parsed if str(item)]
    return []


def _parse_tick_size(row: dict[str, Any]) -> float:
    for key in ("tick_size", "tickSize"):
        raw = row.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.01


def _parse_allowed_tick_sizes(raw: str | None) -> frozenset[float]:
    raw_value = "0.01" if raw is None or raw.strip() == "" else raw
    values: list[float] = []
    for part in raw_value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            raise ValueError(
                "POLYMARKET_MAKER_ALLOWED_TICKS must be comma-separated floats "
                f"(got {raw_value!r})"
            ) from None
        if value <= 0:
            raise ValueError(
                "POLYMARKET_MAKER_ALLOWED_TICKS values must be > 0 "
                f"(got {value})"
            )
        values.append(value)
    if not values:
        raise ValueError("POLYMARKET_MAKER_ALLOWED_TICKS must include at least one tick")
    return frozenset(values)


def _http_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": _HTTP_USER_AGENT})


def _book_prices(rows: Any) -> list[float]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    prices: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            price = float(row.get("price"))
        except (TypeError, ValueError):
            continue
        prices.append(price)
    return prices


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "trades", "items", "results"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _row_ts(row: dict[str, Any]) -> datetime | None:
    raw = row.get("timestamp") or row.get("ts_utc")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value > 10_000_000_000:
            value = value / 1000.0
        return datetime.fromtimestamp(value, timezone.utc)
    if isinstance(raw, str):
        try:
            if raw.isdigit():
                return datetime.fromtimestamp(float(raw), timezone.utc)
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
