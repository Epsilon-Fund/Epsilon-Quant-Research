"""Real-venue wrapper around the kernel's PolymarketVenueAdapter.

Architecture (Strategy A from REAL_VENUE_PLAN.md): mirror_engine talks
to the bot's local kwargs-style :class:`VenueAdapter` Protocol. The
kernel adapter speaks an intent-style API with int-quantised shares
and tick-quantised prices, and returns acknowledgement-only — fills
arrive asynchronously via ``poll_or_process_order_updates``. This
wrapper translates between the two and runs a background polling
thread that journals fills as they arrive.

Key design points:

  * ``set_tick_size(asset_id, tick_size)`` is called on the substitute
    HTTP client (see :mod:`polymarket.execution.mirror.clob_http_client`)
    BEFORE every kernel submit. This is mandatory — the substitute
    falls back to ``default_tick_size`` only as a last-resort guard
    against missing tick_size lookups; relying on the default
    silently mis-prices sub-penny markets.
  * Fractional shares are encoded into the kernel's ``int`` quantity
    field via ``quantity_int = int(round(size_shares * quantity_scale))``.
    The substitute HTTP client decodes back to a decimal string for
    py-clob-client. Both sides must agree on the scale; this wrapper
    holds it as a constructor argument with the same default
    (10_000) as :class:`ClobHttpClient`.
  * Polling thread reads the kernel's event stream every
    ``polling_interval_seconds``. Fills (``VenueFillEvent``) are
    journaled as :class:`FillRecorded`; other events are logged to
    stderr but not journaled.
  * State maps (``_coid_to_fields``, ``_venue_to_coid``) are
    journal-backed: rebuilt from today's + yesterday's journal on
    construction. Recovery is best-effort — orders submitted before
    that window are not tracked, and fills that arrive for them are
    logged but not journaled.

The transaction_hash field on :class:`FillRecorded` is a synthetic
placeholder (``f"{client_order_id}:fill:{ts_ns}"``) until the kernel
exposes the on-chain tx hash on ``VenueFillEvent``.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any

from polymarket.execution._kernel.polymarket_adapter import (
    PolymarketVenueAdapter,
    VenueTimeoutError,
    VenueTransportError,
)
from polymarket.execution._kernel.state_machine import (
    Side,
    TimeInForce,
    VenueCancelEvent,
    VenueFillEvent,
    VenueOrderAck,
    VenueRejectEvent,
)
from polymarket.execution._kernel.venue import (
    SubmitOrderResult,
    SubmitOrderStatus,
    VenueOrderIntent,
)
from polymarket.execution.journal import FillRecorded, JsonlWriter
from polymarket.execution.mirror.clob_http_client import ClobHttpClient
from polymarket.execution.mirror.mirror_engine import SubmitResult


_TICK_FETCH_TIMEOUT_S = 5.0
_POLLING_TIMEOUT_MS = 700
_POLLING_LIMIT = 200
_STOP_JOIN_TIMEOUT_S = 5.0


class RealVenueAdapter:
    """Bot-side wrapper conforming to mirror_engine's VenueAdapter Protocol.

    Wraps the kernel adapter for HTTP-mediated Polymarket order flow.
    """

    def __init__(
        self,
        kernel_adapter: PolymarketVenueAdapter,
        http_client: ClobHttpClient,
        journal: JsonlWriter,
        clob_url: str,
        bot_proxy_wallet: str,
        *,
        polling_interval_seconds: float = 0.7,
        tick_size_default: float = 0.01,
        quantity_scale: int = 10_000,
        today_utc: date | None = None,
    ) -> None:
        self._kernel: PolymarketVenueAdapter = kernel_adapter
        self._http_client: ClobHttpClient = http_client
        self._journal: JsonlWriter = journal
        self._clob_url: str = clob_url.rstrip("/")
        self._bot_proxy_wallet: str = bot_proxy_wallet
        self._polling_interval: float = polling_interval_seconds
        self._tick_default: float = tick_size_default
        self._quantity_scale: int = quantity_scale

        # Journal-backed state.
        self._coid_to_fields: dict[str, tuple[str, str, str]] = {}
        self._venue_to_coid: dict[str, str] = {}
        self._tick_cache: dict[str, float] = {}

        # Polling thread.
        self._stop_event: threading.Event = threading.Event()
        self._polling_thread: threading.Thread | None = None

        self._rebuild_from_journal(today_utc)

    # ------------------------------------------------------------------
    # Public Protocol surface
    # ------------------------------------------------------------------

    def is_real_venue(self) -> bool:
        return True

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
    ) -> SubmitResult:
        # 1. Tick size — mandatory; populate the HTTP client side channel.
        tick_size = self._get_tick_size(asset_id)
        self._http_client.set_tick_size(asset_id, tick_size)

        # 2. Encode for the kernel.
        quantity_int = int(round(size_shares * self._quantity_scale))
        if quantity_int < 1:
            return SubmitResult(
                accepted=False,
                ambiguous=False,
                message=(
                    f"size_shares={size_shares} encodes to quantity_int<1 "
                    f"(below resolution at scale={self._quantity_scale})"
                ),
            )
        price_ticks = int(round(price / tick_size)) if tick_size > 0 else 0
        if price_ticks <= 0:
            return SubmitResult(
                accepted=False,
                ambiguous=False,
                message=(
                    f"price={price} encodes to price_ticks<=0 "
                    f"(tick_size={tick_size})"
                ),
            )

        # 3. Translate side / TIF.
        try:
            kernel_side = Side(side.upper())
        except ValueError:
            return SubmitResult(
                accepted=False, ambiguous=False,
                message=f"unknown side: {side!r}",
            )
        # Polymarket exposes IOC and GTC; "FOK" maps to IOC + immediate
        # expiry per the closest-behavioural reduction (see PLAN.md
        # decision 7 follow-up in REAL_VENUE_PLAN.md §7).
        if order_type.upper() in ("FOK", "IOC"):
            tif = TimeInForce.IOC
        elif order_type.upper() == "GTC":
            tif = TimeInForce.GTC
        else:
            return SubmitResult(
                accepted=False, ambiguous=False,
                message=f"unsupported order_type: {order_type!r}",
            )

        ts_ns = time.monotonic_ns()
        intent = VenueOrderIntent(
            package_id=client_order_id,  # 1:1 for single-leg copy trades
            leg_id="leg-0",
            market_id=condition_id,
            token_id=asset_id,
            side=kernel_side,
            quantity=quantity_int,
            limit_price_ticks=price_ticks,
            tif=tif,
            ts_ns=ts_ns,
            expires_at_ns=ts_ns + 1,  # IOC: expire immediately if not filled
            client_order_id=client_order_id,
        )

        # 4. Track BEFORE the call so polling can correlate even if
        # we lose the result (e.g. raised exception).
        self._coid_to_fields[client_order_id] = (condition_id, asset_id, side)

        # 5. Call kernel.
        try:
            result: SubmitOrderResult = self._kernel.submit_order(intent)
        except (VenueTimeoutError, VenueTransportError) as exc:
            return SubmitResult(
                accepted=True,
                ambiguous=True,
                message=f"{type(exc).__name__}: {exc}",
            )

        # 6. Map kernel status → bot's SubmitResult.
        if result.status == SubmitOrderStatus.ACKNOWLEDGED:
            if result.venue_order_id:
                self._venue_to_coid[result.venue_order_id] = client_order_id
            return SubmitResult(
                accepted=True,
                ambiguous=result.ambiguous,
                venue_order_id=result.venue_order_id,
                message=result.message,
            )
        if result.status == SubmitOrderStatus.ALREADY_SUBMITTED:
            if result.venue_order_id:
                self._venue_to_coid[result.venue_order_id] = client_order_id
            return SubmitResult(
                accepted=True,
                ambiguous=True,
                venue_order_id=result.venue_order_id,
                message=result.message or "already_submitted",
            )
        if result.status == SubmitOrderStatus.REJECTED:
            return SubmitResult(
                accepted=False,
                ambiguous=False,
                venue_order_id=result.venue_order_id,
                message=result.message,
            )
        # AMBIGUOUS, FAILED_RETRYABLE, anything else → treat as ambiguous.
        return SubmitResult(
            accepted=True,
            ambiguous=True,
            venue_order_id=result.venue_order_id,
            message=result.message,
        )

    def cancel_order(
        self,
        *,
        client_order_id: str | None = None,
        venue_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Delegate to kernel.cancel_order. Returns a dict snapshot.

        mirror_engine doesn't currently cancel anything; this is here
        for completeness and to keep the wrapper a strict superset of
        the bot's protocol.
        """
        result = self._kernel.cancel_order(
            client_order_id=client_order_id,
            venue_order_id=venue_order_id,
            now_ns=time.monotonic_ns(),
        )
        return {
            "status": result.status.value,
            "client_order_id": result.client_order_id,
            "venue_order_id": result.venue_order_id,
            "ambiguous": result.ambiguous,
            "message": result.message,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._polling_thread is not None and self._polling_thread.is_alive():
            return
        self._stop_event.clear()
        self._polling_thread = threading.Thread(
            target=self._polling_loop,
            name="real_venue_poller",
            daemon=True,
        )
        self._polling_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._polling_thread is not None:
            self._polling_thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
            if self._polling_thread.is_alive():
                print(
                    "[real_venue] WARNING: polling thread did not stop in "
                    f"{_STOP_JOIN_TIMEOUT_S}s",
                    file=sys.stderr, flush=True,
                )

    # ------------------------------------------------------------------
    # Internals: polling thread + event handling
    # ------------------------------------------------------------------

    def _polling_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                events = self._kernel.poll_or_process_order_updates(None)
            except Exception as exc:  # noqa: BLE001 — defensive: thread must not die
                print(
                    f"[real_venue] poll error: {type(exc).__name__}: {exc}",
                    file=sys.stderr, flush=True,
                )
                if self._stop_event.wait(self._polling_interval):
                    break
                continue
            for event in events:
                try:
                    self._handle_kernel_event(event)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[real_venue] event handler error: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr, flush=True,
                    )
            if self._stop_event.wait(self._polling_interval):
                break

    def _handle_kernel_event(self, event: Any) -> None:
        if isinstance(event, VenueFillEvent):
            self._handle_fill(event)
        elif isinstance(event, VenueOrderAck):
            # Acks for orders we already saw via submit_order's return.
            # If a venue_order_id arrives here that we haven't seen, map it.
            if event.venue_order_id and event.venue_order_id not in self._venue_to_coid:
                self._venue_to_coid[event.venue_order_id] = event.client_order_id
        elif isinstance(event, VenueRejectEvent):
            print(
                f"[real_venue] post-ack reject coid={event.client_order_id} "
                f"reason={event.reason}",
                file=sys.stderr, flush=True,
            )
        elif isinstance(event, VenueCancelEvent):
            print(
                f"[real_venue] cancel ack coid={event.client_order_id} "
                f"qty={event.canceled_qty}",
                file=sys.stderr, flush=True,
            )

    def _handle_fill(self, event: VenueFillEvent) -> None:
        coid = event.client_order_id
        fields = self._coid_to_fields.get(coid)
        if fields is None:
            print(
                f"[real_venue] fill for unknown coid={coid} — skipped",
                file=sys.stderr, flush=True,
            )
            return
        condition_id, asset_id, side = fields

        # Decode shares from the kernel's int representation.
        shares = event.fill_qty / self._quantity_scale
        # Decode dollar price from ticks * tick_size. Cache was populated
        # at submit time; fall back to default if missing.
        tick_size = self._tick_cache.get(asset_id, self._tick_default)
        price = event.fill_price_ticks * tick_size

        # Synthetic transaction_hash placeholder until kernel exposes
        # the on-chain tx hash. Made unique per fill via ts_ns so
        # multiple fills on the same coid don't collide.
        synthetic_tx = f"{coid}:fill:{event.ts_ns}"

        self._journal.write(FillRecorded(
            ts_utc=datetime.now(timezone.utc),
            transaction_hash=synthetic_tx,
            condition_id=condition_id,
            asset_id=asset_id,
            side=side,
            size=shares,
            price=price,
            proxy_wallet=self._bot_proxy_wallet,
        ))

    # ------------------------------------------------------------------
    # Internals: state rebuild + tick lookup
    # ------------------------------------------------------------------

    def _rebuild_from_journal(self, today_utc: date | None) -> None:
        """Recover in-flight order state from today's + yesterday's journal.

        After restart the wrapper recovers state for orders that were
        submitted but not yet finalised (filled or rejected). Orders
        outside the today+yesterday window are not recovered; if a
        late fill arrives for one, it's logged but no FillRecorded
        is journaled.
        """
        today = today_utc if today_utc is not None else datetime.now(timezone.utc).date()
        for day in (today - timedelta(days=1), today):
            for event in self._journal.read_today(today_utc=day):
                self._absorb_journal_event(event)

    def _absorb_journal_event(self, event: dict[str, Any]) -> None:
        et = event.get("event_type")
        if et == "ORDER_SUBMITTED":
            coid = event.get("client_order_id")
            cond = event.get("condition_id")
            asset = event.get("asset_id")
            side = event.get("side")
            if all(isinstance(x, str) for x in (coid, cond, asset, side)):
                self._coid_to_fields[coid] = (cond, asset, side)  # type: ignore[index]
        elif et == "ORDER_ACKNOWLEDGED":
            coid = event.get("client_order_id")
            voi = event.get("venue_order_id")
            if isinstance(coid, str) and isinstance(voi, str) and voi:
                self._venue_to_coid[voi] = coid
        elif et == "FILL_RECORDED":
            # Order is considered closed for tracking purposes once a
            # fill is journaled; remove from both maps.
            # FillRecorded uses transaction_hash, not client_order_id.
            # Synthetic tx hashes contain the coid prefix; parse it back.
            tx = event.get("transaction_hash")
            if isinstance(tx, str) and ":fill:" in tx:
                coid = tx.split(":fill:", 1)[0]
                self._coid_to_fields.pop(coid, None)
                # Also drop the venue_order_id back-pointer, if any.
                stale = [v for v, c in self._venue_to_coid.items() if c == coid]
                for v in stale:
                    self._venue_to_coid.pop(v, None)
        elif et == "ORDER_REJECTED":
            coid = event.get("client_order_id")
            if isinstance(coid, str):
                self._coid_to_fields.pop(coid, None)
                stale = [v for v, c in self._venue_to_coid.items() if c == coid]
                for v in stale:
                    self._venue_to_coid.pop(v, None)

    def _get_tick_size(self, asset_id: str) -> float:
        cached = self._tick_cache.get(asset_id)
        if cached is not None:
            return cached
        tick = self._fetch_tick_size(asset_id)
        self._tick_cache[asset_id] = tick
        return tick

    def _fetch_tick_size(self, asset_id: str) -> float:
        url = f"{self._clob_url}/book?token_id={asset_id}"
        try:
            with urllib.request.urlopen(url, timeout=_TICK_FETCH_TIMEOUT_S) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            tick = data.get("tick_size")
            if tick is not None:
                value = float(tick)
                if value > 0:
                    return value
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            OSError,
        ) as exc:
            print(
                f"[real_venue] tick_size fetch failed for asset={asset_id}: "
                f"{type(exc).__name__}: {exc} — using default {self._tick_default}",
                file=sys.stderr, flush=True,
            )
        return self._tick_default
