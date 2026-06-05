from __future__ import annotations

import asyncio
import signal
import sqlite3
import time
from dataclasses import dataclass

from executor.risk import ExecutionRiskManager, RiskManagerConfig
from executor.venue import NullLogger, StructuredLogger, VenueAdapter

from .market_data import BookUpdateEvent, MarketDataClient, UserOrderEvent
from .oms import OrderManagementSystem, OrderStatus
from .persistence import check_kill_switch, delete_all_orders, delete_order, set_kill_switch, upsert_order, write_heartbeat
from .registry import TokenRegistry
from .strategy import StrategySignal, TailHarvesterStrategy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    shutdown_timeout_s: float = 10.0           # max time to wait for cancel_all on shutdown
    stop_loss_price: float = 0.0               # 0 = disabled; exit position when bid drops below this
    kill_check_interval_s: float = 3.0         # how often to check the dashboard kill switch
    position_reconcile_interval_s: float = 10.0   # how often to sync positions from venue (ground truth)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ExecutionEngine:
    """Async orchestration layer.

    Wires together:
        MarketDataClient → TokenRegistry → TailHarvesterStrategy → OMS → VenueAdapter

    Threading model:
        - All state mutations (registry, OMS) run on the asyncio event loop thread.
        - Blocking HTTP calls (submit_order, cancel_order, poll, reconcile) are
          offloaded to the default thread pool via asyncio.to_thread so the event
          loop stays free to process incoming price/fill events.

    Lifecycle:
        await engine.run()   — blocks until SIGINT / SIGTERM
    """

    __slots__ = (
        "_client", "_registry", "_strategy", "_oms", "_adapter", "_config", "_logger", "_risk",
        "_db", "_last_market_msg_ns", "_stop_loss_price", "_stop_loss_slugs", "_placing_slugs",
    )

    def __init__(
        self,
        *,
        client: MarketDataClient,
        registry: TokenRegistry,
        strategy: TailHarvesterStrategy,
        oms: OrderManagementSystem,
        adapter: VenueAdapter,
        config: ExecutionConfig,
        logger: StructuredLogger | None = None,
        risk: ExecutionRiskManager | None = None,
        db: sqlite3.Connection | None = None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._strategy = strategy
        self._oms = oms
        self._adapter = adapter
        self._config = config
        self._logger: StructuredLogger = logger or NullLogger()
        self._risk = risk or ExecutionRiskManager(RiskManagerConfig())
        self._db = db
        self._last_market_msg_ns: int = 0
        self._stop_loss_price: float = config.stop_loss_price
        self._stop_loss_slugs: set[str] = set()
        # Slugs with an OMS call currently in-flight on the thread pool.
        # Prevents duplicate submissions when rapid book updates arrive while
        # place_or_replace is awaited (asyncio.to_thread yields the event loop).
        self._placing_slugs: set[str] = set()

    async def run(self) -> None:
        """Run the engine until a shutdown signal is received."""
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, OSError):
                # Windows: add_signal_handler not supported for all signals
                signal.signal(sig, lambda _s, _f: loop.call_soon_threadsafe(stop.set))

        async with self._client:
            # Clear any previous kill signal, wipe stale orders, write initial heartbeat
            self._logger.info("engine_ready", db_connected=self._db is not None)
            if self._db is not None:
                set_kill_switch(self._db, False)
                delete_all_orders(self._db)
            self._write_heartbeat()

            # Cancel any stale open orders from previous sessions before placing new ones.
            # reconcile_open_orders can't match them (Polymarket doesn't echo clientOrderId
            # back in GET /data/orders), so the OMS would be blind to them and place
            # duplicate orders alongside the stale ones.
            cancelled = await asyncio.to_thread(self._adapter.cancel_all_open_at_venue)
            if cancelled:
                self._logger.info("startup_cancelled_stale_orders", count=cancelled)

            # Sync open orders from previous run, then sync actual positions from venue.
            # Position sync runs first so the cap is correct before any new orders go out.
            await self._reconcile()
            await self._reconcile_positions()

            kill_chk_task  = asyncio.create_task(self._kill_check_loop(stop),         name="kill-check")
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(stop),          name="heartbeat")
            pos_task       = asyncio.create_task(self._position_reconcile_loop(stop), name="pos-reconcile")
            try:
                async for event in self._client.events():
                    if stop.is_set():
                        break
                    await self._dispatch(event)
            finally:
                kill_chk_task.cancel()
                heartbeat_task.cancel()
                pos_task.cancel()
                await asyncio.gather(
                    kill_chk_task, heartbeat_task, pos_task,
                    return_exceptions=True,
                )

        # Client is stopped — run shutdown sequence before exiting
        await self._shutdown()

    # ------------------------------------------------------------------
    # Event dispatch — runs on event loop thread
    # ------------------------------------------------------------------

    async def _dispatch(self, event: BookUpdateEvent | UserOrderEvent) -> None:
        if isinstance(event, BookUpdateEvent):
            await self._on_book_update(event)
        else:
            if event.fill_qty:
                self._logger.info(
                    "fill",
                    client_order_id=event.client_order_id,
                    status=event.status,
                    fill_qty=event.fill_qty,
                    fill_price=event.fill_price,
                )
            self._oms.on_user_order_event(event)

    async def _on_book_update(self, event: BookUpdateEvent) -> None:
        self._last_market_msg_ns = event.ts_ns
        self._registry.update(event)

        token_state = self._registry.state_for(event.token_id)
        if token_state is None:
            return

        slug = token_state.record.event_slug
        self._logger.debug(
            "book_update",
            slug=slug,
            token_id=event.token_id[:8] + "…",
            best_bid_c=round(event.best_bid * 100, 3),
            best_ask_c=round(event.best_ask * 100, 3),
        )

        now_ns = time.time_ns()

        # --- Stop-loss: check on every tick for the token we own ---
        if self._stop_loss_price > 0.0 and slug not in self._stop_loss_slugs:
            sl_state = self._oms.order_state(slug)
            if (
                sl_state is not None
                and sl_state.filled_qty > 0
                and sl_state.token_id == event.token_id
                and event.best_bid < self._stop_loss_price
            ):
                self._stop_loss_slugs.add(slug)  # guard BEFORE await
                self._logger.warning(
                    "stop_loss.triggered",
                    slug=slug,
                    token_id=event.token_id[:8] + "…",
                    best_bid_c=round(event.best_bid * 100, 1),
                    threshold_c=round(self._stop_loss_price * 100, 1),
                    filled_qty=sl_state.filled_qty,
                    filled_usdc=sl_state.filled_usdc,
                )
                self._registry.mark_closed(slug)
                await asyncio.to_thread(
                    self._oms.sell_position, event_slug=slug, now_ns=now_ns, bid_price=event.best_bid
                )
                estimated_loss = max(
                    0.0, sl_state.filled_usdc - event.best_bid * round(sl_state.filled_qty)
                )
                loss_decision = self._risk.record_loss(estimated_loss, now_ns)
                if loss_decision is not None:
                    self._logger.warning(
                        "daily_loss_cap.breached",
                        slug=slug,
                        trigger="stop_loss",
                        estimated_loss_usdc=round(estimated_loss, 4),
                    )

        # --- Zero-resolution: bid hit 0 means market resolved against us ---
        if event.best_bid == 0.0 and slug not in self._stop_loss_slugs:
            zero_state = self._oms.order_state(slug)
            if (
                zero_state is not None
                and zero_state.filled_qty > 0
                and zero_state.token_id == event.token_id
                and zero_state.filled_usdc > 0.0
            ):
                self._stop_loss_slugs.add(slug)  # prevent double-recording
                self._logger.warning(
                    "zero_resolution.triggered",
                    slug=slug,
                    token_id=event.token_id[:8] + "…",
                    filled_usdc=zero_state.filled_usdc,
                    filled_qty=zero_state.filled_qty,
                )
                loss_decision = self._risk.record_loss(zero_state.filled_usdc, now_ns)
                if loss_decision is not None:
                    self._logger.warning(
                        "daily_loss_cap.breached",
                        slug=slug,
                        trigger="zero_resolution",
                        loss_usdc=round(zero_state.filled_usdc, 4),
                    )

        order_state = self._oms.order_state(slug)
        has_open = order_state is not None and order_state.status in {OrderStatus.OPEN, OrderStatus.AMBIGUOUS}
        current_price_ticks = order_state.price_ticks if order_state else None
        current_token_id = order_state.token_id if order_state else None

        sig = self._strategy.evaluate(
            slug,
            has_open_order=has_open,
            current_price_ticks=current_price_ticks,
            current_token_id=current_token_id,
        )
        if sig.action == "NO_OP":
            self._logger.debug(
                "signal.noop",
                slug=slug,
                open_ticks=current_price_ticks,
                bid_c=round(event.best_bid * 100, 3),
            )
        await self._act(sig, now_ns)

    async def _act(self, sig: StrategySignal, now_ns: int) -> None:
        """Translate a strategy signal into an OMS call.

        OMS logic itself is synchronous and stays on the event loop thread.
        The HTTP calls it triggers are offloaded to a thread pool via to_thread.
        """
        if sig.action == "PLACE":
            if sig.token_id is None or sig.condition_id is None or sig.price_ticks is None:
                return
            order_state = self._oms.order_state(sig.event_slug)
            filled_qty = order_state.filled_qty if order_state else 0.0
            decision = self._risk.check_order_allowed(
                event_slug=sig.event_slug,
                filled_qty=filled_qty,
                order_qty=self._oms._order_qty,
                ts_ns=now_ns,
            )
            if not decision.allowed:
                self._logger.warning(
                    "risk.blocked",
                    slug=sig.event_slug,
                    code=decision.code.value if decision.code else None,
                    reason=decision.reason,
                    filled_qty=round(filled_qty, 1),
                    cap=self._risk._config.max_position_qty_per_event,
                )
                return
            # In-flight guard: asyncio.to_thread yields the event loop, so a rapid
            # book update can enter _act for the same slug before the OMS state is
            # updated. Without this guard, three concurrent submissions are possible.
            if sig.event_slug in self._placing_slugs:
                return
            self._logger.info(
                "signal.place",
                slug=sig.event_slug,
                token_id=sig.token_id,
                price_ticks=sig.price_ticks,
                tick_size=sig.tick_size,
            )
            self._placing_slugs.add(sig.event_slug)
            try:
                await asyncio.to_thread(
                    self._oms.place_or_replace,
                    event_slug=sig.event_slug,
                    token_id=sig.token_id,
                    condition_id=sig.condition_id,
                    price_ticks=sig.price_ticks,
                    now_ns=now_ns,
                    tick_size=sig.tick_size,
                )
            finally:
                self._placing_slugs.discard(sig.event_slug)
            if self._db is not None:
                state = self._oms.order_state(sig.event_slug)
                token_record = self._registry.state_for(sig.token_id)
                is_yes = token_record.record.is_yes if token_record else True
                try:
                    upsert_order(
                        self._db,
                        event_slug=sig.event_slug,
                        token_id=sig.token_id,
                        is_yes=is_yes,
                        price_ticks=sig.price_ticks,
                        tick_size=sig.tick_size,
                        qty=self._oms._order_qty,
                        status=state.status.value if state else "OPEN",
                        venue_order_id=state.client_order_id if state else None,
                    )
                except Exception:
                    pass
        elif sig.action == "CANCEL":
            self._logger.info("signal.cancel", slug=sig.event_slug)
            await asyncio.to_thread(
                self._oms.cancel,
                event_slug=sig.event_slug,
                now_ns=now_ns,
            )
            if self._db is not None:
                try:
                    delete_order(self._db, sig.event_slug)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Periodic tasks — background loops
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self, stop: asyncio.Event) -> None:
        """Write a DB heartbeat every 10 s."""
        self._logger.info("heartbeat_loop_started", db_connected=self._db is not None)
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                try:
                    self._write_heartbeat()
                except Exception as exc:
                    self._logger.warning("heartbeat_loop_error", error=str(exc))

    async def _kill_check_loop(self, stop: asyncio.Event) -> None:
        """Check the dashboard kill switch every kill_check_interval_s seconds."""
        if self._db is None:
            return
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._config.kill_check_interval_s)
            except asyncio.TimeoutError:
                if check_kill_switch(self._db):
                    self._logger.warning(
                        "kill_switch.triggered",
                        detail="Dashboard kill switch activated — shutting down",
                    )
                    stop.set()

    def _write_heartbeat(self) -> None:
        if self._db is None:
            return
        now_ns = time.time_ns()
        age_s = (now_ns - self._last_market_msg_ns) / 1e9 if self._last_market_msg_ns else None
        active = len({
            s.record.event_slug for s in self._registry.qualified()
        })
        open_orders = len(self._oms.all_open_client_order_ids())
        self._logger.info("heartbeat_write_attempt", active=active, open_orders=open_orders)
        try:
            write_heartbeat(
                self._db,
                ws_market_connected=self._last_market_msg_ns > 0 and age_s is not None and age_s < 60,
                ws_user_connected=True,
                last_market_msg_age_s=age_s,
                active_markets=active,
                open_orders=open_orders,
            )
        except Exception as exc:
            self._logger.warning("heartbeat_write_failed", error=str(exc))
            raise

    async def _reconcile(self) -> None:
        """Full reconciliation: compare our expected open orders against the venue."""
        expected = self._oms.all_open_client_order_ids()
        result = await asyncio.to_thread(
            self._adapter.reconcile_open_orders, expected
        )
        if result.generated_events:
            self._oms.on_venue_events(result.generated_events)

    async def _position_reconcile_loop(self, stop: asyncio.Event) -> None:
        """Periodically fetch actual positions from the venue and correct OMS upward.

        This is the ground-truth safety net: even if every WS fill event and every
        poll fill update fails, the venue knows what we hold and will correct the cap.
        Runs every position_reconcile_interval_s seconds (default 2 minutes).
        """
        interval = self._config.position_reconcile_interval_s
        if interval <= 0:
            return
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                try:
                    await self._reconcile_positions()
                except Exception as exc:
                    self._logger.warning("position_reconcile_failed", error=str(exc))

    async def _reconcile_positions(self) -> None:
        """Single position reconciliation pass: fetch, map to slugs, correct OMS.

        Positions are aggregated PER SLUG before updating the OMS. A market with
        multiple token positions (e.g. YES on one outcome + NO on another) must be
        summed so the cap sees the true total exposure, not just the last token seen.
        Direction is upward-only: Data API serves as a floor, not an override.  If
        the API hasn't caught up with a very recent fill the WS already counted, we
        keep the higher WS value — never open a gap that allows extra orders.
        """
        try:
            positions = await asyncio.to_thread(self._adapter.get_positions)
        except Exception as exc:
            self._logger.warning("position_fetch_failed", error=str(exc))
            return

        # Build conditionId+(Yes|No) → token_id map so Data API positions
        # (which carry conditionId+outcome instead of token_id) can be resolved.
        cond_map: dict[tuple[str, str], str] = {}
        for tid in self._registry.all_token_ids():
            ts = self._registry.state_for(tid)
            if ts is not None:
                r = ts.record
                cond_map[(r.condition_id, "Yes" if r.is_yes else "No")] = r.token_id

        # Aggregate all token positions for the same slug into a single total.
        # Maps slug → (total_qty, total_cost_usdc, first_token_id, condition_id, tick_size)
        slug_agg: dict[str, list] = {}
        for pos in positions:
            token_id = pos.token_id
            if not token_id and pos.condition_id:
                token_id = cond_map.get((pos.condition_id, pos.outcome), "")
            if not token_id:
                continue
            token_state = self._registry.state_for(token_id)
            if token_state is None:
                continue
            slug = token_state.record.event_slug
            if slug not in slug_agg:
                slug_agg[slug] = [
                    0.0, 0.0,
                    token_id,
                    token_state.record.condition_id,
                    token_state.record.tick_size,
                ]
            slug_agg[slug][0] += pos.size
            slug_agg[slug][1] += pos.cost_usdc

        corrections = 0
        for slug, (total_qty, total_cost, token_id, condition_id, tick_size) in slug_agg.items():
            corrected = self._oms.update_filled_from_position(
                event_slug=slug,
                actual_qty=total_qty,
                actual_cost_usdc=total_cost,
                token_id=token_id,
                condition_id=condition_id,
                tick_size=tick_size,
            )
            if corrected:
                corrections += 1
                self._logger.warning(
                    "position_drift_corrected",
                    slug=slug,
                    total_qty=round(total_qty, 2),
                    total_cost_usdc=round(total_cost, 2),
                )

        self._logger.info(
            "position_reconcile_complete",
            positions=len(positions),
            corrections=corrections,
        )

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        """Cancel all open orders then clear DB."""
        now_ns = time.time_ns()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._oms.cancel_all, now_ns=now_ns),
                timeout=self._config.shutdown_timeout_s,
            )
        except asyncio.TimeoutError:
            pass
        # Clear DB so dashboard shows no orders immediately after bot stops
        if self._db is not None:
            try:
                delete_all_orders(self._db)
            except Exception:
                pass
