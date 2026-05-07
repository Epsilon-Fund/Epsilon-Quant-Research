from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import dataclass

from executor.risk import ExecutionRiskManager, RiskManagerConfig
from executor.venue import NullLogger, StructuredLogger, VenueAdapter

from .market_data import BookUpdateEvent, MarketDataClient, UserOrderEvent
from .oms import OrderManagementSystem, OrderStatus
from .registry import TokenRegistry
from .strategy import StrategySignal, TailHarvesterStrategy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    poll_interval_s: float = 15.0     # how often to poll venue for missed fills
    shutdown_timeout_s: float = 10.0  # max time to wait for cancel_all on shutdown


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
    ) -> None:
        self._client = client
        self._registry = registry
        self._strategy = strategy
        self._oms = oms
        self._adapter = adapter
        self._config = config
        self._logger: StructuredLogger = logger or NullLogger()
        self._risk = risk or ExecutionRiskManager(RiskManagerConfig())

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
            # Sync any orders left open from a previous run before we start
            await self._reconcile()

            poll_task = asyncio.create_task(self._poll_loop(stop), name="poll")
            try:
                async for event in self._client.events():
                    if stop.is_set():
                        break
                    await self._dispatch(event)
            finally:
                poll_task.cancel()
                await asyncio.gather(poll_task, return_exceptions=True)

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
        self._registry.update(event)

        token_state = self._registry.state_for(event.token_id)
        if token_state is None:
            return

        slug = token_state.record.event_slug
        self._logger.debug(
            "book_update",
            slug=slug,
            token_id=event.token_id[:8] + "…",
            best_bid_c=round(event.best_bid * 100, 3),   # e.g. 99.7 (cents, matches screen)
            best_ask_c=round(event.best_ask * 100, 3),
        )

        now_ns = time.time_ns()
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
            filled_usdc = order_state.filled_usdc if order_state else 0.0
            decision = self._risk.check_order_allowed(
                event_slug=sig.event_slug,
                filled_usdc=filled_usdc,
                ts_ns=now_ns,
            )
            if not decision.allowed:
                self._logger.warning(
                    "risk.blocked",
                    slug=sig.event_slug,
                    code=decision.code.value if decision.code else None,
                    reason=decision.reason,
                )
                return
            self._logger.info(
                "signal.place",
                slug=sig.event_slug,
                token_id=sig.token_id,
                price_ticks=sig.price_ticks,
                tick_size=sig.tick_size,
            )
            await asyncio.to_thread(
                self._oms.place_or_replace,
                event_slug=sig.event_slug,
                token_id=sig.token_id,
                condition_id=sig.condition_id,
                price_ticks=sig.price_ticks,
                now_ns=now_ns,
                tick_size=sig.tick_size,
            )
        elif sig.action == "CANCEL":
            self._logger.info("signal.cancel", slug=sig.event_slug)
            await asyncio.to_thread(
                self._oms.cancel,
                event_slug=sig.event_slug,
                now_ns=now_ns,
            )

    # ------------------------------------------------------------------
    # Periodic reconciliation — background task
    # ------------------------------------------------------------------

    async def _poll_loop(self, stop: asyncio.Event) -> None:
        """Poll the venue every poll_interval_s to catch fills missed by the WS."""
        while not stop.is_set():
            try:
                # Wait for the stop signal; timeout triggers a poll cycle
                await asyncio.wait_for(stop.wait(), timeout=self._config.poll_interval_s)
            except asyncio.TimeoutError:
                await self._poll_once()

    async def _poll_once(self) -> None:
        updates = await asyncio.to_thread(self._adapter.poll_or_process_order_updates)
        if updates:
            self._oms.on_venue_events(updates)

    async def _reconcile(self) -> None:
        """Full reconciliation: compare our expected open orders against the venue."""
        expected = self._oms.all_open_client_order_ids()
        result = await asyncio.to_thread(
            self._adapter.reconcile_open_orders, expected
        )
        if result.generated_events:
            self._oms.on_venue_events(result.generated_events)

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        """Cancel all open orders, then do a final poll to capture last fills."""
        now_ns = time.time_ns()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._oms.cancel_all, now_ns=now_ns),
                timeout=self._config.shutdown_timeout_s,
            )
        except asyncio.TimeoutError:
            pass
        await self._poll_once()
