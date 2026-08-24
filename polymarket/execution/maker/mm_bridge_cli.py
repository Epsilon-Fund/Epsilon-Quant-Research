"""Operator entry point for the MM-engine → maker live bridge (Join 2a surface).

Run with:
    uv run python -m polymarket.execution --mode mm_bridge
    # a REAL market-data feed needs websocket-client (mm_engine's live_shadow transport):
    uv run --with websocket-client python -m polymarket.execution --mode mm_bridge

Behaviour:
  - Loads ``ExecutionConfig`` + ``BridgeConfig`` from the environment
    (``polymarket/execution/.env``). Returns exit code 2 on validation failure.
  - Builds the venue via the SAME ``cli.build_venue_adapter`` the copytrade and maker loops
    use (``POLYMARKET_VENUE=fake`` by default → DRY-RUN, no network orders).
  - Instantiates the SAME strategy/queue/latency models used in the JOIN-1 backtest
    (``SymmetricQuoter`` + ``OptimisticQueue`` + ``ConstantLatency``) so the live loop runs
    identical quoting code.
  - Resolves the YES token id from Gamma if ``POLYMARKET_MM_BRIDGE_ASSET_ID`` is unset.
  - Streams read-only market data via ``mm_engine.feeds.live_shadow`` and ingests our own
    fills via ``DataApiFillSource`` (``data-api/trades`` polled for the funder wallet).

Safety: ``POLYMARKET_MAX_REAL_ORDERS`` + ``POLYMARKET_REQUIRE_OPERATOR_CONFIRM`` (shared
``RealOrderGate``), the kill-switch, and the per-trade/market/deployed/daily caps all sit in
the order path (see ``mm_engine_bridge.VenueOrderRouter``). This task never places real
orders — the real 1-contract run is operator-driven (Join 2b/2c).

Exit codes mirror the maker cli: 0 clean, 2 config error, 3 real-stub, 4 unknown
venue/credential validation, 5 market-lookup failure.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone

from polymarket.execution.cli import build_venue_adapter
from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import (
    JsonlWriter,
    MakerSessionStarted,
    MakerSessionStopped,
)

from .event_calendar import EventCalendar
from .maker_engine import DataApiTradeClient, GammaMarketLookup
from .mm_engine_bridge import (
    BridgeConfig,
    DataApiFillSource,
    MMEngineBridge,
    VenueOrderRouter,
    ensure_mm_engine_importable,
)
from .order_safety import RealOrderGate

ensure_mm_engine_importable()
from mm_engine.latency_models import ConstantLatency  # noqa: E402
from mm_engine.queue_models import OptimisticQueue  # noqa: E402
from mm_engine.strategies import SymmetricQuoter  # noqa: E402
from mm_engine.telemetry import Telemetry  # noqa: E402


def build_bridge(
    *,
    config: ExecutionConfig,
    bridge_config: BridgeConfig,
    venue: object,
    journal: JsonlWriter,
    telemetry: Telemetry | None = None,
    event_calendar: EventCalendar | None = None,
) -> MMEngineBridge:
    """Wire the JOIN-1 models + reused venue/safety into an :class:`MMEngineBridge`.

    Factored out of :func:`main` so tests can build the bridge from config without a
    network feed. Uses ``OptimisticQueue`` (the JOIN-1 stub) so backtest and live share the
    same queue model until Join-2 calibration collapses the bracket.
    """
    gate = RealOrderGate(config=config, journal=journal, label="mm_bridge")
    router = VenueOrderRouter(
        venue=venue,
        config=config,
        journal=journal,
        gate=gate,
        order_type=bridge_config.order_type,
        coid_prefix=bridge_config.coid_prefix,
    )
    return MMEngineBridge(
        strategy=SymmetricQuoter(),
        queue_model=OptimisticQueue(),
        # 2b's measured submit→ack constant (0.0 until the latency harness has run — then
        # POLYMARKET_MM_BRIDGE_LATENCY_MS carries the fit into the queue/latency telemetry).
        latency_model=ConstantLatency(bridge_config.latency_ms),
        router=router,
        bridge_config=bridge_config,
        journal=journal,
        telemetry=telemetry,
        event_calendar=event_calendar,
    )


def _resolve_asset_id(config: ExecutionConfig, bridge_config: BridgeConfig) -> str | None:
    if bridge_config.asset_id:
        return bridge_config.asset_id
    market = GammaMarketLookup(config.gamma_url).get_market(bridge_config.condition_id)
    return market.asset_id if market is not None else None


def main(env: Mapping[str, str] | None = None) -> int:
    src: Mapping[str, str] = os.environ if env is None else env

    try:
        config = ExecutionConfig.from_env(src)
        bridge_config = BridgeConfig.from_env(dict(src))
    except ValueError as exc:
        print(f"[mm_bridge:startup] Config error: {exc}", file=sys.stderr)
        return 2

    journal = JsonlWriter(config.journal_dir, "mm_bridge")
    print(
        f"[mm_bridge:startup] Journal: {config.journal_dir}/mm_bridge-<date>.jsonl",
        flush=True,
    )

    venue_mode = src.get("POLYMARKET_VENUE", "fake")
    try:
        venue = build_venue_adapter(venue_mode, config, journal=journal)
    except NotImplementedError as exc:
        print(f"[mm_bridge:startup] {exc}", file=sys.stderr)
        journal.close()
        return 3
    except ValueError as exc:
        print(f"[mm_bridge:startup] {exc}", file=sys.stderr)
        journal.close()
        return 4
    print(f"[mm_bridge:startup] Venue: {venue_mode}", flush=True)

    asset_id = _resolve_asset_id(config, bridge_config)
    if not asset_id:
        print(
            "[mm_bridge:startup] could not resolve YES token id "
            "(set POLYMARKET_MM_BRIDGE_ASSET_ID or check the condition id)",
            file=sys.stderr,
        )
        _stop_venue(venue)
        journal.close()
        return 5
    bridge_config = _with_asset(bridge_config, asset_id)

    bridge = build_bridge(
        config=config,
        bridge_config=bridge_config,
        venue=venue,
        journal=journal,
        event_calendar=EventCalendar.default(),
    )

    print(
        f"[mm_bridge:startup] condition={bridge_config.condition_id} "
        f"asset={asset_id} half_spread={bridge_config.half_spread} "
        f"size={bridge_config.size_contracts} venue={venue_mode} "
        f"max_real_orders={config.max_real_orders} "
        f"operator_confirm={config.require_operator_confirm} "
        f"latency_ms={bridge_config.latency_ms} "
        f"max_inventory={bridge_config.max_inventory_contracts} "
        f"reconcile_every={bridge_config.reconcile_interval_events}",
        flush=True,
    )
    if bridge_config.max_inventory_contracts <= 0:
        print(
            "[mm_bridge:startup] WARNING: POLYMARKET_MM_BRIDGE_MAX_INVENTORY is unset/0 — "
            "no hard contracts cap on inventory (USD risk caps still apply). The live "
            "runbook sets this.",
            flush=True,
        )
    if bridge_config.reconcile_interval_events <= 0:
        print(
            "[mm_bridge:startup] WARNING: POLYMARKET_MM_BRIDGE_RECONCILE_EVERY is unset/0 — "
            "periodic venue-state reconcile disabled. The live runbook sets this.",
            flush=True,
        )

    stop_event = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        print(
            f"\n[mm_bridge:shutdown] received signal {signum}, stopping...",
            file=sys.stderr,
            flush=True,
        )
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
    except ValueError:
        pass

    journal.write(MakerSessionStarted(
        ts_utc=datetime.now(timezone.utc),
        condition_id=bridge_config.condition_id,
        venue=venue_mode,
        size_contracts=float(bridge_config.size_contracts),
    ))

    fill_source = DataApiFillSource(
        condition_id=bridge_config.condition_id,
        asset_id=asset_id,
        funder=config.funder,
        session_start=datetime.now(timezone.utc),
        data_client=DataApiTradeClient(config.data_url),
        tick=bridge_config.tick,
    )

    stop_reason = "operator_signal"
    max_events = _int_env(src, "POLYMARKET_MM_BRIDGE_MAX_EVENTS", 0) or None
    try:
        # mm_engine's live_shadow feed is read-only (no auth, no orders). A real connection
        # needs websocket-client in the venv; imported lazily so this module stays importable.
        from mm_engine.feeds.live_shadow import live_shadow_feed

        feed = _interruptible(
            live_shadow_feed([asset_id], max_events=max_events), stop_event
        )
        bridge.run(feed, fill_source=fill_source)
        if bridge.router.halted:
            stop_reason = "router_halted"
    except Exception as exc:  # noqa: BLE001
        stop_reason = f"feed_error:{type(exc).__name__}"
        print(f"[mm_bridge:shutdown] feed error: {exc!r}", file=sys.stderr, flush=True)
    finally:
        _stop_venue(venue)
        journal.write(MakerSessionStopped(
            ts_utc=datetime.now(timezone.utc),
            condition_id=bridge_config.condition_id,
            reason=stop_reason,
        ))
        journal.close()

    print(f"[mm_bridge:shutdown] clean exit (reason={stop_reason})", flush=True)
    return 0


def _interruptible(feed, stop_event: threading.Event):
    for item in feed:
        if stop_event.is_set():
            break
        yield item


def _with_asset(bridge_config: BridgeConfig, asset_id: str) -> BridgeConfig:
    from dataclasses import replace

    return replace(bridge_config, asset_id=asset_id)


def _stop_venue(venue: object) -> None:
    venue_stop = getattr(venue, "stop", None)
    if callable(venue_stop):
        try:
            venue_stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[mm_bridge:shutdown] venue.stop() raised: {exc!r}", file=sys.stderr)


def _int_env(src: Mapping[str, str], name: str, default: int) -> int:
    raw = src.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


if __name__ == "__main__":
    sys.exit(main())
