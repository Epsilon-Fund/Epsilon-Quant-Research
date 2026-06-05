"""Entry point for the politics NegRisk passive maker measurement loop.

Run with:
    uv run python -m polymarket.execution --mode maker
    uv run python -m polymarket.execution --mode maker --check-auth
    uv run python -m polymarket.execution.maker.cli   # via this module's guard

Behaviour:
  - Loads ExecutionConfig + MakerEngineConfig from environment (or an
    injected mapping in tests). Returns exit code 2 on validation failure.
  - Builds journal, venue adapter (reusing cli.build_venue_adapter — fake
    by default, real on POLYMARKET_VENUE=real), NegRiskInventoryTracker,
    ResolutionHandler, EventCalendar, and MakerEngine.
  - Starts all three background threads (inventory, resolution, engine).
  - Installs SIGINT/SIGTERM handlers that flip stop_event.
  - Optional auto-stop after POLYMARKET_MAKER_MAX_RUNTIME_SECONDS (used by
    smoke runs; 0/unset means run until signalled).
  - On shutdown: cancels all open quotes, stops the three components in
    reverse start order, stops the venue (duck-typed), and closes journal.
  - Logs MakerSessionStarted / MakerSessionStopped to the journal.

Safety env vars (POLYMARKET_MAX_REAL_ORDERS, POLYMARKET_REQUIRE_OPERATOR_CONFIRM)
are read by ExecutionConfig and consulted inside MakerEngine exactly as the
copytrade bot consults them — they gate real-venue submits only and never
affect fake-venue runs.

Exit codes mirror the copytrade cli: 0 clean, 2 config error, 3 real-stub,
4 unknown venue/credential validation, 5 auth read failure.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from polymarket.execution.cli import build_venue_adapter
from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import (
    JsonlWriter,
    MakerSessionStarted,
    MakerSessionStopped,
)
from polymarket.execution.maker.event_calendar import EventCalendar
from polymarket.execution.maker.maker_engine import MakerEngine, MakerEngineConfig
from polymarket.execution.maker.negrisk_inventory import NegRiskInventoryTracker
from polymarket.execution.maker.resolution_handler import ResolutionHandler

_INVENTORY_STATE_FILENAME = "negrisk_inventory.jsonl"
_STATUS_INTERVAL_S = 60.0
_WAIT_TICK_S = 0.5


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def main(env: Mapping[str, str] | None = None, *, check_auth: bool = False) -> int:
    src: Mapping[str, str] = os.environ if env is None else env
    if check_auth:
        return check_real_auth(src)

    # 1. Config (execution + maker).
    try:
        config = ExecutionConfig.from_env(src)
        maker_config = MakerEngineConfig.from_env(dict(src))
    except ValueError as exc:
        print(f"[maker:startup] Config error: {exc}", file=sys.stderr)
        return 2

    # 2. Journal.
    journal = JsonlWriter(config.journal_dir, "maker")
    print(
        f"[maker:startup] Journal: {config.journal_dir}/maker-<date>.jsonl",
        flush=True,
    )

    # 3. Venue adapter (reuse the copytrade builder + its safety semantics).
    venue_mode = src.get("POLYMARKET_VENUE", "fake")
    try:
        venue = build_venue_adapter(venue_mode, config, journal=journal)
    except NotImplementedError as exc:
        print(f"[maker:startup] {exc}", file=sys.stderr)
        journal.close()
        return 3
    except ValueError as exc:
        print(f"[maker:startup] {exc}", file=sys.stderr)
        journal.close()
        return 4
    print(f"[maker:startup] Venue: {venue_mode}", flush=True)

    # 4. Inventory tracker, resolution handler, event calendar, maker engine.
    inventory = NegRiskInventoryTracker(
        funder=config.funder,
        data_url=config.data_url,
        state_path=Path(config.journal_dir) / _INVENTORY_STATE_FILENAME,
        journal=journal,
    )
    resolution = ResolutionHandler(
        condition_ids_provider=lambda: {maker_config.condition_id.lower()},
        funder=config.funder,
        gamma_url=config.gamma_url,
        data_url=config.data_url,
        journal=journal,
        private_key=config.private_key,
        rpc_url=src.get("POLYMARKET_RPC_URL"),
        chain_id=config.chain_id,
    )
    event_calendar = EventCalendar.default()
    engine = MakerEngine(
        execution_config=config,
        maker_config=maker_config,
        journal=journal,
        venue=venue,
        inventory=inventory,
        event_calendar=event_calendar,
        resolution_state=resolution,
    )

    print(
        f"[maker:startup] condition={maker_config.condition_id} "
        f"size={maker_config.size_contracts} "
        f"refresh={maker_config.refresh_interval_seconds}s "
        f"max_real_orders={config.max_real_orders} "
        f"operator_confirm={config.require_operator_confirm} "
        f"events={len(event_calendar.events)}",
        flush=True,
    )

    # 5. Signal handlers (install before any thread starts).
    stop_event = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        print(
            f"\n[maker:shutdown] received signal {signum}, stopping cleanly...",
            file=sys.stderr,
            flush=True,
        )
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
    except ValueError:
        # signal.signal must run on the main thread; tolerate being called
        # from a worker (tests) by skipping the install rather than aborting.
        pass

    # 6. Journal the session start, then start the three threads.
    journal.write(MakerSessionStarted(
        ts_utc=_utc_now(),
        condition_id=maker_config.condition_id,
        venue=venue_mode,
        size_contracts=float(maker_config.size_contracts),
    ))
    inventory.start()
    resolution.start()
    engine.start()
    print("[maker:startup] inventory + resolution + engine threads started", flush=True)

    # 7. Wait until signalled (or optional max-runtime auto-stop).
    max_runtime = _float_env(src, "POLYMARKET_MAKER_MAX_RUNTIME_SECONDS", 0.0)
    deadline = (time.monotonic() + max_runtime) if max_runtime > 0 else None
    stop_reason = "operator_signal"
    last_status = time.monotonic()
    while not stop_event.is_set():
        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "max_runtime_reached"
            break
        now = time.monotonic()
        if now - last_status >= _STATUS_INTERVAL_S:
            print(
                f"[maker:status] open_quotes={len(engine._quotes)} "
                f"exposure={inventory.get_basket_exposure(maker_config.condition_id):.2f} "
                f"resolved={resolution.is_resolved(maker_config.condition_id)}",
                flush=True,
            )
            last_status = now
        stop_event.wait(_WAIT_TICK_S)

    # 8. Shutdown: cancel quotes, stop components in reverse start order.
    print(f"[maker:shutdown] reason={stop_reason}; cancelling open quotes...", flush=True)
    try:
        engine.cancel_all(reason="shutdown")
    except Exception as exc:  # noqa: BLE001
        print(f"[maker:shutdown] cancel_all raised: {exc!r}", file=sys.stderr, flush=True)
    for name, component in (
        ("engine", engine),
        ("resolution", resolution),
        ("inventory", inventory),
    ):
        try:
            component.stop()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[maker:shutdown] {name}.stop() raised: {exc!r}",
                file=sys.stderr, flush=True,
            )

    venue_stop = getattr(venue, "stop", None)
    if callable(venue_stop):
        try:
            venue_stop()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[maker:shutdown] venue.stop() raised: {exc!r}",
                file=sys.stderr, flush=True,
            )

    journal.write(MakerSessionStopped(
        ts_utc=_utc_now(),
        condition_id=maker_config.condition_id,
        reason=stop_reason,
    ))
    journal.close()
    print(f"[maker:shutdown] clean exit (reason={stop_reason})", flush=True)
    return 0


def check_real_auth(env: Mapping[str, str] | None = None) -> int:
    """Read CLOB open orders with real credentials and submit nothing."""
    src: Mapping[str, str] = os.environ if env is None else env
    try:
        config = ExecutionConfig.from_env(src)
    except ValueError as exc:
        print(f"[maker:auth] Config error: {exc}", file=sys.stderr)
        return 2

    journal = JsonlWriter(config.journal_dir, "maker")
    venue = None
    try:
        venue = build_venue_adapter("real", config, journal=journal)
        count = _read_open_order_count(venue)
    except NotImplementedError as exc:
        print(f"[maker:auth] {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"[maker:auth] {exc}", file=sys.stderr)
        return 4
    except Exception as exc:  # noqa: BLE001
        print(
            f"[maker:auth] auth read failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 5
    finally:
        if venue is not None:
            venue_stop = getattr(venue, "stop", None)
            if callable(venue_stop):
                try:
                    venue_stop()
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[maker:auth] venue.stop() raised: {exc!r}",
                        file=sys.stderr,
                        flush=True,
                    )
        journal.close()

    print(f"[maker:auth] open_orders={count}", flush=True)
    return 0


def _read_open_order_count(venue: object) -> int:
    reconcile = getattr(venue, "reconcile_open_orders", None)
    if callable(reconcile):
        result = reconcile(set())
        open_ids = getattr(result, "venue_open_client_order_ids", None)
        if open_ids is not None:
            return len(open_ids)
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            return len(result)

    get_open_orders = getattr(venue, "get_open_orders", None)
    if callable(get_open_orders):
        result = get_open_orders()
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            return len(result)

    raise RuntimeError("venue does not expose an open-order read method")


def _float_env(src: Mapping[str, str], name: str, default: float) -> float:
    raw = src.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


if __name__ == "__main__":
    sys.exit(main(check_auth="--check-auth" in sys.argv[1:]))
