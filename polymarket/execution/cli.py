"""Entry point. Wires watcher → signal → risk → mirror engine.

Run with:
    python -m polymarket.execution.cli           # via this module's __main__ guard
    python -m polymarket.execution               # via execution/__main__.py

Behaviour:
  - Loads ExecutionConfig from environment (or an injected mapping
    in tests). Returns exit code 2 on validation failure.
  - Builds journal, dedup, classifier, mirror_engine, watcher, queues.
  - Installs SIGINT/SIGTERM handlers that flip stop_event and exit.
  - Main loop: drain fill_queue → classifier → signal_queue →
    mirror_engine, with halt check between iterations.
  - Periodic status to stdout every ~60 s.
  - On halt or signal: stop watcher, drain residual queues briefly,
    close journal, print final stats, exit 0 if clean / 1 if halted.

Venue dispatch:
  POLYMARKET_VENUE=fake → _PrintVenueAdapter (default).
  POLYMARKET_VENUE=real → NotImplementedError, exit 3.
  anything else → ValueError, exit 4.
"""
from __future__ import annotations

import os
import queue
import signal
import sys
import threading
import time
from collections.abc import Mapping

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import JsonlWriter
from polymarket.execution.mirror import MirrorEngine, SubmitResult, VenueAdapter
from polymarket.execution.signal import Classifier, Deduplicator
from polymarket.execution.watcher import LeaderWatcher

_FILL_QUEUE_MAXSIZE: int = 1000
_SIGNAL_QUEUE_MAXSIZE: int = 1000
_FILL_POLL_TIMEOUT_S: float = 0.5
_STATUS_INTERVAL_S: float = 60.0
_SHUTDOWN_DRAIN_BUDGET_S: float = 5.0


class _PrintVenueAdapter:
    """Trivial dry-run adapter: prints submissions, returns immediate-fill success.

    Used when POLYMARKET_VENUE=fake. Suitable for end-to-end smoke
    against live RTDS without sending real orders.
    """

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
        print(
            f"[fake_venue] coid={client_order_id} {side} "
            f"{size_shares} @ {price} cond={condition_id[:18]} "
            f"asset=...{asset_id[-12:]} type={order_type}",
            flush=True,
        )
        return SubmitResult(
            accepted=True,
            ambiguous=False,
            venue_order_id=f"fake-{client_order_id}",
            fill_price=price,
            fill_size_shares=size_shares,
        )

    def cancel_order(
        self,
        *,
        client_order_id: str | None = None,
        venue_order_id: str | None = None,
    ) -> dict[str, object]:
        print(
            f"[fake_venue] cancel coid={client_order_id} venue_order_id={venue_order_id}",
            flush=True,
        )
        return {"ambiguous": False}


_PLACEHOLDER_VALUES: frozenset[str] = frozenset({
    "", "dummy", "placeholder", "todo", "xxx", "test",
})


def _looks_like_placeholder(value: str) -> bool:
    """Heuristic for refuse-to-start credentials in real mode."""
    if not value:
        return True
    s = value.strip().lower()
    if s in _PLACEHOLDER_VALUES:
        return True
    # All zeros (with or without leading 0x) — every soak run uses these.
    if set(s) <= {"0", "x"}:
        return True
    return False


def _validate_real_credentials(config: ExecutionConfig) -> None:
    """Refuse to start in real mode with placeholder-looking credentials.

    Raises ValueError naming the offending env var. Catches the common
    failure mode where soak-style dummies get accidentally promoted to
    a real-venue run.
    """
    required: tuple[tuple[str, str], ...] = (
        ("POLYMARKET_PRIVATE_KEY", config.private_key),
        ("POLYMARKET_API_KEY", config.api_key),
        ("POLYMARKET_API_SECRET", config.api_secret),
        ("POLYMARKET_PASSPHRASE", config.passphrase),
        ("POLYMARKET_FUNDER", config.funder),
    )
    for name, value in required:
        if _looks_like_placeholder(value):
            raise ValueError(
                f"{name} looks like a placeholder; refusing to start in real mode"
            )


def build_venue_adapter(venue_mode: str, config: ExecutionConfig,
                          journal: JsonlWriter | None = None) -> VenueAdapter:
    if venue_mode == "fake":
        return _PrintVenueAdapter()
    if venue_mode == "real":
        _validate_real_credentials(config)
        if journal is None:
            raise ValueError("real venue requires a journal handle")
        # Construct the chain: signer → substitute http_client → kernel
        # adapter → wrapper. The substitute (mirror/clob_http_client)
        # replaces the kernel's broken HTTP impl; see PLAN.md and
        # the Phase-1 finding logs for context.
        from polymarket.execution._kernel.polymarket_adapter import (
            PolymarketAdapterConfig,
            PolymarketVenueAdapter,
        )
        from polymarket.execution.mirror.clob_http_client import (
            ClobHttpClient,
            ClobHttpClientConfig,
        )
        from polymarket.execution.mirror.clob_signer import (
            ClobSigner,
            ClobSignerConfig,
        )
        from polymarket.execution.mirror.market_metadata import (
            MarketMetadataCache,
        )
        from polymarket.execution.mirror.real_venue_adapter import (
            RealVenueAdapter,
        )

        # ClobSigner replaces the kernel signer so we can pass
        # `neg_risk` (and `tick_size`) through to py-clob-client's
        # PartialCreateOrderOptions at signing time — the kernel
        # signer ignores those flags, which would make every NegRisk
        # order get rejected as "invalid signature".
        signer = ClobSigner(
            ClobSignerConfig(
                api_url=config.clob_url,
                chain_id=config.chain_id,
                signature_type=config.signature_type,
                funder=config.funder,
            )
        )
        http_client = ClobHttpClient(
            ClobHttpClientConfig(
                api_url=config.clob_url,
                api_key=config.api_key,
                api_secret=config.api_secret,
                passphrase=config.passphrase,
                private_key=config.private_key,
                chain_id=config.chain_id,
            ),
            signer=signer,
        )
        kernel_adapter = PolymarketVenueAdapter(
            client=http_client,
            config=PolymarketAdapterConfig(api_url=config.clob_url),
        )
        market_metadata = MarketMetadataCache(gamma_url=config.gamma_url)
        wrapper = RealVenueAdapter(
            kernel_adapter=kernel_adapter,
            http_client=http_client,
            journal=journal,
            clob_url=config.clob_url,
            bot_proxy_wallet=config.funder,
            market_metadata=market_metadata,
        )
        wrapper.start()
        return wrapper
    raise ValueError(
        f"Unknown POLYMARKET_VENUE: {venue_mode!r} (expected 'fake' or 'real')"
    )


def _redacted_config_summary(config: ExecutionConfig) -> str:
    return (
        "[startup] Config:\n"
        f"  leader_address    = {config.leader_address}\n"
        f"  funder            = {config.funder}\n"
        f"  ws_url            = {config.ws_url}\n"
        f"  clob_url          = {config.clob_url}\n"
        f"  sizing_usd        = ${config.sizing_usd:.2f}\n"
        f"  max_capital_usd   = ${config.max_capital_usd:.2f}\n"
        f"  per_trade_cap     = ${config.per_trade_cap_usd:.2f}\n"
        f"  per_market_cap    = ${config.per_market_cap_usd:.2f}\n"
        f"  max_open_pos      = {config.max_open_positions}\n"
        f"  default_order     = {config.default_order_type}\n"
        f"  price_dev_pct     = {config.price_deviation_pct}\n"
        f"  daily_loss_halt   = ${config.daily_loss_halt_usd:.2f}\n"
        f"  killswitch_path   = {config.killswitch_path}\n"
        f"  journal_dir       = {config.journal_dir}\n"
        f"  log_level         = {config.log_level}\n"
        f"  api_key/secret/passphrase/private_key = <redacted>"
    )


def _print_status(
    *,
    fills_processed: int,
    signals_emitted: int,
    bot_positions: int,
    leader_positions: int,
    halted: bool,
) -> None:
    print(
        f"[status] fills={fills_processed} signals={signals_emitted} "
        f"bot_positions={bot_positions} leader_positions={leader_positions} "
        f"halted={halted}",
        flush=True,
    )


def main(env: Mapping[str, str] | None = None) -> int:
    """Bot entry point. Returns exit code (0 clean, 1 halted, 2 config, 3 real-stub, 4 unknown-venue)."""
    src: Mapping[str, str] = os.environ if env is None else env

    # 1. Config
    try:
        config = ExecutionConfig.from_env(src)
    except ValueError as exc:
        print(f"[startup] Config error: {exc}", file=sys.stderr)
        return 2
    print(_redacted_config_summary(config), flush=True)

    # 2. Journal
    journal = JsonlWriter(config.journal_dir, "execution")
    print(
        f"[startup] Journal: {config.journal_dir}/execution-<date>.jsonl",
        flush=True,
    )

    # 3. Venue adapter
    venue_mode = src.get("POLYMARKET_VENUE", "fake")
    try:
        venue = build_venue_adapter(venue_mode, config, journal=journal)
    except NotImplementedError as exc:
        print(f"[startup] {exc}", file=sys.stderr)
        journal.close()
        return 3
    except ValueError as exc:
        print(f"[startup] {exc}", file=sys.stderr)
        journal.close()
        return 4
    print(f"[startup] Venue: {venue_mode}", flush=True)

    # 4. Queues
    fill_queue: queue.Queue = queue.Queue(maxsize=_FILL_QUEUE_MAXSIZE)
    signal_queue: queue.Queue = queue.Queue(maxsize=_SIGNAL_QUEUE_MAXSIZE)

    # 5. Dedup, classifier, mirror_engine
    dedup = Deduplicator(journal)
    classifier = Classifier(config, journal, dedup, signal_queue)
    mirror_engine = MirrorEngine(
        config, journal, venue, signal_queue, config.killswitch_path
    )
    print(
        f"[startup] Rebuild: dedup={len(dedup)} txhashes, "
        f"classifier bot={len(classifier._bot_positions)} "
        f"leader={len(classifier._leader_positions)}, "
        f"mirror bot={len(mirror_engine._bot_positions)} "
        f"daily_pnl=${mirror_engine._daily_realised_pnl_usd:.2f}",
        flush=True,
    )

    # 6. Watcher
    watcher = LeaderWatcher(config, journal, fill_queue)

    # 7. Signal handlers (install before any thread starts)
    stop_event = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        print(
            f"\n[shutdown] received signal {signum}, stopping cleanly...",
            file=sys.stderr,
            flush=True,
        )
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
    except ValueError:
        # signal.signal must be called from the main thread; tolerate
        # being invoked from a worker (e.g. tests) by skipping the
        # install rather than aborting.
        pass

    # 8. Start watcher
    watcher.start()
    print(
        f"[startup] Watcher started — leader {config.leader_address}",
        flush=True,
    )

    # 9. Main loop
    fills_processed = 0
    signals_emitted = 0
    last_status_ts = time.monotonic()

    while not stop_event.is_set():
        # Pull one fill (blocks up to _FILL_POLL_TIMEOUT_S) → classifier.
        try:
            fill = fill_queue.get(timeout=_FILL_POLL_TIMEOUT_S)
        except queue.Empty:
            pass
        else:
            try:
                classifier.process_fill(fill)
                fills_processed += 1
            except Exception as exc:
                print(
                    f"[error] classifier.process_fill raised: {exc!r}",
                    file=sys.stderr, flush=True,
                )
            fill_queue.task_done()

        # Drain all currently-queued signals → mirror_engine.
        while True:
            try:
                sig = signal_queue.get_nowait()
            except queue.Empty:
                break
            try:
                mirror_engine.handle_signal(sig)
                signals_emitted += 1
            except Exception as exc:
                print(
                    f"[error] mirror_engine.handle_signal raised: {exc!r}",
                    file=sys.stderr, flush=True,
                )
            signal_queue.task_done()

        # Halt check
        if mirror_engine.is_halted():
            print(
                "[halt] mirror_engine reports halted state — stopping.",
                file=sys.stderr, flush=True,
            )
            stop_event.set()

        # Periodic status
        now = time.monotonic()
        if now - last_status_ts >= _STATUS_INTERVAL_S:
            _print_status(
                fills_processed=fills_processed,
                signals_emitted=signals_emitted,
                bot_positions=len(mirror_engine._bot_positions),
                leader_positions=len(classifier._leader_positions),
                halted=mirror_engine.is_halted(),
            )
            last_status_ts = now

    # 10. Shutdown
    print("[shutdown] stopping watcher...", flush=True)
    watcher.stop()
    # Real-venue wrapper has its own polling thread; stop it before the
    # final journal close so any last events flush. Fake adapter has no
    # stop method — duck-typed.
    venue_stop = getattr(venue, "stop", None)
    if callable(venue_stop):
        try:
            venue_stop()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[shutdown] venue.stop() raised: {exc!r}",
                file=sys.stderr, flush=True,
            )

    print("[shutdown] draining residual queues (≤5 s)...", flush=True)
    deadline = time.monotonic() + _SHUTDOWN_DRAIN_BUDGET_S
    while time.monotonic() < deadline:
        progress = False
        try:
            fill = fill_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            try:
                classifier.process_fill(fill)
                fills_processed += 1
            except Exception:
                pass
            fill_queue.task_done()
            progress = True
        try:
            sig = signal_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            try:
                mirror_engine.handle_signal(sig)
                signals_emitted += 1
            except Exception:
                pass
            signal_queue.task_done()
            progress = True
        if not progress:
            break

    journal.close()
    halted = mirror_engine.is_halted()
    print(
        f"[shutdown] final: fills={fills_processed} signals={signals_emitted} "
        f"bot_positions={len(mirror_engine._bot_positions)} "
        f"halted={halted}",
        flush=True,
    )
    return 1 if halted else 0


if __name__ == "__main__":
    sys.exit(main())
