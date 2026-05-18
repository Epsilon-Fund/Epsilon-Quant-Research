"""
Binance WebSocket manager for live price streaming.

Process-wide singleton: exactly one ``ThreadedWebsocketManager`` ever runs
inside this Python process.  Each dashboard calls ``get_shared_ws(symbols)``
to subscribe its symbols; the subscription set is the UNION across all
callers.  Symbols are added incrementally — no restart on subscription
growth — which avoids gaps in the price feed.

Why a singleton: python-binance's TWM owns its own asyncio event loop in
a daemon thread.  Spinning up two TWMs in the same process (one per
dashboard tab) is unsafe — on Python 3.14 the second instance reliably
hits "This event loop is already running" and fails to initialise.  One
shared TWM, many subscribers, sidesteps the problem entirely.

The callback runs in the TWM's background thread.  It only mutates the
shared price dict under a lock — it never touches Streamlit APIs —
which keeps it safe to read from a Streamlit fragment.

Public reads (``get_price``, ``get_prices``, ``is_connected``) are all
lock-guarded.
"""

from __future__ import annotations

import threading
import time
from typing import Iterable, Optional

try:
    from binance import ThreadedWebsocketManager
    _BINANCE_AVAILABLE = True
except Exception:                       # python-binance missing / broken
    ThreadedWebsocketManager = None     # type: ignore[assignment]
    _BINANCE_AVAILABLE = False


# Silent-disconnect threshold: if no message has arrived in this many seconds
# the connection is treated as dead.  Binance pushes at least one ticker
# tick per second per active symbol, so 120 s of silence across every
# subscribed symbol is unambiguously a dropped connection.
_STALE_SECONDS = 120

# Per-symbol grace window: a symbol subscribed via start_symbol_ticker_socket
# should deliver its first tick within ~1 s (Binance pushes one per second).
# If we've waited this many seconds since the subscription attempt and still
# have no price, we conclude the subscription silently failed (the coroutine
# was submitted to TWM's asyncio loop but its handshake never completed —
# e.g. transient rate-limit while bulk-subscribing a hot loop) and retry.
# This is what fixed LINK/POL on bbbreakout: their _add() succeeded
# synchronously, but they never delivered a tick, and the prior code
# never retried because they were already in _symbols.
_STUCK_SUBSCRIBE_SECONDS = 10


class PriceWebSocket:
    """Live price feed for a set of Binance spot symbols.

    Not intended to be instantiated directly — use ``get_shared_ws()`` so
    every dashboard shares the same TWM and the same price dict.
    """

    def __init__(self) -> None:
        self._prices: dict[str, float] = {}
        self._lock = threading.Lock()
        self._symbols: set[str] = set()
        # Last subscription attempt timestamp, per symbol.  Used together
        # with _prices to detect symbols that "subscribed" synchronously
        # but never delivered a tick — those get re-subscribed after the
        # stuck-grace window elapses.
        self._subscribed_at: dict[str, float] = {}
        self._twm: Optional["ThreadedWebsocketManager"] = None
        self._running = False
        self._last_msg_at: float = 0.0
        self._started_at: float = 0.0
        self._start_error: Optional[str] = None
        # Guard against log-spam when the TWM emits a fatal error in a
        # tight loop (e.g. ReadLoopClosed firing on every socket read).
        self._fatal_logged = False
        # Set True by _handle_message on a fatal error.  ``get_shared_ws``
        # treats this as a signal to discard the singleton entirely
        # (force-reset) instead of attempting an in-place _stop+_start
        # rebuild — the in-place path can race the dying asyncio loop on
        # Python 3.14 and keep failing forever.
        self._needs_force_reset = False

    # ── lifecycle (caller must hold the module-level lock) ───────────────

    def _subscribe_one(self, symbol: str) -> bool:
        """Submit one start_symbol_ticker_socket call on the live TWM.

        Records the attempt timestamp in _subscribed_at regardless of
        whether the call raises — that way the retry logic in
        _ensure_subscribed throttles re-attempts on persistent failures
        rather than hammering the asyncio loop every render.

        Note: returning True only means the call didn't raise.  The
        actual handshake completes asynchronously inside the TWM thread;
        a "successful" return here can still result in zero ticks if
        the subscription silently fails downstream.  _ensure_subscribed
        catches that by retrying any symbol still missing from _prices
        after _STUCK_SUBSCRIBE_SECONDS.
        """
        try:
            self._twm.start_symbol_ticker_socket(
                callback=self._handle_message,
                symbol=symbol,
            )
        except Exception as e:
            print(f"WebSocket subscribe {symbol} failed: {e}")
            with self._lock:
                self._subscribed_at[symbol] = time.time()
            return False
        with self._lock:
            self._symbols.add(symbol)
            self._subscribed_at[symbol] = time.time()
        return True

    def _ensure_subscribed(self, symbols: Iterable[str]) -> None:
        """Subscribe each wanted symbol whose subscription needs (re-)trying.

        A symbol needs subscribing if any of the following holds:
          - we've never attempted it (not in _subscribed_at);
          - we attempted it but no tick has arrived AND the stuck-grace
            window has elapsed since the last attempt (silent failure).

        Healthy symbols (in _prices) are left alone.  Recently-attempted
        symbols still inside the grace window are also left alone so we
        don't re-fire the same subscription every render.
        """
        if not self._running or self._twm is None:
            return
        now = time.time()
        wanted = {s.upper() for s in symbols}
        with self._lock:
            ticking = set(self._prices.keys())
            attempts = dict(self._subscribed_at)
        to_subscribe: list[str] = []
        for sym in sorted(wanted):
            if sym in ticking:
                continue
            last_try = attempts.get(sym)
            if last_try is None:
                to_subscribe.append(sym)
            elif (now - last_try) >= _STUCK_SUBSCRIBE_SECONDS:
                to_subscribe.append(sym)
                print(
                    f"WebSocket retry (no tick after {now - last_try:.1f}s): "
                    f"{sym}"
                )
        for sym in to_subscribe:
            self._subscribe_one(sym)

    def _start(self, symbols: Iterable[str]) -> bool:
        if not _BINANCE_AVAILABLE:
            self._start_error = "python-binance not installed"
            return False
        self._stop()
        try:
            self._twm = ThreadedWebsocketManager(api_key="", api_secret="")
            self._twm.start()
        except Exception as e:
            print(f"PriceWebSocket start failed (TWM init): {e}")
            self._start_error = str(e)
            self._stop()
            return False

        # Fresh start — clear stale tracking so the new TWM gets clean
        # attempt timestamps for every symbol.
        with self._lock:
            self._symbols.clear()
            self._subscribed_at.clear()
            self._prices.clear()

        wanted_sorted = sorted({s.upper() for s in symbols})
        print(f"PriceWebSocket subscribing: {wanted_sorted}")
        failed: list[str] = []
        for symbol in wanted_sorted:
            if not self._subscribe_one(symbol):
                failed.append(symbol)

        # If literally every symbol failed, treat the start as a no-op
        # so is_connected stays False and a future call rebuilds.
        if not self._symbols:
            print("PriceWebSocket start: no symbols subscribed")
            self._start_error = "all symbol subscriptions failed"
            self._stop()
            return False

        self._running = True
        self._started_at = time.time()
        self._start_error = None
        self._fatal_logged = False        # reset spam-guard on fresh start
        self._needs_force_reset = False   # cleared once we're back online
        ok_list = sorted(self._symbols)
        if failed:
            print(f"WebSocket started for: {ok_list}  (failed: {sorted(failed)})")
        else:
            print(f"WebSocket started for: {ok_list}")
        return True

    def _add(self, symbols: Iterable[str]) -> None:
        """Add or re-subscribe symbols, retrying any that subscribed but
        never delivered a tick.  See _ensure_subscribed for the policy."""
        self._ensure_subscribed(symbols)

    def _stop(self) -> None:
        twm = self._twm
        self._twm = None
        self._running = False
        if twm is None:
            return
        try:
            twm.stop()
        except Exception as e:
            print(f"PriceWebSocket stop error: {e}")
        # Wait for the TWM's asyncio loop thread to actually finish
        # before returning.  Without this, a subsequent _start() races
        # the still-tearing-down loop and Python 3.14 raises
        # "This event loop is already running" — the same failure mode
        # we saw with two TWMs in the same process.  3 s is plenty;
        # python-binance's stop() schedules loop teardown immediately.
        try:
            if hasattr(twm, "join"):
                twm.join(timeout=3)
        except Exception as e:
            print(f"PriceWebSocket join error: {e}")

    def stop(self) -> None:
        with _SINGLETON_LOCK:
            self._stop()
            with self._lock:
                self._symbols.clear()

    # ── callback (runs in TWM thread) ────────────────────────────────────

    # Error types from python-binance that mean the underlying socket is
    # gone and won't recover inside this TWM instance — we have to tear
    # down and rebuild on the next get_shared_ws() call.
    _FATAL_ERROR_TYPES = frozenset({"ReadLoopClosed", "BinanceWebsocketUnableToConnect"})

    def _handle_message(self, msg) -> None:
        if not isinstance(msg, dict):
            return
        if msg.get("e") == "error":
            err_type = str(msg.get("type", ""))
            err_msg  = str(msg.get("m",    ""))
            is_fatal = (
                err_type in self._FATAL_ERROR_TYPES
                or "closed" in err_msg.lower()
            )
            if is_fatal:
                # Mark not-running so the next get_shared_ws() detects
                # the dead connection.  Log once — the same error will
                # keep firing on every read attempt (python-binance
                # also prints its own "Error receiving message" on each).
                # Additionally, set _needs_force_reset so the recovery
                # is via full singleton replacement, not an in-place
                # _stop+_start (which can race the dying asyncio loop).
                if not self._fatal_logged:
                    print(f"WebSocket fatal error: {msg} — will force-reset on next access")
                    self._fatal_logged = True
                self._running = False
                self._needs_force_reset = True
            else:
                # Non-fatal: log every occurrence but don't kill the WS.
                print(f"WebSocket error: {msg}")
            return
        symbol = msg.get("s")
        price_raw = msg.get("c")
        if not symbol or price_raw is None:
            return
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            return
        with self._lock:
            first = symbol not in self._prices
            self._prices[symbol] = price
            self._last_msg_at = time.time()
        if first:
            print(f"First price received: {symbol}={price:,.6f}")

    # ── thread-safe reads ────────────────────────────────────────────────

    def get_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._prices.get(symbol.upper())

    def get_prices(self, symbols: Iterable[str]) -> dict[str, float]:
        with self._lock:
            return {
                s: self._prices[s.upper()]
                for s in symbols
                if s.upper() in self._prices
            }

    @property
    def symbols(self) -> list[str]:
        with self._lock:
            return sorted(self._symbols)

    @property
    def start_error(self) -> Optional[str]:
        return self._start_error

    @property
    def is_connected(self) -> bool:
        """Healthy if the TWM is running, ≥1 price has been received, and
        the most recent message was within ``_STALE_SECONDS``."""
        if not self._running:
            return False
        with self._lock:
            if not self._prices:
                return False
            return (time.time() - self._last_msg_at) < _STALE_SECONDS


# ── Module-level singleton ────────────────────────────────────────────────

_SINGLETON_LOCK = threading.Lock()
_SINGLETON: Optional[PriceWebSocket] = None


def force_reset_shared_ws() -> None:
    """
    Discard the current singleton entirely so the next ``get_shared_ws()``
    call constructs a fresh ``PriceWebSocket``.

    Use this as a recovery hammer when the TWM is wedged in a state that
    in-place rebuilds (`_stop` + `_start`) can't break out of — typically
    Python 3.14's asyncio refusing to spin a new event loop because the
    previous one didn't fully tear down.  Drops cached prices and the
    subscribed-symbol set; the next caller will resubscribe from scratch.
    """
    global _SINGLETON
    with _SINGLETON_LOCK:
        old = _SINGLETON
        _SINGLETON = None
    if old is not None:
        try:
            old._stop()
        except Exception as e:
            print(f"force_reset_shared_ws: stop error ignored: {e}")
        print("force_reset_shared_ws: singleton discarded; next call will rebuild fresh")


def get_shared_ws(symbols: Iterable[str]) -> PriceWebSocket:
    """
    Return the process-wide ``PriceWebSocket`` instance, ensuring it is
    running and subscribed to ``symbols`` (union-merged with any
    pre-existing subscriptions).

    Behaviour:
      - First caller starts the TWM with ``symbols``.
      - Later callers add their symbols to the existing TWM with no
        restart (price dict and connection are preserved).
      - A silently-stale connection (no message in ``_STALE_SECONDS``)
        triggers a single in-place rebuild here.
      - A fatal error (ReadLoopClosed etc.) sets ``_needs_force_reset``
        on the singleton; the next call sees it and discards the whole
        singleton, building a fresh one — bypasses the in-place rebuild
        path which can hang on Python 3.14's asyncio teardown.
    """
    global _SINGLETON
    # Fast-path: if the previous singleton flagged itself for force
    # reset (fatal error), discard it entirely.  Done OUTSIDE the
    # singleton lock so force_reset_shared_ws()'s own locking works.
    if _SINGLETON is not None and _SINGLETON._needs_force_reset:
        force_reset_shared_ws()

    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = PriceWebSocket()

        wanted = {s.upper() for s in symbols}
        current = set(_SINGLETON._symbols)
        running = _SINGLETON._running

        # Silently-stale: rebuild once, with the full union
        stale = (
            running
            and _SINGLETON._last_msg_at > 0
            and (time.time() - _SINGLETON._last_msg_at) >= _STALE_SECONDS
        )

        if not running or stale:
            _SINGLETON._start(current | wanted)
        else:
            # Always call _ensure_subscribed (not just when wanted - current
            # is non-empty) so a previously-subscribed-but-silent symbol
            # gets retried.  _ensure_subscribed itself decides whether each
            # symbol needs a (re-)attempt based on _prices and the stuck
            # grace window — see its docstring.
            _SINGLETON._ensure_subscribed(wanted)

        return _SINGLETON
