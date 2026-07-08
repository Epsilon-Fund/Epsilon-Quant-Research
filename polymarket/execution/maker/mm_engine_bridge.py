"""Join-2a bridge: run the research MM engine's quoting code against the live venue.

This is the *machine bridge* between the research market-making backtest engine
(``polymarket/research/mm_engine/``) and the live execution stack
(``polymarket/execution/maker/``). Its single job: let the **same** ``Strategy`` +
``QueueModel`` + ``LatencyModel`` + ``BookTracker`` that ran in the JOIN-1 backtest place
real orders through the **existing** NegRisk-aware signer + venue adapter + safety harness —
**reuse, do not rebuild**.

**What is identical to the backtest** (imported unmodified from ``mm_engine``): the book
builder (:class:`~mm_engine.book.BookTracker`), the strategy's ``quote()``, the order
reconciliation (:class:`~mm_engine.orders.OrderManager`), the queue/latency models, the
telemetry schema (:class:`~mm_engine.telemetry.Telemetry`), the fee/rebate model
(:class:`~mm_engine.fees.FeeModel`), and the average-cost PnL accounting
(``mm_engine.engine._apply_fill`` / ``_Pos`` / :class:`~mm_engine.engine.EngineResult`). The
per-event quote loop in :meth:`MMEngineBridge.on_market_event` is a line-for-line mirror of
``run_engine``'s steps 2–4 (mark equity → re-quote → reconcile → queue snapshot).

**The ONLY thing that swaps is the fill path.** In the backtest, fills are realized by the
:class:`~mm_engine.fills.FillSimulator` against the recorded trade tape. Here, the
:class:`~mm_engine.orders.OrderManager`'s place/cancel/replace ops are routed to the real
venue via :class:`VenueOrderRouter` (which wraps the existing venue adapter + safety), and
fills come back from a :class:`FillSource` (the venue) and are fed into the *same* PnL +
telemetry code. Nothing in this module simulates fills.

**Venv boundary (documented choice).** The bridge runs in the **execution venv** and imports
``mm_engine`` as a *path dependency* (``polymarket/research`` inserted on ``sys.path`` by
:func:`ensure_mm_engine_importable`). Rationale: the execution side owns the irreplaceable,
independently-audited pieces — the NegRisk-aware ``ClobSigner``, the venue adapter chain
(``cli.build_venue_adapter``), the safety harness, resolution/redemption, and the journal —
none of which may be duplicated into the research venv (which has no ``py-clob-client``). The
``mm_engine`` quoting stack is pure/stdlib (``lib.clob_book`` and every leaf import cleanly
under the execution runner), so it drops in unchanged. Copying signing into ``mm_engine`` is
explicitly forbidden; this direction is the least-duplicating integration.

**DRY-RUN default.** Nothing here places real orders unless the injected venue self-reports
``is_real_venue() == True`` *and* the operator provisions real credentials + raises
``POLYMARKET_MAX_REAL_ORDERS``. The safety harness (``MAX_REAL_ORDERS`` +
``REQUIRE_OPERATOR_CONFIRM`` via the shared :class:`~.order_safety.RealOrderGate`, plus the
kill-switch and per-trade/market/deployed/daily caps reused from ``risk``) sits in the order
path. The real 1-contract live run is operator-driven (Join 2b/2c) — not this task.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


def ensure_mm_engine_importable() -> str:
    """Insert ``polymarket/research`` on ``sys.path`` so ``import mm_engine`` resolves.

    Idempotent. Returns the research root path. This file lives at
    ``polymarket/execution/maker/mm_engine_bridge.py``; the research root is three parents
    up, then ``research``. The mm_engine quoting stack (and its ``lib.clob_book`` /
    ``scripts.dali_live_clob_capture`` deps) is stdlib-only, so this is a pure path
    dependency — no code is copied across the venv boundary in either direction.
    """
    research_root = str(Path(__file__).resolve().parents[2] / "research")
    if research_root not in sys.path:
        sys.path.insert(0, research_root)
    return research_root


ensure_mm_engine_importable()

# --- mm_engine (research venv, imported unmodified as a path dependency) ---------------
from mm_engine.book import BookTracker  # noqa: E402
from mm_engine.engine import (  # noqa: E402
    EngineResult,
    _apply_fill,
    _Pos,
)
from mm_engine.events import GapMarker  # noqa: E402
from mm_engine.fees import FeeModel  # noqa: E402
from mm_engine.interfaces import (  # noqa: E402
    LatencyModel,
    MarketEvent,
    Order,
    QueueModel,
    Strategy,
)
from mm_engine.orders import ActiveOrder, OrderManager, OrderOp  # noqa: E402
from mm_engine.strategies import best_ask, best_bid, mid  # noqa: E402
from mm_engine.telemetry import Telemetry  # noqa: E402

# --- execution venv (safety + signing + journal — reused, never duplicated) ------------
from polymarket.execution.config import ExecutionConfig  # noqa: E402
from polymarket.execution.journal import (  # noqa: E402
    JsonlWriter,
    MakerFillTelemetry,
    MakerQuoteCanceled,
    MakerQuotePlaced,
    MakerQuoteSkipped,
    OrderAcknowledged,
    OrderRejected,
    OrderSubmitted,
    RiskHalt,
)
from polymarket.execution.mirror.mirror_engine import SubmitResult  # noqa: E402
from polymarket.execution.risk import (  # noqa: E402
    CandidateOrder,
    RiskState,
    check_daily_loss,
    check_deployed_cap,
    check_kill_switch,
    check_market_cap,
    check_size_cap,
)

from .event_calendar import EventCalendar  # noqa: E402
from .maker_engine import _row_ts  # noqa: E402  (reuse the robust int/float/ISO ts parser)
from .order_safety import RealOrderGate  # noqa: E402

LIVE_BRIDGE = "live_bridge"
_EPS = 1e-9


class BridgeVenue(Protocol):
    """The venue surface the bridge needs — satisfied by ``cli.build_venue_adapter`` output.

    Structurally identical to ``maker.maker_engine.MakerVenue`` (``submit_order`` +
    ``cancel_order``); the fake ``_PrintVenueAdapter`` and the real ``RealVenueAdapter`` both
    satisfy it, so the bridge routes through the exact same signing/venue path the copytrade
    and maker loops use.
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
    ) -> SubmitResult: ...

    def cancel_order(
        self,
        *,
        client_order_id: str | None = None,
        venue_order_id: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class VenueFill:
    """One fill reported by the venue for one of our orders (the live analog of a RawFill).

    ``client_order_id`` is the venue coid we submitted with (``None`` for an unmatched funder
    fill — a taker fill on an order that already left our resting set, still ours). ``ts_ms``
    is the fill's exchange timestamp (ms epoch), used for the equity/markout trajectory.
    """

    token_id: str
    side: str            # "BUY" | "SELL"
    price: float
    qty: float
    ts_ms: int
    client_order_id: str | None = None
    venue_order_id: str | None = None
    transaction_hash: str | None = None


class FillSource(Protocol):
    """Where the bridge learns about its own fills — the live replacement for the fill sim.

    ``poll`` is called once per market event with the currently-resting orders (so a
    Data-API-backed source can match funder fills to resting quotes by price/side). It
    returns the *new* fills since the last poll. In DRY-RUN/mock, the mock venue is its own
    fill source; in live ops, :class:`DataApiFillSource` polls ``data-api/trades``.
    """

    def poll(self, active_orders: list[ActiveOrder]) -> list[VenueFill]: ...


@dataclass
class BridgeConfig:
    """Bridge run parameters. Defaults to DRY-RUN, 1-contract, join-the-touch."""

    condition_id: str
    asset_id: str                    # the YES token id we quote on
    # SymmetricQuoter params — same shape mm_engine.strategies.DEFAULT_PARAMS uses.
    half_spread: float = 0.01
    size_contracts: float = 1.0
    tick: float = 0.001
    order_type: str = "GTC"
    coid_prefix: str = ""            # prepended to the OrderManager client id for venue coids
    throttle_ms: int = 0
    # periodic venue-state reconcile cadence, in market events (0 = disabled). The live loop
    # sets this so external cancels / missed acks can't silently diverge the resting set.
    reconcile_interval_events: int = 0
    # Join-2b measured submit→ack round-trip (ms) → ConstantLatency for queue telemetry.
    # 0.0 keeps the JOIN-1 zero-latency stub until the operator runs the latency harness.
    latency_ms: float = 0.0
    # HARD inventory cap in CONTRACTS (0 = disabled). The reused risk breakers bound USD
    # notional only; this caps |position| in contracts regardless of price or account
    # balance: at/above +cap the BUY side is withheld, at/below −cap the SELL side is —
    # reduce-only quoting until inventory re-enters the band. The live loop sets this.
    max_inventory_contracts: float = 0.0
    dry_run: bool = True             # informational; real submits are gated by the venue + harness

    @property
    def params(self) -> dict:
        return {"half_spread": self.half_spread, "size": self.size_contracts, "tick": self.tick}

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "BridgeConfig":
        cond = (env.get("POLYMARKET_MAKER_CONDITION_ID", "") or "").strip().lower()
        if not cond:
            raise ValueError("POLYMARKET_MAKER_CONDITION_ID is required")
        asset = (env.get("POLYMARKET_MM_BRIDGE_ASSET_ID", "") or "").strip()
        return cls(
            condition_id=cond,
            asset_id=asset,
            half_spread=float(env.get("POLYMARKET_MM_BRIDGE_HALF_SPREAD", "0.01")),
            size_contracts=float(env.get("MAKER_SIZE_CONTRACTS", "1")),
            tick=float(env.get("POLYMARKET_MM_BRIDGE_TICK", "0.001")),
            order_type=env.get("POLYMARKET_MAKER_ORDER_TYPE", "GTC").upper(),
            coid_prefix=env.get("POLYMARKET_MM_BRIDGE_COID_PREFIX", ""),
            throttle_ms=int(env.get("POLYMARKET_MM_BRIDGE_THROTTLE_MS", "0")),
            reconcile_interval_events=int(
                env.get("POLYMARKET_MM_BRIDGE_RECONCILE_EVERY", "0")
            ),
            latency_ms=float(env.get("POLYMARKET_MM_BRIDGE_LATENCY_MS", "0")),
            max_inventory_contracts=float(
                env.get("POLYMARKET_MM_BRIDGE_MAX_INVENTORY", "0")
            ),
        )


@dataclass(frozen=True)
class PlaceOutcome:
    """Result of routing one place to the venue.

    ``accepted`` is ``True`` only when the order is actually resting at the venue (the venue
    ACKed and it was not ambiguous). A safety-gate skip, a cap veto, a venue NACK, an
    ambiguous submit, or a transport exception all yield ``accepted=False`` with a ``reason``
    — the caller then rolls the intent back out of the OrderManager so it is re-proposed
    next reconcile instead of leaving a phantom resting order the venue never took.
    """

    accepted: bool
    venue_order_id: str | None = None
    reason: str | None = None


@dataclass
class VenueOrderRouter:
    """Routes OrderManager ops to the venue behind the full reused safety harness.

    Safety order (fail-fast): the reused ``risk`` breakers (kill-switch → per-trade →
    per-market → deployed → daily-loss caps, all pure functions of ``(config, state,
    order)``), then the shared :class:`RealOrderGate` (``MAX_REAL_ORDERS`` +
    ``REQUIRE_OPERATOR_CONFIRM``). The gate is a no-op on fake venues, so DRY-RUN placements
    flow freely; caps + kill-switch apply on every venue so a stray kill-switch file stops
    even a dry run. Journals the same event vocabulary the maker loop uses.
    """

    venue: BridgeVenue
    config: ExecutionConfig
    journal: JsonlWriter
    gate: RealOrderGate
    order_type: str = "GTC"
    coid_prefix: str = ""
    halted: bool = field(default=False, init=False)

    def venue_coid(self, client_id: str) -> str:
        return f"{self.coid_prefix}{client_id}"

    def _skip(self, order: Order, reason: str, detail: str, condition_id: str) -> None:
        self.journal.write(MakerQuoteSkipped(
            ts_utc=datetime.now(timezone.utc),
            condition_id=condition_id,
            asset_id=order.token_id,
            side=order.side,
            reason=reason,
            detail=detail or reason,
        ))

    def _risk_veto(self, order: Order, *, condition_id: str, state: RiskState):
        candidate = CandidateOrder(
            client_order_id="",
            condition_id=condition_id,
            asset_id=order.token_id,
            side=order.side,
            size_usd=order.price * order.size,
            leader_fill_price=order.price,   # maker: no leader; equals quote price → price_deviation n/a
        )
        for check in (
            check_kill_switch,
            check_size_cap,
            check_market_cap,
            check_deployed_cap,
            check_daily_loss,
        ):
            veto = check(self.config, state, candidate)
            if veto is not None:
                return veto
        return None

    def place(
        self,
        order: Order,
        client_id: str,
        *,
        condition_id: str,
        state: RiskState,
    ) -> PlaceOutcome:
        """Submit one resting order through safety + the venue.

        Returns a :class:`PlaceOutcome` — ``accepted=True`` only when the order is actually
        resting at the venue, so the caller can roll back a rejected/skipped intent. Mirrors
        ``MakerEngine._place_quote``'s event vocabulary and reuses the same gate, so the
        bridge and the maker loop are indistinguishable to the safety harness.
        """
        veto = self._risk_veto(order, condition_id=condition_id, state=state)
        if veto is not None:
            if veto.reason == "kill_switch":
                self.halted = True
                self.journal.write(RiskHalt(
                    ts_utc=datetime.now(timezone.utc),
                    reason="kill_switch",
                    detail=veto.detail,
                ))
            self._skip(order, veto.reason, veto.detail, condition_id)
            return PlaceOutcome(accepted=False, reason=veto.reason)

        gate_block = self.gate.check(
            venue=self.venue,
            condition_id=condition_id,
            asset_id=order.token_id,
            side=order.side,
            size=order.size,
            price=order.price,
        )
        if gate_block is not None:
            reason, detail = gate_block
            self._skip(order, reason, detail, condition_id)
            return PlaceOutcome(accepted=False, reason=reason)

        coid = self.venue_coid(client_id)
        self.journal.write(OrderSubmitted(
            ts_utc=datetime.now(timezone.utc),
            client_order_id=coid,
            condition_id=condition_id,
            asset_id=order.token_id,
            side=order.side,
            size=order.size,
            price=order.price,
            order_type=self.order_type,
        ))
        try:
            result = self.venue.submit_order(
                client_order_id=coid,
                condition_id=condition_id,
                asset_id=order.token_id,
                side=order.side,
                size_shares=order.size,
                price=order.price,
                order_type=self.order_type,
            )
        except Exception as exc:  # noqa: BLE001
            self.journal.write(OrderRejected(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                reason="mm_bridge_submit_exception",
                detail=f"{type(exc).__name__}: {exc}",
            ))
            return PlaceOutcome(accepted=False, reason="mm_bridge_submit_exception")
        if result.ambiguous:
            self.halted = True
            self.journal.write(RiskHalt(
                ts_utc=datetime.now(timezone.utc),
                reason="ambiguous_submit",
                detail=result.message or "mm bridge quote submit ambiguous",
            ))
            return PlaceOutcome(accepted=False, reason="ambiguous_submit")
        if not result.accepted:
            self.journal.write(OrderRejected(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                reason="mm_bridge_venue_rejected",
                detail=result.message or "venue rejected mm bridge quote",
            ))
            return PlaceOutcome(accepted=False, reason="mm_bridge_venue_rejected")
        if result.venue_order_id:
            self.journal.write(OrderAcknowledged(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                venue_order_id=result.venue_order_id,
            ))
        self.journal.write(MakerQuotePlaced(
            ts_utc=datetime.now(timezone.utc),
            client_order_id=coid,
            condition_id=condition_id,
            asset_id=order.token_id,
            side=order.side,
            size=order.size,
            price=order.price,
            order_type=self.order_type,
            venue_order_id=result.venue_order_id,
        ))
        return PlaceOutcome(accepted=True, venue_order_id=result.venue_order_id)

    def cancel(
        self,
        ao: ActiveOrder,
        *,
        condition_id: str,
        venue_order_id: str | None,
        reason: str,
    ) -> None:
        coid = self.venue_coid(ao.client_id)
        ambiguous = False
        try:
            result = self.venue.cancel_order(client_order_id=coid, venue_order_id=venue_order_id)
            ambiguous = bool(result.get("ambiguous", False))
        except Exception as exc:  # noqa: BLE001
            ambiguous = True
            self.journal.write(OrderRejected(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                reason="mm_bridge_cancel_exception",
                detail=f"{type(exc).__name__}: {exc}",
            ))
        self.journal.write(MakerQuoteCanceled(
            ts_utc=datetime.now(timezone.utc),
            client_order_id=coid,
            condition_id=condition_id,
            asset_id=ao.order.token_id,
            side=ao.order.side,
            price=ao.order.price,
            reason=reason,
            ambiguous=ambiguous,
        ))


class MMEngineBridge:
    """Live event loop reusing the mm_engine quoting stack; only the fill path is the venue.

    Construct with the *same* ``strategy`` / ``queue_model`` / ``latency_model`` objects the
    JOIN-1 backtest used, plus a :class:`VenueOrderRouter` wrapping the reused venue+safety.
    Drive it with :meth:`run` over any feed of ``MarketEvent | GapMarker`` (a live-shadow WS
    feed in ops; recorded frames or hand-built events in tests) and a :class:`FillSource`.
    """

    def __init__(
        self,
        *,
        strategy: Strategy,
        queue_model: QueueModel,
        latency_model: LatencyModel,
        router: VenueOrderRouter,
        bridge_config: BridgeConfig,
        journal: JsonlWriter,
        fee_model: FeeModel | None = None,
        tracker: BookTracker | None = None,
        order_manager: OrderManager | None = None,
        telemetry: Telemetry | None = None,
        event_calendar: EventCalendar | None = None,
    ) -> None:
        self.strategy = strategy
        self.queue_model = queue_model
        self.latency_model = latency_model
        self.router = router
        self.cfg = bridge_config
        self.journal = journal
        # fee_free by default: the live rebate policy is not in the L2 capture (a
        # measurement-loop unknown), and the category fallback would import pandas. This
        # keeps net_ex_rebate == net_with_rebate and matches the JOIN-1 honesty caveat.
        self.fees = fee_model if fee_model is not None else FeeModel.fee_free_model()
        self.tracker = tracker if tracker is not None else BookTracker()
        self.om = order_manager if order_manager is not None else OrderManager(
            throttle_ms=bridge_config.throttle_ms
        )
        self.telemetry = telemetry if telemetry is not None else Telemetry.in_memory()
        self.event_calendar = event_calendar

        self.positions: dict[str, _Pos] = {}
        self.last_mid: dict[str, float] = {}
        self.gross_cash = 0.0
        self.realized_pnl = 0.0
        self.rebates_earned = 0.0
        self.taker_fees_paid = 0.0
        self.equity_path: list[tuple[int, float]] = []
        self.fill_count = 0
        self.filled_qty = 0.0
        self.placed_count = 0
        self.quote_count = 0
        self.event_count = 0
        self._last_ts = 0
        # venue bookkeeping: OrderManager client_id -> resting ActiveOrder / venue_order_id
        self._client_to_active: dict[str, ActiveOrder] = {}
        self._venue_order_id: dict[str, str | None] = {}
        # fill_share_this_market = our fills / total market trades on the tape (per token).
        # Same numerator/denominator the backtest run_engine uses, so the two fills streams
        # carry a comparable fill_share. The tape = the feed's last_trade events (in live,
        # the public market-trades channel; the unfiltered data-api/trades feed is the same
        # market's trades). markout stays DERIVED downstream from mid_at_fill + the quotes mid
        # trajectory — no field here, exactly as in the backtest.
        self._market_trades: dict[str, int] = {}
        self._our_fills_by_token: dict[str, int] = {}

    # -- PnL helpers (same accounting as run_engine) ------------------------------------
    def _unrealized_total(self) -> float:
        out = 0.0
        for tok, pos in self.positions.items():
            m = self.last_mid.get(tok)
            if m is not None and abs(pos.qty) > _EPS:
                out += pos.qty * (m - pos.cost_basis)
        return out

    def _equity(self) -> float:
        return (
            self.realized_pnl
            + self._unrealized_total()
            + self.rebates_earned
            - self.taker_fees_paid
        )

    def _position_notional_usd(self, token_id: str, price: float) -> float:
        pos = self.positions.get(token_id)
        return abs(pos.qty) * price if pos is not None else 0.0

    def _deployed_usd(self) -> float:
        out = 0.0
        for tok, pos in self.positions.items():
            m = self.last_mid.get(tok, pos.cost_basis)
            out += abs(pos.qty) * (m if m is not None else pos.cost_basis)
        return out

    def _risk_state(self, token_id: str, price: float) -> RiskState:
        return RiskState(
            current_market_price=price,
            deployed_usd=self._deployed_usd(),
            deployed_in_market_usd=self._position_notional_usd(token_id, price),
            open_positions_count=sum(1 for p in self.positions.values() if abs(p.qty) > _EPS),
            realised_pnl_today_usd=self.realized_pnl,
            killswitch_present=self.router.config.killswitch_path.exists(),
        )

    # -- the event loop (steps 2–4 of run_engine; fill path swapped) --------------------
    def on_market_event(self, ev: MarketEvent) -> None:
        self.event_count += 1
        self._last_ts = ev.ts_exchange
        book = self.tracker.apply(ev)
        self.queue_model.on_event(ev, book)

        # market-trade tape counter (denominator for fill_share) — mirrors run_engine.
        if ev.type == "last_trade":
            self._market_trades[ev.token_id] = self._market_trades.get(ev.token_id, 0) + 1

        m = mid(book)
        if m is not None:
            self.last_mid[ev.token_id] = m

        # (step 2) mark net-with-rebate equity to current mids — identical to run_engine
        self.equity_path.append((ev.ts_exchange, self._equity()))

        # (step 3) re-quote and reconcile — the SAME strategy + OrderManager as the backtest
        inventory = self.positions.get(ev.token_id, _Pos()).qty
        desired = self.strategy.quote(book, inventory, self.cfg.params)
        # HARD inventory cap (contracts): withhold the side that would grow |inventory|
        # beyond the cap — reduce-only quoting until the position re-enters the band. This
        # bounds inventory independently of the USD risk caps AND of any balance read (the
        # account's cash is not visible to the bridge). No-op when the cap is 0 or slack, so
        # the fills-free orders/quotes streams stay byte-identical to the backtest's.
        desired = self._apply_inventory_cap(desired, inventory)
        ops, removed = self.om.reconcile(desired, ev.ts_exchange)

        # telemetry emitted FIRST so the orders stream is byte-identical to the backtest's,
        # then the fill-path swap routes the same ops to the venue.
        for op in ops:
            if op.op == "place":
                self.placed_count += 1
            self.telemetry.orders.emit(op.as_dict())
        for ao in removed:
            self.queue_model.forget(ao.order)

        # --- fill-path swap: drive the real venue off the reconcile result ---
        for ao in removed:  # cancels + the old side of every replace
            self.router.cancel(
                ao,
                condition_id=self.cfg.condition_id,
                venue_order_id=self._venue_order_id.pop(ao.client_id, None),
                reason="reconcile",
            )
            self._client_to_active.pop(ao.client_id, None)
        for op in ops:
            if op.op in ("place", "replace"):
                self._route_place(op)

        self.quote_count += 1

        # (step 4) per-quote queue snapshot — identical shape to run_engine's quotes stream
        snap = [
            {
                "client_id": ao.client_id,
                "side": ao.order.side,
                "price": ao.order.price,
                "size": ao.order.size,
                "remaining": ao.remaining,
                "queue_ahead": self.queue_model.get_queue_ahead(ao.order),
            }
            for ao in self.om.active_orders()
        ]
        self.telemetry.quotes.emit({
            "ts_exchange": ev.ts_exchange,
            "token_id": ev.token_id,
            "event_type": ev.type,
            "stale": book.stale,
            "best_bid": best_bid(book),
            "best_ask": best_ask(book),
            "mid": m,
            "orders": snap,
        })

    def _apply_inventory_cap(self, desired: list[Order], inventory: float) -> list[Order]:
        """Clamp quotes so a FULL fill can never push the position beyond ±cap contracts.

        With ``max_inventory_contracts <= 0`` this is the identity (cap disabled). The cap
        is a true worst-case position bound (adversarial-review fix — side-withholding
        alone lets one full-clip fill overshoot the cap by up to a clip): a BUY may rest at
        most ``cap − inventory`` contracts, a SELL at most ``cap + inventory``. A side whose
        allowance is < 1 contract is withheld entirely (journaled ``inventory_cap``, like a
        gate skip); a side whose allowance is smaller than the desired clip is size-clamped
        (visible in the orders/quotes telemetry). No-op when the cap is slack, so the
        fills-free streams stay byte-identical to the backtest's.
        """
        cap = self.cfg.max_inventory_contracts
        if cap <= 0.0 or not desired:
            return desired
        kept: list[Order] = []
        for order in desired:
            allowed = (cap - inventory) if order.side == "BUY" else (cap + inventory)
            if allowed < 1.0 - _EPS:      # no room for even one contract → withhold
                self.journal.write(MakerQuoteSkipped(
                    ts_utc=datetime.now(timezone.utc),
                    condition_id=self.cfg.condition_id,
                    asset_id=order.token_id,
                    side=order.side,
                    reason="inventory_cap",
                    detail=(
                        f"inventory {inventory:+.2f} vs hard cap ±{cap:.2f} contracts — "
                        f"{order.side} allowance {max(allowed, 0.0):.2f} < 1, side withheld"
                    ),
                ))
                continue
            if order.size > allowed + _EPS:
                order = Order(order.token_id, order.side, order.price, allowed, tag=order.tag)
            kept.append(order)
        return kept

    def _route_place(self, op: OrderOp) -> None:
        ao = self._active_for_client(op.client_id)
        if ao is None:
            return
        outcome = self.router.place(
            ao.order,
            op.client_id,
            condition_id=self.cfg.condition_id,
            state=self._risk_state(op.token_id, op.price),
        )
        if outcome.accepted:
            # resting at the venue → track it (client_id → ActiveOrder / venue_order_id).
            self._client_to_active[op.client_id] = ao
            self._venue_order_id[op.client_id] = outcome.venue_order_id
        else:
            # NOT resting at the venue (gate/cap skip, NACK, ambiguous, or exception). Roll
            # the intent back out of the OrderManager so the next reconcile RE-PROPOSES it
            # instead of treating the phantom as an idempotent no-op — the engine's resting
            # set must never claim an order the venue doesn't hold.
            dropped = self.om.drop_order(op.client_id)
            if dropped is not None:
                self.queue_model.forget(dropped.order)
            self._client_to_active.pop(op.client_id, None)
            self._venue_order_id.pop(op.client_id, None)
            # Ambiguous/exception outcomes are the one case where "rolled back in the
            # engine" and "resting at the venue" can diverge (the venue MAY hold the order
            # even though we treat it as not placed). Cancels are ungated and risk-reducing:
            # issue a best-effort cancel-by-coid so nothing rests unattended (adversarial-
            # review fix; the halt path below also runs a final venue reconcile).
            if outcome.reason in ("ambiguous_submit", "mm_bridge_submit_exception") and ao is not None:
                self.router.cancel(
                    ao,
                    condition_id=self.cfg.condition_id,
                    venue_order_id=None,
                    reason="ambiguous_rollback",
                )

    def _active_for_client(self, client_id: str) -> ActiveOrder | None:
        for ao in self.om.active_orders():
            if ao.client_id == client_id:
                return ao
        return None

    def on_gap(self, gap: GapMarker) -> None:
        self.tracker.note_gap(gap)
        ops, removed = self.om.cancel_all(self._last_ts)
        for op in ops:
            self.telemetry.orders.emit(op.as_dict())
        for ao in removed:
            self.queue_model.forget(ao.order)
            self.router.cancel(
                ao,
                condition_id=self.cfg.condition_id,
                venue_order_id=self._venue_order_id.pop(ao.client_id, None),
                reason="capture_gap",
            )
            self._client_to_active.pop(ao.client_id, None)

    # -- periodic venue-state reconcile (resting set must equal the venue's open orders) ---
    def _venue_open_coids(self) -> set[str] | None:
        """Read the venue's ACTUAL open-order client ids (read-only; submits nothing).

        Reuses the existing venue open-order read — ``reconcile_open_orders(set())`` on the
        real adapter (returns ``venue_open_client_order_ids``), or a ``get_open_orders()``
        fallback. Returns ``None`` when the venue can't report open orders (then a reconcile
        is a no-op — we never guess the venue's state).
        """
        venue = self.router.venue
        fn = getattr(venue, "reconcile_open_orders", None)
        if callable(fn):
            try:
                result = fn(set())
            except Exception:  # noqa: BLE001 — a failed read must not corrupt local state
                return None
            ids = getattr(result, "venue_open_client_order_ids", None)
            if ids is not None:
                return {str(c) for c in ids}
        fn2 = getattr(venue, "get_open_orders", None)
        if callable(fn2):
            try:
                orders = fn2()
            except Exception:  # noqa: BLE001
                return None
            out: set[str] = set()
            for o in orders or []:
                cid = o.get("client_order_id") if isinstance(o, dict) else getattr(o, "client_order_id", None)
                if cid:
                    out.add(str(cid))
            return out
        return None

    def reconcile_with_venue(self) -> dict:
        """Sync the OrderManager's resting set to the venue's ACTUAL open orders.

        Two directions, so external cancels / missed acks / partial-fills-to-zero cannot
        silently diverge the engine from reality:

        * **engine-has / venue-doesn't** → drop the order from the OrderManager (+ queue
          state + tracking) and journal a ``MakerQuoteCanceled`` (``venue_reconcile_missing``);
        * **venue-has / engine-doesn't** (orphan) → cancel it at the venue (cancels are safe,
          reduce risk) and journal ``venue_reconcile_orphan``.

        After this returns, ``{venue_coid(ao) for ao in om.active_orders()}`` equals the
        venue's open set. Returns ``{dropped, orphans_cancelled, reconciled}``; a no-op
        (``reconciled=False``) when the venue can't report open orders.
        """
        venue_coids = self._venue_open_coids()
        if venue_coids is None:
            return {"dropped": 0, "orphans_cancelled": 0, "reconciled": False}

        dropped = 0
        for ao in list(self.om.active_orders()):
            coid = self.router.venue_coid(ao.client_id)
            if coid not in venue_coids:
                self.om.drop_order(ao.client_id)
                self.queue_model.forget(ao.order)
                self._client_to_active.pop(ao.client_id, None)
                self._venue_order_id.pop(ao.client_id, None)
                self.journal.write(MakerQuoteCanceled(
                    ts_utc=datetime.now(timezone.utc),
                    client_order_id=coid,
                    condition_id=self.cfg.condition_id,
                    asset_id=ao.order.token_id,
                    side=ao.order.side,
                    price=ao.order.price,
                    reason="venue_reconcile_missing",
                    ambiguous=False,
                ))
                dropped += 1

        engine_coids = {self.router.venue_coid(ao.client_id) for ao in self.om.active_orders()}
        orphans = 0
        for coid in venue_coids - engine_coids:
            try:
                self.router.venue.cancel_order(client_order_id=coid, venue_order_id=None)
            except Exception as exc:  # noqa: BLE001
                self.journal.write(OrderRejected(
                    ts_utc=datetime.now(timezone.utc),
                    client_order_id=coid,
                    reason="mm_bridge_orphan_cancel_exception",
                    detail=f"{type(exc).__name__}: {exc}",
                ))
                continue
            self.journal.write(MakerQuoteCanceled(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                condition_id=self.cfg.condition_id,
                asset_id=self.cfg.asset_id,
                side="",
                price=0.0,
                reason="venue_reconcile_orphan",
                ambiguous=False,
            ))
            orphans += 1
        return {"dropped": dropped, "orphans_cancelled": orphans, "reconciled": True}

    # -- fill ingestion (the live replacement for FillSimulator) ------------------------
    def ingest_fill(self, fill: VenueFill) -> None:
        """Apply one venue-reported fill to the SAME PnL + telemetry code the backtest uses.

        Matches the fill to a resting order by venue coid, else by ``(side, price≈)``; an
        unmatched funder fill is still ours and is booked against a synthetic order (rank
        ``None``). ``queue_ahead`` and ``own_round_trip_ms`` are the model's *estimates* at
        fill time (telemetry only — the fill qty/price are the venue's ground truth).
        """
        ao = self._match_fill(fill)
        matched = ao is not None
        order = ao.order if ao is not None else Order(
            fill.token_id, fill.side, fill.price, fill.qty, tag="unmatched"
        )

        dq = fill.qty if order.side == "BUY" else -fill.qty
        pos = self.positions.setdefault(order.token_id, _Pos())
        realized_delta = _apply_fill(pos, dq, order.price)
        self.realized_pnl += realized_delta
        self.gross_cash += -dq * order.price
        rebate = self.fees.maker_rebate(order.token_id, fill.qty, order.price)
        self.rebates_earned += rebate
        self.fill_count += 1
        self.filled_qty += fill.qty

        sched = self.fees.schedule_for(order.token_id)
        m = self.last_mid.get(order.token_id)
        queue_ahead = self.queue_model.get_queue_ahead(order) if matched else None
        own_round_trip_ms = (
            self.latency_model.round_trip_ms(ao.placement_ts) if ao is not None else None
        )
        news = self._news_proximate(fill.ts_ms)

        # fill_share = our fills / total market trades on the tape (per token). Same ratio the
        # backtest computes. None until at least one market trade has been observed (a venue
        # fill can momentarily precede its tape print live — an accepted, documented artifact).
        self._our_fills_by_token[order.token_id] = (
            self._our_fills_by_token.get(order.token_id, 0) + 1
        )
        total_mkt = self._market_trades.get(order.token_id, 0)
        fill_share = (
            self._our_fills_by_token[order.token_id] / total_mkt if total_mkt > 0 else None
        )

        # SAME core schema as engine.run_engine's fills stream (the fields Join-2d joins on),
        # plus live-only measurement fields. See module docstring / the findings note.
        self.telemetry.fills.emit({
            "ts_exchange": fill.ts_ms,
            "token_id": order.token_id,
            "side": order.side,
            "price": order.price,
            "qty": fill.qty,
            "queue_ahead": queue_ahead,
            "mid_at_fill": m,
            "maker_fee": 0.0,
            "maker_rebate": rebate,
            "taker_fee_ref": sched.taker_fee(fill.qty, order.price),
            "fee_rate": sched.fee_rate,
            "rebate_rate": sched.rebate_rate,
            "fee_source": sched.source,
            "realized_delta": realized_delta,
            "position_after": pos.qty,
            "cost_basis_after": pos.cost_basis,
            "gross_cash_after": self.gross_cash,
            "client_id": ao.client_id if ao is not None else "",
            "trade_ts": fill.ts_ms,
            "trade_price": fill.price,
            # The backtest's trade_size is the AGGRESSOR trade's total print size (which can
            # exceed our fill qty). A venue fill reports only OUR fill, so the aggressor total
            # is unknown live — emit None rather than fill.qty, so Join-2d never treats this as
            # a comparable aggressor-size column (that would bias any qty/trade_size ratio to
            # 1.0). Our own fill size is already carried in the `qty` field.
            "trade_size": None,
            # --- live-only fields (superset; the backtest join ignores these) ---
            "source": "venue",
            "matched": matched,
            "venue_order_id": fill.venue_order_id,
            "transaction_hash": fill.transaction_hash,
            "own_round_trip_ms": own_round_trip_ms,
            "news_proximate": news,
            "fill_share_this_market": fill_share,
        })

        # execution-journal continuity (the maker loop's native telemetry event)
        self.journal.write(MakerFillTelemetry(
            ts_utc=datetime.now(timezone.utc),
            client_order_id=(
                self.router.venue_coid(ao.client_id) if ao is not None
                else (fill.client_order_id or "")
            ),
            condition_id=self.cfg.condition_id,
            asset_id=order.token_id,
            side=order.side,
            size=fill.qty,
            price=order.price,
            top_maker_rank_at_fill=None,
            post_fill_price_drift_60s=None,
            news_proximate=news,
            fill_share_this_market=fill_share,
        ))

        if ao is not None:
            ao.remaining -= fill.qty
            for dropped in self.om.drop_filled():
                self.queue_model.forget(dropped.order)
                self._client_to_active.pop(dropped.client_id, None)
                self._venue_order_id.pop(dropped.client_id, None)

    def _match_fill(self, fill: VenueFill) -> ActiveOrder | None:
        if fill.client_order_id is not None:
            # strip the coid prefix back to the OrderManager client id
            prefix = self.router.coid_prefix
            cid = fill.client_order_id
            if prefix and cid.startswith(prefix):
                cid = cid[len(prefix):]
            ao = self._client_to_active.get(cid)
            if ao is not None:
                return ao
        # fall back to (side, price within half-tick) matching, like MakerEngine
        half_tick = self.cfg.tick / 2.0
        for ao in self._client_to_active.values():
            if (
                ao.order.token_id == fill.token_id
                and ao.order.side == fill.side
                and abs(ao.order.price - fill.price) <= half_tick
            ):
                return ao
        return None

    def _news_proximate(self, ts_ms: int) -> bool | None:
        if self.event_calendar is None:
            return None
        ts = datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc)
        return self.event_calendar.is_event_proximate(ts, window_minutes=30)

    # -- driver -------------------------------------------------------------------------
    def run(
        self,
        feed: Iterable[MarketEvent | GapMarker],
        *,
        fill_source: FillSource | None = None,
    ) -> EngineResult:
        """Drive the bridge over ``feed``, polling ``fill_source`` after each event.

        Stops early if the router trips the kill-switch / an ambiguous submit halts it.
        Returns an :class:`EngineResult` — the *same* result shape the backtest returns, so
        reconciliation / Join-2d consume live and backtest runs identically.
        """
        interval = self.cfg.reconcile_interval_events
        for item in feed:
            if self.router.halted:
                break
            if isinstance(item, GapMarker):
                self.on_gap(item)
            else:
                self.on_market_event(item)
            if fill_source is not None:
                for fill in fill_source.poll(self.om.active_orders()):
                    self.ingest_fill(fill)
            # periodic venue-state reconcile: keep the resting set == the venue's open orders.
            if interval > 0 and self.event_count > 0 and self.event_count % interval == 0:
                self.reconcile_with_venue()

        # A halt (kill-switch / ambiguous submit) breaks the loop BEFORE the next periodic
        # reconcile — exactly when an orphan may rest at the venue. One final read-only
        # reconcile (cancels only, never submits) so nothing rests unattended on the way out.
        if self.router.halted:
            try:
                self.reconcile_with_venue()
            except Exception:  # noqa: BLE001 — best-effort cleanup must not mask the halt
                pass

        flush = getattr(self.queue_model, "flush_pending_cancels", None)
        if callable(flush):
            flush()
        return self.result()

    def result(self) -> EngineResult:
        return EngineResult(
            mode=LIVE_BRIDGE,
            fills=self.telemetry.fills.records,
            orders=self.telemetry.orders.records,
            quotes=self.telemetry.quotes.records,
            position={t: p.qty for t, p in self.positions.items()},
            open_positions={
                t: (p.qty, p.cost_basis)
                for t, p in self.positions.items()
                if abs(p.qty) > _EPS
            },
            last_mid=dict(self.last_mid),
            gross_cash=self.gross_cash,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self._unrealized_total(),
            rebates_earned=self.rebates_earned,
            taker_fees_paid=self.taker_fees_paid,
            equity_path=self.equity_path,
            fill_count=self.fill_count,
            filled_qty=self.filled_qty,
            placed_count=self.placed_count,
            quote_count=self.quote_count,
            event_count=self.event_count,
            l1_crosscheck=self.tracker.l1_crosscheck_summary(),
        )


@dataclass
class DataApiFillSource:
    """Live fill source: poll ``data-api/trades`` for our funder's fills, match to quotes.

    Reuses ``maker.maker_engine.DataApiTradeClient`` (the same HTTP client the maker loop
    uses) and its ``_row_ts`` parser — no venue/signer code is duplicated. Kept deliberately
    thin; the DRY-RUN/mock tests exercise the bridge via the mock venue's own fill source,
    and the operator wires this for the Join-2b/2c real run.

    Mirrors ``MakerEngine.process_fills_once``'s two guards that keep it honest:
    (1) **session filter** — ``data-api/trades`` returns the funder's recent trade *history*,
    not a post-session delta, so a row whose timestamp is before ``session_start`` is skipped
    (else the funder's prior fills would be booked as phantom live fills); and
    (2) **per-fill dedup** — the seen-set is keyed on the repo's true trade key semantics
    (``(tx, asset, side, price, size, ts)``, since ``data-api`` exposes no ``log_index``), so
    a single transaction that fills several of our resting levels emits *all* its fills, not
    just the first (see execution ``CLAUDE.md`` rule 3 / ``PLAN.md`` on multi-fill txs).
    """

    condition_id: str
    asset_id: str
    funder: str
    session_start: datetime
    data_client: Any                     # maker_engine.DataApiTradeClient (duck-typed)
    tick: float = 0.001
    _seen: set[tuple] = field(default_factory=set)

    def poll(self, active_orders: list[ActiveOrder]) -> list[VenueFill]:
        out: list[VenueFill] = []
        funder = self.funder.lower()
        half_tick = self.tick / 2.0
        for row in self.data_client.get_trades(self.condition_id):
            tx = str(row.get("transactionHash") or row.get("transaction_hash") or "")
            if not tx:
                continue
            asset_id = str(row.get("asset") or row.get("assetId") or "")
            if asset_id != self.asset_id:
                continue
            wallet = str(row.get("proxyWallet") or row.get("proxy_wallet") or "")
            if wallet.lower() != funder:
                continue
            ts = _row_ts(row)
            # session filter: skip the funder's pre-session history. Stricter than the maker
            # loop on one edge (adversarial-review fix): a row with NO parseable timestamp is
            # treated as pre-session and skipped — a genuinely-live fill always carries one,
            # while a null-ts row would bypass the filter AND collapse the dedup key to
            # ts_ms=0 for every such row.
            if ts is None or ts < self.session_start:
                continue
            price = _to_float(row.get("price"))
            size = _to_float(row.get("size"))
            side = str(row.get("side") or "").upper()
            if price is None or size is None or size <= 0 or side not in ("BUY", "SELL"):
                continue
            ts_ms = _dt_to_ms(ts)
            # composite dedup key: data-api has no log_index, so distinguish multiple fills
            # within one tx by (side, price, size, ts) — one tx can fill several levels.
            key = (tx, asset_id, side, round(price, 6), round(size, 6), ts_ms)
            if key in self._seen:
                continue
            self._seen.add(key)
            coid = None
            for ao in active_orders:
                if ao.order.side == side and abs(ao.order.price - price) <= half_tick:
                    coid = ao.client_id
                    break
            out.append(VenueFill(
                token_id=asset_id,
                side=side,
                price=price,
                qty=size,
                ts_ms=ts_ms,
                client_order_id=coid,
                transaction_hash=tx,
            ))
        return out


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dt_to_ms(ts: datetime | None) -> int:
    """Datetime → ms epoch (0 when the row carried no parseable timestamp)."""
    return int(ts.timestamp() * 1000) if ts is not None else 0
