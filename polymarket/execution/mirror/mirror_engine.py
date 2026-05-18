"""Main loop. Consumes signals, applies risk and sizing, submits via _kernel adapter.

Pipeline per signal (handle_signal):
  1. Build CandidateOrder from MirrorSignal (ENTRY → fixed-USD sizing,
     EXIT → target_size_shares × leader_fill_price as USD value).
  2. Fetch current best price from CLOB (orderbook.get_best_price).
     If unavailable, drop with reason "no_orderbook_price".
  3. Build a RiskState snapshot from current bot ledgers + kill-switch
     file presence.
  4. Run risk breakers (run_all_checks). On veto, journal RiskHalt
     (the historical event name — see PLAN.md naming reconciliation).
     The kill_switch breaker firing is the only veto that halts the
     bot; other vetos are soft skips.
  5. Compute submission price by widening current_price by
     ±price_deviation_pct (slippage tolerance — distinct from the
     guard in step 4 which compared current to leader's price).
  6. Compute target_shares = size_usd / submission_price.
  7. Submit via venue adapter. Catch any exception → OrderRejected
     reason "submit_exception", halt.
  8. Handle SubmitResult: ambiguous → AmbiguousSubmit + halt;
     not accepted → OrderRejected reason "venue_rejected" (no halt);
     accepted → OrderAcknowledged + (if immediate FOK fill info)
     FillRecorded + in-process bot_positions update.

Position math is duplicated from signal/classifier.py per PLAN.md
note: kept separate during PoC, refactor when patterns are clearer.
"""
from __future__ import annotations

import hashlib
import pathlib
import queue
import sys
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import (
    AmbiguousSubmit,
    FillRecorded,
    JsonlWriter,
    OrderAcknowledged,
    OrderRejected,
    OrderSubmitted,
    RiskHalt,
)
from polymarket.execution.risk import (
    CandidateOrder,
    RiskState,
    run_all_checks,
)
from polymarket.execution.signal import MirrorSignal, Position, SignalKind, position_key

from . import orderbook


@dataclass(frozen=True, slots=True)
class SubmitResult:
    accepted: bool
    ambiguous: bool
    venue_order_id: str | None = None
    message: str | None = None
    fill_price: float | None = None
    fill_size_shares: float | None = None


class VenueAdapter(Protocol):
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


def _make_client_order_id(signal: MirrorSignal) -> str:
    payload = (signal.signal_id + signal.condition_id + str(time.time())).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _apply_slippage(current_price: float, side: str, deviation_pct: float) -> float:
    """Widen current_price by ±deviation_pct as the limit submission price.

    BUY  → cross the spread upward (willing to pay more).
    SELL → cross downward  (willing to accept less).
    """
    pct = deviation_pct / 100.0
    if side == "BUY":
        return round(current_price * (1.0 + pct), 2)
    if side == "SELL":
        return round(current_price * (1.0 - pct), 2)
    raise ValueError(f"Unknown side: {side!r}")


class MirrorEngine:
    def __init__(
        self,
        config: ExecutionConfig,
        journal: JsonlWriter,
        venue_adapter: VenueAdapter,
        signal_queue: "queue.Queue[MirrorSignal]",
        kill_switch_path: pathlib.Path,
        today_utc: date | None = None,
    ) -> None:
        self._config: ExecutionConfig = config
        self._journal: JsonlWriter = journal
        self._venue: VenueAdapter = venue_adapter
        self._signals: queue.Queue[MirrorSignal] = signal_queue
        self._kill_switch_path: pathlib.Path = kill_switch_path
        self._bot_positions: dict[tuple[str, str], Position] = {}
        self._daily_realised_pnl_usd: float = 0.0
        self._halted: bool = False
        # Real-venue safety harness: counts every real-venue submit
        # attempt (accepted or rejected). Halts when >= config.max_real_orders.
        # Per-process; resets on restart. Has no effect for fake venues.
        self._real_attempts: int = 0
        self._rebuild_state_from_journal(today_utc)

    # ------------------------------------------------------------------
    # State rebuild
    # ------------------------------------------------------------------

    def _rebuild_state_from_journal(self, today_utc: date | None) -> None:
        today = today_utc if today_utc is not None else datetime.now(timezone.utc).date()
        for day in (today - timedelta(days=1), today):
            for event in self._journal.read_today(today_utc=day):
                if event.get("event_type") != "FILL_RECORDED":
                    continue
                self._absorb_fill(event, count_pnl=(day == today))

    def _absorb_fill(self, event: dict[str, Any], *, count_pnl: bool) -> None:
        try:
            shares = float(event.get("size", 0.0))
            price = float(event.get("price", 0.0))
        except (TypeError, ValueError):
            return
        side = event.get("side")
        condition_id = event.get("condition_id")
        asset_id = event.get("asset_id")
        if not isinstance(condition_id, str) or not isinstance(asset_id, str):
            return
        if side == "BUY":
            self._apply_buy(condition_id, asset_id, side, shares, price)
            if count_pnl:
                self._daily_realised_pnl_usd -= shares * price
        elif side == "SELL":
            self._apply_sell(condition_id, asset_id, shares)
            if count_pnl:
                self._daily_realised_pnl_usd += shares * price

    def _apply_buy(
        self,
        condition_id: str,
        asset_id: str,
        side: str,
        shares: float,
        price: float,
    ) -> None:
        key = position_key(condition_id, asset_id)
        existing = self._bot_positions.get(key)
        if existing is None:
            self._bot_positions[key] = Position(
                condition_id=condition_id.lower(),
                asset_id=asset_id,
                side=side,
                shares=shares,
                avg_entry_price=price,
                total_entry_usd=shares * price,
            )
            return
        new_shares = existing.shares + shares
        new_total = existing.total_entry_usd + (shares * price)
        new_avg = new_total / new_shares if new_shares > 0 else 0.0
        self._bot_positions[key] = replace(
            existing,
            shares=new_shares,
            avg_entry_price=new_avg,
            total_entry_usd=new_total,
        )

    def _apply_sell(self, condition_id: str, asset_id: str, shares: float) -> None:
        key = position_key(condition_id, asset_id)
        existing = self._bot_positions.get(key)
        if existing is None:
            return
        new_shares = max(0.0, existing.shares - shares)
        if new_shares == 0.0:
            del self._bot_positions[key]
            return
        new_total = existing.total_entry_usd * (new_shares / existing.shares)
        self._bot_positions[key] = replace(
            existing, shares=new_shares, total_entry_usd=new_total
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_halted(self) -> bool:
        return self._halted

    def handle_signal(self, signal: MirrorSignal) -> None:
        if self._halted:
            return

        # 1. Branch by kind. ENTRYs are dollar-sized (sizing_usd ÷ price);
        # EXITs are share-sized (target_size_shares is canonical, USD is
        # only derived for risk caps). leader_fill_price is passed
        # through to risk in both branches so the price_deviation breaker
        # answers the same question — "did the price drift between the
        # leader's fill and our submit?" — for entries and exits alike.
        if signal.kind == SignalKind.ENTRY:
            side = signal.side
        elif signal.kind == SignalKind.EXIT:
            side = "SELL"
        else:
            raise ValueError(f"Unknown signal kind: {signal.kind!r}")

        coid = _make_client_order_id(signal)

        # 2. Pricing dispatch.
        # leader_fill: submit at the leader's exact fill price; no
        #   orderbook fetch, no slippage adjustment. Suited to
        #   information-driven leaders where the price level is the
        #   signal. The price_deviation breaker becomes a no-op
        #   (state.current_market_price == leader_fill_price ⇒
        #   pct_diff is 0); that's an explicit trade-off in this mode.
        # current_book: fetch best price from CLOB and widen by
        #   ±price_deviation_pct as a slippage cap. Suited to
        #   speed-driven leaders where copying the level would be
        #   wrong because it has already moved.
        if self._config.pricing_mode == "leader_fill":
            submission_price = signal.leader_fill_price
            current_price = signal.leader_fill_price
        elif self._config.pricing_mode == "current_book":
            fetched = orderbook.get_best_price(
                self._config.clob_url, signal.asset_id, side
            )
            if fetched is None:
                self._journal.write(OrderRejected(
                    ts_utc=datetime.now(timezone.utc),
                    client_order_id=coid,
                    reason="no_orderbook_price",
                    detail=f"no best price for asset_id={signal.asset_id} side={side}",
                ))
                return
            current_price = fetched
            submission_price = _apply_slippage(
                current_price, side, self._config.price_deviation_pct
            )
        else:  # config validates; defensive guard.
            raise ValueError(
                f"Unknown pricing_mode: {self._config.pricing_mode!r}"
            )

        if submission_price <= 0:
            self._journal.write(OrderRejected(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                reason="bad_submission_price",
                detail=f"submission_price={submission_price} after slippage",
            ))
            return

        if signal.kind == SignalKind.ENTRY:
            size_usd = float(self._config.sizing_usd)
            target_shares = round(size_usd / submission_price, 4)
        else:
            target_shares = round(signal.target_size_shares, 4)
            size_usd = target_shares * submission_price  # for risk caps only

        # 4. Build candidate + risk-state snapshot
        candidate = CandidateOrder(
            client_order_id=coid,
            condition_id=signal.condition_id,
            asset_id=signal.asset_id,
            side=side,
            size_usd=size_usd,
            leader_fill_price=signal.leader_fill_price,
        )
        deployed_in_market = sum(
            p.total_entry_usd for k, p in self._bot_positions.items()
            if k[0] == signal.condition_id.lower()
        )
        state = RiskState(
            current_market_price=current_price,
            deployed_usd=sum(p.total_entry_usd for p in self._bot_positions.values()),
            deployed_in_market_usd=deployed_in_market,
            open_positions_count=len(self._bot_positions),
            realised_pnl_today_usd=self._daily_realised_pnl_usd,
            killswitch_present=self._kill_switch_path.exists(),
        )

        # 5. Run risk
        veto = run_all_checks(self._config, state, candidate)
        if veto is not None:
            self._journal.write(RiskHalt(
                ts_utc=datetime.now(timezone.utc),
                reason=veto.reason,
                detail=veto.detail,
            ))
            if veto.reason == "kill_switch":
                self._halted = True
            return

        # 6. Real-venue safety harness — only fires when the venue
        # reports is_real_venue=True. Per-order operator confirmation
        # can skip a single order without halting; max_real_orders
        # halts the bot once the per-process budget is exhausted.
        if _is_real_venue(self._venue):
            if self._real_attempts >= self._config.max_real_orders:
                self._journal.write(RiskHalt(
                    ts_utc=datetime.now(timezone.utc),
                    reason="max_real_orders",
                    detail=(
                        f"reached limit of {self._config.max_real_orders} "
                        f"real-venue submits"
                    ),
                ))
                self._halted = True
                return
            if self._config.require_operator_confirm:
                if not self._prompt_operator(
                    candidate=candidate,
                    submission_price=submission_price,
                    target_shares=target_shares,
                ):
                    self._journal.write(RiskHalt(
                        ts_utc=datetime.now(timezone.utc),
                        reason="operator_aborted",
                        detail="operator declined order via stdin",
                    ))
                    return  # per-order skip; bot is NOT halted
            # Counts ALL attempts (accepted or rejected) — even rejections
            # consume the budget. The intent is to bound exposure during
            # the first N runs, not just successful submits.
            self._real_attempts += 1

        # Journal the candidate before sending it.
        self._journal.write(OrderSubmitted(
            ts_utc=datetime.now(timezone.utc),
            client_order_id=coid,
            condition_id=signal.condition_id,
            asset_id=signal.asset_id,
            side=side,
            size=target_shares,
            price=submission_price,
            order_type=self._config.default_order_type,
        ))

        # 7. Submit
        try:
            result = self._venue.submit_order(
                client_order_id=coid,
                condition_id=signal.condition_id,
                asset_id=signal.asset_id,
                side=side,
                size_shares=target_shares,
                price=submission_price,
                order_type=self._config.default_order_type,
            )
        except Exception as exc:
            self._journal.write(OrderRejected(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                reason="submit_exception",
                detail=f"{type(exc).__name__}: {exc}",
            ))
            self._halted = True
            return

        # 8. Dispatch on result
        if result.ambiguous:
            self._journal.write(AmbiguousSubmit(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                detail=result.message or "venue returned ambiguous=True",
            ))
            self._halted = True
            return

        if not result.accepted:
            self._journal.write(OrderRejected(
                ts_utc=datetime.now(timezone.utc),
                client_order_id=coid,
                reason="venue_rejected",
                detail=result.message or "venue did not accept order",
            ))
            return

        # Accepted
        self._journal.write(OrderAcknowledged(
            ts_utc=datetime.now(timezone.utc),
            client_order_id=coid,
            venue_order_id=result.venue_order_id or "",
        ))

        # Immediate FOK fill: record + update in-process state.
        if result.fill_price is not None and result.fill_size_shares is not None:
            self._journal.write(FillRecorded(
                ts_utc=datetime.now(timezone.utc),
                # PoC: use venue_order_id as a stand-in for transaction_hash
                # until the kernel exposes the on-chain tx hash.
                transaction_hash=result.venue_order_id or coid,
                condition_id=signal.condition_id,
                asset_id=signal.asset_id,
                side=side,
                size=result.fill_size_shares,
                price=result.fill_price,
                proxy_wallet=self._config.funder,
            ))
            if side == "BUY":
                self._apply_buy(
                    signal.condition_id, signal.asset_id, side,
                    result.fill_size_shares, result.fill_price,
                )
                self._daily_realised_pnl_usd -= (
                    result.fill_size_shares * result.fill_price
                )
            elif side == "SELL":
                self._apply_sell(
                    signal.condition_id, signal.asset_id,
                    result.fill_size_shares,
                )
                self._daily_realised_pnl_usd += (
                    result.fill_size_shares * result.fill_price
                )

    # ------------------------------------------------------------------
    # Operator interaction (real-venue safety harness)
    # ------------------------------------------------------------------

    def _prompt_operator(
        self,
        *,
        candidate: CandidateOrder,
        submission_price: float,
        target_shares: float,
    ) -> bool:
        """Block on stdin for explicit per-order operator confirmation.

        Returns True iff the operator types ``yes`` (case-insensitive,
        whitespace-trimmed) and presses ENTER. Any other input — empty
        line, ``no``, EOF, etc. — returns False.

        Intended for tethered manual supervision. If
        ``POLYMARKET_REQUIRE_OPERATOR_CONFIRM=true`` is set in a
        daemon/non-interactive context, this hangs indefinitely
        waiting for stdin. Don't enable in detached runs.
        """
        print(
            f"\n[operator confirm] Submit order:\n"
            f"  market   : {candidate.condition_id}\n"
            f"  asset    : {candidate.asset_id}\n"
            f"  side     : {candidate.side}\n"
            f"  shares   : {target_shares}\n"
            f"  price    : ${submission_price:.4f}\n"
            f"  type     : {self._config.default_order_type}\n"
            f"  size_usd : ${candidate.size_usd:.2f}\n"
            f"Type 'yes' to proceed: ",
            end="", flush=True,
        )
        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            return False
        return line.strip().lower() == "yes"


def _is_real_venue(venue: Any) -> bool:
    """Duck-type check for the real-venue safety harness.

    The kwargs-style :class:`VenueAdapter` Protocol does not require
    ``is_real_venue``; we don't extend the Protocol because that would
    force every existing fake/stub in tests to implement it. Instead,
    we ask the venue if it identifies as real and treat absence as
    fake — symmetrical with the default in :class:`_PrintVenueAdapter`.
    """
    fn = getattr(venue, "is_real_venue", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 — defensive
        return False
