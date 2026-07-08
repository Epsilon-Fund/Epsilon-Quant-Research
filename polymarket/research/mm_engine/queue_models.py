"""Queue models — how a resting order advances through the queue and when it fills.

Three variants behind the frozen :class:`~mm_engine.interfaces.QueueModel` protocol, in
ascending pessimism. They share **all** mechanics — join at the back of the queue, the
book re-anchor clamp, the trade-through fill, and an **order-invariant** trade-vs-cancel
netting — and differ in exactly ONE thing: how a **cancel** (the aggregate size-decrease
left over after netting out coincident trades) is attributed between the size ahead of us
and the size behind us.

* :class:`OptimisticQueue` — every cancel is assumed **ahead** of us, so we advance by the
  full decrease. **Upper bound** on fills.
* :class:`RiskAverseQueue` — cancels are **ignored** except for the *logical floor* below;
  we advance only on real trades plus the cancel volume that provably cannot fit behind us.
  **Lower bound** on fills. (hftbacktest ``RiskAverseQueueModel``.)
* :class:`ProbQueue` — each cancel is attributed **probabilistically**: the fraction taken
  from ahead is ``front^f / (front^f + back^f)`` (hftbacktest's ``PowerProbQueueFunc``, the
  function behind ``ProbQueueModel`` / ``ProbQueueModel2``), with the power ``f`` a
  constructor knob (default 0.5). Sits **between** the two bounds.

Over any one event stream the realized fills bracket as **Optimistic ≥ Prob ≥ RiskAverse**
(more queue advance ⇒ we reach the front sooner ⇒ we fill more) — the sanity test in
``test_mm_queue_models.py``.

### Two correctness properties baked in (audit fixes)

**Logical floor (all models).** Of a cancel of size ``Δ`` at our price, at most ``back``
of it could have been resting *behind* us, so at least ``floor = max(0, Δ − back)`` MUST
have been ahead and is removed from ``queue_ahead`` regardless of model. Concretely
``advance = min(max(model_attribution, floor), Δ)``. This makes a cancel advance **fully**
when nobody is behind us (``back == 0`` ⇒ the cancel must be ahead), keeps
``queue_ahead ≤ level depth``, and preserves the bracket (``floor ≤ Prob ≤ Δ``). It is what
makes :class:`RiskAverseQueue` *logically tight* rather than naively "ignore every cancel."

**Order-invariant netting within a timestamp (FIX 3).** A trade does not mutate the book;
PM reports its depletion as a separate ``price_change`` that usually shares the trade's
``ts_exchange``. Attributing cancels *immediately* would only net the trade out if the
trade event happened to be processed before its ``price_change`` — the reverse order would
double-count the trade as a cancel. So cancel attribution is **deferred**: within a
``ts_exchange`` we only *accumulate* per ``(token, side, price)`` the net depth-decrease and
the total trade qty, and we attribute once at the timestamp boundary (when ``ts_exchange``
strictly advances) or on :meth:`flush_pending_cancels` (stream end). The cancel volume is
``max(0, net_decrease − trade_qty)``. Trade-through **fills** still happen in :meth:`fill`
the instant the trade is processed, against the pre-cancel ``queue_ahead`` — and because the
deferred cancel never touches ``queue_ahead`` before that fill, the fill (and the final
``queue_ahead``) are identical no matter which order the trade and its ``price_change``
arrive in. (Scope: net-within-timestamp; a single trade + its single depletion
``price_change`` — the real-world case, see [[mm_engine_queue_models]] — is exactly
order-invariant. Pathological multiple distinct absolute-size updates to the *same* level in
one millisecond are best-effort, matching whatever the book builder reconstructs.)

### How the three protocol methods map onto the engine's call order

The engine (``mm_engine/engine.py``) does, per market event::

    book = tracker.apply(ev)        # book is POST-event state
    queue_model.on_event(ev, book)  # (A) every event — may flush the PREVIOUS ts here
    if backtest and ev.type == "last_trade":
        ... queue_model.fill(order, book, ev) ...   # (B) only trades, AFTER on_event
    ... om.reconcile(...) / drop_filled() / cancel_all() -> queue_model.forget(removed) ...

So for a trade, ``on_event`` runs *before* ``fill`` on the **same** trade. ``on_event`` does
not advance ``queue_ahead`` on a ``last_trade`` (it only accumulates the trade qty for the
deferred netting); the trade-through advance and our overflow fill live in :meth:`fill`,
which needs the *pre-trade* ``queue_ahead``. After re-quoting, the engine calls
:meth:`forget` for every order the order-manager removed (a cancel, a replaced side, or a
fully-filled order) so a re-quote at a previously-used price re-seeds at the **back** of the
queue instead of inheriting a stale (over-optimistic) ``queue_ahead``.

The realism framing (optimistic/pessimistic bounds until our own live fills calibrate the
attribution) is [[mm_backtesting_methodology_explainer]] §1; the trade-vs-cancel data limits
are [[mm_clob_capture_semantics]]. ``calibrate(live_fills)`` is a Phase-2 stub on every
model — public anonymous L2 cannot give us our own fill rate, so the attribution is bounded
now and tuned from real fills later (Join 2).
"""
from __future__ import annotations

from mm_engine.interfaces import BookState, FillResult, MarketEvent, Order


_PRICE_DP = 6   # PM prices live on a 0.001 grid; round order/level prices to 6dp to key them


def _to_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out


def _order_key(order: Order) -> tuple[str, str, float]:
    return (order.token_id, order.side, round(float(order.price), _PRICE_DP))


def _size_at_price(book: BookState | None, side: str, price: float) -> float:
    """Resting size at ``price`` on the book side our order would join."""
    if book is None:
        return 0.0
    levels = book.bids if side == "BUY" else book.asks
    for lvl_price, lvl_size in levels:
        if abs(lvl_price - price) < 1e-9:
            return lvl_size
    return 0.0


def _resting_side(trade_side: str) -> str | None:
    """Which resting book side a trade consumes.

    A **SELL** aggressor lifts resting **bids** (our ``BUY`` orders); a **BUY** aggressor
    lifts resting **asks** (our ``SELL`` orders). PM reports the level depletion as a
    ``price_change`` whose ``side`` is that resting side, so this is the mapping that lets a
    trade be netted against the coincident ``price_change``.
    """
    if trade_side == "SELL":
        return "BUY"
    if trade_side == "BUY":
        return "SELL"
    return None


def _prob_ahead_power(front: float, back: float, n: float) -> float:
    """hftbacktest ``PowerProbQueueFunc``: P(a cancel came from ahead) = fⁿ/(fⁿ+bⁿ).

    ``front``/``back`` are the resting sizes ahead of / behind our order; ``n`` is the power
    (``ProbQueue.f``, required > 0). At ``n=1`` this is the ``front/(front+back)`` form the
    spec names; at the default ``n=0.5`` a symmetric queue (``front==back``) attributes half
    the cancel ahead. Degenerate cases: nothing ahead ⇒ 0; nothing behind ⇒ 1. Result is
    always in ``[0, 1]``. (The hard ``back==0 ⇒ advance fully`` guarantee is enforced by the
    logical floor in :meth:`_QueueBase._flush_ts`, not here.)

    Computed **scale-free** as ``1/(1+(b/a)ⁿ)`` (dividing through by the larger of ``aⁿ``,
    ``bⁿ``) so the ratio raised to the power is always ≤ 1 — ``frontⁿ`` is never formed, so
    a large ``n`` over a deep book cannot ``OverflowError`` or silently overflow to ``inf``.
    """
    if front <= 0.0:
        return 0.0
    if back <= 0.0:
        return 1.0
    if front >= back:
        r = (back / front) ** n        # ratio <= 1 -> no overflow even for large n
        return 1.0 / (1.0 + r)
    r = (front / back) ** n            # ratio <= 1
    return r / (1.0 + r)


class _QueueBase:
    """Shared queue-tracking machinery; the three models override only cancel attribution.

    Per-order state, keyed by :func:`_order_key`:

    * ``_ahead`` — ``queue_ahead``: resting size in front of our order at its price.
    * ``_depth`` — last-seen **total** resting depth at our price (the level's absolute size),
      so a ``price_change`` (which reports the new *absolute* level size) yields a signed Δ.

    Plus ``_books`` (latest :class:`BookState` per token, to seed a freshly-placed order).

    Deferred per-timestamp cancel accumulation (FIX 3 — order-invariant netting):

    * ``_ts`` — the ``ts_exchange`` currently being accumulated.
    * ``_ts_keys`` — order keys touched this ts.
    * ``_ts_start_depth`` — each touched key's level depth at the *start* of the ts.
    * ``_ts_trade`` — accumulated trade qty per key this ts.

    Nothing is attributed to ``queue_ahead`` until the ts completes (:meth:`_flush_ts`).
    """

    def __init__(self) -> None:
        self._ahead: dict[tuple[str, str, float], float] = {}
        self._depth: dict[tuple[str, str, float], float] = {}
        self._books: dict[str, BookState] = {}
        self._ts: int | None = None
        self._ts_keys: set[tuple[str, str, float]] = set()
        self._ts_start_depth: dict[tuple[str, str, float], float] = {}
        self._ts_trade: dict[tuple[str, str, float], float] = {}

    # --- the ONLY model-specific hook -------------------------------------------------
    def _cancel_attribution(self, delta_cancel: float, front: float, back: float) -> float:
        """Raw size to remove from ``queue_ahead`` for a cancel of ``delta_cancel``.

        ``front`` is our current ``queue_ahead``; ``back`` is the size resting behind us. The
        base clamps the return into ``[floor, delta_cancel]`` (see :meth:`_flush_ts`), so a
        subclass may return any non-negative value and the logical floor still holds.
        """
        raise NotImplementedError

    # --- event handling (shared) ------------------------------------------------------
    def on_event(self, ev: MarketEvent, book: BookState) -> None:
        # Cache the latest book per token so get_queue_ahead() can seed new orders.
        self._books[ev.token_id] = book

        # Timestamp boundary: finalize the buffered (now-complete) timestamp first.
        ts = ev.ts_exchange
        if self._ts is None:
            self._ts = ts
        elif ts != self._ts:
            self._flush_ts()
            self._ts = ts

        if ev.type == "book":
            self._on_book(ev.token_id, book)
        elif ev.type == "last_trade":
            self._accumulate_trade(ev)
        elif ev.type == "price_change":
            self._accumulate_pc(ev)

    def _touch(self, key: tuple[str, str, float]) -> None:
        """Register a key as active this ts and snapshot its start-of-ts level depth."""
        if key not in self._ts_keys:
            self._ts_keys.add(key)
            self._ts_start_depth[key] = self._depth.get(key, 0.0)

    def _accumulate_trade(self, ev: MarketEvent) -> None:
        trade_side = str(ev.payload.get("side") or "").upper()
        price = _to_float(ev.payload.get("price"))
        size = _to_float(ev.payload.get("size"))
        if price is None or size is None or size <= 0:
            return
        resting_side = _resting_side(trade_side)
        if resting_side is None:
            return
        key = (ev.token_id, resting_side, round(price, _PRICE_DP))
        if key not in self._ahead:
            return                       # no resting order here -> nothing to attribute
        self._touch(key)
        self._ts_trade[key] = self._ts_trade.get(key, 0.0) + size

    def _accumulate_pc(self, ev: MarketEvent) -> None:
        side = str(ev.payload.get("side") or "").upper()
        price = _to_float(ev.payload.get("price"))
        new_size = _to_float(ev.payload.get("size"))
        if side not in ("BUY", "SELL") or price is None or new_size is None:
            return
        new_size = max(0.0, new_size)
        key = (ev.token_id, side, round(price, _PRICE_DP))
        if key not in self._ahead:
            return                       # not a level we rest at -> ignore for attribution
        self._touch(key)
        self._depth[key] = new_size      # depth_end tracks the latest absolute size

    def _flush_ts(self) -> None:
        """Attribute the completed timestamp's net cancel (decrease − trade) per key."""
        for key in self._ts_keys:
            if key not in self._ahead:
                continue                  # order cancelled/filled during the ts -> drop
            depth_start = self._ts_start_depth.get(key, 0.0)
            depth_end = self._depth.get(key, depth_start)
            trade_qty = self._ts_trade.get(key, 0.0)
            net_decrease = max(0.0, depth_start - depth_end)
            cancel = max(0.0, net_decrease - trade_qty)   # trade-through is NOT a cancel
            if cancel <= 0.0:
                continue
            front = self._ahead[key]
            effective_prev = max(0.0, depth_start - trade_qty)   # level after the trade
            back = max(0.0, effective_prev - front)
            floor = max(0.0, cancel - back)                # cancel that cannot fit behind us
            raw = self._cancel_attribution(cancel, front, back)
            advance = min(max(raw, floor), cancel)
            if advance > 0.0:
                self._ahead[key] = max(0.0, front - advance)
        self._ts_keys.clear()
        self._ts_start_depth.clear()
        self._ts_trade.clear()

    def flush_pending_cancels(self) -> None:
        """Finalize the in-progress timestamp's deferred cancel attribution (stream end)."""
        self._flush_ts()

    def _on_book(self, token_id: str, book: BookState) -> None:
        """Re-anchor every resting order on ``token_id`` to a full ``book`` snapshot.

        A snapshot is an absolute re-sync, so it **supersedes** any incremental guess: drop
        this token's pending per-ts accumulation, then clamp ``queue_ahead`` DOWN to the
        observed depth (never raise it — size beyond our belief joined behind us) and reset
        the depth baseline.
        """
        for key in [k for k in self._ts_keys if k[0] == token_id]:
            self._ts_keys.discard(key)
            self._ts_start_depth.pop(key, None)
            self._ts_trade.pop(key, None)
        for key, ahead in self._ahead.items():
            if key[0] != token_id:
                continue
            _, side, price = key
            depth = _size_at_price(book, side, price)
            if depth < ahead:
                self._ahead[key] = depth
            self._depth[key] = depth

    # --- queue position + fill (shared) -----------------------------------------------
    def _ensure(self, order: Order, book: BookState | None = None) -> tuple[str, str, float]:
        """Seed an order's queue state on first sight: it joins at the BACK of its level."""
        key = _order_key(order)
        if key not in self._ahead:
            ref = book if book is not None else self._books.get(order.token_id)
            depth = _size_at_price(ref, order.side, order.price)
            self._ahead[key] = depth
            self._depth[key] = depth     # baseline for delta tracking
        return key

    def get_queue_ahead(self, our_order: Order) -> float:
        return self._ahead[self._ensure(our_order)]

    def fill(self, our_order: Order, book: BookState, trade: MarketEvent | None) -> FillResult:
        """Trade-through fill — identical across all three models.

        The trade first consumes the size ahead of us; any **overflow** beyond
        ``queue_ahead`` fills our order, up to its size. ``queue_ahead`` is decremented by
        what the trade consumed of it. Latency gating is the engine's job (``fills.py``), not
        ours; the backtest book already excludes our own order. This reads the *pre-cancel*
        ``queue_ahead`` (deferred cancels for the current ts have not been attributed yet),
        which is what makes the trade-vs-cancel netting order-invariant.
        """
        key = self._ensure(our_order, book)
        if trade is None or trade.type != "last_trade":
            return FillResult(qty=0.0, queue_ahead=self._ahead[key])

        trade_side = str(trade.payload.get("side") or "").upper()
        trade_price = _to_float(trade.payload.get("price"))
        trade_size = _to_float(trade.payload.get("size"))
        if (
            trade_price is None
            or trade_size is None
            or trade_size <= 0
            or not self._eligible(our_order, trade_side, trade_price)
        ):
            return FillResult(qty=0.0, queue_ahead=self._ahead[key])

        ahead = self._ahead[key]
        consumed_ahead = min(ahead, trade_size)
        self._ahead[key] = ahead - consumed_ahead
        overflow = trade_size - consumed_ahead
        qty = min(overflow, float(our_order.size)) if overflow > 0 else 0.0
        return FillResult(qty=qty, queue_ahead=self._ahead[key])

    def forget(self, our_order: Order) -> None:
        """Drop per-order queue state (called when an order is cancelled/replaced/filled).

        The next order at the same ``(token, side, price)`` then re-seeds at the back of the
        queue rather than inheriting this order's (possibly fully-consumed) ``queue_ahead``.
        """
        key = _order_key(our_order)
        self._ahead.pop(key, None)
        self._depth.pop(key, None)
        self._ts_keys.discard(key)
        self._ts_start_depth.pop(key, None)
        self._ts_trade.pop(key, None)

    def calibrate(self, live_fills) -> None:
        # Phase-2 stub: real calibration tunes the cancel attribution from our own live
        # fills — the one thing public anonymous L2 can never give us offline.
        return None

    @staticmethod
    def _eligible(order: Order, trade_side: str, trade_price: float) -> bool:
        # our BUY (resting bid) fills from a SELL trade at price <= our price;
        # our SELL (resting ask) fills from a BUY  trade at price >= our price.
        if order.side == "BUY":
            return trade_side == "SELL" and trade_price <= order.price + 1e-9
        if order.side == "SELL":
            return trade_side == "BUY" and trade_price >= order.price - 1e-9
        return False


class OptimisticQueue(_QueueBase):
    """Optimistic bound: every cancel is assumed to be **ahead** of us.

    ``queue_ahead -= Δ`` on each cancel, so we reach the front as fast as the data allows.
    Combined with the shared trade-through fill, this is the **upper bound** on fills — the
    "once your price is reached you fill" optimism the methodology explainer warns flatters a
    backtest, kept here deliberately as the optimistic end of the bracket.
    """

    def _cancel_attribution(self, delta_cancel: float, front: float, back: float) -> float:
        return delta_cancel


class RiskAverseQueue(_QueueBase):
    """Pessimistic bound: advance only on real trades plus the logical floor.

    hftbacktest's ``RiskAverseQueueModel``. The raw attribution is 0 — cancels are assumed to
    be behind us — but the base then clamps up to ``floor = max(0, Δ − back)``, the cancel
    volume that *cannot* fit behind us. So ``queue_ahead`` advances by 0 when there is room
    behind us (``back ≥ Δ``) and by exactly ``Δ − back`` when there is not. This is the
    **lower bound** on fills while staying logically consistent (``queue_ahead ≤ depth``).
    """

    def _cancel_attribution(self, delta_cancel: float, front: float, back: float) -> float:
        return 0.0


class ProbQueue(_QueueBase):
    """Probabilistic queue (hftbacktest ``ProbQueueModel`` / ``ProbQueueModel2`` power-law).

    Each cancel is split ahead-vs-behind: the fraction taken from the size ahead of us is
    ``front^f / (front^f + back^f)`` (:func:`_prob_ahead_power`), so the raw attribution is
    ``prob_ahead · Δ`` and the base clamps it into ``[floor, Δ]``. The power ``f`` (default
    0.5) is the only knob; at ``f=1`` the probability is the plain ``front/(front+back)``
    ratio the spec names. Because the clamped attribution lies in ``[floor, Δ]``, ProbQueue
    always lands between :class:`OptimisticQueue` and :class:`RiskAverseQueue`.

    ``f`` is meant to be tuned from our own live fills; until then it is a fixed assumption
    and results should be read as the *middle* of an optimistic/pessimistic bound, not a
    point estimate. :meth:`calibrate` is the Phase-2 hook for that fit.
    """

    def __init__(self, f: float = 0.5) -> None:
        super().__init__()
        f = float(f)
        if not (f > 0.0):   # power-law exponent must be positive for a monotone attribution
            raise ValueError(f"ProbQueue.f must be > 0, got {f}")
        self.f = f

    def _cancel_attribution(self, delta_cancel: float, front: float, back: float) -> float:
        prob_ahead = _prob_ahead_power(front, back, self.f)
        return prob_ahead * delta_cancel

    def calibrate(self, live_fills) -> None:
        # Phase-2 stub: fit `self.f` to minimize the gap between modeled and observed fill
        # rates on our own live fills (Join 2). No-op until those fills exist.
        return None
