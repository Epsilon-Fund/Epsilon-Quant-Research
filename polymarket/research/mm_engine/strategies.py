"""Strategies for the MM engine.

Phase 0 shipped the :class:`SymmetricQuoter` (the 0-parameter A/B baseline). Task 5 adds the
inventory-managed quoters, all conforming to the frozen
:class:`~mm_engine.interfaces.Strategy` protocol (``quote(book, inventory, params)``) with
**every knob and the τ anchor injected via** ``params`` — the interface is untouched:

* :class:`InventoryAwareQuoter` — v1: linear skew around the **microprice**
  (``r = microprice − k·q``), a tight position cap with one-sided reduce-only quoting at cap,
  a near-expiry flatten (τ from ``params['end_date_ms']`` vs ``book.ts_exchange``), and a
  separable, individually-toggleable toxicity gate (velocity / book imbalance / depth
  evaporation — the strategy-observable subset of the v0-certified signals).
* :class:`ASQuoter` — the Avellaneda-Stoikov build-up ladder. Rung 1 replaces the hand-tuned
  ``k`` with the derived slope ``γσ²τ`` (σ = causal EWMA of mid-change variance); rung 2 adds
  the A-S optimal half-spread ``γσ²τ/2 + (1/γ)·ln(1+γ/k_arr)`` (``k_arr`` IS-calibrated by
  the runner, injected via params); rung 3 = rung 2 + the toxicity overlay.
  **Sign warning honored:** the A-S τ-skew *shrinks* toward expiry, while our near-expiry
  toxicity *grows* — so the flatten rule is a separate overlay (``pull_hours``), never wired
  through the same τ term.
* :class:`BasketCarryQuoter` — the v2 *alternative* to flattening: joint quoting across a
  NegRisk event's legs, skewing each leg toward a balanced basket (netting complementary
  same-condition tokens), carrying inventory through expiry instead of pulling.

All internal state is derived from the event stream only (deterministic replay).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from mm_engine.interfaces import BookState, Order

_LN2 = math.log(2.0)
CENTS_PER_PRICE = 100.0   # 1 price unit = 100 cents (chase-rate knob is quoted in ¢/s)


def best_bid(book: BookState) -> float | None:
    return book.bids[0][0] if book.bids else None


def best_ask(book: BookState) -> float | None:
    return book.asks[0][0] if book.asks else None


def mid(book: BookState) -> float | None:
    bid = best_bid(book)
    ask = best_ask(book)
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


# Defaults: 1c half-spread, 100-contract clips, PM's 0.001 price tick.
DEFAULT_PARAMS = {"half_spread": 0.01, "size": 100.0, "tick": 0.001}


@dataclass
class SymmetricQuoter:
    """Fixed-width, inventory-agnostic two-sided quoter (the A-S A/B baseline).

    Quotes ``mid ± half_spread`` with a constant clip size, ignoring inventory entirely.
    Returns no orders when the book is stale or has no two-sided mid — a strategy must
    never quote off a stale book. Prices are rounded to ``tick`` and clamped to the open
    interval ``(0, 1)`` so a quote is always a valid resting price.

    Example: with ``half_spread=0.01`` and a book of best bid 0.47 / best ask 0.49
    (mid 0.48), it emits BUY 0.47 and SELL 0.49 at the configured size. If the book is
    marked stale (gap, or >5s since the last depth update), it emits nothing.
    """

    name: str = "symmetric"

    def quote(self, book: BookState, inventory: float, params: dict) -> list[Order]:
        if book.stale:
            return []
        m = mid(book)
        if m is None:
            return []

        cfg = {**DEFAULT_PARAMS, **(params or {})}
        half_spread = float(cfg["half_spread"])
        size = float(cfg["size"])
        tick = float(cfg["tick"])

        bid_price = self._round_clamp(m - half_spread, tick)
        ask_price = self._round_clamp(m + half_spread, tick)
        if bid_price is None or ask_price is None or bid_price >= ask_price:
            return []

        return [
            Order(book.token_id, "BUY", bid_price, size, tag=self.name),
            Order(book.token_id, "SELL", ask_price, size, tag=self.name),
        ]

    @staticmethod
    def _round_clamp(price: float, tick: float) -> float | None:
        return _round_clamp(price, tick)


def _round_clamp(price: float, tick: float) -> float | None:
    if tick > 0:
        price = round(round(price / tick) * tick, 10)
    lo, hi = (tick if tick > 0 else 0.0), 1.0 - (tick if tick > 0 else 0.0)
    if price <= 0.0 or price >= 1.0:
        return None
    return min(max(price, lo), hi)


def microprice(book: BookState) -> float | None:
    """Size-weighted mid: ``(bid·ask_size + ask·bid_size)/(bid_size+ask_size)``.

    The 0-parameter, imbalance-aware fair value the Task-5 quoters center on (a bid-heavy
    book pushes the microprice toward the ask — the direction the next move leans). Falls
    back to the raw mid when top-of-book sizes are missing or degenerate.
    """
    if not book.bids or not book.asks:
        return None
    bid_p, bid_s = book.bids[0]
    ask_p, ask_s = book.asks[0]
    tot = bid_s + ask_s
    if tot <= 0:
        return (bid_p + ask_p) / 2.0
    return (bid_p * ask_s + ask_p * bid_s) / tot


def tau_hours(book: BookState, params: dict) -> float:
    """Time-to-resolution in hours, from the runner-injected ``end_date_ms`` anchor.

    The frozen ``BookState`` carries no τ; the eval runner knows each market's Gamma
    ``end_date`` and injects it via ``params`` — the strategy derives the per-event τ from
    the book's own ``ts_exchange`` (lookahead-free). Missing anchor → +inf (no τ behavior).
    """
    end = params.get("end_date_ms")
    if end is None:
        return float("inf")
    return (float(end) - book.ts_exchange) / 3.6e6


# Fixed (non-knob) toxicity-gate constants — declared, v0-grounded, NOT tuned per config.
# The knob is the SUBSET of enabled signals, not these thresholds.
TOX_DEFAULTS = {
    "vel_window_s": 10.0,        # microprice-velocity lookback
    "vel_thresh": 0.005,         # price units per window (= 0.5 cents / 10 s)
    "imb_thresh": 0.85,          # |top-of-book imbalance| beyond which we pull
    "depth_ewma_alpha": 0.01,    # per-event EWMA for the trailing-depth reference (~100-event memory)
    "depth_evap_thresh": 0.4,    # current depth / trailing EWMA below this = evaporation
}


@dataclass
class _ToxicityState:
    """Causal, event-stream-only signal state shared by the Task-5 quoters.

    Hot-path design: the mid history keeps only the velocity window (+1 anchor entry just
    older than it), pruned amortized-O(1) per event; the depth reference is a per-event EWMA
    (declared constant, not a knob) instead of a rolling median.
    """

    mid_hist: deque = field(default_factory=deque)   # (ts_ms, microprice), window + 1 anchor
    _depth_ewma: float = 0.0
    _depth_n: int = 0

    def update(self, ts: int, micro: float | None, depth: float | None,
               window_ms: float, depth_alpha: float) -> None:
        if micro is not None:
            self.mid_hist.append((ts, micro))
            cutoff = ts - window_ms
            h = self.mid_hist
            while len(h) >= 2 and h[1][0] <= cutoff:
                h.popleft()
        if depth is not None and depth > 0:
            if self._depth_n == 0:
                self._depth_ewma = depth
            else:
                self._depth_ewma += depth_alpha * (depth - self._depth_ewma)
            self._depth_n += 1

    def velocity(self) -> float:
        """|Δmicroprice| across the retained window (price units). 0 until history exists."""
        h = self.mid_hist
        if len(h) < 2:
            return 0.0
        return abs(h[-1][1] - h[0][1])

    def depth_ratio(self, current: float | None) -> float:
        """Current top depth / trailing EWMA. 1.0 until warmed (≥20 depth observations)."""
        if current is None or current <= 0 or self._depth_n < 20 or self._depth_ewma <= 0:
            return 1.0
        return current / self._depth_ewma


def _tox_triggered(state: _ToxicityState, book: BookState, cfg: dict) -> bool:
    """Any ENABLED toxicity signal firing? Signals are separable (each its own toggle).

    ``cfg`` is the pre-merged per-run config (see ``_parse_cfg``) — no per-event dict work.
    """
    if cfg["tox_velocity"] and state.velocity() > cfg["vel_thresh"]:
        return True
    if cfg["tox_imbalance"] and book.bids and book.asks:
        bs, as_ = book.bids[0][1], book.asks[0][1]
        tot = bs + as_
        if tot > 0 and abs((bs - as_) / tot) > cfg["imb_thresh"]:
            return True
    if cfg["tox_depth"]:
        depth = ((book.bids[0][1] + book.asks[0][1]) / 2.0) if (book.bids and book.asks) else None
        if state.depth_ratio(depth) < cfg["depth_evap_thresh"]:
            return True
    return False


# Per-run parsed-config cache: `params` is one stable dict per engine run, so each strategy
# instance parses it once and reuses the result on every event (the merge was a hot-path cost).
def _parse_cfg(params: dict) -> dict:
    cfg = {**DEFAULT_PARAMS, **TOX_DEFAULTS, **(params or {})}
    cfg["half_spread"] = float(cfg["half_spread"])
    cfg["size"] = float(cfg["size"])
    cfg["tick"] = float(cfg["tick"])
    cfg["skew_k"] = float(cfg.get("skew_k", 0.0))
    cfg["inv_cap"] = float(cfg.get("inv_cap", float("inf")))
    cfg["pull_hours"] = float(cfg.get("pull_hours", 0.0))
    cfg["as_gamma"] = float(cfg.get("as_gamma", 1e-5))
    cfg["tox_velocity"] = bool(cfg.get("tox_velocity", False))
    cfg["tox_imbalance"] = bool(cfg.get("tox_imbalance", False))
    cfg["tox_depth"] = bool(cfg.get("tox_depth", False))
    cfg["_vel_window_ms"] = float(cfg["vel_window_s"]) * 1000.0
    return cfg


def _reduce_only(book: BookState, inventory: float, size: float, tick: float,
                 tag: str, at_touch: bool, reservation: float | None = None,
                 half_spread: float = 0.0) -> list[Order]:
    """One-sided quote that only reduces |inventory| (flatten / at-cap behavior).

    ``at_touch=True`` joins the touch on the reducing side (the near-expiry flatten —
    passive but at the front price). Otherwise rests at ``reservation ± half_spread``.
    """
    if abs(inventory) < 1e-9:
        return []
    clip = min(size, abs(inventory))
    if inventory > 0:      # long -> SELL to reduce
        px = best_ask(book) if at_touch else (reservation + half_spread)
        side = "SELL"
    else:                  # short -> BUY to reduce
        px = best_bid(book) if at_touch else (reservation - half_spread)
        side = "BUY"
    if px is None:
        return []
    px = _round_clamp(px, tick)
    if px is None:
        return []
    return [Order(book.token_id, side, px, clip, tag=tag)]


@dataclass
class InventoryAwareQuoter:
    """Task-5 v1: microprice-centered linear inventory skew + tight cap + τ-flatten + toxicity gate.

    Knobs (all via ``params``; ≤4 per the PRD budget):

    1. ``skew_k`` — reservation shift per contract of inventory: ``r = microprice − k·q``.
    2. ``inv_cap`` — tight position cap; **at/above cap quoting goes one-sided** (reduce-only).
    3. ``pull_hours`` — near-expiry flatten: for τ below this, stop opening quotes and rest a
       reduce-only quote AT the touch until flat (both terminal tails deliberately clipped —
       the edge is the spread, not the resolution coin-flip).
    4. ``tox_velocity`` / ``tox_imbalance`` / ``tox_depth`` — the separable toxicity gate
       (subset selection is the knob; thresholds are fixed declared constants). Triggered →
       pull opening quotes (reduce-only quote stays if we hold inventory).

    Example: microprice 0.480, ``skew_k=2e-5``, ``q=+400`` → r = 0.472; with a 0.5¢
    half-spread the quoter rests BUY 0.467 / SELL 0.477 — both shifted down, selling
    inventory into strength and backing the bid away.
    """

    name: str = "inventory_v1"
    _tox: _ToxicityState = field(default_factory=_ToxicityState)
    _cfg_key: int | None = None
    _cfg: dict | None = None

    def quote(self, book: BookState, inventory: float, params: dict) -> list[Order]:
        if book.stale:
            return []
        micro = microprice(book)
        if micro is None:
            return []
        if self._cfg_key != id(params):
            self._cfg = _parse_cfg(params)
            self._cfg_key = id(params)
        cfg = self._cfg
        half_spread = cfg["half_spread"]
        size = cfg["size"]
        tick = cfg["tick"]
        k = cfg["skew_k"]
        cap = cfg["inv_cap"]
        pull_h = cfg["pull_hours"]

        depth = ((book.bids[0][1] + book.asks[0][1]) / 2.0) if (book.bids and book.asks) else None
        self._tox.update(book.ts_exchange, micro, depth,
                         cfg["_vel_window_ms"], cfg["depth_ewma_alpha"])

        # near-expiry flatten: no opening quotes, reduce-only AT the touch until flat
        if tau_hours(book, cfg) < pull_h:
            return _reduce_only(book, inventory, size, tick, self.name, at_touch=True)

        r = micro - k * inventory
        toxic = _tox_triggered(self._tox, book, cfg)

        if toxic or abs(inventory) >= cap:
            # pull opening quotes; keep only the reducing side (resting at r ± half_spread)
            return _reduce_only(book, inventory, size, tick, self.name, at_touch=False,
                                reservation=r, half_spread=half_spread)

        bid = _round_clamp(r - half_spread, tick)
        ask = _round_clamp(r + half_spread, tick)
        if bid is None or ask is None or bid >= ask:
            return []
        return [
            Order(book.token_id, "BUY", bid, size, tag=self.name),
            Order(book.token_id, "SELL", ask, size, tag=self.name),
        ]


# A-S modeling constants (declared, not tuned): τ is capped so the γσ²τ term reflects a
# maker's risk horizon rather than a 6-month calendar residual (Dec-2026 outrights would
# otherwise dominate the skew); σ uses a fixed 1h EWMA half-life.
AS_TAU_CAP_H = 168.0
AS_SIGMA_HALFLIFE_S = 3600.0
AS_MAX_HALF_SPREAD = 0.05


@dataclass
class _SigmaEWMA:
    """Causal per-hour variance of microprice changes (EWMA over event-time)."""

    halflife_s: float = AS_SIGMA_HALFLIFE_S
    _last: tuple[int, float] | None = None
    _var_per_h: float = 0.0
    _warm: int = 0

    def update(self, ts: int, micro: float) -> None:
        if self._last is not None:
            dt_ms = ts - self._last[0]
            if dt_ms > 0:
                inst = (micro - self._last[1]) ** 2 / dt_ms * 3.6e6   # price²/hour
                alpha = 1.0 - math.exp(-_LN2 * (dt_ms / 1000.0) / self.halflife_s)
                self._var_per_h += alpha * (inst - self._var_per_h)
                self._warm += 1
        self._last = (ts, micro)

    @property
    def var_per_h(self) -> float:
        return self._var_per_h if self._warm >= 30 else 0.0


@dataclass
class ASQuoter:
    """A-S build-up ladder (rungs 1–3) — same cap/flatten overlays as v1, derived skew/spread.

    * Rung 1 (``as_use_spread=False``, tox off): ``r = microprice − q·γσ²τ`` — the hand-tuned
      ``k`` replaced by the A-S slope. σ² = causal EWMA (price²/h); τ = capped time-to-
      resolution (hours). γ (``as_gamma``) is the knob.
    * Rung 2 (``as_use_spread=True``): half-spread becomes the A-S optimum
      ``γσ²τ/2 + (1/γ)·ln(1 + γ/k_arr)`` with ``as_k_arr`` (arrival-decay) IS-calibrated by
      the runner per token.
    * Rung 3: rung 2 + the toxicity overlay (``tox_*`` toggles — the piece A-S omits).

    The near-expiry flatten (``pull_hours``) stays a SEPARATE overlay with the opposite sign
    to the A-S τ term, per the design decision (they must never share the τ wire).
    """

    name: str = "as_ladder"
    _tox: _ToxicityState = field(default_factory=_ToxicityState)
    _sigma: _SigmaEWMA = field(default_factory=_SigmaEWMA)
    _cfg_key: int | None = None
    _cfg: dict | None = None

    def quote(self, book: BookState, inventory: float, params: dict) -> list[Order]:
        if book.stale:
            return []
        micro = microprice(book)
        if micro is None:
            return []
        if self._cfg_key != id(params):
            self._cfg = _parse_cfg(params)
            self._cfg_key = id(params)
        cfg = self._cfg
        size = cfg["size"]
        tick = cfg["tick"]
        gamma = cfg["as_gamma"]
        cap = cfg["inv_cap"]
        pull_h = cfg["pull_hours"]

        depth = ((book.bids[0][1] + book.asks[0][1]) / 2.0) if (book.bids and book.asks) else None
        self._tox.update(book.ts_exchange, micro, depth,
                         cfg["_vel_window_ms"], cfg["depth_ewma_alpha"])
        self._sigma.update(book.ts_exchange, micro)

        if tau_hours(book, cfg) < pull_h:
            return _reduce_only(book, inventory, size, tick, self.name, at_touch=True)

        tau = min(max(tau_hours(book, cfg), 0.0), AS_TAU_CAP_H)
        var_h = self._sigma.var_per_h
        r = micro - inventory * gamma * var_h * tau

        if cfg.get("as_use_spread"):
            k_arr = max(float(cfg.get("as_k_arr", 50.0)), 1e-9)
            half_spread = gamma * var_h * tau / 2.0 + (1.0 / gamma) * math.log1p(gamma / k_arr)
            half_spread = min(max(half_spread, tick), AS_MAX_HALF_SPREAD)
        else:
            half_spread = cfg["half_spread"]

        toxic = _tox_triggered(self._tox, book, cfg)
        if toxic or abs(inventory) >= cap:
            return _reduce_only(book, inventory, size, tick, self.name, at_touch=False,
                                reservation=r, half_spread=half_spread)

        bid = _round_clamp(r - half_spread, tick)
        ask = _round_clamp(r + half_spread, tick)
        if bid is None or ask is None or bid >= ask:
            return []
        return [
            Order(book.token_id, "BUY", bid, size, tag=self.name),
            Order(book.token_id, "SELL", ask, size, tag=self.name),
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Task 5.1 — NeutralSpikeQuoter (LOTECH-grounded neutral spike avoidance)
# ──────────────────────────────────────────────────────────────────────────────

# Declared (non-knob) constants of the NeutralSpikeQuoter. The TUNED knobs are only
# ``skew_k``, ``inv_cap``, ``nsq_vpin_window`` and ``nsq_damp_coeff`` (+ the component
# toggles, which are ladder RUNGS, not tuned values). Everything below is a declared
# statistical/structural constant per the Task-5.1 PRD §5 knob discipline — the AS
# z-threshold (−2) in particular is standard, not fitted.
NSQ_DEFAULTS = {
    # Lens 1 — VPIN volume clock + LOTECH sweep weighting
    "nsq_vpin_window": 20,        # TUNED: buckets averaged into the VPIN reading
    "nsq_bucket_trades": 25.0,    # bucket volume = EWMA typical trade size × this
    "nsq_size_ewma_alpha": 0.02,  # EWMA for the typical trade size (session-relative clock)
    "nsq_sweep_lambda": 0.5,      # sweep weight = exp(min(dist_ticks, cap) · λ)
    "nsq_sweep_cap_ticks": 6.0,   # bounded LOTECH exp(dist/tick) — weight ≤ e^3 ≈ 20×
    "nsq_vpin_q": 0.90,           # session-relative elevated band: VPIN ≥ its own q-quantile
    "nsq_vpin_floor": 0.30,       # absolute floor under the literature 0.4 "elevated" band
    "nsq_vpin_min_buckets": 30,   # session warm-up before Lens 1 may fire
    "nsq_vpin_hist_max": 500,     # session history window for the quantile band
    # Lens 2 — adverse-selection z-score (post-fill drift vs calm baseline)
    "nsq_as_horizon_s": 30.0,     # post-fill drift horizon (matches the eval's primary markout)
    "nsq_as_baseline_n": 50,      # rolling calm-baseline window (# matured drifts)
    "nsq_as_min_fills": 10,       # baseline warm-up before Lens 2 may fire
    "nsq_as_z_thresh": -2.0,      # STATISTICAL constant (LOTECH Lens 2), never tuned
    "nsq_as_flag_decay_s": 120.0, # as_flag persists this long past the last bad drift
    # graduated response + unwind
    "nsq_as_size_factor": 0.5,    # adverse-only: reduce opening size to this fraction
    "nsq_widen_mult": 2.0,        # both-lenses: reduce-only rests at r ± widen·half_spread
    # asymmetric repricing (slow to chase, fast to withdraw)
    "nsq_chase_rate_c_s": 0.5,    # max chase speed, cents per second
    # OFI size-dampening
    "nsq_damp_coeff": 0.0,        # TUNED when the rung is on (0 = off)
    "nsq_ofi_halflife_s": 60.0,   # EWMA half-life for the OFI state variable
    "nsq_size_floor": 0.2,        # dampened size never below this fraction of the clip
}


def _parse_nsq_cfg(params: dict) -> dict:
    cfg = {**DEFAULT_PARAMS, **NSQ_DEFAULTS, **(params or {})}
    cfg["half_spread"] = float(cfg["half_spread"])
    cfg["size"] = float(cfg["size"])
    cfg["tick"] = float(cfg["tick"])
    cfg["skew_k"] = float(cfg.get("skew_k", 0.0))
    cfg["inv_cap"] = float(cfg.get("inv_cap", float("inf")))
    cfg["nsq_lens1"] = bool(cfg.get("nsq_lens1", False))
    cfg["nsq_lens2"] = bool(cfg.get("nsq_lens2", False))
    cfg["nsq_asym"] = bool(cfg.get("nsq_asym", False))
    cfg["nsq_vpin_window"] = int(cfg["nsq_vpin_window"])
    cfg["nsq_damp_coeff"] = float(cfg["nsq_damp_coeff"])
    return cfg


@dataclass
class _VPINState:
    """Lens 1 — volume-clock VPIN with LOTECH sweep-distance weighting (causal, session-relative).

    Trades are read from the injected :class:`~mm_eval.tape.TradeTape` (the public prints a
    live strategy sees). Each trade's (weighted) volume fills the current volume bucket;
    when the bucket closes, its buy/sell imbalance ``|B−S|/(B+S)`` joins the rolling VPIN
    window. The elevated band is the session's own quantile — no fixed cross-market
    threshold (retires Task-5's fixed-threshold caveat).
    """

    cursor: int = 0                 # tape rows consumed
    typ_size: float = 0.0           # EWMA typical trade size (the volume clock's unit)
    _n_size: int = 0
    buy_vol: float = 0.0            # weighted volume in the OPEN bucket
    sell_vol: float = 0.0
    buckets: deque = field(default_factory=deque)      # closed-bucket imbalances (window)
    bucket_net: deque = field(default_factory=deque)   # closed-bucket signed net (direction)
    vpin_hist: deque = field(default_factory=deque)    # session VPIN readings (quantile band)
    last_touch: tuple[float, float] | None = None      # (best_bid, best_ask) BEFORE this event
    # cached per bucket-close (the quantile sort must not run on every event)
    _flag_cache: tuple[bool, str | None] = (False, None)
    _net_cache: float = 0.0

    def ingest(self, tape, book: BookState, cfg: dict) -> None:
        rows = getattr(tape, "rows", None)
        if rows is None:
            return
        win = cfg["nsq_vpin_window"]
        for i in range(self.cursor, len(rows)):
            ts, price, size, side = rows[i]
            if size <= 0:
                continue
            # typical-size EWMA (session-relative volume clock)
            if self._n_size == 0:
                self.typ_size = size
            else:
                self.typ_size += cfg["nsq_size_ewma_alpha"] * (size - self.typ_size)
            self._n_size += 1
            # aggressor side: captured taker side, else tick rule vs pre-event touch
            s = side
            if s not in ("BUY", "SELL") and self.last_touch is not None:
                m = (self.last_touch[0] + self.last_touch[1]) / 2.0
                s = "BUY" if price >= m else "SELL"
            if s not in ("BUY", "SELL"):
                self.cursor = i + 1
                continue
            # LOTECH sweep weighting: how far through the book the trade executed
            w = 1.0
            if self.last_touch is not None:
                bb, ba = self.last_touch
                dist = max(0.0, price - ba) if s == "BUY" else max(0.0, bb - price)
                ticks = min(dist / max(cfg["tick"], 1e-9), cfg["nsq_sweep_cap_ticks"])
                w = math.exp(ticks * cfg["nsq_sweep_lambda"])
            if s == "BUY":
                self.buy_vol += w * size
            else:
                self.sell_vol += w * size
            # close the bucket when the volume clock ticks
            v_bucket = max(self.typ_size * cfg["nsq_bucket_trades"], 1e-9)
            tot = self.buy_vol + self.sell_vol
            if tot >= v_bucket:
                self.buckets.append(abs(self.buy_vol - self.sell_vol) / tot)
                self.bucket_net.append(self.buy_vol - self.sell_vol)
                while len(self.buckets) > win:
                    self.buckets.popleft()
                    self.bucket_net.popleft()
                if len(self.buckets) >= max(win // 2, 2):
                    self.vpin_hist.append(sum(self.buckets) / len(self.buckets))
                    while len(self.vpin_hist) > cfg["nsq_vpin_hist_max"]:
                        self.vpin_hist.popleft()
                self.buy_vol = 0.0
                self.sell_vol = 0.0
                self._recompute_flag(cfg)   # quantile sort only at bucket close
            self.cursor = i + 1
        # touch AFTER processing this event's trades: next trades compare to this book
        if book.bids and book.asks:
            self.last_touch = (book.bids[0][0], book.asks[0][0])

    def reading(self) -> float:
        return self.vpin_hist[-1] if self.vpin_hist else 0.0

    def _recompute_flag(self, cfg: dict) -> None:
        """Refresh the cached (flag, exposed-side) — called only when a bucket closes."""
        self._net_cache = sum(self.bucket_net)
        if len(self.vpin_hist) < cfg["nsq_vpin_min_buckets"]:
            self._flag_cache = (False, None)
            return
        v = self.vpin_hist[-1]
        hist = sorted(self.vpin_hist)
        qi = min(int(cfg["nsq_vpin_q"] * (len(hist) - 1)), len(hist) - 1)
        if v >= max(hist[qi], cfg["nsq_vpin_floor"]):
            # net buy flow lifts our asks (we stack short) → the SELL side is exposed
            self._flag_cache = (True, "SELL" if self._net_cache > 0 else "BUY")
        else:
            self._flag_cache = (False, None)

    def flag_and_side(self, cfg: dict) -> tuple[bool, str | None]:
        """(directional_flag, exposed side) — cached; recomputed at each bucket close."""
        return self._flag_cache


@dataclass
class _ASZState:
    """Lens 2 — adverse-selection z-score of post-fill mid drift vs a calm session baseline.

    Own fills are detected from the inventory delta between successive ``quote`` calls
    (the frozen interface passes inventory; a live strategy gets fill notifications).
    ``signed_drift = side · (mid_{t+T} − mid_at_fill)`` — negative = adverse. The rolling
    baseline only updates while NO regime flag is active (LOTECH: calm-period baseline),
    so a spike cannot normalize itself into the reference distribution.
    """

    last_inv: float = 0.0
    pending: deque = field(default_factory=deque)      # (fill_ts, mid_at_fill, side_sign)
    drifts: deque = field(default_factory=deque)       # matured signed drifts (baseline)
    last_z: float = float("nan")
    last_bad_ts: float = -1e18

    def on_quote(self, book: BookState, inventory: float, m: float | None,
                 regime_active: bool, cfg: dict) -> None:
        ts = book.ts_exchange
        dq = inventory - self.last_inv
        if abs(dq) > 1e-12 and m is not None:
            self.pending.append((ts, m, 1.0 if dq > 0 else -1.0))
            self.last_inv = inventory
        elif abs(dq) > 1e-12:
            self.last_inv = inventory
        # mature pending fills whose horizon has elapsed
        horizon_ms = cfg["nsq_as_horizon_s"] * 1000.0
        n_base = int(cfg["nsq_as_baseline_n"])
        while self.pending and ts - self.pending[0][0] >= horizon_ms:
            f_ts, m_fill, sign = self.pending.popleft()
            if m is None:
                continue
            signed = sign * (m - m_fill)
            if len(self.drifts) >= int(cfg["nsq_as_min_fills"]):
                mu = sum(self.drifts) / len(self.drifts)
                var = sum((d - mu) ** 2 for d in self.drifts) / max(len(self.drifts) - 1, 1)
                sd = max(math.sqrt(var), 1e-4)
                self.last_z = (signed - mu) / sd
                if self.last_z < cfg["nsq_as_z_thresh"]:
                    self.last_bad_ts = ts
            # calm-only baseline update (frozen during any flagged regime)
            if not regime_active:
                self.drifts.append(signed)
                while len(self.drifts) > n_base:
                    self.drifts.popleft()

    def flag(self, ts: int, cfg: dict) -> bool:
        return (ts - self.last_bad_ts) <= cfg["nsq_as_flag_decay_s"] * 1000.0


@dataclass
class _OFIState:
    """Order-flow imbalance (Cont et al.) EWMA — the continuous size-dampening input."""

    prev: tuple[float, float, float, float] | None = None   # (bid_px, bid_sz, ask_px, ask_sz)
    prev_ts: int | None = None
    ofi: float = 0.0
    scale: float = 0.0

    def update(self, book: BookState, cfg: dict) -> None:
        if not book.bids or not book.asks:
            return
        bp, bs = book.bids[0]
        ap, asz = book.asks[0]
        ts = book.ts_exchange
        if self.prev is not None:
            pbp, pbs, pap, pas = self.prev
            e = ((bs if bp >= pbp else 0.0) - (pbs if bp <= pbp else 0.0)
                 - (asz if ap <= pap else 0.0) + (pas if ap >= pap else 0.0))
            dt_s = max((ts - (self.prev_ts or ts)) / 1000.0, 0.0)
            alpha = 1.0 - math.exp(-_LN2 * dt_s / cfg["nsq_ofi_halflife_s"]) if dt_s > 0 else 0.0
            self.ofi += alpha * (e - self.ofi) if alpha > 0 else 0.0
            self.scale += alpha * (abs(e) - self.scale) if alpha > 0 else 0.0
        self.prev = (bp, bs, ap, asz)
        self.prev_ts = ts

    def norm(self) -> float:
        """Signed pressure in (−1, 1): >0 = net buy pressure."""
        if self.scale <= 1e-12:
            return 0.0
        return math.tanh(self.ofi / (3.0 * self.scale))


@dataclass
class NeutralSpikeQuoter:
    """Task-5.1 controller: microprice skew + two-lens toxicity gate, NO calendar flatten.

    Replaces the Task-5 ``pull_hours`` flatten with graduated **spike avoidance** (the
    LOTECH course of action): carry balanced/small inventory to resolution, but refuse to
    stack a one-sided book into informed flow. Components (each toggleable via ``params``
    so the ladder can ablate them):

    * **Core** (always on): reservation ``r = microprice − skew_k·q`` + tight ``inv_cap``
      with passive reduce-only quoting at cap. τ is deliberately ignored — this quoter has
      no calendar behavior at all.
    * **Two-lens gate** (``nsq_lens1``/``nsq_lens2``): Lens 1 = volume-clock VPIN with
      sweep-distance weighting over the injected public-trade tape (``params["trade_tape"]``,
      see :mod:`mm_eval.tape`), session-relative elevated band; Lens 2 = post-fill drift
      z-score vs a calm rolling baseline (z < −2, statistical). Graduated response:
      Lens 1 only → suspend NEW quotes on the exposed side; Lens 2 only → reduce opening
      size; both → pull opening quotes, rest a reduce-only quote at a widened offset
      (passive unwind — never dump at the touch, never cross).
    * **Asymmetric repricing** (``nsq_asym``): opening quotes chase a moving market at a
      capped rate (slow to follow price away) but withdraw instantly (fast to back off) —
      the LOTECH idea-1 asymmetry.
    * **OFI size-dampening** (``nsq_damp_coeff`` > 0): opening size on the pressured side
      shrinks continuously with the Cont-style OFI state variable — exposure fades during
      one-sided flow without a hard threshold.

    Single-token state (the eval replays one token per run, like the other Task-5 quoters).
    """

    name: str = "neutral_spike"
    _vpin: _VPINState = field(default_factory=_VPINState)
    _asz: _ASZState = field(default_factory=_ASZState)
    _ofi: _OFIState = field(default_factory=_OFIState)
    _regime_active: bool = False
    _last_bid: tuple[int, float] | None = None    # (ts, px) — asymmetric repricing memory
    _last_ask: tuple[int, float] | None = None
    _cfg_key: int | None = None
    _cfg: dict | None = None

    def quote(self, book: BookState, inventory: float, params: dict) -> list[Order]:
        if self._cfg_key != id(params):
            self._cfg = _parse_nsq_cfg(params)
            self._cfg_key = id(params)
        cfg = self._cfg
        if book.stale:
            return []
        micro = microprice(book)
        m = mid(book)
        if micro is None:
            return []
        tick = cfg["tick"]
        half_spread = cfg["half_spread"]
        cap = cfg["inv_cap"]

        # ── signal state updates (always causal: past events only) ──
        tape = cfg.get("trade_tape") or (params or {}).get("trade_tape")
        if cfg["nsq_lens1"] and tape is not None:
            self._vpin.ingest(tape, book, cfg)
        if cfg["nsq_damp_coeff"] > 0:
            self._ofi.update(book, cfg)
        if cfg["nsq_lens2"]:
            self._asz.on_quote(book, inventory, m, self._regime_active, cfg)
        else:
            self._asz.last_inv = inventory   # keep the fill detector coherent if toggled

        directional, exposed = (self._vpin.flag_and_side(cfg)
                                if cfg["nsq_lens1"] else (False, None))
        adverse = self._asz.flag(book.ts_exchange, cfg) if cfg["nsq_lens2"] else False
        self._regime_active = directional or adverse

        r = micro - cfg["skew_k"] * inventory

        # ── graduated response ──
        if (directional and adverse) or abs(inventory) >= cap:
            # full adverse regime (or at cap): passive reduce-only, widened when toxic
            widen = cfg["nsq_widen_mult"] if (directional and adverse) else 1.0
            self._last_bid = self._last_ask = None
            return _reduce_only(book, inventory, cfg["size"], tick, self.name,
                                at_touch=False, reservation=r,
                                half_spread=widen * half_spread)

        size_bid = size_ask = cfg["size"]
        if adverse:                      # Lens 2 only: reduce opening size both sides
            size_bid *= cfg["nsq_as_size_factor"]
            size_ask *= cfg["nsq_as_size_factor"]
        if cfg["nsq_damp_coeff"] > 0:    # continuous OFI dampening on the pressured side
            n = self._ofi.norm()
            floor = cfg["nsq_size_floor"]
            if n > 0:                    # buy pressure → ask side exposed
                size_ask *= max(1.0 - cfg["nsq_damp_coeff"] * n, floor)
            elif n < 0:
                size_bid *= max(1.0 - cfg["nsq_damp_coeff"] * (-n), floor)

        bid = _round_clamp(r - half_spread, tick)
        ask = _round_clamp(r + half_spread, tick)
        if bid is None or ask is None or bid >= ask:
            self._last_bid = self._last_ask = None
            return []

        # ── asymmetric repricing: slow to chase, instant to withdraw ──
        ts = book.ts_exchange
        if cfg["nsq_asym"]:
            rate = cfg["nsq_chase_rate_c_s"] / CENTS_PER_PRICE   # price units per second
            if self._last_bid is not None and bid > self._last_bid[1]:
                dt_s = max((ts - self._last_bid[0]) / 1000.0, 0.0)
                bid = min(bid, self._last_bid[1] + rate * dt_s)
                bid = _round_clamp(bid, tick)
            if self._last_ask is not None and ask < self._last_ask[1]:
                dt_s = max((ts - self._last_ask[0]) / 1000.0, 0.0)
                ask = max(ask, self._last_ask[1] - rate * dt_s)
                ask = _round_clamp(ask, tick)
            if bid is None or ask is None or bid >= ask:
                self._last_bid = self._last_ask = None
                return []

        orders: list[Order] = []
        # Lens 1 only: suspend ADDING on the exposed side (stop one-sided stacking);
        # the other side keeps quoting (and passively reduces any exposed inventory).
        if not (directional and exposed == "BUY"):
            orders.append(Order(book.token_id, "BUY", bid, size_bid, tag=self.name))
            self._last_bid = (ts, bid)
        else:
            self._last_bid = None
        if not (directional and exposed == "SELL"):
            orders.append(Order(book.token_id, "SELL", ask, size_ask, tag=self.name))
            self._last_ask = (ts, ask)
        else:
            self._last_ask = None
        return orders


# Basket staleness: legs whose last-seen book is older than this are not re-quoted
# (their resting quotes get cancelled by the order manager's reconcile).
BASKET_BOOK_MAX_AGE_MS = 300_000


@dataclass
class BasketCarryQuoter:
    """v2 alternative to flattening: joint NegRisk-basket quoting, balance-skew, carry to expiry.

    ``params['basket_legs']`` (runner-injected) maps every leg's ``token_id`` to
    ``{"cond": condition_id, "sign": ±1}`` — same-condition complementary tokens net against
    each other (sign −1), sibling conditions in the event are mutually exclusive outcomes.
    Per-condition net exposure ``s_c = Σ sign_i·q_i``; each leg's reservation is skewed
    toward the *basket-balanced* state (all ``s_c`` equal): a leg whose condition is
    over-held quotes to shed, an under-held one to add — the inventory that remains tends
    toward a complementary basket whose resolution payoff is floored (one outcome pays).

    NO near-expiry flatten (that is the whole point of the comparison vs v1); the tight cap
    applies per condition. Quotes ALL legs on every event (books cached per leg; a leg
    whose book is stale/aged out is simply not re-quoted this tick).
    """

    name: str = "basket_carry"
    _books: dict = field(default_factory=dict)        # token_id -> (ts, BookState)
    _inv: dict = field(default_factory=dict)          # token_id -> last seen inventory

    def quote(self, book: BookState, inventory: float, params: dict) -> list[Order]:
        cfg = {**DEFAULT_PARAMS, **(params or {})}
        legs: dict = cfg.get("basket_legs") or {}
        size = float(cfg["size"])
        tick = float(cfg["tick"])
        k = float(cfg.get("skew_k", 0.0))
        cap = float(cfg.get("inv_cap", float("inf")))
        hs_by_token: dict = cfg.get("half_spread_by_token") or {}

        self._inv[book.token_id] = inventory
        if not book.stale:
            self._books[book.token_id] = (book.ts_exchange, book)

        if not legs:   # degenerate single-token use: behave like an unskewed micro quoter
            legs = {book.token_id: {"cond": book.token_id, "sign": 1}}

        # per-condition net exposure and the balance target (mean across conditions)
        s: dict[str, float] = {}
        for tok, leg in legs.items():
            s.setdefault(leg["cond"], 0.0)
            s[leg["cond"]] += leg["sign"] * self._inv.get(tok, 0.0)
        s_bar = sum(s.values()) / len(s) if s else 0.0

        now = book.ts_exchange
        orders: list[Order] = []
        for tok, leg in legs.items():
            cached = self._books.get(tok)
            if cached is None or (now - cached[0]) > BASKET_BOOK_MAX_AGE_MS:
                continue
            b = cached[1]
            micro = microprice(b)
            if micro is None:
                continue
            excess = s[leg["cond"]] - s_bar          # >0: this condition over-held
            r = micro - leg["sign"] * k * excess     # shed when over-held, add when under
            hs = float(hs_by_token.get(tok, cfg["half_spread"]))
            if abs(s[leg["cond"]]) >= cap:
                q_tok = self._inv.get(tok, 0.0)
                # reduce only the condition's exposure via this leg
                reduce_dir = -1.0 if (s[leg["cond"]] > 0) == (leg["sign"] > 0) else 1.0
                side = "SELL" if reduce_dir < 0 else "BUY"
                px = _round_clamp(r + (hs if side == "SELL" else -hs), tick)
                if px is not None:
                    clip = min(size, abs(q_tok)) if abs(q_tok) > 1e-9 else size
                    orders.append(Order(tok, side, px, clip, tag=self.name))
                continue
            bid = _round_clamp(r - hs, tick)
            ask = _round_clamp(r + hs, tick)
            if bid is None or ask is None or bid >= ask:
                continue
            orders.append(Order(tok, "BUY", bid, size, tag=self.name))
            orders.append(Order(tok, "SELL", ask, size, tag=self.name))
        return orders
