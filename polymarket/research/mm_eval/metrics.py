"""The pure numerics of the validation layer — markout, adverse selection, scorecard, breakeven.

All functions take plain telemetry (the engine's raw ``fills``/``quotes`` dicts) or arrays and
return values + confidence intervals. No engine state, no I/O — so they are unit-testable in
isolation and reusable for the Task-5 parameterized strategy.

**Units.** Polymarket prices are probabilities in ``[0,1]`` and a contract pays ``$1`` at
resolution, so price units *are dollars per contract*; ``0.01`` of price = **1 cent/contract**.
Internally everything is in price (dollar) units; the public markout/breakeven numbers are
reported in **cents per contract** (``× 100``) because that is the unit a per-contract edge is
read in (``brain/CODEX.md`` rule 4: report the deployable per-contract edge in absolute terms).

**Sign conventions (markout / adverse selection).** For a fill on ``side`` with sign
``+1`` (BUY → we are long, we want mid to rise) or ``-1`` (SELL → short, we want mid to fall):

* ``markout_to_fill(T)  = side · (mid_{t+T} − fill_price)`` — the realized per-contract edge to
  mid at horizon ``T``. It already **includes** the half-spread captured at the touch and **nets**
  the post-fill drift. This is the headline per-contract economic quantity.
* ``adverse_drift(T)     = side · (mid_{t+T} − mid_at_fill)`` — the **pure** post-fill mid move.
  Negative ⇒ the market moved against us ⇒ adverse selection. ``A(T) ≡ −mean(adverse_drift)``.

Per fill, ``markout_to_fill = realized_half_spread + adverse_drift`` (the ``realized_half_spread``
is ``side·(mid_at_fill − fill_price)`` — the spread actually captured at the instant of the fill).
The maker's net per-contract edge is ``E(T) = markout_to_fill(T) + rebate``, and the **verdict gates
directly on** ``E(T)``'s bootstrap lower CI (``BreakevenRead.clears``). The reported
``A* = half_spread + rebate`` uses the **static quoted** ``half_spread`` (median touch spread / 2),
so ``A*`` is a **design-target reference**, NOT algebraically equal to the ``E(T)`` gate: under
adverse selection and tick rounding the realized half-spread drifts from the quoted one, so
``A(T) ≤ A*`` and ``E(T) ≥ 0`` can disagree at the margin. Trust ``net_edge_cents`` (the gate);
read ``A*`` vs measured ``A(T)`` as the intuition, not the decision.

**CIs, never point estimates.** Means carry a **moving-block bootstrap** CI (block length
``≈ √n``) because fills cluster in time (adverse selection is autocorrelated); an iid bootstrap
would understate the interval. Per-contract means are **quantity-weighted** (a 100-lot fill
counts more than a 10-lot).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Task-specified markout horizons (seconds).
MARKOUT_HORIZONS_S = (1, 5, 30, 60)
CENTS = 100.0  # price (dollars/contract) -> cents/contract


# ──────────────────────────────────────────────────────────────────────────────
# Block bootstrap
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float
    n: int

    def cents(self) -> "CI":
        return CI(self.point * CENTS, self.lo * CENTS, self.hi * CENTS, self.n)

    @property
    def clears_zero(self) -> bool:
        """Lower CI strictly above 0 — the bar for calling a per-contract edge positive."""
        return np.isfinite(self.lo) and self.lo > 0.0


def block_bootstrap_mean_ci(
    values, weights=None, *, n_boot: int = 2000, block: int | None = None,
    seed: int = 0, confidence: float = 0.95,
) -> CI:
    """Quantity-weighted mean with a moving-block bootstrap percentile CI.

    ``values``/``weights`` are aligned per-fill arrays (time-ordered). NaNs (censored fills) are
    dropped pairwise. Block length defaults to ``round(√n)`` (rate-optimal for short-range
    dependence). Degrades gracefully: ``n < 3`` returns the point with NaN bounds.
    """
    v = np.asarray(values, dtype=float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[mask], w[mask]
    n = v.size
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    point = float(np.sum(v * w) / np.sum(w))
    if n < 3:
        return CI(point, float("nan"), float("nan"), n)
    blk = block or max(1, int(round(math.sqrt(n))))
    n_blocks = int(math.ceil(n / blk))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    base = np.arange(blk)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + base[None, :]).ravel() % n)[:n]
        sw = np.sum(w[idx])
        draws[b] = np.sum(v[idx] * w[idx]) / sw if sw > 0 else np.nan
    alpha = (1 - confidence) / 2
    lo, hi = np.nanquantile(draws, [alpha, 1 - alpha])
    return CI(point, float(lo), float(hi), n)


# ──────────────────────────────────────────────────────────────────────────────
# Mid trajectory + markout
# ──────────────────────────────────────────────────────────────────────────────

def mid_trajectory(quotes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(ts_ms, mid) arrays from the quotes log, two-sided mids only, sorted, deduped-by-last.

    The engine logs ``mid`` per event; this is the per-token mid trajectory markout joins onto.
    """
    pts = [(int(q["ts_exchange"]), float(q["mid"])) for q in quotes if q.get("mid") is not None]
    if not pts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    pts.sort(key=lambda x: x[0])
    ts = np.fromiter((p[0] for p in pts), dtype=np.int64, count=len(pts))
    mid = np.fromiter((p[1] for p in pts), dtype=float, count=len(pts))
    return ts, mid


def _asof_backward(ts: np.ndarray, mid: np.ndarray, target_ts: int) -> float:
    """Prevailing mid at ``target_ts`` = last observation with ts ≤ target_ts (backward as-of)."""
    if ts.size == 0:
        return float("nan")
    pos = int(np.searchsorted(ts, target_ts, side="right")) - 1
    if pos < 0:
        return float("nan")
    return float(mid[pos])


@dataclass
class MarkoutResult:
    """Per-fill markout arrays at each horizon (cents/contract is applied by the curve builder)."""

    horizons: tuple[int, ...]
    side_sign: np.ndarray                       # +1 BUY / -1 SELL per fill
    qty: np.ndarray                             # per-fill quantity
    fill_ts: np.ndarray                         # per-fill ts_exchange (ms) — for temporal stability
    fill_price: np.ndarray
    mid_at_fill: np.ndarray
    markout_to_fill: dict[int, np.ndarray]      # horizon -> per-fill markout (price units, NaN=censored)
    adverse_drift: dict[int, np.ndarray]        # horizon -> per-fill drift (price units, NaN=censored)
    censored: dict[int, int] = field(default_factory=dict)  # horizon -> #fills past the data edge


def compute_markout(
    fills: list[dict], quotes: list[dict], *, horizons=MARKOUT_HORIZONS_S,
) -> MarkoutResult:
    """Signed markout + adverse drift per fill at each horizon, right-censored at the data edge.

    A fill's horizon is **censored** (NaN) when ``fill_ts + T`` exceeds the last event timestamp —
    the capture ends before the horizon elapses, so the future mid is genuinely unobserved (no
    extrapolation, no lookahead). Censored counts are reported per horizon.
    """
    ts, mid = mid_trajectory(quotes)
    last_ts = int(ts[-1]) if ts.size else 0
    n = len(fills)
    side_sign = np.empty(n); qty = np.empty(n); fp = np.empty(n); maf = np.empty(n)
    fts = np.empty(n, dtype=np.int64)
    m2f = {h: np.full(n, np.nan) for h in horizons}
    drift = {h: np.full(n, np.nan) for h in horizons}
    cens = {h: 0 for h in horizons}
    for i, f in enumerate(fills):
        sign = 1.0 if f["side"] == "BUY" else -1.0
        side_sign[i] = sign
        qty[i] = float(f["qty"])
        fp[i] = float(f["price"])
        m0 = f.get("mid_at_fill")
        maf[i] = float(m0) if m0 is not None else float("nan")
        ft = int(f["ts_exchange"])
        fts[i] = ft
        for h in horizons:
            target = ft + h * 1000
            if target > last_ts:
                cens[h] += 1
                continue
            mt = _asof_backward(ts, mid, target)
            if not np.isfinite(mt):
                continue
            m2f[h][i] = sign * (mt - fp[i])
            if np.isfinite(maf[i]):
                drift[h][i] = sign * (mt - maf[i])
    return MarkoutResult(tuple(horizons), side_sign, qty, fts, fp, maf, m2f, drift, cens)


@dataclass(frozen=True)
class MarkoutCurvePoint:
    horizon_s: int
    markout_to_fill_cents: CI       # realized per-contract edge to mid (incl. half-spread)
    adverse_drift_cents: CI         # pure post-fill drift (negative = adverse)
    adverse_rate: float             # qty-weighted share of fills with markout_to_fill < 0
    n_fills: int
    n_censored: int


def markout_curve(mr: MarkoutResult, *, seed: int = 0, n_boot: int = 2000) -> list[MarkoutCurvePoint]:
    """Per-horizon markout curve: quantity-weighted mean (cents/contract) + block-bootstrap CI."""
    out = []
    for h in mr.horizons:
        m2f = mr.markout_to_fill[h]
        drift = mr.adverse_drift[h]
        ci_m = block_bootstrap_mean_ci(m2f, mr.qty, seed=seed, n_boot=n_boot).cents()
        ci_d = block_bootstrap_mean_ci(drift, mr.qty, seed=seed, n_boot=n_boot).cents()
        mask = np.isfinite(m2f)
        if mask.any():
            wq = mr.qty[mask]
            adverse_rate = float(np.sum(wq[m2f[mask] < 0]) / np.sum(wq)) if np.sum(wq) > 0 else float("nan")
        else:
            adverse_rate = float("nan")
        out.append(MarkoutCurvePoint(h, ci_m, ci_d, adverse_rate, int(mask.sum()), mr.censored[h]))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Scorecard
# ──────────────────────────────────────────────────────────────────────────────

def _inventory_path(fills: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(ts_ms, signed position) step path from fills' ``position_after``."""
    if not fills:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    ts = np.fromiter((int(f["ts_exchange"]) for f in fills), dtype=np.int64, count=len(fills))
    pos = np.fromiter((float(f["position_after"]) for f in fills), dtype=float, count=len(fills))
    return ts, pos


def time_weighted_abs_inventory(fills: list[dict], end_ts: int | None = None) -> float:
    """Time-weighted mean |position| over the run (the denominator for PnLMAP)."""
    ts, pos = _inventory_path(fills)
    if ts.size == 0:
        return 0.0
    end = end_ts if end_ts is not None else int(ts[-1])
    dur = max(end - int(ts[0]), 1)
    # position is held from each fill ts until the next; |pos| weighted by the dwell time
    seg_ts = np.append(ts, end)
    dwell = np.diff(seg_ts).astype(float)
    return float(np.sum(np.abs(pos) * dwell) / dur) if dur > 0 else float(np.mean(np.abs(pos)))


def max_drawdown(equity_path: list[tuple[int, float]]) -> float:
    """Max peak-to-trough drawdown ($) of the net-with-rebate equity path. Returns a positive $."""
    if not equity_path:
        return 0.0
    eq = np.array([e for _, e in equity_path], dtype=float)
    peak = np.maximum.accumulate(eq)
    return float(np.max(peak - eq))


def interval_sharpe(equity_path: list[tuple[int, float]], *, bucket_s: int = 300) -> tuple[float, int]:
    """Sharpe of PnL increments bucketed at ``bucket_s`` wall-clock seconds (per-bucket, not annualised).

    Sampling the equity path at coarse buckets (vs per-event) avoids letting same-timestamp
    mark-to-mid noise dominate. Returned un-annualised because the bucket→year factor is large and
    misleading on a 39.5 h sample; the note reports it as a *shape* statistic, heavily caveated
    (the symmetric quoter's equity is inventory-mark-dominated, so Sharpe is diagnostic only).
    """
    if len(equity_path) < 3:
        return float("nan"), 0
    ts = np.array([t for t, _ in equity_path], dtype=np.int64)
    eq = np.array([e for _, e in equity_path], dtype=float)
    t0 = ts[0]
    bucket = ((ts - t0) // (bucket_s * 1000)).astype(np.int64)
    # last equity in each bucket
    last_idx = {}
    for i, b in enumerate(bucket):
        last_idx[int(b)] = i
    bs = sorted(last_idx)
    bucket_eq = np.array([eq[last_idx[b]] for b in bs], dtype=float)
    if bucket_eq.size < 3:
        return float("nan"), int(bucket_eq.size)
    rets = np.diff(bucket_eq)
    sd = rets.std(ddof=1)
    return (float(rets.mean() / sd) if sd > 0 else 0.0), int(rets.size)


def profit_ratio(fills: list[dict]) -> float:
    """Gross realized profit / |gross realized loss| over offsetting round-trips (``realized_delta``)."""
    deltas = np.array([float(f.get("realized_delta", 0.0)) for f in fills], dtype=float)
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def quote_uptime(quotes: list[dict]) -> dict:
    """Share of events where the quoter rested a two-sided / any quote, and the stale share."""
    n = len(quotes)
    if n == 0:
        return {"two_sided": float("nan"), "any": float("nan"), "stale": float("nan"), "events": 0}
    two = sum(1 for q in quotes if len(q.get("orders", [])) >= 2)
    any_ = sum(1 for q in quotes if len(q.get("orders", [])) >= 1)
    stale = sum(1 for q in quotes if q.get("stale"))
    return {"two_sided": two / n, "any": any_ / n, "stale": stale / n, "events": n}


@dataclass(frozen=True)
class Scorecard:
    """Per-market, per-queue-model scorecard. PnL is reported three ways (gross / ex-rebate / with-rebate)."""

    # PnL ($)
    realized: float
    unrealized: float
    rebates: float
    gross_pnl: float
    net_ex_rebate: float
    net_with_rebate: float
    settled_available: bool
    settled_pnl: float
    # normalized
    days: float
    nd_pnl_gross: float            # Normalized Daily PnL (per capture-day)
    nd_pnl_ex_rebate: float
    nd_pnl_with_rebate: float
    pnlmap_ex_rebate: float        # PnL per unit Mean Absolute Position (capital/inventory efficiency)
    pnlmap_with_rebate: float
    mean_abs_inventory: float
    # activity / risk
    fills: int
    placed: int
    fill_rate: float
    filled_qty: float
    max_inventory: float
    net_position: float
    profit_ratio: float
    max_drawdown_usd: float
    interval_sharpe: float
    interval_sharpe_buckets: int
    quote_uptime_two_sided: float
    quote_uptime_any: float
    stale_share: float
    rebate_per_contract: float     # mean rebate $/contract under the schedule used (0 if fee_free)
    l1_both_match_frac: float | None


def build_scorecard(result, *, days: float, settle=None, sharpe_bucket_s: int = 300) -> Scorecard:
    """Assemble the scorecard from an ``EngineResult`` (+ optional ``Settlement``)."""
    fills = result.fills
    quotes = result.quotes
    net_ex = result.net_ex_rebate
    net_wr = result.net_with_rebate
    gross = result.gross_pnl
    end_ts = int(result.equity_path[-1][0]) if result.equity_path else None
    mean_abs_inv = time_weighted_abs_inventory(fills, end_ts=end_ts)
    up = quote_uptime(quotes)
    max_inv = max((abs(float(f["position_after"])) for f in fills), default=0.0)
    rpc = (result.rebates_earned / result.filled_qty) if result.filled_qty > 0 else 0.0
    shp, nb = interval_sharpe(result.equity_path, bucket_s=sharpe_bucket_s)

    settled_ok = settle is not None
    settled_pnl = float(settle.settled_pnl) if settled_ok else float("nan")
    # If settled, net flavors fold settled inventory in place of mark-to-mid (the honest carry read).
    if settled_ok:
        gross = settle.gross_pnl
        net_ex = settle.net_ex_rebate
        net_wr = settle.net_with_rebate

    def _safe_div(a, b):
        return float(a / b) if b not in (0, 0.0) and np.isfinite(b) else float("nan")

    return Scorecard(
        realized=result.realized_pnl,
        unrealized=result.unrealized_pnl,
        rebates=result.rebates_earned,
        gross_pnl=gross, net_ex_rebate=net_ex, net_with_rebate=net_wr,
        settled_available=settled_ok, settled_pnl=settled_pnl,
        days=days,
        nd_pnl_gross=_safe_div(gross, days),
        nd_pnl_ex_rebate=_safe_div(net_ex, days),
        nd_pnl_with_rebate=_safe_div(net_wr, days),
        pnlmap_ex_rebate=_safe_div(net_ex, mean_abs_inv),
        pnlmap_with_rebate=_safe_div(net_wr, mean_abs_inv),
        mean_abs_inventory=mean_abs_inv,
        fills=result.fill_count, placed=result.placed_count,
        fill_rate=(result.fill_count / result.placed_count) if result.placed_count else 0.0,
        filled_qty=result.filled_qty,
        max_inventory=max_inv,
        net_position=sum(result.position.values()),
        profit_ratio=profit_ratio(fills),
        max_drawdown_usd=max_drawdown(result.equity_path),
        interval_sharpe=shp, interval_sharpe_buckets=nb,
        quote_uptime_two_sided=up["two_sided"], quote_uptime_any=up["any"], stale_share=up["stale"],
        rebate_per_contract=rpc,
        l1_both_match_frac=result.l1_crosscheck.get("both_match_frac"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Breakeven + viable/fragile/dead verdict
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BreakevenRead:
    """Per-market breakeven economics at one markout horizon, for one queue model + fee assumption."""

    horizon_s: int
    queue: str
    fee_mode: str                  # "no_rebate" (captured fee=0) | "representative"
    half_spread_cents: float       # A* with no rebate
    rebate_cents: float            # per-contract rebate under the fee mode
    breakeven_adverse_cents: float # A* = half_spread + rebate (max adverse selection absorbable)
    measured_adverse_cents: CI     # A(T) = -mean(adverse_drift) (positive = adverse)
    net_edge_cents: CI             # E(T) = markout_to_fill + rebate (>0 = clears breakeven)
    clears: bool                   # net-edge lower-CI > 0


def breakeven_read(
    curve_point: MarkoutCurvePoint, *, queue: str, half_spread: float,
    rebate_per_contract: float, fee_mode: str,
) -> BreakevenRead:
    """Map a markout-curve point + spread + rebate to the breakeven economics.

    ``net_edge = markout_to_fill + rebate`` (markout_to_fill already includes the half-spread).
    The CI on net_edge is the markout CI shifted by the (≈constant) per-contract rebate.
    """
    h_c = half_spread * CENTS
    r_c = rebate_per_contract * CENTS
    a_star = h_c + r_c
    m = curve_point.markout_to_fill_cents
    net = CI(m.point + r_c, m.lo + r_c, m.hi + r_c, m.n)
    adverse = curve_point.adverse_drift_cents
    measured_adverse = CI(-adverse.point, -adverse.hi, -adverse.lo, adverse.n)  # flip sign + bounds
    return BreakevenRead(
        horizon_s=curve_point.horizon_s, queue=queue, fee_mode=fee_mode,
        half_spread_cents=h_c, rebate_cents=r_c, breakeven_adverse_cents=a_star,
        measured_adverse_cents=measured_adverse, net_edge_cents=net,
        clears=net.clears_zero,
    )


def verdict_from_bracket(optimistic: BreakevenRead, pessimistic: BreakevenRead) -> str:
    """viable / fragile / dead from the optimistic↔pessimistic queue bracket.

    * **VIABLE**  — even the pessimistic queue's net-edge lower-CI clears 0.
    * **FRAGILE** — only the optimistic queue clears; the pessimistic does not.
    * **DEAD**    — even the optimistic (best-case) queue does not clear.

    This is the task's "DEAD (optimistic < breakeven) / FRAGILE (only optimistic clears)" mapped
    onto the per-contract net edge (the breakeven being the zero line of ``E(T)``).
    """
    if pessimistic.clears:
        return "VIABLE"
    if optimistic.clears:
        return "FRAGILE"
    return "DEAD"
