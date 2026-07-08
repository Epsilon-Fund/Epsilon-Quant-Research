"""Join-2b latency-measurement harness — submit→ack round-trip via UNEXECUTABLE probes.

Implements the protocol in
``polymarket/research/notes/overview/data_quality/mm_latency_measurement_spec.md``:

* A probe is an order priced so far from the touch that it **cannot fill** — BUY at
  ``0.001`` (the lowest tick) or SELL at ``0.999`` (the highest), size 1 contract — and it
  is **cancelled immediately** after the venue acknowledges it.
* The measured quantity is the **submit→ack round-trip on the local monotonic clock**:
  ``t_submit`` stamped immediately before the signed request leaves us, ``t_ack`` when the
  venue's synchronous ack (the ``submit_order`` return) arrives. Cancel round-trips are
  logged separately.
* Safety guard (the spec's "one fill path"): only probe a side whose probe price is
  strictly outside the current book — ``best_bid > 0.001 + tick`` for a BUY probe,
  ``best_ask < 0.999 - tick`` for a SELL probe — i.e. never probe a near-resolved market
  trading within a tick of 0/1. When both sides are safe, prefer the side **further** from
  the touch.
* Samples accumulate to a fit — mean / std / p50 / p90 / p99, reported **raw and trimmed**
  (winsorized) per the spec's honest-in-both-directions rule — which then **sets the
  latency model**: the trimmed mean → ``ConstantLatency`` (politics), the full ``(mean,
  std)`` → ``SampledLatency.from_samples`` (fast markets). The bridge picks the constant up
  via ``POLYMARKET_MM_BRIDGE_LATENCY_MS``.

**Safety.** Every probe placement goes through the SAME order path as the bridge — the
reused :class:`~polymarket.execution.maker.mm_engine_bridge.VenueOrderRouter` (kill-switch +
per-trade/market/deployed/daily caps + the shared
:class:`~polymarket.execution.maker.order_safety.RealOrderGate`). A probe on a real venue
therefore consumes the ``MAX_REAL_ORDERS`` budget and honors ``REQUIRE_OPERATOR_CONFIRM``
per order, exactly like a quote. On a fake venue (the default) the gate is a no-op and no
network request exists anywhere in this module. **This build never runs it against a real
venue** — the real measurement run is the operator's Join-2b step (see the runbook).

Timing is taken by a transparent venue wrapper (:class:`TimingVenue`) that stamps
``time.monotonic_ns()`` around the inner ``submit_order`` / ``cancel_order`` calls, so the
safety checks and journal writes of the router are **excluded** from the measured
round-trip (we want the venue's latency, not our own bookkeeping).

No secret is ever read or printed here; the harness sees only the already-built venue
adapter, prices, and clock values.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import JsonlWriter, MakerQuoteSkipped

from .mm_engine_bridge import (
    BridgeVenue,
    PlaceOutcome,
    VenueOrderRouter,
    ensure_mm_engine_importable,
)

ensure_mm_engine_importable()
from mm_engine.interfaces import Order  # noqa: E402
from mm_engine.orders import ActiveOrder  # noqa: E402

# The unexecutable probe prices — the extreme ticks of PM's 0.001 grid (spec §1).
PROBE_BUY_PRICE = 0.001
PROBE_SELL_PRICE = 0.999
PROBE_SIZE_CONTRACTS = 1.0


# --------------------------------------------------------------------------------------
# timing wrapper — measures the venue call only, transparently delegating everything else
# --------------------------------------------------------------------------------------

class TimingVenue:
    """Wraps a venue adapter; stamps ``monotonic_ns`` around submit/cancel calls.

    Everything else (``is_real_venue``, ``reconcile_open_orders``, ``stop``, …) delegates to
    the wrapped venue via ``__getattr__``, so the safety gate still sees the REAL venue's
    ``is_real_venue()`` — wrapping can never turn a real venue into a fake one.
    """

    def __init__(self, inner: BridgeVenue, *, clock_ns: Callable[[], int] | None = None) -> None:
        self._inner = inner
        self._clock_ns = clock_ns if clock_ns is not None else time.monotonic_ns
        self.last_submit_ms: float | None = None
        self.last_cancel_ms: float | None = None

    def submit_order(self, **kwargs: Any):
        t0 = self._clock_ns()
        result = self._inner.submit_order(**kwargs)
        self.last_submit_ms = (self._clock_ns() - t0) / 1e6
        return result

    def cancel_order(self, **kwargs: Any):
        t0 = self._clock_ns()
        result = self._inner.cancel_order(**kwargs)
        self.last_cancel_ms = (self._clock_ns() - t0) / 1e6
        return result

    def is_real_venue(self) -> bool:
        fn = getattr(self._inner, "is_real_venue", None)
        return bool(fn()) if callable(fn) else False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# --------------------------------------------------------------------------------------
# probe-side selection (the spec's safety guard)
# --------------------------------------------------------------------------------------

def choose_probe_side(
    best_bid: float | None,
    best_ask: float | None,
    *,
    tick: float = 0.001,
    prefer: str = "auto",
) -> str | None:
    """Pick the safe probe side, or ``None`` when no probe is safe.

    A BUY probe at 0.001 is safe only if the book rests strictly above it
    (``best_bid > 0.001 + tick``); a SELL probe at 0.999 only if the book rests strictly
    below it (``best_ask < 0.999 - tick``). Books within a tick of 0/1 (near-resolved) are
    never probed. With both sides safe and ``prefer='auto'``, pick the side whose probe
    price sits FURTHER from the touch.
    """
    if best_bid is None or best_ask is None:
        return None
    buy_safe = best_bid > PROBE_BUY_PRICE + tick
    sell_safe = best_ask < PROBE_SELL_PRICE - tick
    if prefer == "buy":
        return "BUY" if buy_safe else None
    if prefer == "sell":
        return "SELL" if sell_safe else None
    if buy_safe and sell_safe:
        dist_buy = best_bid - PROBE_BUY_PRICE      # gap between the book and the BUY probe
        dist_sell = PROBE_SELL_PRICE - best_ask    # gap between the book and the SELL probe
        return "BUY" if dist_buy >= dist_sell else "SELL"
    if buy_safe:
        return "BUY"
    if sell_safe:
        return "SELL"
    return None


# --------------------------------------------------------------------------------------
# the fit (spec §3): mean/std/percentiles, raw AND trimmed
# --------------------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _stats(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    s = sorted(vals)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": var ** 0.5,
        "p50": _percentile(s, 0.50),
        "p90": _percentile(s, 0.90),
        "p99": _percentile(s, 0.99),
    }


def fit_latency(samples_ms: list[float], *, trim_frac: float = 0.02) -> dict[str, Any]:
    """Fit the round-trip samples: raw stats + winsorized ("trimmed") stats.

    Winsorizing clamps the top/bottom ``trim_frac`` tails to the remaining extremes, so one
    multi-second network stall cannot inflate the whole model, while the raw fit still
    reports the genuine tail (spec §3: report BOTH). The recommended latency-model inputs:
    ``constant_ms`` = the trimmed mean (politics ``ConstantLatency``), ``sampled`` = the
    trimmed ``(mean, std)`` (crypto/sports ``SampledLatency``).
    """
    raw = _stats(samples_ms)
    if samples_ms and 0.0 < trim_frac < 0.5:
        s = sorted(samples_ms)
        k = int(math.floor(len(s) * trim_frac))
        if k > 0 and len(s) > 2 * k:
            lo, hi = s[k], s[-k - 1]
            winsorized = [min(max(x, lo), hi) for x in s]
        else:
            winsorized = s
    else:
        winsorized = sorted(samples_ms)
    trimmed = _stats(winsorized)
    return {
        "raw": raw,
        "trimmed": trimmed,
        "trim_frac": trim_frac,
        # the numbers that SET the latency model (see module docstring)
        "constant_ms": trimmed["mean"],
        "sampled": {"mean": trimmed["mean"], "std": trimmed["std"]},
    }


# --------------------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------------------

@dataclass
class LatencyProbeConfig:
    """Probe-run parameters. Defaults are DRY-RUN friendly (no sleeps, few samples)."""

    condition_id: str
    asset_id: str
    tick: float = 0.001
    samples_target: int = 20          # operator raises to the spec's K≈200–500 for a real fit
    cadence_s: float = 0.0            # seconds between probes (0 for dry runs / tests)
    prefer: str = "auto"              # "auto" | "buy" | "sell"
    trim_frac: float = 0.02
    out_dir: Path | None = None       # where samples JSONL + fit JSON land (None = no files)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "LatencyProbeConfig":
        cond = (env.get("POLYMARKET_MAKER_CONDITION_ID", "") or "").strip().lower()
        if not cond:
            raise ValueError("POLYMARKET_MAKER_CONDITION_ID is required")
        out_dir_raw = (env.get("POLYMARKET_MM_LATENCY_OUT_DIR", "") or "").strip()
        return cls(
            condition_id=cond,
            asset_id=(env.get("POLYMARKET_MM_BRIDGE_ASSET_ID", "") or "").strip(),
            tick=float(env.get("POLYMARKET_MM_BRIDGE_TICK", "0.001")),
            samples_target=int(env.get("POLYMARKET_MM_LATENCY_SAMPLES", "20")),
            cadence_s=float(env.get("POLYMARKET_MM_LATENCY_CADENCE_S", "30")),
            prefer=(env.get("POLYMARKET_MM_LATENCY_SIDE", "auto") or "auto").lower(),
            trim_frac=float(env.get("POLYMARKET_MM_LATENCY_TRIM", "0.02")),
            out_dir=Path(out_dir_raw) if out_dir_raw else None,
        )


@dataclass
class ProbeRecord:
    """One probe's outcome — a sample when accepted, a skip reason otherwise."""

    ts_utc: str
    side: str | None
    price: float | None
    accepted: bool
    submit_ms: float | None
    cancel_ms: float | None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts_utc": self.ts_utc, "side": self.side, "price": self.price,
            "accepted": self.accepted, "submit_ms": self.submit_ms,
            "cancel_ms": self.cancel_ms, "reason": self.reason,
        }


@dataclass
class LatencyProbeHarness:
    """Drives probe→time→cancel cycles through the reused safety + venue path.

    ``book_fn`` returns the current ``(best_bid, best_ask)`` — injected so tests and the
    dry-run need no network (live ops pass a ``ClobBookClient``-backed reader). ``sleep_fn``
    is injected for the cadence so tests never sleep.
    """

    router: VenueOrderRouter
    timing: TimingVenue
    cfg: LatencyProbeConfig
    journal: JsonlWriter
    book_fn: Callable[[], tuple[float | None, float | None]]
    risk_state_fn: Callable[[str, float], Any]
    sleep_fn: Callable[[float], None] = time.sleep
    records: list[ProbeRecord] = field(default_factory=list)
    _counter: int = field(default=0, init=False)

    def probe_once(self) -> ProbeRecord:
        best_bid, best_ask = self.book_fn()
        side = choose_probe_side(best_bid, best_ask, tick=self.cfg.tick, prefer=self.cfg.prefer)
        now_iso = datetime.now(timezone.utc).isoformat()
        if side is None:
            rec = ProbeRecord(ts_utc=now_iso, side=None, price=None, accepted=False,
                              submit_ms=None, cancel_ms=None, reason="unsafe_book")
            self.journal.write(MakerQuoteSkipped(
                ts_utc=datetime.now(timezone.utc),
                condition_id=self.cfg.condition_id,
                asset_id=self.cfg.asset_id,
                side="",
                reason="latency_probe_unsafe_book",
                detail=f"best_bid={best_bid} best_ask={best_ask} — probe prices not strictly outside",
            ))
            self.records.append(rec)
            return rec

        price = PROBE_BUY_PRICE if side == "BUY" else PROBE_SELL_PRICE
        self._counter += 1
        client_id = f"probe{self._counter}"
        order = Order(self.cfg.asset_id, side, price, PROBE_SIZE_CONTRACTS, tag="latency_probe")

        # SAME safety path as a bridge quote: risk breakers + RealOrderGate + venue.
        outcome: PlaceOutcome = self.router.place(
            order, client_id,
            condition_id=self.cfg.condition_id,
            state=self.risk_state_fn(self.cfg.asset_id, price),
        )
        if not outcome.accepted:
            # Ambiguous/exception outcomes may have left the probe RESTING at the venue
            # (timeout after the order landed). A GTC at the extreme tick must never rest
            # unattended — cancels are ungated and risk-reducing, so fire a best-effort
            # cancel-by-coid before reporting the block (adversarial-review fix).
            if outcome.reason in ("ambiguous_submit", "mm_bridge_submit_exception"):
                ao = ActiveOrder(order=order, client_id=client_id, placement_ts=0,
                                 last_change_ts=0, remaining=PROBE_SIZE_CONTRACTS)
                self.router.cancel(
                    ao,
                    condition_id=self.cfg.condition_id,
                    venue_order_id=None,
                    reason="latency_probe_ambiguous_rollback",
                )
            rec = ProbeRecord(ts_utc=now_iso, side=side, price=price, accepted=False,
                              submit_ms=None, cancel_ms=None, reason=outcome.reason)
            self.records.append(rec)
            return rec

        submit_ms = self.timing.last_submit_ms
        # cancel IMMEDIATELY (spec §1) — cancels reduce risk and are not gated.
        ao = ActiveOrder(order=order, client_id=client_id, placement_ts=0,
                         last_change_ts=0, remaining=PROBE_SIZE_CONTRACTS)
        self.router.cancel(
            ao,
            condition_id=self.cfg.condition_id,
            venue_order_id=outcome.venue_order_id,
            reason="latency_probe",
        )
        rec = ProbeRecord(ts_utc=now_iso, side=side, price=price, accepted=True,
                          submit_ms=submit_ms, cancel_ms=self.timing.last_cancel_ms)
        self.records.append(rec)
        return rec

    def run(self) -> dict[str, Any]:
        """Probe until ``samples_target`` accepted samples (or the router halts)."""
        accepted = 0
        consecutive_blocked = 0
        while accepted < self.cfg.samples_target:
            if self.router.halted:
                break
            rec = self.probe_once()
            if rec.accepted:
                accepted += 1
                consecutive_blocked = 0
            else:
                consecutive_blocked += 1
                # a persistently-blocked probe (gate exhausted / unsafe book) must not spin
                if consecutive_blocked >= 3:
                    break
            if self.cfg.cadence_s > 0 and accepted < self.cfg.samples_target:
                self.sleep_fn(self.cfg.cadence_s)
        return self.report()

    def report(self) -> dict[str, Any]:
        submit_samples = [r.submit_ms for r in self.records if r.accepted and r.submit_ms is not None]
        cancel_samples = [r.cancel_ms for r in self.records if r.accepted and r.cancel_ms is not None]
        fit = fit_latency(submit_samples, trim_frac=self.cfg.trim_frac)
        out: dict[str, Any] = {
            "condition_id": self.cfg.condition_id,
            "asset_id": self.cfg.asset_id,
            "probes_attempted": len(self.records),
            "samples": len(submit_samples),
            "submit_ack_fit": fit,
            "cancel_fit": fit_latency(cancel_samples, trim_frac=self.cfg.trim_frac),
            "skips": {r.reason: sum(1 for x in self.records if x.reason == r.reason)
                      for r in self.records if r.reason},
            # the assumption ledger the spec requires: what this number is and is not
            "measures": "order-entry submit→ack round-trip (local monotonic clock)",
            "excludes": "feed latency (market-data staleness) — a separate tick-to-trade term",
        }
        if self.cfg.out_dir is not None:
            self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            samples_path = self.cfg.out_dir / f"latency_samples_{stamp}.jsonl"
            with samples_path.open("a", encoding="utf-8") as fh:   # append-only
                for r in self.records:
                    fh.write(json.dumps(r.as_dict()) + "\n")
            fit_path = self.cfg.out_dir / f"latency_fit_{stamp}.json"
            fit_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            out["samples_path"] = str(samples_path)
            out["fit_path"] = str(fit_path)
        return out


def render_fit(report: dict[str, Any]) -> str:
    """Human-readable fit summary (stdout; contains prices/latencies only, never secrets)."""
    fit = report["submit_ack_fit"]
    raw, trm = fit["raw"], fit["trimmed"]

    def line(tag: str, s: dict[str, float]) -> str:
        return (f"  {tag:8s} n={s['n']:>4}  mean={s['mean']:8.2f}ms  std={s['std']:7.2f}ms  "
                f"p50={s['p50']:8.2f}  p90={s['p90']:8.2f}  p99={s['p99']:8.2f}")

    return "\n".join([
        f"[mm_latency] condition={report['condition_id']} asset={report['asset_id']}",
        f"[mm_latency] probes={report['probes_attempted']} accepted_samples={report['samples']}",
        line("raw", raw),
        line("trimmed", trm),
        f"[mm_latency] → ConstantLatency(round_trip={fit['constant_ms']:.1f})  (politics)",
        f"[mm_latency] → SampledLatency(mean={fit['sampled']['mean']:.1f}, "
        f"std={fit['sampled']['std']:.1f})  (fast markets)",
        f"[mm_latency] → set POLYMARKET_MM_BRIDGE_LATENCY_MS={fit['constant_ms']:.0f}",
        f"[mm_latency] measures: {report['measures']}",
        f"[mm_latency] excludes: {report['excludes']}",
    ])


# --------------------------------------------------------------------------------------
# operator entry (`--mode mm_latency`) — DRY-RUN by default, like the bridge CLI
# --------------------------------------------------------------------------------------

def main(env: Mapping[str, str] | None = None) -> int:
    src: Mapping[str, str] = os.environ if env is None else env

    try:
        config = ExecutionConfig.from_env(src)
        probe_cfg = LatencyProbeConfig.from_env(src)
    except ValueError as exc:
        print(f"[mm_latency:startup] Config error: {exc}", file=sys.stderr)
        return 2

    from polymarket.execution.cli import build_venue_adapter

    from .maker_engine import ClobBookClient, GammaMarketLookup
    from .order_safety import RealOrderGate

    journal = JsonlWriter(config.journal_dir, "mm_latency")
    venue_mode = src.get("POLYMARKET_VENUE", "fake")
    try:
        venue = build_venue_adapter(venue_mode, config, journal=journal)
    except NotImplementedError as exc:
        print(f"[mm_latency:startup] {exc}", file=sys.stderr)
        journal.close()
        return 3
    except ValueError as exc:
        print(f"[mm_latency:startup] {exc}", file=sys.stderr)
        journal.close()
        return 4

    if not probe_cfg.asset_id:
        market = GammaMarketLookup(config.gamma_url).get_market(probe_cfg.condition_id)
        if market is None:
            print("[mm_latency:startup] could not resolve YES token id", file=sys.stderr)
            journal.close()
            return 5
        probe_cfg.asset_id = market.asset_id

    if probe_cfg.out_dir is None:
        probe_cfg.out_dir = config.journal_dir

    timing = TimingVenue(venue)
    gate = RealOrderGate(config=config, journal=journal, label="mm_latency")
    router = VenueOrderRouter(
        venue=timing, config=config, journal=journal, gate=gate,
        order_type="GTC", coid_prefix="lat-",
    )
    book_client = ClobBookClient(config.clob_url)

    def book_fn() -> tuple[float | None, float | None]:
        top = book_client.get_top_of_book(probe_cfg.asset_id)
        return (top.best_bid, top.best_ask) if top is not None else (None, None)

    from polymarket.execution.risk import RiskState

    def risk_state_fn(_asset: str, _price: float) -> RiskState:
        return RiskState(
            current_market_price=_price,
            deployed_usd=0.0,
            deployed_in_market_usd=0.0,
            open_positions_count=0,
            realised_pnl_today_usd=0.0,
            killswitch_present=config.killswitch_path.exists(),
        )

    harness = LatencyProbeHarness(
        router=router, timing=timing, cfg=probe_cfg, journal=journal,
        book_fn=book_fn, risk_state_fn=risk_state_fn,
    )
    print(
        f"[mm_latency:startup] condition={probe_cfg.condition_id} asset={probe_cfg.asset_id} "
        f"venue={venue_mode} samples_target={probe_cfg.samples_target} "
        f"cadence={probe_cfg.cadence_s}s max_real_orders={config.max_real_orders} "
        f"operator_confirm={config.require_operator_confirm}",
        flush=True,
    )
    try:
        report = harness.run()
        print(render_fit(report), flush=True)
    finally:
        stop = getattr(venue, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception as exc:  # noqa: BLE001
                print(f"[mm_latency:shutdown] venue.stop() raised: {exc!r}", file=sys.stderr)
        journal.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
