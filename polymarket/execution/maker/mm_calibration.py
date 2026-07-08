"""Join-2d calibration pipeline — fit ``ProbQueue.f`` + the latency constant from fills.

The backtest's fill counts are a **bracket** (``OptimisticQueue`` ≥ ``ProbQueue`` ≥
``RiskAverseQueue``) until our own fills pin down how cancel volume should be attributed
ahead-vs-behind our resting orders. This module turns a set of fills (SYNTHETIC in this
build; the operator's real Join-2b/2c fills later) plus the recorded market-event stream
into:

1. a **latency fit** — submit→ack samples → the round-trip constant
   (:func:`~polymarket.execution.maker.mm_latency_harness.fit_latency`), loaded into
   ``ConstantLatency`` and, via ``SampledLatency.calibrate`` (a REAL in-place refit on the
   frozen model), the dispersion model;
2. a **queue fit** — a grid search over ``ProbQueue``'s power-law ``f``: each candidate
   ``f`` replays the SAME strategy over the SAME event stream through the real
   ``run_engine`` (backtest mode), and the ``f`` whose modeled filled quantity best matches
   the observed filled quantity wins;
3. the **bracket-collapse report** — Optimistic / RiskAverse / fitted-Prob replays side by
   side, showing the [RA, Opt] bound and how far the fitted point sits from the observed
   rate: the "bracket collapses toward the live-measured rate" acceptance of the build plan.

**Frozen-code constraint (explicit).** ``QueueModel.calibrate()`` on Alvaro's models is a
frozen Phase-2 STUB (returns ``None`` — see ``mm_engine/queue_models.py``); it cannot fit
anything in place. This pipeline still calls it (the protocol hook is wired end-to-end, as
the PRD asks) and then performs the actual ``f`` selection EXTERNALLY, constructing a fresh
``ProbQueue(f=f_star)`` — no frozen file is modified. ``SampledLatency.calibrate`` is not a
stub and is used as-is.

Everything here is a pure computation over recorded/synthetic data: **no venue, no orders,
no network, no secrets.**
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mm_engine_bridge import ensure_mm_engine_importable

ensure_mm_engine_importable()
from mm_engine.engine import BACKTEST, EngineResult, run_engine  # noqa: E402
from mm_engine.events import GapMarker  # noqa: E402
from mm_engine.fees import FeeModel  # noqa: E402
from mm_engine.interfaces import MarketEvent  # noqa: E402
from mm_engine.latency_models import ConstantLatency, SampledLatency  # noqa: E402
from mm_engine.queue_models import OptimisticQueue, ProbQueue, RiskAverseQueue  # noqa: E402
from mm_engine.strategies import SymmetricQuoter  # noqa: E402

from .mm_latency_harness import fit_latency  # noqa: E402

# Default f grid: log-ish spacing around the shipped prior f=0.5, spanning "cancels mostly
# behind us" (small f → conservative) to "cancels mostly ahead" (large f → optimistic).
DEFAULT_F_GRID: tuple[float, ...] = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class ObservedFills:
    """The observed (live or synthetic) fill totals the fit targets.

    ``filled_qty`` is the primary objective (contracts filled); ``fill_count`` breaks ties.
    Build one with :meth:`from_fill_records` from a fills stream (bridge telemetry or a
    backtest), or directly in tests.
    """

    filled_qty: float
    fill_count: int

    @classmethod
    def from_fill_records(cls, fills: Iterable[Mapping[str, Any]]) -> "ObservedFills":
        qty = 0.0
        n = 0
        for rec in fills:
            q = rec.get("qty")
            if q is None:
                continue
            qty += float(q)
            n += 1
        return cls(filled_qty=qty, fill_count=n)


def _replay(
    feed: Sequence[MarketEvent | GapMarker],
    *,
    queue_model,
    latency_ms: float,
    params: dict,
    strategy=None,
) -> EngineResult:
    """One deterministic backtest replay with the given queue model (same code as JOIN-1)."""
    return run_engine(
        list(feed),
        strategy=strategy if strategy is not None else SymmetricQuoter(),
        queue_model=queue_model,
        latency_model=ConstantLatency(latency_ms),
        mode=BACKTEST,
        params=params,
        fee_model=FeeModel.fee_free_model(),
    )


def fit_prob_queue_f(
    feed: Sequence[MarketEvent | GapMarker],
    observed: ObservedFills,
    *,
    params: dict,
    latency_ms: float = 0.0,
    f_grid: Sequence[float] = DEFAULT_F_GRID,
    strategy_factory=None,
) -> dict[str, Any]:
    """Grid-fit ``ProbQueue.f``: minimize |modeled filled qty − observed filled qty|.

    Each candidate replays the full event stream through ``run_engine`` with a FRESH
    ``ProbQueue(f)`` (queue models are stateful — never reuse one across replays). Ties on
    the qty objective break on fill-count distance, then on the smaller ``f`` (the more
    conservative attribution).

    ``strategy_factory`` (optional) supplies a fresh strategy per replay for stateful
    strategies; the default ``SymmetricQuoter`` is stateless.
    """
    rows: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    for f in f_grid:
        strat = strategy_factory() if strategy_factory is not None else None
        res = _replay(feed, queue_model=ProbQueue(f=f), latency_ms=latency_ms,
                      params=params, strategy=strat)
        row = {
            "f": float(f),
            "modeled_filled_qty": res.filled_qty,
            "modeled_fill_count": float(res.fill_count),
            "qty_gap": abs(res.filled_qty - observed.filled_qty),
            "count_gap": abs(res.fill_count - observed.fill_count),
        }
        rows.append(row)
        if best is None or (row["qty_gap"], row["count_gap"], row["f"]) < (
            best["qty_gap"], best["count_gap"], best["f"]
        ):
            best = row
    assert best is not None, "empty f_grid"
    return {"f_star": best["f"], "best": best, "grid": rows}


def bracket_collapse_report(
    feed: Sequence[MarketEvent | GapMarker],
    observed: ObservedFills,
    *,
    params: dict,
    latency_samples_ms: Sequence[float] = (),
    f_grid: Sequence[float] = DEFAULT_F_GRID,
    strategy_factory=None,
) -> dict[str, Any]:
    """The end-to-end Join-2d pipeline: latency fit → f fit → bracket-collapse report.

    Steps (each numbered key appears in the returned report):

    1. ``latency`` — :func:`fit_latency` over the submit→ack samples (empty ⇒ the 0ms JOIN-1
       stub is kept and flagged). The trimmed mean becomes the replay latency constant, and
       a ``SampledLatency`` is refit in place via its REAL ``calibrate`` hook.
    2. ``calibrate_hook`` — ``QueueModel.calibrate(observed)`` is invoked on a ``ProbQueue``
       (the frozen no-op stub; wired so the protocol path is exercised end-to-end).
    3. ``fit`` — the external grid fit of ``f`` (:func:`fit_prob_queue_f`).
    4. ``bracket`` — Optimistic / RiskAverse / ProbQueue(f*) replays: the pre-calibration
       bound ``[ra, opt]`` and the fitted point, with the collapse metrics
       (``bracket_width`` vs ``fitted_gap_to_observed``).
    """
    lat_fit = fit_latency(list(latency_samples_ms))
    latency_ms = lat_fit["constant_ms"] if latency_samples_ms else 0.0

    # the frozen protocol hook, wired end-to-end (a no-op stub by design — see docstring)
    hook_model = ProbQueue()
    hook_result = hook_model.calibrate(observed)

    # SampledLatency.calibrate IS a real refit — exercise it on the same samples.
    sampled = SampledLatency()
    if latency_samples_ms:
        sampled.calibrate(list(latency_samples_ms))

    fit = fit_prob_queue_f(
        feed, observed, params=params, latency_ms=latency_ms, f_grid=f_grid,
        strategy_factory=strategy_factory,
    )
    f_star = fit["f_star"]

    def one(queue_model) -> dict[str, float]:
        strat = strategy_factory() if strategy_factory is not None else None
        res = _replay(feed, queue_model=queue_model, latency_ms=latency_ms,
                      params=params, strategy=strat)
        return {"filled_qty": res.filled_qty, "fill_count": float(res.fill_count)}

    opt = one(OptimisticQueue())
    ra = one(RiskAverseQueue())
    fitted = one(ProbQueue(f=f_star))

    report = {
        "latency": {
            **lat_fit,
            "applied_constant_ms": latency_ms,
            "samples": len(latency_samples_ms),
            "sampled_latency_after_calibrate": {"mean": sampled.mean, "std": sampled.std},
            "note": (
                "no latency samples — replays keep the 0ms JOIN-1 stub"
                if not latency_samples_ms else "trimmed mean applied to all replays"
            ),
        },
        "calibrate_hook": {
            "invoked": True,
            "returned": hook_result,   # None — the frozen Phase-2 stub; fit is external
            "note": "QueueModel.calibrate is a frozen no-op stub; f is fitted externally",
        },
        "observed": {"filled_qty": observed.filled_qty, "fill_count": observed.fill_count},
        "fit": fit,
        "bracket": {
            "optimistic": opt,
            "risk_averse": ra,
            "fitted_prob": fitted,
            "f_star": f_star,
            "bracket_width_qty": opt["filled_qty"] - ra["filled_qty"],
            "fitted_gap_to_observed_qty": abs(fitted["filled_qty"] - observed.filled_qty),
            "bracket_holds": ra["filled_qty"] - 1e-9
            <= fitted["filled_qty"]
            <= opt["filled_qty"] + 1e-9,
        },
    }
    return report


def render_report(report: dict[str, Any]) -> str:
    """Human-readable bracket-collapse summary (numbers only; no secrets anywhere)."""
    b = report["bracket"]
    obs = report["observed"]
    lat = report["latency"]
    lines = [
        "[mm_calibrate] === Join-2d bracket-collapse report ===",
        f"[mm_calibrate] latency: constant={lat['applied_constant_ms']:.1f}ms "
        f"({lat['samples']} samples; {lat['note']})",
        f"[mm_calibrate] observed fills: qty={obs['filled_qty']:.1f} n={obs['fill_count']}",
        f"[mm_calibrate] bracket BEFORE calibration: "
        f"RiskAverse={b['risk_averse']['filled_qty']:.1f} ≤ "
        f"Optimistic={b['optimistic']['filled_qty']:.1f} "
        f"(width {b['bracket_width_qty']:.1f} contracts)",
        f"[mm_calibrate] fitted ProbQueue(f={b['f_star']:g}): "
        f"{b['fitted_prob']['filled_qty']:.1f} contracts "
        f"(gap to observed {b['fitted_gap_to_observed_qty']:.2f})",
        f"[mm_calibrate] bracket holds (RA ≤ fitted ≤ Opt): {b['bracket_holds']}",
        "[mm_calibrate] read: the [RA, Opt] bound is the pre-calibration uncertainty; the "
        "fitted point is where live fills pin it. Re-run after every live batch.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# operator entry (`--mode mm_calibrate`) — pure file-in/file-out, no venue, no orders
# --------------------------------------------------------------------------------------

def _load_events_jsonl(path: Path) -> list[MarketEvent | GapMarker]:
    """Load a recorded MarketEvent stream (one JSON object per line).

    Two accepted line shapes: a flat MarketEvent dict (``type/token_id/ts_exchange/...``) —
    the bridge/backtest telemetry convention — or ``{"gap": "<reason>"}`` for a GapMarker.
    """
    out: list[MarketEvent | GapMarker] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "gap" in rec:
            out.append(GapMarker(reason=str(rec["gap"])))
            continue
        out.append(MarketEvent(
            type=str(rec["type"]),
            token_id=str(rec["token_id"]),
            ts_exchange=int(rec["ts_exchange"]),
            ts_local_iso=str(rec.get("ts_local_iso", "")),
            ts_monotonic_ns=int(rec.get("ts_monotonic_ns", 0)),
            payload=dict(rec.get("payload", {})),
        ))
    return out


def _load_fills_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load_latency_samples(path: Path) -> list[float]:
    """Accept the latency harness's samples JSONL (submit_ms per accepted probe)."""
    out: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("accepted") and rec.get("submit_ms") is not None:
            out.append(float(rec["submit_ms"]))
    return out


def main(env: Mapping[str, str] | None = None) -> int:
    src: Mapping[str, str] = os.environ if env is None else env
    events_path = (src.get("POLYMARKET_MM_CALIBRATE_EVENTS", "") or "").strip()
    fills_path = (src.get("POLYMARKET_MM_CALIBRATE_FILLS", "") or "").strip()
    if not events_path or not fills_path:
        print(
            "[mm_calibrate:startup] Config error: POLYMARKET_MM_CALIBRATE_EVENTS and "
            "POLYMARKET_MM_CALIBRATE_FILLS are required (JSONL paths)",
            file=sys.stderr,
        )
        return 2

    try:
        feed = _load_events_jsonl(Path(events_path))
        fills = _load_fills_jsonl(Path(fills_path))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[mm_calibrate:startup] input error: {exc!r}", file=sys.stderr)
        return 2

    latency_raw = (src.get("POLYMARKET_MM_CALIBRATE_LATENCY_SAMPLES", "") or "").strip()
    latency_samples: list[float] = []
    if latency_raw:
        try:
            latency_samples = _load_latency_samples(Path(latency_raw))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[mm_calibrate:startup] latency-samples error: {exc!r}", file=sys.stderr)
            return 2

    params = {
        "half_spread": float(src.get("POLYMARKET_MM_BRIDGE_HALF_SPREAD", "0.01")),
        "size": float(src.get("MAKER_SIZE_CONTRACTS", "1")),
        "tick": float(src.get("POLYMARKET_MM_BRIDGE_TICK", "0.001")),
    }
    observed = ObservedFills.from_fill_records(fills)
    report = bracket_collapse_report(
        feed, observed, params=params, latency_samples_ms=latency_samples,
    )
    print(render_report(report), flush=True)

    out_path = (src.get("POLYMARKET_MM_CALIBRATE_OUT", "") or "").strip()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[mm_calibrate] report written: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
