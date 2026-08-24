"""CPCV-style temporal stability — is the per-market read driven by ONE window, or robust?

This is the **meaningful-now** use of the cross-validation machinery on a fixed-spread,
0-parameter quoter (per the task brief): NOT an overfitting verdict (there is nothing fit to
overfit — see :mod:`mm_eval.overfitting_hook`), but the honest question "does the headline
per-contract markout hold across the ~39.5 h capture, or is it manufactured by a single burst?"

Method: split the capture window into ``n_blocks`` equal **time** blocks, compute the
quantity-weighted per-contract markout (cents) within each block at the primary horizon, then:

* report the per-block series + its dispersion,
* **sign stability** — the share of (non-empty) blocks whose edge has the same sign as the
  pooled edge,
* **leave-one-block-out (LOBO)** — re-pool excluding each block in turn; flag if dropping any
  single block flips the pooled sign (i.e. the read is concentrated in / destroyed by one block).

Lookahead-free: blocks are wall-clock partitions of already-recorded fills; markout itself is the
same right-censored forward measurement used everywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mm_eval.metrics import CENTS, compute_markout


@dataclass(frozen=True)
class BlockEdge:
    block: int
    t_lo: int
    t_hi: int
    n_fills: int
    edge_cents: float          # qty-weighted mean markout_to_fill in the block
    qty: float


@dataclass(frozen=True)
class StabilityReport:
    horizon_s: int
    n_blocks: int
    pooled_edge_cents: float
    blocks: list[BlockEdge]
    n_nonempty_blocks: int
    sign_stability: float           # share of non-empty blocks agreeing in sign with the pool
    lobo_min_edge_cents: float      # most adverse leave-one-block-out pooled edge
    lobo_max_edge_cents: float
    lobo_sign_flips: bool           # does any single-block removal flip the pooled sign?
    concentration_note: str


def temporal_stability(
    fills: list[dict], quotes: list[dict], *, horizon_s: int = 30, n_blocks: int = 8,
    span: tuple[int, int] | None = None,
) -> StabilityReport:
    """Per-time-block markout stability for one (token × queue-model) run."""
    mr = compute_markout(fills, quotes, horizons=(horizon_s,))
    m = mr.markout_to_fill[horizon_s]
    q = mr.qty
    ts = mr.fill_ts
    valid = np.isfinite(m) & np.isfinite(q) & (q > 0)
    m, q, ts = m[valid], q[valid], ts[valid]

    def wmean(vals, wts):
        s = np.sum(wts)
        return float(np.sum(vals * wts) / s * CENTS) if s > 0 else float("nan")

    if m.size == 0:
        return StabilityReport(horizon_s, n_blocks, float("nan"), [], 0, float("nan"),
                               float("nan"), float("nan"), False, "no non-censored fills")

    pooled = wmean(m, q)
    t0, t1 = span or (int(ts.min()), int(ts.max()))
    edges = np.linspace(t0, t1, n_blocks + 1)
    # assign each fill to a block [edges[k], edges[k+1]); last block is closed on the right
    bidx = np.clip(np.searchsorted(edges[1:-1], ts, side="right"), 0, n_blocks - 1)

    blocks: list[BlockEdge] = []
    for k in range(n_blocks):
        sel = bidx == k
        if not sel.any():
            blocks.append(BlockEdge(k, int(edges[k]), int(edges[k + 1]), 0, float("nan"), 0.0))
            continue
        blocks.append(BlockEdge(k, int(edges[k]), int(edges[k + 1]), int(sel.sum()),
                                wmean(m[sel], q[sel]), float(np.sum(q[sel]))))

    nonempty = [b for b in blocks if b.n_fills > 0 and np.isfinite(b.edge_cents)]
    if nonempty and np.isfinite(pooled) and pooled != 0:
        agree = sum(1 for b in nonempty if np.sign(b.edge_cents) == np.sign(pooled))
        sign_stability = agree / len(nonempty)
    else:
        sign_stability = float("nan")

    # leave-one-block-out pooled edge
    lobo = []
    for k in range(n_blocks):
        sel = bidx != k
        if sel.any():
            lobo.append(wmean(m[sel], q[sel]))
    lobo = [x for x in lobo if np.isfinite(x)]
    lobo_min = float(min(lobo)) if lobo else float("nan")
    lobo_max = float(max(lobo)) if lobo else float("nan")
    lobo_flip = bool(lobo and np.isfinite(pooled) and any(np.sign(x) != np.sign(pooled) for x in lobo if x != 0))

    if not np.isfinite(sign_stability):
        note = "too few fills for a stability read"
    elif lobo_flip:
        note = "FRAGILE-IN-TIME: removing one block flips the pooled sign (concentrated in a window)"
    elif sign_stability >= 0.75:
        note = "stable: sign holds across ≥75% of blocks and no single-block removal flips it"
    else:
        note = "mixed: sign varies across blocks but no single-block removal flips the pool"

    return StabilityReport(horizon_s, n_blocks, pooled, blocks, len(nonempty), sign_stability,
                           lobo_min, lobo_max, lobo_flip, note)
