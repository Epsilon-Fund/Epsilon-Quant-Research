"""Task-5 IS/OOS protocol — event-group splits, windowed costed PnL, group-CSCV PBO, DSR.

This module pins the load-bearing methodology from the Task-5 PRD so every config in the
ladder is evaluated identically:

* **Unit of observation = the event / NegRisk group, never the token.** Complementary legs
  of one event share a resolution fingerprint (Petro+Starmer are literally one NegRisk
  event; same-condition tokens are exact complements), so token-level splits leak. All CIs
  on cross-config comparisons resample GROUPS.
* **Temporal walk-forward (carry-forward) = the primary ship gate.** Knobs are selected on
  the IS window only; the reported verdict is the OOS bracket. The boundary carries a purge
  (markout horizon) + embargo. Categories (politics vs esports) are split and gated
  separately; cross-regime transfer is a robustness read, not the gate.
* **Costed PnL, not mark-to-mid.** Window PnL = realized round-trips inside the window +
  the change in the *liquidation-marked* inventory (exit at the executable touch: longs
  marked to best bid, shorts to best ask) across the window edges — the carry/exit cost the
  Task-4 naive PnL ignored.
* **Overfitting apparatus live:** group-CSCV PBO (probability the IS-best config is
  OOS-suboptimal, combinatorial splits over event-groups) + DSR (Sharpe deflated by the
  number of configs tried) via the shared ``infrastructure/validation/overfitting_audit``.

Small-K honesty: politics has 6 event groups and esports 7 on this capture — group-level
bootstrap CIs at K≤7 are *approximate* (directional), and are labelled as such wherever
they appear. That is the honest ceiling of an 11-day sample; do not dress it up.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mm_eval.metrics import CENTS, CI, block_bootstrap_mean_ci, compute_markout

# ── The pinned split (pre-registered before any config was run) ────────────────
# IS = 2026-06-19 .. 06-23 (5 days), OOS = 06-24 .. 06-30 (7 days). The boundary puts the
# first Musk-group expiry (06-26) and the second (06-30) in OOS, so the near-expiry regime
# is *faced* out-of-sample exactly as live deployment would face it. Esports groups mostly
# resolve inside IS (matches end 06-19..21) — the esports OOS carries only the 06-25/06-29
# events; that thinness is reported, not hidden.
SPLIT_TS_MS = int(datetime(2026, 6, 24, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
EMBARGO_MS = 3_600_000          # 1 h embargo after the boundary (OOS side)
PRIMARY_HORIZON_S = 30
PURGE_MS = 2 * PRIMARY_HORIZON_S * 1000   # fills whose markout window crosses an edge


@dataclass(frozen=True)
class TokenMeta:
    token_id: str
    universe: str
    market: str        # condition id
    group_id: str      # event / NegRisk group — the split unit
    end_ms: float      # Gamma end_date (τ anchor); nan if unknown
    resolved_payoff: float | None   # actual payoff (0/1) if the market resolved, else None


def _end_ms(rec: dict) -> float:
    ed = rec.get("endDate")
    if not ed:
        return float("nan")
    return datetime.fromisoformat(ed.replace("Z", "+00:00")).timestamp() * 1000.0


def _payoff_for_token(rec: dict, token_id: str) -> float | None:
    """Actual resolution payoff for one CLOB token, from Gamma outcomePrices/clobTokenIds."""
    if not rec.get("closed"):
        return None
    try:
        tokens = json.loads(rec["clobTokenIds"]) if isinstance(rec.get("clobTokenIds"), str) \
            else (rec.get("clobTokenIds") or [])
        prices = json.loads(rec["outcomePrices"]) if isinstance(rec.get("outcomePrices"), str) \
            else (rec.get("outcomePrices") or [])
        for tok, px in zip(tokens, prices):
            if str(tok) == str(token_id):
                return float(px)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def load_token_meta(meta_path: Path, verdict_csv: Path) -> dict[str, TokenMeta]:
    """token_id -> TokenMeta for the Task-4 token set (the A/B universe)."""
    meta = json.loads(Path(meta_path).read_text())
    vd = pd.read_csv(verdict_csv, dtype={"token_id": str})
    out: dict[str, TokenMeta] = {}
    for r in vd.itertuples():
        rec = meta.get(r.market, {})
        out[r.token_id] = TokenMeta(
            token_id=r.token_id, universe=r.universe, market=r.market,
            group_id=str(rec.get("group_id", r.market)), end_ms=_end_ms(rec),
            resolved_payoff=_payoff_for_token(rec, r.token_id),
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Windowed costed PnL (realized + liquidation-marked inventory change)
# ──────────────────────────────────────────────────────────────────────────────

def _touch_series(quotes: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ts, best_bid, best_ask) arrays from the quotes log (rows with a two-sided touch)."""
    rows = [(int(q["ts_exchange"]), float(q["best_bid"]), float(q["best_ask"]))
            for q in quotes
            if q.get("best_bid") is not None and q.get("best_ask") is not None]
    if not rows:
        return (np.empty(0, dtype=np.int64),) + (np.empty(0),) * 2
    rows.sort(key=lambda x: x[0])
    ts = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    bb = np.fromiter((r[1] for r in rows), dtype=float, count=len(rows))
    ba = np.fromiter((r[2] for r in rows), dtype=float, count=len(rows))
    return ts, bb, ba


def _asof_idx(ts: np.ndarray, t: int) -> int:
    return int(np.searchsorted(ts, t, side="right")) - 1


@dataclass(frozen=True)
class WindowCosted:
    """Costed economics of one (token × config × queue) run inside one time window."""

    window: str                 # "IS" | "OOS"
    n_fills: int
    filled_qty: float
    realized_usd: float         # realized round-trip PnL from fills inside the window
    carry_usd: float            # Δ liquidation-marked inventory across the window edges
    costed_usd: float           # realized + carry — the honest window PnL
    costed_per_contract_c: float  # costed_usd / filled_qty, in cents/contract
    net_edge_cents: CI          # markout(30s) per-contract CI, purged at the window edges
    end_inventory: float


def _liq_mark(fills: list[dict], ts: np.ndarray, bb: np.ndarray, ba: np.ndarray,
              t: int) -> float:
    """Liquidation mark of the open inventory at time t (longs→best bid, shorts→best ask)."""
    q = 0.0
    basis = 0.0
    for f in fills:
        if int(f["ts_exchange"]) > t:
            break
        q = float(f["position_after"])
        basis = float(f.get("cost_basis_after", 0.0))
    if abs(q) < 1e-12:
        return 0.0
    i = _asof_idx(ts, t)
    if i < 0:
        return 0.0
    exit_px = bb[i] if q > 0 else ba[i]
    return q * (exit_px - basis)


def windowed_costed(fills: list[dict], quotes: list[dict], *,
                    span: tuple[int, int], seed: int = 0, n_boot: int = 1000,
                    horizon_s: int = PRIMARY_HORIZON_S) -> dict[str, WindowCosted]:
    """IS/OOS costed economics for one run, under the pinned split.

    Fills are time-ordered (engine order). The OOS window opens EMBARGO_MS after the
    boundary; markout at the edges is purged (a fill whose 30 s markout crosses an edge is
    excluded from that window's edge CI, not from the costed PnL).
    """
    t0, t1 = span
    fills = sorted(fills, key=lambda f: int(f["ts_exchange"]))
    ts, bb, ba = _touch_series(quotes)
    mr = compute_markout(fills, quotes, horizons=(horizon_s,))
    m = mr.markout_to_fill[horizon_s]
    fts = mr.fill_ts

    windows = {"IS": (t0, SPLIT_TS_MS), "OOS": (SPLIT_TS_MS + EMBARGO_MS, t1)}
    out: dict[str, WindowCosted] = {}
    for name, (a, b) in windows.items():
        sel = [(f, i) for i, f in enumerate(fills)
               if a <= int(f["ts_exchange"]) < b]
        realized = sum(float(f.get("realized_delta", 0.0)) for f, _ in sel)
        qty = sum(float(f["qty"]) for f, _ in sel)
        mark_a = _liq_mark(fills, ts, bb, ba, a)
        mark_b = _liq_mark(fills, ts, bb, ba, b)
        costed = realized + (mark_b - mark_a)
        # markout CI with edge purge
        idx = np.array([i for _, i in sel], dtype=int)
        if idx.size:
            keep = (fts[idx] <= b - PURGE_MS)
            vals = m[idx[keep]]
            wts = mr.qty[idx[keep]]
            edge = block_bootstrap_mean_ci(vals, wts, seed=seed, n_boot=n_boot).cents()
        else:
            edge = CI(float("nan"), float("nan"), float("nan"), 0)
        end_inv = 0.0
        for f, _ in sel:
            end_inv = float(f["position_after"])
        out[name] = WindowCosted(
            window=name, n_fills=len(sel), filled_qty=qty, realized_usd=realized,
            carry_usd=mark_b - mark_a, costed_usd=costed,
            costed_per_contract_c=(costed / qty * CENTS) if qty > 0 else float("nan"),
            net_edge_cents=edge, end_inventory=end_inv,
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Group-cluster A/B delta (the keep/drop gate) + group-CSCV PBO + DSR
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GroupDelta:
    """Paired per-token config-A-minus-config-B delta, group-cluster bootstrapped."""

    metric: str
    point: float
    lo: float
    hi: float
    n_tokens: int
    n_groups: int
    beats: bool     # lower CI > 0 — A beats B at the group-cluster 95% level

    @property
    def improves(self) -> bool:
        return np.isfinite(self.point) and self.point > 0


def group_delta(df: pd.DataFrame, col_a: str, col_b: str, *, group_col: str = "group_id",
                n_boot: int = 4000, seed: int = 0, metric: str = "") -> GroupDelta:
    """Bootstrap the pooled per-token (A − B) delta by resampling event groups.

    ``df`` has one row per token with the two configs' per-token metric. Token weights are
    equal (a token is one market read); the resample unit is the GROUP (the independent
    unit). K≤7 groups → the CI is approximate/directional; the caller labels it.
    """
    sub = df[[group_col, col_a, col_b]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return GroupDelta(metric, float("nan"), float("nan"), float("nan"), 0, 0, False)
    d = (sub[col_a] - sub[col_b]).to_numpy(dtype=float)
    groups = sub[group_col].to_numpy()
    uniq = np.unique(groups)
    point = float(np.mean(d))
    if uniq.size < 2:
        return GroupDelta(metric, point, float("nan"), float("nan"), int(d.size),
                          int(uniq.size), False)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, uniq.size, replace=True)
        rows = np.concatenate([idx_by[g] for g in pick])
        draws[i] = float(np.mean(d[rows]))
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return GroupDelta(metric, point, float(lo), float(hi), int(d.size), int(uniq.size),
                      bool(np.isfinite(lo) and lo > 0))


@dataclass(frozen=True)
class GroupPBO:
    """CSCV-style PBO over event groups: P(IS-best config is below-median OOS)."""

    pbo: float
    n_splits: int
    n_groups: int
    n_configs: int
    note: str


def group_cscv_pbo(matrix: pd.DataFrame, *, group_col: str = "group_id") -> GroupPBO:
    """Combinatorially split the event groups in half; rank configs on each half.

    ``matrix``: one row per token with ``group_col`` + one column per config (the per-token
    metric, e.g. OOS per-contract costed net). For each C(G, ⌊G/2⌋) split: config ranking =
    mean over the tokens of the half's groups; PBO = share of splits where the IS-half best
    config falls in the bottom half of the OOS-half ranking (López de Prado's CSCV logic
    with event-groups as the exchangeable blocks — legitimate because groups don't share
    resolution fingerprints).
    """
    cfg_cols = [c for c in matrix.columns if c != group_col]
    groups = sorted(matrix[group_col].unique())
    G = len(groups)
    if G < 4 or len(cfg_cols) < 2:
        return GroupPBO(float("nan"), 0, G, len(cfg_cols),
                        "undefined: needs ≥4 groups and ≥2 configs")
    half = G // 2
    n_bad = 0
    n_splits = 0
    for is_groups in combinations(groups, half):
        is_set = set(is_groups)
        is_rows = matrix[matrix[group_col].isin(is_set)]
        oos_rows = matrix[~matrix[group_col].isin(is_set)]
        is_mean = is_rows[cfg_cols].mean()
        oos_mean = oos_rows[cfg_cols].mean()
        if is_mean.isna().all() or oos_mean.isna().all():
            continue
        best = is_mean.idxmax()
        rank = (oos_mean.rank(ascending=True, pct=True))[best]   # 1.0 = best OOS
        if np.isfinite(rank) and rank <= 0.5:
            n_bad += 1
        n_splits += 1
    pbo = n_bad / n_splits if n_splits else float("nan")
    return GroupPBO(pbo, n_splits, G, len(cfg_cols),
                    f"K={G} groups → approximate; directional at this sample size")


def daily_pnl_series(fills: list[dict], quotes: list[dict], *, span: tuple[int, int],
                     window: str = "OOS") -> np.ndarray:
    """Per-UTC-day costed PnL ($) inside a window — the return series DSR deflates."""
    a, b = ((span[0], SPLIT_TS_MS) if window == "IS"
            else (SPLIT_TS_MS + EMBARGO_MS, span[1]))
    fills = sorted(fills, key=lambda f: int(f["ts_exchange"]))
    ts, bb, ba = _touch_series(quotes)
    day_ms = 86_400_000
    d0 = int(a // day_ms)
    d1 = int((b - 1) // day_ms)
    out = []
    for d in range(d0, d1 + 1):
        lo = max(a, d * day_ms)
        hi = min(b, (d + 1) * day_ms)
        realized = sum(float(f.get("realized_delta", 0.0)) for f in fills
                       if lo <= int(f["ts_exchange"]) < hi)
        carry = _liq_mark(fills, ts, bb, ba, hi) - _liq_mark(fills, ts, bb, ba, lo)
        out.append(realized + carry)
    return np.asarray(out, dtype=float)


def _load_audit():
    import sys
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from infrastructure.validation import overfitting_audit as oa
    return oa


def dsr_for_config(daily_pnl: np.ndarray, all_config_daily: dict[str, np.ndarray]) -> dict:
    """Deflated Sharpe of the selected config's daily costed PnL.

    ``all_config_daily`` = every config's daily-PnL series over the same window — the trial
    set whose Sharpe dispersion sets the SR* haircut (n_trials = #configs tried).
    """
    oa = _load_audit()
    r = np.asarray(daily_pnl, dtype=float)
    r = r[np.isfinite(r)]
    n_trials = max(len(all_config_daily), 1)
    if r.size < 5 or np.allclose(r.std(), 0):
        return {"dsr_p": float("nan"), "sharpe_ann": float("nan"), "sr_star_ann": float("nan"),
                "n_days": int(r.size), "n_trials": n_trials,
                "note": "underpowered (<5 daily obs or zero variance)"}
    trial_srs = []
    for series in all_config_daily.values():
        s = np.asarray(series, dtype=float)
        s = s[np.isfinite(s)]
        if s.size >= 3 and s.std(ddof=1) > 0:
            trial_srs.append(float(s.mean() / s.std(ddof=1)))
    if len(trial_srs) < 2:
        return {"dsr_p": float("nan"), "sharpe_ann": float("nan"), "sr_star_ann": float("nan"),
                "n_days": int(r.size), "n_trials": n_trials,
                "note": "fewer than 2 finite trial Sharpes — haircut undefined"}
    res = oa.deflated_sharpe_ratio(r, n_trials=n_trials, periods_per_year=365.0,
                                   trial_sharpes=trial_srs)
    return {"dsr_p": float(res.dsr_prob), "sharpe_ann": float(res.sr_ann),
            "sr_star_ann": float(res.sr_star_ann), "n_days": int(r.size),
            "n_trials": n_trials, "note": f"{r.size} daily obs — thin; DSR is a harsh gate here"}
