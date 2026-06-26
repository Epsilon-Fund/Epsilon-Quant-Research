"""Copy-execution slippage helpers shared across the Phase-5 evaluators (Block SPREAD-2).

Three capabilities, all pure / importable / unit-tested:

  1. ``mid_at_trade`` attach — as-of join of a fill anchor to the lookahead-free
     CLOB ``/prices-history`` midpoint strictly before the fill (the same mid the
     SPREAD-1 surface was built and trade-time-validated on), plus the
     ``leader_vs_mid_cents`` decomposition. The fetch primitive is reused from
     ``scripts.spread_surface_build.fetch_mid_history`` (cached, idempotent); this
     module only owns the pure as-of attach + the K5-category mapping.

  2. ``surface_fallback_cents`` — the category-gated replacement for the flat-3c
     fallback slippage. Uses the FROZEN SPREAD-1 surface (``lib.spread_surface``)
     ONLY for the categories that cleared a SPREAD-1b validation bar
     (``SURFACE_VALIDATED_CATEGORIES``); every other category keeps flat-3c. Full
     spread (2x the half-spread) when the leader fill was maker-side — the copy
     must cross to take that side — half otherwise; floored/capped to [0.5c, 8c].
     For contamination-flagged cells (``frac_negative > 0.4``) it uses the
     SPREAD-1b hybrid BOUNCE replacement level (Roll lost the hybrid arm).

  3. ``decompose_next_fill_slippage`` — splits an observed next-fill copy cost
     into a spread component (the surface's predicted half-spread) and a drift
     component (how far the copy landed beyond mid + predicted half-spread). A
     copy fill beyond mid + half-spread is flagged drift-dominated.

  4. ``mtm_equity_curve`` — lookahead-free daily mark-to-market equity for a copy
     ledger, marking each open position to its forward-filled token mid
     (``t <= grid``) and to resolution at end_date. Returns the daily equity
     series + MTM Sharpe + max drawdown.

Validation provenance: the gated category set and the bounce-over-Roll hybrid
choice are the PASS results of Block SPREAD-1b
(``spread_surface_tradetime_regate_findings``). politics_negrisk is EXCLUDED —
it had only 2 validation cells (no power).
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lib.spread_surface import (
    CATEGORIES,
    CATEGORY_CASE_SQL,
    TICK_FLOOR_CENTS,
    SpreadSurface,
    price_bucket,
)

# ---------------------------------------------------------------------------
# Pre-registered SPREAD-2 constants (provenance: SPREAD-1b PASS)
# ---------------------------------------------------------------------------
# K5 categories that cleared a SPREAD-1b trade-time bar (pooled MedAE 0.75c,
# fast-crypto 0.80c, sign test 71.7% vs flat-3c). politics_negrisk is EXCLUDED
# (n=2 validation cells — unvalidated); it keeps flat-3c.
SURFACE_VALIDATED_CATEGORIES: frozenset[str] = frozenset({
    "crypto_4h", "daily_crypto", "geopolitics", "sports", "tech", "other",
})

CONTAMINATION_FLAG: float = 0.4        # frac_negative threshold for the bounce swap
SURFACE_FALLBACK_FLOOR_C: float = 0.5  # pre-registered floor/cap on the fallback cents
SURFACE_FALLBACK_CAP_C: float = 8.0
FLAT_FALLBACK_CENTS: float = 3.0       # incumbent comparison arm (kept everywhere)


# ---------------------------------------------------------------------------
# K5 category mapping (verbatim taxonomy via the frozen CATEGORY_CASE_SQL)
# ---------------------------------------------------------------------------
def k5_category(df: pd.DataFrame, slug_col: str = "slug",
                question_col: str | None = None,
                neg_risk_col: str | None = None) -> pd.Series:
    """Map rows to the K5 spread-surface category using the SAME SQL CASE the
    surface was built and validated with (so gating is consistent with SPREAD-1/1b).

    ``question``/``neg_risk`` default to ''/false when the column is absent.
    politics_negrisk requires neg_risk=TRUE, so callers that want it resolved
    correctly MUST pass a real ``neg_risk_col`` (otherwise a NegRisk-politics
    market would fall through to 'other' and wrongly become validated)."""
    import duckdb

    work = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "slug_l": df[slug_col].astype("string").str.lower().fillna(""),
        "question_l": (df[question_col].astype("string").str.lower().fillna("")
                       if question_col and question_col in df.columns else ""),
        "neg_risk": (df[neg_risk_col].fillna(False).astype(bool)
                     if neg_risk_col and neg_risk_col in df.columns else False),
    })
    con = duckdb.connect()
    con.register("rows", work)
    out = con.execute(
        f"SELECT row_id, {CATEGORY_CASE_SQL} AS category FROM rows ORDER BY row_id"
    ).df()
    con.close()
    return pd.Series(out["category"].to_numpy(), index=df.index, name="k5_category")


# ---------------------------------------------------------------------------
# mid_at_trade attach (lookahead-free, strictly-before)
# ---------------------------------------------------------------------------
def asof_mid_before(mid_ts: np.ndarray, mid_p: np.ndarray, fill_epoch: float,
                    max_age_s: float | None = None) -> tuple[float | None, float | None]:
    """Last midpoint with ts strictly < fill_epoch (and within max_age_s if set).
    Returns (mid, age_s) or (None, None)."""
    if len(mid_ts) == 0:
        return None, None
    i = bisect_left(list(mid_ts), fill_epoch)  # strictly before
    if i <= 0:
        return None, None
    age = fill_epoch - float(mid_ts[i - 1])
    if max_age_s is not None and age > max_age_s:
        return None, None
    return float(mid_p[i - 1]), age


def leader_vs_mid_cents(price: float, mid: float | None, direction: str) -> float | None:
    """Signed cents the leader's own fill sat away from mid, adverse-positive in
    the leader's trade direction: BUY above mid is positive (paid up), SELL below
    mid is positive (sold down). None when mid is missing."""
    if mid is None or (isinstance(mid, float) and math.isnan(mid)):
        return None
    s = 1.0 if str(direction).upper() == "BUY" else -1.0
    return float(s * (price - mid) * 100.0)


# ---------------------------------------------------------------------------
# Category-gated surface fallback
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FallbackQuote:
    cents: float            # adverse slippage cents to apply to leader_price
    source: str             # 'surface_fallback' | 'surface_fallback_bounce' | 'flat3c'
    used_bounce: bool
    raw_half_cents: float | None  # surface half-spread before maker doubling / floor-cap


def surface_fallback_cents(
    surface: SpreadSurface,
    category: str,
    price_level: float,
    ttr_hours: float | None,
    trade_rate: float,
    leader_is_maker: bool,
    bounce_lookup: dict[tuple[str, str], float] | None = None,
    flat_cents: float = FLAT_FALLBACK_CENTS,
) -> FallbackQuote:
    """Slippage cents for a fallback (no observed next-fill) copy fill.

    For a SURFACE_VALIDATED category: half-spread from the frozen surface (or the
    SPREAD-1b bounce level for contamination-flagged cells), doubled to a full
    spread when the leader was a maker (the copy crosses to take that side),
    floored/capped to [0.5c, 8c]. For any other category (incl. politics_negrisk):
    the incumbent flat-3c."""
    if category not in SURFACE_VALIDATED_CATEGORIES:
        return FallbackQuote(flat_cents, "flat3c", False, None)

    pred = surface.predict(price_level, ttr_hours, trade_rate, category)
    half = pred.half_spread_cents
    used_bounce = False
    if (bounce_lookup is not None and pred.cell_frac_negative is not None
            and pred.cell_frac_negative > CONTAMINATION_FLAG):
        v = bounce_lookup.get((category, price_bucket(price_level)))
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            half = max(float(v), TICK_FLOOR_CENTS)
            used_bounce = True

    cents = (2.0 * half) if leader_is_maker else half
    cents = min(max(cents, SURFACE_FALLBACK_FLOOR_C), SURFACE_FALLBACK_CAP_C)
    return FallbackQuote(
        cents,
        "surface_fallback_bounce" if used_bounce else "surface_fallback",
        used_bounce,
        half,
    )


def apply_fallback_price(leader_price: float, cents: float, direction: str) -> float:
    """Adverse-slippage copy price: leader_price moved against us by `cents`."""
    fb = cents / 100.0
    return leader_price + fb if str(direction).upper() == "BUY" else leader_price - fb


def reprice_fallback_rows(
    df: pd.DataFrame,
    surface: SpreadSurface,
    bounce: dict[tuple[str, str], float] | None,
    *,
    price_col: str,
    ttr_h_col: str,
    dir_col: str,
    maker_col,
    leader_price_col: str,
    is_fallback,
    category_col: str = "k5_category",
) -> pd.DataFrame:
    """Add the gated surface_fallback re-pricing columns for the FALLBACK rows of
    a copy-evaluator frame (the shared core used by all three Phase-5 entry points
    AND the phase5 driver). next_fill rows are untouched.

    Returns the frame with added columns:
      ``sf_fallback_cents``  surface (or bounce) fallback cents (NaN on non-fallback)
      ``sf_source``          'surface_fallback' | 'surface_fallback_bounce' | 'flat3c'
      ``sf_used_bounce``     bool — a contaminated cell used the SPREAD-1b bounce level
      ``sf_copy_price``      copy price under the gated surface fallback
      ``flat3c_copy_price``  copy price under the incumbent flat-3c (comparison arm)

    Fallback rows are quiet-by-construction (no other print in the next-fill
    window) -> trade_rate=0 -> activity quartile act_q1, the principled bucket for
    a quiet moment. With trade_rate fixed, the surface prediction depends only on
    (category, price_bucket, ttr_bucket, is_maker), so this MEMOIZES on that tuple
    rather than calling predict() per row (the largest leader audit is 5.1M fills)."""
    from lib.spread_surface import price_bucket as _pb, ttr_bucket as _tb

    n = len(df)
    fb_cents = np.full(n, np.nan)
    fb_source = np.array(["next_fill"] * n, dtype=object)
    fb_used_bounce = np.zeros(n, dtype=bool)
    fb_price = np.full(n, np.nan)
    flat_price = np.full(n, np.nan)

    price = df[price_col].to_numpy(dtype=float)
    ttr = df[ttr_h_col].to_numpy(dtype=float)
    cat = df[category_col].to_numpy()
    direction = df[dir_col].astype(str).str.upper().to_numpy()
    leader_px = df[leader_price_col].to_numpy(dtype=float)
    maker = (maker_col.to_numpy(dtype=bool) if hasattr(maker_col, "to_numpy")
             else np.asarray(maker_col, dtype=bool))
    fbmask = (is_fallback.to_numpy(dtype=bool) if hasattr(is_fallback, "to_numpy")
              else np.asarray(is_fallback, dtype=bool))

    memo: dict[tuple, FallbackQuote] = {}
    for i in np.nonzero(fbmask)[0]:
        ttr_i = None if not np.isfinite(ttr[i]) else float(ttr[i])
        key = (cat[i], _pb(float(price[i])), _tb(ttr_i), bool(maker[i]))
        q = memo.get(key)
        if q is None:
            q = surface_fallback_cents(surface, cat[i], float(price[i]), ttr_i,
                                       trade_rate=0.0, leader_is_maker=bool(maker[i]),
                                       bounce_lookup=bounce)
            memo[key] = q
        fb_cents[i] = q.cents
        fb_source[i] = q.source
        fb_used_bounce[i] = q.used_bounce
        fb_price[i] = apply_fallback_price(float(leader_px[i]), q.cents, direction[i])
        flat_price[i] = apply_fallback_price(float(leader_px[i]), FLAT_FALLBACK_CENTS, direction[i])

    out = df.copy()
    out["sf_fallback_cents"] = fb_cents
    out["sf_source"] = fb_source
    out["sf_used_bounce"] = fb_used_bounce
    out["sf_copy_price"] = fb_price
    out["flat3c_copy_price"] = flat_price
    return out


# ---------------------------------------------------------------------------
# Drift-vs-spread decomposition of an observed next-fill copy cost
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlippageSplit:
    copy_vs_mid_c: float        # signed cost of the copy fill vs mid (adverse +)
    spread_c: float             # the surface's predicted half-spread (the spread part)
    drift_c: float              # residual = copy_vs_mid - spread (adverse drift +)
    is_drift: bool              # copy fill landed beyond mid + predicted half-spread


def decompose_next_fill_slippage(
    copy_price: float, mid: float, pred_half_c: float, direction: str,
) -> SlippageSplit | None:
    """Split a next-fill copy cost into spread (predicted half-spread) and drift
    (the residual beyond mid + half). A copy fill beyond mid + half-spread is
    flagged drift-dominated (the market moved between the leader and the copy)."""
    if mid is None or (isinstance(mid, float) and math.isnan(mid)):
        return None
    s = 1.0 if str(direction).upper() == "BUY" else -1.0
    copy_vs_mid_c = float(s * (copy_price - mid) * 100.0)
    drift_c = copy_vs_mid_c - pred_half_c
    return SlippageSplit(copy_vs_mid_c, float(pred_half_c), drift_c, drift_c > 0.0)


# ---------------------------------------------------------------------------
# Lookahead-free mark-to-market equity
# ---------------------------------------------------------------------------
def _asof_value(series_ts: list[float], series_v: list[float], t: float) -> float | None:
    """Last value with ts <= t (forward-fill; never peeks ahead). None if empty
    or nothing yet."""
    if not series_ts:
        return None
    i = bisect_right(series_ts, t)
    return series_v[i - 1] if i > 0 else None


@dataclass(frozen=True)
class MTMResult:
    equity: pd.DataFrame        # columns: date, mtm_equity, realized, unrealized, n_open
    sharpe_daily_ann: float
    max_drawdown_usd: float
    max_drawdown_frac: float


def mtm_equity_curve(
    positions: pd.DataFrame,
    mid_histories: dict[str, list[tuple[float, float]]],
    *,
    entry_ts_col: str = "trade_timestamp",
    resolution_ts_col: str = "resolution_date",
    token_col: str = "outcome_token_id",
    direction_col: str = "leader_direction",
    entry_price_col: str = "copy_price",
    size_token_col: str = "copy_token_amount",
    resolution_price_col: str = "position_resolution",
    freq: str = "D",
) -> MTMResult:
    """Daily lookahead-free MTM equity for a copy ledger.

    Each position contributes, at grid time t:
      * entry_cost (signed cash) once t >= entry_ts;
      * if t < resolution_ts: token_qty * mark(t), where mark = forward-filled
        token mid with ts <= t (the SAME /prices-history mid the surface uses;
        falls back to entry_price before the first quote);
      * if t >= resolution_ts: token_qty * resolution_price (realized).

    Equity(t) = sum over positions of (cash + mark-value). Realized vs unrealized
    is split by whether each position has resolved at t. No future quote ever
    enters a past grid point (bisect_right, ts <= t)."""
    if positions.empty:
        empty = pd.DataFrame(columns=["date", "mtm_equity", "realized", "unrealized", "n_open"])
        return MTMResult(empty, 0.0, 0.0, 0.0)

    # internal column names avoid a leading underscore: DataFrame.itertuples()
    # silently renames underscore-leading fields to positional _1/_2/… .
    pos = pd.DataFrame({
        "entry": pd.to_datetime(positions[entry_ts_col], utc=True),
        "res": pd.to_datetime(positions[resolution_ts_col], utc=True),
        "tok": positions[token_col].astype(str),
        "entrypx": positions[entry_price_col].astype(float),
        "resprice": positions[resolution_price_col].astype(float),
    })
    dirn = positions[direction_col].astype(str).str.upper().to_numpy()
    pos["qty"] = positions[size_token_col].astype(float).to_numpy() * np.where(dirn == "BUY", 1.0, -1.0)
    pos["cash"] = (-pos["qty"]) * pos["entrypx"]  # BUY pays (-), SELL receives (+)

    pre = {tok: ([t for t, _ in s], [v for _, v in s]) for tok, s in mid_histories.items()}

    start = pos["entry"].min().normalize()
    end = pos["res"].max().normalize()
    grid = pd.date_range(start, end, freq=freq, tz="UTC")

    rows = []
    for t in grid:
        t_epoch = t.timestamp()
        realized = 0.0
        unrealized = 0.0
        n_open = 0
        live = pos[pos["entry"] <= t]
        for r in live.itertuples(index=False):
            if t >= r.res:
                realized += r.cash + r.qty * r.resprice
            else:
                ts_list, v_list = pre.get(r.tok, ([], []))
                mark = _asof_value(ts_list, v_list, t_epoch)
                if mark is None:
                    mark = r.entrypx
                unrealized += r.cash + r.qty * mark
                n_open += 1
        rows.append({"date": t, "mtm_equity": realized + unrealized,
                     "realized": realized, "unrealized": unrealized, "n_open": n_open})

    eq = pd.DataFrame(rows)
    daily_ret = eq["mtm_equity"].diff().dropna()
    sd = float(daily_ret.std(ddof=0))
    sharpe = float(daily_ret.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0
    cummax = eq["mtm_equity"].cummax()
    dd = (cummax - eq["mtm_equity"])
    max_dd = float(dd.max()) if len(dd) else 0.0
    denom = float(cummax.replace(0.0, np.nan).abs().max() or 1.0)
    return MTMResult(eq, sharpe, max_dd, max_dd / denom if denom else 0.0)


def mtm_equity_curve_fast(
    fills: pd.DataFrame,
    mid_histories: dict[str, list[tuple[float, float]]],
    *,
    entry_ts_col: str = "entry_ts",
    resolution_ts_col: str = "res_ts",       # NaT = never resolves (stays open, marked)
    token_col: str = "token",
    qty_col: str = "qty",                    # signed: + long tokens, - short
    cash_col: str = "cash",                  # signed entry cash: BUY pays (-), SELL receives (+)
    resolution_price_col: str = "res_price",  # NaN when unresolved
) -> MTMResult:
    """Vectorized lookahead-free daily MTM for large per-fill ledgers (the slow
    reference loop in ``mtm_equity_curve`` is O(grid x fills); this is
    O(fills + tokens x grid), needed for 100k+-fill leader ledgers).

    Semantics match ``mtm_equity_curve``: at grid time t a fill contributes
    realized (cash + qty*res_price) once t >= res_ts, else open
    (cash + qty*mark_t) once t >= entry_ts, with mark_t the forward-filled
    token mid (ts <= t). Leading-edge fallback: grid days before a token's
    first quote use the token's first quote (or, if a token has no quotes at
    all, the fill-weighted average entry price) — a documented approximation
    of the slow version's per-fill entry-price fallback."""
    if fills.empty:
        empty = pd.DataFrame(columns=["date", "mtm_equity", "realized", "unrealized", "n_open"])
        return MTMResult(empty, 0.0, 0.0, 0.0)

    entry = pd.to_datetime(fills[entry_ts_col], utc=True)
    res = pd.to_datetime(fills[resolution_ts_col], utc=True)
    tok = fills[token_col].astype(str).to_numpy()
    qty = fills[qty_col].astype(float).to_numpy()
    cash = fills[cash_col].astype(float).to_numpy()
    res_price = fills[resolution_price_col].astype(float).to_numpy()
    resolved = res.notna().to_numpy() & np.isfinite(res_price)

    start = entry.min().normalize()
    end_candidates = [entry.max()]
    if resolved.any():
        end_candidates.append(res[resolved].max())
    end = max(end_candidates).normalize()
    grid = pd.date_range(start, end, freq="D", tz="UTC")
    _epoch0 = pd.Timestamp("1970-01-01", tz="UTC")
    # unit-robust epoch seconds (datetime64 may be [ns] or [us]; .view is gone in pandas 2.2+)
    g_epoch = ((grid - _epoch0).total_seconds()).to_numpy()
    n_g = len(grid)

    entry_idx = np.searchsorted(
        g_epoch, (entry - _epoch0).dt.total_seconds().to_numpy(), side="left")
    res_idx = np.full(len(fills), n_g, dtype=np.int64)
    if resolved.any():
        res_idx[resolved] = np.searchsorted(
            g_epoch, (res[resolved] - _epoch0).dt.total_seconds().to_numpy(), side="left")
    res_idx = np.minimum(res_idx, n_g)
    entry_idx = np.minimum(entry_idx, n_g)

    # realized(t): step function — each resolved fill adds (cash + qty*res_price) at res_idx
    realized_steps = np.zeros(n_g + 1)
    np.add.at(realized_steps, res_idx[resolved], cash[resolved] + qty[resolved] * res_price[resolved])
    realized_t = np.cumsum(realized_steps[:n_g])

    # open cash + open count via diff arrays over [entry_idx, res_idx)
    cash_diff = np.zeros(n_g + 1)
    np.add.at(cash_diff, entry_idx, cash)
    np.add.at(cash_diff, res_idx, -cash)
    open_cash_t = np.cumsum(cash_diff[:n_g])
    cnt_diff = np.zeros(n_g + 1)
    np.add.at(cnt_diff, entry_idx, 1.0)
    np.add.at(cnt_diff, res_idx, -1.0)
    n_open_t = np.cumsum(cnt_diff[:n_g])

    # open token value: per token, qty_open_t x mark_t
    open_val_t = np.zeros(n_g)
    order = np.argsort(tok, kind="stable")
    tok_sorted = tok[order]
    bounds = np.flatnonzero(np.r_[True, tok_sorted[1:] != tok_sorted[:-1], True])
    for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
        idxs = order[b0:b1]
        t_name = tok_sorted[b0]
        qdiff = np.zeros(n_g + 1)
        np.add.at(qdiff, entry_idx[idxs], qty[idxs])
        np.add.at(qdiff, res_idx[idxs], -qty[idxs])
        q_open = np.cumsum(qdiff[:n_g])
        if not np.any(q_open):
            continue
        series = mid_histories.get(t_name) or []
        if series:
            m_ts = np.array([s[0] for s in series])
            m_p = np.array([s[1] for s in series])
            pos_i = np.searchsorted(m_ts, g_epoch, side="right")
            mark = np.where(pos_i > 0, m_p[np.maximum(pos_i - 1, 0)], m_p[0])  # leading edge -> first quote
        else:
            w = np.abs(qty[idxs])
            avg_entry = (np.abs(cash[idxs]).sum() / w.sum()) if w.sum() else 0.0
            mark = np.full(n_g, avg_entry)
        open_val_t += q_open * mark

    eq = pd.DataFrame({"date": grid,
                       "mtm_equity": realized_t + open_cash_t + open_val_t,
                       "realized": realized_t,
                       "unrealized": open_cash_t + open_val_t,
                       "n_open": n_open_t.astype(int)})
    daily_ret = eq["mtm_equity"].diff().dropna()
    sd = float(daily_ret.std(ddof=0))
    sharpe = float(daily_ret.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0
    cummax = eq["mtm_equity"].cummax()
    dd = cummax - eq["mtm_equity"]
    max_dd = float(dd.max()) if len(dd) else 0.0
    denom = float(cummax.replace(0.0, np.nan).abs().max() or 1.0)
    return MTMResult(eq, sharpe, max_dd, max_dd / denom if denom else 0.0)


def load_bounce_lookup(crosschecks_csv) -> dict[tuple[str, str], float]:
    """(category, price_bucket) -> bounce_half_c from the SPREAD-1 cross-checks CSV
    (the SPREAD-1b hybrid replacement levels)."""
    df = pd.read_csv(crosschecks_csv)
    return {(r.category, r.price_bucket): float(r.bounce_half_c)
            for r in df.itertuples() if pd.notna(r.bounce_half_c)}
