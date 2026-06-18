#!/usr/bin/env python3
"""Build README showcase charts.

Outputs (git-tracked, each < 300 KB):
  docs/assets/live_trading_overview.png   — live momentum universe, daily closes
                                            rebased to 1.0 + rolling realised vol
  docs/assets/momentum_cpcv_audit.png     — normalized overfitting-audit panel
                                            (haircut %, PBO/DSR probabilities,
                                            synthetic-null ratio)
  docs/assets/momentum_cpcv_runs.png      — what a CPCV run looks like: the
                                            combinatorial split grid + the fan of
                                            stitched OOS portfolio equity paths
                                            (rebased, y-scale omitted by design)

Inputs read locally:
  live_trading/cache/daily/{SYM}_daily.parquet            (git-ignored cache,
                                                           present on dev machines)
  topics/momentum/strategies/momentum_cpcv/oos/overfitting_audit_*.pkl  (tracked)
  topics/momentum/strategies/momentum_cpcv/oos/synthetic_null_mc_*.pkl  (tracked)

Design constraint: charts show shape and relative diagnostics only — prices
rebased to 1.0, haircuts as % of the observed value, probabilities in [0, 1],
and null-relative ratios. No absolute Sharpe / return / PnL value is rendered.

Idempotent: re-running overwrites the two PNGs in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CPCV_DIR = REPO / "topics" / "momentum" / "strategies" / "momentum_cpcv"

# unpickling the synthetic-null results needs the harness module importable
sys.path.insert(0, str(REPO / "infrastructure" / "validation"))

LIVE_ASSETS = ["ADAUSDT", "AVAXUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
LOOKBACK_DAYS = 730
VOL_WINDOW = 30
SEED_SUBSAMPLE = 42  # deterministic path subsample so reruns are idempotent

INK = "#1f2933"
MUTED = "#637083"
BLUE = "#4878d0"
RED = "#d65f5f"
GREEN = "#6acc64"
ASSET_COLORS = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4", "#8c613c"]


def _style(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.tick_params(labelsize=8, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.grid(True, axis="y", lw=0.4, alpha=0.35)


def build_live_trading_overview() -> Path | None:
    cache = REPO / "live_trading" / "cache" / "daily"
    closes = {}
    for sym in LIVE_ASSETS:
        f = cache / f"{sym}_daily.parquet"
        if not f.exists():
            print(f"[skip] {f} missing — run live_trading/update_cache.py first")
            return None
        closes[sym.replace("USDT", "")] = pd.read_parquet(f)["Close"]
    px = pd.DataFrame(closes).dropna()
    px = px.iloc[-LOOKBACK_DAYS:]
    rebased = px / px.iloc[0]
    vol = px.pct_change().rolling(VOL_WINDOW).std() * np.sqrt(365) * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.0, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for i, col in enumerate(rebased.columns):
        ax1.plot(rebased.index, rebased[col], lw=1.3, label=col,
                 color=ASSET_COLORS[i % len(ASSET_COLORS)])
        ax2.plot(vol.index, vol[col], lw=0.9, alpha=0.85,
                 color=ASSET_COLORS[i % len(ASSET_COLORS)])
    ax1.set_yscale("log")
    tick_vals = [0.25, 0.5, 1, 2, 4, 8]
    lo, hi = rebased.min().min(), rebased.max().max()
    ticks = [t for t in tick_vals if lo * 0.9 <= t <= hi * 1.1]
    ax1.set_yticks(ticks, [f"{t:g}x" for t in ticks])
    ax1.minorticks_off()
    ax1.axhline(1.0, color=MUTED, lw=0.7, ls="--")
    _style(ax1, f"Live momentum universe — daily closes rebased to 1.0 at window start "
                f"(last {len(rebased)} days, log scale)")
    ax1.set_ylabel("growth of 1.0 (log)", fontsize=8, color=INK)
    ax1.legend(fontsize=8, ncols=6, frameon=False, loc="upper left")
    _style(ax2, f"Rolling {VOL_WINDOW}d realised volatility (annualised %) — "
                f"the regime spread the validation stack has to survive")
    ax2.set_ylabel("ann. vol %", fontsize=8, color=INK)
    fig.suptitle("", fontsize=1)
    fig.text(0.995, 0.01, "shape only — rebased market data, no strategy output shown",
             ha="right", fontsize=7, color=MUTED, style="italic")
    fig.tight_layout()
    out = OUT_DIR / "live_trading_overview.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    return out


def _latest(pattern: str) -> Path | None:
    hits = sorted((CPCV_DIR / "oos").glob(pattern))
    return hits[-1] if hits else None


def build_momentum_cpcv_audit() -> Path | None:
    audit_pkl = _latest("overfitting_audit_*.pkl")
    mc_pkl = _latest("synthetic_null_mc_*.pkl")
    if audit_pkl is None:
        print("[skip] no overfitting_audit_*.pkl under momentum_cpcv/oos/")
        return None
    audit = pd.read_pickle(audit_pkl)
    per_asset = audit["per_asset"]
    port = audit["portfolio"]
    syms = sorted(per_asset)
    labels = [s.replace("usdt", "").upper() for s in syms]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.0, 4.3))

    # ── panel A: selection haircut as % of the observed value ────────────────
    haircut_pct = [100 * per_asset[s]["sr_star_ann"] / per_asset[s]["selected_sharpe_ann"]
                   for s in syms]
    port_pct = 100 * port["sr_star_p"] / port["headline_sharpe"]
    port_pct_total = 100 * port["sr_star_p_total"] / port["headline_sharpe"]
    x = np.arange(len(labels) + 1)
    ax1.bar(x, haircut_pct + [port_pct], 0.6, color=[BLUE] * len(labels) + [RED])
    ax1.scatter([x[-1]], [port_pct_total], marker="v", color=INK, zorder=3, s=40,
                label="harshest variant (full 11,200-trial search)")
    ax1.set_xticks(x, labels + ["PORT"])
    ax1.set_ylabel("haircut, % of observed", fontsize=8, color=INK)
    ax1.set_ylim(0, max(port_pct_total, max(haircut_pct)) * 1.35)
    ax1.legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax1, "Selection haircut (DSR)\n% of each sleeve's observed Sharpe explained\nby picking the best of a 400-trial search")

    # ── panel B: PBO vs DSR probability ──────────────────────────────────────
    pbo = [per_asset[s]["pbo"] for s in syms]
    dsr_p = [per_asset[s]["dsr_prob"] for s in syms]
    xb = np.arange(len(labels))
    ax2.bar(xb - 0.18, pbo, 0.36, color=RED, label="PBO (lower is better)")
    ax2.bar(xb + 0.18, dsr_p, 0.36, color=GREEN, label="DSR prob (higher is better)")
    ax2.axhline(0.5, color=INK, lw=0.8, ls="--")
    ax2.text(len(labels) - 0.45, 0.515, "PBO gate 0.5", fontsize=7, color=INK, ha="right")
    ax2.set_xticks(xb, labels)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("probability", fontsize=8, color=INK)
    ax2.legend(fontsize=7, frameon=False, loc="upper left", ncols=1)
    _style(ax2, "The split verdict\nDSR: family edge is real (all ≈ 1.0) ·\nPBO: per-fold winner picking is noise (all > 0.5)")

    # ── panel C: synthetic-null Monte Carlo, portfolio, primary null ─────────
    if mc_pkl is not None:
        mc = pd.read_pickle(mc_pkl)
        res = mc["variants"]["primary_block10"]["portfolio"]
        null_stats = np.asarray(res.null_stats, dtype=float)
        q95 = float(np.quantile(null_stats, 0.95))
        ax3.hist(null_stats / q95, bins=30, color=BLUE, alpha=0.75,
                 label=f"null max-of-search ({len(null_stats)} synthetic paths)")
        ax3.axvline(1.0, color=INK, lw=0.9, ls="--", label="null 95th percentile (gate)")
        ax3.axvline(res.real_stat / q95, color=RED, lw=1.6,
                    label="real search result")
        ax3.set_xlabel("best-of-search statistic ÷ null 95th pct", fontsize=8, color=INK)
        ax3.set_ylabel("synthetic paths", fontsize=8, color=INK)
        ax3.legend(fontsize=7, frameon=False, loc="upper left")
        _style(ax3, "Synthetic-null Monte Carlo (portfolio)\ncan the same search pipeline manufacture the\nreal result on edge-free bootstrap data?")
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "synthetic_null_mc_*.pkl not found", ha="center", fontsize=9)

    fig.text(0.995, 0.01, "normalized diagnostics only — no absolute performance values shown",
             ha="right", fontsize=7, color=MUTED, style="italic")
    fig.tight_layout()
    out = OUT_DIR / "momentum_cpcv_audit.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def build_momentum_cpcv_runs() -> Path | None:
    from itertools import combinations
    from matplotlib.ticker import NullFormatter

    paths_pkl = CPCV_DIR / "oos" / "portfolio_cpcv_paths.pkl"
    if not paths_pkl.exists():
        print(f"[skip] {paths_pkl} missing")
        return None
    paths = pd.read_pickle(paths_pkl)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.0, 1.9]},
    )

    # ── panel A: the combinatorial split structure (N=8 choose k=2) ──────────
    N, K = 8, 2
    splits = list(combinations(range(N), K))
    grid = np.zeros((len(splits), N))
    for i, test in enumerate(splits):
        for g in test:
            grid[i, g] = 1.0
    ax1.imshow(grid, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#e8ecf1", BLUE]),
               interpolation="nearest")
    ax1.set_xticks(range(N), [f"G{g+1}" for g in range(N)])
    ax1.set_yticks(range(0, len(splits), 4), [f"{s+1}" for s in range(0, len(splits), 4)])
    ax1.set_xlabel("time groups (contiguous, purged at boundaries)", fontsize=8, color=INK)
    ax1.set_ylabel("split #", fontsize=8, color=INK)
    _style(ax1, f"The combinatorial schedule\nevery pair of the {N} time groups serves as the\ntest set once — {len(splits)} purged splits per asset")
    ax1.grid(False)

    # ── panel B: fan of stitched OOS portfolio equity paths, shape only ──────
    rng = np.random.default_rng(SEED_SUBSAMPLE)
    idx = rng.choice(len(paths), size=min(250, len(paths)), replace=False)
    med = sorted(paths, key=lambda p: p["sharpe"])[len(paths) // 2]
    for i in idx:
        eq = (1.0 + paths[i]["portfolio_returns"].fillna(0)).cumprod()
        ax2.plot(eq.index, eq.values, lw=0.5, alpha=0.18, color=BLUE)
    eq_med = (1.0 + med["portfolio_returns"].fillna(0)).cumprod()
    ax2.plot(eq_med.index, eq_med.values, lw=1.6, color=RED, label="median path")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(NullFormatter())
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_yticks([])
    ax2.set_ylabel("equity shape, rebased to 1.0 (scale omitted)", fontsize=8, color=INK)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax2, f"{len(idx)} of {len(paths)} stitched out-of-sample portfolio equity paths\n"
                f"every bar is out-of-sample in every path — the verdict is a distribution, not one backtest")
    fig.text(0.995, 0.01, "structural display — y-axis scale intentionally omitted (shape, not returns)",
             ha="right", fontsize=7, color=MUTED, style="italic")
    fig.tight_layout()
    out = OUT_DIR / "momentum_cpcv_runs.png"
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


def main() -> int:
    built, skipped = [], []
    for fn in (build_live_trading_overview, build_momentum_cpcv_audit,
               build_momentum_cpcv_runs):
        out = fn()
        (built if out else skipped).append(fn.__name__ if out is None else out)
    for out in built:
        kb = out.stat().st_size / 1024
        flag = "" if kb < 300 else "  ** OVER 300 KB — shrink before committing **"
        print(f"[ok] {out.relative_to(REPO)}  ({kb:.0f} KB){flag}")
    if skipped:
        print(f"[warn] skipped: {skipped}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
