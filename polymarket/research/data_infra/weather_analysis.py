"""Weather tail-harvest analysis — pure module, supersedes the legacy build_weather_nb.py workflow.

Pure computation. No plots, no prints. All functions return DataFrames or dicts
that the caller (notebook, dashboard, or script) renders however it wants.

Reads parquets produced by the sibling script in this package:
    scripts/weather_tail_analysis.py
Output parquets land in:
    data/analysis/
        weather_tail_analysis.parquet         (primary, 24h)
        weather_tail_analysis_48h.parquet     (sidebar, 48h)
        weather_tail_per_instance.parquet     (inst, one row per market×token×barrier)
        weather_universe.parquet              (uni, market-level catalogue)
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
BARRIERS_EE: list[float] = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
SLIPS: list[int] = [0, 1, 2, 3, 5]  # cents per leg
# parents[0]=data_infra, parents[1]=polymarket/research (the package root).
DEFAULT_DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "analysis"
DEFAULT_TRADES_GLOB: str = str(Path(__file__).resolve().parents[1] / "data" / "trades" / "*.parquet")

# Next-fill slippage defaults (phase5_design.md §3.5)
DEFAULT_MIN_SECONDS: int = 15
DEFAULT_MAX_SECONDS: int = 300
DEFAULT_FALLBACK_CENTS: float = 3.0
DEFAULT_SPREAD_CENTS: float = 2.0  # for Session 2.5 bid fallback

# Block SPREAD-2 (opt-in): the trade-time-validated surface fallback. Weather
# markets map to the K5 category 'other' (a SURFACE_VALIDATED category) and
# resolve same-day (ttr < 6h), so when eval_pair(surface_fallback=True) is set,
# the arbitrary flat-3c is replaced by the surface's 'other' taker half-spread
# at the entry trigger price — a single principled constant per run. Default OFF.
_SURFACE_CACHE: dict = {}


def surface_fallback_cents_other(price: float, ttr_hours: float = 3.0) -> float:
    """Gated surface fallback cents for a weather (category 'other') taker fill at
    a trigger price. Lazily loads the frozen SPREAD-1 surface; returns the flat-3c
    default if the surface artifacts are absent."""
    from lib.copy_slippage import load_bounce_lookup, surface_fallback_cents
    from lib.spread_surface import SpreadSurface

    if "surf" not in _SURFACE_CACHE:
        csv = Path(__file__).resolve().parents[1] / "data" / "analysis" / "csv_outputs" / "copytrade"
        try:
            _SURFACE_CACHE["surf"] = SpreadSurface.load(
                csv / "spread_surface_v1_surface.csv", csv / "spread_surface_v1_activity_breaks.csv")
            _SURFACE_CACHE["bounce"] = load_bounce_lookup(csv / "spread_surface_v1_diag_crosschecks.csv")
        except Exception:  # noqa: BLE001 — no surface built -> keep flat-3c
            _SURFACE_CACHE["surf"] = None
    if _SURFACE_CACHE.get("surf") is None:
        return DEFAULT_FALLBACK_CENTS
    q = surface_fallback_cents(_SURFACE_CACHE["surf"], "other", price, ttr_hours,
                               trade_rate=0.0, leader_is_maker=False,
                               bounce_lookup=_SURFACE_CACHE.get("bounce"))
    return q.cents


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def load_weather_results(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the four analysis parquets and attach derived per-signal columns.

    Derived columns attached to PRIMARY and SIDEBAR:
        win_rate           = 1 - chop_rate
        roi_per_signal_pct = edge_per_signal / barrier_price * 100
        kelly_fraction     = edge_per_signal / (1 - barrier_price)   (NaN when p>=1)
        dollar_pnl_pooled  = n_crossed * edge_per_signal

    Returns {'primary': df, 'sidebar': df, 'inst': df, 'uni': df}.
    """
    data_dir = Path(data_dir)
    primary = pd.read_parquet(data_dir / "weather_tail_analysis.parquet")
    sidebar = pd.read_parquet(data_dir / "weather_tail_analysis_48h.parquet")
    inst    = pd.read_parquet(data_dir / "weather_tail_per_instance.parquet")
    uni     = pd.read_parquet(data_dir / "weather_universe.parquet")

    for df in (primary, sidebar):
        df["win_rate"]           = 1 - df["chop_rate"]
        df["roi_per_signal_pct"] = df["edge_per_signal"] / df["barrier_price"] * 100
        df["kelly_fraction"]     = np.where(
            df["barrier_price"] < 1.0,
            df["edge_per_signal"] / (1 - df["barrier_price"]),
            np.nan,
        )
        df["dollar_pnl_pooled"]  = df["n_crossed"] * df["edge_per_signal"]

    return {"primary": primary, "sidebar": sidebar, "inst": inst, "uni": uni}


def pivot_inst_to_wide(inst: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-instance long-form into one row per (market_id, outcome_token_id).

    Columns: market_id, outcome_token_id, fc_NNN (first_cross_ts per barrier),
    mk_NNN (first_cross_maker), tk_NNN (first_cross_taker), ms_NNN (first_cross_maker_side),
    plus min_price, max_price, resolution, end_ts, slug_family. Implemented with
    DuckDB rather than pandas pivot_table because INST has ~400k rows.
    """
    con = duckdb.connect()
    try:
        con.register("inst", inst)
        anchor_exprs = []
        for p in BARRIERS_EE:
            pct = int(round(p * 100))
            cond = f"ROUND(barrier_price, 2) = {p:.2f}"
            anchor_exprs.append(
                f"MAX(CASE WHEN {cond} THEN first_cross_ts END) AS fc_{pct:03d}"
            )
            anchor_exprs.append(
                f"MAX(CASE WHEN {cond} THEN first_cross_maker END) AS mk_{pct:03d}"
            )
            anchor_exprs.append(
                f"MAX(CASE WHEN {cond} THEN first_cross_taker END) AS tk_{pct:03d}"
            )
            anchor_exprs.append(
                f"MAX(CASE WHEN {cond} THEN first_cross_maker_side END) AS ms_{pct:03d}"
            )
        fc_exprs = ",\n            ".join(anchor_exprs)
        q = f"""
        SELECT
            market_id,
            outcome_token_id,
            ANY_VALUE(slug_family)  AS slug_family,
            ANY_VALUE(min_price)    AS min_price,
            ANY_VALUE(max_price)    AS max_price,
            ANY_VALUE(resolution)   AS resolution,
            ANY_VALUE(end_ts)       AS end_ts,
            {fc_exprs}
        FROM inst
        GROUP BY market_id, outcome_token_id
        """
        return con.execute(q).df()
    finally:
        con.close()


# ----------------------------------------------------------------------------
# Next-fill slippage (phase5_design.md §3.5)
# ----------------------------------------------------------------------------
def lookup_next_fills_batch(
    anchors: pd.DataFrame,
    trades_glob: str = DEFAULT_TRADES_GLOB,
    min_seconds: int = DEFAULT_MIN_SECONDS,
    max_seconds: int = DEFAULT_MAX_SECONDS,
) -> pd.DataFrame:
    """Vectorised next-fill lookup, split by ABSOLUTE maker_side (BUY-maker vs SELL-maker).

    Why absolute, not "same/opposite relative to anchor": Polymarket fills with
    maker_side='SELL' cluster at the ask (taker lifts the offer); maker_side='BUY'
    cluster at the bid (taker hits the bid). Returning both sides absolutely lets
    callers map to bid/ask cleanly regardless of anchor side.

    `anchors` columns (required):
        anchor_idx (unique key), market_id, outcome_token_id,
        anchor_ts, anchor_maker, anchor_taker.
        (anchor_maker_side is NOT required here — used only for "different
         counterparty" filtering.)

    Per anchor, two next-fills are searched in
    (anchor_ts + min_seconds, anchor_ts + max_seconds]:

      - bid: maker_side = 'BUY'  (= bid print), by different counterparties
      - ask: maker_side = 'SELL' (= ask print), by different counterparties

    "Different counterparties" = anchor's maker not in {nf.maker, nf.taker} AND
    anchor's taker not in {nf.maker, nf.taker}.

    Returns one row per anchor_idx with columns:
        anchor_idx,
        bid_nf_ts, bid_nf_price,
        ask_nf_ts, ask_nf_price.
    NaN means no qualifying fill in window (caller applies fallback).
    """
    needed = {"anchor_idx", "market_id", "outcome_token_id",
              "anchor_ts", "anchor_maker", "anchor_taker"}
    missing = needed - set(anchors.columns)
    if missing:
        raise ValueError(f"anchors missing columns: {missing}")
    if anchors["anchor_idx"].duplicated().any():
        raise ValueError("anchor_idx must be unique")

    con = duckdb.connect()
    try:
        a = anchors[["anchor_idx", "market_id", "outcome_token_id",
                     "anchor_ts", "anchor_maker", "anchor_taker"]].copy()
        a["anchor_ts"] = pd.to_datetime(a["anchor_ts"])
        con.register("anchors", a)

        q = f"""
        WITH cand AS (
          SELECT
            a.anchor_idx,
            t.timestamp  AS nf_ts,
            t.price      AS nf_price,
            t.maker_side AS nf_maker_side
          FROM anchors a
          JOIN read_parquet('{trades_glob}') t
            ON t.market_id = a.market_id
           AND (CASE WHEN t.maker_asset_id = '0' OR t.maker_asset_id IS NULL
                     THEN t.taker_asset_id ELSE t.maker_asset_id END) = a.outcome_token_id
           AND t.timestamp >  a.anchor_ts + INTERVAL '{int(min_seconds)} second'
           AND t.timestamp <= a.anchor_ts + INTERVAL '{int(max_seconds)} second'
           AND t.maker NOT IN (a.anchor_maker, a.anchor_taker)
           AND t.taker NOT IN (a.anchor_maker, a.anchor_taker)
        ),
        ranked AS (
          SELECT
            anchor_idx, nf_ts, nf_price, nf_maker_side,
            ROW_NUMBER() OVER (
              PARTITION BY anchor_idx, nf_maker_side
              ORDER BY nf_ts ASC
            ) AS rn
          FROM cand
        ),
        first_each AS (SELECT * FROM ranked WHERE rn = 1)
        SELECT
          a.anchor_idx,
          MAX(CASE WHEN f.nf_maker_side = 'BUY'  THEN f.nf_ts    END) AS bid_nf_ts,
          MAX(CASE WHEN f.nf_maker_side = 'BUY'  THEN f.nf_price END) AS bid_nf_price,
          MAX(CASE WHEN f.nf_maker_side = 'SELL' THEN f.nf_ts    END) AS ask_nf_ts,
          MAX(CASE WHEN f.nf_maker_side = 'SELL' THEN f.nf_price END) AS ask_nf_price
        FROM anchors a LEFT JOIN first_each f USING (anchor_idx)
        GROUP BY a.anchor_idx
        """
        out = con.execute(q).df()
    finally:
        con.close()
    return out


def compute_next_fill_price(
    market_id: str,
    outcome_token_id: str,
    anchor_ts,
    anchor_maker: str,
    anchor_taker: str,
    direction: str,
    trades_glob: str = DEFAULT_TRADES_GLOB,
    min_seconds: int = DEFAULT_MIN_SECONDS,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    fallback_price: float | None = None,
) -> tuple[float, str]:
    """Single-anchor next-fill price lookup. Returns (price, source).

    `direction`: 'BUY' or 'SELL'. Returns the next-fill where the maker had that
    side (BUY-maker = bid print; SELL-maker = ask print). Caller picks ASK for
    taker-buys / maker-sells, BID for taker-sells / maker-buys.

    Internally calls the batched lookup with a 1-row anchors frame. For many
    anchors, call `lookup_next_fills_batch` directly — one SQL pass beats N.
    """
    direction = direction.upper()
    if direction not in ("BUY", "SELL"):
        raise ValueError(f"direction must be 'BUY' or 'SELL', got {direction!r}")

    anchors = pd.DataFrame([{
        "anchor_idx": 0,
        "market_id": market_id,
        "outcome_token_id": outcome_token_id,
        "anchor_ts": pd.to_datetime(anchor_ts),
        "anchor_maker": anchor_maker,
        "anchor_taker": anchor_taker,
    }])
    out = lookup_next_fills_batch(
        anchors, trades_glob=trades_glob,
        min_seconds=min_seconds, max_seconds=max_seconds,
    )
    row = out.iloc[0]
    nf_price = row["bid_nf_price"] if direction == "BUY" else row["ask_nf_price"]
    if pd.notna(nf_price):
        return float(nf_price), "next_fill"
    if fallback_price is None:
        raise ValueError("no next fill in window and no fallback_price provided")
    return float(fallback_price), "fallback"


def pooled_metrics_by_barrier(primary: pd.DataFrame) -> pd.DataFrame:
    """Pooled (POOLED_ALL_WEATHER) per-barrier metrics, sorted by barrier_price.

    Returns the canonical columns; caller can rename for display:
      barrier_price, n_crossed, n_crossed_and_crashed,
      chop_rate, win_rate, edge_per_signal,
      roi_per_signal_pct, kelly_fraction, dollar_pnl_pooled.
    """
    cols = [
        "barrier_price", "n_crossed", "n_crossed_and_crashed",
        "chop_rate", "win_rate", "edge_per_signal",
        "roi_per_signal_pct", "kelly_fraction", "dollar_pnl_pooled",
    ]
    pooled = primary[primary["slug_family"] == "POOLED_ALL_WEATHER"]
    return pooled.sort_values("barrier_price").reset_index(drop=True)[cols]


def compute_ftc_metrics(inst: pd.DataFrame) -> pd.DataFrame:
    """Per-token vs first-to-cross-per-slug chop/edge + both-crossed decomposition.

    One row per barrier with columns:
        p, n_per_token, chop_per_token, edge_per_token,
        n_first_to_cross, chop_first_to_cross, edge_first_to_cross,
        n_both_crossed, chop_first_in_both, chop_second_in_both
    """
    con = duckdb.connect()
    try:
        con.register("inst", inst)
        ranked_cte = """
        WITH ranked AS (
            SELECT
                barrier_price AS p,
                market_id,
                CAST(crossed_and_crashed AS DOUBLE) AS chop,
                first_cross_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY market_id, barrier_price
                    ORDER BY first_cross_ts ASC
                ) AS rn,
                COUNT(*) OVER (PARTITION BY market_id, barrier_price) AS n_in_market
            FROM inst
            WHERE crossed
        )
        """
        per_token = con.execute("""
            SELECT barrier_price AS p,
                   COUNT(*) AS n_per_token,
                   AVG(CAST(crossed_and_crashed AS DOUBLE)) AS chop_per_token
            FROM inst WHERE crossed
            GROUP BY barrier_price
        """).df()

        ftc = con.execute(ranked_cte + """
            SELECT p,
                   COUNT(*) AS n_first_to_cross,
                   AVG(chop) AS chop_first_to_cross
            FROM ranked WHERE rn = 1
            GROUP BY p
        """).df()

        both = con.execute(ranked_cte + """
            SELECT p,
                   COUNT(DISTINCT market_id) AS n_both_crossed,
                   AVG(CASE WHEN rn = 1 THEN chop END) AS chop_first_in_both,
                   AVG(CASE WHEN rn = 2 THEN chop END) AS chop_second_in_both
            FROM ranked
            WHERE n_in_market = 2 AND rn IN (1, 2)
            GROUP BY p
        """).df()
    finally:
        con.close()

    out = (per_token
           .merge(ftc, on="p", how="outer")
           .merge(both, on="p", how="outer")
           .sort_values("p").reset_index(drop=True))
    out["edge_per_token"]      = (1 - out["chop_per_token"]) - out["p"]
    out["edge_first_to_cross"] = (1 - out["chop_first_to_cross"]) - out["p"]
    return out[["p",
                "n_per_token", "chop_per_token", "edge_per_token",
                "n_first_to_cross", "chop_first_to_cross", "edge_first_to_cross",
                "n_both_crossed", "chop_first_in_both", "chop_second_in_both"]]


def compute_family_rankings(
    primary: pd.DataFrame,
    p: float = 0.80,
    min_n: int = 30,
) -> pd.DataFrame:
    """Per-family edge/ROI at a chosen barrier, filtered by n_crossed >= min_n.

    POOLED_ALL_WEATHER is excluded; sorted by edge_per_signal descending.
    """
    cols = [
        "slug_family", "n_crossed", "n_crossed_and_crashed",
        "chop_rate", "win_rate", "edge_per_signal", "roi_per_signal_pct",
    ]
    df = primary[
        (primary["barrier_price"] == p)
        & (primary["n_crossed"] >= min_n)
        & (primary["slug_family"] != "POOLED_ALL_WEATHER")
    ]
    return (df.sort_values("edge_per_signal", ascending=False)
              .reset_index(drop=True)[cols])


def slippage_grid(primary: pd.DataFrame, slips: list[int] = SLIPS) -> pd.DataFrame:
    """Pooled edge under (barrier, per-leg slip in cents) — wide frame for heatmap.

    Index = barrier_price (ascending); one column per cents-value in `slips`,
    named ``slip_Nc``. Cell value = edge_per_signal − slip/100.
    """
    pooled = (primary[primary["slug_family"] == "POOLED_ALL_WEATHER"]
              .sort_values("barrier_price")
              .set_index("barrier_price"))
    out = pd.DataFrame(index=pooled.index)
    for s in slips:
        out[f"slip_{s}c"] = pooled["edge_per_signal"] - s / 100.0
    return out


def window_comparison(primary: pd.DataFrame, sidebar: pd.DataFrame) -> pd.DataFrame:
    """Pooled 24h vs 48h comparison, joined on (slug_family, barrier_price).

    Returns one row per barrier with both windows' n_crossed / chop_rate / edge.
    """
    joined = primary.merge(
        sidebar, on=["slug_family", "barrier_price"],
        how="outer", suffixes=("_24h", "_48h"),
    )
    pooled = (joined[joined["slug_family"] == "POOLED_ALL_WEATHER"]
              .sort_values("barrier_price").reset_index(drop=True))
    return pooled[[
        "barrier_price",
        "n_crossed_24h", "chop_rate_24h", "edge_per_signal_24h",
        "n_crossed_48h", "chop_rate_48h", "edge_per_signal_48h",
    ]]


def _build_audit_log(
    df: pd.DataFrame,
    in_col: str,
    out_col: str,
    p_in: float,
    p_out: float,
    tp_mask: np.ndarray,
    min_seconds: int,
    max_seconds: int,
    fallback_cents: float,
    assumed_spread_cents: float,
    trades_glob: str,
) -> pd.DataFrame:
    """Per-trade next-fill prices (PROXIES for execution cost, not literal bid/ask).

    The audit log carries two proxies per leg, named relative to the strategy's
    trade direction (entry = buy, exit = sell):

      next_same_dir_price : price of the next fill where someone made the SAME
                            aggressive trade we'd be making (entry: aggressive
                            buy / taker lift; exit: aggressive sell / taker hit).
                            Tends to be WORSE than the true touch price — the
                            orderbook is consumed by the leader's fill and
                            re-prints elsewhere within the window.

      next_opp_dir_price  : price of the next fill in the OPPOSITE direction
                            (entry: aggressive sell; exit: aggressive buy).
                            Tends to be BETTER than the true passive price —
                            an actual passive maker order at the touch may
                            not have been the one filled.

    Both proxies are noisy. Treat the gap (spread_estimate) as a sensitivity
    range, not a precise estimate.

    Polymarket CTF maker_side semantics used in the lookup:
      maker_side='BUY'  => resting bid was hit by an aggressive seller (sell-aggressor print)
      maker_side='SELL' => resting ask was lifted by an aggressive buyer (buy-aggressor print)

    Direction mapping (relative to absolute lookup outputs bid_nf / ask_nf):
      entry leg (we buy):    same_dir = ask_nf (buy-aggressor),  opp_dir = bid_nf
      exit  leg (we sell):   same_dir = bid_nf (sell-aggressor), opp_dir = ask_nf

    Fallback rules per leg (when no qualifying next fill):
      next_same_dir fallback: trigger + fallback_cents/100
      next_opp_dir  fallback: max(0, next_same_dir - assumed_spread_cents/100)
      cross-aware: if only one leg is real, the synthetic side is built off it
      with a spread of `assumed_spread_cents`. See _resolve_bid_ask.
    """
    mk_in_col, tk_in_col = in_col.replace("fc_", "mk_"), in_col.replace("fc_", "tk_")
    mk_out_col, tk_out_col = out_col.replace("fc_", "mk_"), out_col.replace("fc_", "tk_")

    audit = df[["market_id", "outcome_token_id", "slug_family",
                in_col, mk_in_col, tk_in_col,
                out_col, mk_out_col, tk_out_col,
                "resolution"]].copy().rename(columns={
        in_col:     "entry_anchor_ts",
        mk_in_col:  "entry_anchor_maker",
        tk_in_col:  "entry_anchor_taker",
        out_col:    "exit_anchor_ts",
        mk_out_col: "exit_anchor_maker",
        tk_out_col: "exit_anchor_taker",
    }).reset_index(drop=True)
    audit["tp_fires"]   = tp_mask
    audit["anchor_idx"] = audit.index.values

    entry_anchors = audit[["anchor_idx", "market_id", "outcome_token_id",
                           "entry_anchor_ts", "entry_anchor_maker",
                           "entry_anchor_taker"]].rename(columns={
        "entry_anchor_ts":    "anchor_ts",
        "entry_anchor_maker": "anchor_maker",
        "entry_anchor_taker": "anchor_taker",
    }).dropna(subset=["anchor_ts"])
    entry_nf = lookup_next_fills_batch(
        entry_anchors, trades_glob=trades_glob,
        min_seconds=min_seconds, max_seconds=max_seconds,
    )
    audit = audit.merge(
        entry_nf.rename(columns={
            "bid_nf_ts":    "entry_bid_nf_ts",
            "bid_nf_price": "entry_bid_nf_price",
            "ask_nf_ts":    "entry_ask_nf_ts",
            "ask_nf_price": "entry_ask_nf_price",
        }),
        on="anchor_idx", how="left",
    )

    tp_audit = audit[audit["tp_fires"]].copy()
    if len(tp_audit):
        exit_anchors = tp_audit[["anchor_idx", "market_id", "outcome_token_id",
                                 "exit_anchor_ts", "exit_anchor_maker",
                                 "exit_anchor_taker"]].rename(columns={
            "exit_anchor_ts":    "anchor_ts",
            "exit_anchor_maker": "anchor_maker",
            "exit_anchor_taker": "anchor_taker",
        }).dropna(subset=["anchor_ts"])
        exit_nf = lookup_next_fills_batch(
            exit_anchors, trades_glob=trades_glob,
            min_seconds=min_seconds, max_seconds=max_seconds,
        )
        audit = audit.merge(
            exit_nf.rename(columns={
                "bid_nf_ts":    "exit_bid_nf_ts",
                "bid_nf_price": "exit_bid_nf_price",
                "ask_nf_ts":    "exit_ask_nf_ts",
                "ask_nf_price": "exit_ask_nf_price",
            }),
            on="anchor_idx", how="left",
        )
    else:
        for c in ("exit_bid_nf_ts", "exit_bid_nf_price",
                  "exit_ask_nf_ts", "exit_ask_nf_price"):
            audit[c] = pd.NA

    fb = fallback_cents / 100.0
    spread = assumed_spread_cents / 100.0

    # Entry leg: we BUY → same_dir = ask_nf (buy-aggressor), opp_dir = bid_nf.
    audit["entry_next_same_dir_source"] = np.where(
        audit["entry_ask_nf_price"].notna(), "next_fill", "fallback")
    audit["entry_next_opp_dir_source"] = np.where(
        audit["entry_bid_nf_price"].notna(), "next_fill", "fallback")
    entry_same_resolved, entry_opp_resolved = _resolve_bid_ask(
        ask_real=audit["entry_ask_nf_price"].astype(float),  # ask_nf → entry same_dir
        bid_real=audit["entry_bid_nf_price"].astype(float),  # bid_nf → entry opp_dir
        anchor_price=p_in, fb=fb, spread=spread,
    )
    audit["entry_next_same_dir_price"] = entry_same_resolved
    audit["entry_next_opp_dir_price"]  = entry_opp_resolved

    # Lag (seconds) of the actually-used next fill. NaN when fallback fired.
    entry_anchor_ts_pd = pd.to_datetime(audit["entry_anchor_ts"])
    audit["entry_next_same_dir_lag_seconds"] = (
        pd.to_datetime(audit["entry_ask_nf_ts"]) - entry_anchor_ts_pd
    ).dt.total_seconds()
    audit["entry_next_opp_dir_lag_seconds"] = (
        pd.to_datetime(audit["entry_bid_nf_ts"]) - entry_anchor_ts_pd
    ).dt.total_seconds()

    # Exit leg: we SELL → same_dir = bid_nf (sell-aggressor), opp_dir = ask_nf.
    # _resolve_bid_ask wants (ask=higher-typical, bid=lower-typical). Pass the
    # absolute lookup prices in their natural roles, then relabel: at exit,
    #   helper.ask (= ask_nf, the higher proxy) → exit_next_opp_dir
    #   helper.bid (= bid_nf, the lower  proxy) → exit_next_same_dir
    if audit["tp_fires"].any():
        exit_opp_resolved, exit_same_resolved = _resolve_bid_ask(
            ask_real=audit["exit_ask_nf_price"].astype(float),  # ask_nf → exit opp_dir
            bid_real=audit["exit_bid_nf_price"].astype(float),  # bid_nf → exit same_dir
            anchor_price=p_out, fb=fb, spread=spread,
        )
        audit["exit_next_same_dir_price"] = np.where(audit["tp_fires"], exit_same_resolved, np.nan)
        audit["exit_next_opp_dir_price"]  = np.where(audit["tp_fires"], exit_opp_resolved, np.nan)
    else:
        audit["exit_next_same_dir_price"] = np.nan
        audit["exit_next_opp_dir_price"]  = np.nan

    audit["exit_next_same_dir_source"] = np.where(
        audit["tp_fires"],
        np.where(audit["exit_bid_nf_price"].notna(), "next_fill", "fallback"),
        "n/a",
    )
    audit["exit_next_opp_dir_source"] = np.where(
        audit["tp_fires"],
        np.where(audit["exit_ask_nf_price"].notna(), "next_fill", "fallback"),
        "n/a",
    )

    exit_anchor_ts_pd = pd.to_datetime(audit["exit_anchor_ts"])
    audit["exit_next_same_dir_lag_seconds"] = (
        pd.to_datetime(audit["exit_bid_nf_ts"]) - exit_anchor_ts_pd
    ).dt.total_seconds()
    audit["exit_next_opp_dir_lag_seconds"] = (
        pd.to_datetime(audit["exit_ask_nf_ts"]) - exit_anchor_ts_pd
    ).dt.total_seconds()
    # Mask lag fields on non-TP exit rows.
    audit.loc[~audit["tp_fires"], "exit_next_same_dir_lag_seconds"] = np.nan
    audit.loc[~audit["tp_fires"], "exit_next_opp_dir_lag_seconds"]  = np.nan

    # Spread estimate per leg, per the spec: same_dir − opp_dir.
    # Natural sign: entry positive (ASK−BID), exit negative (BID−ASK).
    audit["spread_estimate_entry"] = (
        audit["entry_next_same_dir_price"] - audit["entry_next_opp_dir_price"]
    )
    audit["spread_estimate_exit"] = (
        audit["exit_next_same_dir_price"] - audit["exit_next_opp_dir_price"]
    )

    # Crossed-market flag — built from RAW next-fill prices (both legs must be
    # actual next_fill, not fallback). The fallback resolver clamps ask>=bid,
    # so post-clamp prices hide genuine crossed observations. Per-leg sign-aware:
    #   entry: anomaly = raw bid_nf > raw ask_nf  (opp_dir > same_dir at entry)
    #   exit:  anomaly = raw ask_nf < raw bid_nf  (opp_dir < same_dir at exit;
    #          equivalently: bid_nf > ask_nf — same numeric condition)
    both_real_entry = audit["entry_ask_nf_price"].notna() & audit["entry_bid_nf_price"].notna()
    audit["crossed_market_entry"] = both_real_entry & (
        audit["entry_bid_nf_price"] > audit["entry_ask_nf_price"]
    )
    both_real_exit = (audit["tp_fires"] &
                      audit["exit_ask_nf_price"].notna() &
                      audit["exit_bid_nf_price"].notna())
    audit["crossed_market_exit"] = both_real_exit & (
        audit["exit_bid_nf_price"] > audit["exit_ask_nf_price"]
    )

    audit["p_in"], audit["p_out"] = p_in, p_out

    # Backward-compat aliases for ftc_tp_sizing.py (which the spec forbids touching).
    # Map the old bid/ask leg names onto the new direction-relative names:
    #   entry: same_dir == ASK (we buy aggressively → SELL-maker print)
    #   exit:  same_dir == BID (we sell aggressively → BUY-maker print)
    audit["entry_ask_price"]  = audit["entry_next_same_dir_price"]
    audit["entry_ask_source"] = audit["entry_next_same_dir_source"]
    audit["exit_bid_price"]   = audit["exit_next_same_dir_price"]
    audit["exit_bid_source"]  = audit["exit_next_same_dir_source"]
    return audit


def _resolve_bid_ask(
    ask_real: pd.Series,
    bid_real: pd.Series,
    anchor_price: float,
    fb: float,
    spread: float,
) -> tuple[pd.Series, pd.Series]:
    """Compute (ask, bid) with cross-aware fallbacks.

    Cases:
      both real:     ask = ask_real, bid = bid_real  (clip to ensure ask >= bid)
      only ask real: bid = max(0, ask - spread)
      only bid real: ask = max(anchor + fb, bid + spread)
      neither real:  ask = anchor + fb, bid = max(0, ask - spread)

    Always enforces bid >= 0 and ask >= bid.
    """
    ask_real = ask_real.astype(float)
    bid_real = bid_real.astype(float)

    have_ask = ask_real.notna()
    have_bid = bid_real.notna()

    # Start with raw real values; then fill the missing leg using the present leg + spread.
    ask = ask_real.copy()
    bid = bid_real.copy()

    only_bid = (~have_ask) & have_bid
    ask = ask.where(~only_bid, np.maximum(anchor_price + fb, bid_real + spread))

    only_ask = have_ask & (~have_bid)
    bid = bid.where(~only_ask, (ask_real - spread).clip(lower=0.0))

    neither = (~have_ask) & (~have_bid)
    ask_neither = pd.Series(anchor_price + fb, index=ask.index)
    bid_neither = (ask_neither - spread).clip(lower=0.0)
    ask = ask.where(~neither, ask_neither)
    bid = bid.where(~neither, bid_neither)

    # Final clamp: if real data has bid > ask (data anomaly — e.g. an old print),
    # nudge ask up to bid. We choose this direction because a stale-ask scenario
    # is the most likely cause; treating both prints as valid is fine since the
    # caller can inspect via the *_source columns.
    ask = np.maximum(ask, bid)
    bid = np.minimum(bid, ask).clip(lower=0.0)
    return ask, bid


def eval_pair(
    p_wide: pd.DataFrame,
    p_in: float,
    p_out: float,
    policy: str = "all",
    family: str | None = None,
    min_seconds: int = DEFAULT_MIN_SECONDS,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    fallback_cents: float = DEFAULT_FALLBACK_CENTS,
    assumed_spread_cents: float = DEFAULT_SPREAD_CENTS,
    trades_glob: str = DEFAULT_TRADES_GLOB,
    return_audit: bool = False,
    surface_fallback: bool = False,
):
    """LONG strategy: buy outcome at p_in, take profit if price reaches p_out, else hold.

    Default pricing model is "next_same_dir" (PROXY: pay the price someone else
    pays to do the same aggressive trade we'd be making):
        entry_price = entry_next_same_dir_price   (next aggressive-buy print)
        exit_price  = exit_next_same_dir_price    on TP (next aggressive-sell print)
                    = resolution (0/1)            on hold-to-resolution

    These are PROXIES, not literal bid/ask quotes. The next_same_dir proxy
    tends to be WORSE than the true touch price (book consumed by leader's
    fill); the next_opp_dir proxy tends to be BETTER than the true passive
    price (actual passive maker may not have been filled). Treat the gap
    between scenarios as a sensitivity range, not a precise estimate.

    Fallback per leg when no qualifying next-fill exists in the window
    (anchor_ts + min_seconds, anchor_ts + max_seconds]:
        next_same_dir fallback: trigger_price + fallback_cents/100
        next_opp_dir  fallback: derived to keep ask>=bid in the helper

    The audit log captures BOTH proxies + lag_seconds + spread_estimate +
    crossed_market flags per leg, so downstream `compute_fill_scenarios` can
    compute next_same_dir / next_opp_dir / midpoint PnL without re-running.

    Returns a dict (and the audit DataFrame if return_audit=True), or None when
    p_in >= p_out or no entries.

    policy:
        'all'            — every (market, token) that level-breaks counts.
        'first_to_cross' — only the first token per market to cross p_in.
    """
    if not (p_in < p_out):
        return None
    # SPREAD-2 (opt-in): replace the flat-3c fallback with the gated surface
    # estimate for category 'other' (weather) at the entry trigger price.
    if surface_fallback:
        fallback_cents = surface_fallback_cents_other(p_in)
    in_col  = f"fc_{int(round(p_in  * 100)):03d}"
    out_col = f"fc_{int(round(p_out * 100)):03d}"
    base = p_wide if family is None else p_wide[p_wide["slug_family"] == family]
    entered = base[in_col].notna() & (base["min_price"] < p_in)
    if entered.sum() == 0:
        return None
    df = base[entered]
    if policy == "first_to_cross":
        df = df.sort_values(in_col).drop_duplicates("market_id", keep="first")

    fc_in, fc_out = df[in_col], df[out_col]
    tp_mask    = (fc_out.notna() & (fc_out > fc_in)).values
    resolution = df["resolution"].fillna(0).astype(float).values
    held       = ~tp_mask
    hold_win   = held & (resolution == 1)
    hold_chop  = held & (resolution == 0)
    n          = len(df)

    audit = _build_audit_log(
        df, in_col, out_col, p_in, p_out, tp_mask,
        min_seconds=min_seconds, max_seconds=max_seconds,
        fallback_cents=fallback_cents,
        assumed_spread_cents=assumed_spread_cents,
        trades_glob=trades_glob,
    )
    audit["resolution"] = resolution
    audit["bucket"]     = np.where(
        tp_mask, "tp", np.where(hold_win, "hold_win", "hold_chop")
    )

    # next_same_dir PnL: pay next_same_dir on entry (buy proxy), receive
    # next_same_dir on TP exit (sell proxy), settle 0/1 on hold.
    entry_p = audit["entry_next_same_dir_price"].astype(float).values
    exit_p  = audit["exit_next_same_dir_price"].astype(float).values
    pnl_next_same_dir = np.where(
        tp_mask,
        exit_p - entry_p,
        np.where(hold_win, 1.0 - entry_p, -entry_p),
    )
    audit["pnl_next_same_dir"] = pnl_next_same_dir

    p_tp        = tp_mask.mean()
    p_hold_win  = hold_win.mean()
    p_hold_chop = hold_chop.mean()

    # Three-bucket decomposition under the next_same_dir scenario.
    tp_gain       = (pnl_next_same_dir[tp_mask].mean()   if tp_mask.any()   else 0.0) * p_tp
    hold_win_gain = (pnl_next_same_dir[hold_win].mean()  if hold_win.any()  else 0.0) * p_hold_win
    chop_drag     = (pnl_next_same_dir[hold_chop].mean() if hold_chop.any() else 0.0) * p_hold_chop
    edge = pnl_next_same_dir.mean()

    entry_slip_cents = (audit["entry_next_same_dir_price"] - p_in) * 100.0
    exit_slip_cents  = ((p_out - audit.loc[tp_mask, "exit_next_same_dir_price"]) * 100.0
                       if tp_mask.any() else pd.Series(dtype=float))

    out = {
        "p_in": p_in, "p_out": p_out,
        "policy": policy, "family": family,
        "fill_assumption": "next_same_dir",
        "min_seconds": min_seconds, "max_seconds": max_seconds,
        "fallback_cents": fallback_cents,
        "fallback_model": "surface_fallback_other" if surface_fallback else "flat",
        "assumed_spread_cents": assumed_spread_cents,
        "n_entries": n,
        "p_tp": p_tp, "p_hold_win": p_hold_win, "p_hold_chop": p_hold_chop,
        "tp_gain": tp_gain, "hold_win_gain": hold_win_gain, "chop_drag": chop_drag,
        "edge": edge,
        "roi_pct": 100 * edge / p_in,
        "mean_entry_slippage_cents": float(entry_slip_cents.mean()),
        "mean_exit_slippage_cents":  float(exit_slip_cents.mean()) if len(exit_slip_cents) else float("nan"),
        "entry_next_same_dir_fallback_rate": float((audit["entry_next_same_dir_source"] == "fallback").mean()),
        "entry_next_opp_dir_fallback_rate":  float((audit["entry_next_opp_dir_source"] == "fallback").mean()),
        "exit_next_same_dir_fallback_rate":  float((audit.loc[tp_mask, "exit_next_same_dir_source"] == "fallback").mean()) if tp_mask.any() else float("nan"),
        "exit_next_opp_dir_fallback_rate":   float((audit.loc[tp_mask, "exit_next_opp_dir_source"] == "fallback").mean()) if tp_mask.any() else float("nan"),
    }
    if return_audit:
        return out, audit
    return out


def compute_fill_scenarios(audit: pd.DataFrame) -> pd.DataFrame:
    """For each trade in the audit log, compute PnL under three fill PROXIES.

    All three are noisy proxies for execution cost — interpret the gap between
    them as a sensitivity range, not a precise estimate.

    For a LONG (buy outcome at p_in, hold or take profit at p_out):
        pnl_next_same_dir   — pay next_same_dir on entry (next buy-aggressor
                              print), receive next_same_dir on TP exit
                              (next sell-aggressor print). Both legs pay the
                              "crossed-the-spread" proxy.
        pnl_next_opp_dir    — pay next_opp_dir on entry (next sell-aggressor
                              print), receive next_opp_dir on TP exit (next
                              buy-aggressor print). Both legs assume you'd
                              have been the passive counterparty.
        pnl_midpoint        — midpoint of the two proxies on each leg.
                              When a leg's market is crossed (per-leg
                              crossed_market flag), the naive midpoint would
                              be WORSE than the aggressive proxy. In that
                              case midpoint is clipped to next_same_dir.

    Invariant enforced: monotonicity per leg-direction
        entry: next_same_dir >= entry_midpoint >= next_opp_dir  (after clip)
        exit:  next_same_dir <= exit_midpoint  <= next_opp_dir  (after clip)
    """
    a = audit.copy()
    tp       = a["bucket"].eq("tp").values
    hold_win = a["bucket"].eq("hold_win").values

    e_same = a["entry_next_same_dir_price"].astype(float).values
    e_opp  = a["entry_next_opp_dir_price"].astype(float).values
    x_same = a["exit_next_same_dir_price"].astype(float).values
    x_opp  = a["exit_next_opp_dir_price"].astype(float).values

    a["pnl_next_same_dir"] = np.where(
        tp, x_same - e_same,
        np.where(hold_win, 1.0 - e_same, -e_same),
    )
    a["pnl_next_opp_dir"] = np.where(
        tp, x_opp - e_opp,
        np.where(hold_win, 1.0 - e_opp, -e_opp),
    )

    # Midpoint with per-leg crossed-market clipping.
    # Entry (we BUY): naive_mid = (e_same + e_opp)/2. If e_opp > e_same
    # (crossed at entry), naive_mid > e_same — worse than aggressive. Clip
    # midpoint = min(naive_mid, e_same).
    entry_mid_naive = (e_same + e_opp) / 2.0
    entry_mid = np.where(e_opp > e_same, np.minimum(entry_mid_naive, e_same), entry_mid_naive)

    # Exit (we SELL): naive_mid = (x_same + x_opp)/2. If x_opp < x_same
    # (crossed at exit, equivalently bid>ask), naive_mid < x_same — worse
    # than aggressive sell. Clip midpoint = max(naive_mid, x_same).
    exit_mid_naive = (x_same + x_opp) / 2.0
    exit_mid = np.where(x_opp < x_same, np.maximum(exit_mid_naive, x_same), exit_mid_naive)

    a["entry_midpoint_price"] = entry_mid
    a["exit_midpoint_price"]  = np.where(tp, exit_mid, np.nan)

    a["pnl_midpoint"] = np.where(
        tp, exit_mid - entry_mid,
        np.where(hold_win, 1.0 - entry_mid, -entry_mid),
    )

    # Post-clip invariants:
    # entry: e_opp ≤ entry_mid ≤ e_same (when not crossed); on crossed,
    # entry_mid = e_same and e_opp > e_same, so the chain breaks at the
    # ≤ on the right of entry_mid. We instead just assert entry_mid ≤ e_same.
    assert np.all(entry_mid <= e_same + 1e-9), "entry midpoint exceeded next_same_dir"
    if tp.any():
        assert np.all(exit_mid[tp] >= x_same[tp] - 1e-9), \
            "exit midpoint fell below next_same_dir"

    return a


def slippage_diagnostic(
    audit: pd.DataFrame,
    p_in: float | None = None,
    p_out: float | None = None,
    high_fallback_threshold: float = 0.50,
    median_spread_thin_cents: float = 5.0,
    anomaly_share_threshold: float = 0.15,
) -> dict:
    """Characterise the next-fill slippage proxies: distribution, lag, spread, reliability.

    For each (leg, direction) of {entry, exit} × {same_dir, opp_dir}, reports:
        n_total, n_fallback, n_next_fill, fallback_pct
        slip_p10/p25/median/p75/p90 in cents — slip is (next_fill_price − trigger_price)
        lag_p10/median/p90 in seconds (NaN where fallback fired)
        lag_under_30s_share, lag_30s_to_2min_share, lag_2min_to_5min_share
        shape: 'unimodal' | 'long_tail' | 'bimodal_or_thin_data'
            heuristic: long_tail if (p90 − median) > 3 × (median − p10)
        interpretation: 'noisy_taker_proxy' | 'constant_slippage_with_labelling'
            interpretation='constant_slippage_with_labelling' when fallback_pct
            > high_fallback_threshold (default 50%). In that regime the scenario
            edge is dominated by the fallback constant; the labelling
            ("next_same_dir" / "next_opp_dir") is mostly cosmetic.

    Also reports per-leg spread_estimate distribution + crossed_market flags
    + reliability flags:
        median_spread_cents
        spread_anomaly_share — fraction with the per-leg crossed-market sign
        flag_thin_markets — median |spread_estimate| > median_spread_thin_cents
        flag_noisy_spread — spread_anomaly_share > anomaly_share_threshold

    trigger_price is p_in for entry legs, p_out for exit legs. If not passed,
    inferred from the audit's `p_in` / `p_out` columns.
    """
    if p_in is None:
        p_in = float(audit["p_in"].iloc[0])
    if p_out is None:
        p_out = float(audit["p_out"].iloc[0])

    def _leg_stats(leg: str, dirn: str, trigger: float, mask: pd.Series) -> dict:
        price_col  = f"{leg}_next_{dirn}_price"
        source_col = f"{leg}_next_{dirn}_source"
        lag_col    = f"{leg}_next_{dirn}_lag_seconds"

        sub_all = audit.loc[mask, [price_col, source_col, lag_col]].copy()
        n_total = int(len(sub_all))
        if n_total == 0:
            return {"leg": leg, "direction": dirn, "n_total": 0,
                    "n_fallback": 0, "n_next_fill": 0, "fallback_pct": float("nan"),
                    "interpretation": "no_data"}

        is_fallback = sub_all[source_col] == "fallback"
        n_fallback  = int(is_fallback.sum())
        n_next_fill = int((sub_all[source_col] == "next_fill").sum())
        fallback_pct = n_fallback / n_total

        # Slippage = price − trigger, in cents.
        slip_cents = (sub_all[price_col].astype(float) - trigger) * 100.0
        p10, p25, med, p75, p90 = np.nanpercentile(slip_cents, [10, 25, 50, 75, 90])

        # Lag stats only over rows where a real next-fill was used.
        lag = sub_all.loc[~is_fallback, lag_col].astype(float)
        if len(lag):
            lag_med = float(np.nanmedian(lag))
            lag_p10 = float(np.nanpercentile(lag, 10))
            lag_p90 = float(np.nanpercentile(lag, 90))
            lag_u30  = float((lag < 30).mean())
            lag_30_120 = float(((lag >= 30) & (lag < 120)).mean())
            lag_120_300 = float((lag >= 120).mean())
        else:
            lag_med = lag_p10 = lag_p90 = float("nan")
            lag_u30 = lag_30_120 = lag_120_300 = float("nan")

        # Shape heuristic on slip distribution.
        upper = p90 - med
        lower = med - p10
        if not np.isfinite(upper) or not np.isfinite(lower) or lower <= 1e-9:
            shape = "bimodal_or_thin_data"
        elif upper > 3 * lower or lower > 3 * upper:
            shape = "long_tail"
        else:
            shape = "unimodal"

        interpretation = ("constant_slippage_with_labelling"
                          if fallback_pct > high_fallback_threshold
                          else "noisy_taker_proxy")

        return {
            "leg": leg, "direction": dirn,
            "n_total": n_total, "n_fallback": n_fallback, "n_next_fill": n_next_fill,
            "fallback_pct": fallback_pct,
            "slip_cents_p10": float(p10), "slip_cents_p25": float(p25),
            "slip_cents_median": float(med),
            "slip_cents_p75": float(p75), "slip_cents_p90": float(p90),
            "lag_median_s": lag_med, "lag_p10_s": lag_p10, "lag_p90_s": lag_p90,
            "lag_under_30s_share": lag_u30,
            "lag_30s_to_2min_share": lag_30_120,
            "lag_2min_to_5min_share": lag_120_300,
            "shape": shape,
            "interpretation": interpretation,
        }

    all_mask = pd.Series(True, index=audit.index)
    tp_mask  = audit["bucket"].eq("tp")

    legs = [
        _leg_stats("entry", "same_dir", p_in,  all_mask),
        _leg_stats("entry", "opp_dir",  p_in,  all_mask),
        _leg_stats("exit",  "same_dir", p_out, tp_mask),
        _leg_stats("exit",  "opp_dir",  p_out, tp_mask),
    ]

    # Spread estimate stats.
    def _spread_stats(leg: str, trigger: float, mask: pd.Series, crossed_col: str) -> dict:
        sub = audit.loc[mask]
        if len(sub) == 0:
            return {"leg": leg, "n": 0}
        spread = sub[f"spread_estimate_{leg}"].astype(float) * 100.0  # cents
        abs_spread = spread.abs()
        n_crossed = int(sub[crossed_col].sum())
        return {
            "leg": leg,
            "n": int(len(sub)),
            "median_signed_cents": float(np.nanmedian(spread)),
            "median_abs_cents":    float(np.nanmedian(abs_spread)),
            "p10_signed_cents":    float(np.nanpercentile(spread, 10)),
            "p90_signed_cents":    float(np.nanpercentile(spread, 90)),
            "n_crossed_market":    n_crossed,
            "crossed_market_share": float(n_crossed / len(sub)),
            "flag_thin_markets":   float(np.nanmedian(abs_spread)) > median_spread_thin_cents,
            "flag_noisy_spread":   float(n_crossed / len(sub)) > anomaly_share_threshold,
        }

    spread = [
        _spread_stats("entry", p_in,  all_mask, "crossed_market_entry"),
        _spread_stats("exit",  p_out, tp_mask,  "crossed_market_exit"),
    ]

    # Overall flags
    any_high_fallback = any(
        leg.get("fallback_pct", 0) > high_fallback_threshold for leg in legs
    )

    return {
        "high_fallback_threshold": high_fallback_threshold,
        "median_spread_thin_cents": median_spread_thin_cents,
        "anomaly_share_threshold": anomaly_share_threshold,
        "any_leg_high_fallback": bool(any_high_fallback),
        "legs": legs,
        "spread": spread,
    }


def compare_windows_diagnostic(
    p_wide: pd.DataFrame,
    p_in: float,
    p_out: float,
    policy: str = "all",
    windows_seconds: tuple[int, ...] = (300, 600),
    min_seconds: int = DEFAULT_MIN_SECONDS,
    fallback_cents: float = DEFAULT_FALLBACK_CENTS,
    assumed_spread_cents: float = DEFAULT_SPREAD_CENTS,
    trades_glob: str = DEFAULT_TRADES_GLOB,
) -> dict:
    """Compare fallback rates and edge under multiple next-fill window caps.

    For each window cap, runs the lookup once (the cost is dominated by the
    parquet scan, so this is roughly linear in len(windows_seconds)). Then for
    each window:
      - per-leg fallback rate
      - lag distribution (for trades where a real fill was found)
      - edge under next_same_dir / next_opp_dir / midpoint (using each window
        as the data source, the SAME fallback constants for missing rows)

    Returns a dict keyed by window-seconds with these per-window stats, plus a
    'window_delta' section comparing the longest window to the shortest:
      - fallback drop per leg (pp)
      - edge change per scenario (cents/share)
      - additional rows that gained a fill in the longer window:
        their lag distribution (i.e. fills in (short_window, long_window])
    """
    if not (p_in < p_out):
        raise ValueError("p_in must be < p_out")
    if min(windows_seconds) < 1:
        raise ValueError("windows_seconds must be positive")

    in_col  = f"fc_{int(round(p_in  * 100)):03d}"
    out_col = f"fc_{int(round(p_out * 100)):03d}"
    base = p_wide
    entered = base[in_col].notna() & (base["min_price"] < p_in)
    df = base[entered].copy()
    if policy == "first_to_cross":
        df = df.sort_values(in_col).drop_duplicates("market_id", keep="first")
    df = df.reset_index(drop=True)

    fc_in, fc_out = df[in_col], df[out_col]
    tp_mask    = (fc_out.notna() & (fc_out > fc_in)).values
    resolution = df["resolution"].fillna(0).astype(float).values
    hold_win   = (~tp_mask) & (resolution == 1)
    n = len(df)

    mk_in_col, tk_in_col   = in_col.replace("fc_", "mk_"),  in_col.replace("fc_", "tk_")
    mk_out_col, tk_out_col = out_col.replace("fc_", "mk_"), out_col.replace("fc_", "tk_")

    # Build entry/exit anchors ONCE.
    entry_anchors = pd.DataFrame({
        "anchor_idx":       df.index.values,
        "market_id":        df["market_id"].astype(str).values,
        "outcome_token_id": df["outcome_token_id"].astype(str).values,
        "anchor_ts":        pd.to_datetime(df[in_col]).values,
        "anchor_maker":     df[mk_in_col].astype(str).values,
        "anchor_taker":     df[tk_in_col].astype(str).values,
    }).dropna(subset=["anchor_ts"])
    exit_anchors = pd.DataFrame({
        "anchor_idx":       df.index.values[tp_mask],
        "market_id":        df.loc[tp_mask, "market_id"].astype(str).values,
        "outcome_token_id": df.loc[tp_mask, "outcome_token_id"].astype(str).values,
        "anchor_ts":        pd.to_datetime(df.loc[tp_mask, out_col]).values,
        "anchor_maker":     df.loc[tp_mask, mk_out_col].astype(str).values,
        "anchor_taker":     df.loc[tp_mask, tk_out_col].astype(str).values,
    }).dropna(subset=["anchor_ts"])

    fb = fallback_cents / 100.0
    spread = assumed_spread_cents / 100.0

    per_window = {}
    for w in sorted(windows_seconds):
        entry_nf = lookup_next_fills_batch(
            entry_anchors, trades_glob=trades_glob,
            min_seconds=min_seconds, max_seconds=w,
        ).set_index("anchor_idx")
        exit_nf = lookup_next_fills_batch(
            exit_anchors, trades_glob=trades_glob,
            min_seconds=min_seconds, max_seconds=w,
        ).set_index("anchor_idx") if len(exit_anchors) else pd.DataFrame(
            columns=["bid_nf_ts", "bid_nf_price", "ask_nf_ts", "ask_nf_price"]
        )

        # Resolve entry prices with fallbacks.
        e_ask_raw = pd.Series(
            entry_nf["ask_nf_price"].reindex(df.index).astype(float).values, index=df.index)
        e_bid_raw = pd.Series(
            entry_nf["bid_nf_price"].reindex(df.index).astype(float).values, index=df.index)
        e_same_resolved, e_opp_resolved = _resolve_bid_ask(
            ask_real=e_ask_raw, bid_real=e_bid_raw,
            anchor_price=p_in, fb=fb, spread=spread,
        )

        # Resolve exit prices with fallbacks (only TP rows).
        if tp_mask.any():
            x_ask_raw = pd.Series(
                exit_nf["ask_nf_price"].reindex(df.index).astype(float).values, index=df.index)
            x_bid_raw = pd.Series(
                exit_nf["bid_nf_price"].reindex(df.index).astype(float).values, index=df.index)
            x_opp_resolved, x_same_resolved = _resolve_bid_ask(
                ask_real=x_ask_raw, bid_real=x_bid_raw,
                anchor_price=p_out, fb=fb, spread=spread,
            )
        else:
            x_same_resolved = pd.Series(np.nan, index=df.index)
            x_opp_resolved  = pd.Series(np.nan, index=df.index)

        e_same = e_same_resolved.values
        e_opp  = e_opp_resolved.values
        x_same = np.where(tp_mask, x_same_resolved.values, np.nan)
        x_opp  = np.where(tp_mask, x_opp_resolved.values,  np.nan)

        # Per-scenario PnL.
        pnl_same = np.where(tp_mask, x_same - e_same,
                            np.where(hold_win, 1.0 - e_same, -e_same))
        pnl_opp  = np.where(tp_mask, x_opp - e_opp,
                            np.where(hold_win, 1.0 - e_opp, -e_opp))
        e_mid_naive = (e_same + e_opp) / 2.0
        e_mid = np.where(e_opp > e_same, np.minimum(e_mid_naive, e_same), e_mid_naive)
        x_mid_naive = (x_same + x_opp) / 2.0
        x_mid = np.where(x_opp < x_same, np.maximum(x_mid_naive, x_same), x_mid_naive)
        pnl_mid = np.where(tp_mask, x_mid - e_mid,
                           np.where(hold_win, 1.0 - e_mid, -e_mid))

        # Fallback rates.
        fb_e_same = float(e_ask_raw.isna().mean())
        fb_e_opp  = float(e_bid_raw.isna().mean())
        if tp_mask.any():
            fb_x_same = float(pd.Series(exit_nf["bid_nf_price"]
                              .reindex(df.index[tp_mask]).astype(float).values).isna().mean())
            fb_x_opp  = float(pd.Series(exit_nf["ask_nf_price"]
                              .reindex(df.index[tp_mask]).astype(float).values).isna().mean())
        else:
            fb_x_same = fb_x_opp = float("nan")

        # Lag distribution for real-fill rows (entry, all directions pooled).
        entry_lags = []
        for col, ts_col in (("ask_nf_price", "ask_nf_ts"), ("bid_nf_price", "bid_nf_ts")):
            sub = entry_nf[entry_nf[col].notna()]
            if len(sub):
                anch_ts = pd.to_datetime(df.loc[sub.index, in_col].values)
                lag = (pd.to_datetime(sub[ts_col].values) - anch_ts).total_seconds()
                entry_lags.extend(lag.tolist())
        entry_lags = np.array(entry_lags, dtype=float)

        if len(entry_lags):
            lag_p10, lag_p25, lag_med, lag_p75, lag_p90 = np.nanpercentile(
                entry_lags, [10, 25, 50, 75, 90]
            )
        else:
            lag_p10 = lag_p25 = lag_med = lag_p75 = lag_p90 = float("nan")

        per_window[w] = {
            "window_seconds": w,
            "n_entries": n,
            "fallback_rate_entry_same_dir": fb_e_same,
            "fallback_rate_entry_opp_dir":  fb_e_opp,
            "fallback_rate_exit_same_dir":  fb_x_same,
            "fallback_rate_exit_opp_dir":   fb_x_opp,
            "edge_next_same_dir": float(np.nanmean(pnl_same)),
            "edge_next_opp_dir":  float(np.nanmean(pnl_opp)),
            "edge_midpoint":      float(np.nanmean(pnl_mid)),
            "entry_lag_p10_s": float(lag_p10),
            "entry_lag_p25_s": float(lag_p25),
            "entry_lag_median_s": float(lag_med),
            "entry_lag_p75_s": float(lag_p75),
            "entry_lag_p90_s": float(lag_p90),
        }

    # Window delta — shortest vs longest.
    w_short = min(per_window.keys())
    w_long  = max(per_window.keys())
    short = per_window[w_short]
    longw = per_window[w_long]
    delta = {
        "short_seconds": w_short,
        "long_seconds":  w_long,
        "fallback_drop_entry_same_dir_pp": (short["fallback_rate_entry_same_dir"]
                                             - longw["fallback_rate_entry_same_dir"]) * 100,
        "fallback_drop_entry_opp_dir_pp":  (short["fallback_rate_entry_opp_dir"]
                                             - longw["fallback_rate_entry_opp_dir"]) * 100,
        "edge_change_same_dir_cents": (longw["edge_next_same_dir"]
                                       - short["edge_next_same_dir"]) * 100,
        "edge_change_opp_dir_cents":  (longw["edge_next_opp_dir"]
                                       - short["edge_next_opp_dir"]) * 100,
        "edge_change_midpoint_cents": (longw["edge_midpoint"]
                                       - short["edge_midpoint"]) * 100,
    }

    return {
        "p_in": p_in, "p_out": p_out, "n_entries": n,
        "windows": per_window,
        "delta_short_to_long": delta,
    }


def time_to_first_any_fill_diagnostic(
    p_wide: pd.DataFrame,
    p_in: float,
    policy: str = "all",
    min_seconds: int = DEFAULT_MIN_SECONDS,
    max_seconds: int = 1800,  # 30 min cap for the diagnostic
    trades_glob: str = DEFAULT_TRADES_GLOB,
) -> dict:
    """For each entry anchor at p_in, find the time to ANY next fill (any side, any
    direction) by a different trader in the same (market, token).

    This is the rawest measure of post-cross market activity — independent of
    maker_side / bid_ask interpretation. If lag percentiles are >>5min, the
    market is simply illiquid and no proxy can recover; if they're <30s, the
    main next-fill model is just looking at the wrong side.

    Returns a dict with:
      n_anchors, n_with_any_fill_within_max
      pct_within_30s, pct_within_60s, pct_within_120s, pct_within_300s,
      pct_within_600s, pct_no_fill_within_max
      lag_p10/p25/median/p75/p90/p99 in seconds (only over anchors that found a fill)
    """
    in_col  = f"fc_{int(round(p_in * 100)):03d}"
    base = p_wide
    entered = base[in_col].notna() & (base["min_price"] < p_in)
    df = base[entered].copy()
    if policy == "first_to_cross":
        df = df.sort_values(in_col).drop_duplicates("market_id", keep="first")
    df = df.reset_index(drop=True)

    mk_col, tk_col = in_col.replace("fc_", "mk_"), in_col.replace("fc_", "tk_")
    anchors = pd.DataFrame({
        "anchor_idx":       df.index.values,
        "market_id":        df["market_id"].astype(str).values,
        "outcome_token_id": df["outcome_token_id"].astype(str).values,
        "anchor_ts":        pd.to_datetime(df[in_col]).values,
        "anchor_maker":     df[mk_col].astype(str).values,
        "anchor_taker":     df[tk_col].astype(str).values,
    }).dropna(subset=["anchor_ts"])

    con = duckdb.connect()
    try:
        con.register("anchors", anchors)
        q = f"""
        SELECT
          a.anchor_idx,
          MIN(t.timestamp) AS first_other_fill_ts
        FROM anchors a
        JOIN read_parquet('{trades_glob}') t
          ON t.market_id = a.market_id
         AND (CASE WHEN t.maker_asset_id = '0' OR t.maker_asset_id IS NULL
                   THEN t.taker_asset_id ELSE t.maker_asset_id END) = a.outcome_token_id
         AND t.timestamp >  a.anchor_ts + INTERVAL '{int(min_seconds)} second'
         AND t.timestamp <= a.anchor_ts + INTERVAL '{int(max_seconds)} second'
         AND t.maker NOT IN (a.anchor_maker, a.anchor_taker)
         AND t.taker NOT IN (a.anchor_maker, a.anchor_taker)
        GROUP BY a.anchor_idx
        """
        first_fill = con.execute(q).df()
    finally:
        con.close()

    n = len(anchors)
    merged = anchors[["anchor_idx", "anchor_ts"]].merge(first_fill, on="anchor_idx", how="left")
    lag = (pd.to_datetime(merged["first_other_fill_ts"])
           - pd.to_datetime(merged["anchor_ts"])).dt.total_seconds()
    has_fill = lag.notna()
    n_with = int(has_fill.sum())

    def share_within(t: float) -> float:
        return float((lag < t).sum() / n) if n else float("nan")

    if n_with:
        p10, p25, med, p75, p90, p99 = np.nanpercentile(
            lag.dropna(), [10, 25, 50, 75, 90, 99]
        )
    else:
        p10 = p25 = med = p75 = p90 = p99 = float("nan")

    return {
        "p_in": p_in,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "n_anchors": n,
        "n_with_any_fill_within_max": n_with,
        "pct_within_30s":  share_within(30),
        "pct_within_60s":  share_within(60),
        "pct_within_120s": share_within(120),
        "pct_within_300s": share_within(300),
        "pct_within_600s": share_within(600),
        "pct_no_fill_within_max": 1.0 - (n_with / n) if n else float("nan"),
        "lag_p10_s":    float(p10),
        "lag_p25_s":    float(p25),
        "lag_median_s": float(med),
        "lag_p75_s":    float(p75),
        "lag_p90_s":    float(p90),
        "lag_p99_s":    float(p99),
    }


def eval_pair_user_passive(
    p_wide: pd.DataFrame,
    p_in: float,
    p_out: float,
    bankroll: float = 10_000.0,
    size_pct: float = 0.05,
    entry_model: str = "sticky",
    policy: str = "all",
    family: str | None = None,
    min_seconds: int = DEFAULT_MIN_SECONDS,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    fallback_cents: float = DEFAULT_FALLBACK_CENTS,
    assumed_spread_cents: float = DEFAULT_SPREAD_CENTS,
    trades_glob: str = DEFAULT_TRADES_GLOB,
    return_audit: bool = False,
):
    """Calibrated to the user's CLOB-WS execution model.

    `entry_model`:
      'sticky' (default, REALISTIC) — bot posts at the best bid initially,
              chases up to a p_in cap, but does NOT chase down. In a crash,
              our bid at p_in becomes the highest in the book and gets hit
              FIRST by aggressive sellers — fill price = p_in, regardless of
              how far price has fallen. This captures adverse selection:
              sticky bids get filled exactly when the market is moving
              against us. Realistic for low-liquidity venues like Polymarket
              weather markets, where queue rebuild + cancel latency mean
              you can't dodge crash hits.

      'track_down' (UPPER BOUND, unrealistic) — bot continuously cancels and
              re-posts at every new best bid, in both directions. Fill price
              = the actual touch level at fill time (bid_nf_price). This
              gives the bot CHEAP entries during crashes (sometimes
              recovering to p_out → 8x trades). Requires sub-second
              cancel-replace and queue-priority advantage that no realistic
              bot achieves in a low-liquidity weather market. Use this only
              as a sensitivity check.

              ADDITIONAL CAVEAT — interpretation of bid_nf_price: a low
              bid_nf could reflect (a) the market actually crashed to that
              level (narrow spread, both sides at the new low) OR (b) the
              spread simply widened (bid lonely at the low, ask still high).
              Under (b), filling at bid_nf is filling into a one-sided dump
              — the price hasn't really "moved down," only the bid side has.
              The §4 spread diagnostic measures spread on a 5-min average,
              not a snapshot, so it can't fully distinguish these cases.
              This adds further reason to treat track_down as an unreachable
              upper bound, not a deployable target.

    Common execution semantics:
      Entry-fill condition: aggressive seller arrives in (15s, 5min] of cross,
                             AND bid_nf_price ≤ p_in (i.e., we were the best
                             bid level — or would have been if our p_in bid
                             was in the book).
      Exit: when price reaches p_out, bot sells ACTIVE (hits the bid). Receives
            p_out − assumed_spread_cents/100 per share. (Active exit because
            you don't want your sell order to sit unhit while price falls.)
      Hold: if TP doesn't fire, hold to resolution.
      Sizing: size_pct × bankroll per FILLED trade, fixed-fractional on
              INITIAL bankroll (non-compounding). Unfilled crosses → no capital.

    Per-trade PnL ($/share):
      TP:        (p_out − spread) − entry_price
      hold_win:  1.0              − entry_price
      hold_chop: 0.0              − entry_price

    where entry_price = p_in (sticky) or bid_nf_price (track_down).

    Per-trade $ PnL: shares × pps, where shares = size_pct·bankroll / entry_price.

    Returns a summary dict; pass return_audit=True for the per-trade DataFrame
    + daily + equity + DD curves.
    """
    if entry_model not in ("sticky", "track_down"):
        raise ValueError(
            f"entry_model must be 'sticky' or 'track_down', got {entry_model!r}"
        )
    r = eval_pair(
        p_wide, p_in, p_out, policy=policy, family=family,
        min_seconds=min_seconds, max_seconds=max_seconds,
        fallback_cents=fallback_cents,
        assumed_spread_cents=assumed_spread_cents,
        trades_glob=trades_glob,
        return_audit=True,
    )
    if r is None:
        return None
    _summary, audit = r

    spread_dollars = assumed_spread_cents / 100.0
    p_in_v  = float(p_in)
    p_out_v = float(p_out)

    bid_price  = audit["entry_next_opp_dir_price"].astype(float).values
    bid_source = audit["entry_next_opp_dir_source"].values
    tp_fires   = audit["tp_fires"].values
    resolution = audit["resolution"].astype(float).values
    entry_ts   = pd.to_datetime(audit["entry_anchor_ts"]).values

    has_real_bid = (bid_source == "next_fill")
    filled = has_real_bid & (bid_price <= p_in_v + 1e-9)
    # Entry price model:
    # - 'sticky' (realistic): our bid sits at p_in; on a crash our high bid is
    #   the first one hit, so we fill at p_in regardless of how low the
    #   historical bid_nf print was. This captures adverse selection: sticky
    #   bids get picked off during exactly the moves that go against us.
    # - 'track_down': bot continuously chases best bid down; fills at the
    #   actual touch level. Unrealistic for queue/latency reasons.
    if entry_model == "sticky":
        entry_price = np.where(filled, p_in_v, np.nan)
    else:  # track_down
        entry_price = np.where(filled, bid_price, np.nan)

    tp  = filled & tp_fires
    win = filled & ~tp_fires & (resolution == 1)
    chop = filled & ~tp_fires & (resolution == 0)

    pps = np.zeros(len(audit))
    pps[tp]   = (p_out_v - spread_dollars) - entry_price[tp]
    pps[win]  = 1.0                        - entry_price[win]
    pps[chop] = 0.0                        - entry_price[chop]

    # $ PnL per trade with fixed-fractional on INITIAL bankroll
    notional = size_pct * bankroll
    dollar_pnl = np.zeros(len(audit))
    dollar_pnl[filled] = notional * pps[filled] / entry_price[filled]

    # Daily aggregation (only filled trades contribute).
    filled_idx = np.where(filled)[0]
    if len(filled_idx) == 0:
        # Empty cell — return zeros
        return {
            "p_in": p_in, "p_out": p_out,
            "bankroll_initial": bankroll, "size_pct": size_pct,
            "assumed_spread_cents": assumed_spread_cents,
            "n_total": len(audit), "n_filled": 0,
            "fill_rate": 0.0, "n_tp": 0,
            "p_tp_of_filled":      float("nan"),
            "p_hold_win_of_filled": float("nan"),
            "p_hold_chop_of_filled": float("nan"),
            "mean_entry_price":    float("nan"),
            "mean_entry_cents_above_p_in": float("nan"),
            "edge_per_filled":     float("nan"),
            "roi_per_filled_pct":  float("nan"),
            "total_dollar_pnl":    0.0,
            "final_bankroll":      bankroll,
            "pnl_pct_of_initial_bankroll": 0.0,
            "sharpe_ann":          float("nan"),
            "max_dd_pct":          0.0,
            "annualization_days":  0,
        }

    trades_df = pd.DataFrame({
        "entry_ts":     entry_ts[filled_idx],
        "entry_price":  entry_price[filled_idx],
        "pnl_per_share":pps[filled_idx],
        "dollar_pnl":   dollar_pnl[filled_idx],
        "bucket":       np.where(tp[filled_idx],  "tp",
                          np.where(win[filled_idx], "hold_win", "hold_chop")),
    })
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_ts"]).dt.floor("D")

    # Universe span for honest annualisation (same approach as ftc_tp_sizing.backtest).
    uni_end_ts = p_wide["end_ts"].dropna()
    universe_start = pd.to_datetime(uni_end_ts).dt.floor("D").min()
    universe_end   = pd.to_datetime(uni_end_ts).dt.floor("D").max()
    full_idx = pd.date_range(universe_start, universe_end, freq="D")

    daily_pnl = trades_df.groupby("entry_date")["dollar_pnl"].sum()
    daily_pnl = daily_pnl.reindex(full_idx, fill_value=0.0)
    daily_ret = daily_pnl / bankroll  # return as fraction of initial bankroll

    equity = bankroll + daily_pnl.cumsum()
    dd     = equity / equity.cummax() - 1.0
    mu, sd = daily_ret.mean(), daily_ret.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(365) if sd > 0 else float("nan")
    days   = (full_idx[-1] - full_idx[0]).days or 1

    out = {
        "p_in": p_in, "p_out": p_out,
        "bankroll_initial":     bankroll,
        "size_pct":             size_pct,
        "assumed_spread_cents": assumed_spread_cents,
        "n_total":  int(len(audit)),
        "n_filled": int(filled.sum()),
        "fill_rate": float(filled.mean()),
        "n_tp":      int(tp.sum()),
        "p_tp_of_filled":       float(tp.sum()  / filled.sum()),
        "p_hold_win_of_filled": float(win.sum() / filled.sum()),
        "p_hold_chop_of_filled":float(chop.sum()/ filled.sum()),
        "mean_entry_price":     float(entry_price[filled].mean()),
        "mean_entry_cents_above_p_in":
                                float((entry_price[filled] - p_in_v).mean() * 100),
        "edge_per_filled":      float(pps[filled].mean()),
        "roi_per_filled_pct":   100 * float(pps[filled].mean()) / p_in_v,
        "total_dollar_pnl":     float(dollar_pnl.sum()),
        "final_bankroll":       float(bankroll + dollar_pnl.sum()),
        "pnl_pct_of_initial_bankroll":
                                100 * float(dollar_pnl.sum()) / bankroll,
        "sharpe_ann":           float(sharpe),
        "max_dd_pct":           float(dd.min() * 100),
        "annualization_days":   int(days),
    }
    if return_audit:
        return out, trades_df, daily_pnl, equity, dd
    return out


def grid_user_passive(
    p_wide: pd.DataFrame,
    bankroll: float = 10_000.0,
    size_pct: float = 0.05,
    entry_model: str = "sticky",
    barriers: list[float] = BARRIERS_EE,
    policy: str = "all",
    min_seconds: int = DEFAULT_MIN_SECONDS,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    fallback_cents: float = DEFAULT_FALLBACK_CENTS,
    assumed_spread_cents: float = DEFAULT_SPREAD_CENTS,
    trades_glob: str = DEFAULT_TRADES_GLOB,
) -> pd.DataFrame:
    """Sweep all (p_in, p_out) pairs under the user's calibrated execution model.

    `entry_model='sticky'` (default, realistic) vs `'track_down'` (upper-bound).
    See eval_pair_user_passive docstring for the semantics.

    ~7 min for the default 21-pair grid (one trades-parquet scan per cell).
    """
    rows = []
    for p_in in barriers:
        for p_out in barriers:
            if not (p_in < p_out):
                continue
            r = eval_pair_user_passive(
                p_wide, p_in, p_out,
                bankroll=bankroll, size_pct=size_pct,
                entry_model=entry_model,
                policy=policy,
                min_seconds=min_seconds, max_seconds=max_seconds,
                fallback_cents=fallback_cents,
                assumed_spread_cents=assumed_spread_cents,
                trades_glob=trades_glob,
            )
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


def chase_bid_cutoff_sweep(
    audit: pd.DataFrame,
    cutoffs_cents: list[float] | None = None,
    exit_passive: str = "optimistic",
) -> pd.DataFrame:
    """Sweep "chase-best-bid up to p_in + N cents" cutoffs under WS-passive entry.

    Execution model: the bot follows the best bid up — if someone overbids us,
    we raise our quote to match. The effective fill price is whatever the
    observed `bid_nf_price` is (i.e. the bid that the next aggressive seller hit).
    The cutoff p_in + N cents bounds how far we'll chase before stepping out.

    For each cutoff:
      - fill condition: real `entry_next_opp_dir_price` exists AND <= p_in + N/100
      - entry_price (when filled): max(p_in, bid_nf_price)
        (We never fill BELOW p_in — if best bid sat below p_in we'd just be the
         best bid at p_in waiting. But the print itself executes at the best
         bid level at that instant, which is what bid_nf_price observes.
         Take the max as a conservative read: we paid at least p_in.)
      - exit on TP at exactly p_out under `exit_passive` ('optimistic' = TP fires;
        'strict' = also need real aggressive-buy at exit)
      - hold to resolution otherwise

    Returns one row per cutoff with:
      cutoff_cents, p_cutoff, n_filled, fill_rate, p_tp_of_filled,
      edge_per_filled (mean PnL/share over fills), mean_entry_price,
      total_pnl_per_entered (sum PnL across all crosses; unfilled contribute 0),
      roi_per_filled_pct
    """
    if cutoffs_cents is None:
        cutoffs_cents = [0, 1, 2, 3, 5, 8, 12, 20, 30]

    p_in  = float(audit["p_in"].iloc[0])
    p_out = float(audit["p_out"].iloc[0])
    n_total = len(audit)

    bid_price_raw = audit["entry_next_opp_dir_price"].astype(float).values
    bid_source    = audit["entry_next_opp_dir_source"].values
    tp_fires      = audit["tp_fires"].values
    resolution    = audit["resolution"].astype(float).values
    has_real_bid  = (bid_source == "next_fill")

    if exit_passive == "optimistic":
        exit_fills_when_filled = tp_fires
    else:
        exit_fills_when_filled = tp_fires & (audit["exit_next_opp_dir_source"].values == "next_fill")

    rows = []
    for cents in cutoffs_cents:
        p_cutoff = p_in + cents / 100.0
        # Fill condition: a real bid_nf exists and is at or below the cutoff.
        filled = has_real_bid & (bid_price_raw <= p_cutoff + 1e-9)
        # Entry price under chase model: never below p_in (we wouldn't lower our bid).
        entry_price = np.where(filled, np.maximum(bid_price_raw, p_in), np.nan)

        exit_filled = filled & exit_fills_when_filled
        pnl = np.zeros(n_total)
        pnl[exit_filled] = p_out - entry_price[exit_filled]
        held = filled & ~exit_fills_when_filled
        pnl[held] = resolution[held] - entry_price[held]

        n_filled = int(filled.sum())
        pnl_filled = pnl[filled]
        n_tp = int(exit_filled.sum())
        rows.append({
            "cutoff_cents": cents,
            "p_cutoff":     float(p_cutoff),
            "n_total":      n_total,
            "n_filled":     n_filled,
            "fill_rate":    n_filled / n_total if n_total else float("nan"),
            "n_tp":         n_tp,
            "p_tp_of_filled":       (n_tp / n_filled) if n_filled else float("nan"),
            "mean_entry_price":     float(entry_price[filled].mean()) if n_filled else float("nan"),
            "mean_entry_cents_above_p_in": (
                float((entry_price[filled] - p_in).mean() * 100) if n_filled else float("nan")
            ),
            "edge_per_filled":      float(pnl_filled.mean()) if n_filled else float("nan"),
            "edge_per_entered":     float(pnl.mean()),
            "roi_per_filled_pct":   (100 * float(pnl_filled.mean()) / p_in) if n_filled else float("nan"),
            "total_pnl_per_entered": float(pnl.sum()),
        })
    return pd.DataFrame(rows)


def passive_pnl_from_audit(
    audit: pd.DataFrame,
    exit_passive: str = "optimistic",
    _suppress_queue_warning: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """PnL under WS-passive execution — OPTIMISTIC QUEUE MODEL.

    ⚠️  KNOWN OPTIMISM: this function assumes our posted bid at p_in fills
    whenever ANY aggressive seller arrives in the 5-min window. In reality
    that's only true when we have queue priority at the touch — when the
    observed `bid_nf_price` is at or below p_in. When `bid_nf_price > p_in`,
    the print hit someone else's higher bid, not ours, and we would NOT have
    filled. Treat this function's output as an UPPER BOUND on passive-fill
    PnL, achievable only with perfect queue priority.

    For the realistic queue model (which correctly excludes prints above our
    level), use `chase_bid_cutoff_sweep(audit, cutoffs_cents=[0, ...])`. At
    canonical (0.60, 0.90), this function reports fill_rate ≈ 26% and ROI
    ≈ +0.87%, while the realistic chase-0¢ model gives 12% and −14.7%.

    Entry model: bot posts a passive bid at p_in. Filled rows execute at
    EXACTLY p_in (per the optimistic-queue assumption above).

    Exit model on TP:
      'optimistic' (default): if tp_fires (price reached p_out), assume our
        passive ask at p_out was lifted.
      'strict': require ALSO a real exit_next_opp_dir fill. Filters out cases
        where the leader's offer at p_out absorbed all aggressive buying.

    Non-TP filled rows hold to resolution → PnL = resolution − p_in.
    Unfilled rows: no trade, PnL = 0 (drop out of the trade count).

    Returns (audit_with_new_columns, summary_dict). New columns:
        entry_filled_passive, exit_filled_passive, pnl_passive, outcome_passive.

    Set _suppress_queue_warning=True to silence the runtime warning (e.g.
    when you're explicitly comparing optimistic vs realistic).
    """
    import warnings as _warnings
    if not _suppress_queue_warning:
        _warnings.warn(
            "passive_pnl_from_audit uses an OPTIMISTIC queue model — fill rate "
            "and ROI are upper bounds, achievable only with perfect queue "
            "priority. Use chase_bid_cutoff_sweep(audit, cutoffs_cents=[0]) "
            "for the realistic queue model. See docstring.",
            UserWarning, stacklevel=2,
        )
    if exit_passive not in ("optimistic", "strict"):
        raise ValueError(f"exit_passive must be 'optimistic' or 'strict', got {exit_passive!r}")
    a = audit.copy()
    p_in  = float(a["p_in"].iloc[0])
    p_out = float(a["p_out"].iloc[0])

    entry_filled = (a["entry_next_opp_dir_source"].values == "next_fill")
    tp_fires     = a["tp_fires"].values
    if exit_passive == "optimistic":
        exit_passive_filled = entry_filled & tp_fires
    else:
        exit_passive_filled = (entry_filled & tp_fires
                                & (a["exit_next_opp_dir_source"].values == "next_fill"))
    resolution = a["resolution"].astype(float).values

    pnl = np.zeros(len(a))
    pnl[exit_passive_filled] = p_out - p_in
    held = entry_filled & ~exit_passive_filled
    pnl[held] = resolution[held] - p_in

    outcome = np.array(["no_entry"] * len(a), dtype=object)
    outcome[exit_passive_filled]               = "tp"
    outcome[held & (resolution == 1)]          = "hold_win"
    outcome[held & (resolution == 0)]          = "hold_chop"

    a["entry_filled_passive"] = entry_filled
    a["exit_filled_passive"]  = exit_passive_filled
    a["pnl_passive"]          = pnl
    a["outcome_passive"]      = outcome

    n = len(a)
    n_filled = int(entry_filled.sum())
    pnl_filled = pnl[entry_filled]

    summary = {
        "p_in": p_in, "p_out": p_out,
        "exit_passive_assumption": exit_passive,
        "n_total":        n,
        "n_entry_filled": n_filled,
        "fill_rate":      float(n_filled / n) if n else float("nan"),
        "n_tp":           int((outcome == "tp").sum()),
        "n_hold_win":     int((outcome == "hold_win").sum()),
        "n_hold_chop":    int((outcome == "hold_chop").sum()),
        "p_tp_of_filled":        float((outcome == "tp").sum() / n_filled) if n_filled else float("nan"),
        "p_hold_win_of_filled":  float((outcome == "hold_win").sum() / n_filled) if n_filled else float("nan"),
        "p_hold_chop_of_filled": float((outcome == "hold_chop").sum() / n_filled) if n_filled else float("nan"),
        "edge_per_entered": float(pnl.mean()),
        "edge_per_filled":  float(pnl_filled.mean()) if n_filled else float("nan"),
        "roi_per_filled_pct":  (100 * float(pnl_filled.mean()) / p_in) if n_filled else float("nan"),
        "total_pnl_per_entered": float(pnl.sum()),
    }
    return a, summary


def eval_pair_passive(
    p_wide: pd.DataFrame,
    p_in: float,
    p_out: float,
    policy: str = "all",
    family: str | None = None,
    min_seconds: int = DEFAULT_MIN_SECONDS,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    fallback_cents: float = DEFAULT_FALLBACK_CENTS,
    assumed_spread_cents: float = DEFAULT_SPREAD_CENTS,
    trades_glob: str = DEFAULT_TRADES_GLOB,
    exit_passive: str = "optimistic",
    return_audit: bool = False,
):
    """Convenience: eval_pair + passive_pnl_from_audit in one call.

    ⚠️  Inherits the OPTIMISTIC QUEUE assumption from `passive_pnl_from_audit`
    — see that function's docstring. Use `chase_bid_cutoff_sweep` for the
    realistic-queue answer.

    Returns the passive summary dict (and the augmented audit if return_audit=True)."""
    r = eval_pair(
        p_wide, p_in, p_out, policy=policy, family=family,
        min_seconds=min_seconds, max_seconds=max_seconds,
        fallback_cents=fallback_cents,
        assumed_spread_cents=assumed_spread_cents,
        trades_glob=trades_glob,
        return_audit=True,
    )
    if r is None:
        return None
    _summary, audit = r
    aug_audit, passive_summary = passive_pnl_from_audit(audit, exit_passive=exit_passive)
    if return_audit:
        return passive_summary, aug_audit
    return passive_summary


def grid_passive(
    p_wide: pd.DataFrame,
    barriers: list[float] = BARRIERS_EE,
    policy: str = "all",
    exit_passive: str = "optimistic",
    min_seconds: int = DEFAULT_MIN_SECONDS,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    fallback_cents: float = DEFAULT_FALLBACK_CENTS,
    assumed_spread_cents: float = DEFAULT_SPREAD_CENTS,
    trades_glob: str = DEFAULT_TRADES_GLOB,
) -> pd.DataFrame:
    """Sweep (p_in, p_out) under the WS-passive PnL model. ~7 min for 21 pairs.

    ⚠️  Uses the OPTIMISTIC QUEUE model from `passive_pnl_from_audit` — fill
    rate and ROI per cell are upper bounds. For realistic queue ROI, build
    an audit at the cell's (p_in, p_out) and run `chase_bid_cutoff_sweep`."""
    rows = []
    for p_in in barriers:
        for p_out in barriers:
            if not (p_in < p_out):
                continue
            r = eval_pair_passive(
                p_wide, p_in, p_out, policy=policy,
                min_seconds=min_seconds, max_seconds=max_seconds,
                fallback_cents=fallback_cents,
                assumed_spread_cents=assumed_spread_cents,
                trades_glob=trades_glob,
                exit_passive=exit_passive,
            )
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


def subset_pnl_summary(
    scen: pd.DataFrame, mask: pd.Series, label: str,
    p_in: float | None = None,
) -> dict:
    """Bucket distribution + per-scenario edges on a subset.

    `scen` is the output of compute_fill_scenarios. `mask` is a boolean Series
    aligned to scen.index. Returns a dict suitable for tabulating across subsets.
    """
    sub = scen.loc[mask]
    n = int(len(sub))
    if n == 0:
        return {"label": label, "n": 0, "p_tp": float("nan"),
                "p_hold_win": float("nan"), "p_hold_chop": float("nan"),
                "edge_next_same_dir": float("nan"),
                "edge_midpoint": float("nan"),
                "edge_next_opp_dir": float("nan"),
                "mean_entry_slip_cents": float("nan"),
                "roi_next_same_dir_pct": float("nan"),
                "roi_midpoint_pct": float("nan"),
                "roi_next_opp_dir_pct": float("nan")}
    if p_in is None:
        p_in = float(sub["p_in"].iloc[0])
    edges = {s: float(sub[f"pnl_{s}"].mean()) for s in
             ("next_same_dir", "midpoint", "next_opp_dir")}
    mean_entry_slip = float(((sub["entry_next_same_dir_price"].astype(float) - p_in) * 100).mean())
    return {
        "label": label, "n": n,
        "p_tp":        float(sub["bucket"].eq("tp").mean()),
        "p_hold_win":  float(sub["bucket"].eq("hold_win").mean()),
        "p_hold_chop": float(sub["bucket"].eq("hold_chop").mean()),
        "edge_next_same_dir": edges["next_same_dir"],
        "edge_midpoint":      edges["midpoint"],
        "edge_next_opp_dir":  edges["next_opp_dir"],
        "roi_next_same_dir_pct": 100 * edges["next_same_dir"] / p_in,
        "roi_midpoint_pct":      100 * edges["midpoint"] / p_in,
        "roi_next_opp_dir_pct":  100 * edges["next_opp_dir"] / p_in,
        "mean_entry_slip_cents": mean_entry_slip,
    }


def _city_from_slug_family(sf: str) -> str:
    """Collapse range-bucket suffixes: 'highest-temperature-in-miami-84forhigher'
    → 'highest-temperature-in-miami'. Mirrors ftc_tp_sizing.city_from_slug_family
    so we don't have to import across packages."""
    import re as _re
    return _re.sub(r"-(?:neg-?)?[0-9].*$", "", sf or "")


def subset_selection_bias(
    scen: pd.DataFrame,
    p_wide: pd.DataFrame,
    mask: pd.Series,
    baseline_mask: pd.Series | None = None,
    top_k: int = 5,
) -> dict:
    """Compare a subset's composition vs the baseline.

    Stats: top cities by share (baseline vs subset), hour-of-day distribution,
    day-of-week distribution, distance-to-resolution at cross time (hours).
    """
    if baseline_mask is None:
        baseline_mask = pd.Series(True, index=scen.index)

    # Attach end_ts via p_wide on (market_id, outcome_token_id).
    end_ts_map = (p_wide[["market_id", "outcome_token_id", "end_ts"]]
                  .drop_duplicates(["market_id", "outcome_token_id"]))
    s = scen.merge(end_ts_map, on=["market_id", "outcome_token_id"], how="left")
    s["city"] = s["slug_family"].map(_city_from_slug_family)
    s["entry_anchor_ts"] = pd.to_datetime(s["entry_anchor_ts"])
    s["end_ts"] = pd.to_datetime(s["end_ts"])
    s["hours_to_resolution"] = (s["end_ts"] - s["entry_anchor_ts"]).dt.total_seconds() / 3600.0
    s["hour_utc"]   = s["entry_anchor_ts"].dt.hour
    s["dow"]        = s["entry_anchor_ts"].dt.day_name()

    base = s.loc[baseline_mask.values]
    sub  = s.loc[mask.values]

    def _city_table(df: pd.DataFrame) -> pd.DataFrame:
        return (df.groupby("city").size().sort_values(ascending=False)
                  .head(top_k).rename("n").to_frame()
                  .assign(share=lambda x: x["n"] / len(df)))

    base_cities = _city_table(base)
    sub_cities  = _city_table(sub)

    return {
        "n_baseline": int(len(base)),
        "n_subset":   int(len(sub)),
        "subset_share_of_baseline": float(len(sub) / max(len(base), 1)),
        "top_cities_baseline": base_cities.reset_index().to_dict("records"),
        "top_cities_subset":   sub_cities.reset_index().to_dict("records"),
        "hours_to_resolution_baseline": {
            "p10": float(np.nanpercentile(base["hours_to_resolution"], 10)) if len(base) else float("nan"),
            "median": float(np.nanmedian(base["hours_to_resolution"])) if len(base) else float("nan"),
            "p90": float(np.nanpercentile(base["hours_to_resolution"], 90)) if len(base) else float("nan"),
        },
        "hours_to_resolution_subset": {
            "p10": float(np.nanpercentile(sub["hours_to_resolution"], 10)) if len(sub) else float("nan"),
            "median": float(np.nanmedian(sub["hours_to_resolution"])) if len(sub) else float("nan"),
            "p90": float(np.nanpercentile(sub["hours_to_resolution"], 90)) if len(sub) else float("nan"),
        },
        "hour_utc_baseline_top3": base["hour_utc"].value_counts(normalize=True).head(3).to_dict(),
        "hour_utc_subset_top3":   sub["hour_utc"].value_counts(normalize=True).head(3).to_dict(),
        "dow_baseline_top3":      base["dow"].value_counts(normalize=True).head(3).to_dict(),
        "dow_subset_top3":        sub["dow"].value_counts(normalize=True).head(3).to_dict(),
    }


def grid_subset(
    p_wide: pd.DataFrame,
    mask_builder,
    barriers: list[float] = BARRIERS_EE,
    policy: str = "all",
    min_seconds: int = DEFAULT_MIN_SECONDS,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    fallback_cents: float = DEFAULT_FALLBACK_CENTS,
    assumed_spread_cents: float = DEFAULT_SPREAD_CENTS,
    trades_glob: str = DEFAULT_TRADES_GLOB,
) -> pd.DataFrame:
    """Sweep (p_in, p_out) on a subset defined by mask_builder(scen)→bool Series.

    For each pair: runs eval_pair, builds compute_fill_scenarios, filters by
    the subset mask, returns per-cell n + bucket dist + 3 scenario edges.

    Caller-supplied `mask_builder` lets you reuse the same definition across
    cells (e.g. `lambda a: a['entry_next_same_dir_source'] == 'next_fill'`).
    """
    rows = []
    for p_in in barriers:
        for p_out in barriers:
            if not (p_in < p_out):
                continue
            r = eval_pair(
                p_wide, p_in, p_out, policy=policy,
                min_seconds=min_seconds, max_seconds=max_seconds,
                fallback_cents=fallback_cents,
                assumed_spread_cents=assumed_spread_cents,
                trades_glob=trades_glob,
                return_audit=True,
            )
            if r is None:
                continue
            _summary, audit = r
            scen = compute_fill_scenarios(audit)
            mask = mask_builder(scen)
            sub  = scen.loc[mask]
            n = int(len(sub))
            if n == 0:
                rows.append({"p_in": p_in, "p_out": p_out, "n": 0})
                continue
            rows.append({
                "p_in": p_in, "p_out": p_out, "n": n,
                "n_baseline": int(len(scen)),
                "subset_share": float(n / len(scen)),
                "p_tp":        float(sub["bucket"].eq("tp").mean()),
                "p_hold_win":  float(sub["bucket"].eq("hold_win").mean()),
                "p_hold_chop": float(sub["bucket"].eq("hold_chop").mean()),
                "edge_next_same_dir": float(sub["pnl_next_same_dir"].mean()),
                "edge_midpoint":      float(sub["pnl_midpoint"].mean()),
                "edge_next_opp_dir":  float(sub["pnl_next_opp_dir"].mean()),
                "roi_next_same_dir_pct": 100 * float(sub["pnl_next_same_dir"].mean()) / p_in,
                "roi_midpoint_pct":      100 * float(sub["pnl_midpoint"].mean()) / p_in,
                "roi_next_opp_dir_pct":  100 * float(sub["pnl_next_opp_dir"].mean()) / p_in,
            })
    return pd.DataFrame(rows)


def grid_search_entry_exit(
    p_wide: pd.DataFrame,
    barriers: list[float] = BARRIERS_EE,
    policy: str = "all",
    min_seconds: int = DEFAULT_MIN_SECONDS,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    fallback_cents: float = DEFAULT_FALLBACK_CENTS,
    assumed_spread_cents: float = DEFAULT_SPREAD_CENTS,
    trades_glob: str = DEFAULT_TRADES_GLOB,
) -> pd.DataFrame:
    """Sweep eval_pair over all (p_in, p_out) pairs in `barriers` with p_in < p_out.

    Per-cell call to eval_pair (next-fill slippage). Heavy: each cell runs
    one ASOF-style scan over the trades parquet. Expect minutes for the
    21-pair sweep.
    """
    rows = []
    for p_in in barriers:
        for p_out in barriers:
            r = eval_pair(
                p_wide, p_in, p_out, policy=policy,
                min_seconds=min_seconds, max_seconds=max_seconds,
                fallback_cents=fallback_cents,
                assumed_spread_cents=assumed_spread_cents,
                trades_glob=trades_glob,
            )
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)
