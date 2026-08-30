"""The time series: one token's tape, or an aligned pair / event. Per token, never whole-table.

Columns (post-H0 units):
  l1     : ts(UTC), timestamp_ms, received_ns, asset_id, best_bid, best_ask, mid, spread_c
           best_bid/best_ask/mid are DOLLARS (0-1); spread_c is CENTS.
  trades : ts(UTC), timestamp_ms, received_ns, asset_id, price, size, side, fee_rate_bps, transaction_hash
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import _internal as _i


def load_l1(ref, start=None, end=None) -> pd.DataFrame:
    """One token's L1 tape (deduped to touch-moving rows — NOT every message). ref is an
    asset_id, path or market_slug. start/end accept UTC datetimes or epoch-ms."""
    aid = _i.resolve_ref(ref)
    return _i.read_token_tape("l1", aid, _i.token_row(aid)["universe"], start, end)


def load_trades(ref, start=None, end=None) -> pd.DataFrame:
    """One token's trade prints. ref/start/end as in load_l1."""
    aid = _i.resolve_ref(ref)
    return _i.read_token_tape("trades", aid, _i.token_row(aid)["universe"], start, end)


def load_pair(condition_id, start=None, end=None) -> pd.DataFrame:
    """Both sides of a market, mids time-aligned on a common index (the YES/NO mirror in one
    call). Columns are the two outcome_index values (0, 1); .attrs['labels'] maps index->label.
    A well-behaved binary market has col0 + col1 ~= 1 at every row."""
    t = _i.tokens()
    m = t[t["condition_id"] == str(condition_id)].sort_values("outcome_index")
    if len(m) != 2:
        raise ValueError(f"condition {condition_id} has {len(m)} tokens (expected 2)")
    frames = {}
    labels = {}
    for _, row in m.iterrows():
        d = _i.read_token_tape("l1", row["asset_id"], row["universe"], start, end)
        frames[int(row["outcome_index"])] = d
        labels[int(row["outcome_index"])] = row["outcome_label"]
    wide = _i.align_mids(frames, value_col="mid")
    wide.attrs["labels"] = labels
    wide.attrs["condition_id"] = str(condition_id)
    return wide


def markout(ref, horizons=(10, 30, 60), start=None, end=None) -> pd.DataFrame:
    """Per-trade adverse-selection markout from the LIQUIDITY PROVIDER (maker) perspective.

    `trades.side` is the TAKER (aggressor) side — verified against the book: BUY prints sit at
    the ask, SELL at the bid (both universes, ~200k trades each). So a taker BUY means the maker
    SOLD (is short); a taker SELL means the maker BOUGHT (is long).

    For each trade, markout at horizon Δ (seconds) = maker_sign * (mid[t+Δ] - trade_price), where
    maker_sign = -1 for a BUY, +1 for a SELL. **NEGATIVE markout = the resting quote was adversely
    selected** (price moved against the maker after the fill). Returns the trades with `ts`,
    `price`, `size`, `side`, and `mid_<h>` / `markout_<h>` (dollars) columns per horizon."""
    aid = _i.resolve_ref(ref)
    r = _i.token_row(aid)
    l1 = _i.read_token_tape("l1", aid, r["universe"])[["timestamp_ms", "mid"]].dropna().sort_values("timestamp_ms")
    tr = _i.read_token_tape("trades", aid, r["universe"], start, end)
    if l1.empty or tr.empty:
        return tr
    tr = tr.sort_values("timestamp_ms").reset_index(drop=True)
    maker_sign = np.where(tr["side"].eq("BUY"), -1.0, 1.0)
    for h in horizons:
        tgt = pd.DataFrame({"target": tr["timestamp_ms"].to_numpy() + int(h) * 1000})
        j = pd.merge_asof(tgt, l1.rename(columns={"timestamp_ms": "lt"}),
                          left_on="target", right_on="lt", direction="backward")
        tr[f"mid_{h}"] = j["mid"].to_numpy()
        tr[f"markout_{h}"] = maker_sign * (tr[f"mid_{h}"] - tr["price"])
    return tr


def load_event(event_slug, start=None, end=None) -> pd.DataFrame:
    """Every token in an event, mids time-aligned on a common index (one column per asset_id).
    For a NegRisk politics event, summing the YES-side columns should sit near 1. Use catalog()
    to map asset_id -> outcome/market. .attrs['tokens'] carries that mapping."""
    t = _i.tokens()
    m = t[t["event_slug"] == str(event_slug)]
    if m.empty:
        raise KeyError(event_slug)
    frames = {}
    meta = {}
    for _, row in m.iterrows():
        d = _i.read_token_tape("l1", row["asset_id"], row["universe"], start, end)
        frames[row["asset_id"]] = d
        meta[row["asset_id"]] = {"outcome": row["outcome"], "outcome_label": row["outcome_label"],
                                 "market_slug": row["market_slug"], "outcome_index": int(row["outcome_index"])}
    wide = _i.align_mids(frames, value_col="mid")
    wide.attrs["tokens"] = meta
    return wide
