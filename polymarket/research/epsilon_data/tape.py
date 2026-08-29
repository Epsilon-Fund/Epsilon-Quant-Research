"""The time series: one token's tape, or an aligned pair / event. Per token, never whole-table.

Columns (post-H0 units):
  l1     : ts(UTC), timestamp_ms, received_ns, asset_id, best_bid, best_ask, mid, spread_c
           best_bid/best_ask/mid are DOLLARS (0-1); spread_c is CENTS.
  trades : ts(UTC), timestamp_ms, received_ns, asset_id, price, size, side, fee_rate_bps, transaction_hash
"""
from __future__ import annotations
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
