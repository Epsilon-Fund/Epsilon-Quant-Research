"""Spot-check the next-fill slippage model on 5 random crossings.

Loads the inst parquet, picks 5 random (market, token) pairs that crossed
p=0.60, runs `compute_next_fill_price` for each (both BUY-maker and SELL-maker
direction), and prints the anchor + next-fill side by side so you can eyeball
plausibility.

Run:
    python scripts/test_next_fill_slippage.py [N]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_PKG = Path(__file__).resolve().parents[1]
if str(REPO_PKG) not in sys.path:
    sys.path.insert(0, str(REPO_PKG))

from data_infra import weather_analysis as wa  # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    inst = pd.read_parquet(wa.DEFAULT_DATA_DIR / "weather_tail_per_instance.parquet")
    crossed = inst[(inst["barrier_price"] == 0.60) & inst["crossed"]].copy()
    print(f"crossings at p=0.60: {len(crossed):,}")
    if len(crossed) < n:
        n = len(crossed)
    sample = crossed.sample(n, random_state=42).reset_index(drop=True)

    print(f"\nrunning batched lookup over {n} anchors (min={wa.DEFAULT_MIN_SECONDS}s, "
          f"max={wa.DEFAULT_MAX_SECONDS}s)…")
    anchors = pd.DataFrame({
        "anchor_idx":       range(n),
        "market_id":        sample["market_id"].astype(str).values,
        "outcome_token_id": sample["outcome_token_id"].astype(str).values,
        "anchor_ts":        pd.to_datetime(sample["first_cross_ts"]).values,
        "anchor_maker":     sample["first_cross_maker"].astype(str).values,
        "anchor_taker":     sample["first_cross_taker"].astype(str).values,
    })
    nf = wa.lookup_next_fills_batch(anchors)

    print("\n" + "=" * 100)
    for i in range(n):
        row    = sample.iloc[i]
        nf_row = nf[nf["anchor_idx"] == i].iloc[0]

        print(f"\n--- ANCHOR #{i+1} ---")
        print(f"  slug             : {row['slug']}")
        print(f"  market_id        : {row['market_id']}")
        print(f"  outcome_token_id : {row['outcome_token_id'][:24]}…")
        print(f"  first_cross_ts   : {row['first_cross_ts']}  (price >= 0.60)")
        print(f"  anchor_maker     : {row['first_cross_maker']}")
        print(f"  anchor_taker     : {row['first_cross_taker']}")
        print(f"  anchor_maker_side: {row['first_cross_maker_side']}")
        print(f"  resolution       : {row['resolution']}")

        bid_ts, bid_px = nf_row["bid_nf_ts"], nf_row["bid_nf_price"]
        ask_ts, ask_px = nf_row["ask_nf_ts"], nf_row["ask_nf_price"]
        print(f"  → BID next-fill (BUY-maker): "
              + (f"{bid_ts}  px={bid_px:.4f}  Δt={(bid_ts - row['first_cross_ts']).total_seconds():.0f}s"
                 if pd.notna(bid_ts) else "NONE in window — fallback fires"))
        print(f"  → ASK next-fill (SELL-maker): "
              + (f"{ask_ts}  px={ask_px:.4f}  Δt={(ask_ts - row['first_cross_ts']).total_seconds():.0f}s"
                 if pd.notna(ask_ts) else "NONE in window — fallback fires"))

        # also exercise the per-row API
        ask_px2, ask_src = wa.compute_next_fill_price(
            row["market_id"], row["outcome_token_id"], row["first_cross_ts"],
            row["first_cross_maker"], row["first_cross_taker"],
            direction="SELL",  # SELL-maker = ASK side
            fallback_price=0.60 + wa.DEFAULT_FALLBACK_CENTS / 100.0,
        )
        bid_px2, bid_src = wa.compute_next_fill_price(
            row["market_id"], row["outcome_token_id"], row["first_cross_ts"],
            row["first_cross_maker"], row["first_cross_taker"],
            direction="BUY",   # BUY-maker = BID side
            fallback_price=max(0.0, 0.60 + wa.DEFAULT_FALLBACK_CENTS / 100.0 - wa.DEFAULT_SPREAD_CENTS / 100.0),
        )
        print(f"  per-row API: ASK={ask_px2:.4f} ({ask_src})   BID={bid_px2:.4f} ({bid_src})")

    print("\n" + "=" * 100)
    has_bid = nf["bid_nf_price"].notna().sum()
    has_ask = nf["ask_nf_price"].notna().sum()
    print(f"summary: {has_bid}/{n} have a BID next-fill   {has_ask}/{n} have an ASK next-fill")
    return 0


if __name__ == "__main__":
    sys.exit(main())
