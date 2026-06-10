"""Analyze MM NegRisk basket-consistency poller output.

Input: a run dir produced by scripts/mm_negrisk_consistency_poll.py
(universe_*.json + cycles.jsonl + cycle_meta.jsonl).

Definitions (per event per cycle):
  ask_sum  = sum of best asks across ALL legs (None unless every leg had a live ask)
  bid_sum  = sum of best bids across legs with a live bid (missing legs add 0,
             so bid_sum is a conservative LOWER bound on sell-all proceeds)
  buy violation  : ask_sum is complete and ask_sum < 1 - fee_buffer
                   (buy every YES leg, hold to resolution, redeem $1)
  sell violation : bid_sum > 1 + fee_buffer
                   (mint a full YES set for net $1 via split+convert, sell every leg at bid)

Episodes: consecutive cycles in violation for the same (event, direction).
Persistence is measured at poll cadence (~cycle_seconds); durations are
interval-censored: an episode seen in k consecutive cycles lasted at least
(k-1)*cadence and its true start/end are unobserved at finer resolution.

Outputs:
  CSVs under data/analysis/csv_outputs/market_making/:
    mm_negrisk_consistency_episodes.csv     one row per violation episode
    mm_negrisk_consistency_event_summary.csv one row per event ever in violation
    mm_negrisk_consistency_prevalence.csv   universe-level counts/prevalence
  Printed summary with bootstrap CIs (cluster = event).

Run from polymarket/research:
  PYTHONPATH=. uv run python scripts/mm_negrisk_consistency_analyze.py <run_dir>
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data/analysis/csv_outputs/market_making")
FEE_BUFFER_DEFAULT = 0.0  # detection is gross; fees netted per-episode from leg detail
CAPACITY_HAIRCUTS = (1.0, 0.25, 0.05)
# Fee schedules (assumption ledger):
#   repo-canonical (K5-STRESS / A14c): fee = FEE_RATE_REPO * p * (1-p) per share
#   harsh sensitivity (Gamma-declared 1000bps base): fee = 0.10 * min(p, 1-p)
FEE_RATE_REPO = 0.05


def basket_fee(prices: list[float], schedule: str) -> float:
    """Total taker fee in $ for trading 1 share of every leg at the given prices."""
    if schedule == "repo":
        return sum(FEE_RATE_REPO * p * (1.0 - p) for p in prices)
    if schedule == "gamma_harsh":
        return sum(0.10 * min(p, 1.0 - p) for p in prices)
    raise ValueError(schedule)


def load_universe(run_dir: Path) -> dict:
    """Latest universe snapshot wins per event id (metadata only)."""
    meta = {}
    for p in sorted(run_dir.glob("universe_*.json")):
        for ev in json.loads(p.read_text()):
            fees = any(bool(l.get("fees_enabled")) for l in ev.get("legs", []))
            meta[ev["event_id"]] = {
                "event_slug": ev.get("event_slug"),
                "title": ev.get("title"),
                "n_legs": ev.get("n_legs"),
                "liquidity": float(ev.get("liquidity") or 0),
                "end_date": ev.get("end_date"),
                "neg_risk_augmented": bool(ev.get("neg_risk_augmented")),
                "fees_enabled_any": fees,
            }
    return meta


def load_cycles(run_dir: Path) -> pd.DataFrame:
    rows = []
    with open(run_dir / "cycles.jsonl") as f:
        for line in f:
            r = json.loads(line)
            row = {
                "ts": r["ts"],
                "cycle": r["cycle"],
                "event_id": r["event_id"],
                "n_legs": r["n_legs"],
                "n_book_missing": r["n_book_missing"],
                "ask_sum": r["ask_sum"],
                "bid_sum": r["bid_sum"],
                "min_ask_depth_usd": r.get("min_ask_depth_usd"),
                "min_bid_depth_usd": r.get("min_bid_depth_usd"),
                "has_detail": "legs" in r,
                "fee_buy_repo": None,
                "fee_buy_harsh": None,
                "fee_sell_repo": None,
                "fee_sell_harsh": None,
            }
            row["min_ask_size"] = None
            row["min_bid_size"] = None
            legs = r.get("legs")
            if legs:
                asks = [l["ba"] for l in legs if l.get("ba") is not None]
                bids = [l["bb"] for l in legs if l.get("bb") is not None]
                ask_sizes = [l["as"] for l in legs if l.get("ba") is not None and l.get("as") is not None]
                bid_sizes = [l["bs"] for l in legs if l.get("bb") is not None and l.get("bs") is not None]
                if asks:
                    row["fee_buy_repo"] = basket_fee(asks, "repo")
                    row["fee_buy_harsh"] = basket_fee(asks, "gamma_harsh")
                if bids:
                    row["fee_sell_repo"] = basket_fee(bids, "repo")
                    row["fee_sell_harsh"] = basket_fee(bids, "gamma_harsh")
                # n baskets assemblable at top-of-book = min SHARES across legs
                if ask_sizes:
                    row["min_ask_size"] = min(ask_sizes)
                if bid_sizes:
                    row["min_bid_size"] = min(bid_sizes)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    return df


def detect_episodes(df: pd.DataFrame, fee_buffer: float) -> pd.DataFrame:
    df = df.sort_values(["event_id", "cycle"]).copy()
    df["buy_violation"] = df["ask_sum"].notna() & (df["ask_sum"] < 1.0 - fee_buffer)
    df["sell_violation"] = df["bid_sum"] > 1.0 + fee_buffer

    episodes = []
    for direction, col, edge_fn, depth_col, fee_repo_col, fee_harsh_col, size_col in (
        ("buy_all", "buy_violation", lambda s: 1.0 - s["ask_sum"], "min_ask_depth_usd", "fee_buy_repo", "fee_buy_harsh", "min_ask_size"),
        ("sell_all", "sell_violation", lambda s: s["bid_sum"] - 1.0, "min_bid_depth_usd", "fee_sell_repo", "fee_sell_harsh", "min_bid_size"),
    ):
        for event_id, g in df.groupby("event_id", sort=False):
            g = g.reset_index(drop=True)
            run_start = None
            for i in range(len(g) + 1):
                active = i < len(g) and bool(g.loc[i, col])
                if active and run_start is None:
                    run_start = i
                elif not active and run_start is not None:
                    seg = g.iloc[run_start:i]
                    edges = edge_fn(seg)
                    fee_repo = seg[fee_repo_col].astype(float)
                    fee_harsh = seg[fee_harsh_col].astype(float)
                    episodes.append(
                        {
                            "event_id": event_id,
                            "direction": direction,
                            "first_ts": seg["ts"].iloc[0],
                            "last_ts": seg["ts"].iloc[-1],
                            "n_cycles": len(seg),
                            "min_duration_s": (seg["ts"].iloc[-1] - seg["ts"].iloc[0]).total_seconds(),
                            "edge_mean_c": float(edges.mean() * 100),
                            "edge_max_c": float(edges.max() * 100),
                            "net_edge_repo_c": float((edges - fee_repo).mean() * 100) if fee_repo.notna().any() else np.nan,
                            "net_edge_harsh_c": float((edges - fee_harsh).mean() * 100) if fee_harsh.notna().any() else np.nan,
                            "min_basket_shares_median": float(seg[size_col].median()) if seg[size_col].notna().any() else np.nan,
                            "binding_depth_usd_median": float(seg[depth_col].median()) if seg[depth_col].notna().any() else np.nan,
                            "binding_depth_usd_min": float(seg[depth_col].min()) if seg[depth_col].notna().any() else np.nan,
                            "open_at_run_start": bool(run_start == 0),
                            "open_at_run_end": bool(i == len(g)),
                        }
                    )
                    run_start = None
    return pd.DataFrame(episodes)


def rough_category(slug: str, title: str) -> str:
    """Heuristic event-family tag for breakdowns (labeled heuristic, not Gamma taxonomy)."""
    s = f"{slug} {title}".lower()
    if "temperature" in s or "weather" in s:
        return "weather"
    if any(k in s for k in ("fed-", "fed ", "cpi", "payroll", "durable-goods", "gdp", "interest-rate", "inflation", "unemployment")):
        return "economics"
    if any(k in s for k in ("cup", "nba", "nhl", "nfl", "mlb", "ucl", "premier", "open", "f1", "ufc", "fifwc", "grand-slam", "wimbledon", "masters", "olympic")):
        return "sports"
    if any(k in s for k in ("election", "president", "senate", "house", "governor", "mayor", "nominee", "midterm", "minister", "chancellor", "party")):
        return "politics"
    if any(k in s for k in ("bitcoin", "btc", "eth", "crypto", "solana")):
        return "crypto"
    return "other"


def bootstrap_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 4000, seed: int = 7):
    """Cluster bootstrap (resample clusters) for the mean."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return (np.nan, np.nan)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([values[clusters == c] for c in pick])
        means.append(vals.mean())
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--fee-buffer", type=float, default=FEE_BUFFER_DEFAULT)
    args = ap.parse_args()

    meta = load_universe(args.run_dir)
    df = load_cycles(args.run_dir)
    cyc_meta = pd.read_json(args.run_dir / "cycle_meta.jsonl", lines=True)
    cadence = float(np.median(np.diff(pd.to_datetime(cyc_meta["ts"], format="ISO8601").values).astype("timedelta64[s]").astype(float))) if len(cyc_meta) > 1 else np.nan

    n_cycles = df["cycle"].nunique()
    n_events = df["event_id"].nunique()
    complete_ask_share = df["ask_sum"].notna().mean()

    episodes = detect_episodes(df, args.fee_buffer)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # prevalence
    prevalence_rows = []
    for direction, col in (("buy_all", "buy_violation"), ("sell_all", "sell_violation")):
        flag = df["ask_sum"].notna() & (df["ask_sum"] < 1.0 - args.fee_buffer) if direction == "buy_all" else (df["bid_sum"] > 1.0 + args.fee_buffer)
        ev_any = df.loc[flag, "event_id"].nunique()
        prevalence_rows.append(
            {
                "direction": direction,
                "events_scanned": n_events,
                "events_ever_violating": ev_any,
                "violating_event_share": ev_any / n_events if n_events else np.nan,
                "event_cycle_rows": len(df),
                "violating_rows": int(flag.sum()),
                "violating_row_share": float(flag.mean()),
            }
        )
    prevalence = pd.DataFrame(prevalence_rows)
    prevalence["n_cycles"] = n_cycles
    prevalence["cadence_s_median"] = cadence
    prevalence["complete_ask_row_share"] = complete_ask_share
    prevalence.to_csv(OUT_DIR / "mm_negrisk_consistency_prevalence.csv", index=False)

    if len(episodes):
        for k, v in (("event_slug", "event_slug"), ("title", "title"), ("n_legs_total", "n_legs"),
                     ("liquidity", "liquidity"), ("end_date", "end_date"),
                     ("neg_risk_augmented", "neg_risk_augmented"), ("fees_enabled_any", "fees_enabled_any")):
            episodes[k] = episodes["event_id"].map(lambda e, key=v: meta.get(e, {}).get(key))
        episodes["category"] = [
            rough_category(s or "", t or "") for s, t in zip(episodes["event_slug"], episodes["title"])
        ]
        end_dt = pd.to_datetime(episodes["end_date"], errors="coerce", utc=True)
        first_ts = pd.to_datetime(episodes["first_ts"], utc=True)
        episodes["days_to_resolution"] = (end_dt - first_ts).dt.total_seconds() / 86400.0
        # Extractable $ per episode: net edge (repo fee schedule) x number of
        # baskets assemblable at top-of-book (min shares across legs), haircutted.
        n_baskets = episodes["min_basket_shares_median"].fillna(0)
        for h in CAPACITY_HAIRCUTS:
            episodes[f"extractable_usd_at_{int(h*100)}pct"] = (
                episodes["net_edge_repo_c"].clip(lower=0).fillna(0) / 100.0 * n_baskets * h
            ).round(2)
        episodes = episodes.sort_values("edge_max_c", ascending=False)
        episodes.to_csv(OUT_DIR / "mm_negrisk_consistency_episodes.csv", index=False)

        ev_summary = (
            episodes.groupby(["event_id", "event_slug", "direction"], dropna=False)
            .agg(
                n_episodes=("n_cycles", "size"),
                total_violation_cycles=("n_cycles", "sum"),
                max_min_duration_s=("min_duration_s", "max"),
                edge_mean_c=("edge_mean_c", "mean"),
                edge_max_c=("edge_max_c", "max"),
                depth_median_usd=("binding_depth_usd_median", "median"),
            )
            .reset_index()
            .sort_values("edge_max_c", ascending=False)
        )
        ev_summary.to_csv(OUT_DIR / "mm_negrisk_consistency_event_summary.csv", index=False)
    else:
        ev_summary = pd.DataFrame()

    print(f"run: {args.run_dir.name}")
    print(f"cycles: {n_cycles} | cadence median {cadence:.0f}s | events {n_events} | complete-ask row share {complete_ask_share:.1%}")
    print(prevalence.to_string(index=False))
    if len(episodes):
        print(f"\nepisodes: {len(episodes)}")
        for direction in ("buy_all", "sell_all"):
            sub = episodes[episodes["direction"] == direction]
            if not len(sub):
                continue
            vals = sub["edge_mean_c"].to_numpy()
            cl = sub["event_id"].to_numpy()
            lo, hi = bootstrap_ci(vals, cl)
            print(
                f"{direction}: n_ep {len(sub)} | events {sub['event_id'].nunique()} | "
                f"edge mean {vals.mean():.2f}c CI [{lo:.2f}, {hi:.2f}] | "
                f"median min-duration {sub['min_duration_s'].median():.0f}s | "
                f"median binding depth ${sub['binding_depth_usd_median'].median():.0f}"
            )
        print("\ntop 15 episodes by max edge:")
        cols = ["event_slug", "direction", "n_cycles", "min_duration_s", "edge_mean_c", "edge_max_c",
                "binding_depth_usd_median", "n_legs_total", "neg_risk_augmented", "open_at_run_end"]
        print(episodes[cols].head(15).to_string(index=False))
    else:
        print("no violation episodes detected")


if __name__ == "__main__":
    main()
