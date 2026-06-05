"""Prepare a fee-aware Block A0 capture config from current Dali candidates."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml

from scripts.dali_live_clob_capture import parse_token_ids


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "block_a0_capture.yaml"
DEFAULT_OUT = ROOT / "configs" / "block_a0_capture.generated.yaml"
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def fee_category_from_family(family: str) -> str:
    return {
        "geopolitics_policy": "Geopolitics",
        "sports_game_lines": "Sports",
        "ai_product": "Tech",
        "daily_equity_index": "Finance",
        "daily_single_stock": "Finance",
        "daily_crypto_up_down": "Crypto",
    }.get(family, "Other / General")


def geopolitics_score(slug: str, question: str) -> int:
    text = f"{slug} {question}".lower()
    positive = [
        "iran",
        "israel",
        "ukraine",
        "russia",
        "china",
        "taiwan",
        "hezbollah",
        "hamas",
        "gaza",
        "nato",
        "khamenei",
        "netanyahu",
        "zelensky",
        "strait",
        "hormuz",
        "ceasefire",
        "nuclear",
        "military",
        "war",
        "diplomatic",
    ]
    negative = ["2028", "presidential", "election", "governor", "mayor", "primary"]
    return sum(term in text for term in positive) - 2 * sum(term in text for term in negative)


def peak_fee_rate(category: str) -> float:
    return {
        "Geopolitics": 0.0,
        "Sports": 0.0075,
        "Crypto": 0.0175,
        "Finance": 0.0100,
        "Politics": 0.0100,
        "Tech": 0.0100,
        "Culture": 0.0125,
        "Weather": 0.0125,
        "Economics": 0.0125,
        "Other / General": 0.0125,
    }.get(category, 0.0125)


def fetch_json(client: httpx.Client, url: str) -> dict[str, Any]:
    r = client.get(url)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, dict) else {}


def enrich_market(client: httpx.Client, row: pd.Series) -> dict[str, Any]:
    market_id = str(row["id"])
    gamma = fetch_json(client, f"{GAMMA_BASE}/markets/{market_id}")
    condition_id = str(gamma.get("conditionId") or gamma.get("condition_id") or "")
    fee_info: dict[str, Any] = {}
    if condition_id:
        try:
            fee_info = fetch_json(client, f"{CLOB_BASE}/clob-markets/{condition_id}")
        except Exception as exc:  # keep shortlist generation best-effort
            fee_info = {"fetch_error": repr(exc)}
    gamma_category = (
        gamma.get("category")
        or gamma.get("categoryName")
        or fee_category_from_family(str(row.get("family") or ""))
    )
    fee_details = fee_info.get("fd") or {}
    fee_rate = fee_details.get("r")
    category = str(gamma_category)
    if fee_rate == 0:
        category = "Geopolitics"
    elif fee_rate == 0.03:
        category = "Sports"
    elif fee_rate == 0.07:
        category = "Crypto"
    elif category == "Geopolitics" and fee_rate:
        category = "Politics/Policy"
    return {
        "id": market_id,
        "condition_id": condition_id,
        "question": str(row["question"]),
        "slug": str(row["slug"]),
        "family": str(row["family"]),
        "category": category,
        "end_date": row.get("endDate"),
        "volume": float(row.get("volume") or 0),
        "liquidity": float(row.get("liquidity") or 0),
        "best_bid": float(row.get("bestBid")),
        "best_ask": float(row.get("bestAsk")),
        "mid": float((row["bestBid"] + row["bestAsk"]) / 2),
        "spread": float(row["bestAsk"] - row["bestBid"]),
        "clob_token_ids": parse_token_ids(str(row.get("clobTokenIds") or "")),
        "fee": {
            "fees_enabled": bool(gamma.get("feesEnabled", fee_details.get("to", False))),
            "fee_rate": fee_details.get("r"),
            "fee_exponent": fee_details.get("e"),
            "taker_only": fee_details.get("to"),
            "maker_base_fee_bps": gamma.get("makerBaseFee"),
            "taker_base_fee_bps": gamma.get("takerBaseFee"),
            "fee_schedule": gamma.get("feeSchedule"),
            "peak_effective_rate_estimate": peak_fee_rate(str(category)),
            "clob_market_info": fee_info,
        },
        "selection_rationale": (
            "Selected by Block A0 structural screen: mid in band, end date > 7d, "
            "two-sided visible top of book, spread/liquidity acceptable. Recent "
            "7d two-sided trade activity should be rechecked after capture/Block E."
        ),
    }


def build_config(config: dict[str, Any]) -> dict[str, Any]:
    sel = config["selection"]
    df = pd.read_csv(rel(sel["candidates_csv"]))
    df["end_dt"] = pd.to_datetime(df["endDate"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz=UTC)
    df["days_to_end"] = (df["end_dt"] - now).dt.total_seconds() / 86400.0
    df["mid"] = (df["bestBid"] + df["bestAsk"]) / 2
    df["spread"] = df["bestAsk"] - df["bestBid"]
    df["geo_score"] = [
        geopolitics_score(str(slug), str(question))
        for slug, question in zip(df["slug"], df["question"], strict=False)
    ]
    mask = (
        df["bestBid"].notna()
        & df["bestAsk"].notna()
        & df["mid"].between(float(sel["min_mid"]), float(sel["max_mid"]))
        & df["days_to_end"].ge(float(sel["min_days_to_end"]))
        & df["spread"].le(float(sel["max_spread"]))
        & df["liquidity"].fillna(0).ge(float(sel["min_liquidity"]))
    )
    df = df[mask].sort_values(["family", "liquidity", "volume"], ascending=[True, False, False])

    chosen = []
    for family, n in (sel.get("families") or {}).items():
        family_df = df[df["family"].eq(family)]
        if family == "geopolitics_policy":
            family_df = family_df[family_df["geo_score"].gt(0)].sort_values(
                ["geo_score", "liquidity", "volume"],
                ascending=[False, False, False],
            )
        family_df = family_df.head(int(n))
        chosen.extend(row for _, row in family_df.iterrows())
    if len(chosen) < int(sel["max_markets"]):
        used = {str(row["id"]) for row in chosen}
        fill = df[~df["id"].astype(str).isin(used)].head(int(sel["max_markets"]) - len(chosen))
        chosen.extend(row for _, row in fill.iterrows())

    with httpx.Client(timeout=15) as client:
        markets = [enrich_market(client, row) for row in chosen[: int(sel["max_markets"])]]

    out = dict(config)
    out["markets"] = markets
    out["prepared_at"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    out = build_config(config)
    args.out.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")
    print(f"markets: {len(out['markets'])}")
    for market in out["markets"]:
        fee = market["fee"]
        print(
            f"- {market['family']}: {market['slug']} "
            f"mid={market['mid']:.3f} spread={market['spread']:.3f} "
            f"fee_rate={fee.get('fee_rate')} category={market['category']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
