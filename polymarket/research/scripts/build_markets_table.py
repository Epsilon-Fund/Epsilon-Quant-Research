import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from data_infra.gamma import GammaClient

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "markets"


def parse_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v:
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return []
    return []


def to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize(m: dict) -> dict:
    raw_id = m.get("id")
    return {
        "id": str(raw_id) if raw_id is not None else None,
        "condition_id": m.get("conditionId"),
        "question": m.get("question"),
        "slug": m.get("slug"),
        "outcomes": parse_list(m.get("outcomes")),
        "outcome_prices": parse_list(m.get("outcomePrices")),
        "volume": to_float(m.get("volume")),
        "liquidity": to_float(m.get("liquidity")),
        "active": m.get("active"),
        "closed": m.get("closed"),
        "end_date": m.get("endDate"),
        "created_at": m.get("createdAt"),
        "neg_risk": m.get("negRisk"),
        "clob_token_ids": parse_list(m.get("clobTokenIds")),
    }


LIMIT = 500
SAFE_OFFSET = 100_000  # API caps offsets at ~100,500 with 422


def fetch_combo(gc: GammaClient, seen: dict[str, dict], filters: dict[str, str]) -> None:
    cursor: str | None = None
    pbar = tqdm(unit=" markets", desc=str(filters))
    while True:
        offset = 0
        max_end_date = cursor
        while offset < SAFE_OFFSET:
            params = {
                "limit": LIMIT,
                "offset": offset,
                "order": "endDate",
                "ascending": "true",
                **filters,
            }
            if cursor is not None:
                params["end_date_min"] = cursor
            page = gc.fetch_markets(**params)
            if not page:
                pbar.close()
                return
            for m in page:
                norm = normalize(m)
                mid = norm["id"]
                if mid is not None and mid not in seen:
                    seen[mid] = norm
                ed = norm["end_date"]
                if ed and (max_end_date is None or ed > max_end_date):
                    max_end_date = ed
            pbar.update(len(page))
            offset += LIMIT
        if max_end_date is None or max_end_date == cursor:
            pbar.close()
            return
        cursor = max_end_date


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict] = {}
    combos = [
        {"closed": "true", "archived": "true"},
        {"closed": "true", "archived": "false"},
        {"closed": "false", "archived": "true"},
        {"closed": "false", "archived": "false"},
    ]
    with GammaClient() as gc:
        for filters in combos:
            fetch_combo(gc, seen, filters)
    rows = list(seen.values())

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = OUT_DIR / f"markets_{today}.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression="zstd")

    n_total = len(rows)
    n_active = sum(1 for r in rows if r["active"])
    n_closed = sum(1 for r in rows if r["closed"])
    n_neg_risk = sum(1 for r in rows if r["neg_risk"])
    end_dates = [r["end_date"] for r in rows if r["end_date"]]

    print(f"\nWrote {n_total:,} markets to {out_path}")
    print(f"  active        : {n_active:,}")
    print(f"  closed        : {n_closed:,}")
    print(f"  neg_risk=True : {n_neg_risk:,}")
    if end_dates:
        print(f"  end_date min  : {min(end_dates)}")
        print(f"  end_date max  : {max(end_dates)}")


if __name__ == "__main__":
    main()
