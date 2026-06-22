"""Market discovery daemon: polls the Polymarket Gamma API on a schedule,
filters active markets to our target universes, and writes ``live_universe.json``
(atomic rename) for the capture daemon to consume.

Run modes:
    python discovery/daemon.py --once      # one cycle, then exit (testing)
    python discovery/daemon.py             # loop forever (production)

The daemon never calls the capture daemon directly. It communicates only by
writing ``data/live_universe.json``; capture hot-reloads that file. See the
package README and infrastructure/data/polymarket_l2_ingestion.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

# --- Reuse the existing, battle-tested Gamma client (HTTP + retry/backoff) ---
# This pipeline is Polymarket MM data infra, so importing Polymarket's own
# GammaClient is in-project (not a crypto<->polymarket cross-import).
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "polymarket" / "research"))
from data_infra.gamma import GammaClient  # noqa: E402

# --- Package-relative paths --------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../l2_ingestion/discovery
PKG_ROOT = HERE.parent                          # .../l2_ingestion
CONFIG_PATH = PKG_ROOT / "config" / "universes.yaml"
DEFAULT_OUTPUT = PKG_ROOT / "data" / "live_universe.json"

# How many Gamma pages (100 markets each) to scan per universe before giving up.
# Markets are fetched ordered by 24h volume desc, so the liquid markets come
# first. The volume floor (min_volume24hr) normally stops paging far earlier;
# this is a hard ceiling. It is kept <= 19 so the deepest offset (1900) stays
# under Gamma's ~offset-2000 paging limit, which returns HTTP 422.
MAX_PAGES_PER_UNIVERSE = 19
PAGE_LIMIT = 100

logger = logging.getLogger("l2.discovery")


# ---------------------------------------------------------------------------
# Small field-parsing helpers — adapted from mm_stage1_live_control.py.
# Gamma returns booleans as real bools OR strings ("true"); clobTokenIds as a
# JSON-encoded string OR a list. These normalize both.
# ---------------------------------------------------------------------------
def boolish(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def market_volume(row: dict[str, Any]) -> float:
    return fnum(row.get("volume24hr") or row.get("volume24h") or row.get("volume24Hour"), 0.0)


def token_ids(row: dict[str, Any]) -> list[str]:
    """Extract the market's CLOB outcome-token asset IDs (YES/NO)."""
    raw = row.get("clobTokenIds") or row.get("clob_token_ids") or row.get("clobTokenIDs")
    if isinstance(raw, str):
        try:
            values = json.loads(raw)
        except Exception:
            values = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    return [str(v) for v in values if str(v)]


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def slugish(row: dict[str, Any]) -> str:
    return str(row.get("slug") or row.get("eventSlug") or row.get("marketSlug") or "")


def active_market(row: dict[str, Any], now: datetime) -> bool:
    """A market is capturable only if it has a live, order-book-enabled CLOB
    with two outcome tokens and an end date that hasn't passed. Adapted from
    mm_stage1_live_control.active_market."""
    if boolish(row.get("closed"), False):
        return False
    if row.get("active") is not None and not boolish(row.get("active"), True):
        return False
    if row.get("enableOrderBook") is not None and not boolish(row.get("enableOrderBook"), True):
        return False
    if row.get("acceptingOrders") is not None and not boolish(row.get("acceptingOrders"), True):
        return False
    if len(token_ids(row)) < 2:
        return False
    end = parse_dt(row.get("endDate") or row.get("end_date"))
    return not (end and end < now - timedelta(minutes=2))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Per-universe fetch
# ---------------------------------------------------------------------------
def fetch_universe(
    client: GammaClient,
    universe_key: str,
    spec: dict[str, Any],
    max_markets: int,
    now: datetime,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch + filter the markets for one universe.

    Selection is driven by ``min_volume24hr`` (the floor): markets are fetched
    ordered by 24h volume desc, and because that order is monotonic, the first
    market below the floor means every remaining market is too — so we stop
    paging immediately. ``max_markets_per_universe`` is only a safety backstop.

    Returns ``(market_records, stop_reason)`` where stop_reason is one of
    ``"floor"`` (normal — hit the volume floor), ``"cap"`` (backstop bound —
    investigate), ``"exhausted"`` (ran out of markets), or ``"page_ceiling"``
    (hit MAX_PAGES_PER_UNIVERSE — should not happen with a sane floor)."""
    flt = spec.get("filter", {})
    tag_id = flt["tag_id"]
    neg_risk_only = boolish(flt.get("neg_risk"), False)
    min_volume = fnum(flt.get("min_volume24hr"), 0.0)

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    reason = "page_ceiling"
    for page_idx in range(MAX_PAGES_PER_UNIVERSE):
        page = client.fetch_markets(
            tag_id=tag_id,
            active="true",
            closed="false",
            order="volume24hr",
            ascending="false",
            limit=PAGE_LIMIT,
            offset=page_idx * PAGE_LIMIT,
        )
        if not page:
            reason = "exhausted"
            break
        for row in page:
            if not isinstance(row, dict):
                continue
            # Volume is the API sort key (desc), so the first sub-floor market
            # marks the tail — nothing after it can qualify. Stop everything.
            if market_volume(row) < min_volume:
                return kept, "floor"
            if not active_market(row, now):
                continue
            is_neg_risk = boolish(row.get("negRisk") or row.get("neg_risk"), False)
            if neg_risk_only and not is_neg_risk:
                continue
            condition_id = str(row.get("conditionId") or row.get("condition_id") or "").lower()
            ids = token_ids(row)
            key = condition_id or ",".join(ids)
            if not key or key in seen:
                continue
            seen.add(key)
            kept.append(
                {
                    "universe": universe_key,
                    "condition_id": condition_id,
                    "question": str(row.get("question") or row.get("title") or ""),
                    "slug": slugish(row),
                    "neg_risk": is_neg_risk,
                    "asset_ids": ids[:2],
                    "volume24hr": market_volume(row),
                }
            )
            if len(kept) >= max_markets:
                return kept, "cap"
        if len(page) < PAGE_LIMIT:
            reason = "exhausted"
            break
    return kept, reason


# ---------------------------------------------------------------------------
# Build the full universe payload
# ---------------------------------------------------------------------------
def build_payload(config: dict[str, Any], now: datetime) -> dict[str, Any]:
    settings = config.get("settings", {})
    max_markets = int(settings.get("max_markets_per_universe", 400))
    universes_cfg = config["universes"]

    markets: list[dict[str, Any]] = []
    per_universe: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    with GammaClient(base_url=settings.get("gamma_api_url", "https://gamma-api.polymarket.com")) as client:
        for key, spec in universes_cfg.items():
            try:
                found, reason = fetch_universe(client, key, spec, max_markets, now)
            except Exception as exc:  # GammaClient already retried 5x w/ backoff
                logger.error("universe %s fetch FAILED: %s", key, exc)
                failures.append(key)
                per_universe[key] = {"market_count": 0, "asset_count": 0, "error": str(exc)}
                continue
            markets.extend(found)
            asset_count = sum(len(m["asset_ids"]) for m in found)
            floor = fnum(spec["filter"].get("min_volume24hr"), 0.0)
            per_universe[key] = {
                "market_count": len(found),
                "asset_count": asset_count,
                "min_volume24hr": floor,
                "stop_reason": reason,
            }
            logger.info(
                "universe %-18s tag_id=%-3s floor=$%-7s -> %d markets, %d assets (stop: %s)%s",
                key,
                spec["filter"]["tag_id"],
                f"{floor:,.0f}",
                len(found),
                asset_count,
                reason,
                " [neg_risk only]" if boolish(spec["filter"].get("neg_risk")) else "",
            )
            if reason == "cap":
                logger.warning(
                    "universe %s hit the max_markets_per_universe=%d BACKSTOP — volume likely "
                    "surged past plan. Consider raising the floor or adding a WS connection; "
                    "we are NOT capturing every market above the floor.",
                    key,
                    max_markets,
                )

    # Flat asset list — what the capture daemon actually subscribes to.
    assets = [
        {
            "asset_id": aid,
            "universe": m["universe"],
            "condition_id": m["condition_id"],
            "neg_risk": m["neg_risk"],
        }
        for m in markets
        for aid in m["asset_ids"]
    ]

    return {
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "ws_url": settings.get("ws_url"),
        "discovery_interval_minutes": settings.get("discovery_interval_minutes", 15),
        "counts": {
            "total_markets": len(markets),
            "total_assets": len(assets),
            "failed_universes": failures,
        },
        "universes": per_universe,
        "markets": markets,
        "assets": assets,
    }


# ---------------------------------------------------------------------------
# Atomic write — write to a temp file, then os.replace (atomic on the same
# filesystem) so the capture daemon never reads a half-written JSON file.
# ---------------------------------------------------------------------------
def write_atomic(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, output_path)  # atomic rename


# ---------------------------------------------------------------------------
# One discovery cycle
# ---------------------------------------------------------------------------
def run_once(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    now = datetime.now(UTC)
    payload = build_payload(config, now)
    total = payload["counts"]["total_assets"]

    if total == 0:
        # API down / everything failed: keep the last known good file, retry next cycle.
        if output_path.exists():
            logger.error(
                "0 assets discovered this cycle — KEEPING last known %s, will retry next cycle.",
                output_path.name,
            )
        else:
            logger.error("0 assets discovered and no previous file exists — nothing written.")
        return payload

    write_atomic(output_path, payload)
    logger.info(
        "wrote %s: %d markets / %d assets across %d universes",
        output_path,
        payload["counts"]["total_markets"],
        total,
        len(payload["universes"]),
    )
    return payload


def run_loop(config: dict[str, Any], output_path: Path, interval_minutes: float) -> None:
    interval_s = interval_minutes * 60.0
    logger.info("discovery loop starting — interval %.0f min", interval_minutes)
    while True:
        try:
            run_once(config, output_path)
        except Exception as exc:  # never let one bad cycle kill the daemon
            logger.exception("discovery cycle raised, continuing: %s", exc)
        time.sleep(interval_s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Polymarket L2 market discovery daemon")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit (testing)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="path to universes.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="path to live_universe.json")
    parser.add_argument("--interval", type=float, default=None, help="override loop interval in minutes")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    config = load_config(args.config)
    interval = args.interval if args.interval is not None else config.get("settings", {}).get(
        "discovery_interval_minutes", 15
    )

    if args.once:
        run_once(config, args.output)
    else:
        run_loop(config, args.output, interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
