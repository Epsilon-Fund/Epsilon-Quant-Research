"""READ-ONLY smoke: DataApiFillSource against the REAL public data-api.

Tests the live fill-ingestion path against reality (not a fixture). It needs only the funder
ADDRESS (from .env) — NO private key, and it places NOTHING (data-api /trades is a public
GET). It confirms DataApiFillSource: parses real rows; matches the funder's ACTUAL recent
fills by (side, price); applies the session filter; handles multi-fill txs; is idempotent
across polls. Prints counts.

Run:
    PYTHONPATH=. uv run --no-project --with py-clob-client --with websockets \
        python polymarket/execution/tests/probes/mm_bridge_dataapi_smoke.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from polymarket.execution.maker.maker_engine import DataApiTradeClient, _row_ts
from polymarket.execution.maker.mm_engine_bridge import DataApiFillSource

DATA_URL = "https://data-api.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"
_UA = {"User-Agent": "curl/8.0"}


def _get(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=15) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _funder_from_env() -> str | None:
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("POLYMARKET_FUNDER="):
            return line.split("=", 1)[1].strip().strip('"').strip("'").lower()
    return None


def _rows(payload) -> list[dict]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for k in ("data", "trades", "items", "results"):
            if isinstance(payload.get(k), list):
                return [r for r in payload[k] if isinstance(r, dict)]
    return []


class _StubOrder:
    def __init__(self, side, price):
        self.order = type("O", (), {"side": side, "price": price, "token_id": ""})()
        self.client_id = f"stub-{side}-{price}"


def _pick_politics_market_with_activity(dc: DataApiTradeClient) -> tuple[str, str, list[dict]]:
    """Scan top politics NegRisk markets by volume; return the (condition_id, busiest asset,
    feed rows) of the first whose feed has real trades — so the ingestion battery runs
    against real, non-empty data."""
    markets = _get(
        f"{GAMMA_URL}/markets?closed=false&limit=15&order=volume24hr&ascending=false&tag=Politics"
    )
    markets = markets if isinstance(markets, list) else markets.get("data", [])
    for m in markets:
        if not m.get("negRisk"):
            continue
        cid = str(m.get("conditionId") or "").lower()
        toks = m.get("clobTokenIds")
        if not cid or not toks:
            continue
        rows = dc.get_trades(cid)
        if len(rows) < 5:
            continue
        by_asset: dict[str, int] = {}
        for r in rows:
            a = str(r.get("asset") or r.get("assetId") or "")
            if a:
                by_asset[a] = by_asset.get(a, 0) + 1
        if not by_asset:
            continue
        return cid, max(by_asset, key=by_asset.get), rows
    # fallback: the single top market even if thin
    m = markets[0]
    cid = str(m.get("conditionId")).lower()
    asset = str(json.loads(m.get("clobTokenIds", "[]"))[0])
    return cid, asset, dc.get_trades(cid)


def _run_battery(dc: DataApiTradeClient, condition_id: str, asset_id: str,
                 wallet: str, feed_rows: list[dict], label: str) -> bool:
    wallet_rows = [r for r in feed_rows
                   if str(r.get("proxyWallet") or r.get("proxy_wallet") or "").lower() == wallet
                   and str(r.get("asset") or r.get("assetId") or "") == asset_id]
    print(f"\n[{label}] wallet = {wallet[:10]}…{wallet[-4:]}")
    print(f"[{label}] condition feed rows seen (unfiltered): {len(feed_rows)}")
    print(f"[{label}] wallet rows for this asset:            {len(wallet_rows)}")

    far_past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    src = DataApiFillSource(condition_id=condition_id, asset_id=asset_id, funder=wallet,
                            session_start=far_past, data_client=dc, tick=0.01)
    fills = src.poll([])
    print(f"[{label}] poll#1 (session=far-past) emitted fills: {len(fills)}  "
          f"matched(coid)={sum(1 for f in fills if f.client_order_id)} "
          f"unmatched={sum(1 for f in fills if not f.client_order_id)}")

    if fills:
        seen = {(f.side, round(f.price, 2)) for f in fills[:8]}
        stubs = [_StubOrder(side, price) for side, price in seen]
        src2 = DataApiFillSource(condition_id=condition_id, asset_id=asset_id, funder=wallet,
                                 session_start=far_past, data_client=dc, tick=0.01)
        matched = [f for f in src2.poll(stubs) if f.client_order_id]
        print(f"[{label}] match-by-(side,price): {len(stubs)} synth levels → "
              f"{len(matched)} matched fills")

    src_now = DataApiFillSource(condition_id=condition_id, asset_id=asset_id, funder=wallet,
                                session_start=datetime.now(timezone.utc), data_client=dc, tick=0.01)
    post = src_now.poll([])
    print(f"[{label}] session filter (session=now) emitted: {len(post)} "
          f"(expect << {len(fills)} — history is pre-session)")

    per_tx: dict[str, int] = {}
    for r in wallet_rows:
        tx = str(r.get("transactionHash") or r.get("transaction_hash") or "")
        if tx:
            per_tx[tx] = per_tx.get(tx, 0) + 1
    multi = {tx: n for tx, n in per_tx.items() if n > 1}
    emitted_per_tx: dict[str, int] = {}
    for f in fills:
        if f.transaction_hash:
            emitted_per_tx[f.transaction_hash] = emitted_per_tx.get(f.transaction_hash, 0) + 1
    multi_ok = all(emitted_per_tx.get(tx, 0) == n for tx, n in multi.items())
    print(f"[{label}] multi-fill txs: {len(multi)} (max legs/tx="
          f"{max(per_tx.values()) if per_tx else 0}); all legs emitted: "
          f"{multi_ok if multi else 'n/a'}")

    again = src.poll([])
    print(f"[{label}] idempotency re-poll emitted: {len(again)} (expect 0)")

    return len(again) == 0 and len(fills) == len(wallet_rows) and (not multi or multi_ok)


def main() -> int:
    funder = _funder_from_env()
    if not funder:
        print("[smoke] POLYMARKET_FUNDER not found in .env — cannot run")
        return 2
    print(f"[smoke] funder = {funder[:10]}…{funder[-4:]}")
    dc = DataApiTradeClient(DATA_URL)

    # 1) Our funder's live status (a dormant funder = no pre-session history to phantom-book).
    fx = _rows(_get(f"{DATA_URL}/trades?{urllib.parse.urlencode({'user': funder, 'limit': 500})}"))
    print(f"[smoke] funder recent trades (any market): {len(fx)}  "
          f"→ {'has fills' if fx else 'DORMANT (no pre-session history)'}")

    # 2) Pick a real, active politics-NegRisk market and run the ingestion battery against
    #    the busiest real wallet on it — proving parse/match/session/multi-fill/idempotency
    #    against REALITY (our funder is dormant, so a live active wallet is the reality proxy).
    condition_id, asset_id, feed = _pick_politics_market_with_activity(dc)
    print(f"[smoke] market = {condition_id[:16]}…  asset = …{asset_id[-10:]}  feed_rows={len(feed)}")
    by_wallet: dict[str, int] = {}
    for r in feed:
        if str(r.get("asset") or r.get("assetId") or "") != asset_id:
            continue
        w = str(r.get("proxyWallet") or r.get("proxy_wallet") or "").lower()
        if w:
            by_wallet[w] = by_wallet.get(w, 0) + 1
    reality_wallet = max(by_wallet, key=by_wallet.get) if by_wallet else funder

    ok_reality = _run_battery(dc, condition_id, asset_id, reality_wallet, feed,
                              label="reality")
    # 3) Also run the funder battery (0 fills expected while dormant) for completeness.
    ok_funder = _run_battery(dc, condition_id, asset_id, funder, feed, label="funder")

    print(f"\n[smoke] RESULT: reality-battery={'PASS' if ok_reality else 'CHECK'} "
          f"funder-battery={'PASS' if ok_funder else 'CHECK'} "
          f"(funder dormant = {not fx})")
    return 0 if (ok_reality and ok_funder) else 1


if __name__ == "__main__":
    sys.exit(main())
