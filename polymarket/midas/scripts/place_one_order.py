"""
Minimal one-shot order placement script.

Discovers the first YES token for the target slug, signs a BUY order, submits it,
and prints exactly what was sent and what Polymarket returned.  No strategy, no OMS.

Run from the midas/ directory:
    python scripts/place_one_order.py

Set DRY_RUN=1 to skip actual submission and just show what would be sent.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(".env", override=False)

# ---------------------------------------------------------------------------
# Config — edit these or set env vars
# ---------------------------------------------------------------------------
SLUG = os.getenv("PM_SLUG_TEST", "highest-temperature-in-dallas-on-may-6-2026")
ORDER_PRICE = float(os.getenv("PM_TEST_PRICE", "0.95"))    # bid price (0–1)
ORDER_SIZE  = int(os.getenv("PM_TEST_SIZE", "3"))           # shares
DRY_RUN     = os.getenv("DRY_RUN", "").strip().lower() in {"1", "true", "yes"}
TOKEN_INDEX = int(os.getenv("PM_TEST_TOKEN_INDEX", "0"))    # 0=first token, 1=second

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
API_KEY     = os.environ["PM_API_KEY"]
API_SECRET  = os.environ["PM_API_SECRET"]
PASSPHRASE  = os.environ["PM_PASSPHRASE"]
PRIVATE_KEY = os.getenv("PM_PRIVATE_KEY") or None

SIGNATURE_TYPE = int(os.getenv("PM_SIGNATURE_TYPE", "0"))  # 0=EOA, 1=POLY_PROXY
# FUNDER is only needed for POLY_PROXY (sig_type=1).
# For EOA (sig_type=0) leave it None — ClobClient defaults to the EOA address.
_funder_raw = os.getenv("PM_FUNDER", "").strip()
FUNDER = _funder_raw if _funder_raw else None

CLOB_URL = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
from executor.polymarket_discovery import fetch_markets_for_slug, SlugResolutionConfig

print(f"\n{'='*60}")
print(f"  place_one_order.py — DRY_RUN={DRY_RUN}")
print(f"{'='*60}\n")

print(f"[1] Discovering markets for slug: {SLUG!r}")
markets = fetch_markets_for_slug(SLUG, SlugResolutionConfig())
real_markets = [m for m in markets if m.token_ids]

if not real_markets:
    print("  ERROR: no markets with token_ids found for that slug")
    sys.exit(1)

market = real_markets[0]
if TOKEN_INDEX >= len(market.token_ids):
    print(f"  ERROR: TOKEN_INDEX={TOKEN_INDEX} but market only has {len(market.token_ids)} tokens")
    sys.exit(1)

token_id   = market.token_ids[TOKEN_INDEX]
condition_id = market.market_id
tick_size  = market.tick_size

print(f"  market_id  : {condition_id}")
print(f"  token_id   : {token_id[:20]}…")
print(f"  tick_size  : {tick_size}")
print(f"  accepting  : {market.accepting_orders}")

# Snap price to tick_size
tick = float(tick_size)
snapped_price = round(round(ORDER_PRICE / tick) * tick, 6)
tick_str = str(tick_size)
decimal_places = len(tick_str.rstrip("0").split(".")[-1]) if "." in tick_str else 0
price_str = f"{snapped_price:.{decimal_places}f}"

print(f"\n[2] Order parameters")
print(f"  side       : BUY")
print(f"  price      : {price_str}  (requested {ORDER_PRICE})")
print(f"  size       : {ORDER_SIZE}")
sig_label = {0: "EOA", 1: "POLY_PROXY", 2: "POLY_GNOSIS_SAFE"}.get(SIGNATURE_TYPE, str(SIGNATURE_TYPE))
print(f"  sig_type   : {SIGNATURE_TYPE}  ({sig_label})")
print(f"  funder     : {(FUNDER[:10] + '…') if FUNDER else '<EOA address>'}")

# ---------------------------------------------------------------------------
# Sign
# ---------------------------------------------------------------------------
print(f"\n[3] Signing order…")

if PRIVATE_KEY:
    from executor.polymarket_sdk_signer import (
        PyClobClientOrderSigner,
        PyClobClientOrderSignerConfig,
    )
    signer = PyClobClientOrderSigner(
        PyClobClientOrderSignerConfig(
            api_url=CLOB_URL,
            signature_type=SIGNATURE_TYPE,
            funder=FUNDER,
        )
    )

    unsigned = {
        "market_id": condition_id,
        "token_id": token_id,
        "side": "BUY",
        "size": str(ORDER_SIZE),
        "price": price_str,
        "tif": "GTC",
        "client_order_id": "test-place-one-order-001",
    }

    try:
        signed_fields = signer(unsigned, PRIVATE_KEY)
    except Exception as exc:
        print(f"  ERROR during signing: {type(exc).__name__}: {exc}")
        sys.exit(1)

    print(f"  OK — signed fields: {list(signed_fields.keys())}")
    print(f"  signatureType : {signed_fields.get('signatureType')}")
    print(f"  maker         : {str(signed_fields.get('maker', ''))[:20]}…")
    print(f"  signer        : {str(signed_fields.get('signer', ''))[:20]}…")
    print(f"  tokenId       : {str(signed_fields.get('tokenId', ''))[:20]}…")
    print(f"  makerAmount   : {signed_fields.get('makerAmount')}")
    print(f"  takerAmount   : {signed_fields.get('takerAmount')}")
    print(f"  signature     : {str(signed_fields.get('signature', ''))[:20]}…")
else:
    print("  WARNING: no PM_PRIVATE_KEY — order will not be signed")
    signed_fields = {
        "market_id": condition_id,
        "token_id": token_id,
        "side": "BUY",
        "size": str(ORDER_SIZE),
        "price": price_str,
    }

# ---------------------------------------------------------------------------
# Build submission body
# ---------------------------------------------------------------------------
body = {
    "order": dict(signed_fields),
    "owner": API_KEY,
    "orderType": "GTC",
    "postOnly": False,
}

print(f"\n[4] Submission body:")
print(json.dumps(body, indent=2, default=str))

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
if DRY_RUN:
    print(f"\n[5] DRY_RUN — skipping submission")
    sys.exit(0)

print(f"\n[5] Submitting to {CLOB_URL}/order …")

from executor.polymarket_clob_client import (
    PolymarketCLOBHttpClient,
    PolymarketCLOBHttpClientConfig,
)

http_client = PolymarketCLOBHttpClient(
    PolymarketCLOBHttpClientConfig(
        api_url=CLOB_URL,
        api_key=API_KEY,
        api_secret=API_SECRET,
        passphrase=PASSPHRASE,
        private_key=PRIVATE_KEY,
        request_timeout_ms=10_000,
    )
)

import json as _json

# Manually POST so we can print raw response
body_str = _json.dumps(body, separators=(",", ":"))
print(f"  body bytes : {len(body_str)}")

try:
    response = http_client.submit_order(body, timeout_ms=10_000)
except Exception as exc:
    print(f"  ERROR during submission: {type(exc).__name__}: {exc}")
    sys.exit(1)

print(f"\n[6] Raw response:")
print(json.dumps(dict(response), indent=2, default=str))

http_status = response.get("http_status", "?")
if http_status == 200 or str(response.get("status", "")).upper() in {"ACCEPTED", "OPEN", "LIVE"}:
    print(f"\n  ✓ ORDER PLACED — venue_id: {response.get('orderID') or response.get('id') or response.get('order_id')}")
else:
    print(f"\n  ✗ ORDER REJECTED — http={http_status}  reason={response.get('error') or response.get('reason')}")
