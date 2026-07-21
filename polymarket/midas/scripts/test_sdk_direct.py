"""
Uses py_clob_client directly (no custom code) to place one order.
If this works, the account is fine and the issue is in our signing code.
If this also fails, the issue is in the account setup.

Run: python scripts/test_sdk_direct.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(".env", override=False)

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, PartialCreateOrderOptions

API_KEY    = os.environ["PM_API_KEY"]
API_SECRET = os.environ["PM_API_SECRET"]
PASSPHRASE = os.environ["PM_PASSPHRASE"]
PRIVATE_KEY = os.environ["PM_PRIVATE_KEY"]
CLOB_URL   = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")

TOKEN_ID = "86566348136142331846636184268712774662342020145430121309412586468100186377810"
PRICE    = 0.95
SIZE     = 3.0

print(f"\n=== test_sdk_direct.py ===\n")
print(f"CLOB_URL : {CLOB_URL}")
print(f"API_KEY  : {API_KEY[:12]}…")

# --- Init ClobClient (L1 only — just to sign orders)
client_l1 = ClobClient(
    host=CLOB_URL,
    chain_id=137,
    key=PRIVATE_KEY,
    signature_type=0,  # EOA
)
print(f"EOA address : {client_l1.get_address()}")

# --- Check neg_risk for the token
neg_risk = client_l1.get_neg_risk(TOKEN_ID)
tick_size = client_l1.get_tick_size(TOKEN_ID)
print(f"neg_risk    : {neg_risk}")
print(f"tick_size   : {tick_size}")

# --- Build the order
order_args = OrderArgs(
    token_id=TOKEN_ID,
    price=PRICE,
    size=SIZE,
    side="BUY",
)

print(f"\nCreating signed order (neg_risk={neg_risk}, tick_size={tick_size})…")
try:
    signed = client_l1.create_order(order_args)
except Exception as exc:
    print(f"  ERROR creating order: {type(exc).__name__}: {exc}")
    sys.exit(1)

d = signed.dict()
print(f"  signatureType : {d.get('signatureType')}")
print(f"  maker         : {d.get('maker')}")
print(f"  signer        : {d.get('signer')}")
print(f"  makerAmount   : {d.get('makerAmount')}")
print(f"  takerAmount   : {d.get('takerAmount')}")
print(f"  feeRateBps    : {d.get('feeRateBps')}")
print(f"  signature     : {str(d.get('signature',''))[:30]}…")

# --- Now post it using L2 auth
print(f"\nPosting via L2 client…")
client_l2 = ClobClient(
    host=CLOB_URL,
    chain_id=137,
    key=PRIVATE_KEY,
    creds=ApiCreds(
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_passphrase=PASSPHRASE,
    ),
    signature_type=0,  # EOA
)

# Build body manually so we can print it
import json
body = {"order": signed.dict(), "owner": API_KEY, "orderType": "GTC", "postOnly": False}
print(f"\nBody to POST:")
print(json.dumps(body, indent=2))

# Use the SDK's own post_order
try:
    response = client_l2.post_order(signed)
    print(f"\nResponse: {response}")
except Exception as exc:
    print(f"\nERROR posting: {type(exc).__name__}: {exc}")
