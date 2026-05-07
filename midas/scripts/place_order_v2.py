"""
Place one BUY order using py-clob-client-v2 (V2 CLOB protocol).
Run: python scripts/place_order_v2.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(".env", override=False)

PRIVATE_KEY  = os.environ["PM_PRIVATE_KEY"]
API_KEY      = os.environ["PM_API_KEY"]
API_SECRET   = os.environ["PM_API_SECRET"]
PASSPHRASE   = os.environ["PM_PASSPHRASE"]
PROXY_WALLET = os.environ["PM_FUNDER"]
CLOB_URL     = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
TOKEN_ID     = "86566348136142331846636184268712774662342020145430121309412586468100186377810"

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import ApiCreds, OrderArgsV2, OrderType

creds = ApiCreds(
    api_key=API_KEY,
    api_secret=API_SECRET,
    api_passphrase=PASSPHRASE,
)

client = ClobClient(
    host=CLOB_URL,
    chain_id=137,
    key=PRIVATE_KEY,
    creds=creds,
    signature_type=1,      # POLY_PROXY (Magic.link / Google sign-in)
    funder=PROXY_WALLET,
)

print(f"EOA address  : {client.get_address()}")
print(f"Proxy wallet : {PROXY_WALLET}")

# Fetch market info
tick_size = client.get_tick_size(TOKEN_ID)
neg_risk  = client.get_neg_risk(TOKEN_ID)
version   = client.get_version()
print(f"tick_size    : {tick_size}")
print(f"neg_risk     : {neg_risk}")
print(f"CLOB version : {version}")

order_args = OrderArgsV2(
    token_id=TOKEN_ID,
    price=0.95,
    size=5.0,
    side="BUY",
)

print("\nCreating and posting order...")
result = client.create_and_post_order(order_args)
print(f"Result: {result}")
