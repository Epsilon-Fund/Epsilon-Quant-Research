"""
Tries multiple signing variations to find what Polymarket actually accepts.
Run: python scripts/debug_order.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(".env", override=False)

import httpx, base64, hashlib, hmac as _hmac, time

PRIVATE_KEY  = os.environ["PM_PRIVATE_KEY"]
API_KEY      = os.environ["PM_API_KEY"]
API_SECRET   = os.environ["PM_API_SECRET"]
PASSPHRASE   = os.environ["PM_PASSPHRASE"]
PROXY_WALLET = os.environ["PM_FUNDER"]          # 0x52409...
CLOB_URL     = "https://clob.polymarket.com"
TOKEN_ID     = "86566348136142331846636184268712774662342020145430121309412586468100186377810"

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, CreateOrderOptions
from py_order_utils.builders.base_builder import BaseBuilder
from poly_eip712_structs import make_domain


def make_headers(method, path, body=""):
    ts = str(int(time.time()))
    msg = ts + method.upper() + path
    if body:
        msg += body.replace("'", '"')
    raw_key = base64.urlsafe_b64decode(API_SECRET + "=" * ((4 - len(API_SECRET) % 4) % 4))
    sig = base64.urlsafe_b64encode(_hmac.new(raw_key, msg.encode(), hashlib.sha256).digest()).decode()
    from eth_account import Account
    addr = Account.from_key(PRIVATE_KEY).address
    return {
        "Content-Type": "application/json",
        "POLY_ADDRESS": addr,
        "POLY_SIGNATURE": sig,
        "POLY_TIMESTAMP": ts,
        "POLY_API_KEY": API_KEY,
        "POLY_PASSPHRASE": PASSPHRASE,
    }


def post_order(order_dict, label):
    body = {"order": order_dict, "owner": API_KEY, "orderType": "GTC", "postOnly": False}
    body_str = json.dumps(body, separators=(",", ":"))
    headers = make_headers("POST", "/order", body_str)
    r = httpx.post(f"{CLOB_URL}/order", content=body_str.encode(), headers=headers, timeout=10)
    result = r.json() if r.content else {}
    print(f"  [{label}] HTTP {r.status_code} → {result}")
    return result


def sign_with_domain_version(version_str, fee_rate, sig_type, funder):
    """Override domain version, build and sign an order."""
    def _domain(self, chain_id, verifying_contract):
        return make_domain(
            name="Polymarket CTF Exchange",
            version=version_str,
            chainId=str(chain_id),
            verifyingContract=verifying_contract,
        )
    BaseBuilder._get_domain_separator = _domain

    client = ClobClient(
        host=CLOB_URL, chain_id=137, key=PRIVATE_KEY,
        signature_type=sig_type, funder=funder,
    )
    order_args = OrderArgs(token_id=TOKEN_ID, price=0.95, size=3.0, side="BUY", fee_rate_bps=fee_rate)
    neg_risk = client.get_neg_risk(TOKEN_ID)
    tick_size = client.get_tick_size(TOKEN_ID)
    signed = client.builder.create_order(order_args, CreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk))
    d = signed.dict()
    return d


print(f"\n{'='*60}")
print(f"  debug_order.py — trying all combinations")
print(f"  proxy_wallet : {PROXY_WALLET}")
print(f"{'='*60}\n")

# Also check what CLOB says about API version
r0 = httpx.get(f"{CLOB_URL}/", headers={"User-Agent": "py_clob_client"}, timeout=5)
print(f"CLOB health: {r0.status_code} → {r0.text[:200]}\n")

# Variations to try
variations = [
    # (domain_version, fee_rate_bps, sig_type, funder,   label)
    ("1", 1000, 1, PROXY_WALLET, "v1/fee1000/PROXY"),
    ("2", 1000, 1, PROXY_WALLET, "v2/fee1000/PROXY"),
    ("1",    0, 1, PROXY_WALLET, "v1/fee0/PROXY"),
    ("2",    0, 1, PROXY_WALLET, "v2/fee0/PROXY"),
    ("1",    0, 0, None,         "v1/fee0/EOA"),
    ("2",    0, 0, None,         "v2/fee0/EOA"),
    ("4", 1000, 1, PROXY_WALLET, "v4/fee1000/PROXY"),
    ("4",    0, 1, PROXY_WALLET, "v4/fee0/PROXY"),
]

for domain_v, fee, stype, funder, label in variations:
    try:
        order_dict = sign_with_domain_version(domain_v, fee, stype, funder)
        post_order(order_dict, label)
    except Exception as exc:
        print(f"  [{label}] ERROR: {type(exc).__name__}: {exc}")
