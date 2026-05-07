"""
Tries to discover the Polymarket proxy wallet address for this EOA.

Run: python scripts/find_proxy_wallet.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(".env", override=False)

PRIVATE_KEY = os.environ["PM_PRIVATE_KEY"]
API_KEY     = os.environ["PM_API_KEY"]
API_SECRET  = os.environ["PM_API_SECRET"]
PASSPHRASE  = os.environ["PM_PASSPHRASE"]
CLOB_URL    = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams
from py_clob_client.headers.headers import create_level_1_headers
from py_clob_client.http_helpers.helpers import get

# L1-only client (just needs private key)
client = ClobClient(host=CLOB_URL, chain_id=137, key=PRIVATE_KEY)
eoa = client.get_address()
print(f"EOA address : {eoa}")

# --- Derive API key raw response (might contain proxy address) ---
print("\n[1] Raw derive-api-key response:")
headers = create_level_1_headers(client.signer)
try:
    raw = get(f"{CLOB_URL}/auth/derive-api-key", headers=headers)
    print(json.dumps(raw, indent=2))
except Exception as exc:
    print(f"  ERROR: {exc}")

# --- L2 client ---
client2 = ClobClient(
    host=CLOB_URL, chain_id=137, key=PRIVATE_KEY,
    creds=ApiCreds(api_key=API_KEY, api_secret=API_SECRET, api_passphrase=PASSPHRASE),
    signature_type=0,
)

# --- Balance with sig_type=0 (EOA) ---
print("\n[2] Balance allowance sig_type=0 (EOA):")
try:
    from py_clob_client.clob_types import AssetType
    r = client2.get_balance_allowance(
        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=0)
    )
    print(json.dumps(r, indent=2))
except Exception as exc:
    print(f"  ERROR: {exc}")

# --- Balance with sig_type=1 (POLY_PROXY) ---
print("\n[3] Balance allowance sig_type=1 (POLY_PROXY):")
try:
    r = client2.get_balance_allowance(
        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=1)
    )
    print(json.dumps(r, indent=2))
except Exception as exc:
    print(f"  ERROR: {exc}")

# --- Check the maker address embedded in the exchange ---
print("\n[4] Trying to call NegRisk exchange getOrderStatus or similar:")
import httpx
from eth_utils import keccak

NEG_EXCHANGE = '0xC5d563A36AE78145C45a50134d48A1215220f80a'
eoa_padded = eoa[2:].lower().zfill(64)

# Try 'operators(address,address)' — checks if an address is an approved operator for another
# Selector: keccak('operators(address,address)')[:4]
sel = keccak(text='operators(address,address)')[:4].hex()
# For POLY_PROXY: the proxy wallet is an operator for the EOA
# operators(eoa, proxy) should return true
# But we need to know the proxy address... let's try other methods

# Try 'getNonce(address)' — returns the nonce for a maker
sel2 = keccak(text='nonces(address)')[:4].hex()
calldata2 = '0x' + sel2 + eoa_padded
payload2 = {'jsonrpc':'2.0','method':'eth_call','params':[{'to':NEG_EXCHANGE,'data':calldata2},'latest'],'id':1}
r2 = httpx.post('https://1rpc.io/matic', json=payload2, timeout=8)
try:
    res2 = r2.json().get('result','')
    print(f"  nonces({eoa[:10]}…): {res2}")
except Exception:
    pass

# Also try fetching open orders through the CLOB API (they might show the maker address)
print("\n[5] Fetching open orders (may show maker address):")
try:
    open_orders = client2.get_orders()
    if open_orders:
        for o in open_orders[:2]:
            print(json.dumps(o, indent=2, default=str))
    else:
        print("  No open orders")
except Exception as exc:
    print(f"  ERROR: {exc}")

# Fetch recent trades (may show maker address)
print("\n[6] Fetching recent trades:")
try:
    trades = client2.get_trades()
    if trades:
        for t in trades[:2]:
            print(json.dumps(t, indent=2, default=str))
    else:
        print("  No trades")
except Exception as exc:
    print(f"  ERROR: {exc}")
