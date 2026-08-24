"""Read-only pre-flight probe: open positions + API-credential liveness for the funder.

Companion to ``mm_join2_balance_probe.py`` (Join-2 C1 pre-flight). Answers two questions
with **read-only** calls (no order, no state change):

1. **Open positions** — the public data-api ``/positions?user=<funder>`` (the same
   authoritative source the copytrade PoC + v0 gate used; outcome-token holdings DO show
   here even though custodied *cash* does not). Prints one line per open position.
2. **API-credential liveness** — an authed CLOB GET (``get_api_keys``) that proves the L2
   credentials in ``.env`` are still valid *for authed calls* (a dead/rotated key 401s here).
   Prints only a boolean + the count of associated keys — **never** the key material.

No secret is ever printed; config values are consumed and discarded. Numbers/ids only.

Run (from the repo root, execution venv):

    PYTHONPATH=. uv run --no-project --with py-clob-client --with python-dotenv --with requests \
        python polymarket/execution/tests/probes/mm_join2_positions_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def _load_env() -> dict[str, str]:
    exec_dir = Path(__file__).resolve().parents[2]
    env_path = exec_dir / ".env"
    if not env_path.exists():
        print(f"FAIL: {env_path} not found", file=sys.stderr)
        sys.exit(2)
    env: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> int:
    env = _load_env()
    funder = (env.get("POLYMARKET_FUNDER") or "").lower()
    if not funder:
        print("FAIL: POLYMARKET_FUNDER missing in .env", file=sys.stderr)
        return 2
    data_url = env.get("POLYMARKET_DATA_URL", "https://data-api.polymarket.com").rstrip("/")

    # 1. Open positions — public read-only.
    print(f"[positions-probe] funder: {funder}")
    try:
        import requests
        r = requests.get(f"{data_url}/positions", params={"user": funder}, timeout=30)
        r.raise_for_status()
        rows = r.json() if r.content else []
    except Exception as exc:  # noqa: BLE001
        print(f"[positions-probe] positions read FAILED: {type(exc).__name__}: {str(exc)[:160]}",
              file=sys.stderr)
        rows = None

    if rows is not None:
        rows = [p for p in rows if float(p.get("size", 0) or 0) != 0.0]
        if not rows:
            print("[positions-probe] open positions: NONE (clean book)")
        else:
            print(f"[positions-probe] open positions: {len(rows)}")
            for p in rows:
                print(f"    - {p.get('title', p.get('conditionId', '?'))[:60]} | "
                      f"outcome={p.get('outcome')} size={p.get('size')} "
                      f"avgPx={p.get('avgPrice')} curPx={p.get('curPrice')} "
                      f"redeemable={p.get('redeemable')}")

    # 2. API-credential liveness — authed GET, prints only a boolean + count.
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        client = ClobClient(
            env.get("POLYMARKET_CLOB_URL", "https://clob.polymarket.com"),
            key=env["POLYMARKET_PRIVATE_KEY"],
            chain_id=int(env.get("POLYMARKET_CHAIN_ID", "137")),
            creds=ApiCreds(
                api_key=env["POLYMARKET_API_KEY"],
                api_secret=env["POLYMARKET_API_SECRET"],
                api_passphrase=env["POLYMARKET_PASSPHRASE"],
            ),
            signature_type=int(env.get("POLYMARKET_SIGNATURE_TYPE", "1")),
            funder=funder,
        )
        keys = client.get_api_keys()
        n = len(keys.get("apiKeys", keys)) if isinstance(keys, dict) else len(keys)
        print(f"[positions-probe] API creds AUTHED OK (associated api-key count: {n})")
    except KeyError as exc:
        print(f"[positions-probe] creds check skipped — missing .env key {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[positions-probe] API creds authed check FAILED: {type(exc).__name__}: "
              f"{str(exc)[:120]}", file=sys.stderr)

    print("[positions-probe] read-only; no order placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
