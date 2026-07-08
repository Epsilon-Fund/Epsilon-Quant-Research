"""Read-only pre-flight probe: the AUTHORITATIVE collateral balance for the funder.

Why this exists (v0 gate finding, 2026-07-08): the public surfaces show **$0** for a funded
account — data-api ``/value`` counts open-position value only, and the proxy wallet holds
no on-chain USDC because Polymarket custodies deposited cash off the proxy. The only
authoritative read is the **authed CLOB balance-allowance endpoint**, which requires the
operator's L2 credentials — so this probe is an OPERATOR pre-flight step (runbook §1), not
something an agent runs.

What it does: builds a ``py-clob-client`` with the credentials already present in
``polymarket/execution/.env``, calls ``get_balance_allowance`` for COLLATERAL (a GET — no
order, no state change), and prints the balance/allowance **numbers only**. No secret is
ever printed; config values are consumed and discarded.

Run (from the repo root, in the execution venv):

    PYTHONPATH=. uv run --no-project --with py-clob-client --with python-dotenv \
        python polymarket/execution/tests/probes/mm_join2_balance_probe.py

Expected: `collateral balance: $<your cash>` matching the Polymarket UI (~$109 at the v0
gate). A zero here with cash visible in the UI means the credentials/funder in .env do not
belong to the account you are looking at — STOP and reconcile before any live step.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    exec_dir = Path(__file__).resolve().parents[2]
    env_path = exec_dir / ".env"
    if not env_path.exists():
        print(f"FAIL: {env_path} not found", file=sys.stderr)
        return 2

    env: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")

    required = ("POLYMARKET_PRIVATE_KEY", "POLYMARKET_API_KEY", "POLYMARKET_API_SECRET",
                "POLYMARKET_PASSPHRASE", "POLYMARKET_FUNDER")
    missing = [k for k in required if not env.get(k)]
    if missing:
        print(f"FAIL: missing in .env: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds, AssetType, BalanceAllowanceParams
    except ImportError as exc:
        print(f"FAIL: py-clob-client not importable ({exc}) — run with --with py-clob-client",
              file=sys.stderr)
        return 2

    funder = env["POLYMARKET_FUNDER"].lower()
    print(f"[balance-probe] funder: {funder}")
    print("[balance-probe] endpoint: GET /balance-allowance (authed, read-only, no order)")

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
    # Never let an exception repr from the third-party client reach stderr un-scrubbed —
    # some HTTP-client exception chains can embed request material. Redact every secret
    # value we hold, then truncate (adversarial-review hardening).
    secrets = [env[k] for k in ("POLYMARKET_PRIVATE_KEY", "POLYMARKET_API_KEY",
                                "POLYMARKET_API_SECRET", "POLYMARKET_PASSPHRASE")]

    def _scrubbed(text: str, limit: int = 200) -> str:
        for s in secrets:
            if s:
                text = text.replace(s, "<redacted>")
        return text[:limit]

    try:
        out = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
    except Exception as exc:  # noqa: BLE001 — a probe reports, never raises secrets
        print(f"FAIL: balance-allowance read failed: {type(exc).__name__}: "
              f"{_scrubbed(str(exc))}", file=sys.stderr)
        return 1

    balance_raw = out.get("balance") if isinstance(out, dict) else None
    allowance_raw = None
    if isinstance(out, dict):
        allowances = out.get("allowances") or {}
        allowance_raw = allowances if allowances else out.get("allowance")
    try:
        usd = int(balance_raw) / 1e6 if balance_raw is not None else None  # USDC 6dp
    except (TypeError, ValueError):
        usd = None

    if usd is not None:
        print(f"[balance-probe] collateral balance: ${usd:,.2f}")
    else:
        print(f"[balance-probe] raw response (no parseable balance): {out}")
    if allowance_raw is not None:
        print(f"[balance-probe] allowances: {allowance_raw}")
    print("[balance-probe] PASS iff this matches the Polymarket UI cash balance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
