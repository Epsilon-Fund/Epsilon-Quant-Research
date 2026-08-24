"""Polymarket py-sdk order gateway — STANDALONE child process (V2 order struct).

Why this exists: Polymarket archived ``py-clob-client`` (2026-05) and the venue now
rejects V1 signed orders with "invalid order version, please use the latest
clob-client". Order placement requires the successor SDK (``polymarket-client``,
GitHub ``Polymarket/py-sdk``) — but that SDK installs the top-level package name
``polymarket``, which collides with this repo's local ``polymarket/`` namespace dir:
importing it in-process would shadow ``polymarket.execution`` and break the whole
stack. So the SDK runs HERE, in its own interpreter/venv, and the execution process
talks to it over a JSON-lines stdin/stdout protocol
(:mod:`polymarket.execution.mirror.pysdk_order_gateway` is the parent-side client).

HARD RULES for this file:
* It must NEVER import ``polymarket.execution`` (or anything from this repo) — in the
  child env, ``polymarket`` IS the SDK.
* Secrets arrive on stdin in the one-time ``init`` request (never argv, never echoed);
  every outgoing error string is scrubbed of secret substrings before it is written.
* stdout carries exactly one JSON object per line per request; all logging goes to
  stderr so the protocol stream stays clean.

Protocol (request → response, both single-line JSON):
    {"id", "op": "ping"}                          → {"id", "ok": true}
    {"id", "op": "init", "private_key", "wallet",
     "api_key", "api_secret", "passphrase"}       → {"id", "ok": true, "address": …}
    {"id", "op": "balance"}                       → {"id", "ok": true, "balance_usdc": …}
    {"id", "op": "create_limit_order", "token_id",
     "price", "size", "side"}                     → {"id", "ok": true, "handle", "order": {…}}
    {"id", "op": "post_order", "handle"}          → {"id", "ok": true, "order_id", "status", …}
    {"id", "op": "cancel_order", "order_id"}      → {"id", "ok": true, "canceled": […], "not_canceled": {…}}
    {"id", "op": "cancel_all"}                    → same shape as cancel_order
Failures: {"id", "ok": false, "error": "<Type>: <scrubbed>", "kind": "timeout"|"transport"|"rejected"|"error"}
``kind`` lets the parent map venue-unreachable/timeout onto the kernel's ambiguous-
submit semantics (TimeoutError/OSError) and clean rejections onto plain NACKs.
"""
from __future__ import annotations

import json
import sys
import uuid


def _scrubber(secrets: list[str]):
    def scrub(text: str) -> str:
        for s in secrets:
            if s:
                text = text.replace(s, "<redacted>")
        return text[:400]
    return scrub


def main() -> int:
    client = None
    signed_by_handle: dict[str, object] = {}
    scrub = _scrubber([])

    def classify(exc: Exception) -> str:
        # Import lazily: these names exist only once the SDK is importable.
        try:
            import polymarket as pm
            if isinstance(exc, pm.TimeoutError):
                return "timeout"
            if isinstance(exc, (pm.TransportError, pm.RateLimitError)):
                return "transport"
            if isinstance(exc, pm.PolymarketError):
                return "rejected"
        except Exception:  # noqa: BLE001 — classification must never crash the loop
            pass
        if isinstance(exc, TimeoutError):
            return "timeout"
        if isinstance(exc, (ConnectionError, OSError)):
            return "transport"
        return "error"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"id": None, "ok": False, "error": "bad json", "kind": "error"}),
                  flush=True)
            continue
        rid = req.get("id")
        op = req.get("op")
        try:
            if op == "ping":
                out = {"ok": True}

            elif op == "init":
                from polymarket import SecureClient
                from polymarket.models.clob.api_key import ApiKeyCreds
                secrets = [req.get("private_key", ""), req.get("api_secret", ""),
                           req.get("passphrase", ""), req.get("api_key", "")]
                scrub = _scrubber([s for s in secrets if s])
                creds = None
                if req.get("api_key") and req.get("api_secret") and req.get("passphrase"):
                    creds = ApiKeyCreds(key=req["api_key"], secret=req["api_secret"],
                                        passphrase=req["passphrase"])
                client = SecureClient.create(
                    private_key=req["private_key"],
                    wallet=req.get("wallet") or None,
                    credentials=creds,
                )
                out = {"ok": True}

            elif client is None:
                out = {"ok": False, "error": "gateway not initialized", "kind": "error"}

            elif op == "balance":
                bal = client.get_balance_allowance(asset_type="COLLATERAL")
                try:
                    payload = bal.model_dump(mode="json")   # numbers only — no secrets here
                except Exception:  # noqa: BLE001
                    payload = {"raw": str(bal)[:200]}
                out = {"ok": True, "balance": payload}

            elif op == "create_limit_order":
                signed = client.create_limit_order(
                    token_id=str(req["token_id"]),
                    price=str(req["price"]),
                    size=str(req["size"]),
                    side=str(req["side"]).upper(),
                )
                handle = uuid.uuid4().hex
                # Bound the stash: a create whose response the parent never read (its
                # timeout) or an ambiguous submit that halts before post_order leaves an
                # orphaned handle; cap so a long session of timeouts can't leak memory
                # (adversarial-review LOW finding). An orphaned handle is never posted.
                if len(signed_by_handle) >= 256:
                    for stale in list(signed_by_handle)[:128]:
                        signed_by_handle.pop(stale, None)
                signed_by_handle[handle] = signed
                try:
                    order_dump = signed.model_dump(mode="json")
                except Exception:  # noqa: BLE001
                    order_dump = {"repr": str(signed)[:400]}
                out = {"ok": True, "handle": handle, "order": order_dump}

            elif op == "post_order":
                signed = signed_by_handle.pop(str(req.get("handle", "")), None)
                if signed is None:
                    out = {"ok": False, "error": "unknown or already-posted handle",
                           "kind": "error"}
                else:
                    resp = client.post_order(signed)
                    try:
                        payload = resp.model_dump(mode="json")
                    except Exception:  # noqa: BLE001
                        payload = {"repr": str(resp)[:400]}
                    ok = bool(payload.get("ok", True))
                    out = {"ok": ok, **payload}
                    if not ok:
                        out.setdefault("kind", "rejected")
                        out.setdefault("error", str(payload.get("message") or payload))

            elif op == "cancel_order":
                resp = client.cancel_order(order_id=str(req["order_id"]))
                try:
                    payload = resp.model_dump(mode="json")
                except Exception:  # noqa: BLE001
                    payload = {"repr": str(resp)[:400]}
                out = {"ok": True, **payload}

            elif op == "cancel_all":
                resp = client.cancel_all()
                try:
                    payload = resp.model_dump(mode="json")
                except Exception:  # noqa: BLE001
                    payload = {"repr": str(resp)[:400]}
                out = {"ok": True, **payload}

            else:
                out = {"ok": False, "error": f"unknown op: {op!r}", "kind": "error"}
        except Exception as exc:  # noqa: BLE001 — the loop must survive any request
            out = {"ok": False, "error": scrub(f"{type(exc).__name__}: {exc}"),
                   "kind": classify(exc)}
        out["id"] = rid
        print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
