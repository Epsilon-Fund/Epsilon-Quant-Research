"""Direct tests for the py-sdk order gateway boundary (the real-money path).

The gateway spawns the SDK in an isolated child process and speaks JSON-lines. The
kernel's entire ambiguity model rests on this boundary mapping child failures onto the
right exception TYPES (TimeoutError / OSError) and never leaking secrets. These tests
drive the PARENT (:class:`PySdkOrderGateway`) against STUB children (plain ``python -c``
scripts that speak the protocol) — no SDK, no network — and exercise the child's pure
helpers directly. (Adversarial-review MEDIUM/HIGH test-gap finding, 2026-07-09.)
"""
from __future__ import annotations

import sys

import pytest

from polymarket.execution.mirror.pysdk_order_gateway import PySdkOrderGateway

# A stub child that echoes a fixed op→response map. Reads JSON lines, replies with the
# mapped payload (merging the request id), so we test protocol/id-matching/errors
# without the SDK. Behaviors: "sleep" (never replies → parent timeout), "die" (exit
# mid-stream → parent OSError), "stale" (reply with a wrong id first, then the right one).
_STUB = r'''
import json, sys, time
mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    op, rid = req.get("op"), req.get("id")
    if op == "ping":
        print(json.dumps({"id": rid, "ok": True})); sys.stdout.flush(); continue
    if mode == "sleep":
        time.sleep(30); continue
    if mode == "die":
        sys.exit(1)
    if mode == "stale":
        print(json.dumps({"id": 999999, "ok": True, "note": "stale"})); sys.stdout.flush()
        print(json.dumps({"id": rid, "ok": True, "order_id": "0xLIVE", "status": "live"}))
        sys.stdout.flush(); continue
    print(json.dumps({"id": rid, "ok": True, "order_id": "0xLIVE", "status": "live"}))
    sys.stdout.flush()
'''


def _gateway(mode: str, timeout_s: float = 5.0) -> PySdkOrderGateway:
    return PySdkOrderGateway(
        spawn_cmd=(sys.executable, "-c", _STUB, mode),
        request_timeout_s=timeout_s,
    )


def test_ping_and_post_ok() -> None:
    gw = _gateway("ok")
    gw.start()
    try:
        resp = gw.post_order(handle="h1")
        assert resp["ok"] is True
        assert resp["order_id"] == "0xLIVE"
    finally:
        gw.close()


def test_slow_child_raises_timeout_not_hang() -> None:
    """A wedged post must surface as TimeoutError (kernel → ambiguous), bounded."""
    gw = _gateway("sleep", timeout_s=1.0)
    gw.start()
    try:
        with pytest.raises(TimeoutError):
            gw.post_order(handle="h1", timeout_s=1.0)
    finally:
        gw.close()


def test_dead_child_raises_oserror() -> None:
    """A child that exits mid-stream must surface as OSError (kernel → ambiguous)."""
    gw = _gateway("die")
    gw.start()
    try:
        with pytest.raises(OSError):
            gw.post_order(handle="h1")
    finally:
        gw.close()


def test_stale_response_id_is_skipped_not_misattributed() -> None:
    """A response with a non-matching id must be discarded, not returned for this call."""
    gw = _gateway("stale")
    gw.start()
    try:
        resp = gw.post_order(handle="h1")
        assert resp["order_id"] == "0xLIVE"   # got OUR reply, not the stale id=999999
        assert resp.get("note") != "stale"
    finally:
        gw.close()


def test_request_before_start_raises_oserror() -> None:
    gw = _gateway("ok")
    with pytest.raises(OSError):
        gw.post_order(handle="h1")


# --- child-side pure helpers (no SDK import needed) -------------------------------

def test_child_scrubber_redacts_all_secrets() -> None:
    from polymarket.execution.mirror.pysdk_gateway import _scrubber
    scrub = _scrubber(["PRIVKEY123", "APISECRET456", "PASS789", "APIKEY000"])
    msg = "error with PRIVKEY123 and APISECRET456 and PASS789 and APIKEY000"
    out = scrub(msg)
    for secret in ("PRIVKEY123", "APISECRET456", "PASS789", "APIKEY000"):
        assert secret not in out
    assert "<redacted>" in out


def test_child_scrubber_truncates() -> None:
    from polymarket.execution.mirror.pysdk_gateway import _scrubber
    scrub = _scrubber([])
    assert len(scrub("x" * 5000)) <= 400
