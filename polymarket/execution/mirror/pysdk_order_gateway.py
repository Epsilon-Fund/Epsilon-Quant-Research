"""Parent-side client for the py-sdk order gateway (see ``pysdk_gateway.py``).

Spawns the standalone gateway script in an ISOLATED interpreter env (``uv run
--with polymarket-client``, ``PYTHONPATH`` stripped) so the SDK's top-level
``polymarket`` package can never shadow this repo's ``polymarket/`` namespace, and
speaks single-line JSON over the child's stdin/stdout.

Failure semantics are chosen to slot into the frozen kernel's ambiguity model:
* request timeout            → ``TimeoutError``  (kernel: ambiguous submit/cancel)
* dead/broken child process  → ``OSError``       (kernel: ambiguous, transport)
* venue rejection            → returned as a normal mapping (kernel: clean NACK)

Secrets are sent to the child once, on stdin, inside ``init`` — never argv (ps-visible),
never logged; the child scrubs them from every error string it emits.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

_GATEWAY_SCRIPT = Path(__file__).resolve().parent / "pysdk_gateway.py"
_DEFAULT_SPAWN: tuple[str, ...] = (
    "uv", "run", "--no-project", "--prerelease=allow", "--with", "polymarket-client",
    "python", str(_GATEWAY_SCRIPT),
)


class PySdkOrderGateway:
    """Long-lived subprocess boundary around the Polymarket py-sdk ``SecureClient``."""

    def __init__(self, *, spawn_cmd: tuple[str, ...] | None = None,
                 request_timeout_s: float = 30.0) -> None:
        self._spawn_cmd = spawn_cmd or _DEFAULT_SPAWN
        self._timeout_s = float(request_timeout_s)
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        # A background thread drains the child's stdout into this queue as parsed JSON
        # dicts (or the _EOF sentinel). Reading via a queue — NOT select()+readline() —
        # is race-free when the child emits multiple lines close together (e.g. a stale
        # response to a prior timed-out request followed by the current one): buffered
        # lines can't hide from a queue the way they hide from an fd-level select.
        self._inbox: "queue.Queue[dict | object]" = queue.Queue()
        self._reader: threading.Thread | None = None

    _EOF = object()

    # -- lifecycle ---------------------------------------------------------------

    def _reader_loop(self, stdout: Any) -> None:
        try:
            for line in stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._inbox.put(json.loads(line))
                except json.JSONDecodeError:
                    continue   # tolerate stray non-protocol output (uv/SDK chatter)
        finally:
            self._inbox.put(self._EOF)

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)   # the child must resolve `polymarket` to the SDK
        # Secrets travel over stdin `init` ONLY — never inherit them into the child's
        # environment (readable via /proc/<pid>/environ, ps eww, uv resolver subprocs,
        # any SDK telemetry). Strip every POLYMARKET_* secret var (adversarial-review
        # MEDIUM finding, 2026-07-09).
        for key in ("POLYMARKET_PRIVATE_KEY", "POLYMARKET_API_KEY",
                    "POLYMARKET_API_SECRET", "POLYMARKET_PASSPHRASE"):
            env.pop(key, None)
        self._proc = subprocess.Popen(
            list(self._spawn_cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,   # SDK/uv chatter; the protocol lives on stdout
            cwd=str(_GATEWAY_SCRIPT.parent),
            env=env,
            text=True,
            bufsize=1,
        )
        # Drain any queued responses/EOF from a prior process before reusing the queue.
        while not self._inbox.empty():
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                break
        self._reader = threading.Thread(
            target=self._reader_loop, args=(self._proc.stdout,),
            name="pysdk_gateway_reader", daemon=True)
        self._reader.start()
        # First ping may pay the uv resolve/venv cost — allow a generous one-time wait.
        self._request({"op": "ping"}, timeout_s=max(self._timeout_s, 120.0))

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 — best-effort shutdown
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    # -- protocol ----------------------------------------------------------------

    def _request(self, payload: dict[str, Any], *, timeout_s: float | None = None) -> dict:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or not proc.stdin or not proc.stdout:
                raise OSError("py-sdk gateway process is not running")
            self._next_id += 1
            rid = self._next_id
            line = json.dumps({**payload, "id": rid})
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise OSError(f"py-sdk gateway pipe closed: {exc}") from exc

            deadline = timeout_s if timeout_s is not None else self._timeout_s
            end = time.monotonic() + deadline
            while True:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"py-sdk gateway request op={payload.get('op')!r} timed out "
                        f"after {deadline}s")
                try:
                    item = self._inbox.get(timeout=remaining)
                except queue.Empty:
                    raise TimeoutError(
                        f"py-sdk gateway request op={payload.get('op')!r} timed out "
                        f"after {deadline}s") from None
                if item is self._EOF:
                    raise OSError("py-sdk gateway closed its stdout (crashed?)")
                if isinstance(item, dict) and item.get("id") == rid:
                    return item
                # Non-matching id: a stale reply to a prior timed-out request — discard
                # and keep waiting for OURS (the never-reused monotonic id guarantees no
                # mis-attribution).

    # -- operations ---------------------------------------------------------------

    def init(self, *, private_key: str, wallet: str, api_key: str = "",
             api_secret: str = "", passphrase: str = "") -> dict:
        out = self._request({
            "op": "init", "private_key": private_key, "wallet": wallet,
            "api_key": api_key, "api_secret": api_secret, "passphrase": passphrase,
        }, timeout_s=max(self._timeout_s, 120.0))
        if not out.get("ok"):
            raise OSError(f"py-sdk gateway init failed: {out.get('error')}")
        return out

    def balance(self) -> dict:
        return self._request({"op": "balance"})

    def create_limit_order(self, *, token_id: str, price: str, size: str,
                           side: str) -> dict:
        return self._request({"op": "create_limit_order", "token_id": token_id,
                              "price": price, "size": size, "side": side})

    def post_order(self, *, handle: str, timeout_s: float | None = None) -> dict:
        return self._request({"op": "post_order", "handle": handle},
                             timeout_s=timeout_s)

    def cancel_order(self, *, order_id: str, timeout_s: float | None = None) -> dict:
        return self._request({"op": "cancel_order", "order_id": order_id},
                             timeout_s=timeout_s)

    def cancel_all(self, *, timeout_s: float | None = None) -> dict:
        return self._request({"op": "cancel_all"}, timeout_s=timeout_s)


def maybe_print_gateway_banner(gateway: "PySdkOrderGateway") -> None:
    """Convenience: one stderr line confirming the gateway is up (no secrets)."""
    print("[pysdk_gateway] child up (isolated env; V2 order struct via polymarket-client)",
          file=sys.stderr, flush=True)


__all__ = ["PySdkOrderGateway", "maybe_print_gateway_banner"]
