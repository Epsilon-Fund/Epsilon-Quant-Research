"""Order manager — place / cancel / replace with idempotency + a cancel-replace throttle.

Tracks the engine's *intended* resting orders, at most one per ``(token_id, side)`` (the
shape the symmetric quoter produces: one bid + one ask per token). Each tick the strategy's
desired quotes are reconciled against the active set:

* desired with no active counterpart -> **place** (new deterministic client id);
* desired equal to the active order (same price & size) -> **no-op** (idempotent);
* desired differs from the active order -> **replace** (cancel+place; new client id, new
  placement ts -> the queue position resets), unless the cancel-replace **throttle** is
  still cooling down, in which case the change is **throttled** (existing order kept);
* active with no desired counterpart -> **cancel**.

Every operation is returned as an :class:`OrderOp` for the raw order log. Backtest routes
fills against the active set; live-shadow logs the ops only (no execution).
Client ids are a deterministic counter so replay is reproducible (no wall-clock/random).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mm_engine.interfaces import Order


@dataclass
class ActiveOrder:
    order: Order
    client_id: str
    placement_ts: int      # ts_exchange when placed/replaced — the latency clock origin
    last_change_ts: int    # ts_exchange of the last place/replace — drives the throttle
    remaining: float       # size still resting (decremented by partial fills)


@dataclass(frozen=True)
class OrderOp:
    ts_exchange: int
    op: str                # "place" | "cancel" | "replace" | "throttled"
    client_id: str
    token_id: str
    side: str
    price: float
    size: float

    def as_dict(self) -> dict:
        return {
            "ts_exchange": self.ts_exchange,
            "op": self.op,
            "client_id": self.client_id,
            "token_id": self.token_id,
            "side": self.side,
            "price": self.price,
            "size": self.size,
        }


def _same(a: Order, b: Order) -> bool:
    return abs(a.price - b.price) < 1e-9 and abs(a.size - b.size) < 1e-9


@dataclass
class OrderManager:
    throttle_ms: int = 0   # min ms between place/replace on the same (token, side); 0 = no throttle
    _active: dict[tuple[str, str], ActiveOrder] = field(default_factory=dict)
    _counter: int = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"o{self._counter}"

    def active_orders(self) -> list[ActiveOrder]:
        return list(self._active.values())

    def reconcile(self, desired: list[Order], ts: int) -> list[OrderOp]:
        """Diff ``desired`` against the active set; return the ops to log/execute."""
        ops: list[OrderOp] = []
        desired_by_key: dict[tuple[str, str], Order] = {
            (o.token_id, o.side): o for o in desired
        }

        # cancels: active keys no longer desired
        for key in list(self._active):
            if key not in desired_by_key:
                ao = self._active.pop(key)
                ops.append(
                    OrderOp(ts, "cancel", ao.client_id, ao.order.token_id, ao.order.side,
                            ao.order.price, ao.order.size)
                )

        # places / replaces
        for key, o in desired_by_key.items():
            ao = self._active.get(key)
            if ao is None:
                cid = self._next_id()
                self._active[key] = ActiveOrder(o, cid, ts, ts, float(o.size))
                ops.append(OrderOp(ts, "place", cid, o.token_id, o.side, o.price, o.size))
            elif _same(ao.order, o):
                continue  # idempotent: same resting order, do nothing
            elif self.throttle_ms > 0 and (ts - ao.last_change_ts) < self.throttle_ms:
                ops.append(OrderOp(ts, "throttled", ao.client_id, o.token_id, o.side, o.price, o.size))
            else:
                cid = self._next_id()
                self._active[key] = ActiveOrder(o, cid, ts, ts, float(o.size))
                ops.append(OrderOp(ts, "replace", cid, o.token_id, o.side, o.price, o.size))

        return ops

    def cancel_all(self, ts: int) -> list[OrderOp]:
        """Cancel every active order (used on a capture gap / disconnect)."""
        ops: list[OrderOp] = []
        for key in list(self._active):
            ao = self._active.pop(key)
            ops.append(
                OrderOp(ts, "cancel", ao.client_id, ao.order.token_id, ao.order.side,
                        ao.order.price, ao.order.size)
            )
        return ops

    def drop_filled(self, eps: float = 1e-9) -> list[str]:
        """Remove fully-filled orders (remaining <= eps); return their client ids."""
        dropped: list[str] = []
        for key, ao in list(self._active.items()):
            if ao.remaining <= eps:
                dropped.append(ao.client_id)
                del self._active[key]
        return dropped
