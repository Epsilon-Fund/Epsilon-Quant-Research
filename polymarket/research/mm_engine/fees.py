"""Fee + maker-rebate model for the MM engine — reuses the canonical schedule, doesn't reinvent.

Polymarket fee formula (per the captured CLOB ``fd`` field and the research convention in
``scripts/dali_block_a1_analyze.py``): ``fee = fee_rate · qty · p · (1 − p)`` (dollars;
observed exponent is 1). The **maker rebate** is a fraction of that fee:
``rebate = rebate_rate · fee_rate · qty · p · (1 − p)``. We quote **passively**, so we pay
**zero maker fee** and *earn* the rebate; the taker-fee path is modeled too for any future
crossing leg (the symmetric quoter never crosses).

Resolution order for a token's schedule (per the task):
1. **per-market schedule** parsed from the captured data's ``fee`` field (``fees_enabled`` + rate
   + rebateRate) — most accurate;
2. **category fallback** via the canonical ``FEE_BY_CATEGORY`` (lazy-imported from
   ``dali_block_a1_analyze`` so this stays the single source of truth — no duplicated numbers);
3. **fee_free** override for sensitivity runs.

The ``net_without_rebate`` discipline ([[block_k5_stress_findings]]): always also report PnL
*excluding* the rebate, so a rebate-only "edge" is visible as policy-fragile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache


def _to_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out


def _clip01(p: float) -> float:
    return 0.0 if p < 0.0 else 1.0 if p > 1.0 else p


@lru_cache(maxsize=1)
def _fee_table():
    """Lazily import the canonical fee schedule (keeps `import mm_engine` free of pandas/mpl)."""
    from scripts.dali_block_a1_analyze import FEE_BY_CATEGORY, family_category

    return FEE_BY_CATEGORY, family_category


@dataclass(frozen=True)
class FeeSchedule:
    """One market's (or category's) fee schedule. ``exponent`` is assumed 1 (PM observed)."""

    fee_rate: float
    rebate_rate: float
    fees_enabled: bool = True
    source: str = "explicit"   # "market" | "category:<cat>" | "fee_free" | "explicit"

    def taker_fee(self, qty: float, price: float) -> float:
        if not self.fees_enabled:
            return 0.0
        p = _clip01(price)
        return self.fee_rate * abs(qty) * p * (1.0 - p)

    def maker_rebate(self, qty: float, price: float) -> float:
        # We are the maker: pay no fee, earn rebate_rate * (the taker fee at this print).
        return self.rebate_rate * self.taker_fee(qty, price)

    @classmethod
    def from_fee_field(cls, fee: dict, *, source: str = "market") -> "FeeSchedule":
        """Parse the captured per-market ``fee``/``fd`` dict ({fees_enabled, rate, rebateRate})."""
        enabled = bool(fee.get("fees_enabled", fee.get("feesEnabled", True)))
        rate = _to_float(fee.get("rate", fee.get("fee_rate"))) or 0.0
        reb = _to_float(fee.get("rebateRate", fee.get("rebate_rate"))) or 0.0
        return cls(fee_rate=rate, rebate_rate=reb, fees_enabled=enabled, source=source)


FEE_FREE = FeeSchedule(0.0, 0.0, fees_enabled=False, source="fee_free")


@dataclass
class FeeModel:
    """Resolves a :class:`FeeSchedule` per token: per-market → category → default."""

    market_schedules: dict[str, FeeSchedule] = field(default_factory=dict)   # from data 'fee' field
    token_category: dict[str, str] = field(default_factory=dict)             # token_id -> category name
    default_category: str = "Other"
    fee_free: bool = False

    def schedule_for(self, token_id: str) -> FeeSchedule:
        if self.fee_free:
            return FEE_FREE
        sched = self.market_schedules.get(token_id)
        if sched is not None:
            return sched
        table, _ = _fee_table()
        cat = self.token_category.get(token_id, self.default_category)
        params = table.get(cat, table["Other"])
        return FeeSchedule(
            fee_rate=params["fee_rate"],
            rebate_rate=params["maker_rebate_pct"],
            fees_enabled=True,
            source=f"category:{cat}",
        )

    def maker_rebate(self, token_id: str, qty: float, price: float) -> float:
        return self.schedule_for(token_id).maker_rebate(qty, price)

    def taker_fee(self, token_id: str, qty: float, price: float) -> float:
        return self.schedule_for(token_id).taker_fee(qty, price)

    @classmethod
    def fee_free_model(cls) -> "FeeModel":
        return cls(fee_free=True)

    @classmethod
    def from_token_families(
        cls,
        token_family: dict[str, str],
        *,
        market_schedules: dict[str, FeeSchedule] | None = None,
        default_category: str = "Other",
    ) -> "FeeModel":
        """Build category map from token families via the canonical ``family_category``."""
        _, family_category = _fee_table()
        token_category = {tok: family_category(fam) for tok, fam in token_family.items()}
        return cls(
            market_schedules=market_schedules or {},
            token_category=token_category,
            default_category=default_category,
        )
