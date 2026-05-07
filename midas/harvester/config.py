from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from executor.polymarket_adapter import PolymarketAdapterConfig
from executor.polymarket_discovery import SlugResolutionConfig
from executor.risk import RiskManagerConfig
from .execution import ExecutionConfig
from .market_data import MarketDataConfig
from .strategy import StrategyConfig


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HarvesterConfig:
    """All runtime parameters for the tail harvester, loaded from env vars.

    Credentials use field(repr=False) so they never appear in logs or tracebacks.
    Call HarvesterConfig.from_env() at process startup — it validates everything
    upfront and fails fast before any network connection is made.
    """

    # Credentials
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    passphrase: str = field(repr=False)
    private_key: str | None = field(default=None, repr=False)  # L1 auth (optional)
    signature_type: int = field(default=0, repr=False)  # 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE
    funder: str | None = field(default=None, repr=False)  # proxy wallet address (only for POLY_PROXY)

    # What to trade
    slugs: tuple[str, ...] = field(default_factory=tuple)
    bid_threshold: float = 0.90  # passed to TokenRegistry

    # Sub-configs — each component receives only what it needs
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    discovery: SlugResolutionConfig = field(default_factory=SlugResolutionConfig)
    adapter: PolymarketAdapterConfig = field(
        default_factory=lambda: PolymarketAdapterConfig(api_url="https://clob.polymarket.com"),
        repr=False,
    )
    risk: RiskManagerConfig = field(default_factory=RiskManagerConfig)

    # OMS
    oms_package_id: str = "tail-harvester-v1"
    oms_order_qty: int = 10

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, dotenv_path: str | None = ".env") -> HarvesterConfig:
        """Load and validate config from environment variables.

        If dotenv_path points to a file it is loaded first; real environment
        variables always take precedence (override=False).  Pass dotenv_path=None
        to skip .env loading entirely (e.g. in containerised deployments).

        Raises ValueError on the first invalid or missing required variable.
        """
        if dotenv_path:
            load_dotenv(dotenv_path, override=False)

        # --- Required credentials ---
        api_key = _require("PM_API_KEY")
        api_secret = _require("PM_API_SECRET")
        passphrase = _require("PM_PASSPHRASE")
        private_key = os.getenv("PM_PRIVATE_KEY") or None  # optional L1 key

        # --- What markets to watch ---
        # PM_SLUGS overrides auto-generation. When unset, all 59 weather city
        # slugs are generated for today's UTC date and rolled automatically.
        raw_slugs = os.getenv("PM_SLUGS", "").strip()
        if raw_slugs:
            slugs = _parse_slugs(raw_slugs)
            if not slugs:
                raise ValueError("PM_SLUGS was set but contains no valid slugs")
        else:
            from harvester.city_slugs import generate_today_slugs
            slugs = generate_today_slugs()

        # --- Strategy tuning ---
        bid_threshold = _float("STRATEGY_BID_THRESHOLD", default=0.90)
        if not 0.0 < bid_threshold <= 1.0:
            raise ValueError(f"STRATEGY_BID_THRESHOLD must be in (0, 1], got {bid_threshold}")

        min_reprice = _int("STRATEGY_MIN_REPRICE_TICKS", default=2)
        if min_reprice < 0:
            raise ValueError(f"STRATEGY_MIN_REPRICE_TICKS must be >= 0, got {min_reprice}")

        # --- OMS ---
        oms_order_qty = _int("OMS_ORDER_QTY", default=10)
        if oms_order_qty <= 0:
            raise ValueError(f"OMS_ORDER_QTY must be > 0, got {oms_order_qty}")
        oms_package_id = os.getenv("OMS_PACKAGE_ID", "tail-harvester-v1")

        # --- Risk ---
        risk_daily_loss_cap = _float("RISK_DAILY_LOSS_CAP_USDC", default=200.0)
        risk_max_notional = _float("RISK_MAX_NOTIONAL_PER_EVENT_USDC", default=20.0)
        risk_auto_kill_raw = os.getenv("RISK_ENABLE_AUTO_KILL", "true").strip().lower()
        risk_auto_kill = risk_auto_kill_raw not in {"0", "false", "no"}

        # --- Execution ---
        poll_interval_s = _float("EXECUTION_POLL_INTERVAL_S", default=15.0)
        shutdown_timeout_s = _float("EXECUTION_SHUTDOWN_TIMEOUT_S", default=10.0)

        # --- Signing ---
        # For Polymarket proxy-wallet accounts (the standard web-UI setup):
        #   PM_SIGNATURE_TYPE=1 (POLY_PROXY) + PM_FUNDER=<proxy wallet address>
        # For raw EOA accounts:
        #   PM_SIGNATURE_TYPE=0 (EOA), PM_FUNDER not needed
        # Default: POLY_PROXY (1) — standard for web-UI / Magic.link accounts.
        # Set PM_SIGNATURE_TYPE=0 only for raw EOA-only API accounts (no proxy wallet).
        signature_type = _int("PM_SIGNATURE_TYPE", default=1)
        _funder_raw = os.getenv("PM_FUNDER", "").strip()
        funder: str | None = _funder_raw if _funder_raw else None

        # --- Infrastructure URLs ---
        clob_api_url = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
        gamma_api_url = os.getenv("GAMMA_API_URL", "https://gamma-api.polymarket.com")

        return cls(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            private_key=private_key,
            signature_type=signature_type,
            funder=funder,
            slugs=slugs,
            bid_threshold=bid_threshold,
            strategy=StrategyConfig(
                min_reprice_ticks=min_reprice,
            ),
            execution=ExecutionConfig(
                poll_interval_s=poll_interval_s,
                shutdown_timeout_s=shutdown_timeout_s,
            ),
            market_data=MarketDataConfig(),
            discovery=SlugResolutionConfig(
                gamma_api_url=gamma_api_url,
            ),
            adapter=PolymarketAdapterConfig(
                api_url=clob_api_url,
                api_key=api_key,
                api_secret=api_secret,
                passphrase=passphrase,
                private_key=private_key,
            ),
            oms_package_id=oms_package_id,
            oms_order_qty=oms_order_qty,
            risk=RiskManagerConfig(
                daily_loss_cap_usdc=risk_daily_loss_cap,
                max_notional_per_event_usdc=risk_max_notional,
                enable_auto_kill_switch=risk_auto_kill,
            ),
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Required environment variable {name!r} is not set or empty")
    return value


def _parse_slugs(raw: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _float(name: str, *, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {name!r} must be a number, got {raw!r}"
        ) from None


def _int(name: str, *, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {name!r} must be an integer, got {raw!r}"
        ) from None
