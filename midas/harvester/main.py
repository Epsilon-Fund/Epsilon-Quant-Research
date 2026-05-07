from __future__ import annotations

import asyncio
import os

from executor.fake_venue_adapter import FakeVenueAdapter
from executor.polymarket_adapter import PolymarketVenueAdapter
from executor.polymarket_clob_client import (
    PolymarketCLOBHttpClient,
    PolymarketCLOBHttpClientConfig,
)
from executor.polymarket_discovery import fetch_active_markets_for_slugs, fetch_markets_for_slug
from executor.polymarket_sdk_signer import (
    PyClobClientOrderSigner,
    PyClobClientOrderSignerConfig,
)
from executor.risk import ExecutionRiskManager
from executor.venue import StructuredLogger, VenueAdapter
from harvester.city_slugs import next_day_slug
from harvester.config import HarvesterConfig
from harvester.execution import ExecutionEngine
from harvester.logger import build_logger
from harvester.market_data import MarketDataClient
from harvester.oms import OrderManagementSystem
from harvester.registry import TokenRecord, TokenRegistry
from harvester.strategy import TailHarvesterStrategy


def _is_dry_run() -> bool:
    return os.getenv("DRY_RUN", "").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _discover_records(config: HarvesterConfig, logger: StructuredLogger) -> list[TokenRecord]:
    """Fetch all active markets for every configured slug and convert to TokenRecords.

    Uses a thread pool (max_concurrency from SlugResolutionConfig, default 8) so all
    59 slugs are resolved concurrently rather than sequentially. Startup goes from
    ~5 minutes to ~40 seconds on a cold run.
    """
    logger.info("discovery_start", slug_count=len(config.slugs))
    markets_by_slug = fetch_active_markets_for_slugs(list(config.slugs), config.discovery)

    records: list[TokenRecord] = []
    empty_slugs: list[str] = []

    for slug in config.slugs:
        markets = markets_by_slug.get(slug, ())
        active = [m for m in markets if m.token_ids]

        if not active:
            empty_slugs.append(slug)
            continue

        for market in active:
            yes_id = market.yes_token_id
            no_id = market.no_token_id
            if not yes_id:
                continue  # skip markets where YES token cannot be identified
            records.append(TokenRecord(
                token_id=yes_id,
                event_slug=market.slug,
                condition_id=market.market_id,
                end_date_ns=market.end_date_ns,
                tick_size=market.tick_size,
                is_yes=True,
            ))
            if no_id and no_id != yes_id:
                records.append(TokenRecord(
                    token_id=no_id,
                    event_slug=market.slug,
                    condition_id=market.market_id,
                    end_date_ns=market.end_date_ns,
                    tick_size=market.tick_size,
                    is_yes=False,
                ))

    if empty_slugs:
        logger.warning("no_active_markets", slugs=empty_slugs)

    if not records:
        logger.error("no_tradable_markets", slugs=list(config.slugs))
        raise SystemExit(1)

    logger.info("discovery_complete", tokens=len(records), active_slugs=len(config.slugs) - len(empty_slugs), empty_slugs=len(empty_slugs))
    return records


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------


def _build_adapter(
    config: HarvesterConfig,
    logger: StructuredLogger,
    dry_run: bool,
) -> VenueAdapter:
    if dry_run:
        logger.warning(
            "dry_run_active",
            detail="FakeVenueAdapter in use — no orders will reach Polymarket",
        )
        return FakeVenueAdapter(client_order_prefix="dry")

    signer = None
    if config.private_key:
        logger.info(
            "signer_init",
            signature_type=config.signature_type,
            funder=config.funder[:10] + "…" if config.funder else None,
        )
        signer = PyClobClientOrderSigner(
            PyClobClientOrderSignerConfig(
                api_url=config.adapter.api_url,
                signature_type=config.signature_type,
                funder=config.funder,
            )
        )
    else:
        logger.warning("no_private_key", detail="orders will not be signed")

    clob_client = PolymarketCLOBHttpClient(
        PolymarketCLOBHttpClientConfig(
            api_url=config.adapter.api_url,
            api_key=config.adapter.api_key,
            api_secret=config.adapter.api_secret,
            passphrase=config.adapter.passphrase,
            private_key=config.private_key,
        ),
        signer=signer,
    )
    return PolymarketVenueAdapter(clob_client, config.adapter, logger=logger)


# ---------------------------------------------------------------------------
# Component factory + run
# ---------------------------------------------------------------------------


async def _complete_roll(
    from_slug: str,
    tomorrow_slug: str,
    config: HarvesterConfig,
    registry: TokenRegistry,
    client: MarketDataClient,
    logger: StructuredLogger,
    active_slugs: set[str],
) -> bool:
    """Discover next-day market and wire it into the registry + WS.

    Returns True on success, False if discovery failed (caller should retry).
    """
    try:
        next_markets = await asyncio.to_thread(
            fetch_markets_for_slug, tomorrow_slug, config.discovery
        )
    except Exception as exc:
        logger.warning("next_day_discovery_failed", slug=tomorrow_slug, error=str(exc))
        return False

    if not next_markets:
        logger.warning("next_day_market_not_found", slug=tomorrow_slug)
        return False

    new_records: list[TokenRecord] = []
    new_token_ids: list[str] = []
    new_condition_ids: list[str] = []
    for market in next_markets:
        yes_id = market.yes_token_id
        no_id = market.no_token_id
        if not yes_id:
            continue  # skip markets where YES token cannot be identified
        new_records.append(TokenRecord(
            token_id=yes_id,
            event_slug=tomorrow_slug,
            condition_id=market.market_id,
            end_date_ns=market.end_date_ns,
            tick_size=market.tick_size,
            is_yes=True,
        ))
        new_token_ids.append(yes_id)
        if no_id and no_id != yes_id:
            new_records.append(TokenRecord(
                token_id=no_id,
                event_slug=tomorrow_slug,
                condition_id=market.market_id,
                end_date_ns=market.end_date_ns,
                tick_size=market.tick_size,
                is_yes=False,
            ))
            new_token_ids.append(no_id)
        new_condition_ids.append(market.market_id)

    registry.add_records(new_records)
    await client.add_token_ids(new_token_ids)
    await client.add_condition_ids(new_condition_ids)
    active_slugs.discard(from_slug)
    active_slugs.add(tomorrow_slug)

    logger.info(
        "slug_rolled",
        from_slug=from_slug,
        to_slug=tomorrow_slug,
        new_tokens=len(new_token_ids),
    )
    return True


async def _rolling_slug_manager(
    initial_slugs: tuple[str, ...],
    config: HarvesterConfig,
    registry: TokenRegistry,
    client: MarketDataClient,
    logger: StructuredLogger,
    interval_s: float = 60.0,
    close_confirm_count: int = 2,
) -> None:
    """Background task: poll market status and roll each slug to the next day when it closes.

    Each slug is monitored independently. When a city's market closes:
    1. Two consecutive polls must confirm accepting_orders=False before rolling
       (guards against transient Gamma API glitches that could cause false-positive rolls).
    2. The registry marks it closed (strategy cancels the resting order).
    3. The next day's slug is discovered and its tokens are added to the registry.
    4. The WebSocket reconnects to subscribe to the new token IDs.
    5. If next-day discovery fails, the roll is queued and retried every poll cycle.

    Weather markets open 2-3 days in advance, so the next day's market always
    exists by the time today's closes.
    """
    active_slugs: set[str] = set(initial_slugs)
    # slug -> number of consecutive polls that showed accepting_orders=False
    close_votes: dict[str, int] = {}
    # tomorrow_slug -> from_slug for rolls that failed discovery and need retry
    pending_rolls: dict[str, str] = {}

    while True:
        await asyncio.sleep(interval_s)

        # --- Retry any previously-failed next-day discoveries ---
        for tomorrow_slug, from_slug in list(pending_rolls.items()):
            logger.info("retry_pending_roll", slug=tomorrow_slug)
            ok = await _complete_roll(
                from_slug, tomorrow_slug, config, registry, client, logger, active_slugs
            )
            if ok:
                del pending_rolls[tomorrow_slug]

        # --- Poll active slugs for close signals ---
        for slug in list(active_slugs):
            if registry.is_closed(slug):
                close_votes.pop(slug, None)
                continue

            try:
                markets = await asyncio.to_thread(
                    fetch_markets_for_slug, slug, config.discovery
                )
            except Exception as exc:
                logger.warning("status_poll_failed", slug=slug, error=str(exc))
                close_votes.pop(slug, None)  # reset on poll error — don't count bad data
                continue

            if not markets or any(m.accepting_orders for m in markets):
                close_votes.pop(slug, None)  # still open — reset counter
                continue

            # All polls show closed — increment confirmation counter.
            votes = close_votes.get(slug, 0) + 1
            close_votes[slug] = votes
            if votes < close_confirm_count:
                logger.info(
                    "close_signal_pending",
                    slug=slug,
                    confirmations=votes,
                    required=close_confirm_count,
                )
                continue

            # Confirmed closed — mark and roll.
            close_votes.pop(slug, None)
            registry.mark_closed(slug)
            logger.info("market_closed", slug=slug)

            tomorrow_slug = next_day_slug(slug)
            if tomorrow_slug is None:
                logger.warning("slug_roll_skipped", slug=slug, reason="could not compute next day slug")
                continue

            ok = await _complete_roll(
                slug, tomorrow_slug, config, registry, client, logger, active_slugs
            )
            if not ok:
                pending_rolls[tomorrow_slug] = slug
                logger.warning(
                    "roll_queued_for_retry",
                    from_slug=slug,
                    to_slug=tomorrow_slug,
                )


async def _run(
    config: HarvesterConfig,
    records: list[TokenRecord],
    logger: StructuredLogger,
    dry_run: bool,
) -> None:
    """Build all components and run the engine until shutdown."""
    adapter = _build_adapter(config, logger, dry_run)
    risk = ExecutionRiskManager(config.risk, logger=logger)

    registry = TokenRegistry(records, bid_threshold=config.bid_threshold)
    token_ids = list(registry.all_token_ids())
    condition_ids = list({r.condition_id for r in records})

    strategy = TailHarvesterStrategy(registry, config.strategy)
    oms = OrderManagementSystem(
        adapter,
        package_id=config.oms_package_id,
        order_qty=config.oms_order_qty,
    )
    client = MarketDataClient(
        token_ids=token_ids,
        condition_ids=condition_ids,
        config=config.market_data,
        api_key=config.api_key,
        api_secret=config.api_secret,
        passphrase=config.passphrase,
    )
    engine = ExecutionEngine(
        client=client,
        registry=registry,
        strategy=strategy,
        oms=oms,
        adapter=adapter,
        config=config.execution,
        logger=logger,
        risk=risk,
    )

    slugs = tuple(config.slugs)
    status_task = asyncio.create_task(
        _rolling_slug_manager(slugs, config, registry, client, logger),
        name="slug-roller",
    )

    logger.info("startup_complete", dry_run=dry_run)
    try:
        await engine.run()
    finally:
        status_task.cancel()
        await asyncio.gather(status_task, return_exceptions=True)

    logger.info("shutdown_complete")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    # Load .env before building the logger so LOG_LEVEL is visible immediately.
    # HarvesterConfig.from_env() will call load_dotenv again with override=False,
    # which is a no-op for variables already set by this call.
    from dotenv import load_dotenv
    load_dotenv(".env", override=False)

    logger = build_logger()
    dry_run = _is_dry_run()
    config = HarvesterConfig.from_env()
    records = _discover_records(config, logger)
    asyncio.run(_run(config, records, logger, dry_run=dry_run))


if __name__ == "__main__":
    main()
