"""
executor/__init__.py

Public interface of the executor package for the Tail Harvester strategy.

Exports only the components actually used:
  - Risk management (kill switch, loss cap)
  - Event journal (JSONL logging and replay)
  - Venue abstractions (order intent, result types, logger protocol)
  - Polymarket connectivity (HTTP client, signing, adapter)
  - Paper mode (fake adapter for testing)
  - Market discovery (slug -> token IDs via Gamma API)
"""

from .risk import (
    ExecutionRiskManager,
    RiskManagerConfig,
    RiskDecision,
    RiskReasonCode,
)
from .journal import (
    ExecutorJournal,
    InMemoryJournalStorage,
    JSONLFileJournalStorage,
    JournalWriter,
    JournalWriterConfig,
    JournalReplayLoader,
    partition_key_from_ts_ns,
)
from .venue import (
    VenueOrderIntent,
    SubmitOrderResult,
    SubmitOrderStatus,
    CancelOrderResult,
    CancelOrderStatus,
    NullLogger,
    NullMetrics,
    StructuredLogger,
    VenueAdapter,
)
from .polymarket_adapter import (
    PolymarketVenueAdapter,
    PolymarketAdapterConfig,
    PolymarketOrderRequest,
    VenueTimeoutError,
    VenueTransportError,
)
from .polymarket_clob_client import (
    PolymarketCLOBHttpClient,
    PolymarketCLOBHttpClientConfig,
)
from .polymarket_sdk_signer import (
    PyClobClientOrderSigner,
    PyClobClientOrderSignerConfig,
    build_py_clob_client_signer,
)
from .fake_venue_adapter import FakeVenueAdapter
from .polymarket_discovery import (
    fetch_active_markets_for_slugs,
    fetch_markets_for_slug,
    SlugMarket,
    SlugResolutionConfig,
)
