"""Gamma-backed per-market metadata cache (NegRisk flag + tick size).

Consolidates the two side-channel lookups the substitute HTTP client
needs before signing an order:

  - ``is_neg_risk`` (bool, from Gamma ``negRisk``) — flips the EIP-712
    ``verifyingContract`` between the binary CTF Exchange and the
    NegRisk CTF Exchange. Without the right value, NegRisk orders are
    rejected with ``invalid signature``.
  - ``tick_size`` (float, from Gamma ``tick_size``) — decoder for the
    kernel's integer ``limit_price_ticks`` field. Default ``0.01`` is
    safe for most political/sports markets but wrong for sub-penny
    longshot markets (e.g. ``0.001`` on gravia-style penny BUYs).

Both are looked up together with a single Gamma call per condition_id
and cached for the process lifetime — NegRisk-ness doesn't change for
a market post-launch, and Polymarket's documented tick sizes are
stable per asset.

If the Gamma fetch fails, callers must refuse to submit
(``OrderRejected reason="cannot_classify_market"``) — falling back to
defaults silently mis-prices and mis-signs orders.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

_HTTP_USER_AGENT = "curl/8.0"


def _http_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": _HTTP_USER_AGENT})


@dataclass(frozen=True, slots=True)
class MarketMetadata:
    condition_id: str
    is_neg_risk: bool
    tick_size: float
    fetched_at_utc: datetime


class MarketMetadataCache:
    """Per-condition Gamma lookups with permanent in-process caching."""

    __slots__ = (
        "_gamma_url",
        "_default_tick_size",
        "_fetch_timeout",
        "_cache",
        "_asset_to_condition",
        "_urlopen",
    )

    def __init__(
        self,
        gamma_url: str,
        *,
        default_tick_size: float = 0.01,
        fetch_timeout_seconds: float = 5.0,
        urlopen_fn=None,
    ) -> None:
        if not gamma_url:
            raise ValueError("gamma_url must be non-empty")
        if default_tick_size <= 0:
            raise ValueError("default_tick_size must be > 0")
        if fetch_timeout_seconds <= 0:
            raise ValueError("fetch_timeout_seconds must be > 0")
        self._gamma_url = gamma_url.rstrip("/")
        self._default_tick_size = default_tick_size
        self._fetch_timeout = fetch_timeout_seconds
        self._cache: dict[str, MarketMetadata] = {}
        self._asset_to_condition: dict[str, str] = {}
        self._urlopen = urlopen_fn if urlopen_fn is not None else urllib.request.urlopen

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_by_condition(self, condition_id: str) -> MarketMetadata | None:
        cached = self._cache.get(condition_id)
        if cached is not None:
            return cached
        fetched = self._fetch_from_gamma(condition_id)
        if fetched is None:
            return None
        self._cache[condition_id] = fetched
        return fetched

    def get_by_asset(
        self, asset_id: str, condition_id: str
    ) -> MarketMetadata | None:
        """Lookup keyed by asset_id; condition_id is passed because
        Gamma is keyed by it. Caches the asset→condition mapping so
        subsequent calls for the same asset hit the cache."""
        # If we've already resolved this asset, use its remembered
        # condition_id (it's stable).
        resolved = self._asset_to_condition.get(asset_id, condition_id)
        meta = self.get_by_condition(resolved)
        if meta is not None:
            self._asset_to_condition[asset_id] = resolved
        return meta

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_from_gamma(self, condition_id: str) -> MarketMetadata | None:
        qs = urllib.parse.urlencode({"condition_ids": condition_id})
        url = f"{self._gamma_url}/markets?{qs}"
        try:
            with self._urlopen(_http_request(url), timeout=self._fetch_timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            TimeoutError,
            OSError,
        ) as exc:
            print(
                f"[market_metadata] Gamma fetch failed for "
                f"condition_id={condition_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr, flush=True,
            )
            return None

        if not isinstance(data, list) or not data:
            print(
                f"[market_metadata] Gamma returned empty/non-list for "
                f"condition_id={condition_id}: {str(data)[:200]}",
                file=sys.stderr, flush=True,
            )
            return None

        row = data[0]
        if not isinstance(row, dict):
            print(
                f"[market_metadata] Gamma row not a dict for "
                f"condition_id={condition_id}: {str(row)[:200]}",
                file=sys.stderr, flush=True,
            )
            return None

        # Defaults: treat missing negRisk as False (binary), missing
        # tick_size as the configured default.
        is_neg_risk = bool(row.get("negRisk", False))
        tick_size = self._parse_tick_size(row)

        return MarketMetadata(
            condition_id=condition_id,
            is_neg_risk=is_neg_risk,
            tick_size=tick_size,
            fetched_at_utc=datetime.now(timezone.utc),
        )

    def _parse_tick_size(self, row: dict) -> float:
        # Gamma response keys we've observed for tick_size: "tick_size",
        # "tickSize". Both have been strings in practice ("0.01"); accept
        # numerics defensively.
        for key in ("tick_size", "tickSize"):
            raw = row.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return self._default_tick_size
