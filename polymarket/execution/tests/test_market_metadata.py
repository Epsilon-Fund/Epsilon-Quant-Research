"""Tests for mirror/market_metadata.py — Gamma cache for NegRisk + tick."""
from __future__ import annotations

import io
import json
from urllib.error import URLError

import pytest

from polymarket.execution.mirror.market_metadata import (
    MarketMetadata,
    MarketMetadataCache,
)


def _fake_response(payload) -> io.BytesIO:
    body = json.dumps(payload).encode("utf-8")
    return io.BytesIO(body)


def _cache(*, urlopen_fn=None, **kwargs) -> MarketMetadataCache:
    return MarketMetadataCache(
        gamma_url="https://gamma-api.polymarket.com",
        urlopen_fn=urlopen_fn,
        **kwargs,
    )


# --- get_by_condition --------------------------------------------------

def test_cache_hit_avoids_urlopen() -> None:
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("urlopen must not be called on cache hit")

    cache = _cache(urlopen_fn=boom)
    # Pre-populate via a direct write (testing-only).
    from datetime import datetime, timezone
    cache._cache["cid-A"] = MarketMetadata(
        condition_id="cid-A", is_neg_risk=True, tick_size=0.01,
        fetched_at_utc=datetime.now(timezone.utc),
    )
    meta = cache.get_by_condition("cid-A")
    assert meta is not None and meta.is_neg_risk is True
    assert calls["n"] == 0


def test_cache_miss_fetches_negrisk_true_tick_001() -> None:
    payload = [{"conditionId": "cid-B", "negRisk": True, "tick_size": "0.01"}]
    cache = _cache(urlopen_fn=lambda *a, **k: _fake_response(payload))
    meta = cache.get_by_condition("cid-B")
    assert meta is not None
    assert meta.is_neg_risk is True
    assert meta.tick_size == 0.01
    assert cache.get_by_condition("cid-B") is meta  # cached now


def test_cache_miss_fetches_negrisk_false_tick_0001() -> None:
    payload = [{"conditionId": "cid-C", "negRisk": False, "tick_size": "0.001"}]
    cache = _cache(urlopen_fn=lambda *a, **k: _fake_response(payload))
    meta = cache.get_by_condition("cid-C")
    assert meta is not None
    assert meta.is_neg_risk is False
    assert meta.tick_size == 0.001


def test_url_error_returns_none_not_cached() -> None:
    state = {"calls": 0}

    def flaky(*a, **k):
        state["calls"] += 1
        raise URLError("connection refused")

    cache = _cache(urlopen_fn=flaky)
    assert cache.get_by_condition("cid-D") is None
    # Failure not cached → second call retries (and fails again).
    assert cache.get_by_condition("cid-D") is None
    assert state["calls"] == 2


def test_malformed_json_returns_none() -> None:
    cache = _cache(urlopen_fn=lambda *a, **k: io.BytesIO(b"not json"))
    assert cache.get_by_condition("cid-E") is None


def test_empty_list_returns_none() -> None:
    cache = _cache(urlopen_fn=lambda *a, **k: _fake_response([]))
    assert cache.get_by_condition("cid-F") is None


def test_missing_negrisk_defaults_to_false() -> None:
    payload = [{"conditionId": "cid-G", "tick_size": "0.01"}]
    cache = _cache(urlopen_fn=lambda *a, **k: _fake_response(payload))
    meta = cache.get_by_condition("cid-G")
    assert meta is not None
    assert meta.is_neg_risk is False


def test_missing_tick_size_falls_back_to_default() -> None:
    payload = [{"conditionId": "cid-H", "negRisk": True}]
    cache = _cache(
        urlopen_fn=lambda *a, **k: _fake_response(payload),
        default_tick_size=0.01,
    )
    meta = cache.get_by_condition("cid-H")
    assert meta is not None
    assert meta.tick_size == 0.01


def test_tick_size_alt_key_tickSize_accepted() -> None:
    payload = [{"conditionId": "cid-I", "negRisk": False, "tickSize": "0.001"}]
    cache = _cache(urlopen_fn=lambda *a, **k: _fake_response(payload))
    meta = cache.get_by_condition("cid-I")
    assert meta is not None and meta.tick_size == 0.001


def test_get_by_asset_caches_asset_to_condition_mapping() -> None:
    state = {"n": 0}
    payload = [{"conditionId": "cid-J", "negRisk": True, "tick_size": "0.01"}]

    def once(*a, **k):
        state["n"] += 1
        return _fake_response(payload)

    cache = _cache(urlopen_fn=once)
    meta1 = cache.get_by_asset("asset-X", "cid-J")
    meta2 = cache.get_by_asset("asset-X", "cid-J")
    assert meta1 is meta2
    assert state["n"] == 1


def test_construction_validates_args() -> None:
    with pytest.raises(ValueError):
        MarketMetadataCache(gamma_url="")
    with pytest.raises(ValueError):
        MarketMetadataCache(
            gamma_url="https://g", default_tick_size=0,
        )
    with pytest.raises(ValueError):
        MarketMetadataCache(
            gamma_url="https://g", fetch_timeout_seconds=0,
        )
