"""Feed adapters: both emit identical ``MarketEvent`` streams for the same code path."""
from __future__ import annotations

from mm_engine.feeds.live_shadow import live_shadow_feed
from mm_engine.feeds.replay import load_capture_gaps, replay_feed

__all__ = ["live_shadow_feed", "load_capture_gaps", "replay_feed"]
