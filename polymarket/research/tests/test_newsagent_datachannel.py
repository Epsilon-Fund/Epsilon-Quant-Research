"""Tests for the data-evidence channel (Option C, live 2026-08-24).

Covers the three things that can go wrong quietly: the AGPL import boundary, the
§ 4d double-count guard, and the degradation path when the offline snapshot is
missing or stale. No network.

Run: ``PYTHONPATH=. uv run pytest tests/test_newsagent_datachannel.py``
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from newsagent import config, dashboard, datachannel

NEWSAGENT_DIR = Path(datachannel.__file__).parent


# ------------------------------------------------------------ AGPL boundary ---

def test_newsagent_never_imports_openbb():
    """OpenBB is AGPL-3.0-only. It may be imported by the OFFLINE snapshot script,
    whose output is data; it must never be imported by the package that renders a
    public page. Asserted over the real AST, not by convention."""
    offenders = []
    for py in NEWSAGENT_DIR.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] in ("openbb", "openbb_core") for n in names):
                offenders.append(f"{py.name}:{node.lineno}")
    assert offenders == [], f"newsagent/* must not import OpenBB: {offenders}"


def test_the_snapshot_script_is_the_only_place_openbb_may_live():
    """Guards the other direction: the boundary is a real separation, so the read
    module must not quietly grow a compute path."""
    src = (NEWSAGENT_DIR / "datachannel.py").read_text()
    for forbidden in ("urllib", "requests", "fred", "alfred"):
        assert f"import {forbidden}" not in src


# ------------------------------------------------- § 4d double-count guard ----

def A(title="", trail="", lede=""):
    return {"title": title, "trail": trail, "lede": lede}


def F(event_type="economic"):
    return {"relevance": 0.8, "stance": "toward_no", "event_phase": "completed",
            "strength": 0.6, "tone": 0.0, "event_type": event_type,
            "entities": [], "novelty": 0.8, "clarity": 0.7}


def test_a_report_of_an_already_ingested_release_is_flagged():
    assert datachannel.is_double_count(
        A(title="US core PCE inflation came in at 0.19% in July"), F()) is True


def test_both_legs_are_required_event_type_alone_is_not_enough():
    """A Fed official's speech is `economic` and carries genuinely new information;
    zeroing it would throw away evidence the data channel never saw."""
    assert datachannel.is_double_count(
        A(title="Fed governor signals openness to a further hike this autumn"),
        F()) is False


def test_both_legs_are_required_keywords_alone_are_not_enough():
    """A political story that mentions inflation in passing is not a release report."""
    assert datachannel.is_double_count(
        A(title="Senate candidates trade blows over inflation on the campaign trail"),
        F(event_type="election")) is False


def test_a_missing_feature_record_is_not_double_counted():
    assert datachannel.is_double_count(A(title="core pce"), None) is False


DC_SLUG = next(iter(config.DATA_CHANNEL_MARKETS))


def test_the_guard_zeroes_stage_b_weight_but_keeps_the_row():
    feats = [{"article": A(title="Core PCE inflation rose to 3.4% in July"),
              "features": F(), "source_w": 1.0, "cache_key": "k1"},
             {"article": A(title="Powell speech signals patience"),
              "features": F(), "source_w": 1.0, "cache_key": "k2"}]
    out, n = datachannel.apply_double_count_guard(feats, DC_SLUG)
    assert n == 1
    assert out[0]["source_w"] == 0.0 and out[0]["double_counted"] is True
    assert out[0]["source_w_before_double_count"] == 1.0
    assert out[0]["features"] is not None          # the row survives, labelled
    assert out[1]["source_w"] == 1.0


def test_the_guard_does_not_touch_a_news_only_market():
    feats = [{"article": A(title="Core PCE inflation rose to 3.4%"),
              "features": F(), "source_w": 1.0}]
    out, n = datachannel.apply_double_count_guard(feats, "some-news-market")
    assert n == 0 and out[0]["source_w"] == 1.0


# ------------------------------------------------------ snapshot degradation --

def snap(pct=50.7, asof="2026-08-24", slug=DC_SLUG) -> dict:
    return {"asof": asof, "markets": {slug: {
        "p_struct_pct": pct, "asof": asof, "method": "news+data",
        "market_implied": {"available": False, "why": "the window has opened"}}}}


def test_a_fresh_snapshot_supplies_the_anchor():
    p, why = datachannel.p_struct_for(DC_SLUG, "2026-08-24", snap())
    assert p == 50.7 and why == ""


def test_a_stale_snapshot_degrades_to_the_onboarding_prior():
    p, why = datachannel.p_struct_for(DC_SLUG, "2026-09-30", snap(asof="2026-08-24"))
    assert p is None and "old" in why


def test_a_snapshot_from_the_future_is_refused():
    p, why = datachannel.p_struct_for(DC_SLUG, "2026-08-01", snap(asof="2026-08-24"))
    assert p is None and "after the run date" in why


def test_no_snapshot_at_all_is_a_clean_degradation():
    p, why = datachannel.p_struct_for(DC_SLUG, "2026-08-24", {})
    assert p is None and "no data-channel snapshot" in why


def test_a_market_absent_from_the_snapshot_degrades():
    p, why = datachannel.p_struct_for("other-market", "2026-08-24", snap())
    assert p is None and "no p_struct" in why


# ------------------------------------------------------------ DC-8 labelling --

def test_the_data_channel_market_is_labelled_news_plus_data():
    from newsagent import ledger
    assert ledger.method_for(DC_SLUG) == "news+data"
    assert ledger.method_for("some-other-market") == "news"


def test_only_markets_really_in_the_set_get_the_label():
    """A market must join the set on the day its number really changes method —
    never in advance, or the two calibration tracks become unrecoverable."""
    from newsagent import ledger
    for slug in config.LIVE_MARKETS:
        expected = "news+data" if slug in config.DATA_CHANNEL_MARKETS else "news"
        assert ledger.method_for(slug) == expected


# --------------------------------------------------------------- card copy ----

def _card(**over):
    c = {"slug": DC_SLUG, "question": "No change in Fed rates after September?",
         "region": "US", "mtype": "slow", "deadline": "2026-09-16",
         "tract": "data_scored", "tract_note": "", "fv_pct": 44.0,
         "band": [36.0, 52.0], "market_pct": 66.5, "gap_pp": -22.5,
         "divergence_flag": False, "n_relevant": 6, "evidence_quality": "moderate",
         "volume24h": 1, "liquidity": 1, "drivers": [], "evidence": [],
         "n_private_items": 0, "series": [], "sf_id": "sf-2026-018",
         "method": "news+data", "p0_source": "p_struct", "n_double_counted": 0,
         "market_implied": {"available": False,
                            "why": "the Atlanta Fed stops quoting a window once it opens"}}
    c.update(over)
    return c


def test_the_card_says_data_channel_scored_not_structurally_blind():
    out = dashboard._tract_html(_card())
    assert "data-channel scored" in out
    assert "structurally blind" not in out
    assert "never used as an input" in out


def test_the_card_carries_the_n0_transition_line():
    out = dashboard._tract_html(_card())
    assert "no settled track record yet (n=0)" in out
    assert "news-only method" in out


def test_the_card_renders_an_absent_mpt_in_words_and_does_not_break():
    """The Atlanta Fed stops quoting a 3-month window once that window opens, so
    absence is the NORMAL state near a meeting."""
    out = dashboard._tract_html(_card())
    assert "Market-implied context" in out
    assert "stops quoting a window once it opens" in out


def test_the_card_survives_a_completely_missing_market_implied_block():
    out = dashboard._tract_html(_card(market_implied=None))
    assert "data-channel scored" in out and "Market-implied context" in out


def test_the_card_reports_double_counted_articles_when_there_are_any():
    out = dashboard._tract_html(_card(n_double_counted=2))
    assert "already counted" in out or "contribute zero evidence" in out
    assert "cannot move the number twice" in out


def test_a_market_that_degraded_does_not_claim_to_be_data_channel_scored():
    """If the snapshot was missing or stale the number is news-only, and the card
    must say what it actually is."""
    out = dashboard._tract_html(_card(tract="data", p0_source="onboarding_prior"))
    assert "data-channel scored" not in out
    assert "NOT yet built" in out or "structurally blind" in out


# ------------------------------------------------------- the shipped snapshot --

def test_the_shipped_snapshot_is_well_formed_if_present():
    if not datachannel.LATEST.exists():
        pytest.skip("no snapshot on disk")
    snap = json.loads(datachannel.LATEST.read_text())
    for slug, rec in snap.get("markets", {}).items():
        assert slug in config.DATA_CHANNEL_MARKETS
        assert 0.0 <= rec["p_struct_pct"] <= 100.0
        assert rec["method"] == "news+data"
        assert "market_implied" in rec
        assert rec["inputs"]["target_midpoint_pct"] > 0
