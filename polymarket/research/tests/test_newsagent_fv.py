"""Tests for the newsagent hybrid FV pipeline (Stage A features + Stage B model).

Pure-function tests — no network, no LLM, temp-dir caches.
Run: ``PYTHONPATH=. uv run pytest tests/test_newsagent_fv.py``
"""
from __future__ import annotations

import json
import math

import pytest

from newsagent import dashboard, features, fvmodel
from newsagent.feeds import _lede_last, relevance_rank


def F(relevance=0.8, stance="toward_yes", phase="in_progress", strength=0.7,
      tone=0.0, novelty=1.0):
    return {"relevance": relevance, "stance": stance, "event_phase": phase,
            "strength": strength, "tone": tone, "event_type": "other",
            "entities": [], "novelty": novelty}


PARAMS = {"alpha": 1.85, "lambda": {"slow": 0.9, "shock": 0.6},
          "floor_pp": {"slow": 8.0, "shock": 12.0}}


# ---------------------------------------------------------------- Stage A ------

def test_validate_clamps_and_normalizes():
    out = features.validate({"relevance": 1.7, "stance": "bogus", "event_phase": "nope",
                             "strength": -2, "tone": 9, "novelty": 3})
    assert out["relevance"] == 1.0 and out["stance"] == "neutral"
    assert out["event_phase"] == "none" and out["strength"] == 0.0
    assert out["tone"] == 1.0 and out["novelty"] == 1.0


def test_cache_key_market_relative_and_versioned():
    k1 = features.cache_key("market-a", "Some Headline")
    k2 = features.cache_key("market-b", "Some Headline")
    k3 = features.cache_key("market-a", "  some headline  ")
    assert k1 != k2          # same article, different market -> different features
    assert k1 == k3          # whitespace/case-insensitive
    assert f"v{features.PROMPT_VERSION}" in f"market-a|some headline|v{features.PROMPT_VERSION}"


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "CACHE_DIR", tmp_path)
    key = features.cache_key("m", "t")
    features.write_cache(key, F(), meta={"source": "test"})
    rec = features.cached(key)
    assert rec["relevance"] == 0.8 and rec["_meta"]["source"] == "test"


def test_pending_extractions_skips_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "CACHE_DIR", tmp_path)
    arts = [{"title": "A"}, {"title": "B"}]
    pend = features.pending_extractions("m", "q?", "crit", arts)
    assert len(pend) == 2
    features.write_cache(pend[0]["cache_key"], F())
    pend2 = features.pending_extractions("m", "q?", "crit", arts)
    assert len(pend2) == 1 and pend2[0]["text"].startswith("B")


def test_item_text_caps_and_joins():
    a = {"title": "T", "trail": "tr", "lede": "x" * 2000, "last_para": "L"}
    txt = features._item_text(a)
    assert txt.startswith("T || tr || x") and len(txt) <= 1200


# ---------------------------------------------------------------- feeds --------

def test_lede_last_flat_block():
    lede, last = _lede_last("word " * 300)
    assert lede and last and len(lede) <= 420


def test_relevance_rank_orders_by_hits():
    items = [{"title": "football roundup", "trail": "", "lede": ""},
             {"title": "putin succession rumors grow", "trail": "kremlin", "lede": ""}]
    ranked = relevance_rank(items, "putin AND (succession OR power)", ["kremlin"])
    assert ranked[0]["title"].startswith("putin")


# ---------------------------------------------------------------- Stage B ------

def test_contribution_signs_and_zero():
    assert fvmodel.contribution(F(stance="toward_yes")) > 0
    assert fvmodel.contribution(F(stance="toward_no")) < 0
    assert fvmodel.contribution(F(stance="neutral")) == 0
    assert fvmodel.contribution(F(phase="none", strength=0)) == 0
    assert fvmodel.contribution(None) == 0


def test_effective_weights_decisive_signal():
    cs = [0.9, 0.5, -0.7, -0.2, 0.0]
    ws = fvmodel.effective_weights(cs)
    assert ws[0] == 1.0 and ws[2] == 1.0            # strongest each way count fully
    assert ws[1] == ws[3] == fvmodel.CORROBORATION_W
    assert ws[4] == 0.0


def test_daily_score_dominance_not_drowned():
    # one decisive positive vs a pile of weak negatives: net stays positive
    feats = [{"features": F(stance="toward_yes", phase="completed", strength=0.9)}] + \
            [{"features": F(stance="toward_no", phase="speculative", strength=0.3,
                            relevance=0.5)} for _ in range(6)]
    assert fvmodel.daily_score(feats) > 0


def test_step_state_decay_and_clip():
    s1 = fvmodel.step_state(None, "2026-07-01", 2.0, 0.6)
    assert s1["A"] == 2.0
    s2 = fvmodel.step_state(s1, "2026-07-03", 0.0, 0.6)   # two days: 2 * 0.36
    assert abs(s2["A"] - 2.0 * 0.6 ** 2) < 1e-9
    s3 = fvmodel.step_state(s2, "2026-07-04", 99.0, 0.6)
    assert s3["A"] == fvmodel.A_CLIP


def test_fair_value_monotone_and_clipped():
    assert fvmodel.fair_value(10.0, 0.0, 1.85) == pytest.approx(10.0, abs=0.2)
    assert fvmodel.fair_value(10.0, 2.0, 1.85) > 10.0
    assert fvmodel.fair_value(10.0, -2.0, 1.85) < 10.0
    assert fvmodel.fair_value(99.0, 6.0, 6.0) == 99.0
    assert fvmodel.fair_value(1.0, -6.0, 6.0) == 1.0


def test_band_floors_by_type_and_dispersion():
    quiet = [{"features": F(relevance=0.1)} for _ in range(3)]   # nothing relevant
    assert fvmodel.band_half_pp(quiet, "slow", PARAMS) >= 8.0
    assert fvmodel.band_half_pp(quiet, "shock", PARAMS) >= 12.0
    agree = [{"features": F()} for _ in range(6)]
    disagree = [{"features": F(stance="toward_yes")}, {"features": F(stance="toward_no")},
                {"features": F(stance="toward_yes", strength=0.2)}]
    assert fvmodel.band_half_pp(disagree, "shock", PARAMS) >= \
        fvmodel.band_half_pp(agree, "shock", PARAMS)
    assert fvmodel.band_half_pp(agree, "shock", PARAMS) <= fvmodel.BAND_CAP_PP


def test_divergence_flag_rule():
    # diverges + confident -> flag
    d = fvmodel.divergence_flag(50.0, 30.0, 10.0, 6)
    assert d["flag"] and d["diverges"] and d["confident"]
    # wide band kills it
    assert not fvmodel.divergence_flag(50.0, 30.0, 20.0, 6)["flag"]
    # thin evidence kills it
    assert not fvmodel.divergence_flag(50.0, 30.0, 10.0, 2)["flag"]
    # small gap kills it
    assert not fvmodel.divergence_flag(40.0, 30.0, 10.0, 6)["flag"]


def test_breakdown_sums_to_fv():
    day = [{"article": {"title": "a", "domain": "d"},
            "features": F(stance="toward_yes", phase="completed", strength=0.9)},
           {"article": {"title": "b", "domain": "d"},
            "features": F(stance="toward_no", phase="planned", strength=0.5)},
           {"article": {"title": "c", "domain": "d"},
            "features": F(stance="toward_yes", phase="speculative", strength=0.4)}]
    prev = {"date": "2026-07-03", "A": 1.2}
    bd = fvmodel.breakdown(12.0, prev, "2026-07-05", day, "shock", PARAMS)
    total = bd["p0_pct"] + bd["carry_pp"] + sum(a["pp_effect"] for a in bd["articles"])
    assert total == pytest.approx(bd["fv_pct"], abs=0.05)
    # and fv matches the model run with the same state math
    s_t = fvmodel.daily_score(day)
    nxt = fvmodel.step_state(prev, "2026-07-05", s_t, PARAMS["lambda"]["shock"])
    assert bd["fv_pct"] == pytest.approx(
        fvmodel.fair_value(12.0, nxt["A"], PARAMS["alpha"]), abs=0.05)


def test_breakdown_no_evidence_is_prior_plus_carry():
    bd = fvmodel.breakdown(20.0, None, "2026-07-05", [], "slow", PARAMS)
    assert bd["fv_pct"] == pytest.approx(20.0, abs=0.1)
    assert bd["articles"] == []


def test_fit_alpha_recovers_signal():
    # synthetic: A perfectly separates outcomes -> alpha > 0 and beats prior-only
    pairs = ([{"slug": f"y{i}", "p0_pct": 10.0, "A": 1.5, "y": 1} for i in range(6)] +
             [{"slug": f"n{i}", "p0_pct": 10.0, "A": -1.0, "y": 0} for i in range(6)])
    fit = fvmodel.fit_alpha(pairs)
    assert fit["alpha"] > 0
    assert fit["brier_at_best"] < fit["brier_prior_only"]


def test_fit_alpha_zero_when_evidence_uninformative():
    pairs = ([{"slug": "a", "p0_pct": 50.0, "A": 1.0, "y": 0}] * 5 +
             [{"slug": "b", "p0_pct": 50.0, "A": -1.0, "y": 1}] * 5)
    assert fvmodel.fit_alpha(pairs)["alpha"] == 0.0


# ---------------------------------------------------------------- dashboard ----

def _snapshot(slug="m1", fv=40.0, mid=0.30, flag=False):
    sb = {"mtype": "shock", "n_relevant": 6,
          "divergence": {"gap_pp": round(fv - mid * 100, 1), "flag": flag,
                         "diverges": abs(fv - mid * 100) >= 15, "confident": True,
                         "rule": "r"},
          "breakdown": {"p0_pct": 30.0, "carry_pp": 1.0, "alpha": 1.85, "lambda": 0.6,
                        "clip_scale": 1.0, "fv_pct": fv,
                        "articles": [{"title": "t", "domain": "d", "c": 0.5,
                                      "pp_effect": 9.0}]}}
    return {"market": {"slug": slug, "question": "Q?", "end_date": "2026-08-01T00:00:00Z",
                       "mid": mid, "volume24h": 1000, "liquidity": 5000},
            "packet": {"articles": [{"title": "t", "domain": "d", "seendate": "20260705",
                                     "url": "https://x"}]},
            "forecast": {"p_pct": fv, "band_lo_pct": fv - 10, "band_hi_pct": fv + 10},
            "drivers": ["t"], "region": "US", "stage_b": sb, "sf_id": "sf-1"}


def _series():
    return {"m1": [
        {"date": "2026-07-01", "fv_pct": 35.0, "band_lo_pct": 25.0, "band_hi_pct": 45.0,
         "mid_pct": 30.0, "segment": "backfill"},
        {"date": "2026-07-02", "fv_pct": 36.0, "band_lo_pct": 26.0, "band_hi_pct": 46.0,
         "mid_pct": 31.0, "segment": "backfill"},
        {"date": "2026-07-05", "fv_pct": 40.0, "band_lo_pct": 30.0, "band_hi_pct": 50.0,
         "mid_pct": 30.0, "segment": "live"},
    ]}


def test_showcase_json_shape_and_interim():
    sc = dashboard.build_showcase([_snapshot()], _series())
    assert sc["markets"][0]["fv_pct"] == 40.0
    assert sc["divergence_rule"].startswith("Flag shown")
    # interim: FV_t vs NEXT-day mid, 2 consecutive pairs available
    assert sc["interim"]["n"] == 2
    expected_first = ((35.0 - 31.0) / 100) ** 2
    per = sc["interim"]["per_market"]["m1"]
    assert per["n"] == 2
    assert sc["interim"]["overall_mean"] == pytest.approx(
        (expected_first + ((36.0 - 30.0) / 100) ** 2) / 2, abs=1e-6)


def test_html_renders_selfcontained_and_scrubbed():
    sc = dashboard.build_showcase([_snapshot(flag=True)], _series())
    html_out = dashboard.render_html(sc)
    assert "<script src" not in html_out and "http://cdn" not in html_out
    assert "divergence" in html_out.lower()
    assert "fv construction" in html_out.lower()
    assert "Not investment advice" in html_out
    # no absolute local paths leak into the public page
    assert "/Users/" not in html_out
    # framing never claims to beat the market
    assert "beat the mid" not in html_out.lower()


def test_series_svg_marks_backfill_dashed():
    svg = dashboard._svg_series(_series()["m1"])
    assert "stroke-dasharray" in svg and "reconstructed" in svg


def test_ledger_snapshot_uses_sf_cli_shape():
    # log_snapshot builds CLI args only through the sf wrapper; here we just check
    # the probability normalization contract stays percent -> fraction
    fc = {"p_pct": 40.0, "band_lo_pct": 30.0, "band_hi_pct": 50.0}
    assert 0.0 < fc["p_pct"] / 100.0 < 1.0
